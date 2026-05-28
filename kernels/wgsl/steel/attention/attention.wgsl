// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Attention WGSL (direct port of cuda/kernels/steel/attention/attention.cu
// and metal/src/shaders/steel/attention/attention.metal).
//
// Three entry points:
//   steel_attention_forward       - FP32 standard MHA (with optional mask)
//   steel_attention_forward_fp16  - WGSL has no f16 storage in core; this entry
//                                   accepts FP32 inputs but mirrors the FP16
//                                   CUDA semantics (kept for kernel-name parity).
//   steel_causal_attention        - FP32 causal MHA (decoder-style)
//
// Grid:  (1, seq_len, batch_size * num_heads)
// Block: 256 threads

const WG_SIZE_ATT: u32 = 256u;
const MAX_SEQ_SCORES: u32 = 8192u;  // upper bound on per-row score buffer
const MAX_HEAD_DIM:   u32 = 256u;

struct AttentionParams {
    scale: f32,
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    has_mask: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> Output: array<f32>;
@group(0) @binding(1) var<storage, read>       Q:      array<f32>;
@group(0) @binding(2) var<storage, read>       K:      array<f32>;
@group(0) @binding(3) var<storage, read>       V:      array<f32>;
@group(0) @binding(4) var<storage, read>       Mask:   array<f32>;
@group(0) @binding(5) var<uniform>             params: AttentionParams;

// Threadgroup scratch
var<workgroup> wg_scores:  array<f32, 8192>;   // MAX_SEQ_SCORES
var<workgroup> wg_out_row: array<f32, 256>;    // MAX_HEAD_DIM
var<workgroup> wg_max:     f32;
var<workgroup> wg_sum:     f32;
var<workgroup> wg_partial: array<f32, 8>;      // 256 / 32 simd-style reduce buckets

fn warp_reduce_max(v: f32, lid: u32) -> f32 {
    wg_partial[lid] = v;
    workgroupBarrier();
    var m = v;
    for (var off: u32 = 4u; off > 0u; off = off / 2u) {
        if (lid < off) {
            wg_partial[lid] = max(wg_partial[lid], wg_partial[lid + off]);
        }
        workgroupBarrier();
    }
    m = wg_partial[0];
    return m;
}

fn warp_reduce_sum(v: f32, lid: u32) -> f32 {
    wg_partial[lid] = v;
    workgroupBarrier();
    for (var off: u32 = 4u; off > 0u; off = off / 2u) {
        if (lid < off) {
            wg_partial[lid] = wg_partial[lid] + wg_partial[lid + off];
        }
        workgroupBarrier();
    }
    return wg_partial[0];
}

// ============================================================================
// Standard MHA (FP32) with optional mask
// ============================================================================
@compute @workgroup_size(256)
fn steel_attention_forward(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let batch_size = params.batch_size;
    let num_heads  = params.num_heads;
    let seq_len    = params.seq_len;
    let head_dim   = params.head_dim;
    let scale      = params.scale;
    let has_mask   = params.has_mask != 0u;

    let b     = wid.z / num_heads;
    let h     = wid.z % num_heads;
    let q_idx = wid.y;
    if (b >= batch_size || q_idx >= seq_len) { return; }

    let head_stride = num_heads * seq_len * head_dim;
    let Q_head_base = (b * num_heads + h) * seq_len * head_dim;
    let K_head_base = Q_head_base;
    let V_head_base = Q_head_base;
    let Q_row_base  = Q_head_base + q_idx * head_dim;

    if (tid == 0u) {
        wg_max = -3.4028235e+38;
        wg_sum = 0.0;
    }
    workgroupBarrier();

    // Step 1: Q @ K^T scaled
    for (var k_idx = tid; k_idx < seq_len; k_idx = k_idx + WG_SIZE_ATT) {
        var dot: f32 = 0.0;
        let K_row = K_head_base + k_idx * head_dim;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            dot = dot + Q[Q_row_base + d] * K[K_row + d];
        }
        dot = dot * scale;
        if (has_mask) {
            let m = Mask[b * seq_len * seq_len + q_idx * seq_len + k_idx];
            if (m == 0.0) { dot = -1.0e9; }
        }
        wg_scores[k_idx] = dot;
    }
    workgroupBarrier();

    // Step 2a: max
    var local_max: f32 = -3.4028235e+38;
    for (var i = tid; i < seq_len; i = i + WG_SIZE_ATT) {
        local_max = max(local_max, wg_scores[i]);
    }
    if (tid < 8u) { wg_partial[tid] = -3.4028235e+38; }
    workgroupBarrier();
    let bucket = tid / 32u;
    if (tid % 32u == 0u) {
        // serial atomic-style: only first lane per bucket records
        wg_partial[bucket] = max(wg_partial[bucket], local_max);
    }
    workgroupBarrier();
    if (tid == 0u) {
        var m: f32 = -3.4028235e+38;
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            m = max(m, wg_partial[i]);
        }
        wg_max = m;
    }
    workgroupBarrier();
    let max_val = wg_max;

    // Step 2b: exp & sum
    var local_sum: f32 = 0.0;
    for (var i = tid; i < seq_len; i = i + WG_SIZE_ATT) {
        let e = exp(wg_scores[i] - max_val);
        wg_scores[i] = e;
        local_sum = local_sum + e;
    }
    if (tid < 8u) { wg_partial[tid] = 0.0; }
    workgroupBarrier();
    if (tid % 32u == 0u) {
        wg_partial[bucket] = wg_partial[bucket] + local_sum;
    }
    workgroupBarrier();
    if (tid == 0u) {
        var s: f32 = 0.0;
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            s = s + wg_partial[i];
        }
        wg_sum = s;
    }
    workgroupBarrier();
    let inv_sum = 1.0 / wg_sum;
    for (var i = tid; i < seq_len; i = i + WG_SIZE_ATT) {
        wg_scores[i] = wg_scores[i] * inv_sum;
    }
    workgroupBarrier();

    // Step 3: scores @ V
    for (var d = tid; d < head_dim; d = d + WG_SIZE_ATT) {
        var acc: f32 = 0.0;
        for (var k: u32 = 0u; k < seq_len; k = k + 1u) {
            acc = acc + wg_scores[k] * V[V_head_base + k * head_dim + d];
        }
        wg_out_row[d] = acc;
    }
    workgroupBarrier();

    let out_base = (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    for (var d = tid; d < head_dim; d = d + WG_SIZE_ATT) {
        Output[out_base + d] = wg_out_row[d];
    }
}

// ============================================================================
// FP16-name parity entry (WGSL core lacks f16 storage; kept FP32-typed)
// ============================================================================
@compute @workgroup_size(256)
fn steel_attention_forward_fp16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    // Behavior is identical to steel_attention_forward; an FP16 storage variant
    // requires the "shader-f16" WGSL extension and is provided when the runtime
    // enables it.
    steel_attention_forward(wid, lid);
}

// ============================================================================
// Causal attention (FP32, decoder-style)
// ============================================================================
@compute @workgroup_size(256)
fn steel_causal_attention(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let batch_size = params.batch_size;
    let num_heads  = params.num_heads;
    let seq_len    = params.seq_len;
    let head_dim   = params.head_dim;
    let scale      = params.scale;

    let b     = wid.z / num_heads;
    let h     = wid.z % num_heads;
    let q_idx = wid.y;
    if (b >= batch_size || q_idx >= seq_len) { return; }

    let valid_len = q_idx + 1u;
    let Q_head_base = (b * num_heads + h) * seq_len * head_dim;
    let K_head_base = Q_head_base;
    let V_head_base = Q_head_base;
    let Q_row_base  = Q_head_base + q_idx * head_dim;

    if (tid == 0u) {
        wg_max = -3.4028235e+38;
        wg_sum = 0.0;
    }
    workgroupBarrier();

    for (var k_idx = tid; k_idx < valid_len; k_idx = k_idx + WG_SIZE_ATT) {
        var dot: f32 = 0.0;
        let K_row = K_head_base + k_idx * head_dim;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            dot = dot + Q[Q_row_base + d] * K[K_row + d];
        }
        wg_scores[k_idx] = dot * scale;
    }
    for (var k_idx = tid + valid_len; k_idx < seq_len; k_idx = k_idx + WG_SIZE_ATT) {
        wg_scores[k_idx] = -1.0e9;
    }
    workgroupBarrier();

    var local_max: f32 = -3.4028235e+38;
    for (var i = tid; i < valid_len; i = i + WG_SIZE_ATT) {
        local_max = max(local_max, wg_scores[i]);
    }
    if (tid < 8u) { wg_partial[tid] = -3.4028235e+38; }
    workgroupBarrier();
    let bucket = tid / 32u;
    if (tid % 32u == 0u) {
        wg_partial[bucket] = max(wg_partial[bucket], local_max);
    }
    workgroupBarrier();
    if (tid == 0u) {
        var m: f32 = -3.4028235e+38;
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            m = max(m, wg_partial[i]);
        }
        wg_max = m;
    }
    workgroupBarrier();
    let max_val = wg_max;

    var local_sum: f32 = 0.0;
    for (var i = tid; i < valid_len; i = i + WG_SIZE_ATT) {
        let e = exp(wg_scores[i] - max_val);
        wg_scores[i] = e;
        local_sum = local_sum + e;
    }
    if (tid < 8u) { wg_partial[tid] = 0.0; }
    workgroupBarrier();
    if (tid % 32u == 0u) {
        wg_partial[bucket] = wg_partial[bucket] + local_sum;
    }
    workgroupBarrier();
    if (tid == 0u) {
        var s: f32 = 0.0;
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            s = s + wg_partial[i];
        }
        wg_sum = s;
    }
    workgroupBarrier();
    let inv_sum = 1.0 / wg_sum;
    for (var i = tid; i < valid_len; i = i + WG_SIZE_ATT) {
        wg_scores[i] = wg_scores[i] * inv_sum;
    }
    workgroupBarrier();

    let out_base = (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    for (var d = tid; d < head_dim; d = d + WG_SIZE_ATT) {
        var acc: f32 = 0.0;
        for (var k: u32 = 0u; k < valid_len; k = k + 1u) {
            acc = acc + wg_scores[k] * V[V_head_base + k * head_dim + d];
        }
        Output[out_base + d] = acc;
    }
}
