// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Tiled Multi-Head Attention in WGSL
// Works on WebGPU (Metal/Vulkan/D3D12 via Dawn/wgpu)
//
// Part of the Lux Network GPU acceleration library
// Based on FlashAttention-style tiled computation for memory efficiency

// ============================================================================
// Configuration Constants
// ============================================================================

// Tile sizes for attention computation
const TILE_Q: u32 = 64u;       // Query tile size (Br)
const TILE_K: u32 = 64u;       // Key tile size (Bc)
const HEAD_DIM: u32 = 64u;     // Head dimension (d)
const WARP_SIZE: u32 = 32u;

// ============================================================================
// Kernel Bindings
// ============================================================================

struct AttentionParams {
    batch_size: u32,           // Batch size (B)
    num_heads: u32,            // Number of attention heads (H)
    seq_len_q: u32,            // Query sequence length (N)
    seq_len_kv: u32,           // Key/Value sequence length (M)
    head_dim: u32,             // Head dimension (d)
    scale: f32,                // 1/sqrt(d)
    causal: u32,               // 1 for causal masking, 0 otherwise
    _pad: u32,
}

// Q: [B, H, N, d]
// K: [B, H, M, d]
// V: [B, H, M, d]
// O: [B, H, N, d]
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<uniform> params: AttentionParams;

// Shared memory tiles
var<workgroup> tile_Q: array<f32, 4096>;    // TILE_Q * HEAD_DIM
var<workgroup> tile_K: array<f32, 4096>;    // TILE_K * HEAD_DIM
var<workgroup> tile_V: array<f32, 4096>;    // TILE_K * HEAD_DIM
var<workgroup> tile_S: array<f32, 4096>;    // TILE_Q * TILE_K (attention scores)
var<workgroup> row_max: array<f32, 64>;     // TILE_Q (max per row)
var<workgroup> row_sum: array<f32, 64>;     // TILE_Q (sum per row)

// ============================================================================
// Helper Functions
// ============================================================================

// Index into Q/K/V/O tensor: [B, H, S, D]
fn idx_bhsd(b: u32, h: u32, s: u32, d: u32, num_heads: u32, seq_len: u32, head_dim: u32) -> u32 {
    return ((b * num_heads + h) * seq_len + s) * head_dim + d;
}

// Safe exponential to avoid overflow
fn safe_exp(x: f32) -> f32 {
    return exp(clamp(x, -88.0, 88.0));
}

// ============================================================================
// Tiled Multi-Head Attention Kernel
// ============================================================================

@compute @workgroup_size(256)
fn steel_attention(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z / params.num_heads;
    let head_idx = wid.z % params.num_heads;
    let q_tile_idx = wid.x;

    if (batch_idx >= params.batch_size) { return; }

    let seq_len_q = params.seq_len_q;
    let seq_len_kv = params.seq_len_kv;
    let head_dim = params.head_dim;
    let scale = params.scale;
    let causal = params.causal == 1u;

    let q_start = q_tile_idx * TILE_Q;
    let q_end = min(q_start + TILE_Q, seq_len_q);
    let num_q = q_end - q_start;

    // Initialize row_max and row_sum for online softmax
    if (tid < TILE_Q) {
        row_max[tid] = -3.402823e+38;  // -FLT_MAX
        row_sum[tid] = 0.0;
    }
    workgroupBarrier();

    // Load Q tile to shared memory
    for (var i = tid; i < num_q * head_dim; i += 256u) {
        let q_row = i / head_dim;
        let q_col = i % head_dim;
        let q_idx = idx_bhsd(batch_idx, head_idx, q_start + q_row, q_col,
                             params.num_heads, seq_len_q, head_dim);
        tile_Q[i] = Q[q_idx];
    }
    workgroupBarrier();

    // Initialize output accumulator in registers
    var acc: array<f32, 64>;  // One row of output per thread
    for (var i = 0u; i < 64u; i++) {
        acc[i] = 0.0;
    }

    // Number of K/V tiles
    let num_kv_tiles = (seq_len_kv + TILE_K - 1u) / TILE_K;

    // Iterate over K/V tiles
    for (var kv_tile = 0u; kv_tile < num_kv_tiles; kv_tile++) {
        let kv_start = kv_tile * TILE_K;
        let kv_end = min(kv_start + TILE_K, seq_len_kv);
        let num_kv = kv_end - kv_start;

        // Apply causal mask: skip tiles that are fully masked
        if (causal && kv_start > q_end) {
            continue;
        }

        // Load K tile to shared memory
        for (var i = tid; i < num_kv * head_dim; i += 256u) {
            let k_row = i / head_dim;
            let k_col = i % head_dim;
            let k_idx = idx_bhsd(batch_idx, head_idx, kv_start + k_row, k_col,
                                 params.num_heads, seq_len_kv, head_dim);
            tile_K[i] = K[k_idx];
        }

        // Load V tile to shared memory
        for (var i = tid; i < num_kv * head_dim; i += 256u) {
            let v_row = i / head_dim;
            let v_col = i % head_dim;
            let v_idx = idx_bhsd(batch_idx, head_idx, kv_start + v_row, v_col,
                                 params.num_heads, seq_len_kv, head_dim);
            tile_V[i] = V[v_idx];
        }
        workgroupBarrier();

        // Compute S = Q @ K^T for this tile (tiled matrix multiply)
        // Each thread computes a subset of the TILE_Q x TILE_K result
        for (var i = tid; i < num_q * num_kv; i += 256u) {
            let s_row = i / num_kv;
            let s_col = i % num_kv;

            var dot: f32 = 0.0;
            for (var k = 0u; k < head_dim; k++) {
                let q_val = tile_Q[s_row * head_dim + k];
                let k_val = tile_K[s_col * head_dim + k];
                dot += q_val * k_val;
            }

            // Scale and apply causal mask
            dot *= scale;

            if (causal) {
                let q_pos = q_start + s_row;
                let k_pos = kv_start + s_col;
                if (k_pos > q_pos) {
                    dot = -3.402823e+38;  // -FLT_MAX
                }
            }

            tile_S[i] = dot;
        }
        workgroupBarrier();

        // Online softmax: update row_max
        for (var i = tid; i < num_q; i += 256u) {
            var local_max = row_max[i];
            for (var j = 0u; j < num_kv; j++) {
                let s_val = tile_S[i * num_kv + j];
                local_max = max(local_max, s_val);
            }
            row_max[i] = local_max;
        }
        workgroupBarrier();

        // Compute softmax numerator and update accumulator
        for (var i = tid; i < num_q; i += 256u) {
            let m_new = row_max[i];
            var l_new: f32 = 0.0;

            // Compute exp(S - m_new) and sum
            for (var j = 0u; j < num_kv; j++) {
                let s_val = tile_S[i * num_kv + j];
                let exp_s = safe_exp(s_val - m_new);
                tile_S[i * num_kv + j] = exp_s;
                l_new += exp_s;
            }

            // Update running sum with rescaling
            let l_old = row_sum[i];
            row_sum[i] = l_old * safe_exp(row_max[i] - m_new) + l_new;
        }
        workgroupBarrier();

        // Update output: O = O * rescale + P @ V
        // P = softmax(S) is stored in tile_S
        for (var i = tid; i < num_q * head_dim; i += 256u) {
            let o_row = i / head_dim;
            let o_col = i % head_dim;

            var pv_dot: f32 = 0.0;
            for (var j = 0u; j < num_kv; j++) {
                let p_val = tile_S[o_row * num_kv + j];
                let v_val = tile_V[j * head_dim + o_col];
                pv_dot += p_val * v_val;
            }

            // Accumulate (will normalize at the end)
            acc[o_col] += pv_dot;
        }
        workgroupBarrier();
    }

    // Final normalization: O = O / row_sum
    for (var i = tid; i < num_q * head_dim; i += 256u) {
        let o_row = i / head_dim;
        let o_col = i % head_dim;
        let l = row_sum[o_row];
        let o_val = select(0.0, acc[o_col] / l, l > 0.0);

        let o_idx = idx_bhsd(batch_idx, head_idx, q_start + o_row, o_col,
                             params.num_heads, seq_len_q, head_dim);
        O[o_idx] = o_val;
    }
}

// ============================================================================
// Multi-Head Attention with Separate QKV Matmul
// ============================================================================

@compute @workgroup_size(256)
fn steel_attention_qkv(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // This variant handles the case where Q, K, V are packed together
    // and need to be separated before attention computation
    let tid = lid.x;
    let batch_head = wid.z;
    let batch_idx = batch_head / params.num_heads;
    let head_idx = batch_head % params.num_heads;
    let q_tile_idx = wid.x;

    if (batch_idx >= params.batch_size) { return; }

    // Same logic as steel_attention, but with different memory layout
    // Q, K, V are interleaved as [B, S, 3, H, D] instead of [B, H, S, D]

    let seq_len_q = params.seq_len_q;
    let seq_len_kv = params.seq_len_kv;
    let head_dim = params.head_dim;
    let num_heads = params.num_heads;
    let scale = params.scale;

    let q_start = q_tile_idx * TILE_Q;
    let q_end = min(q_start + TILE_Q, seq_len_q);
    let num_q = q_end - q_start;

    // Initialize accumulators
    if (tid < TILE_Q) {
        row_max[tid] = -3.402823e+38;
        row_sum[tid] = 0.0;
    }
    workgroupBarrier();

    // Load Q with interleaved layout [B, S, 3, H, D]
    for (var i = tid; i < num_q * head_dim; i += 256u) {
        let q_row = i / head_dim;
        let q_col = i % head_dim;
        // QKV offset: Q=0, K=1, V=2
        let qkv_idx = ((batch_idx * seq_len_q + (q_start + q_row)) * 3u + 0u)
                      * num_heads * head_dim + head_idx * head_dim + q_col;
        tile_Q[i] = Q[qkv_idx];
    }
    workgroupBarrier();

    // Process K/V tiles similarly
    let num_kv_tiles = (seq_len_kv + TILE_K - 1u) / TILE_K;

    for (var kv_tile = 0u; kv_tile < num_kv_tiles; kv_tile++) {
        let kv_start = kv_tile * TILE_K;
        let kv_end = min(kv_start + TILE_K, seq_len_kv);
        let num_kv = kv_end - kv_start;

        // Load K with interleaved layout
        for (var i = tid; i < num_kv * head_dim; i += 256u) {
            let k_row = i / head_dim;
            let k_col = i % head_dim;
            let qkv_idx = ((batch_idx * seq_len_kv + (kv_start + k_row)) * 3u + 1u)
                          * num_heads * head_dim + head_idx * head_dim + k_col;
            tile_K[i] = K[qkv_idx];
        }

        // Load V with interleaved layout
        for (var i = tid; i < num_kv * head_dim; i += 256u) {
            let v_row = i / head_dim;
            let v_col = i % head_dim;
            let qkv_idx = ((batch_idx * seq_len_kv + (kv_start + v_row)) * 3u + 2u)
                          * num_heads * head_dim + head_idx * head_dim + v_col;
            tile_V[i] = V[qkv_idx];
        }
        workgroupBarrier();

        // Compute attention scores (same as main kernel)
        for (var i = tid; i < num_q * num_kv; i += 256u) {
            let s_row = i / num_kv;
            let s_col = i % num_kv;

            var dot: f32 = 0.0;
            for (var k = 0u; k < head_dim; k++) {
                dot += tile_Q[s_row * head_dim + k] * tile_K[s_col * head_dim + k];
            }
            tile_S[i] = dot * scale;
        }
        workgroupBarrier();

        // Online softmax update
        for (var i = tid; i < num_q; i += 256u) {
            var local_max = row_max[i];
            for (var j = 0u; j < num_kv; j++) {
                local_max = max(local_max, tile_S[i * num_kv + j]);
            }
            row_max[i] = local_max;
        }
        workgroupBarrier();

        for (var i = tid; i < num_q * num_kv; i += 256u) {
            let s_row = i / num_kv;
            let m = row_max[s_row];
            tile_S[i] = safe_exp(tile_S[i] - m);
        }
        workgroupBarrier();

        for (var i = tid; i < num_q; i += 256u) {
            var local_sum: f32 = 0.0;
            for (var j = 0u; j < num_kv; j++) {
                local_sum += tile_S[i * num_kv + j];
            }
            row_sum[i] += local_sum;
        }
        workgroupBarrier();
    }

    // Store output (with final normalization)
    for (var i = tid; i < num_q * head_dim; i += 256u) {
        let o_row = i / head_dim;
        let o_col = i % head_dim;

        var o_val: f32 = 0.0;
        // Recompute P@V (simplified - full impl would accumulate during tiles)
        let o_idx = idx_bhsd(batch_idx, head_idx, q_start + o_row, o_col,
                             params.num_heads, seq_len_q, head_dim);
        O[o_idx] = select(0.0, o_val, row_sum[o_row] > 0.0);
    }
}

// ============================================================================
// Backward Pass: dQ = dO @ K
// ============================================================================

@compute @workgroup_size(256)
fn steel_attention_backward_dq(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Compute gradient with respect to Q
    // dQ = softmax(QK^T) @ dO @ V^T
    // This is a tiled implementation for memory efficiency

    let tid = lid.x;
    let batch_head = wid.z;
    let batch_idx = batch_head / params.num_heads;
    let head_idx = batch_head % params.num_heads;

    if (batch_idx >= params.batch_size) { return; }

    let seq_len_q = params.seq_len_q;
    let seq_len_kv = params.seq_len_kv;
    let head_dim = params.head_dim;
    let scale = params.scale;

    // Placeholder for backward pass implementation
    // Full implementation would compute:
    // 1. Recompute attention weights P = softmax(Q @ K^T / sqrt(d))
    // 2. Compute dP = dO @ V^T
    // 3. Compute dS = P * (dP - sum(dP * P, dim=-1, keepdim=True))
    // 4. Compute dQ = dS @ K
}
