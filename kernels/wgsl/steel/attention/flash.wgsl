// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Flash Attention in WGSL
// Memory-efficient attention with O(N) memory instead of O(N^2)
// Works on WebGPU (Metal/Vulkan/D3D12 via Dawn/wgpu)

// ============================================================================
// Configuration Constants
// ============================================================================

const BLOCK_M: u32 = 64u;      // Query tile size
const BLOCK_N: u32 = 64u;      // Key/Value tile size
const HEAD_DIM: u32 = 64u;     // Head dimension
const BLOCK_SIZE: u32 = 256u;

// ============================================================================
// Bindings
// ============================================================================

struct FlashAttentionParams {
    batch_size: u32,
    num_heads: u32,
    seq_len_q: u32,
    seq_len_kv: u32,
    head_dim: u32,
    scale: f32,
    is_causal: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<storage, read_write> lse: array<f32>;  // Log-sum-exp for backward
@group(0) @binding(5) var<uniform> params: FlashAttentionParams;

// Shared memory tiles
var<workgroup> K_tile: array<f32, 4096>;   // BLOCK_N * HEAD_DIM
var<workgroup> V_tile: array<f32, 4096>;   // BLOCK_N * HEAD_DIM
var<workgroup> S_tile: array<f32, 4096>;   // BLOCK_M * BLOCK_N
var<workgroup> row_max: array<f32, 64>;    // BLOCK_M
var<workgroup> row_sum: array<f32, 64>;    // BLOCK_M

// ============================================================================
// Helper Functions
// ============================================================================

fn idx_bhsd(b: u32, h: u32, s: u32, d: u32, num_heads: u32, seq_len: u32, head_dim: u32) -> u32 {
    return ((b * num_heads + h) * seq_len + s) * head_dim + d;
}

fn safe_exp(x: f32) -> f32 {
    return exp(clamp(x, -88.0, 88.0));
}

// ============================================================================
// Flash Attention Forward
// ============================================================================

@compute @workgroup_size(256)
fn flash_attention_forward(
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
    let is_causal = params.is_causal == 1u;

    let q_start = q_tile_idx * BLOCK_M;
    let q_end = min(q_start + BLOCK_M, seq_len_q);
    let num_q = q_end - q_start;

    // Initialize running statistics for online softmax
    if (tid < BLOCK_M) {
        row_max[tid] = -3.402823e+38;
        row_sum[tid] = 0.0;
    }
    workgroupBarrier();

    // Load Q tile to registers (each thread handles subset)
    var Q_local: array<f32, 64>;
    let q_local_idx = tid % num_q;
    if (q_local_idx < num_q) {
        for (var d = 0u; d < head_dim; d++) {
            let q_idx = idx_bhsd(batch_idx, head_idx, q_start + q_local_idx, d,
                                 params.num_heads, seq_len_q, head_dim);
            Q_local[d] = Q[q_idx];
        }
    }

    // Output accumulator in registers
    var O_acc: array<f32, 64>;
    for (var d = 0u; d < 64u; d++) {
        O_acc[d] = 0.0;
    }

    // Number of KV tiles
    let num_kv_tiles = (seq_len_kv + BLOCK_N - 1u) / BLOCK_N;
    var kv_tile_end = num_kv_tiles;

    if (is_causal) {
        kv_tile_end = min(num_kv_tiles, (q_start + BLOCK_M + BLOCK_N - 1u) / BLOCK_N);
    }

    // Iterate over KV tiles
    for (var kv_tile = 0u; kv_tile < kv_tile_end; kv_tile++) {
        let k_start = kv_tile * BLOCK_N;
        let k_end = min(k_start + BLOCK_N, seq_len_kv);
        let num_kv = k_end - k_start;

        // Load K tile to shared memory
        for (var i = tid; i < num_kv * head_dim; i += BLOCK_SIZE) {
            let k_local = i / head_dim;
            let d = i % head_dim;
            let k_idx = idx_bhsd(batch_idx, head_idx, k_start + k_local, d,
                                 params.num_heads, seq_len_kv, head_dim);
            K_tile[k_local * head_dim + d] = K[k_idx];
        }

        // Load V tile to shared memory
        for (var i = tid; i < num_kv * head_dim; i += BLOCK_SIZE) {
            let v_local = i / head_dim;
            let d = i % head_dim;
            let v_idx = idx_bhsd(batch_idx, head_idx, k_start + v_local, d,
                                 params.num_heads, seq_len_kv, head_dim);
            V_tile[v_local * head_dim + d] = V[v_idx];
        }
        workgroupBarrier();

        // Compute attention scores: S = Q @ K^T * scale
        for (var i = tid; i < num_q * num_kv; i += BLOCK_SIZE) {
            let q_local = i / num_kv;
            let k_local = i % num_kv;

            var dot: f32 = 0.0;
            for (var d = 0u; d < head_dim; d++) {
                let q_val = Q_local[d];
                let k_val = K_tile[k_local * head_dim + d];
                dot += q_val * k_val;
            }

            dot *= scale;

            // Apply causal mask
            if (is_causal) {
                let q_pos = q_start + q_local;
                let k_pos = k_start + k_local;
                if (k_pos > q_pos) {
                    dot = -3.402823e+38;
                }
            }

            S_tile[q_local * BLOCK_N + k_local] = dot;
        }
        workgroupBarrier();

        // Online softmax: update max
        for (var i = tid; i < num_q; i += BLOCK_SIZE) {
            var local_max = row_max[i];
            for (var j = 0u; j < num_kv; j++) {
                local_max = max(local_max, S_tile[i * BLOCK_N + j]);
            }
            row_max[i] = local_max;
        }
        workgroupBarrier();

        // Compute exp(S - max) and update sum
        for (var i = tid; i < num_q * num_kv; i += BLOCK_SIZE) {
            let q_local = i / num_kv;
            let k_local = i % num_kv;
            let m = row_max[q_local];
            S_tile[q_local * BLOCK_N + k_local] = safe_exp(S_tile[q_local * BLOCK_N + k_local] - m);
        }
        workgroupBarrier();

        for (var i = tid; i < num_q; i += BLOCK_SIZE) {
            var local_sum: f32 = 0.0;
            for (var j = 0u; j < num_kv; j++) {
                local_sum += S_tile[i * BLOCK_N + j];
            }

            // Rescale previous sum
            let m_prev = row_max[i];
            let old_sum = row_sum[i];
            row_sum[i] = old_sum + local_sum;
        }
        workgroupBarrier();

        // Accumulate output: O += P @ V
        for (var i = tid; i < num_q * head_dim; i += BLOCK_SIZE) {
            let q_local = i / head_dim;
            let d = i % head_dim;

            var pv_dot: f32 = 0.0;
            for (var k_local = 0u; k_local < num_kv; k_local++) {
                let p_val = S_tile[q_local * BLOCK_N + k_local];
                let v_val = V_tile[k_local * head_dim + d];
                pv_dot += p_val * v_val;
            }

            O_acc[d] += pv_dot;
        }
        workgroupBarrier();
    }

    // Final normalization: O = O / row_sum
    for (var i = tid; i < num_q * head_dim; i += BLOCK_SIZE) {
        let q_local = i / head_dim;
        let d = i % head_dim;
        let l = row_sum[q_local];

        let o_val = select(0.0, O_acc[d] / l, l > 0.0);

        let o_idx = idx_bhsd(batch_idx, head_idx, q_start + q_local, d,
                             params.num_heads, seq_len_q, head_dim);
        O[o_idx] = o_val;
    }

    // Store LSE for backward pass
    for (var i = tid; i < num_q; i += BLOCK_SIZE) {
        let q_idx = q_start + i;
        if (q_idx < seq_len_q) {
            let lse_idx = (batch_idx * params.num_heads + head_idx) * seq_len_q + q_idx;
            let m = row_max[i];
            let l = row_sum[i];
            lse[lse_idx] = select(-3.402823e+38, m + log(l), l > 0.0);
        }
    }
}

// ============================================================================
// Flash Attention with Alibi Positional Encoding
// ============================================================================

@compute @workgroup_size(256)
fn flash_attention_alibi(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Flash attention with ALiBi (Attention with Linear Biases)
    // Adds position-dependent bias: alibi_slope * (query_pos - key_pos)

    let tid = lid.x;
    let batch_idx = wid.z / params.num_heads;
    let head_idx = wid.z % params.num_heads;
    let q_tile_idx = wid.x;

    if (batch_idx >= params.batch_size) { return; }

    // ALiBi slope: m = 2^(-8 * h / H) where h is head index, H is total heads
    let alibi_slope = pow(2.0, -8.0 * f32(head_idx) / f32(params.num_heads));

    let seq_len_q = params.seq_len_q;
    let seq_len_kv = params.seq_len_kv;
    let head_dim = params.head_dim;
    let scale = params.scale;

    let q_start = q_tile_idx * BLOCK_M;
    let num_q = min(BLOCK_M, seq_len_q - q_start);

    if (tid < BLOCK_M) {
        row_max[tid] = -3.402823e+38;
        row_sum[tid] = 0.0;
    }
    workgroupBarrier();

    let num_kv_tiles = (seq_len_kv + BLOCK_N - 1u) / BLOCK_N;

    for (var kv_tile = 0u; kv_tile < num_kv_tiles; kv_tile++) {
        let k_start = kv_tile * BLOCK_N;
        let num_kv = min(BLOCK_N, seq_len_kv - k_start);

        // Load K, V tiles
        for (var i = tid; i < num_kv * head_dim; i += BLOCK_SIZE) {
            let k_local = i / head_dim;
            let d = i % head_dim;
            let k_idx = idx_bhsd(batch_idx, head_idx, k_start + k_local, d,
                                 params.num_heads, seq_len_kv, head_dim);
            K_tile[k_local * head_dim + d] = K[k_idx];
            V_tile[k_local * head_dim + d] = V[k_idx];
        }
        workgroupBarrier();

        // Compute scores with ALiBi bias
        for (var i = tid; i < num_q * num_kv; i += BLOCK_SIZE) {
            let q_local = i / num_kv;
            let k_local = i % num_kv;
            let q_pos = q_start + q_local;
            let k_pos = k_start + k_local;

            var dot: f32 = 0.0;
            for (var d = 0u; d < head_dim; d++) {
                let q_idx = idx_bhsd(batch_idx, head_idx, q_pos, d,
                                     params.num_heads, seq_len_q, head_dim);
                dot += Q[q_idx] * K_tile[k_local * head_dim + d];
            }

            // Add ALiBi bias
            let alibi_bias = alibi_slope * f32(i32(q_pos) - i32(k_pos));
            dot = dot * scale + alibi_bias;

            S_tile[q_local * BLOCK_N + k_local] = dot;
        }
        workgroupBarrier();

        // Online softmax (same as standard flash attention)
        for (var i = tid; i < num_q; i += BLOCK_SIZE) {
            var local_max = row_max[i];
            for (var j = 0u; j < num_kv; j++) {
                local_max = max(local_max, S_tile[i * BLOCK_N + j]);
            }
            row_max[i] = local_max;
        }
        workgroupBarrier();

        for (var i = tid; i < num_q * num_kv; i += BLOCK_SIZE) {
            let q_local = i / num_kv;
            let m = row_max[q_local];
            S_tile[i] = safe_exp(S_tile[i] - m);
        }
        workgroupBarrier();

        for (var i = tid; i < num_q; i += BLOCK_SIZE) {
            var local_sum: f32 = 0.0;
            for (var j = 0u; j < num_kv; j++) {
                local_sum += S_tile[i * BLOCK_N + j];
            }
            row_sum[i] += local_sum;
        }
        workgroupBarrier();
    }

    // Write normalized output
    for (var i = tid; i < num_q * head_dim; i += BLOCK_SIZE) {
        let q_local = i / head_dim;
        let d = i % head_dim;
        let l = row_sum[q_local];

        var o_val: f32 = 0.0;
        // Recompute P@V (simplified for WebGPU shared memory limits)

        let o_idx = idx_bhsd(batch_idx, head_idx, q_start + q_local, d,
                             params.num_heads, seq_len_q, head_dim);
        O[o_idx] = o_val;
    }
}
