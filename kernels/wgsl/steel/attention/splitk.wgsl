// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Split-K Attention in WGSL
// Parallelizes attention computation across the K dimension
// Works on WebGPU (Metal/Vulkan/D3D12 via Dawn/wgpu)

// ============================================================================
// Configuration Constants
// ============================================================================

const BLOCK_SIZE: u32 = 256u;
const BLOCK_M: u32 = 64u;       // Query tile size
const BLOCK_N: u32 = 64u;       // Key/Value tile size per split
const HEAD_DIM: u32 = 64u;

// ============================================================================
// Bindings
// ============================================================================

struct SplitKAttentionParams {
    batch_size: u32,
    num_heads: u32,
    seq_len_q: u32,
    seq_len_kv: u32,
    head_dim: u32,
    num_splits: u32,
    scale: f32,
    is_causal: u32,
}

// Query: [batch, num_heads, seq_len_q, head_dim]
// Key: [batch, num_heads, seq_len_kv, head_dim]
// Value: [batch, num_heads, seq_len_kv, head_dim]

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> partial_output: array<f32>;  // [batch, num_heads, num_splits, seq_len_q, head_dim]
@group(0) @binding(4) var<storage, read_write> partial_lse: array<f32>;     // [batch, num_heads, num_splits, seq_len_q]
@group(0) @binding(5) var<storage, read_write> partial_max: array<f32>;     // [batch, num_heads, num_splits, seq_len_q]
@group(0) @binding(6) var<storage, read_write> output: array<f32>;          // [batch, num_heads, seq_len_q, head_dim]
@group(0) @binding(7) var<uniform> params: SplitKAttentionParams;

// Shared memory
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
// Split-K Attention Partial Kernel
// Each workgroup handles one split of the KV sequence
// ============================================================================

@compute @workgroup_size(256)
fn splitk_attention_partial(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z / params.num_heads;
    let head_idx = wid.z % params.num_heads;
    let q_tile_idx = wid.x;
    let split_idx = wid.y;

    if (batch_idx >= params.batch_size) { return; }

    let seq_len_q = params.seq_len_q;
    let seq_len_kv = params.seq_len_kv;
    let head_dim = params.head_dim;
    let num_splits = params.num_splits;
    let scale = params.scale;
    let is_causal = params.is_causal == 1u;

    // Calculate KV range for this split
    let kv_per_split = (seq_len_kv + num_splits - 1u) / num_splits;
    let kv_start = split_idx * kv_per_split;
    let kv_end = min(kv_start + kv_per_split, seq_len_kv);

    if (kv_start >= seq_len_kv) {
        // This split has no work - store sentinel values
        let q_start = q_tile_idx * BLOCK_M;
        for (var i = tid; i < BLOCK_M; i += BLOCK_SIZE) {
            let q_idx = q_start + i;
            if (q_idx < seq_len_q) {
                let stat_idx = ((batch_idx * params.num_heads + head_idx) * num_splits + split_idx) * seq_len_q + q_idx;
                partial_max[stat_idx] = -3.402823e+38;
                partial_lse[stat_idx] = 0.0;
            }
        }
        return;
    }

    let q_start = q_tile_idx * BLOCK_M;
    let q_end = min(q_start + BLOCK_M, seq_len_q);
    let num_q = q_end - q_start;

    // Initialize running statistics
    if (tid < BLOCK_M) {
        row_max[tid] = -3.402823e+38;
        row_sum[tid] = 0.0;
    }
    workgroupBarrier();

    // Load Q tile to registers
    var Q_local: array<f32, 64>;
    let q_local_idx = tid % num_q;
    if (q_local_idx < num_q) {
        for (var d = 0u; d < head_dim; d++) {
            let q_idx = idx_bhsd(batch_idx, head_idx, q_start + q_local_idx, d,
                                 params.num_heads, seq_len_q, head_dim);
            Q_local[d] = Q[q_idx];
        }
    }

    // Output accumulator
    var O_acc: array<f32, 64>;
    for (var d = 0u; d < 64u; d++) {
        O_acc[d] = 0.0;
    }

    // Number of KV tiles in this split
    let num_kv_tiles = (kv_end - kv_start + BLOCK_N - 1u) / BLOCK_N;

    // Iterate over KV tiles within this split
    for (var kv_tile = 0u; kv_tile < num_kv_tiles; kv_tile++) {
        let k_start_local = kv_start + kv_tile * BLOCK_N;
        let k_end_local = min(k_start_local + BLOCK_N, kv_end);
        let num_kv = k_end_local - k_start_local;

        // Load K tile to shared memory
        for (var i = tid; i < num_kv * head_dim; i += BLOCK_SIZE) {
            let k_local = i / head_dim;
            let d = i % head_dim;
            let k_idx = idx_bhsd(batch_idx, head_idx, k_start_local + k_local, d,
                                 params.num_heads, seq_len_kv, head_dim);
            K_tile[k_local * head_dim + d] = K[k_idx];
        }

        // Load V tile to shared memory
        for (var i = tid; i < num_kv * head_dim; i += BLOCK_SIZE) {
            let v_local = i / head_dim;
            let d = i % head_dim;
            let v_idx = idx_bhsd(batch_idx, head_idx, k_start_local + v_local, d,
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
                let k_pos = k_start_local + k_local;
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
            row_sum[i] += local_sum;
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

    // Store partial results for this split
    for (var i = tid; i < num_q; i += BLOCK_SIZE) {
        let q_idx = q_start + i;
        if (q_idx < seq_len_q) {
            let stat_idx = ((batch_idx * params.num_heads + head_idx) * num_splits + split_idx) * seq_len_q + q_idx;
            partial_max[stat_idx] = row_max[i];
            partial_lse[stat_idx] = row_sum[i];
        }
    }

    // Store partial output (unnormalized)
    for (var i = tid; i < num_q * head_dim; i += BLOCK_SIZE) {
        let q_local = i / head_dim;
        let d = i % head_dim;
        let q_idx = q_start + q_local;

        if (q_idx < seq_len_q) {
            let out_idx = (((batch_idx * params.num_heads + head_idx) * num_splits + split_idx) * seq_len_q + q_idx) * head_dim + d;
            partial_output[out_idx] = O_acc[d];
        }
    }
}

// ============================================================================
// Split-K Attention Reduction Kernel
// Combines partial results from all splits
// ============================================================================

@compute @workgroup_size(256)
fn splitk_attention_reduce(
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
    let head_dim = params.head_dim;
    let num_splits = params.num_splits;

    let q_start = q_tile_idx * BLOCK_M;
    let q_end = min(q_start + BLOCK_M, seq_len_q);
    let num_q = q_end - q_start;

    // Each thread handles one query position
    for (var q_local = tid; q_local < num_q; q_local += BLOCK_SIZE) {
        let q_idx = q_start + q_local;

        // Find global max across all splits
        var global_max: f32 = -3.402823e+38;
        for (var s = 0u; s < num_splits; s++) {
            let stat_idx = ((batch_idx * params.num_heads + head_idx) * num_splits + s) * seq_len_q + q_idx;
            global_max = max(global_max, partial_max[stat_idx]);
        }

        // Compute rescaled global sum
        var global_sum: f32 = 0.0;
        for (var s = 0u; s < num_splits; s++) {
            let stat_idx = ((batch_idx * params.num_heads + head_idx) * num_splits + s) * seq_len_q + q_idx;
            let m = partial_max[stat_idx];
            let e = partial_lse[stat_idx];
            if (m > -3.402823e+38) {
                global_sum += e * safe_exp(m - global_max);
            }
        }

        // Combine partial outputs
        for (var d = 0u; d < head_dim; d++) {
            var acc: f32 = 0.0;

            for (var s = 0u; s < num_splits; s++) {
                let stat_idx = ((batch_idx * params.num_heads + head_idx) * num_splits + s) * seq_len_q + q_idx;
                let m = partial_max[stat_idx];
                let e = partial_lse[stat_idx];

                if (m > -3.402823e+38 && global_sum > 0.0) {
                    let weight = e * safe_exp(m - global_max) / global_sum;
                    let part_idx = (((batch_idx * params.num_heads + head_idx) * num_splits + s) * seq_len_q + q_idx) * head_dim + d;
                    acc += weight * partial_output[part_idx];
                }
            }

            let out_idx = idx_bhsd(batch_idx, head_idx, q_idx, d,
                                   params.num_heads, seq_len_q, head_dim);
            output[out_idx] = acc;
        }
    }
}

// ============================================================================
// Split-K Attention with Variable Split Sizes
// For unbalanced workloads
// ============================================================================

struct VariableSplitParams {
    batch_size: u32,
    num_heads: u32,
    seq_len_q: u32,
    seq_len_kv: u32,
    head_dim: u32,
    num_splits: u32,
    scale: f32,
    is_causal: u32,
}

@group(0) @binding(8) var<storage, read> split_offsets: array<u32>;  // [num_splits + 1]
@group(0) @binding(9) var<uniform> var_params: VariableSplitParams;

@compute @workgroup_size(256)
fn splitk_attention_variable_partial(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z / var_params.num_heads;
    let head_idx = wid.z % var_params.num_heads;
    let q_tile_idx = wid.x;
    let split_idx = wid.y;

    if (batch_idx >= var_params.batch_size) { return; }

    let seq_len_q = var_params.seq_len_q;
    let seq_len_kv = var_params.seq_len_kv;
    let head_dim = var_params.head_dim;
    let num_splits = var_params.num_splits;
    let scale = var_params.scale;
    let is_causal = var_params.is_causal == 1u;

    // Get variable split range from offsets
    let kv_start = split_offsets[split_idx];
    let kv_end = split_offsets[split_idx + 1u];

    if (kv_start >= kv_end || kv_start >= seq_len_kv) {
        // Empty split
        let q_start = q_tile_idx * BLOCK_M;
        for (var i = tid; i < BLOCK_M; i += BLOCK_SIZE) {
            let q_idx = q_start + i;
            if (q_idx < seq_len_q) {
                let stat_idx = ((batch_idx * var_params.num_heads + head_idx) * num_splits + split_idx) * seq_len_q + q_idx;
                partial_max[stat_idx] = -3.402823e+38;
                partial_lse[stat_idx] = 0.0;
            }
        }
        return;
    }

    let q_start = q_tile_idx * BLOCK_M;
    let q_end = min(q_start + BLOCK_M, seq_len_q);
    let num_q = q_end - q_start;

    // Initialize
    if (tid < BLOCK_M) {
        row_max[tid] = -3.402823e+38;
        row_sum[tid] = 0.0;
    }
    workgroupBarrier();

    // Load Q
    var Q_local: array<f32, 64>;
    let q_local_idx = tid % num_q;
    if (q_local_idx < num_q) {
        for (var d = 0u; d < head_dim; d++) {
            let q_idx = idx_bhsd(batch_idx, head_idx, q_start + q_local_idx, d,
                                 var_params.num_heads, seq_len_q, head_dim);
            Q_local[d] = Q[q_idx];
        }
    }

    var O_acc: array<f32, 64>;
    for (var d = 0u; d < 64u; d++) {
        O_acc[d] = 0.0;
    }

    // Process KV in this split
    let num_kv_tiles = (kv_end - kv_start + BLOCK_N - 1u) / BLOCK_N;

    for (var kv_tile = 0u; kv_tile < num_kv_tiles; kv_tile++) {
        let k_start_local = kv_start + kv_tile * BLOCK_N;
        let k_end_local = min(k_start_local + BLOCK_N, kv_end);
        let num_kv = k_end_local - k_start_local;

        // Load K, V tiles
        for (var i = tid; i < num_kv * head_dim; i += BLOCK_SIZE) {
            let k_local = i / head_dim;
            let d = i % head_dim;
            let k_idx = idx_bhsd(batch_idx, head_idx, k_start_local + k_local, d,
                                 var_params.num_heads, seq_len_kv, head_dim);
            K_tile[k_local * head_dim + d] = K[k_idx];
            V_tile[k_local * head_dim + d] = V[k_idx];
        }
        workgroupBarrier();

        // Compute scores
        for (var i = tid; i < num_q * num_kv; i += BLOCK_SIZE) {
            let q_local = i / num_kv;
            let k_local = i % num_kv;

            var dot: f32 = 0.0;
            for (var d = 0u; d < head_dim; d++) {
                dot += Q_local[d] * K_tile[k_local * head_dim + d];
            }
            dot *= scale;

            if (is_causal) {
                let q_pos = q_start + q_local;
                let k_pos = k_start_local + k_local;
                if (k_pos > q_pos) {
                    dot = -3.402823e+38;
                }
            }

            S_tile[q_local * BLOCK_N + k_local] = dot;
        }
        workgroupBarrier();

        // Softmax
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

        // Accumulate P @ V
        for (var i = tid; i < num_q * head_dim; i += BLOCK_SIZE) {
            let q_local = i / head_dim;
            let d = i % head_dim;

            var pv_dot: f32 = 0.0;
            for (var k_local = 0u; k_local < num_kv; k_local++) {
                pv_dot += S_tile[q_local * BLOCK_N + k_local] * V_tile[k_local * head_dim + d];
            }
            O_acc[d] += pv_dot;
        }
        workgroupBarrier();
    }

    // Store partial results
    for (var i = tid; i < num_q; i += BLOCK_SIZE) {
        let q_idx = q_start + i;
        if (q_idx < seq_len_q) {
            let stat_idx = ((batch_idx * var_params.num_heads + head_idx) * num_splits + split_idx) * seq_len_q + q_idx;
            partial_max[stat_idx] = row_max[i];
            partial_lse[stat_idx] = row_sum[i];
        }
    }

    for (var i = tid; i < num_q * head_dim; i += BLOCK_SIZE) {
        let q_local = i / head_dim;
        let d = i % head_dim;
        let q_idx = q_start + q_local;

        if (q_idx < seq_len_q) {
            let out_idx = (((batch_idx * var_params.num_heads + head_idx) * num_splits + split_idx) * seq_len_q + q_idx) * head_dim + d;
            partial_output[out_idx] = O_acc[d];
        }
    }
}
