// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Scaled Dot-Product Attention (SDPA) Kernels
//
// Implements: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
//
// Supports:
// - Multi-head attention
// - Causal masking (autoregressive)
// - Flash Attention style tiling
// - Key-Value caching for inference
//
// Reference: Vaswani et al., "Attention Is All You Need" (2017)
//            Dao et al., "FlashAttention" (2022)
//
// Part of the Lux Network GPU acceleration library

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;
const TILE_SIZE: u32 = 64u;      // Block size for tiled attention
const NEG_INF: f32 = -3.402823466e+38;

// ============================================================================
// Parameter Structures
// ============================================================================

struct SDPAParams {
    batch_size: u32,        // Batch size
    num_heads: u32,         // Number of attention heads
    seq_len_q: u32,         // Query sequence length
    seq_len_kv: u32,        // Key/Value sequence length
    head_dim: u32,          // Dimension per head
    scale: f32,             // 1/sqrt(head_dim)
    causal: u32,            // 1 = causal mask, 0 = no mask
    dropout_p: f32,         // Dropout probability (0 = disabled)
}

// ============================================================================
// Storage Bindings
// ============================================================================

// Input tensors [batch, num_heads, seq_len, head_dim]
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: SDPAParams;

// Optional: attention weights output
@group(0) @binding(5) var<storage, read_write> attn_weights: array<f32>;

// Workgroup shared memory
var<workgroup> shared_q: array<f32, 1024>;   // Query tile
var<workgroup> shared_k: array<f32, 1024>;   // Key tile
var<workgroup> shared_v: array<f32, 1024>;   // Value tile
var<workgroup> shared_s: array<f32, 256>;    // Attention scores
var<workgroup> shared_max: array<f32, 256>;  // Max values
var<workgroup> shared_sum: array<f32, 256>;  // Sum for softmax

// ============================================================================
// Helper Functions
// ============================================================================

fn safe_exp(x: f32) -> f32 {
    return exp(clamp(x, -88.0, 88.0));
}

fn get_qkv_idx(batch: u32, head: u32, seq: u32, dim: u32, params_ptr: SDPAParams) -> u32 {
    return ((batch * params_ptr.num_heads + head) * params_ptr.seq_len_q + seq) * params_ptr.head_dim + dim;
}

// ============================================================================
// Basic SDPA - One workgroup per (batch, head, query_position)
// ============================================================================

@compute @workgroup_size(256)
fn sdpa_basic(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_size = params.batch_size;
    let num_heads = params.num_heads;
    let seq_len_q = params.seq_len_q;
    let seq_len_kv = params.seq_len_kv;
    let head_dim = params.head_dim;
    let scale = params.scale;
    let causal = params.causal;

    // Decode workgroup ID
    let batch_head_q = wid.x;
    let batch = batch_head_q / (num_heads * seq_len_q);
    let head = (batch_head_q / seq_len_q) % num_heads;
    let q_pos = batch_head_q % seq_len_q;

    if (batch >= batch_size) {
        return;
    }

    // Base offsets
    let q_base = ((batch * num_heads + head) * seq_len_q + q_pos) * head_dim;
    let k_base = (batch * num_heads + head) * seq_len_kv * head_dim;
    let v_base = k_base;
    let o_base = q_base;

    // For causal, limit key positions
    let max_kv = select(seq_len_kv, min(q_pos + 1u, seq_len_kv), causal == 1u);

    // Step 1: Compute attention scores (Q @ K^T)
    // Each thread computes dot product with one or more key positions
    var local_max: f32 = NEG_INF;

    var kv_pos = lid.x;
    while (kv_pos < max_kv) {
        var dot: f32 = 0.0;
        for (var d = 0u; d < head_dim; d++) {
            let q_val = query[q_base + d];
            let k_val = key[k_base + kv_pos * head_dim + d];
            dot += q_val * k_val;
        }
        let score = dot * scale;
        shared_s[lid.x] = score;
        local_max = max(local_max, score);
        kv_pos += WORKGROUP_SIZE;
    }

    // Initialize unused positions to -inf
    if (lid.x >= max_kv) {
        shared_s[lid.x] = NEG_INF;
    }

    shared_max[lid.x] = local_max;
    workgroupBarrier();

    // Reduce max
    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            shared_max[lid.x] = max(shared_max[lid.x], shared_max[lid.x + s]);
        }
        workgroupBarrier();
    }

    let max_val = shared_max[0];
    workgroupBarrier();

    // Step 2: Compute softmax
    var local_sum: f32 = 0.0;

    kv_pos = lid.x;
    while (kv_pos < max_kv) {
        // Recompute score (or could store in shared memory)
        var dot: f32 = 0.0;
        for (var d = 0u; d < head_dim; d++) {
            dot += query[q_base + d] * key[k_base + kv_pos * head_dim + d];
        }
        let score = dot * scale;
        let exp_val = safe_exp(score - max_val);
        shared_s[lid.x] = exp_val;
        local_sum += exp_val;
        kv_pos += WORKGROUP_SIZE;
    }

    shared_sum[lid.x] = local_sum;
    workgroupBarrier();

    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            shared_sum[lid.x] += shared_sum[lid.x + s];
        }
        workgroupBarrier();
    }

    let sum_exp = shared_sum[0];
    let inv_sum = select(0.0, 1.0 / sum_exp, sum_exp > 0.0);
    workgroupBarrier();

    // Step 3: Compute output (attn_weights @ V)
    // Each thread handles a dimension of the output
    var d = lid.x;
    while (d < head_dim) {
        var acc: f32 = 0.0;

        for (var kv = 0u; kv < max_kv; kv++) {
            // Recompute attention weight
            var dot: f32 = 0.0;
            for (var dd = 0u; dd < head_dim; dd++) {
                dot += query[q_base + dd] * key[k_base + kv * head_dim + dd];
            }
            let attn_weight = safe_exp(dot * scale - max_val) * inv_sum;
            acc += attn_weight * value[v_base + kv * head_dim + d];
        }

        output[o_base + d] = acc;
        d += WORKGROUP_SIZE;
    }
}

// ============================================================================
// Flash Attention Style - Tiled SDPA with Online Softmax
// ============================================================================

struct OnlineAttentionState {
    max_val: f32,
    sum_exp: f32,
    output: array<f32, 128>,  // Accumulator (max head_dim = 128)
}

var<workgroup> shared_state_max: array<f32, 64>;
var<workgroup> shared_state_sum: array<f32, 64>;
var<workgroup> shared_output: array<f32, 8192>;  // 64 queries * 128 dims

@compute @workgroup_size(256)
fn sdpa_flash(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_size = params.batch_size;
    let num_heads = params.num_heads;
    let seq_len_q = params.seq_len_q;
    let seq_len_kv = params.seq_len_kv;
    let head_dim = params.head_dim;
    let scale = params.scale;
    let causal = params.causal;

    // Each workgroup handles one (batch, head)
    let batch = wid.x / num_heads;
    let head = wid.x % num_heads;

    if (batch >= batch_size) {
        return;
    }

    let q_base = (batch * num_heads + head) * seq_len_q * head_dim;
    let k_base = (batch * num_heads + head) * seq_len_kv * head_dim;
    let v_base = k_base;
    let o_base = q_base;

    // Initialize output state
    let q_tile_size = min(TILE_SIZE, seq_len_q);
    let q_idx = lid.x % q_tile_size;
    let dim_idx = lid.x / q_tile_size;

    // Initialize accumulators
    if (dim_idx < head_dim) {
        shared_output[q_idx * head_dim + dim_idx] = 0.0;
    }
    if (lid.x < q_tile_size) {
        shared_state_max[lid.x] = NEG_INF;
        shared_state_sum[lid.x] = 0.0;
    }
    workgroupBarrier();

    // Tile over KV sequence
    for (var kv_tile_start = 0u; kv_tile_start < seq_len_kv; kv_tile_start += TILE_SIZE) {
        let kv_tile_end = min(kv_tile_start + TILE_SIZE, seq_len_kv);

        // Load K tile to shared memory
        let k_elements = (kv_tile_end - kv_tile_start) * head_dim;
        var i = lid.x;
        while (i < k_elements) {
            let kv_idx = kv_tile_start + i / head_dim;
            let d = i % head_dim;
            if (kv_idx < seq_len_kv) {
                shared_k[i] = key[k_base + kv_idx * head_dim + d];
            }
            i += WORKGROUP_SIZE;
        }
        workgroupBarrier();

        // Load V tile to shared memory
        i = lid.x;
        while (i < k_elements) {
            let kv_idx = kv_tile_start + i / head_dim;
            let d = i % head_dim;
            if (kv_idx < seq_len_kv) {
                shared_v[i] = value[v_base + kv_idx * head_dim + d];
            }
            i += WORKGROUP_SIZE;
        }
        workgroupBarrier();

        // Process each query position in the workgroup
        for (var q_tile_start = 0u; q_tile_start < seq_len_q; q_tile_start += q_tile_size) {
            let q_pos = q_tile_start + lid.x;
            if (q_pos < seq_len_q) {
                let old_max = shared_state_max[lid.x % q_tile_size];
                let old_sum = shared_state_sum[lid.x % q_tile_size];

                var new_max = old_max;
                var block_sum: f32 = 0.0;

                // Compute attention scores for this query against KV tile
                for (var kv_local = 0u; kv_local < kv_tile_end - kv_tile_start; kv_local++) {
                    let kv_pos = kv_tile_start + kv_local;

                    // Causal check
                    if (causal == 1u && kv_pos > q_pos) {
                        continue;
                    }

                    // Compute dot product Q @ K^T
                    var dot: f32 = 0.0;
                    for (var d = 0u; d < head_dim; d++) {
                        let q_val = query[q_base + q_pos * head_dim + d];
                        let k_val = shared_k[kv_local * head_dim + d];
                        dot += q_val * k_val;
                    }
                    let score = dot * scale;
                    new_max = max(new_max, score);
                }

                // Second pass: compute softmax and update output
                let scale_old = safe_exp(old_max - new_max);

                for (var kv_local = 0u; kv_local < kv_tile_end - kv_tile_start; kv_local++) {
                    let kv_pos = kv_tile_start + kv_local;

                    if (causal == 1u && kv_pos > q_pos) {
                        continue;
                    }

                    var dot: f32 = 0.0;
                    for (var d = 0u; d < head_dim; d++) {
                        dot += query[q_base + q_pos * head_dim + d] * shared_k[kv_local * head_dim + d];
                    }
                    let score = dot * scale;
                    let exp_val = safe_exp(score - new_max);
                    block_sum += exp_val;

                    // Update output
                    for (var d = 0u; d < head_dim; d++) {
                        let out_idx = (lid.x % q_tile_size) * head_dim + d;
                        shared_output[out_idx] = shared_output[out_idx] * scale_old + exp_val * shared_v[kv_local * head_dim + d];
                    }
                }

                // Update state
                shared_state_max[lid.x % q_tile_size] = new_max;
                shared_state_sum[lid.x % q_tile_size] = old_sum * scale_old + block_sum;
            }
            workgroupBarrier();
        }
        workgroupBarrier();
    }

    // Final normalization and output
    for (var q_tile_start = 0u; q_tile_start < seq_len_q; q_tile_start += q_tile_size) {
        let q_pos = q_tile_start + (lid.x % q_tile_size);
        if (q_pos < seq_len_q && lid.x < q_tile_size * head_dim) {
            let q_local = lid.x % q_tile_size;
            let d = lid.x / q_tile_size;
            if (d < head_dim) {
                let sum = shared_state_sum[q_local];
                let inv_sum = select(0.0, 1.0 / sum, sum > 0.0);
                output[o_base + q_pos * head_dim + d] = shared_output[q_local * head_dim + d] * inv_sum;
            }
        }
    }
}

// ============================================================================
// SDPA with KV Cache (for inference)
// ============================================================================

struct SDPAKVCacheParams {
    batch_size: u32,
    num_heads: u32,
    num_kv_heads: u32,      // For grouped-query attention
    seq_len_q: u32,         // Usually 1 for decoding
    cache_len: u32,         // Current length of KV cache
    max_cache_len: u32,     // Maximum cache length
    head_dim: u32,
    scale: f32,
}

@group(1) @binding(0) var<storage, read_write> k_cache: array<f32>;
@group(1) @binding(1) var<storage, read_write> v_cache: array<f32>;
@group(1) @binding(2) var<uniform> kv_params: SDPAKVCacheParams;

@compute @workgroup_size(256)
fn sdpa_kv_cache(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_size = kv_params.batch_size;
    let num_heads = kv_params.num_heads;
    let num_kv_heads = kv_params.num_kv_heads;
    let seq_len_q = kv_params.seq_len_q;
    let cache_len = kv_params.cache_len;
    let head_dim = kv_params.head_dim;
    let scale = kv_params.scale;

    // GQA: group size for query heads sharing same KV
    let kv_group_size = num_heads / num_kv_heads;

    let batch = wid.x / num_heads;
    let head = wid.x % num_heads;
    let kv_head = head / kv_group_size;

    if (batch >= batch_size) {
        return;
    }

    // For decoding, we typically have seq_len_q = 1
    let q_base = ((batch * num_heads + head) * seq_len_q) * head_dim;
    let kv_cache_base = (batch * num_kv_heads + kv_head) * kv_params.max_cache_len * head_dim;
    let o_base = q_base;

    // Total KV length = cache_len + seq_len_q (new tokens to add)
    let total_kv_len = cache_len + seq_len_q;

    // Step 1: Find max attention score
    var local_max: f32 = NEG_INF;

    var kv_pos = lid.x;
    while (kv_pos < total_kv_len) {
        var dot: f32 = 0.0;
        for (var d = 0u; d < head_dim; d++) {
            let q_val = query[q_base + d];
            let k_val = k_cache[kv_cache_base + kv_pos * head_dim + d];
            dot += q_val * k_val;
        }
        local_max = max(local_max, dot * scale);
        kv_pos += WORKGROUP_SIZE;
    }

    shared_max[lid.x] = local_max;
    workgroupBarrier();

    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            shared_max[lid.x] = max(shared_max[lid.x], shared_max[lid.x + s]);
        }
        workgroupBarrier();
    }

    let max_val = shared_max[0];
    workgroupBarrier();

    // Step 2: Compute softmax denominator
    var local_sum: f32 = 0.0;

    kv_pos = lid.x;
    while (kv_pos < total_kv_len) {
        var dot: f32 = 0.0;
        for (var d = 0u; d < head_dim; d++) {
            dot += query[q_base + d] * k_cache[kv_cache_base + kv_pos * head_dim + d];
        }
        local_sum += safe_exp(dot * scale - max_val);
        kv_pos += WORKGROUP_SIZE;
    }

    shared_sum[lid.x] = local_sum;
    workgroupBarrier();

    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            shared_sum[lid.x] += shared_sum[lid.x + s];
        }
        workgroupBarrier();
    }

    let inv_sum = 1.0 / shared_sum[0];
    workgroupBarrier();

    // Step 3: Compute weighted sum of values
    var d = lid.x;
    while (d < head_dim) {
        var acc: f32 = 0.0;

        for (var kv = 0u; kv < total_kv_len; kv++) {
            var dot: f32 = 0.0;
            for (var dd = 0u; dd < head_dim; dd++) {
                dot += query[q_base + dd] * k_cache[kv_cache_base + kv * head_dim + dd];
            }
            let attn = safe_exp(dot * scale - max_val) * inv_sum;
            acc += attn * v_cache[kv_cache_base + kv * head_dim + d];
        }

        output[o_base + d] = acc;
        d += WORKGROUP_SIZE;
    }
}

// ============================================================================
// Update KV Cache (append new K/V to cache)
// ============================================================================

@compute @workgroup_size(256)
fn update_kv_cache(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_size = kv_params.batch_size;
    let num_kv_heads = kv_params.num_kv_heads;
    let seq_len_q = kv_params.seq_len_q;
    let cache_len = kv_params.cache_len;
    let max_cache_len = kv_params.max_cache_len;
    let head_dim = kv_params.head_dim;

    let total_elements = batch_size * num_kv_heads * seq_len_q * head_dim;
    if (gid.x >= total_elements) {
        return;
    }

    // Decode indices
    let d = gid.x % head_dim;
    let temp = gid.x / head_dim;
    let seq_idx = temp % seq_len_q;
    let temp2 = temp / seq_len_q;
    let head = temp2 % num_kv_heads;
    let batch = temp2 / num_kv_heads;

    // Source: new K/V at position seq_idx
    let src_idx = ((batch * num_kv_heads + head) * seq_len_q + seq_idx) * head_dim + d;

    // Destination: cache at position cache_len + seq_idx
    let cache_pos = cache_len + seq_idx;
    if (cache_pos >= max_cache_len) {
        return;  // Cache overflow
    }

    let dst_idx = ((batch * num_kv_heads + head) * max_cache_len + cache_pos) * head_dim + d;

    k_cache[dst_idx] = key[src_idx];
    v_cache[dst_idx] = value[src_idx];
}

// ============================================================================
// Multi-Query Attention (MQA) - Single KV head shared across all query heads
// ============================================================================

@compute @workgroup_size(256)
fn sdpa_mqa(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_size = params.batch_size;
    let num_heads = params.num_heads;
    let seq_len_q = params.seq_len_q;
    let seq_len_kv = params.seq_len_kv;
    let head_dim = params.head_dim;
    let scale = params.scale;

    // MQA: single KV head (num_kv_heads = 1)
    let batch_head = wid.x;
    let batch = batch_head / num_heads;
    let head = batch_head % num_heads;

    if (batch >= batch_size) {
        return;
    }

    // Query uses head index, KV uses batch only (single head)
    let q_base = (batch * num_heads + head) * seq_len_q * head_dim;
    let kv_base = batch * seq_len_kv * head_dim;  // No head dimension
    let o_base = q_base;

    // Similar to basic SDPA but with shared KV
    // ... (implementation similar to sdpa_basic with different indexing)

    // For each query position
    for (var q_pos = 0u; q_pos < seq_len_q; q_pos++) {
        var max_val: f32 = NEG_INF;

        // Find max
        var kv_pos = lid.x;
        while (kv_pos < seq_len_kv) {
            var dot: f32 = 0.0;
            for (var d = 0u; d < head_dim; d++) {
                dot += query[q_base + q_pos * head_dim + d] * key[kv_base + kv_pos * head_dim + d];
            }
            max_val = max(max_val, dot * scale);
            kv_pos += WORKGROUP_SIZE;
        }

        shared_max[lid.x] = max_val;
        workgroupBarrier();

        for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
            if (lid.x < s) {
                shared_max[lid.x] = max(shared_max[lid.x], shared_max[lid.x + s]);
            }
            workgroupBarrier();
        }

        max_val = shared_max[0];
        workgroupBarrier();

        // Compute sum
        var sum_exp: f32 = 0.0;
        kv_pos = lid.x;
        while (kv_pos < seq_len_kv) {
            var dot: f32 = 0.0;
            for (var d = 0u; d < head_dim; d++) {
                dot += query[q_base + q_pos * head_dim + d] * key[kv_base + kv_pos * head_dim + d];
            }
            sum_exp += safe_exp(dot * scale - max_val);
            kv_pos += WORKGROUP_SIZE;
        }

        shared_sum[lid.x] = sum_exp;
        workgroupBarrier();

        for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
            if (lid.x < s) {
                shared_sum[lid.x] += shared_sum[lid.x + s];
            }
            workgroupBarrier();
        }

        let inv_sum = 1.0 / shared_sum[0];
        workgroupBarrier();

        // Compute output
        var d = lid.x;
        while (d < head_dim) {
            var acc: f32 = 0.0;
            for (var kv = 0u; kv < seq_len_kv; kv++) {
                var dot: f32 = 0.0;
                for (var dd = 0u; dd < head_dim; dd++) {
                    dot += query[q_base + q_pos * head_dim + dd] * key[kv_base + kv * head_dim + dd];
                }
                let attn = safe_exp(dot * scale - max_val) * inv_sum;
                acc += attn * value[kv_base + kv * head_dim + d];
            }
            output[o_base + q_pos * head_dim + d] = acc;
            d += WORKGROUP_SIZE;
        }
        workgroupBarrier();
    }
}
