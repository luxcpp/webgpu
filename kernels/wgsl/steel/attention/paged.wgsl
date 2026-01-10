// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Paged Attention in WGSL
// Efficient KV cache management for LLM inference (vLLM style)
// Works on WebGPU (Metal/Vulkan/D3D12 via Dawn/wgpu)

// ============================================================================
// Configuration Constants
// ============================================================================

const BLOCK_SIZE: u32 = 16u;   // Tokens per block in KV cache
const WARP_SIZE: u32 = 32u;
const MAX_CONTEXT_LEN: u32 = 32768u;

// ============================================================================
// Bindings
// ============================================================================

struct PagedAttentionParams {
    num_seqs: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    max_blocks_per_seq: u32,
    scale: f32,
    _pad: u32,
}

// Query: [num_seqs, num_heads, head_dim]
// Key cache: [num_blocks, num_kv_heads, block_size, head_dim]
// Value cache: [num_blocks, num_kv_heads, block_size, head_dim]
// Block tables: [num_seqs, max_blocks_per_seq]
// Context lens: [num_seqs]
// Output: [num_seqs, num_heads, head_dim]

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key_cache: array<f32>;
@group(0) @binding(2) var<storage, read> value_cache: array<f32>;
@group(0) @binding(3) var<storage, read> block_tables: array<i32>;
@group(0) @binding(4) var<storage, read> context_lens: array<i32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> params: PagedAttentionParams;

// Shared memory
var<workgroup> scores: array<f32, 2048>;  // Max context per workgroup
var<workgroup> output_smem: array<f32, 128>;

// ============================================================================
// Paged Attention V1
// ============================================================================

@compute @workgroup_size(256)
fn paged_attention_v1(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let seq_idx = wid.x;
    let head_idx = wid.y;
    let tid = lid.x;

    if (seq_idx >= params.num_seqs) { return; }

    let context_len = context_lens[seq_idx];
    if (context_len <= 0) { return; }

    let num_heads = params.num_heads;
    let num_kv_heads = params.num_kv_heads;
    let head_dim = params.head_dim;
    let block_size = params.block_size;
    let scale = params.scale;

    // Map query head to KV head (for GQA/MQA)
    let heads_per_kv = num_heads / num_kv_heads;
    let kv_head_idx = head_idx / heads_per_kv;

    // Query for this sequence/head
    let q_offset = (seq_idx * num_heads + head_idx) * head_dim;

    // Block table for this sequence
    let block_table_offset = seq_idx * params.max_blocks_per_seq;

    // Step 1: Compute attention scores
    for (var token_idx = i32(tid); token_idx < context_len; token_idx += i32(256u)) {
        let logical_block = u32(token_idx) / block_size;
        let block_offset = u32(token_idx) % block_size;
        let physical_block = u32(block_tables[block_table_offset + logical_block]);

        // Key location in cache
        let k_offset = (physical_block * num_kv_heads + kv_head_idx) * block_size * head_dim +
                       block_offset * head_dim;

        var dot: f32 = 0.0;
        for (var d = 0u; d < head_dim; d++) {
            dot += query[q_offset + d] * key_cache[k_offset + d];
        }

        scores[token_idx] = dot * scale;
    }
    workgroupBarrier();

    // Step 2: Softmax - find max
    var max_val: f32 = -3.402823e+38;
    for (var i = i32(tid); i < context_len; i += i32(256u)) {
        max_val = max(max_val, scores[i]);
    }

    // Warp reduction for max
    for (var offset = 16u; offset > 0u; offset /= 2u) {
        // Subgroup operations not available in base WGSL, use workgroup shared memory
    }

    // Store partial max to shared memory
    var<workgroup> partial_max: array<f32, 8>;
    if (tid % 32u == 0u) {
        partial_max[tid / 32u] = max_val;
    }
    workgroupBarrier();

    // Reduce partial maxes
    if (tid == 0u) {
        var global_max: f32 = -3.402823e+38;
        for (var i = 0u; i < 8u; i++) {
            global_max = max(global_max, partial_max[i]);
        }
        partial_max[0] = global_max;
    }
    workgroupBarrier();
    max_val = partial_max[0];

    // Compute exp and sum
    var local_sum: f32 = 0.0;
    for (var i = i32(tid); i < context_len; i += i32(256u)) {
        let exp_val = exp(scores[i] - max_val);
        scores[i] = exp_val;
        local_sum += exp_val;
    }

    // Reduce sum
    var<workgroup> partial_sum: array<f32, 8>;
    if (tid % 32u == 0u) {
        partial_sum[tid / 32u] = local_sum;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var global_sum: f32 = 0.0;
        for (var i = 0u; i < 8u; i++) {
            global_sum += partial_sum[i];
        }
        partial_sum[0] = global_sum;
    }
    workgroupBarrier();

    let inv_sum = 1.0 / partial_sum[0];

    // Normalize scores
    for (var i = i32(tid); i < context_len; i += i32(256u)) {
        scores[i] *= inv_sum;
    }
    workgroupBarrier();

    // Step 3: Compute output = sum(scores * V)
    for (var d = tid; d < head_dim; d += 256u) {
        var acc: f32 = 0.0;

        for (var token_idx = 0; token_idx < context_len; token_idx++) {
            let logical_block = u32(token_idx) / block_size;
            let block_offset = u32(token_idx) % block_size;
            let physical_block = u32(block_tables[block_table_offset + logical_block]);

            let v_offset = (physical_block * num_kv_heads + kv_head_idx) * block_size * head_dim +
                           block_offset * head_dim;

            acc += scores[token_idx] * value_cache[v_offset + d];
        }

        output[(seq_idx * num_heads + head_idx) * head_dim + d] = acc;
    }
}

// ============================================================================
// Paged Attention V2 (Partitioned for long contexts)
// ============================================================================

struct PagedAttentionV2Params {
    num_seqs: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    max_blocks_per_seq: u32,
    partition_size: u32,
    max_partitions: u32,
    scale: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(7) var<storage, read_write> exp_sums: array<f32>;      // [num_seqs, num_heads, max_partitions]
@group(0) @binding(8) var<storage, read_write> max_logits: array<f32>;   // [num_seqs, num_heads, max_partitions]
@group(0) @binding(9) var<storage, read_write> partition_output: array<f32>;  // [num_seqs, num_heads, max_partitions, head_dim]
@group(0) @binding(10) var<uniform> params_v2: PagedAttentionV2Params;

@compute @workgroup_size(256)
fn paged_attention_v2_partial(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let seq_idx = wid.x;
    let head_idx = wid.y;
    let partition_idx = wid.z;
    let tid = lid.x;

    if (seq_idx >= params_v2.num_seqs) { return; }

    let context_len = context_lens[seq_idx];
    if (context_len <= 0) { return; }

    let partition_size = params_v2.partition_size;
    let partition_start = i32(partition_idx * partition_size);
    let partition_end = min(partition_start + i32(partition_size), context_len);

    if (partition_start >= context_len) {
        // This partition has no work
        if (tid == 0u) {
            let stat_idx = (seq_idx * params_v2.num_heads + head_idx) * params_v2.max_partitions + partition_idx;
            max_logits[stat_idx] = -3.402823e+38;
            exp_sums[stat_idx] = 0.0;
        }
        return;
    }

    let partition_len = partition_end - partition_start;

    let num_heads = params_v2.num_heads;
    let num_kv_heads = params_v2.num_kv_heads;
    let head_dim = params_v2.head_dim;
    let block_size = params_v2.block_size;
    let scale = params_v2.scale;

    let heads_per_kv = num_heads / num_kv_heads;
    let kv_head_idx = head_idx / heads_per_kv;

    let q_offset = (seq_idx * num_heads + head_idx) * head_dim;
    let block_table_offset = seq_idx * params_v2.max_blocks_per_seq;

    // Compute scores for this partition
    for (var i = i32(tid); i < partition_len; i += 256) {
        let token_idx = partition_start + i;
        let logical_block = u32(token_idx) / block_size;
        let block_offset = u32(token_idx) % block_size;
        let physical_block = u32(block_tables[block_table_offset + logical_block]);

        let k_offset = (physical_block * num_kv_heads + kv_head_idx) * block_size * head_dim +
                       block_offset * head_dim;

        var dot: f32 = 0.0;
        for (var d = 0u; d < head_dim; d++) {
            dot += query[q_offset + d] * key_cache[k_offset + d];
        }

        scores[i] = dot * scale;
    }
    workgroupBarrier();

    // Local softmax
    var max_val: f32 = -3.402823e+38;
    for (var i = i32(tid); i < partition_len; i += 256) {
        max_val = max(max_val, scores[i]);
    }

    var<workgroup> partial_max: array<f32, 8>;
    if (tid % 32u == 0u) {
        partial_max[tid / 32u] = max_val;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var global_max: f32 = -3.402823e+38;
        for (var i = 0u; i < 8u; i++) {
            global_max = max(global_max, partial_max[i]);
        }
        partial_max[0] = global_max;
    }
    workgroupBarrier();
    max_val = partial_max[0];

    var local_sum: f32 = 0.0;
    for (var i = i32(tid); i < partition_len; i += 256) {
        let exp_val = exp(scores[i] - max_val);
        scores[i] = exp_val;
        local_sum += exp_val;
    }

    var<workgroup> partial_sum: array<f32, 8>;
    if (tid % 32u == 0u) {
        partial_sum[tid / 32u] = local_sum;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var global_sum: f32 = 0.0;
        for (var i = 0u; i < 8u; i++) {
            global_sum += partial_sum[i];
        }
        partial_sum[0] = global_sum;
    }
    workgroupBarrier();

    // Store partition statistics
    if (tid == 0u) {
        let stat_idx = (seq_idx * num_heads + head_idx) * params_v2.max_partitions + partition_idx;
        max_logits[stat_idx] = max_val;
        exp_sums[stat_idx] = partial_sum[0];
    }

    // Normalize and compute weighted output
    let inv_sum = 1.0 / partial_sum[0];
    for (var i = i32(tid); i < partition_len; i += 256) {
        scores[i] *= inv_sum;
    }
    workgroupBarrier();

    // Compute partition output
    for (var d = tid; d < head_dim; d += 256u) {
        var acc: f32 = 0.0;

        for (var i = 0; i < partition_len; i++) {
            let token_idx = partition_start + i;
            let logical_block = u32(token_idx) / block_size;
            let block_offset = u32(token_idx) % block_size;
            let physical_block = u32(block_tables[block_table_offset + logical_block]);

            let v_offset = (physical_block * num_kv_heads + kv_head_idx) * block_size * head_dim +
                           block_offset * head_dim;

            acc += scores[i] * value_cache[v_offset + d];
        }

        let out_idx = ((seq_idx * num_heads + head_idx) * params_v2.max_partitions + partition_idx) * head_dim + d;
        partition_output[out_idx] = acc;
    }
}

// ============================================================================
// Paged Attention V2 Reduction
// ============================================================================

@compute @workgroup_size(256)
fn paged_attention_v2_reduce(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let seq_idx = wid.x;
    let head_idx = wid.y;
    let tid = lid.x;

    if (seq_idx >= params_v2.num_seqs) { return; }

    let num_heads = params_v2.num_heads;
    let head_dim = params_v2.head_dim;
    let max_partitions = params_v2.max_partitions;

    // Find global max
    var global_max: f32 = -3.402823e+38;
    for (var p = 0u; p < max_partitions; p++) {
        let idx = (seq_idx * num_heads + head_idx) * max_partitions + p;
        global_max = max(global_max, max_logits[idx]);
    }

    // Compute rescaled global sum
    var global_sum: f32 = 0.0;
    for (var p = 0u; p < max_partitions; p++) {
        let idx = (seq_idx * num_heads + head_idx) * max_partitions + p;
        let m = max_logits[idx];
        let e = exp_sums[idx];
        if (m > -3.402823e+38) {
            global_sum += e * exp(m - global_max);
        }
    }

    // Combine partition outputs
    for (var d = tid; d < head_dim; d += 256u) {
        var acc: f32 = 0.0;

        for (var p = 0u; p < max_partitions; p++) {
            let stat_idx = (seq_idx * num_heads + head_idx) * max_partitions + p;
            let m = max_logits[stat_idx];
            let e = exp_sums[stat_idx];

            if (m > -3.402823e+38) {
                let weight = e * exp(m - global_max) / global_sum;
                let part_idx = ((seq_idx * num_heads + head_idx) * max_partitions + p) * head_dim + d;
                acc += weight * partition_output[part_idx];
            }
        }

        output[(seq_idx * num_heads + head_idx) * head_dim + d] = acc;
    }
}
