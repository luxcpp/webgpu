// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Masked GEMM for Attention in WGSL
// Works on WebGPU (Metal/Vulkan/D3D12 via Dawn/wgpu)
//
// Part of the Lux Network GPU acceleration library
// Implements GEMM with various masking patterns for attention mechanisms

// ============================================================================
// Configuration Constants
// ============================================================================

const TILE_M: u32 = 64u;
const TILE_N: u32 = 64u;
const TILE_K: u32 = 16u;
const BLOCK_SIZE: u32 = 256u;
const NEG_INF: f32 = -3.402823e+38;

// ============================================================================
// Mask Types
// ============================================================================

const MASK_NONE: u32 = 0u;
const MASK_CAUSAL: u32 = 1u;         // Lower triangular
const MASK_CAUSAL_WINDOW: u32 = 2u;  // Sliding window causal
const MASK_BIDIRECTIONAL: u32 = 3u;  // Full attention
const MASK_PREFIX_LM: u32 = 4u;      // Prefix can attend to all, rest is causal
const MASK_ALIBI: u32 = 5u;          // Causal + ALiBi bias
const MASK_CUSTOM: u32 = 6u;         // User-provided mask

// ============================================================================
// Kernel Bindings
// ============================================================================

struct GemmMaskedParams {
    M: u32,                    // Rows of A (query sequence length)
    N: u32,                    // Columns of B (key sequence length)
    K: u32,                    // Inner dimension (head dimension)
    batch_size: u32,
    num_heads: u32,
    mask_type: u32,            // Mask type
    window_size: u32,          // For sliding window attention
    prefix_length: u32,        // For prefix LM
    scale: f32,                // Attention scale (1/sqrt(d))
    alibi_slope: f32,          // ALiBi slope per head
    _pad: vec2<u32>,
}

// Q: [batch, heads, M, K]
// K: [batch, heads, N, K]  (or [batch, kv_heads, N, K] for GQA)
// mask: [batch, 1, M, N] or [1, 1, M, N] (broadcastable)
// output: [batch, heads, M, N] (attention scores before softmax)
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: GemmMaskedParams;

// Shared memory
var<workgroup> tile_Q: array<f32, 1024>;
var<workgroup> tile_K: array<f32, 1024>;
var<workgroup> tile_S: array<f32, 4096>;

// ============================================================================
// Mask Generation Functions
// ============================================================================

fn causal_mask(q_pos: u32, k_pos: u32) -> f32 {
    return select(NEG_INF, 0.0, k_pos <= q_pos);
}

fn causal_window_mask(q_pos: u32, k_pos: u32, window: u32) -> f32 {
    let in_window = k_pos <= q_pos && q_pos - k_pos < window;
    return select(NEG_INF, 0.0, in_window);
}

fn prefix_lm_mask(q_pos: u32, k_pos: u32, prefix_len: u32) -> f32 {
    // Prefix tokens can attend to all prefix tokens
    // Non-prefix tokens use causal mask
    if (q_pos < prefix_len) {
        return select(NEG_INF, 0.0, k_pos < prefix_len);
    } else {
        return causal_mask(q_pos, k_pos);
    }
}

fn alibi_bias(q_pos: u32, k_pos: u32, slope: f32) -> f32 {
    let dist = i32(k_pos) - i32(q_pos);
    if (dist > 0) {
        return NEG_INF;  // Causal part
    }
    return slope * f32(dist);  // Negative distance penalty
}

// ============================================================================
// Causal Masked GEMM
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_masked_causal(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_head = wid.z;
    let batch_idx = batch_head / params.num_heads;
    let head_idx = batch_head % params.num_heads;

    if (batch_idx >= params.batch_size) { return; }

    let M = params.M;
    let N = params.N;
    let K = params.K;
    let scale = params.scale;

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;
    let num_rows = min(TILE_M, M - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    // Skip if entire tile is masked (causal optimization)
    if (tile_col > tile_row + TILE_M) {
        // This tile is fully above the diagonal, all masked
        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;
            let out_idx = ((batch_head * M + tile_row + c_row) * N) + tile_col + c_col;
            output[out_idx] = NEG_INF;
        }
        return;
    }

    // Initialize
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        tile_S[i] = 0.0;
    }
    workgroupBarrier();

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, K - k_start);

        // Load Q tile
        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let q_idx = ((batch_head * M + tile_row + row) * K) + k_start + col;
            tile_Q[row * TILE_K + col] = Q[q_idx];
        }

        // Load K tile
        for (var i = tid; i < num_cols * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let k_idx = ((batch_head * N + tile_col + row) * K) + k_start + col;
            tile_K[row * TILE_K + col] = K[k_idx];
        }
        workgroupBarrier();

        // Compute Q @ K^T
        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;

            var sum: f32 = 0.0;
            for (var k = 0u; k < num_k; k++) {
                sum += tile_Q[c_row * TILE_K + k] * tile_K[c_col * TILE_K + k];
            }
            tile_S[c_row * TILE_N + c_col] += sum;
        }
        workgroupBarrier();
    }

    // Apply scale and causal mask, then store
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let q_pos = tile_row + c_row;
        let k_pos = tile_col + c_col;

        var score = tile_S[c_row * TILE_N + c_col] * scale;

        // Apply causal mask
        if (k_pos > q_pos) {
            score = NEG_INF;
        }

        let out_idx = ((batch_head * M + q_pos) * N) + k_pos;
        output[out_idx] = score;
    }
}

// ============================================================================
// Sliding Window Masked GEMM
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_masked_window(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_head = wid.z;
    let batch_idx = batch_head / params.num_heads;
    let head_idx = batch_head % params.num_heads;

    if (batch_idx >= params.batch_size) { return; }

    let M = params.M;
    let N = params.N;
    let K = params.K;
    let scale = params.scale;
    let window = params.window_size;

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;
    let num_rows = min(TILE_M, M - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    // Check if tile is completely outside window
    let min_q_pos = tile_row;
    let max_q_pos = tile_row + num_rows - 1u;
    let min_k_pos = tile_col;
    let max_k_pos = tile_col + num_cols - 1u;

    // Skip if outside window (above diagonal or beyond window)
    if (min_k_pos > max_q_pos || (max_q_pos >= window && max_k_pos < min_q_pos - window + 1u)) {
        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;
            let out_idx = ((batch_head * M + tile_row + c_row) * N) + tile_col + c_col;
            output[out_idx] = NEG_INF;
        }
        return;
    }

    // Standard GEMM
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        tile_S[i] = 0.0;
    }
    workgroupBarrier();

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, K - k_start);

        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let q_idx = ((batch_head * M + tile_row + row) * K) + k_start + col;
            tile_Q[row * TILE_K + col] = Q[q_idx];
        }

        for (var i = tid; i < num_cols * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let k_idx = ((batch_head * N + tile_col + row) * K) + k_start + col;
            tile_K[row * TILE_K + col] = K[k_idx];
        }
        workgroupBarrier();

        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;

            var sum: f32 = 0.0;
            for (var k = 0u; k < num_k; k++) {
                sum += tile_Q[c_row * TILE_K + k] * tile_K[c_col * TILE_K + k];
            }
            tile_S[c_row * TILE_N + c_col] += sum;
        }
        workgroupBarrier();
    }

    // Apply sliding window mask
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let q_pos = tile_row + c_row;
        let k_pos = tile_col + c_col;

        var score = tile_S[c_row * TILE_N + c_col] * scale;

        // Apply sliding window causal mask
        score += causal_window_mask(q_pos, k_pos, window);

        let out_idx = ((batch_head * M + q_pos) * N) + k_pos;
        output[out_idx] = score;
    }
}

// ============================================================================
// ALiBi Masked GEMM
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_masked_alibi(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_head = wid.z;
    let batch_idx = batch_head / params.num_heads;
    let head_idx = batch_head % params.num_heads;

    if (batch_idx >= params.batch_size) { return; }

    let M = params.M;
    let N = params.N;
    let K = params.K;
    let scale = params.scale;

    // ALiBi slope is different for each head
    // Common pattern: slopes = 2^(-8/num_heads * (head_idx + 1))
    let base_slope = params.alibi_slope;
    let slope = base_slope * pow(2.0, -8.0 * f32(head_idx + 1u) / f32(params.num_heads));

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;
    let num_rows = min(TILE_M, M - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    // Skip fully masked tiles
    if (tile_col > tile_row + TILE_M) {
        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;
            let out_idx = ((batch_head * M + tile_row + c_row) * N) + tile_col + c_col;
            output[out_idx] = NEG_INF;
        }
        return;
    }

    // Standard GEMM
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        tile_S[i] = 0.0;
    }
    workgroupBarrier();

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, K - k_start);

        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let q_idx = ((batch_head * M + tile_row + row) * K) + k_start + col;
            tile_Q[row * TILE_K + col] = Q[q_idx];
        }

        for (var i = tid; i < num_cols * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let k_idx = ((batch_head * N + tile_col + row) * K) + k_start + col;
            tile_K[row * TILE_K + col] = K[k_idx];
        }
        workgroupBarrier();

        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;

            var sum: f32 = 0.0;
            for (var k = 0u; k < num_k; k++) {
                sum += tile_Q[c_row * TILE_K + k] * tile_K[c_col * TILE_K + k];
            }
            tile_S[c_row * TILE_N + c_col] += sum;
        }
        workgroupBarrier();
    }

    // Apply ALiBi bias (causal + distance penalty)
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let q_pos = tile_row + c_row;
        let k_pos = tile_col + c_col;

        var score = tile_S[c_row * TILE_N + c_col] * scale;

        // Apply ALiBi: causal mask + linear distance penalty
        score += alibi_bias(q_pos, k_pos, slope);

        let out_idx = ((batch_head * M + q_pos) * N) + k_pos;
        output[out_idx] = score;
    }
}

// ============================================================================
// Custom Mask GEMM
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_masked_custom(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Uses user-provided mask buffer
    // mask layout: [batch, 1, M, N] with broadcasting

    let tid = lid.x;
    let batch_head = wid.z;
    let batch_idx = batch_head / params.num_heads;
    let head_idx = batch_head % params.num_heads;

    if (batch_idx >= params.batch_size) { return; }

    let M = params.M;
    let N = params.N;
    let K = params.K;
    let scale = params.scale;

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;
    let num_rows = min(TILE_M, M - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    // Standard GEMM
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        tile_S[i] = 0.0;
    }
    workgroupBarrier();

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, K - k_start);

        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let q_idx = ((batch_head * M + tile_row + row) * K) + k_start + col;
            tile_Q[row * TILE_K + col] = Q[q_idx];
        }

        for (var i = tid; i < num_cols * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let k_idx = ((batch_head * N + tile_col + row) * K) + k_start + col;
            tile_K[row * TILE_K + col] = K[k_idx];
        }
        workgroupBarrier();

        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;

            var sum: f32 = 0.0;
            for (var k = 0u; k < num_k; k++) {
                sum += tile_Q[c_row * TILE_K + k] * tile_K[c_col * TILE_K + k];
            }
            tile_S[c_row * TILE_N + c_col] += sum;
        }
        workgroupBarrier();
    }

    // Apply custom mask and store
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let q_pos = tile_row + c_row;
        let k_pos = tile_col + c_col;

        var score = tile_S[c_row * TILE_N + c_col] * scale;

        // Load mask value (with broadcast: use batch_idx for mask batch dim)
        let mask_idx = (batch_idx * M + q_pos) * N + k_pos;
        let mask_val = mask[mask_idx];

        // Add mask (expected to be 0 for attend, NEG_INF for mask)
        score += mask_val;

        let out_idx = ((batch_head * M + q_pos) * N) + k_pos;
        output[out_idx] = score;
    }
}
