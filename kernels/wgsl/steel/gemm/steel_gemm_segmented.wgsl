// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Segmented GEMM for Batched Operations in WGSL
// Works on WebGPU (Metal/Vulkan/D3D12 via Dawn/wgpu)
//
// Part of the Lux Network GPU acceleration library
// Implements GEMM with variable-sized segments for batched processing

// ============================================================================
// Configuration Constants
// ============================================================================

const TILE_M: u32 = 32u;
const TILE_N: u32 = 64u;
const TILE_K: u32 = 16u;
const BLOCK_SIZE: u32 = 256u;
const MAX_SEGMENTS: u32 = 256u;

// ============================================================================
// Kernel Bindings
// ============================================================================

struct GemmSegmentedParams {
    total_m: u32,              // Total rows across all segments
    N: u32,                    // Columns of B (shared)
    K: u32,                    // Inner dimension (shared)
    num_segments: u32,         // Number of segments
    max_segment_size: u32,     // Maximum segment size
    lda: u32,                  // Leading dimension of A
    ldb: u32,                  // Leading dimension of B
    ldc: u32,                  // Leading dimension of C
    alpha: f32,
    beta: f32,
    shared_b: u32,             // 1 if B is shared across segments
    _pad: u32,
}

// A: [total_m, K] - concatenated segments
// B: [K, N] or [num_segments, K, N] if not shared
// C: [total_m, N] - concatenated outputs
// segment_offsets: [num_segments + 1] - cumulative row offsets
// segment_sizes: [num_segments] - size of each segment
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> segment_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> segment_sizes: array<u32>;
@group(0) @binding(5) var<uniform> params: GemmSegmentedParams;

// Shared memory
var<workgroup> tile_A: array<f32, 512>;
var<workgroup> tile_B: array<f32, 1024>;
var<workgroup> tile_C: array<f32, 2048>;
var<workgroup> segment_info: vec4<u32>;  // [offset, size, segment_idx, _]

// ============================================================================
// Segmented GEMM (Variable Segment Sizes)
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_segmented(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let segment_idx = wid.z;

    if (segment_idx >= params.num_segments) { return; }

    // Load segment info
    if (tid == 0u) {
        segment_info.x = segment_offsets[segment_idx];      // Row offset
        segment_info.y = segment_sizes[segment_idx];        // Segment size (M)
        segment_info.z = segment_idx;
    }
    workgroupBarrier();

    let seg_offset = segment_info.x;
    let seg_size = segment_info.y;
    let N = params.N;
    let K = params.K;
    let shared_b = params.shared_b == 1u;

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;

    // Skip if tile is outside this segment
    if (tile_row >= seg_size) { return; }

    let num_rows = min(TILE_M, seg_size - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    // Initialize accumulator
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        tile_C[i] = 0.0;
    }
    workgroupBarrier();

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, K - k_start);

        // Load A tile from segment
        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let global_row = seg_offset + tile_row + row;
            let a_idx = global_row * params.lda + k_start + col;
            tile_A[row * TILE_K + col] = A[a_idx];
        }

        // Load B tile (shared or per-segment)
        for (var i = tid; i < num_k * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;

            var b_idx: u32;
            if (shared_b) {
                b_idx = (k_start + row) * params.ldb + tile_col + col;
            } else {
                // Per-segment B: [segment, K, N]
                b_idx = (segment_idx * K + k_start + row) * params.ldb + tile_col + col;
            }
            tile_B[row * TILE_N + col] = B[b_idx];
        }
        workgroupBarrier();

        // Compute
        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;

            var sum: f32 = 0.0;
            for (var k = 0u; k < num_k; k++) {
                sum += tile_A[c_row * TILE_K + k] * tile_B[k * TILE_N + c_col];
            }
            tile_C[c_row * TILE_N + c_col] += sum;
        }
        workgroupBarrier();
    }

    // Store result
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let global_row = seg_offset + tile_row + c_row;
        let global_col = tile_col + c_col;

        let c_idx = global_row * params.ldc + global_col;
        C[c_idx] = params.alpha * tile_C[c_row * TILE_N + c_col] + params.beta * C[c_idx];
    }
}

// ============================================================================
// Grouped GEMM (Fixed Segment Sizes)
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_grouped(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Optimized for when all segments have the same size
    // Uses fixed stride patterns for better memory access

    let tid = lid.x;
    let segment_idx = wid.z;

    if (segment_idx >= params.num_segments) { return; }

    let seg_size = params.max_segment_size;  // Fixed size per segment
    let seg_offset = segment_idx * seg_size;
    let N = params.N;
    let K = params.K;

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;

    if (tile_row >= seg_size) { return; }

    let num_rows = min(TILE_M, seg_size - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    // Initialize
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        tile_C[i] = 0.0;
    }
    workgroupBarrier();

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, K - k_start);

        // Load A with fixed stride
        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let a_idx = (seg_offset + tile_row + row) * K + k_start + col;
            tile_A[row * TILE_K + col] = A[a_idx];
        }

        // Load B (segment-specific)
        for (var i = tid; i < num_k * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;
            let b_idx = (segment_idx * K + k_start + row) * N + tile_col + col;
            tile_B[row * TILE_N + col] = B[b_idx];
        }
        workgroupBarrier();

        // Compute
        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;

            var sum: f32 = 0.0;
            for (var k = 0u; k < num_k; k++) {
                sum += tile_A[c_row * TILE_K + k] * tile_B[k * TILE_N + c_col];
            }
            tile_C[c_row * TILE_N + c_col] += sum;
        }
        workgroupBarrier();
    }

    // Store
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let c_idx = (seg_offset + tile_row + c_row) * N + tile_col + c_col;
        C[c_idx] = params.alpha * tile_C[c_row * TILE_N + c_col];
    }
}

// ============================================================================
// Strided Batched GEMM
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_strided_batched(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Standard strided batched GEMM
    // A: [batch, M, K] with stride = M * K
    // B: [batch, K, N] with stride = K * N
    // C: [batch, M, N] with stride = M * N

    let tid = lid.x;
    let batch_idx = wid.z;

    if (batch_idx >= params.num_segments) { return; }

    let M = params.max_segment_size;
    let N = params.N;
    let K = params.K;

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;

    if (tile_row >= M) { return; }

    let num_rows = min(TILE_M, M - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    let stride_a = M * K;
    let stride_b = K * N;
    let stride_c = M * N;

    // Initialize
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        tile_C[i] = 0.0;
    }
    workgroupBarrier();

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, K - k_start);

        // Load A
        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let a_idx = batch_idx * stride_a + (tile_row + row) * K + k_start + col;
            tile_A[row * TILE_K + col] = A[a_idx];
        }

        // Load B
        for (var i = tid; i < num_k * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;
            let b_idx = batch_idx * stride_b + (k_start + row) * N + tile_col + col;
            tile_B[row * TILE_N + col] = B[b_idx];
        }
        workgroupBarrier();

        // Compute
        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;

            var sum: f32 = 0.0;
            for (var k = 0u; k < num_k; k++) {
                sum += tile_A[c_row * TILE_K + k] * tile_B[k * TILE_N + c_col];
            }
            tile_C[c_row * TILE_N + c_col] += sum;
        }
        workgroupBarrier();
    }

    // Store
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let c_idx = batch_idx * stride_c + (tile_row + c_row) * N + tile_col + c_col;

        var result = params.alpha * tile_C[c_row * TILE_N + c_col];
        if (params.beta != 0.0) {
            result += params.beta * C[c_idx];
        }
        C[c_idx] = result;
    }
}

// ============================================================================
// Multi-Head Attention Batched GEMM (Q @ K^T per head)
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_mha_batched(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Optimized for attention: Q @ K^T where K is transposed
    // A = Q: [batch, heads, seq, head_dim]
    // B = K: [batch, heads, seq, head_dim] (will be transposed)
    // C = S: [batch, heads, seq, seq]

    let tid = lid.x;
    let batch_head = wid.z;
    let num_heads = params.num_segments;  // Repurpose
    let batch_idx = batch_head / num_heads;
    let head_idx = batch_head % num_heads;

    let seq_len = params.max_segment_size;
    let head_dim = params.K;

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;

    if (tile_row >= seq_len || tile_col >= seq_len) { return; }

    let num_rows = min(TILE_M, seq_len - tile_row);
    let num_cols = min(TILE_N, seq_len - tile_col);

    // Initialize
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        tile_C[i] = 0.0;
    }
    workgroupBarrier();

    let num_k_tiles = (head_dim + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, head_dim - k_start);

        // Load Q rows
        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let q_idx = ((batch_head * seq_len) + tile_row + row) * head_dim + k_start + col;
            tile_A[row * TILE_K + col] = A[q_idx];
        }

        // Load K rows (will use as K^T columns)
        for (var i = tid; i < num_cols * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let k_idx = ((batch_head * seq_len) + tile_col + row) * head_dim + k_start + col;
            tile_B[row * TILE_K + col] = B[k_idx];
        }
        workgroupBarrier();

        // Compute Q @ K^T (tile_B stores K rows, so inner product is correct)
        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;

            var sum: f32 = 0.0;
            for (var k = 0u; k < num_k; k++) {
                sum += tile_A[c_row * TILE_K + k] * tile_B[c_col * TILE_K + k];
            }
            tile_C[c_row * TILE_N + c_col] += sum;
        }
        workgroupBarrier();
    }

    // Store with scaling
    let scale = params.alpha;  // Usually 1/sqrt(head_dim)
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let c_idx = (batch_head * seq_len + tile_row + c_row) * seq_len + tile_col + c_col;

        C[c_idx] = tile_C[c_row * TILE_N + c_col] * scale;
    }
}

// ============================================================================
// Expert-Parallel MoE GEMM
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_moe(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Mixture of Experts: route tokens to experts and compute
    // segment_offsets: tokens per expert
    // B: [num_experts, K, N] - expert weights

    let tid = lid.x;
    let expert_idx = wid.z;

    if (expert_idx >= params.num_segments) { return; }

    // Get tokens routed to this expert
    let expert_start = segment_offsets[expert_idx];
    let expert_end = segment_offsets[expert_idx + 1u];
    let num_tokens = expert_end - expert_start;

    if (num_tokens == 0u) { return; }

    let N = params.N;
    let K = params.K;

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;

    if (tile_row >= num_tokens) { return; }

    let num_rows = min(TILE_M, num_tokens - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    // Initialize
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        tile_C[i] = 0.0;
    }
    workgroupBarrier();

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, K - k_start);

        // Load token activations (A is packed by expert assignment)
        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let token_idx = expert_start + tile_row + row;
            let a_idx = token_idx * K + k_start + col;
            tile_A[row * TILE_K + col] = A[a_idx];
        }

        // Load expert weights
        for (var i = tid; i < num_k * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;
            let b_idx = (expert_idx * K + k_start + row) * N + tile_col + col;
            tile_B[row * TILE_N + col] = B[b_idx];
        }
        workgroupBarrier();

        // Compute
        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;

            var sum: f32 = 0.0;
            for (var k = 0u; k < num_k; k++) {
                sum += tile_A[c_row * TILE_K + k] * tile_B[k * TILE_N + c_col];
            }
            tile_C[c_row * TILE_N + c_col] += sum;
        }
        workgroupBarrier();
    }

    // Store (back to packed format)
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let token_idx = expert_start + tile_row + c_row;
        let c_idx = token_idx * N + tile_col + c_col;

        C[c_idx] = tile_C[c_row * TILE_N + c_col];
    }
}
