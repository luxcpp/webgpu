// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Masked GEMV - General Matrix-Vector Multiplication with Masking
// Computes: y = alpha * mask(A) * x + beta * y
//
// Supports various masking patterns:
// - Causal (lower triangular) for autoregressive attention
// - Block sparse masks for efficient sparse computation
// - Custom binary masks
//
// Part of the Lux Network GPU acceleration library

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// Mask types
const MASK_NONE: u32 = 0u;
const MASK_CAUSAL: u32 = 1u;
const MASK_UPPER_TRIANGULAR: u32 = 2u;
const MASK_BLOCK_SPARSE: u32 = 3u;
const MASK_CUSTOM: u32 = 4u;

// ============================================================================
// Parameter Structures
// ============================================================================

struct MaskedGemvParams {
    M: u32,              // Number of rows in A
    N: u32,              // Number of columns in A
    alpha: f32,          // Scalar multiplier for A*x
    beta: f32,           // Scalar multiplier for y
    lda: u32,            // Leading dimension of A
    mask_type: u32,      // Type of mask to apply
    diagonal_offset: i32, // Offset for triangular masks
    block_size: u32,     // Block size for block-sparse masks
    mask_value: f32,     // Value to use for masked positions (typically -inf or 0)
    query_offset: u32,   // Query position offset for sliding window
    key_offset: u32,     // Key position offset
    _pad: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> vector_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> vector_y: array<f32>;
@group(0) @binding(3) var<uniform> params: MaskedGemvParams;
@group(0) @binding(4) var<storage, read> custom_mask: array<u32>;  // Packed bits for custom masks
@group(0) @binding(5) var<storage, read> block_indices: array<u32>; // For block-sparse

// Workgroup shared memory
var<workgroup> shared_sum: array<f32, WORKGROUP_SIZE>;

// ============================================================================
// Mask Functions
// ============================================================================

// Check if position (row, col) should be masked for causal attention
fn is_causal_masked(row: u32, col: u32, offset: i32) -> bool {
    // Causal mask: col > row + offset
    return i32(col) > i32(row) + offset;
}

// Check if position should be masked for upper triangular
fn is_upper_triangular_masked(row: u32, col: u32, offset: i32) -> bool {
    // Upper triangular mask: col < row + offset
    return i32(col) < i32(row) + offset;
}

// Check custom mask (bit-packed)
fn is_custom_masked(row: u32, col: u32, N: u32) -> bool {
    let bit_idx = row * N + col;
    let word_idx = bit_idx / 32u;
    let bit_offset = bit_idx % 32u;
    return (custom_mask[word_idx] & (1u << bit_offset)) == 0u;
}

// Check if block is active in block-sparse mask
fn is_block_active(block_row: u32, block_col: u32, num_blocks_col: u32) -> bool {
    let block_idx = block_row * num_blocks_col + block_col;
    let word_idx = block_idx / 32u;
    let bit_offset = block_idx % 32u;
    return (block_indices[word_idx] & (1u << bit_offset)) != 0u;
}

// ============================================================================
// Main Masked GEMV Kernel - Causal (Lower Triangular)
// ============================================================================

@compute @workgroup_size(256)
fn gemv_causal(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let M = params.M;
    let N = params.N;
    let alpha = params.alpha;
    let beta = params.beta;
    let lda = params.lda;
    let diagonal_offset = params.diagonal_offset;
    let row = wid.x;

    if (row >= M) {
        return;
    }

    // For causal mask, only process columns up to row + offset
    let max_col = min(u32(max(0, i32(row) + diagonal_offset + 1)), N);

    var partial_sum: f32 = 0.0;

    var col = lid.x;
    while (col < max_col) {
        let a_idx = row * lda + col;
        partial_sum += matrix_a[a_idx] * vector_x[col];
        col += WORKGROUP_SIZE;
    }

    shared_sum[lid.x] = partial_sum;
    workgroupBarrier();

    // Parallel reduction
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_sum[lid.x] += shared_sum[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        vector_y[row] = alpha * shared_sum[0] + beta * vector_y[row];
    }
}

// ============================================================================
// Masked GEMV with Custom Binary Mask
// ============================================================================

@compute @workgroup_size(256)
fn gemv_custom_mask(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let M = params.M;
    let N = params.N;
    let alpha = params.alpha;
    let beta = params.beta;
    let lda = params.lda;
    let row = wid.x;

    if (row >= M) {
        return;
    }

    var partial_sum: f32 = 0.0;

    var col = lid.x;
    while (col < N) {
        // Check mask for this position
        if (!is_custom_masked(row, col, N)) {
            let a_idx = row * lda + col;
            partial_sum += matrix_a[a_idx] * vector_x[col];
        }
        col += WORKGROUP_SIZE;
    }

    shared_sum[lid.x] = partial_sum;
    workgroupBarrier();

    // Parallel reduction
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_sum[lid.x] += shared_sum[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        vector_y[row] = alpha * shared_sum[0] + beta * vector_y[row];
    }
}

// ============================================================================
// Block-Sparse Masked GEMV
// ============================================================================

@compute @workgroup_size(256)
fn gemv_block_sparse(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let M = params.M;
    let N = params.N;
    let alpha = params.alpha;
    let beta = params.beta;
    let lda = params.lda;
    let block_size = params.block_size;
    let row = wid.x;

    if (row >= M) {
        return;
    }

    let block_row = row / block_size;
    let num_blocks_col = (N + block_size - 1u) / block_size;

    var partial_sum: f32 = 0.0;

    // Iterate through column blocks
    for (var block_col = 0u; block_col < num_blocks_col; block_col++) {
        // Check if this block is active
        if (is_block_active(block_row, block_col, num_blocks_col)) {
            let col_start = block_col * block_size;
            let col_end = min(col_start + block_size, N);

            // Each thread processes columns within active blocks
            var col = col_start + lid.x;
            while (col < col_end) {
                let a_idx = row * lda + col;
                partial_sum += matrix_a[a_idx] * vector_x[col];
                col += WORKGROUP_SIZE;
            }
        }
    }

    shared_sum[lid.x] = partial_sum;
    workgroupBarrier();

    // Parallel reduction
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_sum[lid.x] += shared_sum[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        vector_y[row] = alpha * shared_sum[0] + beta * vector_y[row];
    }
}

// ============================================================================
// Sliding Window Causal Mask (for long sequence attention)
// ============================================================================

struct SlidingWindowParams {
    M: u32,
    N: u32,
    alpha: f32,
    beta: f32,
    lda: u32,
    window_size: u32,    // Size of attention window
    query_start: u32,    // Starting position of query
    key_start: u32,      // Starting position of keys
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(1) @binding(0) var<uniform> sw_params: SlidingWindowParams;

@compute @workgroup_size(256)
fn gemv_sliding_window(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let M = sw_params.M;
    let N = sw_params.N;
    let alpha = sw_params.alpha;
    let beta = sw_params.beta;
    let lda = sw_params.lda;
    let window_size = sw_params.window_size;
    let query_start = sw_params.query_start;
    let key_start = sw_params.key_start;
    let row = wid.x;

    if (row >= M) {
        return;
    }

    // Calculate query position
    let query_pos = query_start + row;

    // Sliding window: attend to keys in range [query_pos - window_size + 1, query_pos]
    let window_start = select(0u, query_pos - window_size + 1u, query_pos >= window_size - 1u);
    let window_end = query_pos + 1u;

    // Map to actual column indices
    let col_start = select(0u, window_start - key_start, window_start >= key_start);
    let col_end = min(window_end - key_start, N);

    var partial_sum: f32 = 0.0;

    if (col_start < col_end) {
        var col = col_start + lid.x;
        while (col < col_end) {
            let a_idx = row * lda + col;
            partial_sum += matrix_a[a_idx] * vector_x[col];
            col += WORKGROUP_SIZE;
        }
    }

    shared_sum[lid.x] = partial_sum;
    workgroupBarrier();

    // Parallel reduction
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_sum[lid.x] += shared_sum[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        vector_y[row] = alpha * shared_sum[0] + beta * vector_y[row];
    }
}

// ============================================================================
// Generic Masked GEMV with Mask Value Replacement
// For softmax-based attention where masked positions get -inf
// ============================================================================

@compute @workgroup_size(256)
fn gemv_masked_with_value(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let M = params.M;
    let N = params.N;
    let alpha = params.alpha;
    let beta = params.beta;
    let lda = params.lda;
    let mask_type = params.mask_type;
    let diagonal_offset = params.diagonal_offset;
    let mask_value = params.mask_value;
    let row = wid.x;

    if (row >= M) {
        return;
    }

    var partial_sum: f32 = 0.0;

    var col = lid.x;
    while (col < N) {
        var is_masked = false;

        switch (mask_type) {
            case MASK_CAUSAL: {
                is_masked = is_causal_masked(row, col, diagonal_offset);
            }
            case MASK_UPPER_TRIANGULAR: {
                is_masked = is_upper_triangular_masked(row, col, diagonal_offset);
            }
            case MASK_CUSTOM: {
                is_masked = is_custom_masked(row, col, N);
            }
            default: {
                is_masked = false;
            }
        }

        let a_idx = row * lda + col;
        // Use mask_value for masked positions, actual value otherwise
        let effective_value = select(matrix_a[a_idx], mask_value, is_masked);
        partial_sum += effective_value * vector_x[col];

        col += WORKGROUP_SIZE;
    }

    shared_sum[lid.x] = partial_sum;
    workgroupBarrier();

    // Parallel reduction
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_sum[lid.x] += shared_sum[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        vector_y[row] = alpha * shared_sum[0] + beta * vector_y[row];
    }
}

// ============================================================================
// Prefix Sum Causal GEMV (for parallel scan attention)
// ============================================================================

@compute @workgroup_size(256)
fn gemv_causal_prefix(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let M = params.M;
    let N = params.N;
    let alpha = params.alpha;
    let lda = params.lda;
    let row = wid.x;

    if (row >= M) {
        return;
    }

    // For causal attention, compute cumulative sum
    // y[i] = sum_{j=0}^{i} A[i,j] * x[j]
    let max_col = min(row + 1u, N);

    var partial_sum: f32 = 0.0;

    var col = lid.x;
    while (col < max_col) {
        let a_idx = row * lda + col;
        partial_sum += matrix_a[a_idx] * vector_x[col];
        col += WORKGROUP_SIZE;
    }

    shared_sum[lid.x] = partial_sum;
    workgroupBarrier();

    // Parallel reduction
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_sum[lid.x] += shared_sum[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        vector_y[row] = alpha * shared_sum[0];
    }
}
