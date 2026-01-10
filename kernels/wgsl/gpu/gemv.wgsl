// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// GEMV - General Matrix-Vector Multiplication
// Computes: y = alpha * A * x + beta * y
//
// Optimized for WebGPU with workgroup-level reductions
// Part of the Lux Network GPU acceleration library

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;
const TILE_SIZE: u32 = 4u;  // Elements per thread for coalesced access

// ============================================================================
// Parameter Structures
// ============================================================================

struct GemvParams {
    M: u32,           // Number of rows in A (output dimension)
    N: u32,           // Number of columns in A (input dimension)
    alpha: f32,       // Scalar multiplier for A*x
    beta: f32,        // Scalar multiplier for y
    lda: u32,         // Leading dimension of A
    trans: u32,       // 0 = A, 1 = A^T
    _pad0: u32,
    _pad1: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> vector_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> vector_y: array<f32>;
@group(0) @binding(3) var<uniform> params: GemvParams;

// Workgroup shared memory for parallel reduction
var<workgroup> shared_sum: array<f32, WORKGROUP_SIZE>;

// ============================================================================
// Main GEMV Kernel - Row-parallel version
// Each workgroup computes one output element
// ============================================================================

@compute @workgroup_size(256)
fn gemv_row_parallel(
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

    // Each thread accumulates partial dot product
    var partial_sum: f32 = 0.0;

    // Stride through the row, each thread handles different columns
    var col = lid.x;
    while (col < N) {
        let a_idx = row * lda + col;
        let a_val = matrix_a[a_idx];
        let x_val = vector_x[col];
        partial_sum += a_val * x_val;
        col += WORKGROUP_SIZE;
    }

    // Store partial sum to shared memory
    shared_sum[lid.x] = partial_sum;
    workgroupBarrier();

    // Parallel reduction in shared memory
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_sum[lid.x] += shared_sum[lid.x + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes the final result
    if (lid.x == 0u) {
        let old_y = vector_y[row];
        vector_y[row] = alpha * shared_sum[0] + beta * old_y;
    }
}

// ============================================================================
// Transposed GEMV Kernel - y = alpha * A^T * x + beta * y
// Column-parallel: each workgroup handles one output element
// ============================================================================

@compute @workgroup_size(256)
fn gemv_transpose(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let M = params.M;  // Original rows (now iteration dimension)
    let N = params.N;  // Original cols (now output dimension)
    let alpha = params.alpha;
    let beta = params.beta;
    let lda = params.lda;
    let col = wid.x;  // Output column

    if (col >= N) {
        return;
    }

    // Each thread accumulates partial dot product over rows
    var partial_sum: f32 = 0.0;

    var row = lid.x;
    while (row < M) {
        let a_idx = row * lda + col;
        let a_val = matrix_a[a_idx];
        let x_val = vector_x[row];
        partial_sum += a_val * x_val;
        row += WORKGROUP_SIZE;
    }

    // Store partial sum to shared memory
    shared_sum[lid.x] = partial_sum;
    workgroupBarrier();

    // Parallel reduction
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_sum[lid.x] += shared_sum[lid.x + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes the final result
    if (lid.x == 0u) {
        let old_y = vector_y[col];
        vector_y[col] = alpha * shared_sum[0] + beta * old_y;
    }
}

// ============================================================================
// Tiled GEMV - Multiple rows per workgroup for better occupancy
// ============================================================================

@compute @workgroup_size(256)
fn gemv_tiled(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let M = params.M;
    let N = params.N;
    let alpha = params.alpha;
    let beta = params.beta;
    let lda = params.lda;

    // Each thread handles TILE_SIZE rows
    let base_row = gid.x * TILE_SIZE;

    // Local accumulator for each row
    var sums: array<f32, 4>;
    for (var t = 0u; t < TILE_SIZE; t++) {
        sums[t] = 0.0;
    }

    // Iterate through all columns
    for (var col = 0u; col < N; col++) {
        let x_val = vector_x[col];

        // Process TILE_SIZE rows
        for (var t = 0u; t < TILE_SIZE; t++) {
            let row = base_row + t;
            if (row < M) {
                let a_idx = row * lda + col;
                sums[t] += matrix_a[a_idx] * x_val;
            }
        }
    }

    // Write results
    for (var t = 0u; t < TILE_SIZE; t++) {
        let row = base_row + t;
        if (row < M) {
            let old_y = vector_y[row];
            vector_y[row] = alpha * sums[t] + beta * old_y;
        }
    }
}

// ============================================================================
// Vectorized GEMV using vec4 loads for better memory bandwidth
// ============================================================================

@compute @workgroup_size(256)
fn gemv_vec4(
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

    // Process 4 elements at a time where possible
    let N_vec4 = N / 4u;
    let N_remainder = N % 4u;

    var partial_sum: f32 = 0.0;

    // Vectorized part: process 4 columns per iteration
    var vec_idx = lid.x;
    while (vec_idx < N_vec4) {
        let col = vec_idx * 4u;
        let a_base = row * lda + col;

        // Load 4 elements from matrix and vector
        let a0 = matrix_a[a_base];
        let a1 = matrix_a[a_base + 1u];
        let a2 = matrix_a[a_base + 2u];
        let a3 = matrix_a[a_base + 3u];

        let x0 = vector_x[col];
        let x1 = vector_x[col + 1u];
        let x2 = vector_x[col + 2u];
        let x3 = vector_x[col + 3u];

        // FMA operations
        partial_sum += a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3;

        vec_idx += WORKGROUP_SIZE;
    }

    // Handle remainder
    if (lid.x == 0u) {
        for (var i = N_vec4 * 4u; i < N; i++) {
            partial_sum += matrix_a[row * lda + i] * vector_x[i];
        }
    }

    // Store partial sum to shared memory
    shared_sum[lid.x] = partial_sum;
    workgroupBarrier();

    // Parallel reduction
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_sum[lid.x] += shared_sum[lid.x + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes the final result
    if (lid.x == 0u) {
        let old_y = vector_y[row];
        vector_y[row] = alpha * shared_sum[0] + beta * old_y;
    }
}

// ============================================================================
// Batched GEMV - Process multiple independent GEMV operations
// ============================================================================

struct BatchGemvParams {
    M: u32,
    N: u32,
    alpha: f32,
    beta: f32,
    lda: u32,
    batch_count: u32,
    stride_a: u32,      // Stride between matrices
    stride_x: u32,      // Stride between x vectors
    stride_y: u32,      // Stride between y vectors
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(1) @binding(0) var<uniform> batch_params: BatchGemvParams;

@compute @workgroup_size(256)
fn gemv_batched(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let M = batch_params.M;
    let N = batch_params.N;
    let alpha = batch_params.alpha;
    let beta = batch_params.beta;
    let lda = batch_params.lda;
    let batch_count = batch_params.batch_count;
    let stride_a = batch_params.stride_a;
    let stride_x = batch_params.stride_x;
    let stride_y = batch_params.stride_y;

    // Determine batch index and row
    let batch_idx = wid.x / M;
    let row = wid.x % M;

    if (batch_idx >= batch_count) {
        return;
    }

    // Calculate base offsets for this batch
    let a_offset = batch_idx * stride_a;
    let x_offset = batch_idx * stride_x;
    let y_offset = batch_idx * stride_y;

    // Each thread accumulates partial dot product
    var partial_sum: f32 = 0.0;

    var col = lid.x;
    while (col < N) {
        let a_idx = a_offset + row * lda + col;
        let a_val = matrix_a[a_idx];
        let x_val = vector_x[x_offset + col];
        partial_sum += a_val * x_val;
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
        let y_idx = y_offset + row;
        let old_y = vector_y[y_idx];
        vector_y[y_idx] = alpha * shared_sum[0] + beta * old_y;
    }
}

// ============================================================================
// Simple GEMV - One thread per output element (for small matrices)
// ============================================================================

@compute @workgroup_size(256)
fn gemv_simple(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let M = params.M;
    let N = params.N;
    let alpha = params.alpha;
    let beta = params.beta;
    let lda = params.lda;
    let row = gid.x;

    if (row >= M) {
        return;
    }

    var sum: f32 = 0.0;
    for (var col = 0u; col < N; col++) {
        sum += matrix_a[row * lda + col] * vector_x[col];
    }

    vector_y[row] = alpha * sum + beta * vector_y[row];
}
