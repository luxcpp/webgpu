// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Split-K GEMM for Tall Matrices in WGSL
// Works on WebGPU (Metal/Vulkan/D3D12 via Dawn/wgpu)
//
// Part of the Lux Network GPU acceleration library
// Implements Split-K parallelization for improved throughput on tall/thin matrices

// ============================================================================
// Configuration Constants
// ============================================================================

const TILE_M: u32 = 32u;
const TILE_N: u32 = 64u;
const TILE_K: u32 = 32u;
const BLOCK_SIZE: u32 = 256u;
const MAX_SPLITS: u32 = 32u;

// ============================================================================
// Kernel Bindings
// ============================================================================

struct GemmSplitKParams {
    M: u32,                    // Rows of A and C
    N: u32,                    // Columns of B and C
    K: u32,                    // Columns of A, Rows of B
    batch_size: u32,           // Batch dimension
    lda: u32,                  // Leading dimension of A
    ldb: u32,                  // Leading dimension of B
    ldc: u32,                  // Leading dimension of C
    alpha: f32,
    beta: f32,
    num_splits: u32,           // Number of K splits
    k_per_split: u32,          // K elements per split
    reduction_mode: u32,       // 0=atomic, 1=workspace, 2=two-pass
    _pad: u32,
}

// A: [batch, M, K]
// B: [batch, K, N]
// C: [batch, M, N]
// workspace: [batch, num_splits, M, N] (for workspace reduction)
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read_write> workspace: array<f32>;
@group(0) @binding(4) var<uniform> params: GemmSplitKParams;

// Shared memory
var<workgroup> tile_A: array<f32, 1024>;
var<workgroup> tile_B: array<f32, 2048>;
var<workgroup> tile_C: array<f32, 2048>;
var<workgroup> reduction_buffer: array<f32, 256>;

// ============================================================================
// Split-K GEMM with Workspace Reduction
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_splitk_workspace(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Split-K: each split computes partial results for a range of K
    // Results are written to workspace, then reduced in a second pass

    let tid = lid.x;
    let batch_idx = wid.z / params.num_splits;
    let split_idx = wid.z % params.num_splits;

    if (batch_idx >= params.batch_size) { return; }

    let M = params.M;
    let N = params.N;
    let K = params.K;
    let k_per_split = params.k_per_split;
    let num_splits = params.num_splits;

    // K range for this split
    let k_start = split_idx * k_per_split;
    let k_end = min(k_start + k_per_split, K);
    let local_k = k_end - k_start;

    if (local_k == 0u) { return; }

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;

    if (tile_row >= M || tile_col >= N) { return; }

    let num_rows = min(TILE_M, M - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    // Initialize accumulator
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        tile_C[i] = 0.0;
    }
    workgroupBarrier();

    // Process K range for this split
    let num_k_tiles = (local_k + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_local_start = k_tile * TILE_K;
        let k_global_start = k_start + k_local_start;
        let num_k = min(TILE_K, local_k - k_local_start);

        // Load A tile
        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let a_idx = (batch_idx * M + tile_row + row) * params.lda + k_global_start + col;
            tile_A[row * TILE_K + col] = A[a_idx];
        }

        // Load B tile
        for (var i = tid; i < num_k * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;
            let b_idx = (batch_idx * K + k_global_start + row) * params.ldb + tile_col + col;
            tile_B[row * TILE_N + col] = B[b_idx];
        }
        workgroupBarrier();

        // Compute partial result
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

    // Store partial result to workspace
    // workspace layout: [batch, split, M, N]
    let ws_stride = num_splits * M * N;
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let global_row = tile_row + c_row;
        let global_col = tile_col + c_col;

        let ws_idx = batch_idx * ws_stride + split_idx * M * N + global_row * N + global_col;
        workspace[ws_idx] = params.alpha * tile_C[c_row * TILE_N + c_col];
    }
}

// ============================================================================
// Split-K Reduction Pass
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_splitk_reduce(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Reduce workspace across splits into final output

    let tid = lid.x;
    let batch_idx = wid.z;

    if (batch_idx >= params.batch_size) { return; }

    let M = params.M;
    let N = params.N;
    let num_splits = params.num_splits;

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;

    if (tile_row >= M || tile_col >= N) { return; }

    let num_rows = min(TILE_M, M - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    let ws_stride = num_splits * M * N;

    // Reduce across splits
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let global_row = tile_row + c_row;
        let global_col = tile_col + c_col;

        var sum: f32 = 0.0;
        for (var s = 0u; s < num_splits; s++) {
            let ws_idx = batch_idx * ws_stride + s * M * N + global_row * N + global_col;
            sum += workspace[ws_idx];
        }

        let c_idx = (batch_idx * M + global_row) * params.ldc + global_col;

        if (params.beta != 0.0) {
            sum += params.beta * C[c_idx];
        }

        C[c_idx] = sum;
    }
}

// ============================================================================
// Split-K with In-Register Reduction
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_splitk_inline(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Single-pass split-K with thread-local accumulation and shared memory reduction
    // Better for smaller matrices where workspace overhead is significant

    let tid = lid.x;
    let batch_idx = wid.z;

    if (batch_idx >= params.batch_size) { return; }

    let M = params.M;
    let N = params.N;
    let K = params.K;
    let num_splits = params.num_splits;
    let k_per_split = params.k_per_split;

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;

    if (tile_row >= M || tile_col >= N) { return; }

    let num_rows = min(TILE_M, M - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    // Each thread handles a split, then we reduce
    let split_per_thread = (num_splits + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    // Thread-local accumulator
    var local_acc: array<f32, 8>;  // Limited per-thread storage
    for (var i = 0u; i < 8u; i++) {
        local_acc[i] = 0.0;
    }

    // Process splits assigned to this thread
    for (var s = 0u; s < split_per_thread; s++) {
        let split_idx = tid * split_per_thread + s;
        if (split_idx >= num_splits) { break; }

        let k_start = split_idx * k_per_split;
        let k_end = min(k_start + k_per_split, K);
        let local_k = k_end - k_start;

        if (local_k == 0u) { continue; }

        // Each thread computes for subset of output elements
        let elements_per_thread = (num_rows * num_cols + BLOCK_SIZE - 1u) / BLOCK_SIZE;
        let elem_start = tid * elements_per_thread;
        let elem_end = min(elem_start + elements_per_thread, num_rows * num_cols);

        for (var elem = elem_start; elem < elem_end; elem++) {
            let c_row = elem / num_cols;
            let c_col = elem % num_cols;

            var sum: f32 = 0.0;
            for (var k = k_start; k < k_end; k++) {
                let a_idx = (batch_idx * M + tile_row + c_row) * params.lda + k;
                let b_idx = (batch_idx * K + k) * params.ldb + tile_col + c_col;
                sum += A[a_idx] * B[b_idx];
            }

            local_acc[elem - elem_start] += sum;
        }
    }

    // Shared memory reduction
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        tile_C[i] = 0.0;
    }
    workgroupBarrier();

    // Accumulate local results to shared memory
    let elements_per_thread = (num_rows * num_cols + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    let elem_start = tid * elements_per_thread;
    for (var e = 0u; e < elements_per_thread; e++) {
        let elem = elem_start + e;
        if (elem < num_rows * num_cols) {
            // Atomic-like operation (sequential within workgroup for correctness)
            // Note: WGSL doesn't have atomicAdd for f32, simplified here
            tile_C[elem] += local_acc[e];
        }
    }
    workgroupBarrier();

    // Store final result
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let global_row = tile_row + c_row;
        let global_col = tile_col + c_col;

        let c_idx = (batch_idx * M + global_row) * params.ldc + global_col;

        var result = params.alpha * tile_C[i];
        if (params.beta != 0.0) {
            result += params.beta * C[c_idx];
        }
        C[c_idx] = result;
    }
}

// ============================================================================
// Parallel Reduction for Tall-Thin Matrices (M >> N, K)
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_splitk_tall(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Optimized for tall matrices where M >> K and M >> N
    // Uses M-dimension parallelism with K splits

    let tid = lid.x;
    let batch_idx = wid.z / params.num_splits;
    let split_idx = wid.z % params.num_splits;

    if (batch_idx >= params.batch_size) { return; }

    let M = params.M;
    let N = params.N;
    let K = params.K;
    let num_splits = params.num_splits;
    let k_per_split = params.k_per_split;

    let k_start = split_idx * k_per_split;
    let k_end = min(k_start + k_per_split, K);
    let local_k = k_end - k_start;

    if (local_k == 0u) { return; }

    // Each workgroup handles a tile of M x N
    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;

    if (tile_row >= M || tile_col >= N) { return; }

    let num_rows = min(TILE_M, M - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    // For tall matrices, load all of B into shared memory (B is thin)
    if (N <= TILE_N) {
        // B fits in shared memory, load once
        for (var i = tid; i < local_k * N; i += BLOCK_SIZE) {
            let row = i / N;
            let col = i % N;
            let b_idx = (batch_idx * K + k_start + row) * params.ldb + col;
            tile_B[row * N + col] = B[b_idx];
        }
        workgroupBarrier();
    }

    // Process rows with streaming A
    for (var row_base = 0u; row_base < num_rows; row_base += 4u) {
        let rows_to_process = min(4u, num_rows - row_base);

        // Load A rows
        for (var i = tid; i < rows_to_process * local_k; i += BLOCK_SIZE) {
            let row = i / local_k;
            let col = i % local_k;
            let a_idx = (batch_idx * M + tile_row + row_base + row) * params.lda + k_start + col;
            tile_A[row * TILE_K + col] = A[a_idx];
        }
        workgroupBarrier();

        // Compute for these rows
        for (var i = tid; i < rows_to_process * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;

            var sum: f32 = 0.0;
            for (var k = 0u; k < local_k; k++) {
                let a_val = tile_A[row * TILE_K + k];
                var b_val: f32;
                if (N <= TILE_N) {
                    b_val = tile_B[k * N + tile_col + col];
                } else {
                    let b_idx = (batch_idx * K + k_start + k) * params.ldb + tile_col + col;
                    b_val = B[b_idx];
                }
                sum += a_val * b_val;
            }

            // Store partial result
            let global_row = tile_row + row_base + row;
            let global_col = tile_col + col;
            let ws_idx = batch_idx * (num_splits * M * N) + split_idx * M * N + global_row * N + global_col;

            workspace[ws_idx] = params.alpha * sum;
        }
        workgroupBarrier();
    }
}

// ============================================================================
// Stream-K GEMM (Dynamic Work Distribution)
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_streamk(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wgs: vec3<u32>
) {
    // Stream-K: distribute work evenly across all SMs
    // Each workgroup processes a stream of K iterations

    let tid = lid.x;
    let wg_id = wid.x + wid.y * num_wgs.x + wid.z * num_wgs.x * num_wgs.y;
    let total_wgs = num_wgs.x * num_wgs.y * num_wgs.z;

    let M = params.M;
    let N = params.N;
    let K = params.K;
    let batch_size = params.batch_size;

    // Total tiles across all dimensions
    let tiles_m = (M + TILE_M - 1u) / TILE_M;
    let tiles_n = (N + TILE_N - 1u) / TILE_N;
    let tiles_k = (K + TILE_K - 1u) / TILE_K;
    let total_tiles = batch_size * tiles_m * tiles_n * tiles_k;

    // Distribute tiles evenly
    let tiles_per_wg = (total_tiles + total_wgs - 1u) / total_wgs;
    let tile_start = wg_id * tiles_per_wg;
    let tile_end = min(tile_start + tiles_per_wg, total_tiles);

    // Track current output tile for accumulation
    var current_batch = 0xFFFFFFFFu;
    var current_m_tile = 0xFFFFFFFFu;
    var current_n_tile = 0xFFFFFFFFu;

    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        tile_C[i] = 0.0;
    }

    for (var tile_idx = tile_start; tile_idx < tile_end; tile_idx++) {
        // Decode tile index
        let k_tile = tile_idx % tiles_k;
        let mn_batch = tile_idx / tiles_k;
        let n_tile = mn_batch % tiles_n;
        let m_batch = mn_batch / tiles_n;
        let m_tile = m_batch % tiles_m;
        let batch_idx = m_batch / tiles_m;

        // Check if we switched output tiles
        if (batch_idx != current_batch || m_tile != current_m_tile || n_tile != current_n_tile) {
            // Store previous accumulation (if any)
            if (current_batch != 0xFFFFFFFFu) {
                let prev_tile_row = current_m_tile * TILE_M;
                let prev_tile_col = current_n_tile * TILE_N;
                let prev_num_rows = min(TILE_M, M - prev_tile_row);
                let prev_num_cols = min(TILE_N, N - prev_tile_col);

                for (var i = tid; i < prev_num_rows * prev_num_cols; i += BLOCK_SIZE) {
                    let c_row = i / prev_num_cols;
                    let c_col = i % prev_num_cols;
                    let c_idx = (current_batch * M + prev_tile_row + c_row) * params.ldc + prev_tile_col + c_col;

                    // Atomic add for correctness (simplified - WGSL lacks f32 atomics)
                    C[c_idx] = C[c_idx] + params.alpha * tile_C[c_row * TILE_N + c_col];
                }
            }

            // Reset for new output tile
            current_batch = batch_idx;
            current_m_tile = m_tile;
            current_n_tile = n_tile;

            for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
                tile_C[i] = 0.0;
            }
            workgroupBarrier();
        }

        let tile_row = m_tile * TILE_M;
        let tile_col = n_tile * TILE_N;
        let k_start = k_tile * TILE_K;
        let num_rows = min(TILE_M, M - tile_row);
        let num_cols = min(TILE_N, N - tile_col);
        let num_k = min(TILE_K, K - k_start);

        // Load and compute
        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let a_idx = (batch_idx * M + tile_row + row) * params.lda + k_start + col;
            tile_A[row * TILE_K + col] = A[a_idx];
        }

        for (var i = tid; i < num_k * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;
            let b_idx = (batch_idx * K + k_start + row) * params.ldb + tile_col + col;
            tile_B[row * TILE_N + col] = B[b_idx];
        }
        workgroupBarrier();

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

    // Store final accumulation
    if (current_batch != 0xFFFFFFFFu) {
        let tile_row = current_m_tile * TILE_M;
        let tile_col = current_n_tile * TILE_N;
        let num_rows = min(TILE_M, M - tile_row);
        let num_cols = min(TILE_N, N - tile_col);

        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;
            let c_idx = (current_batch * M + tile_row + c_row) * params.ldc + tile_col + c_col;

            C[c_idx] = C[c_idx] + params.alpha * tile_C[c_row * TILE_N + c_col];
        }
    }
}
