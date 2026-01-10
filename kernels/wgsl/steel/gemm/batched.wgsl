// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Batched GEMM in WGSL
// Efficient batched matrix multiplication for WebGPU
// Works on WebGPU (Metal/Vulkan/D3D12 via Dawn/wgpu)

// ============================================================================
// Configuration Constants
// ============================================================================

const TILE_M: u32 = 64u;
const TILE_N: u32 = 64u;
const TILE_K: u32 = 16u;
const BLOCK_SIZE: u32 = 256u;
const THREADS_PER_ROW: u32 = 16u;

// ============================================================================
// Bindings for Strided Batched GEMM
// ============================================================================

struct BatchedGemmParams {
    M: u32,
    N: u32,
    K: u32,
    batch_count: u32,
    lda: u32,        // Leading dimension of A
    ldb: u32,        // Leading dimension of B
    ldc: u32,        // Leading dimension of C
    stride_a: u32,   // Batch stride for A
    stride_b: u32,   // Batch stride for B
    stride_c: u32,   // Batch stride for C
    alpha: f32,
    beta: f32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: BatchedGemmParams;

// Shared memory tiles
var<workgroup> A_tile: array<f32, 1024>;  // TILE_M * TILE_K
var<workgroup> B_tile: array<f32, 1024>;  // TILE_K * TILE_N

// ============================================================================
// Strided Batched GEMM: C[b] = alpha * A[b] @ B[b] + beta * C[b]
// A[b] is at offset b * stride_a
// B[b] is at offset b * stride_b
// C[b] is at offset b * stride_c
// ============================================================================

@compute @workgroup_size(256)
fn batched_gemm_strided(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z;
    let tile_m = wid.x;
    let tile_n = wid.y;

    if (batch_idx >= params.batch_count) { return; }

    let M = params.M;
    let N = params.N;
    let K = params.K;
    let lda = params.lda;
    let ldb = params.ldb;
    let ldc = params.ldc;
    let alpha = params.alpha;
    let beta = params.beta;

    // Batch offsets
    let A_offset = batch_idx * params.stride_a;
    let B_offset = batch_idx * params.stride_b;
    let C_offset = batch_idx * params.stride_c;

    let m_start = tile_m * TILE_M;
    let n_start = tile_n * TILE_N;

    // Thread's position within tile
    let thread_row = tid / THREADS_PER_ROW;
    let thread_col = tid % THREADS_PER_ROW;

    // Registers for accumulation
    var acc: array<f32, 16>;  // 4x4 per thread
    for (var i = 0u; i < 16u; i++) {
        acc[i] = 0.0;
    }

    // Number of K tiles
    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;

        // Cooperatively load A tile
        for (var i = tid; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            let m_local = i / TILE_K;
            let k_local = i % TILE_K;
            let m_global = m_start + m_local;
            let k_global = k_start + k_local;

            var val: f32 = 0.0;
            if (m_global < M && k_global < K) {
                val = A[A_offset + m_global * lda + k_global];
            }
            A_tile[m_local * TILE_K + k_local] = val;
        }

        // Cooperatively load B tile
        for (var i = tid; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            let k_local = i / TILE_N;
            let n_local = i % TILE_N;
            let k_global = k_start + k_local;
            let n_global = n_start + n_local;

            var val: f32 = 0.0;
            if (k_global < K && n_global < N) {
                val = B[B_offset + k_global * ldb + n_global];
            }
            B_tile[k_local * TILE_N + n_local] = val;
        }
        workgroupBarrier();

        // Compute partial product
        for (var k = 0u; k < TILE_K; k++) {
            // Each thread computes 4x4 output elements
            for (var m_offset = 0u; m_offset < 4u; m_offset++) {
                let m_idx = thread_row * 4u + m_offset;
                let a_val = A_tile[m_idx * TILE_K + k];

                for (var n_offset = 0u; n_offset < 4u; n_offset++) {
                    let n_idx = thread_col * 4u + n_offset;
                    let b_val = B_tile[k * TILE_N + n_idx];
                    acc[m_offset * 4u + n_offset] += a_val * b_val;
                }
            }
        }
        workgroupBarrier();
    }

    // Store results
    for (var m_offset = 0u; m_offset < 4u; m_offset++) {
        for (var n_offset = 0u; n_offset < 4u; n_offset++) {
            let m_global = m_start + thread_row * 4u + m_offset;
            let n_global = n_start + thread_col * 4u + n_offset;

            if (m_global < M && n_global < N) {
                let c_idx = C_offset + m_global * ldc + n_global;
                var result = alpha * acc[m_offset * 4u + n_offset];
                if (beta != 0.0) {
                    result += beta * C[c_idx];
                }
                C[c_idx] = result;
            }
        }
    }
}

// ============================================================================
// Transposed Batched GEMM variants
// ============================================================================

// C = alpha * A^T @ B + beta * C
@compute @workgroup_size(256)
fn batched_gemm_strided_at(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z;
    let tile_m = wid.x;
    let tile_n = wid.y;

    if (batch_idx >= params.batch_count) { return; }

    let M = params.M;
    let N = params.N;
    let K = params.K;
    let lda = params.lda;
    let ldb = params.ldb;
    let ldc = params.ldc;
    let alpha = params.alpha;
    let beta = params.beta;

    let A_offset = batch_idx * params.stride_a;
    let B_offset = batch_idx * params.stride_b;
    let C_offset = batch_idx * params.stride_c;

    let m_start = tile_m * TILE_M;
    let n_start = tile_n * TILE_N;

    let thread_row = tid / THREADS_PER_ROW;
    let thread_col = tid % THREADS_PER_ROW;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) {
        acc[i] = 0.0;
    }

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;

        // Load A^T (transposed access)
        for (var i = tid; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            let m_local = i / TILE_K;
            let k_local = i % TILE_K;
            let m_global = m_start + m_local;
            let k_global = k_start + k_local;

            var val: f32 = 0.0;
            if (m_global < M && k_global < K) {
                // A^T: read A[k, m] instead of A[m, k]
                val = A[A_offset + k_global * lda + m_global];
            }
            A_tile[m_local * TILE_K + k_local] = val;
        }

        // Load B normally
        for (var i = tid; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            let k_local = i / TILE_N;
            let n_local = i % TILE_N;
            let k_global = k_start + k_local;
            let n_global = n_start + n_local;

            var val: f32 = 0.0;
            if (k_global < K && n_global < N) {
                val = B[B_offset + k_global * ldb + n_global];
            }
            B_tile[k_local * TILE_N + n_local] = val;
        }
        workgroupBarrier();

        // Compute
        for (var k = 0u; k < TILE_K; k++) {
            for (var m_offset = 0u; m_offset < 4u; m_offset++) {
                let m_idx = thread_row * 4u + m_offset;
                let a_val = A_tile[m_idx * TILE_K + k];

                for (var n_offset = 0u; n_offset < 4u; n_offset++) {
                    let n_idx = thread_col * 4u + n_offset;
                    let b_val = B_tile[k * TILE_N + n_idx];
                    acc[m_offset * 4u + n_offset] += a_val * b_val;
                }
            }
        }
        workgroupBarrier();
    }

    // Store
    for (var m_offset = 0u; m_offset < 4u; m_offset++) {
        for (var n_offset = 0u; n_offset < 4u; n_offset++) {
            let m_global = m_start + thread_row * 4u + m_offset;
            let n_global = n_start + thread_col * 4u + n_offset;

            if (m_global < M && n_global < N) {
                let c_idx = C_offset + m_global * ldc + n_global;
                var result = alpha * acc[m_offset * 4u + n_offset];
                if (beta != 0.0) {
                    result += beta * C[c_idx];
                }
                C[c_idx] = result;
            }
        }
    }
}

// C = alpha * A @ B^T + beta * C
@compute @workgroup_size(256)
fn batched_gemm_strided_bt(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z;
    let tile_m = wid.x;
    let tile_n = wid.y;

    if (batch_idx >= params.batch_count) { return; }

    let M = params.M;
    let N = params.N;
    let K = params.K;
    let lda = params.lda;
    let ldb = params.ldb;
    let ldc = params.ldc;
    let alpha = params.alpha;
    let beta = params.beta;

    let A_offset = batch_idx * params.stride_a;
    let B_offset = batch_idx * params.stride_b;
    let C_offset = batch_idx * params.stride_c;

    let m_start = tile_m * TILE_M;
    let n_start = tile_n * TILE_N;

    let thread_row = tid / THREADS_PER_ROW;
    let thread_col = tid % THREADS_PER_ROW;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) {
        acc[i] = 0.0;
    }

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;

        // Load A normally
        for (var i = tid; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            let m_local = i / TILE_K;
            let k_local = i % TILE_K;
            let m_global = m_start + m_local;
            let k_global = k_start + k_local;

            var val: f32 = 0.0;
            if (m_global < M && k_global < K) {
                val = A[A_offset + m_global * lda + k_global];
            }
            A_tile[m_local * TILE_K + k_local] = val;
        }

        // Load B^T (transposed access)
        for (var i = tid; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            let k_local = i / TILE_N;
            let n_local = i % TILE_N;
            let k_global = k_start + k_local;
            let n_global = n_start + n_local;

            var val: f32 = 0.0;
            if (k_global < K && n_global < N) {
                // B^T: read B[n, k] instead of B[k, n]
                val = B[B_offset + n_global * ldb + k_global];
            }
            B_tile[k_local * TILE_N + n_local] = val;
        }
        workgroupBarrier();

        // Compute
        for (var k = 0u; k < TILE_K; k++) {
            for (var m_offset = 0u; m_offset < 4u; m_offset++) {
                let m_idx = thread_row * 4u + m_offset;
                let a_val = A_tile[m_idx * TILE_K + k];

                for (var n_offset = 0u; n_offset < 4u; n_offset++) {
                    let n_idx = thread_col * 4u + n_offset;
                    let b_val = B_tile[k * TILE_N + n_idx];
                    acc[m_offset * 4u + n_offset] += a_val * b_val;
                }
            }
        }
        workgroupBarrier();
    }

    // Store
    for (var m_offset = 0u; m_offset < 4u; m_offset++) {
        for (var n_offset = 0u; n_offset < 4u; n_offset++) {
            let m_global = m_start + thread_row * 4u + m_offset;
            let n_global = n_start + thread_col * 4u + n_offset;

            if (m_global < M && n_global < N) {
                let c_idx = C_offset + m_global * ldc + n_global;
                var result = alpha * acc[m_offset * 4u + n_offset];
                if (beta != 0.0) {
                    result += beta * C[c_idx];
                }
                C[c_idx] = result;
            }
        }
    }
}

// C = alpha * A^T @ B^T + beta * C
@compute @workgroup_size(256)
fn batched_gemm_strided_at_bt(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z;
    let tile_m = wid.x;
    let tile_n = wid.y;

    if (batch_idx >= params.batch_count) { return; }

    let M = params.M;
    let N = params.N;
    let K = params.K;
    let lda = params.lda;
    let ldb = params.ldb;
    let ldc = params.ldc;
    let alpha = params.alpha;
    let beta = params.beta;

    let A_offset = batch_idx * params.stride_a;
    let B_offset = batch_idx * params.stride_b;
    let C_offset = batch_idx * params.stride_c;

    let m_start = tile_m * TILE_M;
    let n_start = tile_n * TILE_N;

    let thread_row = tid / THREADS_PER_ROW;
    let thread_col = tid % THREADS_PER_ROW;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) {
        acc[i] = 0.0;
    }

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;

        // Load A^T
        for (var i = tid; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            let m_local = i / TILE_K;
            let k_local = i % TILE_K;
            let m_global = m_start + m_local;
            let k_global = k_start + k_local;

            var val: f32 = 0.0;
            if (m_global < M && k_global < K) {
                val = A[A_offset + k_global * lda + m_global];
            }
            A_tile[m_local * TILE_K + k_local] = val;
        }

        // Load B^T
        for (var i = tid; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            let k_local = i / TILE_N;
            let n_local = i % TILE_N;
            let k_global = k_start + k_local;
            let n_global = n_start + n_local;

            var val: f32 = 0.0;
            if (k_global < K && n_global < N) {
                val = B[B_offset + n_global * ldb + k_global];
            }
            B_tile[k_local * TILE_N + n_local] = val;
        }
        workgroupBarrier();

        // Compute
        for (var k = 0u; k < TILE_K; k++) {
            for (var m_offset = 0u; m_offset < 4u; m_offset++) {
                let m_idx = thread_row * 4u + m_offset;
                let a_val = A_tile[m_idx * TILE_K + k];

                for (var n_offset = 0u; n_offset < 4u; n_offset++) {
                    let n_idx = thread_col * 4u + n_offset;
                    let b_val = B_tile[k * TILE_N + n_idx];
                    acc[m_offset * 4u + n_offset] += a_val * b_val;
                }
            }
        }
        workgroupBarrier();
    }

    // Store
    for (var m_offset = 0u; m_offset < 4u; m_offset++) {
        for (var n_offset = 0u; n_offset < 4u; n_offset++) {
            let m_global = m_start + thread_row * 4u + m_offset;
            let n_global = n_start + thread_col * 4u + n_offset;

            if (m_global < M && n_global < N) {
                let c_idx = C_offset + m_global * ldc + n_global;
                var result = alpha * acc[m_offset * 4u + n_offset];
                if (beta != 0.0) {
                    result += beta * C[c_idx];
                }
                C[c_idx] = result;
            }
        }
    }
}

// ============================================================================
// Grouped Batched GEMM
// Different matrices can have different sizes (variable batch)
// ============================================================================

struct GroupedGemmParams {
    num_groups: u32,
    max_M: u32,
    max_N: u32,
    max_K: u32,
    alpha: f32,
    beta: f32,
    _pad1: u32,
    _pad2: u32,
}

struct GroupInfo {
    M: u32,
    N: u32,
    K: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    A_offset: u32,
    B_offset: u32,
    C_offset: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(4) var<storage, read> group_infos: array<GroupInfo>;
@group(0) @binding(5) var<uniform> grouped_params: GroupedGemmParams;

@compute @workgroup_size(256)
fn batched_gemm_grouped(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let group_idx = wid.z;
    let tile_m = wid.x;
    let tile_n = wid.y;

    if (group_idx >= grouped_params.num_groups) { return; }

    let info = group_infos[group_idx];
    let M = info.M;
    let N = info.N;
    let K = info.K;
    let lda = info.lda;
    let ldb = info.ldb;
    let ldc = info.ldc;
    let alpha = grouped_params.alpha;
    let beta = grouped_params.beta;

    let A_offset = info.A_offset;
    let B_offset = info.B_offset;
    let C_offset = info.C_offset;

    let m_start = tile_m * TILE_M;
    let n_start = tile_n * TILE_N;

    if (m_start >= M || n_start >= N) { return; }

    let thread_row = tid / THREADS_PER_ROW;
    let thread_col = tid % THREADS_PER_ROW;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) {
        acc[i] = 0.0;
    }

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;

        // Load A tile
        for (var i = tid; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            let m_local = i / TILE_K;
            let k_local = i % TILE_K;
            let m_global = m_start + m_local;
            let k_global = k_start + k_local;

            var val: f32 = 0.0;
            if (m_global < M && k_global < K) {
                val = A[A_offset + m_global * lda + k_global];
            }
            A_tile[m_local * TILE_K + k_local] = val;
        }

        // Load B tile
        for (var i = tid; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            let k_local = i / TILE_N;
            let n_local = i % TILE_N;
            let k_global = k_start + k_local;
            let n_global = n_start + n_local;

            var val: f32 = 0.0;
            if (k_global < K && n_global < N) {
                val = B[B_offset + k_global * ldb + n_global];
            }
            B_tile[k_local * TILE_N + n_local] = val;
        }
        workgroupBarrier();

        // Compute
        for (var k = 0u; k < TILE_K; k++) {
            for (var m_offset = 0u; m_offset < 4u; m_offset++) {
                let m_idx = thread_row * 4u + m_offset;
                let a_val = A_tile[m_idx * TILE_K + k];

                for (var n_offset = 0u; n_offset < 4u; n_offset++) {
                    let n_idx = thread_col * 4u + n_offset;
                    let b_val = B_tile[k * TILE_N + n_idx];
                    acc[m_offset * 4u + n_offset] += a_val * b_val;
                }
            }
        }
        workgroupBarrier();
    }

    // Store
    for (var m_offset = 0u; m_offset < 4u; m_offset++) {
        for (var n_offset = 0u; n_offset < 4u; n_offset++) {
            let m_global = m_start + thread_row * 4u + m_offset;
            let n_global = n_start + thread_col * 4u + n_offset;

            if (m_global < M && n_global < N) {
                let c_idx = C_offset + m_global * ldc + n_global;
                var result = alpha * acc[m_offset * 4u + n_offset];
                if (beta != 0.0) {
                    result += beta * C[c_idx];
                }
                C[c_idx] = result;
            }
        }
    }
}

// ============================================================================
// Batched GEMM with Fused Activation
// ============================================================================

@compute @workgroup_size(256)
fn batched_gemm_strided_relu(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z;
    let tile_m = wid.x;
    let tile_n = wid.y;

    if (batch_idx >= params.batch_count) { return; }

    let M = params.M;
    let N = params.N;
    let K = params.K;
    let lda = params.lda;
    let ldb = params.ldb;
    let ldc = params.ldc;
    let alpha = params.alpha;
    let beta = params.beta;

    let A_offset = batch_idx * params.stride_a;
    let B_offset = batch_idx * params.stride_b;
    let C_offset = batch_idx * params.stride_c;

    let m_start = tile_m * TILE_M;
    let n_start = tile_n * TILE_N;

    let thread_row = tid / THREADS_PER_ROW;
    let thread_col = tid % THREADS_PER_ROW;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) {
        acc[i] = 0.0;
    }

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;

        for (var i = tid; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            let m_local = i / TILE_K;
            let k_local = i % TILE_K;
            let m_global = m_start + m_local;
            let k_global = k_start + k_local;

            var val: f32 = 0.0;
            if (m_global < M && k_global < K) {
                val = A[A_offset + m_global * lda + k_global];
            }
            A_tile[m_local * TILE_K + k_local] = val;
        }

        for (var i = tid; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            let k_local = i / TILE_N;
            let n_local = i % TILE_N;
            let k_global = k_start + k_local;
            let n_global = n_start + n_local;

            var val: f32 = 0.0;
            if (k_global < K && n_global < N) {
                val = B[B_offset + k_global * ldb + n_global];
            }
            B_tile[k_local * TILE_N + n_local] = val;
        }
        workgroupBarrier();

        for (var k = 0u; k < TILE_K; k++) {
            for (var m_offset = 0u; m_offset < 4u; m_offset++) {
                let m_idx = thread_row * 4u + m_offset;
                let a_val = A_tile[m_idx * TILE_K + k];

                for (var n_offset = 0u; n_offset < 4u; n_offset++) {
                    let n_idx = thread_col * 4u + n_offset;
                    let b_val = B_tile[k * TILE_N + n_idx];
                    acc[m_offset * 4u + n_offset] += a_val * b_val;
                }
            }
        }
        workgroupBarrier();
    }

    // Store with ReLU activation
    for (var m_offset = 0u; m_offset < 4u; m_offset++) {
        for (var n_offset = 0u; n_offset < 4u; n_offset++) {
            let m_global = m_start + thread_row * 4u + m_offset;
            let n_global = n_start + thread_col * 4u + n_offset;

            if (m_global < M && n_global < N) {
                let c_idx = C_offset + m_global * ldc + n_global;
                var result = alpha * acc[m_offset * 4u + n_offset];
                if (beta != 0.0) {
                    result += beta * C[c_idx];
                }
                // ReLU activation
                C[c_idx] = max(result, 0.0);
            }
        }
    }
}

@compute @workgroup_size(256)
fn batched_gemm_strided_gelu(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z;
    let tile_m = wid.x;
    let tile_n = wid.y;

    if (batch_idx >= params.batch_count) { return; }

    let M = params.M;
    let N = params.N;
    let K = params.K;
    let lda = params.lda;
    let ldb = params.ldb;
    let ldc = params.ldc;
    let alpha = params.alpha;
    let beta = params.beta;

    let A_offset = batch_idx * params.stride_a;
    let B_offset = batch_idx * params.stride_b;
    let C_offset = batch_idx * params.stride_c;

    let m_start = tile_m * TILE_M;
    let n_start = tile_n * TILE_N;

    let thread_row = tid / THREADS_PER_ROW;
    let thread_col = tid % THREADS_PER_ROW;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) {
        acc[i] = 0.0;
    }

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;

        for (var i = tid; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            let m_local = i / TILE_K;
            let k_local = i % TILE_K;
            let m_global = m_start + m_local;
            let k_global = k_start + k_local;

            var val: f32 = 0.0;
            if (m_global < M && k_global < K) {
                val = A[A_offset + m_global * lda + k_global];
            }
            A_tile[m_local * TILE_K + k_local] = val;
        }

        for (var i = tid; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            let k_local = i / TILE_N;
            let n_local = i % TILE_N;
            let k_global = k_start + k_local;
            let n_global = n_start + n_local;

            var val: f32 = 0.0;
            if (k_global < K && n_global < N) {
                val = B[B_offset + k_global * ldb + n_global];
            }
            B_tile[k_local * TILE_N + n_local] = val;
        }
        workgroupBarrier();

        for (var k = 0u; k < TILE_K; k++) {
            for (var m_offset = 0u; m_offset < 4u; m_offset++) {
                let m_idx = thread_row * 4u + m_offset;
                let a_val = A_tile[m_idx * TILE_K + k];

                for (var n_offset = 0u; n_offset < 4u; n_offset++) {
                    let n_idx = thread_col * 4u + n_offset;
                    let b_val = B_tile[k * TILE_N + n_idx];
                    acc[m_offset * 4u + n_offset] += a_val * b_val;
                }
            }
        }
        workgroupBarrier();
    }

    // Store with GELU activation
    let sqrt_2_over_pi = 0.7978845608;
    let gelu_coeff = 0.044715;

    for (var m_offset = 0u; m_offset < 4u; m_offset++) {
        for (var n_offset = 0u; n_offset < 4u; n_offset++) {
            let m_global = m_start + thread_row * 4u + m_offset;
            let n_global = n_start + thread_col * 4u + n_offset;

            if (m_global < M && n_global < N) {
                let c_idx = C_offset + m_global * ldc + n_global;
                var result = alpha * acc[m_offset * 4u + n_offset];
                if (beta != 0.0) {
                    result += beta * C[c_idx];
                }
                // GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let x = result;
                let inner = sqrt_2_over_pi * (x + gelu_coeff * x * x * x);
                C[c_idx] = 0.5 * x * (1.0 + tanh(inner));
            }
        }
    }
}
