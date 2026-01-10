// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel GEMM with Gather Operations in WGSL
// Works on WebGPU (Metal/Vulkan/D3D12 via Dawn/wgpu)
//
// Part of the Lux Network GPU acceleration library
// Implements GEMM with indirect memory access (gather/scatter) for embeddings

// ============================================================================
// Configuration Constants
// ============================================================================

const TILE_M: u32 = 32u;       // Output rows per workgroup
const TILE_N: u32 = 64u;       // Output columns per workgroup
const TILE_K: u32 = 16u;       // Reduction tile size
const BLOCK_SIZE: u32 = 256u;

// ============================================================================
// Gather Modes
// ============================================================================

const GATHER_NONE: u32 = 0u;
const GATHER_A_ROWS: u32 = 1u;      // Gather rows of A using indices
const GATHER_B_COLS: u32 = 2u;      // Gather columns of B using indices
const GATHER_BOTH: u32 = 3u;        // Gather both A rows and B columns
const SCATTER_C: u32 = 4u;          // Scatter results to C using indices

// ============================================================================
// Kernel Bindings
// ============================================================================

struct GemmGatherParams {
    M: u32,                    // Logical rows of A
    N: u32,                    // Logical columns of B
    K: u32,                    // Columns of A, Rows of B
    batch_size: u32,           // Batch size
    num_indices_a: u32,        // Number of gather indices for A
    num_indices_b: u32,        // Number of gather indices for B
    gather_mode: u32,          // Gather mode
    embedding_dim: u32,        // Embedding dimension (for embedding lookups)
    lda: u32,                  // Leading dimension of A
    ldb: u32,                  // Leading dimension of B
    ldc: u32,                  // Leading dimension of C
    alpha: f32,                // Scalar multiplier
    beta: f32,                 // Scalar for C
    _pad: vec3<u32>,
}

// A: [vocab_size, embedding_dim] or [M, K]
// B: [K, N]
// C: [num_indices, N] or [M, N]
// indices_a: [num_indices_a] - row indices for A
// indices_b: [num_indices_b] - column indices for B
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> indices_a: array<u32>;
@group(0) @binding(4) var<storage, read> indices_b: array<u32>;
@group(0) @binding(5) var<uniform> params: GemmGatherParams;

// Shared memory
var<workgroup> tile_A: array<f32, 1024>;
var<workgroup> tile_B: array<f32, 1024>;
var<workgroup> tile_C: array<f32, 2048>;
var<workgroup> local_indices: array<u32, 64>;

// ============================================================================
// Embedding Lookup (Gather A Rows)
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_gather_a(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Gather rows of A using indices, then multiply with B
    // Used for embedding lookups: output = embeddings[indices] @ projection

    let tid = lid.x;
    let batch_idx = wid.z;

    if (batch_idx >= params.batch_size) { return; }

    let num_indices = params.num_indices_a;
    let K = params.K;
    let N = params.N;
    let lda = params.lda;
    let ldb = params.ldb;

    let tile_row = wid.x * TILE_M;  // Index into indices array
    let tile_col = wid.y * TILE_N;  // Output column
    let num_rows = min(TILE_M, num_indices - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    // Load indices for this tile
    for (var i = tid; i < num_rows; i += BLOCK_SIZE) {
        local_indices[i] = indices_a[batch_idx * num_indices + tile_row + i];
    }
    workgroupBarrier();

    // Initialize accumulator
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        tile_C[i] = 0.0;
    }
    workgroupBarrier();

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, K - k_start);

        // Gather A rows using indices
        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let actual_row = local_indices[row];  // Indirect access

            let a_idx = batch_idx * params.M * lda + actual_row * lda + k_start + col;
            tile_A[row * TILE_K + col] = A[a_idx];
        }

        // Load B tile (standard access)
        for (var i = tid; i < num_k * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;
            let b_idx = batch_idx * K * ldb + (k_start + row) * ldb + tile_col + col;
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
        let global_row = tile_row + c_row;
        let global_col = tile_col + c_col;

        let c_idx = batch_idx * num_indices * params.ldc + global_row * params.ldc + global_col;
        C[c_idx] = params.alpha * tile_C[c_row * TILE_N + c_col];
    }
}

// ============================================================================
// Gather B Columns
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_gather_b(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Gather columns of B using indices
    // Used for: output = input @ embeddings[:, indices]

    let tid = lid.x;
    let batch_idx = wid.z;

    if (batch_idx >= params.batch_size) { return; }

    let M = params.M;
    let K = params.K;
    let num_indices = params.num_indices_b;
    let lda = params.lda;
    let ldb = params.ldb;

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;  // Index into indices array
    let num_rows = min(TILE_M, M - tile_row);
    let num_cols = min(TILE_N, num_indices - tile_col);

    // Load column indices
    for (var i = tid; i < num_cols; i += BLOCK_SIZE) {
        local_indices[i] = indices_b[batch_idx * num_indices + tile_col + i];
    }
    workgroupBarrier();

    // Initialize
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        tile_C[i] = 0.0;
    }
    workgroupBarrier();

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, K - k_start);

        // Load A tile (standard)
        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let a_idx = batch_idx * M * lda + (tile_row + row) * lda + k_start + col;
            tile_A[row * TILE_K + col] = A[a_idx];
        }

        // Gather B columns using indices
        for (var i = tid; i < num_k * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;
            let actual_col = local_indices[col];  // Indirect access

            let b_idx = batch_idx * K * ldb + (k_start + row) * ldb + actual_col;
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
        let global_row = tile_row + c_row;
        let global_col = tile_col + c_col;

        let c_idx = batch_idx * M * params.ldc + global_row * params.ldc + global_col;
        C[c_idx] = params.alpha * tile_C[c_row * TILE_N + c_col];
    }
}

// ============================================================================
// Scatter Output to C
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Compute GEMM and scatter results to sparse locations
    // Used for: sparse_output[indices] = input @ weights

    let tid = lid.x;
    let batch_idx = wid.z;

    if (batch_idx >= params.batch_size) { return; }

    let M = params.M;
    let N = params.N;
    let K = params.K;
    let num_indices = params.num_indices_a;  // Scatter indices

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;
    let num_rows = min(TILE_M, M - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    // Load scatter indices for rows
    for (var i = tid; i < num_rows; i += BLOCK_SIZE) {
        local_indices[i] = indices_a[batch_idx * M + tile_row + i];
    }
    workgroupBarrier();

    // Standard GEMM computation
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        tile_C[i] = 0.0;
    }
    workgroupBarrier();

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, K - k_start);

        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let a_idx = batch_idx * M * params.lda + (tile_row + row) * params.lda + k_start + col;
            tile_A[row * TILE_K + col] = A[a_idx];
        }

        for (var i = tid; i < num_k * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;
            let b_idx = batch_idx * K * params.ldb + (k_start + row) * params.ldb + tile_col + col;
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

    // Scatter results using indices
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let scatter_row = local_indices[c_row];  // Indirect store location
        let global_col = tile_col + c_col;

        let c_idx = batch_idx * num_indices * params.ldc + scatter_row * params.ldc + global_col;
        let result = params.alpha * tile_C[c_row * TILE_N + c_col];

        // Atomic add for scatter (in case of duplicate indices)
        // Note: WGSL doesn't have atomicAdd for f32, so this is simplified
        C[c_idx] = C[c_idx] + result;
    }
}

// ============================================================================
// Embedding GEMM (Token Embedding Lookup + Linear)
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_embedding(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Fused embedding lookup + linear projection
    // output[i] = embed[tokens[i]] @ projection + bias

    let tid = lid.x;
    let batch_idx = wid.z;

    if (batch_idx >= params.batch_size) { return; }

    let seq_len = params.num_indices_a;  // Token sequence length
    let embed_dim = params.embedding_dim;
    let hidden_dim = params.N;  // Projection output dimension

    let tile_row = wid.x * TILE_M;  // Sequence position
    let tile_col = wid.y * TILE_N;  // Hidden dimension
    let num_rows = min(TILE_M, seq_len - tile_row);
    let num_cols = min(TILE_N, hidden_dim - tile_col);

    // Load token indices
    for (var i = tid; i < num_rows; i += BLOCK_SIZE) {
        local_indices[i] = indices_a[batch_idx * seq_len + tile_row + i];
    }
    workgroupBarrier();

    // Initialize
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        tile_C[i] = 0.0;
    }
    workgroupBarrier();

    let num_k_tiles = (embed_dim + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, embed_dim - k_start);

        // Gather embeddings for tokens
        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let token_id = local_indices[row];

            // A is embedding table: [vocab_size, embed_dim]
            let a_idx = token_id * embed_dim + k_start + col;
            tile_A[row * TILE_K + col] = A[a_idx];
        }

        // Load projection weights
        for (var i = tid; i < num_k * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;

            // B is projection: [embed_dim, hidden_dim]
            let b_idx = (k_start + row) * hidden_dim + tile_col + col;
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

    // Store with optional bias (bias in indices_b buffer, repurposed)
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let global_row = tile_row + c_row;
        let global_col = tile_col + c_col;

        var result = tile_C[c_row * TILE_N + c_col];

        // Add bias if present (bias stored as f32 reinterpreted from indices_b)
        if (params.gather_mode & 0x10u) != 0u {
            result += bitcast<f32>(indices_b[global_col]);
        }

        let c_idx = batch_idx * seq_len * hidden_dim + global_row * hidden_dim + global_col;
        C[c_idx] = result;
    }
}
