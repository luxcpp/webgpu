// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Fused GEMM with Epilogue in WGSL
// Works on WebGPU (Metal/Vulkan/D3D12 via Dawn/wgpu)
//
// Part of the Lux Network GPU acceleration library
// Implements tiled matrix multiplication with fused activation/bias

// ============================================================================
// Configuration Constants
// ============================================================================

const TILE_M: u32 = 64u;       // Output rows per workgroup
const TILE_N: u32 = 64u;       // Output columns per workgroup
const TILE_K: u32 = 16u;       // Reduction tile size
const WARP_M: u32 = 16u;       // Warp tile rows
const WARP_N: u32 = 16u;       // Warp tile columns
const BLOCK_SIZE: u32 = 256u;

// ============================================================================
// Epilogue Operations
// ============================================================================

const EPILOGUE_NONE: u32 = 0u;
const EPILOGUE_RELU: u32 = 1u;
const EPILOGUE_GELU: u32 = 2u;
const EPILOGUE_SILU: u32 = 3u;
const EPILOGUE_TANH: u32 = 4u;
const EPILOGUE_SIGMOID: u32 = 5u;
const EPILOGUE_ADD_BIAS: u32 = 6u;
const EPILOGUE_ADD_RESIDUAL: u32 = 7u;
const EPILOGUE_LAYER_NORM: u32 = 8u;
const EPILOGUE_RMS_NORM: u32 = 9u;

// ============================================================================
// Kernel Bindings
// ============================================================================

struct GemmFusedParams {
    M: u32,                    // Rows of A and C
    N: u32,                    // Columns of B and C
    K: u32,                    // Columns of A, Rows of B
    lda: u32,                  // Leading dimension of A
    ldb: u32,                  // Leading dimension of B
    ldc: u32,                  // Leading dimension of C
    alpha: f32,                // Scalar multiplier for A @ B
    beta: f32,                 // Scalar multiplier for C
    epilogue: u32,             // Epilogue operation
    batch_size: u32,           // Batch dimension
    stride_a: u32,             // Batch stride for A
    stride_b: u32,             // Batch stride for B
    stride_c: u32,             // Batch stride for C
    transpose_a: u32,          // Transpose A
    transpose_b: u32,          // Transpose B
    _pad: u32,
}

// A: [batch, M, K] or [batch, K, M] if transposed
// B: [batch, K, N] or [batch, N, K] if transposed
// C: [batch, M, N]
// bias: [N] (broadcasted)
// residual: [batch, M, N]
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> residual: array<f32>;
@group(0) @binding(5) var<uniform> params: GemmFusedParams;

// Shared memory tiles
var<workgroup> tile_A: array<f32, 1024>;  // TILE_M * TILE_K
var<workgroup> tile_B: array<f32, 1024>;  // TILE_K * TILE_N
var<workgroup> tile_C: array<f32, 4096>;  // TILE_M * TILE_N

// ============================================================================
// Activation Functions
// ============================================================================

fn relu(x: f32) -> f32 {
    return max(0.0, x);
}

fn gelu(x: f32) -> f32 {
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c = 0.7978845608;  // sqrt(2/pi)
    let a = 0.044715;
    return 0.5 * x * (1.0 + tanh(c * (x + a * x * x * x)));
}

fn silu(x: f32) -> f32 {
    // SiLU (Swish): x * sigmoid(x)
    return x / (1.0 + exp(-x));
}

fn apply_epilogue(x: f32, epilogue: u32, bias_val: f32, residual_val: f32) -> f32 {
    var result = x;

    switch (epilogue) {
        case 1u: {  // RELU
            result = relu(result);
        }
        case 2u: {  // GELU
            result = gelu(result);
        }
        case 3u: {  // SILU
            result = silu(result);
        }
        case 4u: {  // TANH
            result = tanh(result);
        }
        case 5u: {  // SIGMOID
            result = 1.0 / (1.0 + exp(-result));
        }
        case 6u: {  // ADD_BIAS
            result = result + bias_val;
        }
        case 7u: {  // ADD_RESIDUAL
            result = result + residual_val;
        }
        default: {
            // NONE or unsupported
        }
    }

    return result;
}

// ============================================================================
// Tiled GEMM with Fused Epilogue
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_fused(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z;

    if (batch_idx >= params.batch_size) { return; }

    let M = params.M;
    let N = params.N;
    let K = params.K;
    let alpha = params.alpha;
    let beta = params.beta;
    let epilogue = params.epilogue;
    let transpose_a = params.transpose_a == 1u;
    let transpose_b = params.transpose_b == 1u;

    // Tile coordinates
    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;

    let num_rows = min(TILE_M, M - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    // Initialize accumulator tile
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        tile_C[i] = 0.0;
    }
    workgroupBarrier();

    // Number of K tiles
    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    // Iterate over K dimension
    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, K - k_start);

        // Load A tile
        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let a_row = tile_row + row;
            let a_col = k_start + col;

            var a_idx: u32;
            if (transpose_a) {
                a_idx = batch_idx * params.stride_a + a_col * params.lda + a_row;
            } else {
                a_idx = batch_idx * params.stride_a + a_row * params.lda + a_col;
            }

            tile_A[row * TILE_K + col] = A[a_idx];
        }

        // Load B tile
        for (var i = tid; i < num_k * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;
            let b_row = k_start + row;
            let b_col = tile_col + col;

            var b_idx: u32;
            if (transpose_b) {
                b_idx = batch_idx * params.stride_b + b_col * params.ldb + b_row;
            } else {
                b_idx = batch_idx * params.stride_b + b_row * params.ldb + b_col;
            }

            tile_B[row * TILE_N + col] = B[b_idx];
        }
        workgroupBarrier();

        // Compute C += A @ B for this tile
        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;

            var sum: f32 = 0.0;
            for (var k = 0u; k < num_k; k++) {
                let a_val = tile_A[c_row * TILE_K + k];
                let b_val = tile_B[k * TILE_N + c_col];
                sum += a_val * b_val;
            }

            tile_C[c_row * TILE_N + c_col] += sum;
        }
        workgroupBarrier();
    }

    // Apply alpha, beta, and epilogue, then store
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let global_row = tile_row + c_row;
        let global_col = tile_col + c_col;

        let c_idx = batch_idx * params.stride_c + global_row * params.ldc + global_col;

        // Compute result
        var result = alpha * tile_C[c_row * TILE_N + c_col];

        // Add beta * C if beta != 0
        if (beta != 0.0) {
            result += beta * C[c_idx];
        }

        // Get bias and residual values
        let bias_val = bias[global_col];
        let residual_val = residual[c_idx];

        // Apply epilogue
        result = apply_epilogue(result, epilogue, bias_val, residual_val);

        C[c_idx] = result;
    }
}

// ============================================================================
// Fused GEMM + GELU (Common for Transformers)
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_gelu(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z;

    if (batch_idx >= params.batch_size) { return; }

    let M = params.M;
    let N = params.N;
    let K = params.K;

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;
    let num_rows = min(TILE_M, M - tile_row);
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

        // Load tiles
        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let a_idx = batch_idx * params.stride_a + (tile_row + row) * params.lda + k_start + col;
            tile_A[row * TILE_K + col] = A[a_idx];
        }

        for (var i = tid; i < num_k * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;
            let b_idx = batch_idx * params.stride_b + (k_start + row) * params.ldb + tile_col + col;
            tile_B[row * TILE_N + col] = B[b_idx];
        }
        workgroupBarrier();

        // Accumulate
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

    // Apply GELU and store
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let global_row = tile_row + c_row;
        let global_col = tile_col + c_col;

        let c_idx = batch_idx * params.stride_c + global_row * params.ldc + global_col;

        var result = tile_C[c_row * TILE_N + c_col] + bias[global_col];
        result = gelu(result);

        C[c_idx] = result;
    }
}

// ============================================================================
// Fused GEMM + Add + LayerNorm
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_add_layernorm(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin_local_invocation_id lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // This variant computes: LayerNorm(residual + GEMM(A, B))
    // Useful for transformer residual connections

    let tid = lid.x;
    let batch_idx = wid.z;

    if (batch_idx >= params.batch_size) { return; }

    let M = params.M;
    let N = params.N;
    let K = params.K;

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;
    let num_rows = min(TILE_M, M - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    // Standard GEMM accumulation
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
            let a_idx = batch_idx * params.stride_a + (tile_row + row) * params.lda + k_start + col;
            tile_A[row * TILE_K + col] = A[a_idx];
        }

        for (var i = tid; i < num_k * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;
            let b_idx = batch_idx * params.stride_b + (k_start + row) * params.ldb + tile_col + col;
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

    // Add residual
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let global_row = tile_row + c_row;
        let global_col = tile_col + c_col;
        let idx = batch_idx * params.stride_c + global_row * params.ldc + global_col;

        tile_C[c_row * TILE_N + c_col] += residual[idx];
    }
    workgroupBarrier();

    // LayerNorm per row (requires full row in tile - simplified version)
    // Full impl would need reduction across tiles
    for (var row = 0u; row < num_rows; row++) {
        // Compute mean
        var sum: f32 = 0.0;
        for (var col = tid; col < num_cols; col += BLOCK_SIZE) {
            sum += tile_C[row * TILE_N + col];
        }
        // Note: This is simplified - proper reduction would use shared memory
        let mean = sum / f32(N);

        // Compute variance
        var var_sum: f32 = 0.0;
        for (var col = tid; col < num_cols; col += BLOCK_SIZE) {
            let diff = tile_C[row * TILE_N + col] - mean;
            var_sum += diff * diff;
        }
        let variance = var_sum / f32(N);
        let inv_std = 1.0 / sqrt(variance + 1e-5);

        // Normalize and store
        for (var col = tid; col < num_cols; col += BLOCK_SIZE) {
            let global_row = tile_row + row;
            let global_col = tile_col + col;
            let idx = batch_idx * params.stride_c + global_row * params.ldc + global_col;

            let normalized = (tile_C[row * TILE_N + col] - mean) * inv_std;
            C[idx] = normalized;  // * gamma[global_col] + beta[global_col] would be added here
        }
        workgroupBarrier();
    }
}

// ============================================================================
// Fused GEMM + SiLU (for Llama-style FFN)
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_silu_mul(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Llama FFN: silu(x @ W1) * (x @ W2)
    // This kernel computes: silu(A @ B) * (A @ B2)
    // where B and B2 are the gate and up projection matrices

    let tid = lid.x;
    let batch_idx = wid.z;

    if (batch_idx >= params.batch_size) { return; }

    let M = params.M;
    let N = params.N;
    let K = params.K;

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;
    let num_rows = min(TILE_M, M - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    // We need two accumulator tiles: one for gate, one for up
    var acc_gate: array<f32, 256>;  // Simplified size per thread
    var acc_up: array<f32, 256>;

    for (var i = 0u; i < 256u; i++) {
        acc_gate[i] = 0.0;
        acc_up[i] = 0.0;
    }

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, K - k_start);

        // Load A tile
        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let a_idx = batch_idx * params.stride_a + (tile_row + row) * params.lda + k_start + col;
            tile_A[row * TILE_K + col] = A[a_idx];
        }

        // Load B (gate) tile
        for (var i = tid; i < num_k * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;
            let b_idx = batch_idx * params.stride_b + (k_start + row) * params.ldb + tile_col + col;
            tile_B[row * TILE_N + col] = B[b_idx];
        }
        workgroupBarrier();

        // Accumulate for gate projection
        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;

            var sum: f32 = 0.0;
            for (var k = 0u; k < num_k; k++) {
                sum += tile_A[c_row * TILE_K + k] * tile_B[k * TILE_N + c_col];
            }
            acc_gate[i % 256u] += sum;
        }
        workgroupBarrier();

        // Load B2 (up) tile from residual buffer (repurposed)
        for (var i = tid; i < num_k * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;
            let b2_idx = batch_idx * params.stride_b + (k_start + row) * params.ldb + tile_col + col + N;  // Offset for B2
            tile_B[row * TILE_N + col] = B[b2_idx];
        }
        workgroupBarrier();

        // Accumulate for up projection
        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;

            var sum: f32 = 0.0;
            for (var k = 0u; k < num_k; k++) {
                sum += tile_A[c_row * TILE_K + k] * tile_B[k * TILE_N + c_col];
            }
            acc_up[i % 256u] += sum;
        }
        workgroupBarrier();
    }

    // Apply silu(gate) * up and store
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let global_row = tile_row + c_row;
        let global_col = tile_col + c_col;
        let c_idx = batch_idx * params.stride_c + global_row * params.ldc + global_col;

        let gate = acc_gate[i % 256u];
        let up = acc_up[i % 256u];
        let result = silu(gate) * up;

        C[c_idx] = result;
    }
}
