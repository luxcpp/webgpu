// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel NAX Variant Fused GEMM with Epilogue in WGSL
// Works on WebGPU (Metal/Vulkan/D3D12 via Dawn/wgpu)
//
// Part of the Lux Network GPU acceleration library
// NAX layout optimizes for nested axis memory access patterns

// ============================================================================
// Configuration Constants
// ============================================================================

const TILE_M: u32 = 64u;       // Output rows per workgroup
const TILE_N: u32 = 64u;       // Output columns per workgroup
const TILE_K: u32 = 16u;       // Reduction tile size
const BLOCK_SIZE: u32 = 256u;
const VECTOR_SIZE: u32 = 4u;   // For vectorized memory access

// ============================================================================
// NAX Layout Constants
// ============================================================================

// NAX (Nested Axis) layout stores matrices in a hierarchical tiled format:
// Instead of [M, K], store as [M/TM, K/TK, TM, TK] for better cache locality
const NAX_TILE_M: u32 = 32u;
const NAX_TILE_K: u32 = 32u;
const NAX_TILE_N: u32 = 32u;

// ============================================================================
// Kernel Bindings
// ============================================================================

struct GemmNAXParams {
    M: u32,                    // Rows of A and C
    N: u32,                    // Columns of B and C
    K: u32,                    // Columns of A, Rows of B
    batch_size: u32,           // Batch dimension
    alpha: f32,                // Scalar for A @ B
    beta: f32,                 // Scalar for C
    epilogue: u32,             // Epilogue type
    swizzle: u32,              // Enable memory access swizzling
    nax_mode: u32,             // NAX layout mode (0=standard, 1=tiled, 2=blocked)
    acc_dtype: u32,            // Accumulation dtype (0=f32, 1=f16)
    num_warps: u32,            // Warps per workgroup
    stages: u32,               // Pipeline stages
}

// NAX Layout: matrices are stored in nested tile format
// A: [batch, M/TM, K/TK, TM, TK] (physically contiguous tiles)
// B: [batch, K/TK, N/TN, TK, TN]
// C: [batch, M/TM, N/TN, TM, TN]
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> residual: array<f32>;
@group(0) @binding(5) var<uniform> params: GemmNAXParams;

// Shared memory
var<workgroup> smem_A: array<f32, 2048>;
var<workgroup> smem_B: array<f32, 2048>;
var<workgroup> smem_C: array<f32, 4096>;

// ============================================================================
// NAX Index Helpers
// ============================================================================

// Standard linear index: [batch, row, col]
fn idx_linear(batch: u32, row: u32, col: u32, rows: u32, cols: u32) -> u32 {
    return (batch * rows + row) * cols + col;
}

// NAX tiled index: [batch, tile_row, tile_col, local_row, local_col]
fn idx_nax(batch: u32, row: u32, col: u32, rows: u32, cols: u32, tile_size: u32) -> u32 {
    let tile_row = row / tile_size;
    let tile_col = col / tile_size;
    let local_row = row % tile_size;
    let local_col = col % tile_size;

    let tiles_per_row = (cols + tile_size - 1u) / tile_size;
    let tile_idx = tile_row * tiles_per_row + tile_col;
    let local_idx = local_row * tile_size + local_col;

    return batch * ((rows / tile_size) * tiles_per_row * tile_size * tile_size)
           + tile_idx * tile_size * tile_size + local_idx;
}

// Swizzled index for bank conflict avoidance
fn swizzle_idx(row: u32, col: u32, ld: u32) -> u32 {
    let bank = col % 32u;
    let swizzled_bank = bank ^ (row % 32u);
    return row * ld + (col - bank) + swizzled_bank;
}

// ============================================================================
// Activation Functions
// ============================================================================

fn gelu(x: f32) -> f32 {
    let c = 0.7978845608;
    let a = 0.044715;
    return 0.5 * x * (1.0 + tanh(c * (x + a * x * x * x)));
}

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

// ============================================================================
// NAX GEMM with Standard Layout Conversion
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_nax(
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
    let nax_mode = params.nax_mode;
    let use_swizzle = params.swizzle == 1u;

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;
    let num_rows = min(TILE_M, M - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    // Initialize accumulator
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        smem_C[i] = 0.0;
    }
    workgroupBarrier();

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, K - k_start);

        // Load A tile with NAX layout consideration
        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let global_row = tile_row + row;
            let global_col = k_start + col;

            var a_val: f32;
            if (nax_mode == 1u) {
                // NAX tiled layout
                let a_idx = idx_nax(batch_idx, global_row, global_col, M, K, NAX_TILE_K);
                a_val = A[a_idx];
            } else {
                // Standard linear layout
                let a_idx = idx_linear(batch_idx, global_row, global_col, M, K);
                a_val = A[a_idx];
            }

            // Store with optional swizzling for bank conflict avoidance
            var smem_idx = row * TILE_K + col;
            if (use_swizzle) {
                smem_idx = swizzle_idx(row, col, TILE_K);
            }
            smem_A[smem_idx] = a_val;
        }

        // Load B tile with NAX layout consideration
        for (var i = tid; i < num_k * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;
            let global_row = k_start + row;
            let global_col = tile_col + col;

            var b_val: f32;
            if (nax_mode == 1u) {
                let b_idx = idx_nax(batch_idx, global_row, global_col, K, N, NAX_TILE_N);
                b_val = B[b_idx];
            } else {
                let b_idx = idx_linear(batch_idx, global_row, global_col, K, N);
                b_val = B[b_idx];
            }

            var smem_idx = row * TILE_N + col;
            if (use_swizzle) {
                smem_idx = swizzle_idx(row, col, TILE_N);
            }
            smem_B[smem_idx] = b_val;
        }
        workgroupBarrier();

        // Compute tile
        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;

            var sum: f32 = 0.0;
            for (var k = 0u; k < num_k; k++) {
                var a_idx = c_row * TILE_K + k;
                var b_idx = k * TILE_N + c_col;

                if (use_swizzle) {
                    a_idx = swizzle_idx(c_row, k, TILE_K);
                    b_idx = swizzle_idx(k, c_col, TILE_N);
                }

                sum += smem_A[a_idx] * smem_B[b_idx];
            }
            smem_C[c_row * TILE_N + c_col] += sum;
        }
        workgroupBarrier();
    }

    // Store result with NAX layout
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let global_row = tile_row + c_row;
        let global_col = tile_col + c_col;

        var result = alpha * smem_C[c_row * TILE_N + c_col];

        // Add bias if present
        if (params.epilogue >= 1u) {
            result += bias[global_col];
        }

        // Apply activation
        switch (params.epilogue) {
            case 2u: { result = gelu(result); }
            case 3u: { result = silu(result); }
            case 4u: { result = tanh(result); }
            case 5u: { result = 1.0 / (1.0 + exp(-result)); }  // sigmoid
            default: { }
        }

        var c_idx: u32;
        if (nax_mode == 1u) {
            c_idx = idx_nax(batch_idx, global_row, global_col, M, N, NAX_TILE_N);
        } else {
            c_idx = idx_linear(batch_idx, global_row, global_col, M, N);
        }

        if (beta != 0.0) {
            result += beta * C[c_idx];
        }

        C[c_idx] = result;
    }
}

// ============================================================================
// NAX GEMM with Double Buffering (Pipelined)
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_nax_pipelined(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Double-buffered GEMM for hiding memory latency
    // Uses alternating buffers for load/compute overlap

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

    // Double buffer indices
    var buffer_idx = 0u;
    let BUFFER_SIZE = 1024u;

    // Initialize accumulator
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        smem_C[i] = 0.0;
    }

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    // Prefetch first tile
    if (num_k_tiles > 0u) {
        for (var i = tid; i < num_rows * TILE_K; i += BLOCK_SIZE) {
            let row = i / TILE_K;
            let col = i % TILE_K;
            let a_idx = idx_linear(batch_idx, tile_row + row, col, M, K);
            smem_A[i] = A[a_idx];
        }
        for (var i = tid; i < TILE_K * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;
            let b_idx = idx_linear(batch_idx, row, tile_col + col, K, N);
            smem_B[i] = B[b_idx];
        }
    }
    workgroupBarrier();

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, K - k_start);

        // Prefetch next tile while computing current
        let next_k_tile = k_tile + 1u;
        let next_buffer = 1u - buffer_idx;

        if (next_k_tile < num_k_tiles) {
            let next_k_start = next_k_tile * TILE_K;
            let next_num_k = min(TILE_K, K - next_k_start);

            for (var i = tid; i < num_rows * next_num_k; i += BLOCK_SIZE) {
                let row = i / next_num_k;
                let col = i % next_num_k;
                let a_idx = idx_linear(batch_idx, tile_row + row, next_k_start + col, M, K);
                smem_A[next_buffer * BUFFER_SIZE + row * TILE_K + col] = A[a_idx];
            }

            for (var i = tid; i < next_num_k * num_cols; i += BLOCK_SIZE) {
                let row = i / num_cols;
                let col = i % num_cols;
                let b_idx = idx_linear(batch_idx, next_k_start + row, tile_col + col, K, N);
                smem_B[next_buffer * BUFFER_SIZE + row * TILE_N + col] = B[b_idx];
            }
        }

        // Compute with current buffer
        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;

            var sum: f32 = 0.0;
            for (var k = 0u; k < num_k; k++) {
                let a_val = smem_A[buffer_idx * BUFFER_SIZE + c_row * TILE_K + k];
                let b_val = smem_B[buffer_idx * BUFFER_SIZE + k * TILE_N + c_col];
                sum += a_val * b_val;
            }
            smem_C[c_row * TILE_N + c_col] += sum;
        }

        buffer_idx = next_buffer;
        workgroupBarrier();
    }

    // Store result
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let global_row = tile_row + c_row;
        let global_col = tile_col + c_col;

        var result = params.alpha * smem_C[c_row * TILE_N + c_col];
        result += bias[global_col];

        let c_idx = idx_linear(batch_idx, global_row, global_col, M, N);
        C[c_idx] = result;
    }
}

// ============================================================================
// NAX GEMM for Grouped Queries (GQA)
// ============================================================================

@compute @workgroup_size(256)
fn steel_gemm_nax_gqa(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // GQA: Multiple query heads share the same key/value heads
    // A is [batch, num_q_heads, seq, head_dim]
    // B is [batch, num_kv_heads, seq, head_dim] (broadcast to match query heads)

    let tid = lid.x;
    let batch_idx = wid.z / params.num_warps;  // Repurpose num_warps as num_q_heads
    let q_head = wid.z % params.num_warps;

    if (batch_idx >= params.batch_size) { return; }

    let M = params.M;  // seq_q
    let N = params.N;  // seq_kv
    let K = params.K;  // head_dim
    let num_q_heads = params.num_warps;
    let num_kv_heads = params.stages;  // Repurpose stages as num_kv_heads

    // Compute KV head index for this query head
    let kv_head = q_head * num_kv_heads / num_q_heads;

    let tile_row = wid.x * TILE_M;
    let tile_col = wid.y * TILE_N;
    let num_rows = min(TILE_M, M - tile_row);
    let num_cols = min(TILE_N, N - tile_col);

    // Initialize
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        smem_C[i] = 0.0;
    }
    workgroupBarrier();

    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, K - k_start);

        // Load Q (from query head)
        for (var i = tid; i < num_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            // A layout: [batch, q_head, seq, head_dim]
            let a_idx = ((batch_idx * num_q_heads + q_head) * M + tile_row + row) * K + k_start + col;
            smem_A[row * TILE_K + col] = A[a_idx];
        }

        // Load K (from KV head, broadcast)
        for (var i = tid; i < num_k * num_cols; i += BLOCK_SIZE) {
            let row = i / num_cols;
            let col = i % num_cols;
            // B layout: [batch, kv_head, seq, head_dim]
            let b_idx = ((batch_idx * num_kv_heads + kv_head) * N + tile_col + col) * K + k_start + row;
            smem_B[row * TILE_N + col] = B[b_idx];
        }
        workgroupBarrier();

        // Compute Q @ K^T
        for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
            let c_row = i / num_cols;
            let c_col = i % num_cols;

            var sum: f32 = 0.0;
            for (var k = 0u; k < num_k; k++) {
                sum += smem_A[c_row * TILE_K + k] * smem_B[k * TILE_N + c_col];
            }
            smem_C[c_row * TILE_N + c_col] += sum;
        }
        workgroupBarrier();
    }

    // Store (attention scores before softmax)
    for (var i = tid; i < num_rows * num_cols; i += BLOCK_SIZE) {
        let c_row = i / num_cols;
        let c_col = i % num_cols;
        let global_row = tile_row + c_row;
        let global_col = tile_col + c_col;

        let result = smem_C[c_row * TILE_N + c_col] * params.alpha;  // alpha = 1/sqrt(d)
        let c_idx = ((batch_idx * num_q_heads + q_head) * M + global_row) * N + global_col;
        C[c_idx] = result;
    }
}
