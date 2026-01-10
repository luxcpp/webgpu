// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// WebGPU WGSL Tensor Operations
// Optimized GPU compute shaders for ML tensor operations

// =============================================================================
// Elementwise Binary Operations
// =============================================================================

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Params {
    n: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn add_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.n) {
        out[idx] = a[idx] + b[idx];
    }
}

@compute @workgroup_size(256)
fn sub_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.n) {
        out[idx] = a[idx] - b[idx];
    }
}

@compute @workgroup_size(256)
fn mul_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.n) {
        out[idx] = a[idx] * b[idx];
    }
}

@compute @workgroup_size(256)
fn div_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.n) {
        out[idx] = a[idx] / b[idx];
    }
}

// =============================================================================
// Unary Operations (separate bindings for input/output)
// =============================================================================

@group(0) @binding(0) var<storage, read> unary_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> unary_out: array<f32>;
@group(0) @binding(2) var<uniform> unary_params: Params;

@compute @workgroup_size(256)
fn exp_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.n) {
        unary_out[idx] = exp(unary_in[idx]);
    }
}

@compute @workgroup_size(256)
fn log_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.n) {
        unary_out[idx] = log(unary_in[idx]);
    }
}

@compute @workgroup_size(256)
fn sqrt_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.n) {
        unary_out[idx] = sqrt(unary_in[idx]);
    }
}

@compute @workgroup_size(256)
fn neg_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.n) {
        unary_out[idx] = -unary_in[idx];
    }
}

@compute @workgroup_size(256)
fn abs_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.n) {
        unary_out[idx] = abs(unary_in[idx]);
    }
}

@compute @workgroup_size(256)
fn tanh_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.n) {
        unary_out[idx] = tanh(unary_in[idx]);
    }
}

@compute @workgroup_size(256)
fn sigmoid_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.n) {
        unary_out[idx] = 1.0 / (1.0 + exp(-unary_in[idx]));
    }
}

@compute @workgroup_size(256)
fn relu_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.n) {
        unary_out[idx] = max(0.0, unary_in[idx]);
    }
}

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
@compute @workgroup_size(256)
fn gelu_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.n) {
        let x = unary_in[idx];
        let x3 = x * x * x;
        // sqrt(2/pi) = 0.7978845608
        let inner = 0.7978845608 * (x + 0.044715 * x3);
        unary_out[idx] = 0.5 * x * (1.0 + tanh(inner));
    }
}

@compute @workgroup_size(256)
fn copy_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.n) {
        unary_out[idx] = unary_in[idx];
    }
}

// =============================================================================
// Matrix Multiplication (Tiled)
// =============================================================================

const TILE_SIZE: u32 = 16u;

struct MatmulParams {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(0) var<storage, read> matmul_A: array<f32>;
@group(0) @binding(1) var<storage, read> matmul_B: array<f32>;
@group(0) @binding(2) var<storage, read_write> matmul_C: array<f32>;
@group(0) @binding(3) var<uniform> matmul_params: MatmulParams;

var<workgroup> As: array<array<f32, 16>, 16>;
var<workgroup> Bs: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn matmul_tiled_f32(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let M = matmul_params.M;
    let K = matmul_params.K;
    let N = matmul_params.N;

    let row = wg_id.y * TILE_SIZE + lid.y;
    let col = wg_id.x * TILE_SIZE + lid.x;

    var sum: f32 = 0.0;

    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of A
        let a_col = t * TILE_SIZE + lid.x;
        if (row < M && a_col < K) {
            As[lid.y][lid.x] = matmul_A[row * K + a_col];
        } else {
            As[lid.y][lid.x] = 0.0;
        }

        // Load tile of B
        let b_row = t * TILE_SIZE + lid.y;
        if (b_row < K && col < N) {
            Bs[lid.y][lid.x] = matmul_B[b_row * N + col];
        } else {
            Bs[lid.y][lid.x] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product
        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + As[lid.y][k] * Bs[k][lid.x];
        }

        workgroupBarrier();
    }

    // Write result
    if (row < M && col < N) {
        matmul_C[row * N + col] = sum;
    }
}

// =============================================================================
// Matrix Transpose
// =============================================================================

struct TransposeParams {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> transpose_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> transpose_out: array<f32>;
@group(0) @binding(2) var<uniform> transpose_params: TransposeParams;

var<workgroup> tile: array<array<f32, 17>, 16>; // +1 to avoid bank conflicts

@compute @workgroup_size(16, 16)
fn transpose_f32(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let rows = transpose_params.rows;
    let cols = transpose_params.cols;

    var x = wg_id.x * TILE_SIZE + lid.x;
    var y = wg_id.y * TILE_SIZE + lid.y;

    // Read into shared memory (coalesced read)
    if (x < cols && y < rows) {
        tile[lid.y][lid.x] = transpose_in[y * cols + x];
    }

    workgroupBarrier();

    // Transposed coordinates
    x = wg_id.y * TILE_SIZE + lid.x;
    y = wg_id.x * TILE_SIZE + lid.y;

    // Write from shared memory (coalesced write)
    if (x < rows && y < cols) {
        transpose_out[y * rows + x] = tile[lid.x][lid.y];
    }
}

// =============================================================================
// Reduction Operations
// =============================================================================

const BLOCK_SIZE: u32 = 256u;

struct ReduceParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read> reduce_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> reduce_out: atomic<u32>; // Using atomic for reductions
@group(0) @binding(2) var<uniform> reduce_params: ReduceParams;

var<workgroup> reduce_shared: array<f32, 256>;

// Helper: atomicAdd for f32 (using bit casting)
fn atomicAddF32(ptr: ptr<storage, atomic<u32>, read_write>, value: f32) {
    var old_val = atomicLoad(ptr);
    loop {
        let new_val = bitcast<u32>(bitcast<f32>(old_val) + value);
        let result = atomicCompareExchangeWeak(ptr, old_val, new_val);
        if (result.exchanged) {
            break;
        }
        old_val = result.old_value;
    }
}

@compute @workgroup_size(256)
fn reduce_sum_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let n = reduce_params.n;
    let idx = gid.x;
    let tid = lid.x;

    // Grid-stride loop
    var sum: f32 = 0.0;
    var i = idx;
    while (i < n) {
        sum = sum + reduce_in[i];
        i = i + BLOCK_SIZE * 256u;  // Stride by grid size
    }

    reduce_shared[tid] = sum;
    workgroupBarrier();

    // Tree reduction in shared memory
    for (var s: u32 = BLOCK_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            reduce_shared[tid] = reduce_shared[tid] + reduce_shared[tid + s];
        }
        workgroupBarrier();
    }

    // First thread adds to global result
    if (tid == 0u) {
        atomicAddF32(&reduce_out, reduce_shared[0]);
    }
}

// =============================================================================
// Axis Reduction (Last Axis)
// =============================================================================

struct AxisReduceParams {
    outer_size: u32,
    inner_size: u32,
}

@group(0) @binding(0) var<storage, read> axis_reduce_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> axis_reduce_out: array<f32>;
@group(0) @binding(2) var<uniform> axis_reduce_params: AxisReduceParams;

var<workgroup> axis_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce_sum_axis_f32(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let outer_idx = wg_id.x;
    let tid = lid.x;
    let inner_size = axis_reduce_params.inner_size;

    if (outer_idx >= axis_reduce_params.outer_size) {
        return;
    }

    let row_start = outer_idx * inner_size;

    // Accumulate
    var sum: f32 = 0.0;
    var i = tid;
    while (i < inner_size) {
        sum = sum + axis_reduce_in[row_start + i];
        i = i + BLOCK_SIZE;
    }

    axis_shared[tid] = sum;
    workgroupBarrier();

    // Tree reduction
    for (var s: u32 = BLOCK_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            axis_shared[tid] = axis_shared[tid] + axis_shared[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        axis_reduce_out[outer_idx] = axis_shared[0];
    }
}

@compute @workgroup_size(256)
fn reduce_max_axis_f32(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let outer_idx = wg_id.x;
    let tid = lid.x;
    let inner_size = axis_reduce_params.inner_size;

    if (outer_idx >= axis_reduce_params.outer_size) {
        return;
    }

    let row_start = outer_idx * inner_size;

    var local_max: f32 = -1e38;  // Approximate -infinity
    var i = tid;
    while (i < inner_size) {
        local_max = max(local_max, axis_reduce_in[row_start + i]);
        i = i + BLOCK_SIZE;
    }

    axis_shared[tid] = local_max;
    workgroupBarrier();

    for (var s: u32 = BLOCK_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            axis_shared[tid] = max(axis_shared[tid], axis_shared[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        axis_reduce_out[outer_idx] = axis_shared[0];
    }
}

@compute @workgroup_size(256)
fn reduce_mean_axis_f32(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let outer_idx = wg_id.x;
    let tid = lid.x;
    let inner_size = axis_reduce_params.inner_size;

    if (outer_idx >= axis_reduce_params.outer_size) {
        return;
    }

    let row_start = outer_idx * inner_size;

    var sum: f32 = 0.0;
    var i = tid;
    while (i < inner_size) {
        sum = sum + axis_reduce_in[row_start + i];
        i = i + BLOCK_SIZE;
    }

    axis_shared[tid] = sum;
    workgroupBarrier();

    for (var s: u32 = BLOCK_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            axis_shared[tid] = axis_shared[tid] + axis_shared[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        axis_reduce_out[outer_idx] = axis_shared[0] / f32(inner_size);
    }
}

// =============================================================================
// Softmax
// =============================================================================

struct SoftmaxParams {
    batch_size: u32,
    dim: u32,
}

@group(0) @binding(0) var<storage, read> softmax_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> softmax_out: array<f32>;
@group(0) @binding(2) var<uniform> softmax_params: SoftmaxParams;

var<workgroup> softmax_shared: array<f32, 256>;
var<workgroup> s_max: f32;
var<workgroup> s_sum: f32;

@compute @workgroup_size(256)
fn softmax_f32(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let batch_idx = wg_id.x;
    let tid = lid.x;
    let dim = softmax_params.dim;

    if (batch_idx >= softmax_params.batch_size) {
        return;
    }

    let row_start = batch_idx * dim;

    // Pass 1: Find max
    var local_max: f32 = -1e38;
    var i = tid;
    while (i < dim) {
        local_max = max(local_max, softmax_in[row_start + i]);
        i = i + BLOCK_SIZE;
    }

    softmax_shared[tid] = local_max;
    workgroupBarrier();

    for (var s: u32 = BLOCK_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            softmax_shared[tid] = max(softmax_shared[tid], softmax_shared[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        s_max = softmax_shared[0];
    }
    workgroupBarrier();

    let max_val = s_max;

    // Pass 2: Compute exp(x - max) and sum
    var local_sum: f32 = 0.0;
    i = tid;
    while (i < dim) {
        let exp_val = exp(softmax_in[row_start + i] - max_val);
        softmax_out[row_start + i] = exp_val;
        local_sum = local_sum + exp_val;
        i = i + BLOCK_SIZE;
    }

    softmax_shared[tid] = local_sum;
    workgroupBarrier();

    for (var s: u32 = BLOCK_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            softmax_shared[tid] = softmax_shared[tid] + softmax_shared[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        s_sum = softmax_shared[0];
    }
    workgroupBarrier();

    let inv_sum = 1.0 / s_sum;

    // Normalize
    i = tid;
    while (i < dim) {
        softmax_out[row_start + i] = softmax_out[row_start + i] * inv_sum;
        i = i + BLOCK_SIZE;
    }
}

@compute @workgroup_size(256)
fn log_softmax_f32(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let batch_idx = wg_id.x;
    let tid = lid.x;
    let dim = softmax_params.dim;

    if (batch_idx >= softmax_params.batch_size) {
        return;
    }

    let row_start = batch_idx * dim;

    // Find max
    var local_max: f32 = -1e38;
    var i = tid;
    while (i < dim) {
        local_max = max(local_max, softmax_in[row_start + i]);
        i = i + BLOCK_SIZE;
    }

    softmax_shared[tid] = local_max;
    workgroupBarrier();

    for (var s: u32 = BLOCK_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            softmax_shared[tid] = max(softmax_shared[tid], softmax_shared[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        s_max = softmax_shared[0];
    }
    workgroupBarrier();

    let max_val = s_max;

    // Compute sum(exp(x - max))
    var local_sum: f32 = 0.0;
    i = tid;
    while (i < dim) {
        local_sum = local_sum + exp(softmax_in[row_start + i] - max_val);
        i = i + BLOCK_SIZE;
    }

    softmax_shared[tid] = local_sum;
    workgroupBarrier();

    for (var s: u32 = BLOCK_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            softmax_shared[tid] = softmax_shared[tid] + softmax_shared[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        s_sum = log(softmax_shared[0]);
    }
    workgroupBarrier();

    let log_sum = s_sum;

    // Compute log softmax
    i = tid;
    while (i < dim) {
        softmax_out[row_start + i] = softmax_in[row_start + i] - max_val - log_sum;
        i = i + BLOCK_SIZE;
    }
}

// =============================================================================
// Layer Normalization
// =============================================================================

struct LayerNormParams {
    batch_size: u32,
    dim: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read> ln_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> ln_out: array<f32>;
@group(0) @binding(2) var<storage, read> ln_gamma: array<f32>;
@group(0) @binding(3) var<storage, read> ln_beta: array<f32>;
@group(0) @binding(4) var<uniform> ln_params: LayerNormParams;

var<workgroup> ln_shared: array<f32, 256>;
var<workgroup> ln_mean: f32;
var<workgroup> ln_var: f32;

@compute @workgroup_size(256)
fn layer_norm_f32(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let batch_idx = wg_id.x;
    let tid = lid.x;
    let dim = ln_params.dim;

    if (batch_idx >= ln_params.batch_size) {
        return;
    }

    let row_start = batch_idx * dim;

    // Compute mean
    var local_sum: f32 = 0.0;
    var i = tid;
    while (i < dim) {
        local_sum = local_sum + ln_in[row_start + i];
        i = i + BLOCK_SIZE;
    }

    ln_shared[tid] = local_sum;
    workgroupBarrier();

    for (var s: u32 = BLOCK_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            ln_shared[tid] = ln_shared[tid] + ln_shared[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        ln_mean = ln_shared[0] / f32(dim);
    }
    workgroupBarrier();

    let mean = ln_mean;

    // Compute variance
    var local_var: f32 = 0.0;
    i = tid;
    while (i < dim) {
        let diff = ln_in[row_start + i] - mean;
        local_var = local_var + diff * diff;
        i = i + BLOCK_SIZE;
    }

    ln_shared[tid] = local_var;
    workgroupBarrier();

    for (var s: u32 = BLOCK_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            ln_shared[tid] = ln_shared[tid] + ln_shared[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        ln_var = ln_shared[0] / f32(dim);
    }
    workgroupBarrier();

    let inv_std = inverseSqrt(ln_var + ln_params.eps);

    // Normalize and scale
    i = tid;
    while (i < dim) {
        let normalized = (ln_in[row_start + i] - mean) * inv_std;
        ln_out[row_start + i] = normalized * ln_gamma[i] + ln_beta[i];
        i = i + BLOCK_SIZE;
    }
}

// =============================================================================
// RMS Normalization
// =============================================================================

struct RMSNormParams {
    batch_size: u32,
    dim: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read> rms_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> rms_out: array<f32>;
@group(0) @binding(2) var<storage, read> rms_weight: array<f32>;
@group(0) @binding(3) var<uniform> rms_params: RMSNormParams;

var<workgroup> rms_shared: array<f32, 256>;
var<workgroup> rms_scale: f32;

@compute @workgroup_size(256)
fn rms_norm_f32(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let batch_idx = wg_id.x;
    let tid = lid.x;
    let dim = rms_params.dim;

    if (batch_idx >= rms_params.batch_size) {
        return;
    }

    let row_start = batch_idx * dim;

    // Compute sum of squares
    var local_sum_sq: f32 = 0.0;
    var i = tid;
    while (i < dim) {
        let val = rms_in[row_start + i];
        local_sum_sq = local_sum_sq + val * val;
        i = i + BLOCK_SIZE;
    }

    rms_shared[tid] = local_sum_sq;
    workgroupBarrier();

    for (var s: u32 = BLOCK_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            rms_shared[tid] = rms_shared[tid] + rms_shared[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        rms_scale = inverseSqrt(rms_shared[0] / f32(dim) + rms_params.eps);
    }
    workgroupBarrier();

    let scale = rms_scale;

    // Scale
    i = tid;
    while (i < dim) {
        rms_out[row_start + i] = rms_in[row_start + i] * scale * rms_weight[i];
        i = i + BLOCK_SIZE;
    }
}
