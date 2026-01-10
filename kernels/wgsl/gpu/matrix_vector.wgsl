// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// General Matrix-Vector Operations
//
// Collection of common matrix-vector operations:
// - Vector addition/subtraction
// - Element-wise operations
// - Dot product
// - Outer product
// - Matrix-vector add (bias)
// - Reduction operations
//
// Part of the Lux Network GPU acceleration library

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// ============================================================================
// Parameter Structures
// ============================================================================

struct VectorParams {
    size: u32,          // Vector size
    stride: u32,        // Element stride
    offset: u32,        // Starting offset
    _pad: u32,
}

struct MatrixVectorParams {
    M: u32,             // Number of rows
    N: u32,             // Number of columns
    lda: u32,           // Leading dimension of matrix
    alpha: f32,         // Scalar multiplier
    beta: f32,          // Scalar for accumulator
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct ReductionParams {
    size: u32,
    reduce_size: u32,   // Size of dimension to reduce
    outer_size: u32,    // Product of outer dimensions
    _pad: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: VectorParams;

// Additional bindings for matrix operations
@group(0) @binding(4) var<storage, read> matrix: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<uniform> mv_params: MatrixVectorParams;

// Workgroup shared memory
var<workgroup> shared_data: array<f32, WORKGROUP_SIZE>;

// ============================================================================
// Vector Addition: c = a + b
// ============================================================================

@compute @workgroup_size(256)
fn vec_add(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let size = params.size;
    if (gid.x >= size) {
        return;
    }

    output[gid.x] = input_a[gid.x] + input_b[gid.x];
}

// ============================================================================
// Vector Subtraction: c = a - b
// ============================================================================

@compute @workgroup_size(256)
fn vec_sub(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let size = params.size;
    if (gid.x >= size) {
        return;
    }

    output[gid.x] = input_a[gid.x] - input_b[gid.x];
}

// ============================================================================
// Vector Multiply (element-wise): c = a * b
// ============================================================================

@compute @workgroup_size(256)
fn vec_mul(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let size = params.size;
    if (gid.x >= size) {
        return;
    }

    output[gid.x] = input_a[gid.x] * input_b[gid.x];
}

// ============================================================================
// Vector Divide (element-wise): c = a / b
// ============================================================================

@compute @workgroup_size(256)
fn vec_div(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let size = params.size;
    if (gid.x >= size) {
        return;
    }

    let b_val = input_b[gid.x];
    output[gid.x] = select(input_a[gid.x] / b_val, 0.0, abs(b_val) < 1e-10);
}

// ============================================================================
// Scalar-Vector Operations
// ============================================================================

struct ScalarParams {
    size: u32,
    scalar: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(1) @binding(0) var<uniform> scalar_params: ScalarParams;

// c = alpha * a
@compute @workgroup_size(256)
fn vec_scale(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let size = scalar_params.size;
    let alpha = scalar_params.scalar;

    if (gid.x >= size) {
        return;
    }

    output[gid.x] = alpha * input_a[gid.x];
}

// c = a + alpha
@compute @workgroup_size(256)
fn vec_add_scalar(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let size = scalar_params.size;
    let alpha = scalar_params.scalar;

    if (gid.x >= size) {
        return;
    }

    output[gid.x] = input_a[gid.x] + alpha;
}

// c = alpha * a + b (AXPY)
@compute @workgroup_size(256)
fn vec_axpy(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let size = scalar_params.size;
    let alpha = scalar_params.scalar;

    if (gid.x >= size) {
        return;
    }

    output[gid.x] = alpha * input_a[gid.x] + input_b[gid.x];
}

// ============================================================================
// Dot Product: sum(a * b)
// ============================================================================

@compute @workgroup_size(256)
fn dot_product(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let size = params.size;

    // Each thread computes partial dot product
    var partial_sum: f32 = 0.0;

    var i = gid.x;
    while (i < size) {
        partial_sum += input_a[i] * input_b[i];
        i += WORKGROUP_SIZE * 256u;  // Total threads
    }

    shared_data[lid.x] = partial_sum;
    workgroupBarrier();

    // Parallel reduction
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_data[lid.x] += shared_data[lid.x + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes result (atomic add for multi-workgroup)
    if (lid.x == 0u) {
        // Simplified: direct write (for single workgroup)
        // For multi-workgroup, would need atomicAdd
        output[wid.x] = shared_data[0];
    }
}

// ============================================================================
// Vector Norms
// ============================================================================

// L2 norm: sqrt(sum(x^2))
@compute @workgroup_size(256)
fn vec_norm_l2(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let size = params.size;

    var partial_sum: f32 = 0.0;

    var i = gid.x;
    while (i < size) {
        let val = input_a[i];
        partial_sum += val * val;
        i += WORKGROUP_SIZE * 256u;
    }

    shared_data[lid.x] = partial_sum;
    workgroupBarrier();

    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_data[lid.x] += shared_data[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        output[wid.x] = sqrt(shared_data[0]);
    }
}

// L1 norm: sum(|x|)
@compute @workgroup_size(256)
fn vec_norm_l1(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let size = params.size;

    var partial_sum: f32 = 0.0;

    var i = gid.x;
    while (i < size) {
        partial_sum += abs(input_a[i]);
        i += WORKGROUP_SIZE * 256u;
    }

    shared_data[lid.x] = partial_sum;
    workgroupBarrier();

    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_data[lid.x] += shared_data[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        output[wid.x] = shared_data[0];
    }
}

// Inf norm: max(|x|)
@compute @workgroup_size(256)
fn vec_norm_inf(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let size = params.size;

    var local_max: f32 = 0.0;

    var i = gid.x;
    while (i < size) {
        local_max = max(local_max, abs(input_a[i]));
        i += WORKGROUP_SIZE * 256u;
    }

    shared_data[lid.x] = local_max;
    workgroupBarrier();

    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_data[lid.x] = max(shared_data[lid.x], shared_data[lid.x + stride]);
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        output[wid.x] = shared_data[0];
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

@group(2) @binding(0) var<uniform> reduce_params: ReductionParams;

// Sum reduction
@compute @workgroup_size(256)
fn reduce_sum(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let outer_idx = wid.x;
    let reduce_size = reduce_params.reduce_size;
    let outer_size = reduce_params.outer_size;

    if (outer_idx >= outer_size) {
        return;
    }

    let base_idx = outer_idx * reduce_size;

    var partial_sum: f32 = 0.0;

    var i = lid.x;
    while (i < reduce_size) {
        partial_sum += input_a[base_idx + i];
        i += WORKGROUP_SIZE;
    }

    shared_data[lid.x] = partial_sum;
    workgroupBarrier();

    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_data[lid.x] += shared_data[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        output[outer_idx] = shared_data[0];
    }
}

// Max reduction
@compute @workgroup_size(256)
fn reduce_max(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let outer_idx = wid.x;
    let reduce_size = reduce_params.reduce_size;
    let outer_size = reduce_params.outer_size;

    if (outer_idx >= outer_size) {
        return;
    }

    let base_idx = outer_idx * reduce_size;

    var local_max: f32 = -3.402823466e+38;

    var i = lid.x;
    while (i < reduce_size) {
        local_max = max(local_max, input_a[base_idx + i]);
        i += WORKGROUP_SIZE;
    }

    shared_data[lid.x] = local_max;
    workgroupBarrier();

    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_data[lid.x] = max(shared_data[lid.x], shared_data[lid.x + stride]);
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        output[outer_idx] = shared_data[0];
    }
}

// Min reduction
@compute @workgroup_size(256)
fn reduce_min(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let outer_idx = wid.x;
    let reduce_size = reduce_params.reduce_size;
    let outer_size = reduce_params.outer_size;

    if (outer_idx >= outer_size) {
        return;
    }

    let base_idx = outer_idx * reduce_size;

    var local_min: f32 = 3.402823466e+38;

    var i = lid.x;
    while (i < reduce_size) {
        local_min = min(local_min, input_a[base_idx + i]);
        i += WORKGROUP_SIZE;
    }

    shared_data[lid.x] = local_min;
    workgroupBarrier();

    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_data[lid.x] = min(shared_data[lid.x], shared_data[lid.x + stride]);
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        output[outer_idx] = shared_data[0];
    }
}

// Mean reduction
@compute @workgroup_size(256)
fn reduce_mean(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let outer_idx = wid.x;
    let reduce_size = reduce_params.reduce_size;
    let outer_size = reduce_params.outer_size;

    if (outer_idx >= outer_size) {
        return;
    }

    let base_idx = outer_idx * reduce_size;

    var partial_sum: f32 = 0.0;

    var i = lid.x;
    while (i < reduce_size) {
        partial_sum += input_a[base_idx + i];
        i += WORKGROUP_SIZE;
    }

    shared_data[lid.x] = partial_sum;
    workgroupBarrier();

    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_data[lid.x] += shared_data[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        output[outer_idx] = shared_data[0] / f32(reduce_size);
    }
}

// ============================================================================
// Outer Product: C = a * b^T (rank-1 update)
// ============================================================================

struct OuterParams {
    M: u32,             // Size of vector a
    N: u32,             // Size of vector b
    ldc: u32,           // Leading dimension of C
    alpha: f32,         // Scalar
}

@group(3) @binding(0) var<uniform> outer_params: OuterParams;

@compute @workgroup_size(256)
fn outer_product(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let M = outer_params.M;
    let N = outer_params.N;
    let ldc = outer_params.ldc;
    let alpha = outer_params.alpha;

    let total = M * N;
    if (gid.x >= total) {
        return;
    }

    let row = gid.x / N;
    let col = gid.x % N;

    let a_val = input_a[row];
    let b_val = input_b[col];
    let c_idx = row * ldc + col;

    output[c_idx] = alpha * a_val * b_val;
}

// ============================================================================
// Matrix Add Bias (broadcast bias vector across rows)
// ============================================================================

@compute @workgroup_size(256)
fn matrix_add_bias(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let M = mv_params.M;
    let N = mv_params.N;
    let lda = mv_params.lda;

    let total = M * N;
    if (gid.x >= total) {
        return;
    }

    let row = gid.x / N;
    let col = gid.x % N;
    let idx = row * lda + col;

    output[idx] = matrix[idx] + bias[col];
}

// ============================================================================
// Fused Linear + Bias + ReLU
// ============================================================================

@compute @workgroup_size(256)
fn linear_bias_relu(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let M = mv_params.M;
    let N = mv_params.N;
    let lda = mv_params.lda;

    let row = wid.x;
    if (row >= M) {
        return;
    }

    // Each thread handles one or more output elements
    var col = lid.x;
    while (col < N) {
        // Matrix-vector multiplication for this element
        var sum: f32 = 0.0;
        for (var k = 0u; k < lda; k++) {
            sum += matrix[row * lda + k] * input_a[k];
        }

        // Add bias and apply ReLU
        let result = sum + bias[col];
        output[row * N + col] = max(0.0, result);

        col += WORKGROUP_SIZE;
    }
}

// ============================================================================
// Element-wise Activation Functions
// ============================================================================

// ReLU: max(0, x)
@compute @workgroup_size(256)
fn vec_relu(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let size = params.size;
    if (gid.x >= size) {
        return;
    }

    output[gid.x] = max(0.0, input_a[gid.x]);
}

// GELU: x * Phi(x) where Phi is the CDF of standard normal
// Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
@compute @workgroup_size(256)
fn vec_gelu(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let size = params.size;
    if (gid.x >= size) {
        return;
    }

    let x = input_a[gid.x];
    let c = 0.7978845608028654;  // sqrt(2/pi)
    let inner = c * (x + 0.044715 * x * x * x);
    output[gid.x] = 0.5 * x * (1.0 + tanh(inner));
}

// SiLU (Swish): x * sigmoid(x)
@compute @workgroup_size(256)
fn vec_silu(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let size = params.size;
    if (gid.x >= size) {
        return;
    }

    let x = input_a[gid.x];
    let sigmoid_x = 1.0 / (1.0 + exp(-x));
    output[gid.x] = x * sigmoid_x;
}

// Sigmoid: 1 / (1 + exp(-x))
@compute @workgroup_size(256)
fn vec_sigmoid(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let size = params.size;
    if (gid.x >= size) {
        return;
    }

    let x = input_a[gid.x];
    output[gid.x] = 1.0 / (1.0 + exp(-x));
}

// Tanh
@compute @workgroup_size(256)
fn vec_tanh(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let size = params.size;
    if (gid.x >= size) {
        return;
    }

    output[gid.x] = tanh(input_a[gid.x]);
}

// ============================================================================
// Copy and Fill Operations
// ============================================================================

// Fill with constant value
@compute @workgroup_size(256)
fn vec_fill(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let size = scalar_params.size;
    let value = scalar_params.scalar;

    if (gid.x >= size) {
        return;
    }

    output[gid.x] = value;
}

// Copy a to output
@compute @workgroup_size(256)
fn vec_copy(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let size = params.size;
    if (gid.x >= size) {
        return;
    }

    output[gid.x] = input_a[gid.x];
}

// ============================================================================
// Clamp and Clip Operations
// ============================================================================

struct ClampParams {
    size: u32,
    min_val: f32,
    max_val: f32,
    _pad: u32,
}

@group(4) @binding(0) var<uniform> clamp_params: ClampParams;

@compute @workgroup_size(256)
fn vec_clamp(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let size = clamp_params.size;
    let min_val = clamp_params.min_val;
    let max_val = clamp_params.max_val;

    if (gid.x >= size) {
        return;
    }

    output[gid.x] = clamp(input_a[gid.x], min_val, max_val);
}
