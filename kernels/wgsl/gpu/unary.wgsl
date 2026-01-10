// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unary Operations Kernels - Element-wise single-input operations
// Includes mathematical functions: exp, log, sin, cos, sqrt, abs, neg, etc.
// Also includes activation functions and type conversions.
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

// ============================================================================
// Constants
// ============================================================================

const PI: f32 = 3.14159265358979323846f;
const E: f32 = 2.71828182845904523536f;
const LN2: f32 = 0.69314718055994530942f;
const LN10: f32 = 2.30258509299404568402f;
const LOG2E: f32 = 1.44269504088896340736f;
const LOG10E: f32 = 0.43429448190325182765f;
const SQRT2: f32 = 1.41421356237309504880f;

// ============================================================================
// Parameter Structure
// ============================================================================

struct UnaryParams {
    size: u32,           // Number of elements
    alpha: f32,          // Optional parameter (for leaky_relu, elu, etc.)
    _pad1: u32,
    _pad2: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: UnaryParams;

// ============================================================================
// Basic Math Operations
// ============================================================================

@compute @workgroup_size(256)
fn neg(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = -input[idx];
}

@compute @workgroup_size(256)
fn abs_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = abs(input[idx]);
}

@compute @workgroup_size(256)
fn sign_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = sign(input[idx]);
}

@compute @workgroup_size(256)
fn sqrt_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = sqrt(input[idx]);
}

@compute @workgroup_size(256)
fn rsqrt_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = inverseSqrt(input[idx]);
}

@compute @workgroup_size(256)
fn square(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];
    output[idx] = x * x;
}

@compute @workgroup_size(256)
fn cube(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];
    output[idx] = x * x * x;
}

@compute @workgroup_size(256)
fn reciprocal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = 1.0f / input[idx];
}

// ============================================================================
// Exponential and Logarithmic Functions
// ============================================================================

@compute @workgroup_size(256)
fn exp_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = exp(input[idx]);
}

@compute @workgroup_size(256)
fn exp2_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = exp2(input[idx]);
}

@compute @workgroup_size(256)
fn expm1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    // expm1(x) = exp(x) - 1, more accurate for small x
    let x = input[idx];
    if (abs(x) < 1e-5f) {
        // Taylor series for small x
        output[idx] = x + 0.5f * x * x;
    } else {
        output[idx] = exp(x) - 1.0f;
    }
}

@compute @workgroup_size(256)
fn log_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = log(input[idx]);
}

@compute @workgroup_size(256)
fn log2_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = log2(input[idx]);
}

@compute @workgroup_size(256)
fn log10_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = log(input[idx]) * LOG10E;
}

@compute @workgroup_size(256)
fn log1p(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    // log1p(x) = log(1 + x), more accurate for small x
    let x = input[idx];
    if (abs(x) < 1e-5f) {
        output[idx] = x - 0.5f * x * x;
    } else {
        output[idx] = log(1.0f + x);
    }
}

// ============================================================================
// Trigonometric Functions
// ============================================================================

@compute @workgroup_size(256)
fn sin_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = sin(input[idx]);
}

@compute @workgroup_size(256)
fn cos_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = cos(input[idx]);
}

@compute @workgroup_size(256)
fn tan_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = tan(input[idx]);
}

@compute @workgroup_size(256)
fn asin_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = asin(input[idx]);
}

@compute @workgroup_size(256)
fn acos_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = acos(input[idx]);
}

@compute @workgroup_size(256)
fn atan_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = atan(input[idx]);
}

// ============================================================================
// Hyperbolic Functions
// ============================================================================

@compute @workgroup_size(256)
fn sinh_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = sinh(input[idx]);
}

@compute @workgroup_size(256)
fn cosh_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = cosh(input[idx]);
}

@compute @workgroup_size(256)
fn tanh_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = tanh(input[idx]);
}

@compute @workgroup_size(256)
fn asinh_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = asinh(input[idx]);
}

@compute @workgroup_size(256)
fn acosh_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = acosh(input[idx]);
}

@compute @workgroup_size(256)
fn atanh_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = atanh(input[idx]);
}

// ============================================================================
// Rounding Functions
// ============================================================================

@compute @workgroup_size(256)
fn floor_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = floor(input[idx]);
}

@compute @workgroup_size(256)
fn ceil_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = ceil(input[idx]);
}

@compute @workgroup_size(256)
fn round_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = round(input[idx]);
}

@compute @workgroup_size(256)
fn trunc_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = trunc(input[idx]);
}

@compute @workgroup_size(256)
fn fract_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = fract(input[idx]);
}

// ============================================================================
// Activation Functions (Deep Learning)
// ============================================================================

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = max(0.0f, input[idx]);
}

@compute @workgroup_size(256)
fn relu6(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = clamp(input[idx], 0.0f, 6.0f);
}

@compute @workgroup_size(256)
fn leaky_relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];
    output[idx] = select(params.alpha * x, x, x >= 0.0f);
}

@compute @workgroup_size(256)
fn elu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];
    output[idx] = select(params.alpha * (exp(x) - 1.0f), x, x >= 0.0f);
}

@compute @workgroup_size(256)
fn selu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];
    // SELU constants
    let alpha = 1.6732632423543772848170429916717f;
    let scale = 1.0507009873554804934193349852946f;
    output[idx] = scale * select(alpha * (exp(x) - 1.0f), x, x >= 0.0f);
}

@compute @workgroup_size(256)
fn gelu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c = 0.7978845608028654f;  // sqrt(2/pi)
    let inner = c * (x + 0.044715f * x * x * x);
    output[idx] = 0.5f * x * (1.0f + tanh(inner));
}

@compute @workgroup_size(256)
fn gelu_fast(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];
    // Fast GELU approximation: x * sigmoid(1.702 * x)
    output[idx] = x / (1.0f + exp(-1.702f * x));
}

@compute @workgroup_size(256)
fn sigmoid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];
    output[idx] = 1.0f / (1.0f + exp(-x));
}

@compute @workgroup_size(256)
fn silu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];
    // SiLU (Swish): x * sigmoid(x)
    output[idx] = x / (1.0f + exp(-x));
}

@compute @workgroup_size(256)
fn softplus(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];
    // softplus(x) = log(1 + exp(x))
    // Use stable computation for large x
    if (x > 20.0f) {
        output[idx] = x;
    } else {
        output[idx] = log(1.0f + exp(x));
    }
}

@compute @workgroup_size(256)
fn softsign(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];
    output[idx] = x / (1.0f + abs(x));
}

@compute @workgroup_size(256)
fn mish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];
    // mish(x) = x * tanh(softplus(x))
    let sp = select(x, log(1.0f + exp(x)), x <= 20.0f);
    output[idx] = x * tanh(sp);
}

@compute @workgroup_size(256)
fn hardswish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];
    // hardswish(x) = x * relu6(x + 3) / 6
    output[idx] = x * clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f;
}

@compute @workgroup_size(256)
fn hardsigmoid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];
    // hardsigmoid(x) = relu6(x + 3) / 6
    output[idx] = clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f;
}

// ============================================================================
// Special Functions
// ============================================================================

@compute @workgroup_size(256)
fn erf_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];

    // Approximation of error function
    let t = 1.0f / (1.0f + 0.5f * abs(x));
    let tau = t * exp(-x * x - 1.26551223f +
              t * (1.00002368f +
              t * (0.37409196f +
              t * (0.09678418f +
              t * (-0.18628806f +
              t * (0.27886807f +
              t * (-1.13520398f +
              t * (1.48851587f +
              t * (-0.82215223f +
              t * 0.17087277f)))))))));

    output[idx] = select(1.0f - tau, tau - 1.0f, x >= 0.0f);
}

@compute @workgroup_size(256)
fn erfc_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];

    let t = 1.0f / (1.0f + 0.5f * abs(x));
    let tau = t * exp(-x * x - 1.26551223f +
              t * (1.00002368f +
              t * (0.37409196f +
              t * (0.09678418f +
              t * (-0.18628806f +
              t * (0.27886807f +
              t * (-1.13520398f +
              t * (1.48851587f +
              t * (-0.82215223f +
              t * 0.17087277f)))))))));

    output[idx] = select(tau, 2.0f - tau, x >= 0.0f);
}

// ============================================================================
// Logical/Comparison (output 0.0 or 1.0)
// ============================================================================

@compute @workgroup_size(256)
fn logical_not(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = select(1.0f, 0.0f, input[idx] != 0.0f);
}

@compute @workgroup_size(256)
fn is_nan(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];
    output[idx] = select(0.0f, 1.0f, x != x);
}

@compute @workgroup_size(256)
fn is_inf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];
    output[idx] = select(0.0f, 1.0f, abs(x) == 3.402823466e+38f);
}

@compute @workgroup_size(256)
fn is_finite(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let x = input[idx];
    output[idx] = select(0.0f, 1.0f, abs(x) < 3.402823466e+38f && x == x);
}

// ============================================================================
// Integer Unary Operations
// ============================================================================

@group(0) @binding(3) var<storage, read> input_i32: array<i32>;
@group(0) @binding(4) var<storage, read_write> output_i32: array<i32>;

@compute @workgroup_size(256)
fn neg_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output_i32[idx] = -input_i32[idx];
}

@compute @workgroup_size(256)
fn abs_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output_i32[idx] = abs(input_i32[idx]);
}

// ============================================================================
// Unsigned Integer / Bitwise Operations
// ============================================================================

@group(0) @binding(5) var<storage, read> input_u32: array<u32>;
@group(0) @binding(6) var<storage, read_write> output_u32: array<u32>;

@compute @workgroup_size(256)
fn bitwise_not(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output_u32[idx] = ~input_u32[idx];
}

@compute @workgroup_size(256)
fn popcount(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output_u32[idx] = countOneBits(input_u32[idx]);
}

@compute @workgroup_size(256)
fn clz(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output_u32[idx] = countLeadingZeros(input_u32[idx]);
}

@compute @workgroup_size(256)
fn ctz(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output_u32[idx] = countTrailingZeros(input_u32[idx]);
}

@compute @workgroup_size(256)
fn reverse_bits_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output_u32[idx] = reverseBits(input_u32[idx]);
}
