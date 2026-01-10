// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Binary Two-Output Kernels - Operations that produce two outputs
// Includes divmod, modf, frexp, sincos, and similar operations.
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

// ============================================================================
// Parameter Structure
// ============================================================================

struct BinaryTwoParams {
    size: u32,           // Total number of output elements
    a_stride: u32,       // Stride for input A (0 for broadcast)
    b_stride: u32,       // Stride for input B (0 for broadcast)
    _pad: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_1: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_2: array<f32>;
@group(0) @binding(4) var<uniform> params: BinaryTwoParams;

// ============================================================================
// Helper: Calculate input index with broadcasting
// ============================================================================

fn get_a_idx(idx: u32) -> u32 {
    if (params.a_stride == 0u) {
        return 0u;
    }
    return idx * params.a_stride;
}

fn get_b_idx(idx: u32) -> u32 {
    if (params.b_stride == 0u) {
        return 0u;
    }
    return idx * params.b_stride;
}

// ============================================================================
// Divmod - Returns quotient and remainder
// output_1 = a / b (integer division)
// output_2 = a % b (remainder)
// ============================================================================

@compute @workgroup_size(256)
fn divmod(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];

    let quotient = floor(a / b);
    let remainder = a - quotient * b;

    output_1[idx] = quotient;
    output_2[idx] = remainder;
}

// ============================================================================
// Modf - Split into integer and fractional parts
// output_1 = integer part
// output_2 = fractional part
// ============================================================================

@compute @workgroup_size(256)
fn modf_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let x = input_a[get_a_idx(idx)];

    // modf: split into integer and fractional parts
    let int_part = trunc(x);
    let frac_part = x - int_part;

    output_1[idx] = int_part;
    output_2[idx] = frac_part;
}

// ============================================================================
// Frexp - Extract mantissa and exponent
// output_1 = mantissa (in range [0.5, 1.0) or 0)
// output_2 = exponent (as float)
// x = mantissa * 2^exponent
// ============================================================================

@compute @workgroup_size(256)
fn frexp_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let x = input_a[get_a_idx(idx)];

    // Handle special cases
    if (x == 0.0f) {
        output_1[idx] = 0.0f;
        output_2[idx] = 0.0f;
        return;
    }

    let ax = abs(x);
    var exp_val = floor(log2(ax)) + 1.0f;
    var mantissa = ax / pow(2.0f, exp_val);

    // Normalize mantissa to [0.5, 1.0)
    if (mantissa >= 1.0f) {
        mantissa = mantissa * 0.5f;
        exp_val = exp_val + 1.0f;
    } else if (mantissa < 0.5f) {
        mantissa = mantissa * 2.0f;
        exp_val = exp_val - 1.0f;
    }

    // Apply original sign
    if (x < 0.0f) {
        mantissa = -mantissa;
    }

    output_1[idx] = mantissa;
    output_2[idx] = exp_val;
}

// ============================================================================
// Sincos - Compute sine and cosine simultaneously
// output_1 = sin(x)
// output_2 = cos(x)
// ============================================================================

@compute @workgroup_size(256)
fn sincos(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let x = input_a[get_a_idx(idx)];

    output_1[idx] = sin(x);
    output_2[idx] = cos(x);
}

// ============================================================================
// Polar - Convert polar coordinates to Cartesian
// input_a = radius (r)
// input_b = angle (theta)
// output_1 = x = r * cos(theta)
// output_2 = y = r * sin(theta)
// ============================================================================

@compute @workgroup_size(256)
fn polar_to_cartesian(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let r = input_a[get_a_idx(idx)];
    let theta = input_b[get_b_idx(idx)];

    output_1[idx] = r * cos(theta);
    output_2[idx] = r * sin(theta);
}

// ============================================================================
// Cartesian to Polar
// input_a = x
// input_b = y
// output_1 = r = sqrt(x^2 + y^2)
// output_2 = theta = atan2(y, x)
// ============================================================================

@compute @workgroup_size(256)
fn cartesian_to_polar(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let x = input_a[get_a_idx(idx)];
    let y = input_b[get_b_idx(idx)];

    output_1[idx] = sqrt(x * x + y * y);
    output_2[idx] = atan2(y, x);
}

// ============================================================================
// Min/Max with indices
// output_1 = min/max value
// output_2 = index (0 if a, 1 if b) - stored as float
// ============================================================================

@compute @workgroup_size(256)
fn min_with_index(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];

    if (a <= b) {
        output_1[idx] = a;
        output_2[idx] = 0.0f;
    } else {
        output_1[idx] = b;
        output_2[idx] = 1.0f;
    }
}

@compute @workgroup_size(256)
fn max_with_index(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];

    if (a >= b) {
        output_1[idx] = a;
        output_2[idx] = 0.0f;
    } else {
        output_1[idx] = b;
        output_2[idx] = 1.0f;
    }
}

// ============================================================================
// Top-2 (return two largest values)
// output_1 = larger value
// output_2 = smaller value
// ============================================================================

@compute @workgroup_size(256)
fn top2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];

    output_1[idx] = max(a, b);
    output_2[idx] = min(a, b);
}

// ============================================================================
// Clamp with both bounds returned
// input_a = value to clamp
// input_b = low bound (output_2 gets high bound from separate binding)
// output_1 = clamped value
// output_2 = 0 if unclamped, -1 if clamped low, +1 if clamped high
// ============================================================================

@group(0) @binding(5) var<storage, read> input_c: array<f32>;

@compute @workgroup_size(256)
fn clamp_with_status(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let x = input_a[get_a_idx(idx)];
    let low = input_b[get_b_idx(idx)];
    let high = input_c[idx];

    if (x < low) {
        output_1[idx] = low;
        output_2[idx] = -1.0f;
    } else if (x > high) {
        output_1[idx] = high;
        output_2[idx] = 1.0f;
    } else {
        output_1[idx] = x;
        output_2[idx] = 0.0f;
    }
}

// ============================================================================
// Integer Two-Output Operations
// ============================================================================

@group(0) @binding(6) var<storage, read> input_a_i32: array<i32>;
@group(0) @binding(7) var<storage, read> input_b_i32: array<i32>;
@group(0) @binding(8) var<storage, read_write> output_1_i32: array<i32>;
@group(0) @binding(9) var<storage, read_write> output_2_i32: array<i32>;

@compute @workgroup_size(256)
fn divmod_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_i32[get_a_idx(idx)];
    let b = input_b_i32[get_b_idx(idx)];

    output_1_i32[idx] = a / b;
    output_2_i32[idx] = a % b;
}

@compute @workgroup_size(256)
fn min_max_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_i32[get_a_idx(idx)];
    let b = input_b_i32[get_b_idx(idx)];

    output_1_i32[idx] = min(a, b);
    output_2_i32[idx] = max(a, b);
}

// ============================================================================
// Extended GCD (for integer inputs) - returns gcd and quotient
// Not a true extended GCD but a simplified version
// ============================================================================

@compute @workgroup_size(256)
fn gcd_with_quotient(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    var a = input_a_i32[get_a_idx(idx)];
    var b = input_b_i32[get_b_idx(idx)];

    // Handle negative inputs
    if (a < 0) { a = -a; }
    if (b < 0) { b = -b; }

    // GCD using Euclidean algorithm
    let original_a = a;
    while (b != 0) {
        let temp = b;
        b = a % b;
        a = temp;
    }

    output_1_i32[idx] = a; // GCD
    if (a != 0) {
        output_2_i32[idx] = original_a / a; // How many times GCD fits in original a
    } else {
        output_2_i32[idx] = 0;
    }
}

// ============================================================================
// Split high/low bits (u32)
// Useful for extended precision arithmetic
// ============================================================================

@group(0) @binding(10) var<storage, read> input_a_u32: array<u32>;
@group(0) @binding(11) var<storage, read> input_b_u32: array<u32>;
@group(0) @binding(12) var<storage, read_write> output_1_u32: array<u32>;
@group(0) @binding(13) var<storage, read_write> output_2_u32: array<u32>;

@compute @workgroup_size(256)
fn mul_split(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_u32[get_a_idx(idx)];
    let b = input_b_u32[get_b_idx(idx)];

    // Split 32x32 -> 64-bit result into high and low words
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;

    let mid = p1 + p2;
    let lo = p0 + (mid << 16u);
    let hi = p3 + (mid >> 16u) + select(0u, 1u, lo < p0);

    output_1_u32[idx] = lo;  // Low 32 bits
    output_2_u32[idx] = hi;  // High 32 bits
}

@compute @workgroup_size(256)
fn add_with_carry(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_u32[get_a_idx(idx)];
    let b = input_b_u32[get_b_idx(idx)];

    let sum = a + b;
    let carry = select(0u, 1u, sum < a);

    output_1_u32[idx] = sum;
    output_2_u32[idx] = carry;
}

@compute @workgroup_size(256)
fn sub_with_borrow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_u32[get_a_idx(idx)];
    let b = input_b_u32[get_b_idx(idx)];

    let diff = a - b;
    let borrow = select(0u, 1u, a < b);

    output_1_u32[idx] = diff;
    output_2_u32[idx] = borrow;
}
