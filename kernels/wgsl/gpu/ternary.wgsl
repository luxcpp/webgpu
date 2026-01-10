// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Ternary Operations Kernels - Three-input operations
// Includes where/select, fma, clamp, lerp, and conditional operations.
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

// ============================================================================
// Parameter Structure
// ============================================================================

struct TernaryParams {
    size: u32,           // Total number of elements
    a_stride: u32,       // Stride for input A (0 for broadcast)
    b_stride: u32,       // Stride for input B (0 for broadcast)
    c_stride: u32,       // Stride for input C (0 for broadcast)
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read> input_c: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: TernaryParams;

// ============================================================================
// Helper: Calculate input indices with broadcasting
// ============================================================================

fn get_a_idx(idx: u32) -> u32 {
    if (params.a_stride == 0u) { return 0u; }
    return idx * params.a_stride;
}

fn get_b_idx(idx: u32) -> u32 {
    if (params.b_stride == 0u) { return 0u; }
    return idx * params.b_stride;
}

fn get_c_idx(idx: u32) -> u32 {
    if (params.c_stride == 0u) { return 0u; }
    return idx * params.c_stride;
}

// ============================================================================
// Where/Select - Conditional selection
// output = condition ? true_val : false_val
// A = condition (non-zero = true), B = true_val, C = false_val
// ============================================================================

@compute @workgroup_size(256)
fn where_select(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let condition = input_a[get_a_idx(idx)];
    let true_val = input_b[get_b_idx(idx)];
    let false_val = input_c[get_c_idx(idx)];

    output[idx] = select(false_val, true_val, condition != 0.0f);
}

// ============================================================================
// FMA - Fused Multiply-Add
// output = a * b + c
// ============================================================================

@compute @workgroup_size(256)
fn fma_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    let c = input_c[get_c_idx(idx)];

    output[idx] = fma(a, b, c);
}

// ============================================================================
// Multiply-Subtract
// output = a * b - c
// ============================================================================

@compute @workgroup_size(256)
fn fms(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    let c = input_c[get_c_idx(idx)];

    output[idx] = a * b - c;
}

// ============================================================================
// Negative Multiply-Add
// output = -a * b + c = c - a * b
// ============================================================================

@compute @workgroup_size(256)
fn fnma(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    let c = input_c[get_c_idx(idx)];

    output[idx] = c - a * b;
}

// ============================================================================
// Clamp - Constrain value between bounds
// output = clamp(a, low, high)
// A = value, B = low, C = high
// ============================================================================

@compute @workgroup_size(256)
fn clamp_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let value = input_a[get_a_idx(idx)];
    let low = input_b[get_b_idx(idx)];
    let high = input_c[get_c_idx(idx)];

    output[idx] = clamp(value, low, high);
}

// ============================================================================
// Lerp - Linear interpolation
// output = a + t * (b - a) = a * (1 - t) + b * t
// A = start, B = end, C = t (interpolation factor)
// ============================================================================

@compute @workgroup_size(256)
fn lerp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let start = input_a[get_a_idx(idx)];
    let end = input_b[get_b_idx(idx)];
    let t = input_c[get_c_idx(idx)];

    output[idx] = mix(start, end, t);
}

// ============================================================================
// Smoothstep - Smooth Hermite interpolation
// output = smoothstep(edge0, edge1, x)
// A = edge0, B = edge1, C = x
// ============================================================================

@compute @workgroup_size(256)
fn smoothstep_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let edge0 = input_a[get_a_idx(idx)];
    let edge1 = input_b[get_b_idx(idx)];
    let x = input_c[get_c_idx(idx)];

    output[idx] = smoothstep(edge0, edge1, x);
}

// ============================================================================
// AddCmul - Add with conditional multiply
// output = a + condition * b
// A = base, B = addend, C = condition/multiplier
// ============================================================================

@compute @workgroup_size(256)
fn addcmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    let c = input_c[get_c_idx(idx)];

    output[idx] = a + c * b;
}

// ============================================================================
// AddCdiv - Add with conditional divide
// output = a + b / c
// A = base, B = dividend, C = divisor
// ============================================================================

@compute @workgroup_size(256)
fn addcdiv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    let c = input_c[get_c_idx(idx)];

    output[idx] = a + b / c;
}

// ============================================================================
// Clip Gradient - Gradient clipping
// output = clamp(gradient * scale, -clip_val, clip_val)
// A = gradient, B = scale, C = clip_value
// ============================================================================

@compute @workgroup_size(256)
fn clip_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let gradient = input_a[get_a_idx(idx)];
    let scale = input_b[get_b_idx(idx)];
    let clip_val = input_c[get_c_idx(idx)];

    let scaled = gradient * scale;
    output[idx] = clamp(scaled, -clip_val, clip_val);
}

// ============================================================================
// Blend (Alpha blend)
// output = a * alpha + b * (1 - alpha)
// A = first, B = second, C = alpha
// ============================================================================

@compute @workgroup_size(256)
fn blend(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    let alpha = input_c[get_c_idx(idx)];

    output[idx] = a * alpha + b * (1.0f - alpha);
}

// ============================================================================
// Within Range Check
// output = 1.0 if low <= x <= high, else 0.0
// A = value, B = low, C = high
// ============================================================================

@compute @workgroup_size(256)
fn within_range(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let value = input_a[get_a_idx(idx)];
    let low = input_b[get_b_idx(idx)];
    let high = input_c[get_c_idx(idx)];

    output[idx] = select(0.0f, 1.0f, value >= low && value <= high);
}

// ============================================================================
// Ternary Max - Maximum of three values
// ============================================================================

@compute @workgroup_size(256)
fn max3(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    let c = input_c[get_c_idx(idx)];

    output[idx] = max(max(a, b), c);
}

// ============================================================================
// Ternary Min - Minimum of three values
// ============================================================================

@compute @workgroup_size(256)
fn min3(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    let c = input_c[get_c_idx(idx)];

    output[idx] = min(min(a, b), c);
}

// ============================================================================
// Median of Three
// ============================================================================

@compute @workgroup_size(256)
fn median3(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    let c = input_c[get_c_idx(idx)];

    // Median = max(min(a,b), min(max(a,b), c))
    output[idx] = max(min(a, b), min(max(a, b), c));
}

// ============================================================================
// Integer Ternary Operations
// ============================================================================

@group(0) @binding(5) var<storage, read> input_a_i32: array<i32>;
@group(0) @binding(6) var<storage, read> input_b_i32: array<i32>;
@group(0) @binding(7) var<storage, read> input_c_i32: array<i32>;
@group(0) @binding(8) var<storage, read_write> output_i32: array<i32>;

@compute @workgroup_size(256)
fn where_select_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let condition = input_a_i32[get_a_idx(idx)];
    let true_val = input_b_i32[get_b_idx(idx)];
    let false_val = input_c_i32[get_c_idx(idx)];

    output_i32[idx] = select(false_val, true_val, condition != 0);
}

@compute @workgroup_size(256)
fn clamp_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let value = input_a_i32[get_a_idx(idx)];
    let low = input_b_i32[get_b_idx(idx)];
    let high = input_c_i32[get_c_idx(idx)];

    output_i32[idx] = clamp(value, low, high);
}

@compute @workgroup_size(256)
fn fma_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_i32[get_a_idx(idx)];
    let b = input_b_i32[get_b_idx(idx)];
    let c = input_c_i32[get_c_idx(idx)];

    output_i32[idx] = a * b + c;
}

// ============================================================================
// Unsigned Integer Ternary Operations
// ============================================================================

@group(0) @binding(9) var<storage, read> input_a_u32: array<u32>;
@group(0) @binding(10) var<storage, read> input_b_u32: array<u32>;
@group(0) @binding(11) var<storage, read> input_c_u32: array<u32>;
@group(0) @binding(12) var<storage, read_write> output_u32: array<u32>;

@compute @workgroup_size(256)
fn where_select_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let condition = input_a_u32[get_a_idx(idx)];
    let true_val = input_b_u32[get_b_idx(idx)];
    let false_val = input_c_u32[get_c_idx(idx)];

    output_u32[idx] = select(false_val, true_val, condition != 0u);
}

@compute @workgroup_size(256)
fn clamp_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let value = input_a_u32[get_a_idx(idx)];
    let low = input_b_u32[get_b_idx(idx)];
    let high = input_c_u32[get_c_idx(idx)];

    output_u32[idx] = clamp(value, low, high);
}

// ============================================================================
// Bitwise Ternary (for masked operations)
// output = (a & mask) | (b & ~mask)
// A = a, B = b, C = mask
// ============================================================================

@compute @workgroup_size(256)
fn bitwise_select(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_u32[get_a_idx(idx)];
    let b = input_b_u32[get_b_idx(idx)];
    let mask = input_c_u32[get_c_idx(idx)];

    output_u32[idx] = (a & mask) | (b & ~mask);
}

// ============================================================================
// Conditional Add - Add if condition is true
// output = condition ? (a + b) : a
// ============================================================================

@compute @workgroup_size(256)
fn conditional_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    let condition = input_c[get_c_idx(idx)];

    output[idx] = select(a, a + b, condition != 0.0f);
}

// ============================================================================
// Conditional Multiply - Multiply if condition is true
// output = condition ? (a * b) : a
// ============================================================================

@compute @workgroup_size(256)
fn conditional_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    let condition = input_c[get_c_idx(idx)];

    output[idx] = select(a, a * b, condition != 0.0f);
}
