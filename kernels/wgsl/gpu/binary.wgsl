// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Binary Operations Kernels - Element-wise binary operations
// Supports add, sub, mul, div, pow, mod, min, max, and more.
// Handles broadcasting through stride parameters.
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

// ============================================================================
// Parameter Structure
// ============================================================================

struct BinaryParams {
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
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: BinaryParams;

// ============================================================================
// Helper: Calculate input index with broadcasting
// ============================================================================

fn get_a_idx(idx: u32) -> u32 {
    if (params.a_stride == 0u) {
        return 0u; // Broadcast scalar
    }
    return idx * params.a_stride;
}

fn get_b_idx(idx: u32) -> u32 {
    if (params.b_stride == 0u) {
        return 0u; // Broadcast scalar
    }
    return idx * params.b_stride;
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    output[idx] = a + b;
}

@compute @workgroup_size(256)
fn sub(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    output[idx] = a - b;
}

@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    output[idx] = a * b;
}

@compute @workgroup_size(256)
fn div(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    output[idx] = a / b;
}

@compute @workgroup_size(256)
fn pow_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    output[idx] = pow(a, b);
}

// ============================================================================
// Integer-style Operations (floor division, modulo)
// ============================================================================

@compute @workgroup_size(256)
fn floor_div(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    output[idx] = floor(a / b);
}

@compute @workgroup_size(256)
fn mod_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    // fmod: a - floor(a/b) * b (matches C fmod behavior)
    output[idx] = a - floor(a / b) * b;
}

@compute @workgroup_size(256)
fn remainder(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    // Python-style modulo (sign matches divisor)
    let m = a - floor(a / b) * b;
    output[idx] = m;
}

// ============================================================================
// Min/Max Operations
// ============================================================================

@compute @workgroup_size(256)
fn minimum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    output[idx] = min(a, b);
}

@compute @workgroup_size(256)
fn maximum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    output[idx] = max(a, b);
}

// ============================================================================
// Comparison Operations (output 0.0 or 1.0)
// ============================================================================

@compute @workgroup_size(256)
fn equal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    output[idx] = select(0.0f, 1.0f, a == b);
}

@compute @workgroup_size(256)
fn not_equal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    output[idx] = select(0.0f, 1.0f, a != b);
}

@compute @workgroup_size(256)
fn less(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    output[idx] = select(0.0f, 1.0f, a < b);
}

@compute @workgroup_size(256)
fn less_equal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    output[idx] = select(0.0f, 1.0f, a <= b);
}

@compute @workgroup_size(256)
fn greater(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    output[idx] = select(0.0f, 1.0f, a > b);
}

@compute @workgroup_size(256)
fn greater_equal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    output[idx] = select(0.0f, 1.0f, a >= b);
}

// ============================================================================
// Logical Operations (treating non-zero as true)
// ============================================================================

@compute @workgroup_size(256)
fn logical_and(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    output[idx] = select(0.0f, 1.0f, a != 0.0f && b != 0.0f);
}

@compute @workgroup_size(256)
fn logical_or(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    output[idx] = select(0.0f, 1.0f, a != 0.0f || b != 0.0f);
}

@compute @workgroup_size(256)
fn logical_xor(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    let a_bool = a != 0.0f;
    let b_bool = b != 0.0f;
    output[idx] = select(0.0f, 1.0f, (a_bool && !b_bool) || (!a_bool && b_bool));
}

// ============================================================================
// Special Math Operations
// ============================================================================

@compute @workgroup_size(256)
fn atan2_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    output[idx] = atan2(a, b);
}

@compute @workgroup_size(256)
fn hypot(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a[get_a_idx(idx)];
    let b = input_b[get_b_idx(idx)];
    output[idx] = sqrt(a * a + b * b);
}

@compute @workgroup_size(256)
fn copysign(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let magnitude = abs(input_a[get_a_idx(idx)]);
    let sign_val = input_b[get_b_idx(idx)];
    output[idx] = select(magnitude, -magnitude, sign_val < 0.0f);
}

// ============================================================================
// Integer Binary Operations (using bitcast)
// ============================================================================

@group(0) @binding(4) var<storage, read> input_a_i32: array<i32>;
@group(0) @binding(5) var<storage, read> input_b_i32: array<i32>;
@group(0) @binding(6) var<storage, read_write> output_i32: array<i32>;

@compute @workgroup_size(256)
fn add_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_i32[get_a_idx(idx)];
    let b = input_b_i32[get_b_idx(idx)];
    output_i32[idx] = a + b;
}

@compute @workgroup_size(256)
fn sub_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_i32[get_a_idx(idx)];
    let b = input_b_i32[get_b_idx(idx)];
    output_i32[idx] = a - b;
}

@compute @workgroup_size(256)
fn mul_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_i32[get_a_idx(idx)];
    let b = input_b_i32[get_b_idx(idx)];
    output_i32[idx] = a * b;
}

@compute @workgroup_size(256)
fn div_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_i32[get_a_idx(idx)];
    let b = input_b_i32[get_b_idx(idx)];
    output_i32[idx] = a / b;
}

@compute @workgroup_size(256)
fn mod_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_i32[get_a_idx(idx)];
    let b = input_b_i32[get_b_idx(idx)];
    output_i32[idx] = a % b;
}

@compute @workgroup_size(256)
fn min_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_i32[get_a_idx(idx)];
    let b = input_b_i32[get_b_idx(idx)];
    output_i32[idx] = min(a, b);
}

@compute @workgroup_size(256)
fn max_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_i32[get_a_idx(idx)];
    let b = input_b_i32[get_b_idx(idx)];
    output_i32[idx] = max(a, b);
}

// ============================================================================
// Bitwise Operations (u32)
// ============================================================================

@group(0) @binding(7) var<storage, read> input_a_u32: array<u32>;
@group(0) @binding(8) var<storage, read> input_b_u32: array<u32>;
@group(0) @binding(9) var<storage, read_write> output_u32: array<u32>;

@compute @workgroup_size(256)
fn bitwise_and(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_u32[get_a_idx(idx)];
    let b = input_b_u32[get_b_idx(idx)];
    output_u32[idx] = a & b;
}

@compute @workgroup_size(256)
fn bitwise_or(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_u32[get_a_idx(idx)];
    let b = input_b_u32[get_b_idx(idx)];
    output_u32[idx] = a | b;
}

@compute @workgroup_size(256)
fn bitwise_xor(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_u32[get_a_idx(idx)];
    let b = input_b_u32[get_b_idx(idx)];
    output_u32[idx] = a ^ b;
}

@compute @workgroup_size(256)
fn left_shift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_u32[get_a_idx(idx)];
    let b = input_b_u32[get_b_idx(idx)];
    output_u32[idx] = a << (b & 31u);
}

@compute @workgroup_size(256)
fn right_shift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let a = input_a_u32[get_a_idx(idx)];
    let b = input_b_u32[get_b_idx(idx)];
    output_u32[idx] = a >> (b & 31u);
}
