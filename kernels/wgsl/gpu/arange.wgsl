// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Arange Kernel - Generate sequential values [start, start+step, start+2*step, ...]
// Supports i32, u32, and f32 output types with configurable start, stop, and step.
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

// ============================================================================
// Parameter Structure
// ============================================================================

struct ArangeParams {
    size: u32,           // Number of elements to generate
    start_i32: i32,      // Start value (for integer mode)
    step_i32: i32,       // Step value (for integer mode)
    start_f32: f32,      // Start value (for float mode)
    step_f32: f32,       // Step value (for float mode)
    dtype: u32,          // 0 = i32, 1 = u32, 2 = f32
    _pad1: u32,
    _pad2: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<uniform> params: ArangeParams;

// ============================================================================
// Arange Kernel - Generate sequential i32 values
// ============================================================================

@compute @workgroup_size(256)
fn arange_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let value = params.start_i32 + i32(idx) * params.step_i32;
    output[idx] = bitcast<u32>(value);
}

// ============================================================================
// Arange Kernel - Generate sequential u32 values
// ============================================================================

@compute @workgroup_size(256)
fn arange_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let value = u32(params.start_i32) + idx * u32(params.step_i32);
    output[idx] = value;
}

// ============================================================================
// Arange Kernel - Generate sequential f32 values
// ============================================================================

@compute @workgroup_size(256)
fn arange_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let value = params.start_f32 + f32(idx) * params.step_f32;
    output[idx] = bitcast<u32>(value);
}

// ============================================================================
// Arange Kernel - Dynamic type selection
// ============================================================================

@compute @workgroup_size(256)
fn arange(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    if (params.dtype == 0u) {
        // i32 mode
        let value = params.start_i32 + i32(idx) * params.step_i32;
        output[idx] = bitcast<u32>(value);
    } else if (params.dtype == 1u) {
        // u32 mode
        let value = u32(params.start_i32) + idx * u32(params.step_i32);
        output[idx] = value;
    } else {
        // f32 mode
        let value = params.start_f32 + f32(idx) * params.step_f32;
        output[idx] = bitcast<u32>(value);
    }
}

// ============================================================================
// Linspace Kernel - Generate evenly spaced f32 values between start and stop
// ============================================================================

struct LinspaceParams {
    size: u32,           // Number of elements
    start: f32,          // Start value
    stop: f32,           // Stop value
    endpoint: u32,       // 1 = include endpoint, 0 = exclude
}

@group(0) @binding(2) var<uniform> linspace_params: LinspaceParams;

@compute @workgroup_size(256)
fn linspace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= linspace_params.size) { return; }

    var divisor: f32;
    if (linspace_params.endpoint == 1u) {
        divisor = f32(linspace_params.size - 1u);
    } else {
        divisor = f32(linspace_params.size);
    }

    // Handle edge case of size == 1
    if (divisor <= 0.0) {
        output[idx] = bitcast<u32>(linspace_params.start);
        return;
    }

    let t = f32(idx) / divisor;
    let value = linspace_params.start + t * (linspace_params.stop - linspace_params.start);
    output[idx] = bitcast<u32>(value);
}

// ============================================================================
// Logspace Kernel - Generate logarithmically spaced values
// ============================================================================

struct LogspaceParams {
    size: u32,           // Number of elements
    start: f32,          // Start exponent
    stop: f32,           // Stop exponent
    base: f32,           // Logarithm base (default 10.0)
    endpoint: u32,       // 1 = include endpoint, 0 = exclude
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(3) var<uniform> logspace_params: LogspaceParams;

@compute @workgroup_size(256)
fn logspace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= logspace_params.size) { return; }

    var divisor: f32;
    if (logspace_params.endpoint == 1u) {
        divisor = f32(logspace_params.size - 1u);
    } else {
        divisor = f32(logspace_params.size);
    }

    if (divisor <= 0.0) {
        let value = pow(logspace_params.base, logspace_params.start);
        output[idx] = bitcast<u32>(value);
        return;
    }

    let t = f32(idx) / divisor;
    let exponent = logspace_params.start + t * (logspace_params.stop - logspace_params.start);
    let value = pow(logspace_params.base, exponent);
    output[idx] = bitcast<u32>(value);
}

// ============================================================================
// Full Kernel - Fill array with constant value
// ============================================================================

struct FillParams {
    size: u32,
    value: u32,          // Value to fill (bit pattern for any type)
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(4) var<uniform> fill_params: FillParams;

@compute @workgroup_size(256)
fn full(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= fill_params.size) { return; }

    output[idx] = fill_params.value;
}

// ============================================================================
// Zeros Kernel - Fill array with zeros
// ============================================================================

@compute @workgroup_size(256)
fn zeros(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    output[idx] = 0u;
}

// ============================================================================
// Ones Kernel - Fill array with ones (f32 1.0)
// ============================================================================

@compute @workgroup_size(256)
fn ones_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    output[idx] = bitcast<u32>(1.0f);
}

// ============================================================================
// Ones Kernel - Fill array with ones (i32/u32 1)
// ============================================================================

@compute @workgroup_size(256)
fn ones_int(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    output[idx] = 1u;
}
