// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// INT8/INT4 Quantization Kernels for Neural Network Inference
// - Symmetric and asymmetric quantization
// - Per-tensor and per-channel scales
// - INT8, INT4, and mixed-precision support
// - Optimized pack/unpack operations
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// INT8 quantization limits
const INT8_MIN: i32 = -128;
const INT8_MAX: i32 = 127;
const INT8_RANGE: f32 = 255.0;

// INT4 quantization limits
const INT4_MIN: i32 = -8;
const INT4_MAX: i32 = 7;
const INT4_RANGE: f32 = 15.0;

// UINT8 limits (for asymmetric)
const UINT8_MIN: u32 = 0u;
const UINT8_MAX: u32 = 255u;

// ============================================================================
// Parameter Structures
// ============================================================================

struct QuantParams {
    size: u32,              // Total elements
    scale: f32,             // Quantization scale
    zero_point: i32,        // Zero point (for asymmetric)
    quant_type: u32,        // 0=INT8_SYM, 1=INT8_ASYM, 2=INT4_SYM, 3=INT4_ASYM
}

struct ChannelQuantParams {
    num_channels: u32,      // Number of channels
    channel_stride: u32,    // Elements per channel
    _pad0: u32,
    _pad1: u32,
}

struct PerChannelScale {
    scale: f32,
    zero_point: i32,
    _pad0: u32,
    _pad1: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

// Per-tensor quantization
@group(0) @binding(0) var<storage, read> input_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_int: array<i32>;
@group(0) @binding(2) var<uniform> params: QuantParams;

// Per-channel quantization
@group(0) @binding(3) var<storage, read> channel_scales: array<PerChannelScale>;
@group(0) @binding(4) var<uniform> channel_params: ChannelQuantParams;

// Packed output for INT4 (two INT4 values per byte)
@group(0) @binding(5) var<storage, read_write> output_packed: array<u32>;

// Dequantization bindings
@group(1) @binding(0) var<storage, read> input_int: array<i32>;
@group(1) @binding(1) var<storage, read_write> output_f32: array<f32>;
@group(1) @binding(2) var<storage, read> input_packed: array<u32>;

// ============================================================================
// INT8 Quantization Helpers
// ============================================================================

fn quantize_int8_symmetric(val: f32, scale: f32) -> i32 {
    let scaled = round(val / scale);
    return clamp(i32(scaled), INT8_MIN, INT8_MAX);
}

fn dequantize_int8_symmetric(val: i32, scale: f32) -> f32 {
    return f32(val) * scale;
}

fn quantize_int8_asymmetric(val: f32, scale: f32, zero_point: i32) -> i32 {
    let scaled = round(val / scale) + f32(zero_point);
    return clamp(i32(scaled), INT8_MIN, INT8_MAX);
}

fn dequantize_int8_asymmetric(val: i32, scale: f32, zero_point: i32) -> f32 {
    return f32(val - zero_point) * scale;
}

// ============================================================================
// INT4 Quantization Helpers
// ============================================================================

fn quantize_int4_symmetric(val: f32, scale: f32) -> i32 {
    let scaled = round(val / scale);
    return clamp(i32(scaled), INT4_MIN, INT4_MAX);
}

fn dequantize_int4_symmetric(val: i32, scale: f32) -> f32 {
    return f32(val) * scale;
}

fn quantize_int4_asymmetric(val: f32, scale: f32, zero_point: i32) -> i32 {
    let scaled = round(val / scale) + f32(zero_point);
    return clamp(i32(scaled), INT4_MIN, INT4_MAX);
}

fn dequantize_int4_asymmetric(val: i32, scale: f32, zero_point: i32) -> f32 {
    return f32(val - zero_point) * scale;
}

// Pack two INT4 values into a byte (i32 with sign extension handled)
fn pack_int4x2(a: i32, b: i32) -> u32 {
    let a_u = u32(a & 0xF);  // Keep low 4 bits
    let b_u = u32(b & 0xF);
    return (b_u << 4u) | a_u;
}

// Pack 8 INT4 values into a u32
fn pack_int4x8(vals: array<i32, 8>) -> u32 {
    var result = 0u;
    for (var i = 0u; i < 8u; i++) {
        result |= u32(vals[i] & 0xF) << (i * 4u);
    }
    return result;
}

// Unpack byte into two INT4 values (with sign extension)
fn unpack_int4x2(packed: u32) -> vec2<i32> {
    var a = i32(packed & 0xFu);
    var b = i32((packed >> 4u) & 0xFu);
    // Sign extend from 4-bit
    if (a >= 8) { a -= 16; }
    if (b >= 8) { b -= 16; }
    return vec2<i32>(a, b);
}

// Unpack u32 into 8 INT4 values
fn unpack_int4x8(packed: u32) -> array<i32, 8> {
    var result: array<i32, 8>;
    for (var i = 0u; i < 8u; i++) {
        var val = i32((packed >> (i * 4u)) & 0xFu);
        if (val >= 8) { val -= 16; }  // Sign extend
        result[i] = val;
    }
    return result;
}

// Pack 4 INT8 values into a u32
fn pack_int8x4(a: i32, b: i32, c: i32, d: i32) -> u32 {
    return (u32(d & 0xFF) << 24u) | (u32(c & 0xFF) << 16u) |
           (u32(b & 0xFF) << 8u) | u32(a & 0xFF);
}

// Unpack u32 into 4 INT8 values
fn unpack_int8x4(packed: u32) -> vec4<i32> {
    var a = i32(packed & 0xFFu);
    var b = i32((packed >> 8u) & 0xFFu);
    var c = i32((packed >> 16u) & 0xFFu);
    var d = i32((packed >> 24u) & 0xFFu);
    // Sign extend from 8-bit
    if (a >= 128) { a -= 256; }
    if (b >= 128) { b -= 256; }
    if (c >= 128) { c -= 256; }
    if (d >= 128) { d -= 256; }
    return vec4<i32>(a, b, c, d);
}

// ============================================================================
// INT8 Per-Tensor Quantization Kernels
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE)
fn quantize_int8_per_tensor(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let val = input_f32[idx];

    if (params.quant_type == 0u) {
        // Symmetric
        output_int[idx] = quantize_int8_symmetric(val, params.scale);
    } else {
        // Asymmetric
        output_int[idx] = quantize_int8_asymmetric(val, params.scale, params.zero_point);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn dequantize_int8_per_tensor(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let val = input_int[idx];

    if (params.quant_type == 0u) {
        output_f32[idx] = dequantize_int8_symmetric(val, params.scale);
    } else {
        output_f32[idx] = dequantize_int8_asymmetric(val, params.scale, params.zero_point);
    }
}

// Packed INT8 quantization (4 values per u32)
@compute @workgroup_size(WORKGROUP_SIZE)
fn quantize_int8_packed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let quad_idx = gid.x;
    let elem_idx = quad_idx * 4u;

    if (elem_idx >= params.size) { return; }

    var q_vals: vec4<i32>;
    for (var i = 0u; i < 4u; i++) {
        if (elem_idx + i < params.size) {
            let val = input_f32[elem_idx + i];
            if (params.quant_type == 0u) {
                q_vals[i] = quantize_int8_symmetric(val, params.scale);
            } else {
                q_vals[i] = quantize_int8_asymmetric(val, params.scale, params.zero_point);
            }
        } else {
            q_vals[i] = 0;
        }
    }

    output_packed[quad_idx] = pack_int8x4(q_vals.x, q_vals.y, q_vals.z, q_vals.w);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn dequantize_int8_packed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let quad_idx = gid.x;
    let elem_idx = quad_idx * 4u;

    if (elem_idx >= params.size) { return; }

    let packed = input_packed[quad_idx];
    let q_vals = unpack_int8x4(packed);

    for (var i = 0u; i < 4u; i++) {
        if (elem_idx + i < params.size) {
            if (params.quant_type == 0u) {
                output_f32[elem_idx + i] = dequantize_int8_symmetric(q_vals[i], params.scale);
            } else {
                output_f32[elem_idx + i] = dequantize_int8_asymmetric(q_vals[i], params.scale, params.zero_point);
            }
        }
    }
}

// ============================================================================
// INT8 Per-Channel Quantization Kernels
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE)
fn quantize_int8_per_channel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    // Determine channel
    let channel = idx / channel_params.channel_stride;
    let ch_params = channel_scales[channel];

    let val = input_f32[idx];

    if (params.quant_type == 0u) {
        output_int[idx] = quantize_int8_symmetric(val, ch_params.scale);
    } else {
        output_int[idx] = quantize_int8_asymmetric(val, ch_params.scale, ch_params.zero_point);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn dequantize_int8_per_channel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let channel = idx / channel_params.channel_stride;
    let ch_params = channel_scales[channel];

    let val = input_int[idx];

    if (params.quant_type == 0u) {
        output_f32[idx] = dequantize_int8_symmetric(val, ch_params.scale);
    } else {
        output_f32[idx] = dequantize_int8_asymmetric(val, ch_params.scale, ch_params.zero_point);
    }
}

// ============================================================================
// INT4 Quantization Kernels
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE)
fn quantize_int4_per_tensor(@builtin(global_invocation_id) gid: vec3<u32>) {
    let octet_idx = gid.x;
    let elem_idx = octet_idx * 8u;

    if (elem_idx >= params.size) { return; }

    var q_vals: array<i32, 8>;
    for (var i = 0u; i < 8u; i++) {
        if (elem_idx + i < params.size) {
            let val = input_f32[elem_idx + i];
            if (params.quant_type == 2u) {
                q_vals[i] = quantize_int4_symmetric(val, params.scale);
            } else {
                q_vals[i] = quantize_int4_asymmetric(val, params.scale, params.zero_point);
            }
        } else {
            q_vals[i] = 0;
        }
    }

    output_packed[octet_idx] = pack_int4x8(q_vals);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn dequantize_int4_per_tensor(@builtin(global_invocation_id) gid: vec3<u32>) {
    let octet_idx = gid.x;
    let elem_idx = octet_idx * 8u;

    if (elem_idx >= params.size) { return; }

    let packed = input_packed[octet_idx];
    let q_vals = unpack_int4x8(packed);

    for (var i = 0u; i < 8u; i++) {
        if (elem_idx + i < params.size) {
            if (params.quant_type == 2u) {
                output_f32[elem_idx + i] = dequantize_int4_symmetric(q_vals[i], params.scale);
            } else {
                output_f32[elem_idx + i] = dequantize_int4_asymmetric(q_vals[i], params.scale, params.zero_point);
            }
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn quantize_int4_per_channel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let octet_idx = gid.x;
    let elem_idx = octet_idx * 8u;

    if (elem_idx >= params.size) { return; }

    var q_vals: array<i32, 8>;
    for (var i = 0u; i < 8u; i++) {
        if (elem_idx + i < params.size) {
            let channel = (elem_idx + i) / channel_params.channel_stride;
            let ch_params = channel_scales[channel];
            let val = input_f32[elem_idx + i];

            if (params.quant_type == 2u) {
                q_vals[i] = quantize_int4_symmetric(val, ch_params.scale);
            } else {
                q_vals[i] = quantize_int4_asymmetric(val, ch_params.scale, ch_params.zero_point);
            }
        } else {
            q_vals[i] = 0;
        }
    }

    output_packed[octet_idx] = pack_int4x8(q_vals);
}

// ============================================================================
// Scale Computation Kernels
// ============================================================================

var<workgroup> shared_max: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_min: array<f32, WORKGROUP_SIZE>;

struct ScaleOutput {
    scale: f32,
    zero_point: i32,
    min_val: f32,
    max_val: f32,
}

@group(2) @binding(0) var<storage, read_write> scale_output: ScaleOutput;

@compute @workgroup_size(WORKGROUP_SIZE)
fn compute_scale_int8_symmetric(@builtin(global_invocation_id) gid: vec3<u32>,
                                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let local_idx = lid.x;

    // Find local max absolute value
    var local_max = 0.0f;
    var i = local_idx;
    while (i < params.size) {
        local_max = max(local_max, abs(input_f32[i]));
        i += WORKGROUP_SIZE;
    }

    shared_max[local_idx] = local_max;
    workgroupBarrier();

    // Parallel reduction
    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (local_idx < stride) {
            shared_max[local_idx] = max(shared_max[local_idx], shared_max[local_idx + stride]);
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (local_idx == 0u) {
        let max_val = shared_max[0];
        let scale = max_val / f32(INT8_MAX);
        scale_output = ScaleOutput(
            select(scale, 1.0f, scale == 0.0f),  // Avoid division by zero
            0,
            -max_val,
            max_val
        );
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn compute_scale_int8_asymmetric(@builtin(global_invocation_id) gid: vec3<u32>,
                                  @builtin(local_invocation_id) lid: vec3<u32>) {
    let local_idx = lid.x;

    // Find local min/max
    var local_max = -1e30f;
    var local_min = 1e30f;
    var i = local_idx;
    while (i < params.size) {
        let val = input_f32[i];
        local_max = max(local_max, val);
        local_min = min(local_min, val);
        i += WORKGROUP_SIZE;
    }

    shared_max[local_idx] = local_max;
    shared_min[local_idx] = local_min;
    workgroupBarrier();

    // Parallel reduction
    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (local_idx < stride) {
            shared_max[local_idx] = max(shared_max[local_idx], shared_max[local_idx + stride]);
            shared_min[local_idx] = min(shared_min[local_idx], shared_min[local_idx + stride]);
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (local_idx == 0u) {
        let max_val = shared_max[0];
        let min_val = shared_min[0];
        let range = max_val - min_val;
        let scale = range / INT8_RANGE;
        let zero_point = i32(round(f32(INT8_MIN) - min_val / scale));

        scale_output = ScaleOutput(
            select(scale, 1.0f, scale == 0.0f),
            clamp(zero_point, INT8_MIN, INT8_MAX),
            min_val,
            max_val
        );
    }
}

// ============================================================================
// Quantized Matrix Multiply (INT8 x INT8 -> INT32)
// ============================================================================

struct MatMulParams {
    M: u32,
    N: u32,
    K: u32,
    _pad: u32,
}

@group(3) @binding(0) var<storage, read> mat_a_packed: array<u32>;
@group(3) @binding(1) var<storage, read> mat_b_packed: array<u32>;
@group(3) @binding(2) var<storage, read_write> mat_c_int32: array<i32>;
@group(3) @binding(3) var<uniform> matmul_params: MatMulParams;

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<i32, 256>;  // 16x16
var<workgroup> tile_b: array<i32, 256>;  // 16x16

@compute @workgroup_size(16, 16)
fn matmul_int8(@builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>,
               @builtin(workgroup_id) wgid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    let local_row = lid.y;
    let local_col = lid.x;

    var sum = 0i;

    let num_tiles = (matmul_params.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // Load tile A
        let a_row = wgid.y * TILE_SIZE + local_row;
        let a_col = t * TILE_SIZE + local_col;

        if (a_row < matmul_params.M && a_col < matmul_params.K) {
            let flat_idx = a_row * matmul_params.K + a_col;
            let quad_idx = flat_idx / 4u;
            let pos_in_quad = flat_idx % 4u;
            let packed = mat_a_packed[quad_idx];
            let vals = unpack_int8x4(packed);
            tile_a[local_row * TILE_SIZE + local_col] = vals[pos_in_quad];
        } else {
            tile_a[local_row * TILE_SIZE + local_col] = 0;
        }

        // Load tile B
        let b_row = t * TILE_SIZE + local_row;
        let b_col = wgid.x * TILE_SIZE + local_col;

        if (b_row < matmul_params.K && b_col < matmul_params.N) {
            let flat_idx = b_row * matmul_params.N + b_col;
            let quad_idx = flat_idx / 4u;
            let pos_in_quad = flat_idx % 4u;
            let packed = mat_b_packed[quad_idx];
            let vals = unpack_int8x4(packed);
            tile_b[local_row * TILE_SIZE + local_col] = vals[pos_in_quad];
        } else {
            tile_b[local_row * TILE_SIZE + local_col] = 0;
        }

        workgroupBarrier();

        // Compute partial sum
        for (var k = 0u; k < TILE_SIZE; k++) {
            sum += tile_a[local_row * TILE_SIZE + k] * tile_b[k * TILE_SIZE + local_col];
        }

        workgroupBarrier();
    }

    // Store result
    if (row < matmul_params.M && col < matmul_params.N) {
        mat_c_int32[row * matmul_params.N + col] = sum;
    }
}

// ============================================================================
// Requantization (INT32 -> INT8 with scale adjustment)
// ============================================================================

struct RequantParams {
    size: u32,
    input_scale: f32,
    output_scale: f32,
    output_zero_point: i32,
}

@group(4) @binding(0) var<storage, read> requant_input: array<i32>;
@group(4) @binding(1) var<storage, read_write> requant_output: array<i32>;
@group(4) @binding(2) var<uniform> requant_params: RequantParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn requantize_int32_to_int8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= requant_params.size) { return; }

    let val_f32 = f32(requant_input[idx]) * requant_params.input_scale;
    let requant = round(val_f32 / requant_params.output_scale) + f32(requant_params.output_zero_point);
    requant_output[idx] = clamp(i32(requant), INT8_MIN, INT8_MAX);
}

// ============================================================================
// Mixed Precision: INT4 weights, INT8 activations
// ============================================================================

@group(5) @binding(0) var<storage, read> weights_int4: array<u32>;  // Packed INT4
@group(5) @binding(1) var<storage, read> activations_int8: array<u32>;  // Packed INT8
@group(5) @binding(2) var<storage, read_write> output_int32: array<i32>;

@compute @workgroup_size(16, 16)
fn matmul_int4_int8(@builtin(global_invocation_id) gid: vec3<u32>,
                    @builtin(local_invocation_id) lid: vec3<u32>,
                    @builtin(workgroup_id) wgid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    let local_row = lid.y;
    let local_col = lid.x;

    var sum = 0i;

    let num_tiles = (matmul_params.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // Load tile A (INT4 weights)
        let a_row = wgid.y * TILE_SIZE + local_row;
        let a_col = t * TILE_SIZE + local_col;

        if (a_row < matmul_params.M && a_col < matmul_params.K) {
            let flat_idx = a_row * matmul_params.K + a_col;
            let octet_idx = flat_idx / 8u;
            let pos_in_octet = flat_idx % 8u;
            let packed = weights_int4[octet_idx];
            let vals = unpack_int4x8(packed);
            tile_a[local_row * TILE_SIZE + local_col] = vals[pos_in_octet];
        } else {
            tile_a[local_row * TILE_SIZE + local_col] = 0;
        }

        // Load tile B (INT8 activations)
        let b_row = t * TILE_SIZE + local_row;
        let b_col = wgid.x * TILE_SIZE + local_col;

        if (b_row < matmul_params.K && b_col < matmul_params.N) {
            let flat_idx = b_row * matmul_params.N + b_col;
            let quad_idx = flat_idx / 4u;
            let pos_in_quad = flat_idx % 4u;
            let packed = activations_int8[quad_idx];
            let vals = unpack_int8x4(packed);
            tile_b[local_row * TILE_SIZE + local_col] = vals[pos_in_quad];
        } else {
            tile_b[local_row * TILE_SIZE + local_col] = 0;
        }

        workgroupBarrier();

        // Compute
        for (var k = 0u; k < TILE_SIZE; k++) {
            sum += tile_a[local_row * TILE_SIZE + k] * tile_b[k * TILE_SIZE + local_col];
        }

        workgroupBarrier();
    }

    if (row < matmul_params.M && col < matmul_params.N) {
        output_int32[row * matmul_params.N + col] = sum;
    }
}
