// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// FP16 Quantization Kernels for Neural Network Inference
// - FP32 to FP16 conversion with rounding modes
// - FP16 to FP32 dequantization
// - Per-tensor and per-channel scaling
// - Mixed-precision accumulation support
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

// Enable f16 extension when available
// Note: Requires "shader-f16" feature in WebGPU
enable f16;

// ============================================================================
// Constants
// ============================================================================

const FP16_MAX: f32 = 65504.0;
const FP16_MIN: f32 = -65504.0;
const FP16_EPSILON: f32 = 0.0009765625;  // 2^-10
const FP16_DENORM_MIN: f32 = 5.96046448e-8;  // 2^-24

const WORKGROUP_SIZE: u32 = 256u;

// ============================================================================
// Parameter Structures
// ============================================================================

struct FP16QuantParams {
    size: u32,              // Total elements
    scale: f32,             // Global scale factor
    num_channels: u32,      // For per-channel quantization
    channel_stride: u32,    // Elements per channel
}

struct FP16ChannelParams {
    offset: u32,            // Offset in channel scales array
    rounding_mode: u32,     // 0=nearest, 1=stochastic, 2=truncate
    clamp_inf: u32,         // 1=clamp infinities to max
    _pad: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> input_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_f16: array<u32>;  // Packed f16 pairs
@group(0) @binding(2) var<uniform> params: FP16QuantParams;
@group(0) @binding(3) var<storage, read> channel_scales: array<f32>;
@group(0) @binding(4) var<uniform> channel_params: FP16ChannelParams;

// Alternative bindings for dequantization
@group(1) @binding(0) var<storage, read> input_packed_f16: array<u32>;
@group(1) @binding(1) var<storage, read_write> output_f32: array<f32>;

// ============================================================================
// FP16 Bit Manipulation Helpers
// ============================================================================

// Convert f32 to f16 bit pattern (IEEE 754)
fn f32_to_f16_bits(val: f32) -> u32 {
    let bits = bitcast<u32>(val);
    let sign = (bits >> 16u) & 0x8000u;
    let exp = (bits >> 23u) & 0xFFu;
    let mant = bits & 0x7FFFFFu;

    // Handle special cases
    if (exp == 255u) {
        // Infinity or NaN
        if (mant != 0u) {
            // NaN - preserve quiet bit
            return sign | 0x7E00u;
        } else {
            // Infinity
            return sign | 0x7C00u;
        }
    }

    if (exp == 0u) {
        // Zero or denormal -> zero in f16
        return sign;
    }

    // Rebias exponent: f32 bias=127, f16 bias=15
    let new_exp = i32(exp) - 127 + 15;

    if (new_exp >= 31) {
        // Overflow to infinity
        return sign | 0x7C00u;
    }

    if (new_exp <= 0) {
        // Denormal or underflow
        if (new_exp < -10) {
            return sign;  // Too small, flush to zero
        }
        // Denormalize
        let shift = u32(1 - new_exp);
        let mant_with_implicit = mant | 0x800000u;
        let denorm_mant = (mant_with_implicit >> (13u + shift)) +
                          u32((mant_with_implicit >> (12u + shift)) & 1u);  // Round
        return sign | denorm_mant;
    }

    // Normal number - round to nearest even
    let truncated_mant = mant >> 13u;
    let round_bit = (mant >> 12u) & 1u;
    let sticky = u32(mant & 0xFFFu) != 0u;

    var rounded_mant = truncated_mant;
    if (round_bit != 0u && (sticky || (truncated_mant & 1u) != 0u)) {
        rounded_mant += 1u;
        if (rounded_mant >= 0x400u) {
            rounded_mant = 0u;
            if (new_exp < 30) {
                return sign | (u32(new_exp + 1) << 10u);
            } else {
                return sign | 0x7C00u;  // Overflow to infinity
            }
        }
    }

    return sign | (u32(new_exp) << 10u) | rounded_mant;
}

// Convert f16 bit pattern to f32
fn f16_bits_to_f32(bits: u32) -> f32 {
    let sign = (bits & 0x8000u) << 16u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;

    if (exp == 0u) {
        if (mant == 0u) {
            // Zero
            return bitcast<f32>(sign);
        }
        // Denormal - normalize
        var m = mant;
        var e = 0i;
        while ((m & 0x400u) == 0u) {
            m <<= 1u;
            e -= 1;
        }
        m &= 0x3FFu;
        let new_exp = u32(127 - 15 + 1 + e);
        return bitcast<f32>(sign | (new_exp << 23u) | (m << 13u));
    }

    if (exp == 31u) {
        if (mant == 0u) {
            // Infinity
            return bitcast<f32>(sign | 0x7F800000u);
        }
        // NaN
        return bitcast<f32>(sign | 0x7FC00000u | (mant << 13u));
    }

    // Normal number
    let new_exp = exp + 127u - 15u;
    return bitcast<f32>(sign | (new_exp << 23u) | (mant << 13u));
}

// Pack two f16 values into a u32
fn pack_f16x2(a: u32, b: u32) -> u32 {
    return (b << 16u) | (a & 0xFFFFu);
}

// Unpack u32 into two f16 bit patterns
fn unpack_f16x2(packed: u32) -> vec2<u32> {
    return vec2<u32>(packed & 0xFFFFu, packed >> 16u);
}

// ============================================================================
// Stochastic Rounding
// ============================================================================

// Simple hash for stochastic rounding
fn hash_u32(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 0x85EBCA6Bu;
    h = h ^ (h >> 13u);
    h = h * 0xC2B2AE35u;
    h = h ^ (h >> 16u);
    return h;
}

fn stochastic_round_f16(val: f32, idx: u32) -> u32 {
    let bits = bitcast<u32>(val);
    let sign = (bits >> 16u) & 0x8000u;
    let exp = (bits >> 23u) & 0xFFu;
    let mant = bits & 0x7FFFFFu;

    // Handle special cases
    if (exp >= 255u || exp == 0u) {
        return f32_to_f16_bits(val);
    }

    let new_exp = i32(exp) - 127 + 15;
    if (new_exp >= 31 || new_exp <= 0) {
        return f32_to_f16_bits(val);
    }

    // Stochastic rounding: add random value in [0, 1) * ulp
    let truncated_mant = mant >> 13u;
    let residual = mant & 0x1FFFu;  // 13 bits of residual
    let random = hash_u32(idx ^ bits) & 0x1FFFu;

    var rounded_mant = truncated_mant;
    if (random < residual) {
        rounded_mant += 1u;
        if (rounded_mant >= 0x400u) {
            rounded_mant = 0u;
            if (new_exp < 30) {
                return sign | (u32(new_exp + 1) << 10u);
            } else {
                return sign | 0x7C00u;
            }
        }
    }

    return sign | (u32(new_exp) << 10u) | rounded_mant;
}

// ============================================================================
// FP32 to FP16 Quantization Kernels
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE)
fn quantize_fp32_to_fp16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    let elem_idx = pair_idx * 2u;

    if (elem_idx >= params.size) { return; }

    // Load and scale
    let val0 = input_f32[elem_idx] * params.scale;
    var val1 = 0.0f;
    if (elem_idx + 1u < params.size) {
        val1 = input_f32[elem_idx + 1u] * params.scale;
    }

    // Clamp to FP16 range
    let clamped0 = clamp(val0, FP16_MIN, FP16_MAX);
    let clamped1 = clamp(val1, FP16_MIN, FP16_MAX);

    // Convert to f16 bits
    let f16_0 = f32_to_f16_bits(clamped0);
    let f16_1 = f32_to_f16_bits(clamped1);

    // Pack and store
    output_f16[pair_idx] = pack_f16x2(f16_0, f16_1);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn quantize_fp32_to_fp16_stochastic(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    let elem_idx = pair_idx * 2u;

    if (elem_idx >= params.size) { return; }

    // Load and scale
    let val0 = input_f32[elem_idx] * params.scale;
    var val1 = 0.0f;
    if (elem_idx + 1u < params.size) {
        val1 = input_f32[elem_idx + 1u] * params.scale;
    }

    // Clamp to FP16 range
    let clamped0 = clamp(val0, FP16_MIN, FP16_MAX);
    let clamped1 = clamp(val1, FP16_MIN, FP16_MAX);

    // Stochastic rounding
    let f16_0 = stochastic_round_f16(clamped0, elem_idx);
    let f16_1 = stochastic_round_f16(clamped1, elem_idx + 1u);

    // Pack and store
    output_f16[pair_idx] = pack_f16x2(f16_0, f16_1);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn quantize_fp32_to_fp16_per_channel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    let elem_idx = pair_idx * 2u;

    if (elem_idx >= params.size) { return; }

    // Determine channel for each element
    let channel0 = elem_idx / params.channel_stride;
    let scale0 = channel_scales[channel_params.offset + channel0];

    var scale1 = 1.0f;
    if (elem_idx + 1u < params.size) {
        let channel1 = (elem_idx + 1u) / params.channel_stride;
        scale1 = channel_scales[channel_params.offset + channel1];
    }

    // Load and scale
    let val0 = input_f32[elem_idx] * scale0;
    var val1 = 0.0f;
    if (elem_idx + 1u < params.size) {
        val1 = input_f32[elem_idx + 1u] * scale1;
    }

    // Clamp and convert
    let clamped0 = clamp(val0, FP16_MIN, FP16_MAX);
    let clamped1 = clamp(val1, FP16_MIN, FP16_MAX);

    let f16_0 = f32_to_f16_bits(clamped0);
    let f16_1 = f32_to_f16_bits(clamped1);

    output_f16[pair_idx] = pack_f16x2(f16_0, f16_1);
}

// ============================================================================
// FP16 to FP32 Dequantization Kernels
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE)
fn dequantize_fp16_to_fp32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    let elem_idx = pair_idx * 2u;

    if (elem_idx >= params.size) { return; }

    // Load packed f16
    let packed = input_packed_f16[pair_idx];
    let unpacked = unpack_f16x2(packed);

    // Convert to f32 and apply inverse scale
    let inv_scale = 1.0f / params.scale;
    let val0 = f16_bits_to_f32(unpacked.x) * inv_scale;

    output_f32[elem_idx] = val0;

    if (elem_idx + 1u < params.size) {
        let val1 = f16_bits_to_f32(unpacked.y) * inv_scale;
        output_f32[elem_idx + 1u] = val1;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn dequantize_fp16_to_fp32_per_channel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    let elem_idx = pair_idx * 2u;

    if (elem_idx >= params.size) { return; }

    // Load packed f16
    let packed = input_packed_f16[pair_idx];
    let unpacked = unpack_f16x2(packed);

    // Determine channel scales
    let channel0 = elem_idx / params.channel_stride;
    let inv_scale0 = 1.0f / channel_scales[channel_params.offset + channel0];

    let val0 = f16_bits_to_f32(unpacked.x) * inv_scale0;
    output_f32[elem_idx] = val0;

    if (elem_idx + 1u < params.size) {
        let channel1 = (elem_idx + 1u) / params.channel_stride;
        let inv_scale1 = 1.0f / channel_scales[channel_params.offset + channel1];
        let val1 = f16_bits_to_f32(unpacked.y) * inv_scale1;
        output_f32[elem_idx + 1u] = val1;
    }
}

// ============================================================================
// Native f16 Operations (when extension is available)
// ============================================================================

// Native f16 storage bindings (use when shader-f16 is available)
@group(2) @binding(0) var<storage, read> input_native_f32: array<f32>;
@group(2) @binding(1) var<storage, read_write> output_native_f16: array<f16>;
@group(2) @binding(2) var<storage, read> input_native_f16: array<f16>;
@group(2) @binding(3) var<storage, read_write> output_native_f32: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn quantize_native_f16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let val = input_native_f32[idx] * params.scale;
    output_native_f16[idx] = f16(clamp(val, FP16_MIN, FP16_MAX));
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn dequantize_native_f16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let val = f32(input_native_f16[idx]);
    output_native_f32[idx] = val / params.scale;
}

// ============================================================================
// Mixed Precision Dot Product (FP16 inputs, FP32 accumulation)
// ============================================================================

struct DotParams {
    size: u32,
    num_vectors: u32,
    vector_stride: u32,
    _pad: u32,
}

@group(3) @binding(0) var<storage, read> dot_a_f16: array<u32>;  // Packed f16
@group(3) @binding(1) var<storage, read> dot_b_f16: array<u32>;  // Packed f16
@group(3) @binding(2) var<storage, read_write> dot_result: array<f32>;
@group(3) @binding(3) var<uniform> dot_params: DotParams;

var<workgroup> partial_sums: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn dot_product_fp16_fp32(@builtin(global_invocation_id) gid: vec3<u32>,
                         @builtin(local_invocation_id) lid: vec3<u32>,
                         @builtin(workgroup_id) wgid: vec3<u32>) {
    let vector_idx = wgid.x;
    let local_idx = lid.x;

    if (vector_idx >= dot_params.num_vectors) { return; }

    let base = vector_idx * dot_params.vector_stride;
    let num_pairs = (dot_params.size + 1u) / 2u;

    // Accumulate partial sum
    var sum = 0.0f;
    var i = local_idx;
    while (i < num_pairs) {
        let pair_a = dot_a_f16[base / 2u + i];
        let pair_b = dot_b_f16[base / 2u + i];

        let a = unpack_f16x2(pair_a);
        let b = unpack_f16x2(pair_b);

        let a0 = f16_bits_to_f32(a.x);
        let a1 = f16_bits_to_f32(a.y);
        let b0 = f16_bits_to_f32(b.x);
        let b1 = f16_bits_to_f32(b.y);

        sum += a0 * b0;
        if (i * 2u + 1u < dot_params.size) {
            sum += a1 * b1;
        }

        i += WORKGROUP_SIZE;
    }

    partial_sums[local_idx] = sum;
    workgroupBarrier();

    // Parallel reduction
    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (local_idx < stride) {
            partial_sums[local_idx] += partial_sums[local_idx + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (local_idx == 0u) {
        dot_result[vector_idx] = partial_sums[0];
    }
}
