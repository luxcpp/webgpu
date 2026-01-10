// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// NAX Variant FP Quantization Kernels
// - Non-Affine eXtended (NAX) quantization for neural networks
// - Logarithmic scale quantization for dynamic range
// - Power-of-two scaling for efficient inference
// - Adaptive block-wise quantization
//
// NAX format: sign * 2^exponent * mantissa
// Optimized for transformer attention and normalization layers
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

enable f16;

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;
const BLOCK_SIZE: u32 = 128u;  // Elements per quantization block

// NAX format parameters
const NAX_EXP_BITS: u32 = 4u;    // Exponent bits
const NAX_MANT_BITS: u32 = 3u;   // Mantissa bits
const NAX_MAX_EXP: i32 = 7;      // Maximum exponent (2^7 = 128)
const NAX_MIN_EXP: i32 = -8;     // Minimum exponent (2^-8)
const NAX_MANT_LEVELS: u32 = 8u; // 2^3 mantissa levels

// ============================================================================
// Parameter Structures
// ============================================================================

struct NAXQuantParams {
    size: u32,              // Total elements
    num_blocks: u32,        // Number of quantization blocks
    block_size: u32,        // Elements per block
    mode: u32,              // 0=symmetric, 1=asymmetric, 2=adaptive
}

struct NAXBlockInfo {
    scale_exp: i32,         // Block-wise scale exponent
    shift: i32,             // Bit shift for alignment
    min_val: f32,           // Block minimum (for asymmetric)
    max_val: f32,           // Block maximum
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> input_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_nax: array<u32>;  // Packed NAX values
@group(0) @binding(2) var<uniform> params: NAXQuantParams;
@group(0) @binding(3) var<storage, read_write> block_info: array<NAXBlockInfo>;

// Dequantization bindings
@group(1) @binding(0) var<storage, read> input_nax: array<u32>;
@group(1) @binding(1) var<storage, read_write> output_f32: array<f32>;

// Workgroup shared memory
var<workgroup> shared_max: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_min: array<f32, WORKGROUP_SIZE>;

// ============================================================================
// NAX Format Utilities
// ============================================================================

// Encode a single NAX value (8-bit: 1 sign + 4 exp + 3 mantissa)
fn encode_nax8(val: f32, scale_exp: i32) -> u32 {
    if (val == 0.0f) {
        return 0u;  // Zero encoding
    }

    let sign = select(0u, 1u, val < 0.0f);
    let abs_val = abs(val);

    // Scale value by block scale
    let scaled = abs_val * pow(2.0f, f32(-scale_exp));

    // Find the exponent (log2)
    let log2_val = log2(max(scaled, 1e-20f));
    var exp_val = i32(floor(log2_val));

    // Clamp exponent to valid range
    exp_val = clamp(exp_val, NAX_MIN_EXP, NAX_MAX_EXP);

    // Calculate mantissa
    let mantissa_float = scaled / pow(2.0f, f32(exp_val)) - 1.0f;
    var mantissa = u32(round(mantissa_float * f32(NAX_MANT_LEVELS)));
    mantissa = min(mantissa, NAX_MANT_LEVELS - 1u);

    // Bias exponent for storage
    let biased_exp = u32(exp_val - NAX_MIN_EXP);

    // Pack: [sign(1)][exp(4)][mantissa(3)]
    return (sign << 7u) | (biased_exp << 3u) | mantissa;
}

// Decode a single NAX value
fn decode_nax8(nax: u32, scale_exp: i32) -> f32 {
    if (nax == 0u) {
        return 0.0f;
    }

    let sign = (nax >> 7u) & 1u;
    let biased_exp = (nax >> 3u) & 0xFu;
    let mantissa = nax & 0x7u;

    let exp_val = i32(biased_exp) + NAX_MIN_EXP;
    let mantissa_val = 1.0f + f32(mantissa) / f32(NAX_MANT_LEVELS);

    var val = mantissa_val * pow(2.0f, f32(exp_val));
    val = val * pow(2.0f, f32(scale_exp));  // Apply block scale

    return select(val, -val, sign == 1u);
}

// Pack four NAX8 values into a u32
fn pack_nax8x4(a: u32, b: u32, c: u32, d: u32) -> u32 {
    return (d << 24u) | (c << 16u) | (b << 8u) | a;
}

// Unpack u32 into four NAX8 values
fn unpack_nax8x4(packed: u32) -> vec4<u32> {
    return vec4<u32>(
        packed & 0xFFu,
        (packed >> 8u) & 0xFFu,
        (packed >> 16u) & 0xFFu,
        (packed >> 24u) & 0xFFu
    );
}

// ============================================================================
// Block Statistics Computation
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE)
fn compute_block_stats(@builtin(global_invocation_id) gid: vec3<u32>,
                       @builtin(local_invocation_id) lid: vec3<u32>,
                       @builtin(workgroup_id) wgid: vec3<u32>) {
    let block_idx = wgid.x;
    let local_idx = lid.x;

    if (block_idx >= params.num_blocks) { return; }

    let block_start = block_idx * params.block_size;
    let block_end = min(block_start + params.block_size, params.size);

    // Find local max/min
    var local_max = -1e30f;
    var local_min = 1e30f;

    var i = block_start + local_idx;
    while (i < block_end) {
        let val = input_f32[i];
        let abs_val = abs(val);
        local_max = max(local_max, abs_val);
        local_min = min(local_min, abs_val);
        i += WORKGROUP_SIZE;
    }

    shared_max[local_idx] = local_max;
    shared_min[local_idx] = local_min;
    workgroupBarrier();

    // Parallel reduction for max
    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (local_idx < stride) {
            shared_max[local_idx] = max(shared_max[local_idx], shared_max[local_idx + stride]);
            shared_min[local_idx] = min(shared_min[local_idx], shared_min[local_idx + stride]);
        }
        workgroupBarrier();
        stride /= 2u;
    }

    // Thread 0 computes scale exponent
    if (local_idx == 0u) {
        let max_val = shared_max[0];
        let min_val = shared_min[0];

        // Compute scale exponent: largest power of 2 that covers the range
        var scale_exp = 0i;
        if (max_val > 0.0f) {
            scale_exp = i32(ceil(log2(max_val))) - NAX_MAX_EXP;
        }

        block_info[block_idx] = NAXBlockInfo(
            scale_exp,
            0,
            min_val,
            max_val
        );
    }
}

// ============================================================================
// NAX Quantization Kernels
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE)
fn quantize_nax8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let quad_idx = gid.x;
    let elem_idx = quad_idx * 4u;

    if (elem_idx >= params.size) { return; }

    // Determine block
    let block_idx = elem_idx / params.block_size;
    let info = block_info[block_idx];

    // Load and quantize 4 values
    var nax_vals: array<u32, 4>;
    for (var i = 0u; i < 4u; i++) {
        if (elem_idx + i < params.size) {
            let val = input_f32[elem_idx + i];
            nax_vals[i] = encode_nax8(val, info.scale_exp);
        } else {
            nax_vals[i] = 0u;
        }
    }

    // Pack and store
    output_nax[quad_idx] = pack_nax8x4(nax_vals[0], nax_vals[1], nax_vals[2], nax_vals[3]);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn dequantize_nax8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let quad_idx = gid.x;
    let elem_idx = quad_idx * 4u;

    if (elem_idx >= params.size) { return; }

    // Determine block
    let block_idx = elem_idx / params.block_size;
    let info = block_info[block_idx];

    // Load packed NAX values
    let packed = input_nax[quad_idx];
    let nax_vals = unpack_nax8x4(packed);

    // Dequantize and store
    for (var i = 0u; i < 4u; i++) {
        if (elem_idx + i < params.size) {
            output_f32[elem_idx + i] = decode_nax8(nax_vals[i], info.scale_exp);
        }
    }
}

// ============================================================================
// NAX FP16 Variant (4-bit exponent + 4-bit mantissa per value)
// ============================================================================

// NAX16 format: More precision for critical layers
// [sign(1)][exp(4)][mantissa(11)] = 16 bits per value

fn encode_nax16(val: f32, scale_exp: i32) -> u32 {
    if (val == 0.0f) {
        return 0u;
    }

    let sign = select(0u, 1u, val < 0.0f);
    let abs_val = abs(val);
    let scaled = abs_val * pow(2.0f, f32(-scale_exp));

    let log2_val = log2(max(scaled, 1e-20f));
    var exp_val = i32(floor(log2_val));
    exp_val = clamp(exp_val, NAX_MIN_EXP, NAX_MAX_EXP);

    let mantissa_float = scaled / pow(2.0f, f32(exp_val)) - 1.0f;
    var mantissa = u32(round(mantissa_float * 2048.0f));  // 11 bits
    mantissa = min(mantissa, 2047u);

    let biased_exp = u32(exp_val - NAX_MIN_EXP);

    // Pack: [sign(1)][exp(4)][mantissa(11)]
    return (sign << 15u) | (biased_exp << 11u) | mantissa;
}

fn decode_nax16(nax: u32, scale_exp: i32) -> f32 {
    if (nax == 0u) {
        return 0.0f;
    }

    let sign = (nax >> 15u) & 1u;
    let biased_exp = (nax >> 11u) & 0xFu;
    let mantissa = nax & 0x7FFu;

    let exp_val = i32(biased_exp) + NAX_MIN_EXP;
    let mantissa_val = 1.0f + f32(mantissa) / 2048.0f;

    var val = mantissa_val * pow(2.0f, f32(exp_val));
    val = val * pow(2.0f, f32(scale_exp));

    return select(val, -val, sign == 1u);
}

fn pack_nax16x2(a: u32, b: u32) -> u32 {
    return (b << 16u) | (a & 0xFFFFu);
}

fn unpack_nax16x2(packed: u32) -> vec2<u32> {
    return vec2<u32>(packed & 0xFFFFu, packed >> 16u);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn quantize_nax16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    let elem_idx = pair_idx * 2u;

    if (elem_idx >= params.size) { return; }

    let block_idx = elem_idx / params.block_size;
    let info = block_info[block_idx];

    let val0 = input_f32[elem_idx];
    var val1 = 0.0f;
    if (elem_idx + 1u < params.size) {
        val1 = input_f32[elem_idx + 1u];
    }

    let nax0 = encode_nax16(val0, info.scale_exp);
    let nax1 = encode_nax16(val1, info.scale_exp);

    output_nax[pair_idx] = pack_nax16x2(nax0, nax1);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn dequantize_nax16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    let elem_idx = pair_idx * 2u;

    if (elem_idx >= params.size) { return; }

    let block_idx = elem_idx / params.block_size;
    let info = block_info[block_idx];

    let packed = input_nax[pair_idx];
    let nax_vals = unpack_nax16x2(packed);

    output_f32[elem_idx] = decode_nax16(nax_vals.x, info.scale_exp);
    if (elem_idx + 1u < params.size) {
        output_f32[elem_idx + 1u] = decode_nax16(nax_vals.y, info.scale_exp);
    }
}

// ============================================================================
// Adaptive Block-wise Quantization
// ============================================================================

// Dynamically choose NAX8 or NAX16 based on block variance
struct AdaptiveBlockInfo {
    scale_exp: i32,
    use_nax16: u32,      // 1 = use NAX16, 0 = use NAX8
    variance: f32,
    mean: f32,
}

@group(2) @binding(0) var<storage, read_write> adaptive_info: array<AdaptiveBlockInfo>;

var<workgroup> shared_sum: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_sq_sum: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn compute_adaptive_stats(@builtin(global_invocation_id) gid: vec3<u32>,
                          @builtin(local_invocation_id) lid: vec3<u32>,
                          @builtin(workgroup_id) wgid: vec3<u32>) {
    let block_idx = wgid.x;
    let local_idx = lid.x;

    if (block_idx >= params.num_blocks) { return; }

    let block_start = block_idx * params.block_size;
    let block_end = min(block_start + params.block_size, params.size);

    // Compute local sums for mean and variance
    var local_sum = 0.0f;
    var local_sq_sum = 0.0f;
    var local_max = 0.0f;
    var count = 0u;

    var i = block_start + local_idx;
    while (i < block_end) {
        let val = input_f32[i];
        local_sum += val;
        local_sq_sum += val * val;
        local_max = max(local_max, abs(val));
        count += 1u;
        i += WORKGROUP_SIZE;
    }

    shared_sum[local_idx] = local_sum;
    shared_sq_sum[local_idx] = local_sq_sum;
    shared_max[local_idx] = local_max;
    workgroupBarrier();

    // Parallel reduction
    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (local_idx < stride) {
            shared_sum[local_idx] += shared_sum[local_idx + stride];
            shared_sq_sum[local_idx] += shared_sq_sum[local_idx + stride];
            shared_max[local_idx] = max(shared_max[local_idx], shared_max[local_idx + stride]);
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (local_idx == 0u) {
        let n = f32(block_end - block_start);
        let mean = shared_sum[0] / n;
        let variance = (shared_sq_sum[0] / n) - (mean * mean);
        let max_val = shared_max[0];

        // Compute scale exponent
        var scale_exp = 0i;
        if (max_val > 0.0f) {
            scale_exp = i32(ceil(log2(max_val))) - NAX_MAX_EXP;
        }

        // Use NAX16 if variance is high (more precision needed)
        let coefficient_of_variation = sqrt(variance) / (abs(mean) + 1e-8f);
        let use_nax16 = select(0u, 1u, coefficient_of_variation > 0.5f);

        adaptive_info[block_idx] = AdaptiveBlockInfo(
            scale_exp,
            use_nax16,
            variance,
            mean
        );
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn quantize_adaptive(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.size) { return; }

    let block_idx = idx / params.block_size;
    let info = adaptive_info[block_idx];

    let val = input_f32[idx];

    if (info.use_nax16 == 1u) {
        // NAX16: store as packed pairs
        let pair_idx = idx / 2u;
        let is_second = (idx % 2u) == 1u;

        if (!is_second) {
            var val1 = 0.0f;
            if (idx + 1u < params.size && (idx + 1u) / params.block_size == block_idx) {
                val1 = input_f32[idx + 1u];
            }

            let nax0 = encode_nax16(val, info.scale_exp);
            let nax1 = encode_nax16(val1, info.scale_exp);
            output_nax[pair_idx] = pack_nax16x2(nax0, nax1);
        }
    } else {
        // NAX8: store as packed quads
        let quad_idx = idx / 4u;
        let pos_in_quad = idx % 4u;

        if (pos_in_quad == 0u) {
            var vals: array<u32, 4>;
            for (var i = 0u; i < 4u; i++) {
                if (idx + i < params.size && (idx + i) / params.block_size == block_idx) {
                    vals[i] = encode_nax8(input_f32[idx + i], info.scale_exp);
                } else {
                    vals[i] = 0u;
                }
            }
            output_nax[quad_idx] = pack_nax8x4(vals[0], vals[1], vals[2], vals[3]);
        }
    }
}

// ============================================================================
// NAX Matrix Multiply (for transformer layers)
// ============================================================================

struct MatMulParams {
    M: u32,
    N: u32,
    K: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(3) @binding(0) var<storage, read> mat_a_nax: array<u32>;  // Packed NAX
@group(3) @binding(1) var<storage, read> mat_b_nax: array<u32>;  // Packed NAX
@group(3) @binding(2) var<storage, read_write> mat_c: array<f32>;
@group(3) @binding(3) var<uniform> matmul_params: MatMulParams;
@group(3) @binding(4) var<storage, read> block_info_a: array<NAXBlockInfo>;
@group(3) @binding(5) var<storage, read> block_info_b: array<NAXBlockInfo>;

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<f32, 256>;  // 16x16
var<workgroup> tile_b: array<f32, 256>;  // 16x16

@compute @workgroup_size(16, 16)
fn matmul_nax8(@builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>,
               @builtin(workgroup_id) wgid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    let local_row = lid.y;
    let local_col = lid.x;

    var sum = 0.0f;

    let num_tiles = (matmul_params.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // Load tile A (row-major, NAX8 packed)
        let a_row = wgid.y * TILE_SIZE + local_row;
        let a_col = t * TILE_SIZE + local_col;

        if (a_row < matmul_params.M && a_col < matmul_params.K) {
            let flat_idx = a_row * matmul_params.lda + a_col;
            let quad_idx = flat_idx / 4u;
            let pos_in_quad = flat_idx % 4u;
            let block_idx = flat_idx / BLOCK_SIZE;

            let packed = mat_a_nax[quad_idx];
            let nax_vals = unpack_nax8x4(packed);
            tile_a[local_row * TILE_SIZE + local_col] = decode_nax8(nax_vals[pos_in_quad], block_info_a[block_idx].scale_exp);
        } else {
            tile_a[local_row * TILE_SIZE + local_col] = 0.0f;
        }

        // Load tile B (column-major conceptually, but row-major storage)
        let b_row = t * TILE_SIZE + local_row;
        let b_col = wgid.x * TILE_SIZE + local_col;

        if (b_row < matmul_params.K && b_col < matmul_params.N) {
            let flat_idx = b_row * matmul_params.ldb + b_col;
            let quad_idx = flat_idx / 4u;
            let pos_in_quad = flat_idx % 4u;
            let block_idx = flat_idx / BLOCK_SIZE;

            let packed = mat_b_nax[quad_idx];
            let nax_vals = unpack_nax8x4(packed);
            tile_b[local_row * TILE_SIZE + local_col] = decode_nax8(nax_vals[pos_in_quad], block_info_b[block_idx].scale_exp);
        } else {
            tile_b[local_row * TILE_SIZE + local_col] = 0.0f;
        }

        workgroupBarrier();

        // Compute partial dot product
        for (var k = 0u; k < TILE_SIZE; k++) {
            sum += tile_a[local_row * TILE_SIZE + k] * tile_b[k * TILE_SIZE + local_col];
        }

        workgroupBarrier();
    }

    // Store result
    if (row < matmul_params.M && col < matmul_params.N) {
        mat_c[row * matmul_params.ldc + col] = sum;
    }
}
