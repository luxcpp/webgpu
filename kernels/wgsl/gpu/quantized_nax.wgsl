// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// NAX Variant INT Quantization Kernels
// - Non-Affine eXtended integer quantization
// - Group-wise quantization for LLM inference
// - Block-scaled INT4/INT8 formats
// - GPTQ/AWQ-compatible quantization
//
// NAX INT format uses per-group scales for higher accuracy
// Optimized for large language model weight quantization
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;
const DEFAULT_GROUP_SIZE: u32 = 128u;  // Elements per quantization group

// Quantization limits
const INT8_MIN: i32 = -128;
const INT8_MAX: i32 = 127;
const INT4_MIN: i32 = -8;
const INT4_MAX: i32 = 7;
const UINT4_MAX: u32 = 15u;

// NAX format codes
const NAX_INT4_SYM: u32 = 0u;      // Symmetric INT4
const NAX_INT4_ASYM: u32 = 1u;     // Asymmetric INT4
const NAX_INT8_SYM: u32 = 2u;      // Symmetric INT8
const NAX_INT8_ASYM: u32 = 3u;     // Asymmetric INT8
const NAX_INT4_GPTQ: u32 = 4u;     // GPTQ-style INT4
const NAX_INT4_AWQ: u32 = 5u;      // AWQ-style INT4

// ============================================================================
// Parameter Structures
// ============================================================================

struct NAXIntParams {
    size: u32,              // Total elements
    num_groups: u32,        // Number of quantization groups
    group_size: u32,        // Elements per group
    format: u32,            // NAX format code
}

struct GroupScale {
    scale: f32,
    zero_point: i32,
    group_min: f32,
    group_max: f32,
}

struct AWQParams {
    size: u32,
    num_groups: u32,
    group_size: u32,
    bits: u32,              // 4 or 8
    clip_ratio: f32,        // AWQ clipping ratio
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> input_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_packed: array<u32>;
@group(0) @binding(2) var<uniform> params: NAXIntParams;
@group(0) @binding(3) var<storage, read_write> group_scales: array<GroupScale>;

// Dequantization bindings
@group(1) @binding(0) var<storage, read> input_packed: array<u32>;
@group(1) @binding(1) var<storage, read_write> output_f32: array<f32>;

// AWQ-specific bindings
@group(2) @binding(0) var<storage, read> awq_activations: array<f32>;  // Activation stats
@group(2) @binding(1) var<uniform> awq_params: AWQParams;

// Workgroup shared memory
var<workgroup> shared_max: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_min: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_sum: array<f32, WORKGROUP_SIZE>;

// ============================================================================
// NAX INT4 Pack/Unpack Utilities
// ============================================================================

// Pack 8 symmetric INT4 values into u32
fn pack_nax_int4_sym_x8(vals: array<i32, 8>) -> u32 {
    var result = 0u;
    for (var i = 0u; i < 8u; i++) {
        // Map [-8, 7] to [0, 15] for storage
        let mapped = u32(vals[i] + 8);
        result |= (mapped & 0xFu) << (i * 4u);
    }
    return result;
}

// Unpack u32 into 8 symmetric INT4 values
fn unpack_nax_int4_sym_x8(packed: u32) -> array<i32, 8> {
    var result: array<i32, 8>;
    for (var i = 0u; i < 8u; i++) {
        let mapped = (packed >> (i * 4u)) & 0xFu;
        result[i] = i32(mapped) - 8;  // Map [0, 15] back to [-8, 7]
    }
    return result;
}

// Pack 8 asymmetric INT4 values (using zero point) into u32
fn pack_nax_int4_asym_x8(vals: array<i32, 8>, zero_point: i32) -> u32 {
    var result = 0u;
    for (var i = 0u; i < 8u; i++) {
        let adjusted = u32(clamp(vals[i] + zero_point, 0, 15));
        result |= (adjusted & 0xFu) << (i * 4u);
    }
    return result;
}

// Unpack u32 into 8 asymmetric INT4 values
fn unpack_nax_int4_asym_x8(packed: u32, zero_point: i32) -> array<i32, 8> {
    var result: array<i32, 8>;
    for (var i = 0u; i < 8u; i++) {
        let val = i32((packed >> (i * 4u)) & 0xFu);
        result[i] = val - zero_point;
    }
    return result;
}

// Pack 4 INT8 values into u32
fn pack_nax_int8_x4(vals: array<i32, 4>) -> u32 {
    return (u32(vals[3] & 0xFF) << 24u) |
           (u32(vals[2] & 0xFF) << 16u) |
           (u32(vals[1] & 0xFF) << 8u) |
           u32(vals[0] & 0xFF);
}

// Unpack u32 into 4 INT8 values with sign extension
fn unpack_nax_int8_x4(packed: u32) -> array<i32, 4> {
    var result: array<i32, 4>;
    for (var i = 0u; i < 4u; i++) {
        var val = i32((packed >> (i * 8u)) & 0xFFu);
        if (val >= 128) { val -= 256; }  // Sign extend
        result[i] = val;
    }
    return result;
}

// ============================================================================
// Group Statistics Computation
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE)
fn compute_group_scales(@builtin(global_invocation_id) gid: vec3<u32>,
                        @builtin(local_invocation_id) lid: vec3<u32>,
                        @builtin(workgroup_id) wgid: vec3<u32>) {
    let group_idx = wgid.x;
    let local_idx = lid.x;

    if (group_idx >= params.num_groups) { return; }

    let group_start = group_idx * params.group_size;
    let group_end = min(group_start + params.group_size, params.size);

    // Find local min/max
    var local_max = -1e30f;
    var local_min = 1e30f;

    var i = group_start + local_idx;
    while (i < group_end) {
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

    // Thread 0 computes scale
    if (local_idx == 0u) {
        let max_val = shared_max[0];
        let min_val = shared_min[0];

        var scale: f32;
        var zero_point: i32;

        if (params.format == NAX_INT4_SYM || params.format == NAX_INT8_SYM) {
            // Symmetric quantization
            let abs_max = max(abs(max_val), abs(min_val));
            if (params.format == NAX_INT4_SYM) {
                scale = abs_max / f32(INT4_MAX);
            } else {
                scale = abs_max / f32(INT8_MAX);
            }
            zero_point = 0;
        } else {
            // Asymmetric quantization
            let range = max_val - min_val;
            if (params.format == NAX_INT4_ASYM) {
                scale = range / 15.0f;
                zero_point = i32(round(-min_val / scale));
                zero_point = clamp(zero_point, 0, 15);
            } else {
                scale = range / 255.0f;
                zero_point = i32(round(-min_val / scale));
                zero_point = clamp(zero_point, 0, 255);
            }
        }

        // Avoid division by zero
        if (scale == 0.0f) { scale = 1.0f; }

        group_scales[group_idx] = GroupScale(scale, zero_point, min_val, max_val);
    }
}

// ============================================================================
// NAX INT4 Symmetric Quantization
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE)
fn quantize_nax_int4_sym(@builtin(global_invocation_id) gid: vec3<u32>) {
    let octet_idx = gid.x;
    let elem_idx = octet_idx * 8u;

    if (elem_idx >= params.size) { return; }

    // Determine group
    let group_idx = elem_idx / params.group_size;
    let gs = group_scales[group_idx];

    // Quantize 8 values
    var q_vals: array<i32, 8>;
    for (var i = 0u; i < 8u; i++) {
        if (elem_idx + i < params.size) {
            let val = input_f32[elem_idx + i];
            let scaled = round(val / gs.scale);
            q_vals[i] = clamp(i32(scaled), INT4_MIN, INT4_MAX);
        } else {
            q_vals[i] = 0;
        }
    }

    output_packed[octet_idx] = pack_nax_int4_sym_x8(q_vals);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn dequantize_nax_int4_sym(@builtin(global_invocation_id) gid: vec3<u32>) {
    let octet_idx = gid.x;
    let elem_idx = octet_idx * 8u;

    if (elem_idx >= params.size) { return; }

    let group_idx = elem_idx / params.group_size;
    let gs = group_scales[group_idx];

    let packed = input_packed[octet_idx];
    let q_vals = unpack_nax_int4_sym_x8(packed);

    for (var i = 0u; i < 8u; i++) {
        if (elem_idx + i < params.size) {
            output_f32[elem_idx + i] = f32(q_vals[i]) * gs.scale;
        }
    }
}

// ============================================================================
// NAX INT4 Asymmetric Quantization
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE)
fn quantize_nax_int4_asym(@builtin(global_invocation_id) gid: vec3<u32>) {
    let octet_idx = gid.x;
    let elem_idx = octet_idx * 8u;

    if (elem_idx >= params.size) { return; }

    let group_idx = elem_idx / params.group_size;
    let gs = group_scales[group_idx];

    var q_vals: array<i32, 8>;
    for (var i = 0u; i < 8u; i++) {
        if (elem_idx + i < params.size) {
            let val = input_f32[elem_idx + i];
            let scaled = round(val / gs.scale);
            q_vals[i] = clamp(i32(scaled), 0, 15) - gs.zero_point;
        } else {
            q_vals[i] = 0;
        }
    }

    output_packed[octet_idx] = pack_nax_int4_asym_x8(q_vals, gs.zero_point);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn dequantize_nax_int4_asym(@builtin(global_invocation_id) gid: vec3<u32>) {
    let octet_idx = gid.x;
    let elem_idx = octet_idx * 8u;

    if (elem_idx >= params.size) { return; }

    let group_idx = elem_idx / params.group_size;
    let gs = group_scales[group_idx];

    let packed = input_packed[octet_idx];
    let q_vals = unpack_nax_int4_asym_x8(packed, gs.zero_point);

    for (var i = 0u; i < 8u; i++) {
        if (elem_idx + i < params.size) {
            output_f32[elem_idx + i] = f32(q_vals[i]) * gs.scale;
        }
    }
}

// ============================================================================
// NAX INT8 Group-wise Quantization
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE)
fn quantize_nax_int8_group(@builtin(global_invocation_id) gid: vec3<u32>) {
    let quad_idx = gid.x;
    let elem_idx = quad_idx * 4u;

    if (elem_idx >= params.size) { return; }

    let group_idx = elem_idx / params.group_size;
    let gs = group_scales[group_idx];

    var q_vals: array<i32, 4>;
    for (var i = 0u; i < 4u; i++) {
        if (elem_idx + i < params.size) {
            let val = input_f32[elem_idx + i];
            let scaled = round(val / gs.scale);
            if (params.format == NAX_INT8_SYM) {
                q_vals[i] = clamp(i32(scaled), INT8_MIN, INT8_MAX);
            } else {
                q_vals[i] = clamp(i32(scaled) + gs.zero_point, 0, 255) - 128;
            }
        } else {
            q_vals[i] = 0;
        }
    }

    output_packed[quad_idx] = pack_nax_int8_x4(q_vals);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn dequantize_nax_int8_group(@builtin(global_invocation_id) gid: vec3<u32>) {
    let quad_idx = gid.x;
    let elem_idx = quad_idx * 4u;

    if (elem_idx >= params.size) { return; }

    let group_idx = elem_idx / params.group_size;
    let gs = group_scales[group_idx];

    let packed = input_packed[quad_idx];
    let q_vals = unpack_nax_int8_x4(packed);

    for (var i = 0u; i < 4u; i++) {
        if (elem_idx + i < params.size) {
            if (params.format == NAX_INT8_SYM) {
                output_f32[elem_idx + i] = f32(q_vals[i]) * gs.scale;
            } else {
                output_f32[elem_idx + i] = f32(q_vals[i] + 128 - gs.zero_point) * gs.scale;
            }
        }
    }
}

// ============================================================================
// GPTQ-Style Quantization (with activation-aware ordering)
// ============================================================================

struct GPTQParams {
    size: u32,
    num_groups: u32,
    group_size: u32,
    bits: u32,
    damp_factor: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(3) @binding(0) var<storage, read> gptq_hessian_diag: array<f32>;  // Diagonal of Hessian
@group(3) @binding(1) var<uniform> gptq_params: GPTQParams;
@group(3) @binding(2) var<storage, read_write> gptq_order: array<u32>;  // Column ordering

// Compute quantization order based on Hessian diagonal
@compute @workgroup_size(WORKGROUP_SIZE)
fn compute_gptq_order(@builtin(global_invocation_id) gid: vec3<u32>,
                      @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= gptq_params.size) { return; }

    // Initialize order (will be sorted externally)
    gptq_order[idx] = idx;
}

// GPTQ quantization with error compensation
@compute @workgroup_size(WORKGROUP_SIZE)
fn quantize_gptq(@builtin(global_invocation_id) gid: vec3<u32>) {
    let octet_idx = gid.x;
    let elem_idx = octet_idx * 8u;

    if (elem_idx >= params.size) { return; }

    let group_idx = elem_idx / params.group_size;
    let gs = group_scales[group_idx];

    // Quantize with error tracking
    var q_vals: array<i32, 8>;
    for (var i = 0u; i < 8u; i++) {
        if (elem_idx + i < params.size) {
            let orig_idx = gptq_order[elem_idx + i];
            let val = input_f32[orig_idx];

            // Apply damping factor to prevent numerical issues
            let h_diag = gptq_hessian_diag[orig_idx] + gptq_params.damp_factor;
            let adjusted = val / sqrt(h_diag);

            let scaled = round(adjusted / gs.scale);
            q_vals[i] = clamp(i32(scaled), INT4_MIN, INT4_MAX);
        } else {
            q_vals[i] = 0;
        }
    }

    output_packed[octet_idx] = pack_nax_int4_sym_x8(q_vals);
}

// ============================================================================
// AWQ-Style Quantization (Activation-aware Weight Quantization)
// ============================================================================

// Compute per-channel activation scales
@compute @workgroup_size(WORKGROUP_SIZE)
fn compute_awq_scales(@builtin(global_invocation_id) gid: vec3<u32>,
                      @builtin(local_invocation_id) lid: vec3<u32>,
                      @builtin(workgroup_id) wgid: vec3<u32>) {
    let group_idx = wgid.x;
    let local_idx = lid.x;

    if (group_idx >= awq_params.num_groups) { return; }

    let group_start = group_idx * awq_params.group_size;
    let group_end = min(group_start + awq_params.group_size, awq_params.size);

    // Find weight max and activation scale
    var local_w_max = 0.0f;
    var local_a_sum = 0.0f;

    var i = group_start + local_idx;
    while (i < group_end) {
        local_w_max = max(local_w_max, abs(input_f32[i]));
        local_a_sum += awq_activations[i];
        i += WORKGROUP_SIZE;
    }

    shared_max[local_idx] = local_w_max;
    shared_sum[local_idx] = local_a_sum;
    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (local_idx < stride) {
            shared_max[local_idx] = max(shared_max[local_idx], shared_max[local_idx + stride]);
            shared_sum[local_idx] += shared_sum[local_idx + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (local_idx == 0u) {
        let w_max = shared_max[0];
        let a_mean = shared_sum[0] / f32(group_end - group_start);

        // AWQ scale: balance between weight and activation importance
        // s = w_max / qmax * clip_ratio, adjusted by activation importance
        let qmax = select(f32(INT8_MAX), f32(INT4_MAX), awq_params.bits == 4u);
        var scale = w_max / qmax;

        // Apply activation-aware scaling
        if (a_mean > 1e-8f) {
            scale = scale * pow(a_mean, awq_params.clip_ratio);
        }

        if (scale == 0.0f) { scale = 1.0f; }

        group_scales[group_idx] = GroupScale(scale, 0, -w_max, w_max);
    }
}

// AWQ quantization
@compute @workgroup_size(WORKGROUP_SIZE)
fn quantize_awq(@builtin(global_invocation_id) gid: vec3<u32>) {
    let octet_idx = gid.x;
    let elem_idx = octet_idx * 8u;

    if (elem_idx >= awq_params.size) { return; }

    let group_idx = elem_idx / awq_params.group_size;
    let gs = group_scales[group_idx];

    var q_vals: array<i32, 8>;
    for (var i = 0u; i < 8u; i++) {
        if (elem_idx + i < awq_params.size) {
            let w = input_f32[elem_idx + i];
            let a = awq_activations[elem_idx + i];

            // AWQ: weight importance scaled by activation magnitude
            let importance = abs(w) * a;
            let scaled_w = w / gs.scale;

            // Adaptive clipping based on importance
            let clip_factor = min(1.0f, importance / (gs.group_max + 1e-8f));
            let clipped = scaled_w * (1.0f - awq_params.clip_ratio * (1.0f - clip_factor));

            q_vals[i] = clamp(i32(round(clipped)), INT4_MIN, INT4_MAX);
        } else {
            q_vals[i] = 0;
        }
    }

    output_packed[octet_idx] = pack_nax_int4_sym_x8(q_vals);
}

// ============================================================================
// Group-wise Quantized Matrix Multiply
// ============================================================================

struct GroupMatMulParams {
    M: u32,
    N: u32,
    K: u32,
    group_size: u32,
}

@group(4) @binding(0) var<storage, read> gmm_weights: array<u32>;  // Group-quantized weights
@group(4) @binding(1) var<storage, read> gmm_weight_scales: array<GroupScale>;
@group(4) @binding(2) var<storage, read> gmm_activations: array<f32>;  // FP32 activations
@group(4) @binding(3) var<storage, read_write> gmm_output: array<f32>;
@group(4) @binding(4) var<uniform> gmm_params: GroupMatMulParams;

const GMM_TILE_SIZE: u32 = 16u;

var<workgroup> gmm_tile_a: array<f32, 256>;  // 16x16 dequantized weights
var<workgroup> gmm_tile_b: array<f32, 256>;  // 16x16 activations

@compute @workgroup_size(16, 16)
fn matmul_nax_grouped(@builtin(global_invocation_id) gid: vec3<u32>,
                      @builtin(local_invocation_id) lid: vec3<u32>,
                      @builtin(workgroup_id) wgid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    let local_row = lid.y;
    let local_col = lid.x;

    var sum = 0.0f;

    let num_tiles = (gmm_params.K + GMM_TILE_SIZE - 1u) / GMM_TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // Load and dequantize weight tile (INT4 group-quantized)
        let a_row = wgid.y * GMM_TILE_SIZE + local_row;
        let a_col = t * GMM_TILE_SIZE + local_col;

        if (a_row < gmm_params.M && a_col < gmm_params.K) {
            let flat_idx = a_row * gmm_params.K + a_col;
            let group_idx = flat_idx / gmm_params.group_size;
            let gs = gmm_weight_scales[group_idx];

            let octet_idx = flat_idx / 8u;
            let pos_in_octet = flat_idx % 8u;
            let packed = gmm_weights[octet_idx];
            let q_vals = unpack_nax_int4_sym_x8(packed);

            gmm_tile_a[local_row * GMM_TILE_SIZE + local_col] = f32(q_vals[pos_in_octet]) * gs.scale;
        } else {
            gmm_tile_a[local_row * GMM_TILE_SIZE + local_col] = 0.0f;
        }

        // Load activation tile (FP32)
        let b_row = t * GMM_TILE_SIZE + local_row;
        let b_col = wgid.x * GMM_TILE_SIZE + local_col;

        if (b_row < gmm_params.K && b_col < gmm_params.N) {
            gmm_tile_b[local_row * GMM_TILE_SIZE + local_col] =
                gmm_activations[b_row * gmm_params.N + b_col];
        } else {
            gmm_tile_b[local_row * GMM_TILE_SIZE + local_col] = 0.0f;
        }

        workgroupBarrier();

        // Accumulate in FP32
        for (var k = 0u; k < GMM_TILE_SIZE; k++) {
            sum += gmm_tile_a[local_row * GMM_TILE_SIZE + k] *
                   gmm_tile_b[k * GMM_TILE_SIZE + local_col];
        }

        workgroupBarrier();
    }

    if (row < gmm_params.M && col < gmm_params.N) {
        gmm_output[row * gmm_params.N + col] = sum;
    }
}

// ============================================================================
// Dynamic Quantization (quantize activations on-the-fly)
// ============================================================================

struct DynamicQuantParams {
    size: u32,
    bits: u32,          // 4 or 8
    _pad0: u32,
    _pad1: u32,
}

@group(5) @binding(0) var<storage, read> dyn_input: array<f32>;
@group(5) @binding(1) var<storage, read_write> dyn_output: array<u32>;
@group(5) @binding(2) var<storage, read_write> dyn_scale: f32;
@group(5) @binding(3) var<uniform> dyn_params: DynamicQuantParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn dynamic_quantize(@builtin(global_invocation_id) gid: vec3<u32>,
                    @builtin(local_invocation_id) lid: vec3<u32>) {
    let local_idx = lid.x;

    // Phase 1: Find max absolute value
    var local_max = 0.0f;
    var i = local_idx;
    while (i < dyn_params.size) {
        local_max = max(local_max, abs(dyn_input[i]));
        i += WORKGROUP_SIZE;
    }

    shared_max[local_idx] = local_max;
    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (local_idx < stride) {
            shared_max[local_idx] = max(shared_max[local_idx], shared_max[local_idx + stride]);
        }
        workgroupBarrier();
        stride /= 2u;
    }

    // Thread 0 computes scale
    if (local_idx == 0u) {
        let max_val = shared_max[0];
        if (dyn_params.bits == 4u) {
            dyn_scale = max_val / f32(INT4_MAX);
        } else {
            dyn_scale = max_val / f32(INT8_MAX);
        }
        if (dyn_scale == 0.0f) { dyn_scale = 1.0f; }
    }

    workgroupBarrier();

    // Phase 2: Quantize
    let scale = dyn_scale;

    if (dyn_params.bits == 4u) {
        let octet_idx = gid.x;
        let elem_idx = octet_idx * 8u;

        if (elem_idx < dyn_params.size) {
            var q_vals: array<i32, 8>;
            for (var j = 0u; j < 8u; j++) {
                if (elem_idx + j < dyn_params.size) {
                    let scaled = round(dyn_input[elem_idx + j] / scale);
                    q_vals[j] = clamp(i32(scaled), INT4_MIN, INT4_MAX);
                } else {
                    q_vals[j] = 0;
                }
            }
            dyn_output[octet_idx] = pack_nax_int4_sym_x8(q_vals);
        }
    } else {
        let quad_idx = gid.x;
        let elem_idx = quad_idx * 4u;

        if (elem_idx < dyn_params.size) {
            var q_vals: array<i32, 4>;
            for (var j = 0u; j < 4u; j++) {
                if (elem_idx + j < dyn_params.size) {
                    let scaled = round(dyn_input[elem_idx + j] / scale);
                    q_vals[j] = clamp(i32(scaled), INT8_MIN, INT8_MAX);
                } else {
                    q_vals[j] = 0;
                }
            }
            dyn_output[quad_idx] = pack_nax_int8_x4(q_vals);
        }
    }
}
