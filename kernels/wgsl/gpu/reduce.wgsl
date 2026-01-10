// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Modular Reduction Kernels for ML-DSA (Dilithium) and ML-KEM (Kyber)
// - Barrett reduction for q=8380417 and q=3329
// - Centered reduction to [-q/2, q/2]
// - Batch coefficient reduction
// - Montgomery conversion
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// Dilithium parameters
const DILITHIUM_Q: i32 = 8380417;
const DILITHIUM_QINV: u32 = 58728449u;    // q^(-1) mod 2^32
const DILITHIUM_Q_HALF: i32 = 4190208;    // (q-1)/2
const DILITHIUM_BARRETT_V: i32 = 8396807; // floor(2^46 / q)
const DILITHIUM_R2: i32 = 2365951;        // 2^64 mod q (Montgomery R^2)

// Kyber parameters
const KYBER_Q: i32 = 3329;
const KYBER_QINV: u32 = 62209u;           // q^(-1) mod 2^16
const KYBER_Q_HALF: i32 = 1664;           // (q-1)/2
const KYBER_BARRETT_V: i32 = 20159;       // floor(2^26 / q) + 1
const KYBER_R2: i32 = 1353;               // 2^32 mod q (Montgomery R^2)

// ============================================================================
// Parameter Structures
// ============================================================================

struct ReduceParams {
    size: u32,
    gamma2: i32,  // For Dilithium high/low bits
    d: u32,       // Compression parameter for Kyber
    _pad: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read_write> data_i32: array<i32>;
@group(0) @binding(1) var<uniform> params: ReduceParams;

// For copy operations
@group(0) @binding(2) var<storage, read> input_i32: array<i32>;
@group(0) @binding(3) var<storage, read_write> output_i32: array<i32>;

// For 16-bit Kyber operations (packed as i32 pairs)
@group(0) @binding(4) var<storage, read_write> data_i16: array<i32>;
@group(0) @binding(5) var<storage, read> input_i16: array<i32>;
@group(0) @binding(6) var<storage, read_write> output_i16: array<i32>;

// For hint operations
@group(0) @binding(7) var<storage, read> z_data: array<i32>;
@group(0) @binding(8) var<storage, read> r_data: array<i32>;
@group(0) @binding(9) var<storage, read_write> hint_data: array<u32>;

// For compression
@group(0) @binding(10) var<storage, read_write> output_u8: array<u32>;  // Packed bytes

// ============================================================================
// Barrett Reduction for Dilithium (q = 8380417)
// ============================================================================

// Barrett reduction: reduce a to [0, q)
// Input: a in (-2^31, 2^31)
// Output: a mod q in [0, q) approximately
fn barrett_reduce_dilithium(a: i32) -> i32 {
    // t = floor((BARRETT_V * a) / 2^23) - approximate for 32-bit
    let t = (DILITHIUM_BARRETT_V * a) >> 23;
    return a - t * DILITHIUM_Q;
}

// Full reduction ensuring output in [0, q)
fn full_reduce_dilithium(a: i32) -> i32 {
    var t = barrett_reduce_dilithium(a);
    // If negative, add q
    t = t + select(0, DILITHIUM_Q, t < 0);
    t = t - DILITHIUM_Q;
    t = t + select(0, DILITHIUM_Q, t < 0);
    return t;
}

// Centered reduction: reduce to [-q/2, q/2]
fn centered_reduce_dilithium(a: i32) -> i32 {
    let t = full_reduce_dilithium(a);
    return t - select(0, DILITHIUM_Q, t > DILITHIUM_Q_HALF);
}

// ============================================================================
// Barrett Reduction for Kyber (q = 3329)
// ============================================================================

// Barrett reduction: reduce a to approximately [0, q)
fn barrett_reduce_kyber(a: i32) -> i32 {
    let t = (KYBER_BARRETT_V * a + (1 << 25)) >> 26;
    return a - t * KYBER_Q;
}

// Full reduction ensuring output in [0, q)
fn full_reduce_kyber(a: i32) -> i32 {
    var t = barrett_reduce_kyber(a);
    t = t + select(0, KYBER_Q, t < 0);
    t = t - KYBER_Q;
    t = t + select(0, KYBER_Q, t < 0);
    return t;
}

// Centered reduction: reduce to [-(q-1)/2, q/2]
fn centered_reduce_kyber(a: i32) -> i32 {
    let t = full_reduce_kyber(a);
    return t - select(0, KYBER_Q, t > KYBER_Q_HALF);
}

// ============================================================================
// Montgomery Reduction
// ============================================================================

// Montgomery reduction for Dilithium
// Simplified for 32-bit operations
fn montgomery_reduce_dilithium(a: i32) -> i32 {
    let t = a * i32(DILITHIUM_QINV);
    return (a - t * DILITHIUM_Q) >> 16;
}

// Montgomery reduction for Kyber
fn montgomery_reduce_kyber(a: i32) -> i32 {
    let t = (a & 0xFFFF) * i32(KYBER_QINV) & 0xFFFF;
    return (a - t * KYBER_Q) >> 16;
}

// ============================================================================
// Batch Reduction Kernels - Dilithium
// ============================================================================

// Barrett reduction batch
@compute @workgroup_size(256)
fn reduce_barrett_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }
    data_i32[gid.x] = barrett_reduce_dilithium(data_i32[gid.x]);
}

// Full reduction batch (ensure [0, q))
@compute @workgroup_size(256)
fn reduce_full_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }
    data_i32[gid.x] = full_reduce_dilithium(data_i32[gid.x]);
}

// Centered reduction batch (to [-q/2, q/2])
@compute @workgroup_size(256)
fn reduce_centered_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }
    data_i32[gid.x] = centered_reduce_dilithium(data_i32[gid.x]);
}

// ============================================================================
// Batch Reduction Kernels - Kyber
// ============================================================================

// Extract i16 from packed i32 array
fn get_i16(arr_idx: u32, sub_idx: u32) -> i32 {
    let packed = data_i16[arr_idx];
    if (sub_idx == 0u) {
        return (packed << 16) >> 16;  // Sign extend lower 16 bits
    } else {
        return packed >> 16;  // Sign extend upper 16 bits
    }
}

// Set i16 in packed i32 array
fn set_i16(arr_idx: u32, sub_idx: u32, val: i32) {
    let packed = data_i16[arr_idx];
    if (sub_idx == 0u) {
        data_i16[arr_idx] = (packed & i32(0xFFFF0000u)) | (val & 0xFFFF);
    } else {
        data_i16[arr_idx] = (packed & 0x0000FFFF) | ((val & 0xFFFF) << 16);
    }
}

// Barrett reduction batch for Kyber (16-bit coefficients)
@compute @workgroup_size(256)
fn reduce_barrett_kyber(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }

    let arr_idx = gid.x / 2u;
    let sub_idx = gid.x % 2u;
    let val = get_i16(arr_idx, sub_idx);
    set_i16(arr_idx, sub_idx, barrett_reduce_kyber(val));
}

// Full reduction batch for Kyber
@compute @workgroup_size(256)
fn reduce_full_kyber(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }

    let arr_idx = gid.x / 2u;
    let sub_idx = gid.x % 2u;
    let val = get_i16(arr_idx, sub_idx);
    set_i16(arr_idx, sub_idx, full_reduce_kyber(val));
}

// Centered reduction batch for Kyber
@compute @workgroup_size(256)
fn reduce_centered_kyber(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }

    let arr_idx = gid.x / 2u;
    let sub_idx = gid.x % 2u;
    let val = get_i16(arr_idx, sub_idx);
    set_i16(arr_idx, sub_idx, centered_reduce_kyber(val));
}

// ============================================================================
// Montgomery Domain Conversion
// ============================================================================

// Convert to Montgomery domain: a -> a*R mod q
@compute @workgroup_size(256)
fn to_montgomery_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }
    // Multiply by R^2 then reduce to get a*R
    let a = data_i32[gid.x];
    let product = a * DILITHIUM_R2;
    data_i32[gid.x] = montgomery_reduce_dilithium(product);
}

// Convert from Montgomery domain: a*R -> a mod q
@compute @workgroup_size(256)
fn from_montgomery_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }
    data_i32[gid.x] = montgomery_reduce_dilithium(data_i32[gid.x]);
}

// Convert to Montgomery domain for Kyber
@compute @workgroup_size(256)
fn to_montgomery_kyber(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }

    let arr_idx = gid.x / 2u;
    let sub_idx = gid.x % 2u;
    let val = get_i16(arr_idx, sub_idx);
    set_i16(arr_idx, sub_idx, montgomery_reduce_kyber(val * KYBER_R2));
}

// Convert from Montgomery domain for Kyber
@compute @workgroup_size(256)
fn from_montgomery_kyber(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }

    let arr_idx = gid.x / 2u;
    let sub_idx = gid.x % 2u;
    let val = get_i16(arr_idx, sub_idx);
    set_i16(arr_idx, sub_idx, montgomery_reduce_kyber(val));
}

// ============================================================================
// Compression/Decompression for Kyber
// ============================================================================

// Compress: round(2^d / q * x) mod 2^d
@compute @workgroup_size(256)
fn compress_kyber(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }

    let arr_idx = gid.x / 2u;
    let sub_idx = gid.x % 2u;
    let x = full_reduce_kyber(get_i16(arr_idx, sub_idx));

    let d = params.d;

    // Compute round((2^d * x) / q)
    var t = u32(x) << d;
    t = t + u32(KYBER_Q / 2);  // For rounding
    t = t / u32(KYBER_Q);
    t = t & ((1u << d) - 1u);

    // Store compressed byte (pack 4 bytes per u32)
    let byte_idx = gid.x / 4u;
    let byte_offset = (gid.x % 4u) * 8u;
    let mask = ~(0xFFu << byte_offset);

    output_u8[byte_idx] = (output_u8[byte_idx] & mask) | ((t & 0xFFu) << byte_offset);
}

// Decompress: round(q / 2^d * x)
@compute @workgroup_size(256)
fn decompress_kyber(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }

    let d = params.d;

    // Read compressed byte
    let byte_idx = gid.x / 4u;
    let byte_offset = (gid.x % 4u) * 8u;
    let x = (output_u8[byte_idx] >> byte_offset) & 0xFFu;

    // Compute round((q * x) / 2^d)
    var t = x * u32(KYBER_Q) + (1u << (d - 1u));
    t = t >> d;

    let arr_idx = gid.x / 2u;
    let sub_idx = gid.x % 2u;
    set_i16(arr_idx, sub_idx, i32(t));
}

// ============================================================================
// Dilithium-Specific Reductions
// ============================================================================

// HighBits: extract high d bits after dividing by 2*gamma2
@compute @workgroup_size(256)
fn highbits_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }

    let a = full_reduce_dilithium(input_i32[gid.x]);
    let gamma2 = params.gamma2;
    let two_gamma2 = 2 * gamma2;

    // a0 = a mod 2*gamma2 (centered)
    var a0 = a % two_gamma2;
    if (a0 > gamma2) { a0 = a0 - two_gamma2; }

    // a1 = (a - a0) / (2*gamma2)
    // Handle edge case where a0 = q - 1
    if (a - a0 == DILITHIUM_Q - 1) {
        output_i32[gid.x] = 0;
    } else {
        output_i32[gid.x] = (a - a0) / two_gamma2;
    }
}

// LowBits: extract low bits after modular reduction
@compute @workgroup_size(256)
fn lowbits_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }

    let a = full_reduce_dilithium(input_i32[gid.x]);
    let gamma2 = params.gamma2;
    let two_gamma2 = 2 * gamma2;

    // a0 = a mod 2*gamma2 (centered in [-gamma2, gamma2])
    var a0 = a % two_gamma2;
    if (a0 > gamma2) { a0 = a0 - two_gamma2; }

    // Handle edge case
    if (a - a0 == DILITHIUM_Q - 1) {
        a0 = a0 - 1;
    }

    output_i32[gid.x] = a0;
}

// MakeHint: compute hint bit for verification
@compute @workgroup_size(256)
fn make_hint_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }

    let z0 = z_data[gid.x];
    let r0 = r_data[gid.x];
    let gamma2 = params.gamma2;

    // Hint is 1 if HighBits differ
    let h = (z0 > gamma2) || (z0 < -gamma2) || (r0 > gamma2) || (r0 < -gamma2);

    // Pack hints into u32 (32 hints per word)
    let word_idx = gid.x / 32u;
    let bit_idx = gid.x % 32u;

    if (h) {
        hint_data[word_idx] = hint_data[word_idx] | (1u << bit_idx);
    }
}

// UseHint: recover high bits using hint
@compute @workgroup_size(256)
fn use_hint_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }

    let a = full_reduce_dilithium(input_i32[gid.x]);
    let gamma2 = params.gamma2;
    let two_gamma2 = 2 * gamma2;

    var a0 = a % two_gamma2;
    if (a0 > gamma2) { a0 = a0 - two_gamma2; }

    var a1: i32;
    if (a - a0 == DILITHIUM_Q - 1) {
        a1 = 0;
    } else {
        a1 = (a - a0) / two_gamma2;
    }

    // Read hint bit
    let word_idx = gid.x / 32u;
    let bit_idx = gid.x % 32u;
    let hint = (hint_data[word_idx] >> bit_idx) & 1u;

    // Apply hint
    if (hint != 0u) {
        let max_a1 = (DILITHIUM_Q - 1) / two_gamma2;
        if (a0 > 0) {
            a1 = (a1 + 1) % (max_a1 + 1);
        } else {
            a1 = (a1 + max_a1) % (max_a1 + 1);
        }
    }

    output_i32[gid.x] = a1;
}

// ============================================================================
// Freeze: Ensure coefficients are in canonical form [0, q)
// ============================================================================

@compute @workgroup_size(256)
fn freeze_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }
    data_i32[gid.x] = full_reduce_dilithium(data_i32[gid.x]);
}

@compute @workgroup_size(256)
fn freeze_kyber(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }

    let arr_idx = gid.x / 2u;
    let sub_idx = gid.x % 2u;
    let val = get_i16(arr_idx, sub_idx);
    set_i16(arr_idx, sub_idx, full_reduce_kyber(val));
}

// ============================================================================
// Conditional Subtraction (for lazy reduction)
// ============================================================================

// Subtract q if coefficient >= q (lazy reduction cleanup)
@compute @workgroup_size(256)
fn cond_sub_q_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }
    var a = data_i32[gid.x];
    a = a - select(0, DILITHIUM_Q, a >= DILITHIUM_Q);
    data_i32[gid.x] = a;
}

@compute @workgroup_size(256)
fn cond_sub_q_kyber(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.size) { return; }

    let arr_idx = gid.x / 2u;
    let sub_idx = gid.x % 2u;
    var a = get_i16(arr_idx, sub_idx);
    a = a - select(0, KYBER_Q, a >= KYBER_Q);
    set_i16(arr_idx, sub_idx, a);
}

// ============================================================================
// Vectorized Reductions (process 4 elements at once)
// ============================================================================

@compute @workgroup_size(256)
fn reduce_barrett_dilithium_vec4(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let base_idx = gid.x * 4u;
    if (base_idx >= params.size) { return; }

    let remaining = min(4u, params.size - base_idx);

    for (var i = 0u; i < remaining; i++) {
        data_i32[base_idx + i] = barrett_reduce_dilithium(data_i32[base_idx + i]);
    }
}

@compute @workgroup_size(256)
fn reduce_full_dilithium_vec4(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let base_idx = gid.x * 4u;
    if (base_idx >= params.size) { return; }

    let remaining = min(4u, params.size - base_idx);

    for (var i = 0u; i < remaining; i++) {
        data_i32[base_idx + i] = full_reduce_dilithium(data_i32[base_idx + i]);
    }
}
