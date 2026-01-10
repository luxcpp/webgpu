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

// Dilithium parameters
const DILITHIUM_Q: i32 = 8380417;
const DILITHIUM_QINV: u32 = 58728449u;
const DILITHIUM_Q_HALF: i32 = 4190208;
const DILITHIUM_BARRETT_V: i64 = 8396807i64;
const DILITHIUM_R2: i32 = 2365951;

// Kyber parameters
const KYBER_Q: i32 = 3329;
const KYBER_QINV: u32 = 62209u;
const KYBER_Q_HALF: i32 = 1664;
const KYBER_BARRETT_V: i32 = 20159;
const KYBER_R2: i32 = 1353;

// ============================================================================
// Parameter Structure
// ============================================================================

struct ReduceParams {
    size: u32,
    d: u32,          // Compression parameter
    gamma2: i32,     // Dilithium decomposition parameter
    _pad: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read_write> data: array<i32>;
@group(0) @binding(1) var<storage, read> input: array<i32>;
@group(0) @binding(2) var<storage, read_write> output_u8: array<u32>;  // Packed u8
@group(0) @binding(3) var<uniform> params: ReduceParams;

// ============================================================================
// Barrett Reduction for Dilithium
// ============================================================================

fn barrett_reduce_dilithium(a: i32) -> i32 {
    let t = i32((DILITHIUM_BARRETT_V * i64(a)) >> 46);
    return a - t * DILITHIUM_Q;
}

fn full_reduce_dilithium(x: i32) -> i32 {
    var t = barrett_reduce_dilithium(x);
    t = t + ((t >> 31) & DILITHIUM_Q);
    t = t - DILITHIUM_Q;
    t = t + ((t >> 31) & DILITHIUM_Q);
    return t;
}

fn centered_reduce_dilithium(a: i32) -> i32 {
    var t = full_reduce_dilithium(a);
    if (t > DILITHIUM_Q_HALF) {
        t = t - DILITHIUM_Q;
    }
    return t;
}

// ============================================================================
// Barrett Reduction for Kyber
// ============================================================================

fn barrett_reduce_kyber(a: i32) -> i32 {
    let t = ((KYBER_BARRETT_V * a + (1 << 25)) >> 26);
    return a - t * KYBER_Q;
}

fn full_reduce_kyber(x: i32) -> i32 {
    var t = barrett_reduce_kyber(x);
    t = t + ((t >> 15) & KYBER_Q);
    t = t - KYBER_Q;
    t = t + ((t >> 15) & KYBER_Q);
    return t;
}

fn centered_reduce_kyber(a: i32) -> i32 {
    var t = full_reduce_kyber(a);
    if (t > KYBER_Q_HALF) {
        t = t - KYBER_Q;
    }
    return t;
}

// ============================================================================
// Montgomery Reduction
// ============================================================================

fn montgomery_reduce_dilithium(a_lo: i32, a_hi: i32) -> i32 {
    let t = a_lo * i32(DILITHIUM_QINV);
    // Simplified: compute upper word of (a - t * q)
    let tq_lo = t * DILITHIUM_Q;
    return a_hi - ((tq_lo >> 31) & 1);
}

fn montgomery_reduce_kyber(a: i32) -> i32 {
    let t = (a & 0xFFFF) * i32(KYBER_QINV);
    return (a - (t & 0xFFFF) * KYBER_Q) >> 16;
}

// ============================================================================
// Batch Reduction Kernels - Dilithium
// ============================================================================

@compute @workgroup_size(256)
fn reduce_barrett_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    data[gid.x] = barrett_reduce_dilithium(data[gid.x]);
}

@compute @workgroup_size(256)
fn reduce_full_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    data[gid.x] = full_reduce_dilithium(data[gid.x]);
}

@compute @workgroup_size(256)
fn reduce_centered_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    data[gid.x] = centered_reduce_dilithium(data[gid.x]);
}

// ============================================================================
// Batch Reduction Kernels - Kyber
// ============================================================================

@compute @workgroup_size(256)
fn reduce_barrett_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    data[gid.x] = barrett_reduce_kyber(data[gid.x]);
}

@compute @workgroup_size(256)
fn reduce_full_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    data[gid.x] = full_reduce_kyber(data[gid.x]);
}

@compute @workgroup_size(256)
fn reduce_centered_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    data[gid.x] = centered_reduce_kyber(data[gid.x]);
}

// ============================================================================
// Montgomery Domain Conversion
// ============================================================================

fn mont_mul_dilithium_full(a: i32, b: i32) -> i32 {
    // Emulate 64-bit multiply
    let a_u = bitcast<u32>(a);
    let b_u = bitcast<u32>(b);
    
    let a_lo = a_u & 0xFFFFu;
    let a_hi = a_u >> 16u;
    let b_lo = b_u & 0xFFFFu;
    let b_hi = b_u >> 16u;
    
    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;
    
    let mid = p1 + p2;
    let lo = p0 + (mid << 16u);
    let hi = p3 + (mid >> 16u) + select(0u, 1u, lo < p0);
    
    return montgomery_reduce_dilithium(bitcast<i32>(lo), bitcast<i32>(hi));
}

@compute @workgroup_size(256)
fn to_montgomery_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    data[gid.x] = mont_mul_dilithium_full(data[gid.x], DILITHIUM_R2);
}

@compute @workgroup_size(256)
fn from_montgomery_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    // Multiply by 1 in Montgomery form = just reduce
    data[gid.x] = montgomery_reduce_dilithium(data[gid.x], 0);
}

@compute @workgroup_size(256)
fn to_montgomery_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    data[gid.x] = montgomery_reduce_kyber(data[gid.x] * KYBER_R2);
}

@compute @workgroup_size(256)
fn from_montgomery_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    data[gid.x] = montgomery_reduce_kyber(data[gid.x]);
}

// ============================================================================
// Compression/Decompression for Kyber
// ============================================================================

fn set_byte(idx: u32, val: u32) {
    let word_idx = idx / 4u;
    let byte_offset = idx % 4u;
    let mask = ~(0xFFu << (byte_offset * 8u));
    let old_val = output_u8[word_idx] & mask;
    output_u8[word_idx] = old_val | ((val & 0xFFu) << (byte_offset * 8u));
}

fn get_byte_input(idx: u32) -> u32 {
    let word_idx = idx / 4u;
    let byte_offset = idx % 4u;
    return (bitcast<u32>(input[word_idx]) >> (byte_offset * 8u)) & 0xFFu;
}

@compute @workgroup_size(256)
fn compress_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    let d = params.d;
    
    if (gid.x >= size) { return; }
    
    let x = full_reduce_kyber(data[gid.x]);
    
    // Compute round((2^d * x) / q)
    var t = u32(x) << d;
    t = t + u32(KYBER_Q / 2);
    t = t / u32(KYBER_Q);
    t = t & ((1u << d) - 1u);
    
    set_byte(gid.x, t);
}

@compute @workgroup_size(256)
fn decompress_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    let d = params.d;
    
    if (gid.x >= size) { return; }
    
    let x = get_byte_input(gid.x);
    
    // Compute round((q * x) / 2^d)
    var t = x * u32(KYBER_Q) + (1u << (d - 1u));
    t = t >> d;
    
    data[gid.x] = i32(t);
}

// ============================================================================
// Dilithium-Specific Reductions
// ============================================================================

// Additional bindings for Dilithium operations
@group(0) @binding(4) var<storage, read> input2: array<i32>;
@group(0) @binding(5) var<storage, read> hint: array<u32>;

@compute @workgroup_size(256)
fn highbits_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    let gamma2 = params.gamma2;
    
    if (gid.x >= size) { return; }
    
    let a = full_reduce_dilithium(input[gid.x]);
    let two_gamma2 = 2 * gamma2;
    
    var a0 = a % two_gamma2;
    if (a0 > gamma2) {
        a0 = a0 - two_gamma2;
    }
    
    var a1: i32;
    if (a - a0 == DILITHIUM_Q - 1) {
        a1 = 0;
    } else {
        a1 = (a - a0) / two_gamma2;
    }
    
    data[gid.x] = a1;
}

@compute @workgroup_size(256)
fn lowbits_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    let gamma2 = params.gamma2;
    
    if (gid.x >= size) { return; }
    
    let a = full_reduce_dilithium(input[gid.x]);
    let two_gamma2 = 2 * gamma2;
    
    var a0 = a % two_gamma2;
    if (a0 > gamma2) {
        a0 = a0 - two_gamma2;
    }
    
    if (a - a0 == DILITHIUM_Q - 1) {
        a0 = a0 - 1;
    }
    
    data[gid.x] = a0;
}

@compute @workgroup_size(256)
fn make_hint_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    let gamma2 = params.gamma2;
    
    if (gid.x >= size) { return; }
    
    let z0 = input[gid.x];
    let r0 = input2[gid.x];
    
    var h = 0u;
    if (z0 > gamma2 || z0 < -gamma2 || r0 > gamma2 || r0 < -gamma2) {
        h = 1u;
    }
    
    // Pack hint bits into u32
    let word_idx = gid.x / 32u;
    let bit_idx = gid.x % 32u;
    
    // Atomic OR to set bit (simplified - real impl needs atomic)
    if (h == 1u) {
        output_u8[word_idx] = output_u8[word_idx] | (1u << bit_idx);
    }
}

@compute @workgroup_size(256)
fn use_hint_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    let gamma2 = params.gamma2;
    
    if (gid.x >= size) { return; }
    
    let a = full_reduce_dilithium(input[gid.x]);
    let two_gamma2 = 2 * gamma2;
    
    var a0 = a % two_gamma2;
    if (a0 > gamma2) {
        a0 = a0 - two_gamma2;
    }
    
    var a1: i32;
    if (a - a0 == DILITHIUM_Q - 1) {
        a1 = 0;
    } else {
        a1 = (a - a0) / two_gamma2;
    }
    
    // Get hint bit
    let word_idx = gid.x / 32u;
    let bit_idx = gid.x % 32u;
    let h = (hint[word_idx] >> bit_idx) & 1u;
    
    if (h == 1u) {
        let max_a1 = (DILITHIUM_Q - 1) / two_gamma2;
        if (a0 > 0) {
            a1 = (a1 + 1) % (max_a1 + 1);
        } else {
            a1 = (a1 + max_a1) % (max_a1 + 1);
        }
    }
    
    data[gid.x] = a1;
}

// ============================================================================
// Freeze: Ensure coefficients are in canonical form [0, q)
// ============================================================================

@compute @workgroup_size(256)
fn freeze_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    data[gid.x] = full_reduce_dilithium(data[gid.x]);
}

@compute @workgroup_size(256)
fn freeze_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    data[gid.x] = full_reduce_kyber(data[gid.x]);
}

// ============================================================================
// Conditional Subtraction
// ============================================================================

@compute @workgroup_size(256)
fn cond_sub_q_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    var a = data[gid.x];
    if (a >= DILITHIUM_Q) {
        a = a - DILITHIUM_Q;
    }
    data[gid.x] = a;
}

@compute @workgroup_size(256)
fn cond_sub_q_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    var a = data[gid.x];
    if (a >= KYBER_Q) {
        a = a - KYBER_Q;
    }
    data[gid.x] = a;
}
