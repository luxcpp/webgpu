// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Twiddle Factor Generation and Caching for NTT/FFT
// Pre-computed roots of unity for efficient polynomial multiplication
// Supports multiple moduli for RNS (Residue Number System)

// ============================================================================
// Type Definitions
// ============================================================================

// 64-bit unsigned integer (emulated)
struct U64 {
    lo: u32,
    hi: u32,
}

// 128-bit for intermediate results
struct U128 {
    lo: U64,
    hi: U64,
}

// Twiddle factor parameters
struct TwiddleParams {
    N: u32,                  // Transform size (power of 2)
    log_N: u32,              // log2(N)
    modulus_idx: u32,        // Which modulus to use (for RNS)
    num_moduli: u32,         // Total number of moduli
    direction: u32,          // 0 = forward (NTT), 1 = inverse (INTT)
}

// RNS modulus with precomputed values
struct RnsModulus {
    q: U64,                  // The modulus
    root: U64,               // Primitive 2N-th root of unity
    root_inv: U64,           // Inverse of root
    n_inv: U64,              // 1/N mod q (for INTT scaling)
    barrett_mu: U64,         // Barrett reduction constant
}

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read_write> twiddles: array<U64>;       // Output twiddle factors
@group(0) @binding(1) var<storage, read_write> twiddles_inv: array<U64>;   // Inverse twiddles
@group(0) @binding(2) var<storage, read> moduli: array<RnsModulus>;        // RNS moduli parameters
@group(0) @binding(3) var<uniform> params: TwiddleParams;

// Workgroup shared memory
var<workgroup> shared_roots: array<U64, 256>;

// ============================================================================
// 64-bit Arithmetic Helpers
// ============================================================================

fn u64_zero() -> U64 {
    return U64(0u, 0u);
}

fn u64_one() -> U64 {
    return U64(1u, 0u);
}

fn u64_from_u32(x: u32) -> U64 {
    return U64(x, 0u);
}

fn u64_eq(a: U64, b: U64) -> bool {
    return a.lo == b.lo && a.hi == b.hi;
}

fn u64_lt(a: U64, b: U64) -> bool {
    if (a.hi < b.hi) { return true; }
    if (a.hi > b.hi) { return false; }
    return a.lo < b.lo;
}

fn u64_gte(a: U64, b: U64) -> bool {
    return !u64_lt(a, b);
}

fn u64_add(a: U64, b: U64) -> U64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    let hi = a.hi + b.hi + carry;
    return U64(lo, hi);
}

fn u64_sub(a: U64, b: U64) -> U64 {
    let borrow = select(0u, 1u, a.lo < b.lo);
    let lo = a.lo - b.lo;
    let hi = a.hi - b.hi - borrow;
    return U64(lo, hi);
}

// Multiply two 32-bit values to get 64-bit result
fn mul32_to_64(a: u32, b: u32) -> U64 {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    
    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;
    
    let mid = p1 + p2;
    let mid_carry = select(0u, 0x10000u, mid < p1);
    
    let lo = p0 + (mid << 16u);
    let carry1 = select(0u, 1u, lo < p0);
    
    let hi = p3 + (mid >> 16u) + mid_carry + carry1;
    
    return U64(lo, hi);
}

// Multiply U64 by U32, returning low 64 bits
fn u64_mul_u32_lo(a: U64, b: u32) -> U64 {
    let p0 = mul32_to_64(a.lo, b);
    let p1_lo = a.hi * b;
    return U64(p0.lo, p0.hi + p1_lo);
}

// Full U64 x U64 -> U128
fn u64_mul_full(a: U64, b: U64) -> U128 {
    let p0 = mul32_to_64(a.lo, b.lo);
    let p1 = mul32_to_64(a.lo, b.hi);
    let p2 = mul32_to_64(a.hi, b.lo);
    let p3 = mul32_to_64(a.hi, b.hi);
    
    // Combine: p0 + (p1 + p2) << 32 + p3 << 64
    let mid = u64_add(p1, p2);
    
    var lo = U64(p0.lo, p0.hi);
    lo = u64_add(lo, U64(0u, mid.lo));  // Add mid.lo to position 32-95
    
    var hi = p3;
    hi = u64_add(hi, U64(mid.hi, 0u));  // Add mid.hi to position 64-127
    hi = u64_add(hi, U64(select(0u, 1u, u64_lt(U64(0u, mid.lo), p1)), 0u));  // Carry from mid
    
    return U128(lo, hi);
}

// ============================================================================
// Modular Arithmetic
// ============================================================================

// Barrett reduction: x mod q using precomputed mu
fn barrett_reduce(x: U64, q: U64, mu: U64) -> U64 {
    // Approximate quotient: q_approx = (x * mu) >> 64
    let prod = u64_mul_full(x, mu);
    let q_approx = prod.hi;
    
    // r = x - q_approx * q
    let qaq = u64_mul_full(q_approx, q);
    var r = u64_sub(x, qaq.lo);
    
    // Correction step (at most 2 subtractions needed)
    if (u64_gte(r, q)) {
        r = u64_sub(r, q);
    }
    if (u64_gte(r, q)) {
        r = u64_sub(r, q);
    }
    
    return r;
}

// Modular multiplication
fn mod_mul(a: U64, b: U64, q: U64, mu: U64) -> U64 {
    let prod = u64_mul_full(a, b);
    // Need to reduce 128-bit product - simplified version
    return barrett_reduce(prod.lo, q, mu);
}

// Modular exponentiation by squaring
fn mod_pow(base: U64, exp: u32, q: U64, mu: U64) -> U64 {
    var result = u64_one();
    var b = base;
    var e = exp;
    
    while (e > 0u) {
        if ((e & 1u) == 1u) {
            result = mod_mul(result, b, q, mu);
        }
        b = mod_mul(b, b, q, mu);
        e = e >> 1u;
    }
    
    return result;
}

// ============================================================================
// Twiddle Generation Kernels
// ============================================================================

// Generate twiddle factors for a single modulus
// twiddle[i] = root^(bit_reverse(i)) for NTT
@compute @workgroup_size(256)
fn generate_twiddles(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let idx = gid.x;
    let N = params.N;
    
    if (idx >= N) { return; }
    
    let mod_idx = params.modulus_idx;
    let q = moduli[mod_idx].q;
    let mu = moduli[mod_idx].barrett_mu;
    
    let root = select(moduli[mod_idx].root, moduli[mod_idx].root_inv, params.direction == 1u);
    
    // Bit-reverse index for Cooley-Tukey ordering
    var br_idx = 0u;
    var temp = idx;
    for (var i = 0u; i < params.log_N; i++) {
        br_idx = (br_idx << 1u) | (temp & 1u);
        temp = temp >> 1u;
    }
    
    // Compute root^br_idx mod q
    let twiddle = mod_pow(root, br_idx, q, mu);
    
    let output_offset = mod_idx * N;
    if (params.direction == 0u) {
        twiddles[output_offset + idx] = twiddle;
    } else {
        twiddles_inv[output_offset + idx] = twiddle;
    }
}

// Generate twiddles for all NTT stages
// For stage s, we need root^(j * N/2^{s+1}) for j in [0, 2^s)
@compute @workgroup_size(256)
fn generate_stage_twiddles(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let stage = gid.y;
    let N = params.N;
    
    let half_size = 1u << stage;
    if (idx >= half_size) { return; }
    
    let mod_idx = params.modulus_idx;
    let q = moduli[mod_idx].q;
    let mu = moduli[mod_idx].barrett_mu;
    let root = moduli[mod_idx].root;
    
    // Exponent for this twiddle
    let exp = (idx * N) >> (stage + 1u);
    let twiddle = mod_pow(root, exp, q, mu);
    
    let output_offset = mod_idx * N + (1u << stage) - 1u + idx;
    twiddles[output_offset] = twiddle;
}

// Generate inverse twiddles with N^{-1} scaling factor
@compute @workgroup_size(256)
fn generate_inverse_twiddles(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let N = params.N;
    
    if (idx >= N) { return; }
    
    let mod_idx = params.modulus_idx;
    let q = moduli[mod_idx].q;
    let mu = moduli[mod_idx].barrett_mu;
    let root_inv = moduli[mod_idx].root_inv;
    let n_inv = moduli[mod_idx].n_inv;
    
    // Bit-reverse index
    var br_idx = 0u;
    var temp = idx;
    for (var i = 0u; i < params.log_N; i++) {
        br_idx = (br_idx << 1u) | (temp & 1u);
        temp = temp >> 1u;
    }
    
    // Compute root_inv^br_idx * n_inv mod q
    var twiddle = mod_pow(root_inv, br_idx, q, mu);
    twiddle = mod_mul(twiddle, n_inv, q, mu);
    
    let output_offset = mod_idx * N;
    twiddles_inv[output_offset + idx] = twiddle;
}

// ============================================================================
// Multi-Modulus Support (RNS)
// ============================================================================

// Generate twiddles for all moduli in parallel
@compute @workgroup_size(256)
fn generate_rns_twiddles(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let mod_idx = gid.y;
    let N = params.N;
    
    if (idx >= N || mod_idx >= params.num_moduli) { return; }
    
    let q = moduli[mod_idx].q;
    let mu = moduli[mod_idx].barrett_mu;
    let root = moduli[mod_idx].root;
    
    // Bit-reverse index
    var br_idx = 0u;
    var temp = idx;
    for (var i = 0u; i < params.log_N; i++) {
        br_idx = (br_idx << 1u) | (temp & 1u);
        temp = temp >> 1u;
    }
    
    let twiddle = mod_pow(root, br_idx, q, mu);
    
    let output_offset = mod_idx * N + idx;
    twiddles[output_offset] = twiddle;
}

// Precompute Barrett constants for a new modulus
@compute @workgroup_size(1)
fn compute_barrett_constant(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let mod_idx = gid.x;
    
    if (mod_idx >= params.num_moduli) { return; }
    
    // Barrett constant mu = floor(2^128 / q)
    // This is a placeholder - actual computation requires higher precision
    // In practice, these constants are precomputed on CPU
}
