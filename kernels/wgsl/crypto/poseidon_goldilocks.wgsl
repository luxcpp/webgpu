// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Poseidon Hash WebGPU Kernel for Goldilocks Field
// p = 2^64 - 2^32 + 1

// ============================================================================
// Configuration
// ============================================================================

const POSEIDON_WIDTH: u32 = 8u;
const POSEIDON_FULL_ROUNDS: u32 = 8u;
const POSEIDON_PARTIAL_ROUNDS: u32 = 22u;
const POSEIDON_RATE: u32 = 7u;

// Goldilocks prime components
const GL_P_LO: u32 = 0x00000001u;  // Lower 32 bits of 2^64 - 2^32 + 1
const GL_P_HI: u32 = 0xFFFFFFFFu;  // Upper 32 bits

// ============================================================================
// 64-bit Emulation for Goldilocks
// ============================================================================

struct U64 {
    lo: u32,
    hi: u32,
}

fn u64_zero() -> U64 { return U64(0u, 0u); }

fn u64_from_u32(v: u32) -> U64 { return U64(v, 0u); }

fn u64_eq(a: U64, b: U64) -> bool {
    return a.lo == b.lo && a.hi == b.hi;
}

fn u64_lt(a: U64, b: U64) -> bool {
    if (a.hi != b.hi) { return a.hi < b.hi; }
    return a.lo < b.lo;
}

fn u64_ge(a: U64, b: U64) -> bool { return !u64_lt(a, b); }

// Addition with carry tracking
fn u64_add_with_carry(a: U64, b: U64) -> vec3<u32> {
    let lo = a.lo + b.lo;
    let carry1 = select(0u, 1u, lo < a.lo);
    let hi = a.hi + b.hi + carry1;
    let carry2 = select(0u, 1u, hi < a.hi || (carry1 != 0u && hi <= a.hi));
    return vec3<u32>(lo, hi, carry2);
}

fn u64_add(a: U64, b: U64) -> U64 {
    let r = u64_add_with_carry(a, b);
    return U64(r.x, r.y);
}

fn u64_sub(a: U64, b: U64) -> U64 {
    let borrow = select(0u, 1u, a.lo < b.lo);
    let lo = a.lo - b.lo;
    let hi = a.hi - b.hi - borrow;
    return U64(lo, hi);
}

// 32x32 -> 64 multiply
fn mul32_wide(a: u32, b: u32) -> U64 {
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

    let lo = p0 + ((mid & 0xFFFFu) << 16u);
    let lo_carry = select(0u, 1u, lo < p0);

    let hi = p3 + (mid >> 16u) + mid_carry + lo_carry;

    return U64(lo, hi);
}

// 64x64 -> 128 (returns low 64 and high 64)
fn u64_mul_full(a: U64, b: U64) -> array<U64, 2> {
    let p0 = mul32_wide(a.lo, b.lo);
    let p1 = mul32_wide(a.lo, b.hi);
    let p2 = mul32_wide(a.hi, b.lo);
    let p3 = mul32_wide(a.hi, b.hi);

    // Combine: result = p0 + (p1 + p2) << 32 + p3 << 64
    var mid = u64_add(p1, p2);
    let mid_carry = select(0u, 1u, u64_lt(mid, p1));

    // Add mid.lo << 32 to p0
    var lo = U64(p0.lo, p0.hi + mid.lo);
    let lo_carry = select(0u, 1u, lo.hi < p0.hi);

    // High 64 bits
    var hi = u64_add(p3, U64(mid.hi, mid_carry));
    hi = u64_add(hi, u64_from_u32(lo_carry));

    return array<U64, 2>(lo, hi);
}

fn u64_mul(a: U64, b: U64) -> U64 {
    return u64_mul_full(a, b)[0];
}

// ============================================================================
// Goldilocks Field Arithmetic
// p = 2^64 - 2^32 + 1
// ============================================================================

const GL_P: U64 = U64(0x00000001u, 0xFFFFFFFFu);

// Reduce 128-bit value mod p
// Using: 2^64 ≡ 2^32 - 1 (mod p)
fn gl_reduce(lo: U64, hi: U64) -> U64 {
    // hi * 2^64 + lo ≡ hi * (2^32 - 1) + lo (mod p)
    // = lo + hi * 2^32 - hi

    // hi << 32
    let hi_shifted = U64(0u, hi.lo);

    // lo + hi_shifted
    var result = u64_add(lo, hi_shifted);

    // - hi (with possible underflow)
    if (u64_ge(result, hi)) {
        result = u64_sub(result, hi);
    } else {
        // Underflow: add p
        result = u64_add(u64_sub(result, hi), GL_P);
    }

    // Final reduction if >= p
    if (u64_ge(result, GL_P)) {
        result = u64_sub(result, GL_P);
    }

    return result;
}

fn gl_add(a: U64, b: U64) -> U64 {
    let r = u64_add_with_carry(a, b);
    var result = U64(r.x, r.y);

    // Reduce if overflow or >= p
    if (r.z != 0u || u64_ge(result, GL_P)) {
        result = u64_sub(result, GL_P);
    }

    return result;
}

fn gl_sub(a: U64, b: U64) -> U64 {
    if (u64_ge(a, b)) {
        return u64_sub(a, b);
    }
    return u64_sub(u64_add(a, GL_P), b);
}

fn gl_mul(a: U64, b: U64) -> U64 {
    let full = u64_mul_full(a, b);
    return gl_reduce(full[0], full[1]);
}

// x^7 S-box
fn gl_sbox(x: U64) -> U64 {
    let x2 = gl_mul(x, x);
    let x4 = gl_mul(x2, x2);
    let x3 = gl_mul(x2, x);
    return gl_mul(x4, x3);
}

// ============================================================================
// MDS Matrix (8x8)
// ============================================================================

fn poseidon_mds(state: ptr<function, array<U64, 8>>) {
    var new_state: array<U64, 8>;

    // Row 0: all 1s
    new_state[0] = u64_zero();
    for (var j = 0u; j < 8u; j++) { new_state[0] = gl_add(new_state[0], (*state)[j]); }

    // Row 1: 1,2,3,4,5,6,7,8
    new_state[1] = u64_zero();
    for (var j = 0u; j < 8u; j++) {
        new_state[1] = gl_add(new_state[1], gl_mul(u64_from_u32(j + 1u), (*state)[j]));
    }

    // Row 2: 1,4,9,16,25,36,49,64
    new_state[2] = u64_zero();
    for (var j = 0u; j < 8u; j++) {
        let coeff = (j + 1u) * (j + 1u);
        new_state[2] = gl_add(new_state[2], gl_mul(u64_from_u32(coeff), (*state)[j]));
    }

    // Rows 3-7: powers pattern (simplified)
    for (var i = 3u; i < 8u; i++) {
        new_state[i] = u64_zero();
        for (var j = 0u; j < 8u; j++) {
            // MDS[i][j] = (j+1)^i
            var coeff = 1u;
            for (var k = 0u; k < i; k++) { coeff = coeff * (j + 1u); }
            new_state[i] = gl_add(new_state[i], gl_mul(u64_from_u32(coeff), (*state)[j]));
        }
    }

    *state = new_state;
}

// ============================================================================
// Poseidon Permutation
// ============================================================================

fn poseidon_full_round(state: ptr<function, array<U64, 8>>, rc: array<U64, 8>) {
    // Add round constants
    for (var i = 0u; i < 8u; i++) {
        (*state)[i] = gl_add((*state)[i], rc[i]);
    }

    // S-box on all elements
    for (var i = 0u; i < 8u; i++) {
        (*state)[i] = gl_sbox((*state)[i]);
    }

    // MDS matrix
    poseidon_mds(state);
}

fn poseidon_partial_round(state: ptr<function, array<U64, 8>>, rc: U64) {
    // Add round constant to first element
    (*state)[0] = gl_add((*state)[0], rc);

    // S-box only on first element
    (*state)[0] = gl_sbox((*state)[0]);

    // MDS matrix
    poseidon_mds(state);
}

fn poseidon_permutation(state: ptr<function, array<U64, 8>>) {
    // Simplified: using zero round constants (actual would load from buffer)
    let zero_rc = array<U64, 8>(
        u64_zero(), u64_zero(), u64_zero(), u64_zero(),
        u64_zero(), u64_zero(), u64_zero(), u64_zero()
    );

    // First 4 full rounds
    for (var i = 0u; i < 4u; i++) {
        poseidon_full_round(state, zero_rc);
    }

    // 22 partial rounds
    for (var i = 0u; i < 22u; i++) {
        poseidon_partial_round(state, u64_zero());
    }

    // Last 4 full rounds
    for (var i = 0u; i < 4u; i++) {
        poseidon_full_round(state, zero_rc);
    }
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> left: array<U64>;
@group(0) @binding(1) var<storage, read> right: array<U64>;
@group(0) @binding(2) var<storage, read_write> output: array<U64>;
@group(0) @binding(3) var<uniform> count: u32;

// ============================================================================
// Hash Kernels
// ============================================================================

@compute @workgroup_size(256)
fn poseidon_hash_pair(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= count) { return; }

    var state: array<U64, 8>;
    for (var i = 0u; i < 8u; i++) { state[i] = u64_zero(); }

    state[0] = left[idx];
    state[1] = right[idx];

    poseidon_permutation(&state);

    output[idx] = state[0];
}

@compute @workgroup_size(256)
fn poseidon_merkle_layer(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let num_pairs = count / 2u;
    if (idx >= num_pairs) { return; }

    var state: array<U64, 8>;
    for (var i = 0u; i < 8u; i++) { state[i] = u64_zero(); }

    // Read pair from input (reusing left buffer for input)
    state[0] = left[idx * 2u];
    state[1] = left[idx * 2u + 1u];

    poseidon_permutation(&state);

    output[idx] = state[0];
}

// Sponge hash for arbitrary length input
@group(1) @binding(0) var<storage, read> input_data: array<U64>;
@group(1) @binding(1) var<uniform> input_len: u32;
@group(1) @binding(2) var<uniform> batch_size: u32;

@compute @workgroup_size(256)
fn poseidon_hash_batch(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_idx = gid.x;
    if (batch_idx >= batch_size) { return; }

    var state: array<U64, 8>;
    for (var i = 0u; i < 8u; i++) { state[i] = u64_zero(); }

    let base = batch_idx * input_len;
    var absorbed = 0u;

    // Sponge absorb
    while (absorbed < input_len) {
        for (var i = 0u; i < 7u && absorbed < input_len; i++) {
            state[i] = gl_add(state[i], input_data[base + absorbed]);
            absorbed++;
        }
        poseidon_permutation(&state);
    }

    // Squeeze
    output[batch_idx] = state[0];
}
