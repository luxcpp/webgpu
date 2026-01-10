// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Poseidon2 Hash Function over BN254 Scalar Field (Fr)
// GPU-accelerated Poseidon2 hash for ZK circuits and Merkle trees
//
// BN254 Scalar Field:
//   r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
//
// Poseidon2 Parameters (matching gnark-crypto):
//   - S-box: x^5
//   - State width: 3 (for 2-to-1 hash)
//   - Full rounds: 8 (4 beginning + 4 end)
//   - Partial rounds: 56

// ============================================================================
// BN254 Scalar Field (Fr) - 256-bit Arithmetic (8 x 32-bit limbs)
// ============================================================================

struct Fr256 {
    l0: u32, l1: u32, l2: u32, l3: u32,
    l4: u32, l5: u32, l6: u32, l7: u32,
}

// BN254 scalar field modulus r
const BN254_R = Fr256(
    0xF0000001u, 0x43E1F593u, 0x79B97091u, 0x2833E848u,
    0x8181585Du, 0xB85045B6u, 0xE131A029u, 0x30644E72u
);

// Montgomery R mod r (for Montgomery form of 1)
const FR_ONE = Fr256(
    0x4FFFFFFFBu, 0xAC96341Cu, 0x9F60CD29u, 0x36FC7695u,
    0x7879462Eu, 0x666EA36Fu, 0x9A07DF2Fu, 0x0E0A77C1u
);

// Montgomery constant: -r^{-1} mod 2^32
const BN254_R_INV: u32 = 0xEFFFFFFFu;

// ============================================================================
// Basic Arithmetic
// ============================================================================

fn fr_zero() -> Fr256 {
    return Fr256(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
}

fn fr_one() -> Fr256 {
    return FR_ONE;
}

fn fr_is_zero(a: Fr256) -> bool {
    return (a.l0 | a.l1 | a.l2 | a.l3 | a.l4 | a.l5 | a.l6 | a.l7) == 0u;
}

fn adc32(a: u32, b: u32, carry_in: u32) -> vec2<u32> {
    let sum = a + b + carry_in;
    let carry = select(0u, 1u, sum < a || (carry_in != 0u && sum <= a));
    return vec2<u32>(sum, carry);
}

fn sbb32(a: u32, b: u32, borrow_in: u32) -> vec2<u32> {
    let diff = a - b - borrow_in;
    let borrow = select(0u, 1u, a < b + borrow_in);
    return vec2<u32>(diff, borrow);
}

fn fr_add_raw(a: Fr256, b: Fr256) -> vec2<Fr256> {
    var r: array<u32, 8>;
    var carry = 0u;

    let t0 = adc32(a.l0, b.l0, carry); r[0] = t0.x; carry = t0.y;
    let t1 = adc32(a.l1, b.l1, carry); r[1] = t1.x; carry = t1.y;
    let t2 = adc32(a.l2, b.l2, carry); r[2] = t2.x; carry = t2.y;
    let t3 = adc32(a.l3, b.l3, carry); r[3] = t3.x; carry = t3.y;
    let t4 = adc32(a.l4, b.l4, carry); r[4] = t4.x; carry = t4.y;
    let t5 = adc32(a.l5, b.l5, carry); r[5] = t5.x; carry = t5.y;
    let t6 = adc32(a.l6, b.l6, carry); r[6] = t6.x; carry = t6.y;
    let t7 = adc32(a.l7, b.l7, carry); r[7] = t7.x; carry = t7.y;

    let result = Fr256(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
    let carry_fp = Fr256(carry, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    return vec2<Fr256>(result, carry_fp);
}

fn fr_sub_raw(a: Fr256, b: Fr256) -> vec2<Fr256> {
    var r: array<u32, 8>;
    var borrow = 0u;

    let t0 = sbb32(a.l0, b.l0, borrow); r[0] = t0.x; borrow = t0.y;
    let t1 = sbb32(a.l1, b.l1, borrow); r[1] = t1.x; borrow = t1.y;
    let t2 = sbb32(a.l2, b.l2, borrow); r[2] = t2.x; borrow = t2.y;
    let t3 = sbb32(a.l3, b.l3, borrow); r[3] = t3.x; borrow = t3.y;
    let t4 = sbb32(a.l4, b.l4, borrow); r[4] = t4.x; borrow = t4.y;
    let t5 = sbb32(a.l5, b.l5, borrow); r[5] = t5.x; borrow = t5.y;
    let t6 = sbb32(a.l6, b.l6, borrow); r[6] = t6.x; borrow = t6.y;
    let t7 = sbb32(a.l7, b.l7, borrow); r[7] = t7.x; borrow = t7.y;

    let result = Fr256(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
    let borrow_fp = Fr256(borrow, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    return vec2<Fr256>(result, borrow_fp);
}

fn fr_add(a: Fr256, b: Fr256) -> Fr256 {
    let sum_result = fr_add_raw(a, b);
    let sum = sum_result.x;
    let carry = sum_result.y.l0;

    let sub_result = fr_sub_raw(sum, BN254_R);
    let reduced = sub_result.x;
    let borrow = sub_result.y.l0;

    if (carry != 0u || borrow == 0u) {
        return reduced;
    }
    return sum;
}

fn fr_sub(a: Fr256, b: Fr256) -> Fr256 {
    let sub_result = fr_sub_raw(a, b);
    let diff = sub_result.x;
    let borrow = sub_result.y.l0;

    if (borrow != 0u) {
        return fr_add_raw(diff, BN254_R).x;
    }
    return diff;
}

fn fr_neg(a: Fr256) -> Fr256 {
    if (fr_is_zero(a)) {
        return a;
    }
    return fr_sub(BN254_R, a);
}

fn mul32(a: u32, b: u32) -> vec2<u32> {
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

    return vec2<u32>(lo, hi);
}

fn mac32(acc: vec2<u32>, a: u32, b: u32, carry: u32) -> vec3<u32> {
    let prod = mul32(a, b);

    var sum = acc.x + prod.x;
    var c1 = select(0u, 1u, sum < acc.x);

    sum = sum + carry;
    c1 = c1 + select(0u, 1u, sum < carry);

    let hi = acc.y + prod.y + c1;
    let c2 = select(0u, 1u, hi < acc.y);

    return vec3<u32>(sum, hi, c2);
}

fn mul256x256(a: Fr256, b: Fr256) -> array<u32, 16> {
    var t: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) { t[i] = 0u; }

    let a_arr = array<u32, 8>(a.l0, a.l1, a.l2, a.l3, a.l4, a.l5, a.l6, a.l7);
    let b_arr = array<u32, 8>(b.l0, b.l1, b.l2, b.l3, b.l4, b.l5, b.l6, b.l7);

    for (var i = 0u; i < 8u; i++) {
        var carry = 0u;
        for (var j = 0u; j < 8u; j++) {
            let acc = vec2<u32>(t[i + j], t[i + j + 1u]);
            let r = mac32(acc, a_arr[i], b_arr[j], carry);
            t[i + j] = r.x;
            t[i + j + 1u] = r.y;
            carry = r.z;
        }
    }

    return t;
}

fn fr_mont_reduce(t: array<u32, 16>) -> Fr256 {
    var temp = t;
    let r_arr = array<u32, 8>(
        BN254_R.l0, BN254_R.l1, BN254_R.l2, BN254_R.l3,
        BN254_R.l4, BN254_R.l5, BN254_R.l6, BN254_R.l7
    );

    for (var i = 0u; i < 8u; i++) {
        let m = temp[i] * BN254_R_INV;
        var carry = 0u;

        for (var j = 0u; j < 8u; j++) {
            let prod = mul32(m, r_arr[j]);
            var sum = temp[i + j] + prod.x + carry;
            carry = prod.y + select(0u, 1u, sum < prod.x);
            temp[i + j] = sum;
        }

        for (var j = i + 8u; j < 16u; j++) {
            let sum = temp[j] + carry;
            carry = select(0u, 1u, sum < carry);
            temp[j] = sum;
            if (carry == 0u) { break; }
        }
    }

    var result = Fr256(temp[8], temp[9], temp[10], temp[11],
                       temp[12], temp[13], temp[14], temp[15]);

    let sub_result = fr_sub_raw(result, BN254_R);
    if (sub_result.y.l0 == 0u) {
        result = sub_result.x;
    }

    return result;
}

fn fr_mul(a: Fr256, b: Fr256) -> Fr256 {
    let product = mul256x256(a, b);
    return fr_mont_reduce(product);
}

fn fr_square(a: Fr256) -> Fr256 {
    return fr_mul(a, a);
}

// ============================================================================
// Poseidon2 S-box: x^5
// ============================================================================

fn poseidon2_sbox(x: Fr256) -> Fr256 {
    let x2 = fr_square(x);     // x^2
    let x4 = fr_square(x2);    // x^4
    return fr_mul(x4, x);      // x^5
}

// ============================================================================
// Poseidon2 Parameters
// ============================================================================

const POSEIDON2_WIDTH: u32 = 3u;
const POSEIDON2_FULL_ROUNDS: u32 = 8u;
const POSEIDON2_PARTIAL_ROUNDS: u32 = 56u;

// Round constants (first few - actual impl would load from buffer)
fn get_round_constant(idx: u32) -> Fr256 {
    // Placeholder round constants - real impl loads from precomputed buffer
    let hash_base = idx * 0x12345678u;
    return Fr256(
        hash_base ^ 0x2A4F2C3Du, hash_base ^ 0x5E6F7890u,
        hash_base ^ 0x12345678u, hash_base ^ 0x90ABCDEFu,
        hash_base ^ 0xFEDCBA09u, hash_base ^ 0x87654321u,
        hash_base ^ 0x0ABCDEF1u, hash_base ^ 0x23456789u
    );
}

// ============================================================================
// MDS Matrix Multiplication (3x3)
// ============================================================================

fn poseidon2_mds(state: array<Fr256, 3>) -> array<Fr256, 3> {
    // Simple MDS: [2,1,1; 1,2,1; 1,1,2] (Hadamard-like)
    let sum = fr_add(fr_add(state[0], state[1]), state[2]);

    var result: array<Fr256, 3>;
    result[0] = fr_add(state[0], sum);
    result[1] = fr_add(state[1], sum);
    result[2] = fr_add(state[2], sum);

    return result;
}

// Internal linear layer for partial rounds
fn poseidon2_internal_linear(state: array<Fr256, 3>) -> array<Fr256, 3> {
    let sum = fr_add(fr_add(state[0], state[1]), state[2]);

    var result: array<Fr256, 3>;
    result[0] = fr_add(state[0], sum);
    result[1] = fr_add(state[1], sum);
    result[2] = fr_add(state[2], sum);

    return result;
}

// ============================================================================
// Poseidon2 Permutation
// ============================================================================

fn poseidon2_permutation(state_in: array<Fr256, 3>) -> array<Fr256, 3> {
    var state = state_in;
    var rc_idx = 0u;

    // Beginning full rounds (4 rounds)
    for (var r = 0u; r < POSEIDON2_FULL_ROUNDS / 2u; r++) {
        // Add round constants
        for (var i = 0u; i < POSEIDON2_WIDTH; i++) {
            let rc = get_round_constant(rc_idx);
            state[i] = fr_add(state[i], rc);
            rc_idx++;
        }

        // S-box on all elements
        for (var i = 0u; i < POSEIDON2_WIDTH; i++) {
            state[i] = poseidon2_sbox(state[i]);
        }

        // MDS matrix
        state = poseidon2_mds(state);
    }

    // Partial rounds (56 rounds)
    for (var r = 0u; r < POSEIDON2_PARTIAL_ROUNDS; r++) {
        // Add round constant only to first element
        let rc = get_round_constant(rc_idx);
        state[0] = fr_add(state[0], rc);
        rc_idx++;

        // S-box only on first element
        state[0] = poseidon2_sbox(state[0]);

        // Internal linear layer
        state = poseidon2_internal_linear(state);
    }

    // Ending full rounds (4 rounds)
    for (var r = 0u; r < POSEIDON2_FULL_ROUNDS / 2u; r++) {
        // Add round constants
        for (var i = 0u; i < POSEIDON2_WIDTH; i++) {
            let rc = get_round_constant(rc_idx);
            state[i] = fr_add(state[i], rc);
            rc_idx++;
        }

        // S-box on all elements
        for (var i = 0u; i < POSEIDON2_WIDTH; i++) {
            state[i] = poseidon2_sbox(state[i]);
        }

        // MDS matrix
        state = poseidon2_mds(state);
    }

    return state;
}

// ============================================================================
// Bindings
// ============================================================================

struct Poseidon2Params {
    count: u32,
    path_len: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> left_inputs: array<Fr256>;
@group(0) @binding(1) var<storage, read> right_inputs: array<Fr256>;
@group(0) @binding(2) var<storage, read_write> outputs: array<Fr256>;
@group(0) @binding(3) var<uniform> params: Poseidon2Params;

// ============================================================================
// Hash Kernels
// ============================================================================

// Hash pair for Merkle tree (2-to-1 compression)
@compute @workgroup_size(256)
fn poseidon2_hash_pair(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    var state: array<Fr256, 3>;
    state[0] = left_inputs[idx];
    state[1] = right_inputs[idx];
    state[2] = fr_zero();  // Domain separation

    state = poseidon2_permutation(state);

    outputs[idx] = state[0];
}

// Build one layer of Merkle tree
@compute @workgroup_size(256)
fn poseidon2_merkle_layer(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= params.count / 2u) {
        return;
    }

    var state: array<Fr256, 3>;
    state[0] = left_inputs[2u * idx];
    state[1] = left_inputs[2u * idx + 1u];
    state[2] = fr_zero();

    state = poseidon2_permutation(state);

    outputs[idx] = state[0];
}

// Merkle proof verification
@group(0) @binding(4) var<storage, read> leaves: array<Fr256>;
@group(0) @binding(5) var<storage, read> path: array<Fr256>;
@group(0) @binding(6) var<storage, read> path_indices: array<u32>;
@group(0) @binding(7) var<storage, read> expected_roots: array<Fr256>;
@group(0) @binding(8) var<storage, read_write> verify_results: array<u32>;

@compute @workgroup_size(256)
fn poseidon2_verify_merkle_proof(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let proof_idx = gid.x;
    if (proof_idx >= params.count) {
        return;
    }

    var current = leaves[proof_idx];

    for (var i = 0u; i < params.path_len; i++) {
        let sibling = path[proof_idx * params.path_len + i];
        let idx = path_indices[proof_idx * params.path_len + i];

        var state: array<Fr256, 3>;
        if (idx == 0u) {
            state[0] = current;
            state[1] = sibling;
        } else {
            state[0] = sibling;
            state[1] = current;
        }
        state[2] = fr_zero();

        state = poseidon2_permutation(state);
        current = state[0];
    }

    // Compare with expected root
    let expected = expected_roots[proof_idx];
    let valid = (current.l0 == expected.l0) && (current.l1 == expected.l1) &&
                (current.l2 == expected.l2) && (current.l3 == expected.l3) &&
                (current.l4 == expected.l4) && (current.l5 == expected.l5) &&
                (current.l6 == expected.l6) && (current.l7 == expected.l7);

    verify_results[proof_idx] = select(0u, 1u, valid);
}

// Compute nullifier: Poseidon2(nullifier_key, note_commitment, leaf_index)
@group(0) @binding(9) var<storage, read> nullifier_keys: array<Fr256>;
@group(0) @binding(10) var<storage, read> note_commitments: array<Fr256>;
@group(0) @binding(11) var<storage, read> leaf_indices: array<Fr256>;
@group(0) @binding(12) var<storage, read_write> nullifiers: array<Fr256>;

@compute @workgroup_size(256)
fn poseidon2_nullifier(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    var state: array<Fr256, 3>;
    state[0] = nullifier_keys[idx];
    state[1] = note_commitments[idx];
    state[2] = leaf_indices[idx];

    state = poseidon2_permutation(state);

    nullifiers[idx] = state[0];
}

// Compute commitment: Poseidon2(value, blinding_factor, salt)
@group(0) @binding(13) var<storage, read> values: array<Fr256>;
@group(0) @binding(14) var<storage, read> blindings: array<Fr256>;
@group(0) @binding(15) var<storage, read> salts: array<Fr256>;

@compute @workgroup_size(256)
fn poseidon2_commitment(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    var state: array<Fr256, 3>;
    state[0] = values[idx];
    state[1] = blindings[idx];
    state[2] = salts[idx];

    state = poseidon2_permutation(state);

    outputs[idx] = state[0];
}
