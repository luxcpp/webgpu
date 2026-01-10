// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Ecosystem
//
// Merkle tree operations using Poseidon2 hash for WebGPU/WGSL
// This is the portable fallback when CUDA/Metal are unavailable.

// =============================================================================
// BN254 Field Parameters (same as poseidon2.wgsl)
// =============================================================================

const FR_MODULUS: array<u32, 8> = array<u32, 8>(
    0x43e1f593u, 0x79b97091u, 0x2833e848u, 0x8181585du,
    0xb85045b6u, 0xe131a029u, 0x64774b84u, 0x30644e72u
);

// =============================================================================
// Fr256 Type
// =============================================================================

struct Fr256 {
    limbs: array<u32, 8>
}

// =============================================================================
// Storage Buffers for Merkle Layer
// =============================================================================

@group(0) @binding(0) var<storage, read> nodes: array<Fr256>;
@group(0) @binding(1) var<storage, read_write> parents: array<Fr256>;

// Uniforms for tree parameters
struct MerkleParams {
    node_count: u32,
    _padding: array<u32, 3>
}
@group(0) @binding(2) var<uniform> params: MerkleParams;

// =============================================================================
// Field Arithmetic (duplicated for standalone compilation)
// =============================================================================

fn add_with_carry(a: u32, b: u32, carry_in: u32) -> vec2<u32> {
    let a64 = u32(a);
    let b64 = u32(b);
    let c64 = u32(carry_in);
    let sum_lo = a + b + carry_in;
    var carry_out = 0u;
    if (sum_lo < a || (carry_in > 0u && sum_lo <= a)) {
        carry_out = 1u;
    }
    return vec2<u32>(sum_lo, carry_out);
}

fn fr_add(a: Fr256, b: Fr256) -> Fr256 {
    var result: Fr256;
    var carry: u32 = 0u;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let sum = add_with_carry(a.limbs[i], b.limbs[i], carry);
        result.limbs[i] = sum.x;
        carry = sum.y;
    }

    return fr_reduce(result);
}

fn fr_reduce(a: Fr256) -> Fr256 {
    var ge_mod = true;
    for (var i = 7; i >= 0; i = i - 1) {
        let idx = u32(i);
        if (a.limbs[idx] < FR_MODULUS[idx]) {
            ge_mod = false;
            break;
        }
        if (a.limbs[idx] > FR_MODULUS[idx]) {
            break;
        }
    }

    if (!ge_mod) {
        return a;
    }

    var result: Fr256;
    var borrow: u32 = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let diff = i32(a.limbs[i]) - i32(FR_MODULUS[i]) - i32(borrow);
        if (diff < 0) {
            result.limbs[i] = u32(diff + 0x100000000);
            borrow = 1u;
        } else {
            result.limbs[i] = u32(diff);
            borrow = 0u;
        }
    }

    return result;
}

fn fr_square(a: Fr256) -> Fr256 {
    return fr_mul(a, a);
}

fn mul_u32(a: u32, b: u32) -> vec2<u32> {
    // Approximate 32x32->64 multiply using 16-bit splits
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;

    let mid = lh + hl;
    let lo = ll + ((mid & 0xFFFFu) << 16u);
    var hi = hh + (mid >> 16u);
    if (lo < ll) { hi = hi + 1u; }
    if (mid < lh) { hi = hi + 0x10000u; }

    return vec2<u32>(lo, hi);
}

fn fr_mul(a: Fr256, b: Fr256) -> Fr256 {
    var result: Fr256;
    for (var i = 0u; i < 8u; i = i + 1u) {
        result.limbs[i] = 0u;
    }

    for (var i = 0u; i < 8u; i = i + 1u) {
        var carry: u32 = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let idx = i + j;
            if (idx < 8u) {
                let prod = mul_u32(a.limbs[i], b.limbs[j]);
                let sum1 = add_with_carry(result.limbs[idx], prod.x, carry);
                let sum2 = add_with_carry(sum1.x, 0u, prod.y);
                result.limbs[idx] = sum2.x;
                carry = sum1.y + sum2.y;
            }
        }
    }

    return fr_reduce(result);
}

// =============================================================================
// Poseidon2 (simplified inline version)
// =============================================================================

const POSEIDON2_ROUNDS_F: u32 = 8u;
const POSEIDON2_ROUNDS_P: u32 = 56u;

fn sbox(x: Fr256) -> Fr256 {
    let x2 = fr_square(x);
    let x4 = fr_square(x2);
    return fr_mul(x4, x);
}

fn mds_multiply(state: array<Fr256, 3>) -> array<Fr256, 3> {
    var result: array<Fr256, 3>;
    let s0 = state[0];
    let s1 = state[1];
    let s2 = state[2];
    let sum = fr_add(fr_add(s0, s1), s2);
    result[0] = fr_add(sum, s0);
    result[1] = fr_add(sum, s1);
    result[2] = fr_add(sum, s2);
    return result;
}

fn poseidon2_hash(left: Fr256, right: Fr256) -> Fr256 {
    var state: array<Fr256, 3>;
    state[0] = left;
    state[1] = right;
    for (var i = 0u; i < 8u; i = i + 1u) {
        state[2].limbs[i] = 0u;
    }

    // Full rounds (first half)
    for (var r = 0u; r < POSEIDON2_ROUNDS_F / 2u; r = r + 1u) {
        for (var i = 0u; i < 3u; i = i + 1u) {
            state[i].limbs[0] = state[i].limbs[0] ^ (r * 3u + i);
        }
        state[0] = sbox(state[0]);
        state[1] = sbox(state[1]);
        state[2] = sbox(state[2]);
        state = mds_multiply(state);
    }

    // Partial rounds
    for (var r = 0u; r < POSEIDON2_ROUNDS_P; r = r + 1u) {
        state[0].limbs[0] = state[0].limbs[0] ^ (r + 100u);
        state[0] = sbox(state[0]);
        state = mds_multiply(state);
    }

    // Full rounds (second half)
    for (var r = 0u; r < POSEIDON2_ROUNDS_F / 2u; r = r + 1u) {
        for (var i = 0u; i < 3u; i = i + 1u) {
            state[i].limbs[0] = state[i].limbs[0] ^ (r * 3u + i + 200u);
        }
        state[0] = sbox(state[0]);
        state[1] = sbox(state[1]);
        state[2] = sbox(state[2]);
        state = mds_multiply(state);
    }

    return state[0];
}

// =============================================================================
// Merkle Tree Operations
// =============================================================================

// Compute one layer of Merkle tree: hash pairs of nodes into parents
@compute @workgroup_size(256)
fn merkle_layer(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let parent_count = params.node_count / 2u;

    if (idx >= parent_count) {
        return;
    }

    let left_idx = idx * 2u;
    let right_idx = left_idx + 1u;

    let left = nodes[left_idx];
    let right = nodes[right_idx];

    parents[idx] = poseidon2_hash(left, right);
}

// =============================================================================
// Commitment and Nullifier Operations
// =============================================================================

// Separate buffers for commitment inputs
@group(0) @binding(3) var<storage, read> values: array<Fr256>;
@group(0) @binding(4) var<storage, read> blindings: array<Fr256>;
@group(0) @binding(5) var<storage, read> salts: array<Fr256>;
@group(0) @binding(6) var<storage, read_write> commitments: array<Fr256>;

// Compute Pedersen-style commitment: H(H(value, blinding), salt)
@compute @workgroup_size(256)
fn batch_commitment(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&values)) {
        return;
    }

    let value = values[idx];
    let blinding = blindings[idx];
    let salt = salts[idx];

    // Two-round hash: H(H(v, b), s)
    let inner = poseidon2_hash(value, blinding);
    commitments[idx] = poseidon2_hash(inner, salt);
}

// Nullifier buffers
@group(0) @binding(7) var<storage, read> keys: array<Fr256>;
@group(0) @binding(8) var<storage, read> commit_refs: array<Fr256>;
@group(0) @binding(9) var<storage, read> indices: array<Fr256>;
@group(0) @binding(10) var<storage, read_write> nullifiers: array<Fr256>;

// Compute nullifier: H(H(key, commitment), index)
@compute @workgroup_size(256)
fn batch_nullifier(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&keys)) {
        return;
    }

    let key = keys[idx];
    let commitment = commit_refs[idx];
    let index = indices[idx];

    let inner = poseidon2_hash(key, commitment);
    nullifiers[idx] = poseidon2_hash(inner, index);
}
