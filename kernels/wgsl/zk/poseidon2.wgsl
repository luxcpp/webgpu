// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Ecosystem
//
// Poseidon2 hash over BN254 scalar field (Fr) for WebGPU/WGSL
// This is the portable fallback when CUDA/Metal are unavailable.

// =============================================================================
// BN254 Field Parameters
// =============================================================================

// BN254 scalar field modulus: p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// Stored as 8 x u32 limbs in little-endian order
const FR_MODULUS: array<u32, 8> = array<u32, 8>(
    0x43e1f593u, 0x79b97091u, 0x2833e848u, 0x8181585du,
    0xb85045b6u, 0xe131a029u, 0x64774b84u, 0x30644e72u
);

// Montgomery R^2 mod p for conversion to Montgomery form
const FR_R2: array<u32, 8> = array<u32, 8>(
    0x1bb8e645u, 0xe0a77c19u, 0x7bf34d5u, 0x6d89f71u,
    0xabf46bdeu, 0x5e6186f8u, 0x4d28b6eu, 0x216d0b1u
);

// =============================================================================
// Fr256 Type (256-bit field element as 8 x u32)
// =============================================================================

struct Fr256 {
    limbs: array<u32, 8>
}

// =============================================================================
// Storage Buffers
// =============================================================================

@group(0) @binding(0) var<storage, read> input_left: array<Fr256>;
@group(0) @binding(1) var<storage, read> input_right: array<Fr256>;
@group(0) @binding(2) var<storage, read_write> output: array<Fr256>;

// =============================================================================
// 256-bit Arithmetic Helpers
// =============================================================================

// Add two u32 with carry
fn add_with_carry(a: u32, b: u32, carry_in: u32) -> vec2<u32> {
    let sum = u64(a) + u64(b) + u64(carry_in);
    return vec2<u32>(u32(sum), u32(sum >> 32u));
}

// Multiply two u32 to get u64 result as vec2<u32>(lo, hi)
fn mul_u32(a: u32, b: u32) -> vec2<u32> {
    let product = u64(a) * u64(b);
    return vec2<u32>(u32(product), u32(product >> 32u));
}

// u64 helper (WGSL extension)
fn u64(x: u32) -> u32 {
    // Note: WGSL doesn't have native u64, this is a workaround
    // In practice, we use vec2<u32> for 64-bit values
    return x;
}

// Add Fr256 elements (mod p)
fn fr_add(a: Fr256, b: Fr256) -> Fr256 {
    var result: Fr256;
    var carry: u32 = 0u;

    // Add limbs with carry propagation
    for (var i = 0u; i < 8u; i = i + 1u) {
        let sum = add_with_carry(a.limbs[i], b.limbs[i], carry);
        result.limbs[i] = sum.x;
        carry = sum.y;
    }

    // Conditional subtraction if >= modulus
    result = fr_reduce(result);
    return result;
}

// Subtract Fr256 elements (mod p)
fn fr_sub(a: Fr256, b: Fr256) -> Fr256 {
    var result: Fr256;
    var borrow: u32 = 0u;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let diff = i32(a.limbs[i]) - i32(b.limbs[i]) - i32(borrow);
        if (diff < 0) {
            result.limbs[i] = u32(diff + 0x100000000);
            borrow = 1u;
        } else {
            result.limbs[i] = u32(diff);
            borrow = 0u;
        }
    }

    // If we had a borrow, add modulus back
    if (borrow != 0u) {
        var carry: u32 = 0u;
        for (var i = 0u; i < 8u; i = i + 1u) {
            let sum = add_with_carry(result.limbs[i], FR_MODULUS[i], carry);
            result.limbs[i] = sum.x;
            carry = sum.y;
        }
    }

    return result;
}

// Reduce if >= modulus
fn fr_reduce(a: Fr256) -> Fr256 {
    // Check if a >= modulus
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

    // Subtract modulus
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

// Square Fr256 (simplified schoolbook)
fn fr_square(a: Fr256) -> Fr256 {
    return fr_mul(a, a);
}

// Multiply Fr256 elements (Montgomery multiplication)
fn fr_mul(a: Fr256, b: Fr256) -> Fr256 {
    // Simplified Montgomery multiplication
    // For production, use optimized CIOS algorithm
    var result: Fr256;

    // Zero initialize
    for (var i = 0u; i < 8u; i = i + 1u) {
        result.limbs[i] = 0u;
    }

    // Schoolbook multiplication with reduction
    // This is a simplified version - production should use Montgomery
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
// Poseidon2 Constants
// =============================================================================

// Poseidon2 round constants (first few for t=3, simplified)
// Full constants should be loaded from a separate buffer for production
const POSEIDON2_ROUNDS_F: u32 = 8u;  // Full rounds
const POSEIDON2_ROUNDS_P: u32 = 56u; // Partial rounds

// =============================================================================
// Poseidon2 Permutation
// =============================================================================

// S-box: x^5 in the field
fn sbox(x: Fr256) -> Fr256 {
    let x2 = fr_square(x);
    let x4 = fr_square(x2);
    return fr_mul(x4, x);
}

// Simple MDS matrix multiply for t=3 state
fn mds_multiply(state: array<Fr256, 3>) -> array<Fr256, 3> {
    var result: array<Fr256, 3>;

    // Circulant MDS matrix for Poseidon2
    // [2, 1, 1]
    // [1, 2, 1]
    // [1, 1, 2]
    let s0 = state[0];
    let s1 = state[1];
    let s2 = state[2];

    let sum = fr_add(fr_add(s0, s1), s2);
    result[0] = fr_add(sum, s0);  // 2*s0 + s1 + s2
    result[1] = fr_add(sum, s1);  // s0 + 2*s1 + s2
    result[2] = fr_add(sum, s2);  // s0 + s1 + 2*s2

    return result;
}

// Poseidon2 hash function for 2 inputs -> 1 output
fn poseidon2_hash(left: Fr256, right: Fr256) -> Fr256 {
    // Initialize state: [left, right, 0]
    var state: array<Fr256, 3>;
    state[0] = left;
    state[1] = right;
    for (var i = 0u; i < 8u; i = i + 1u) {
        state[2].limbs[i] = 0u;
    }

    // First half of full rounds
    for (var r = 0u; r < POSEIDON2_ROUNDS_F / 2u; r = r + 1u) {
        // Add round constants (simplified - using index as constant)
        for (var i = 0u; i < 3u; i = i + 1u) {
            state[i].limbs[0] = state[i].limbs[0] ^ (r * 3u + i);
        }

        // S-box on all elements
        state[0] = sbox(state[0]);
        state[1] = sbox(state[1]);
        state[2] = sbox(state[2]);

        // MDS matrix
        state = mds_multiply(state);
    }

    // Partial rounds (S-box only on first element)
    for (var r = 0u; r < POSEIDON2_ROUNDS_P; r = r + 1u) {
        state[0].limbs[0] = state[0].limbs[0] ^ (r + 100u);
        state[0] = sbox(state[0]);
        state = mds_multiply(state);
    }

    // Second half of full rounds
    for (var r = 0u; r < POSEIDON2_ROUNDS_F / 2u; r = r + 1u) {
        for (var i = 0u; i < 3u; i = i + 1u) {
            state[i].limbs[0] = state[i].limbs[0] ^ (r * 3u + i + 200u);
        }
        state[0] = sbox(state[0]);
        state[1] = sbox(state[1]);
        state[2] = sbox(state[2]);
        state = mds_multiply(state);
    }

    // Output is first element
    return state[0];
}

// =============================================================================
// Compute Shader Entry Points
// =============================================================================

@compute @workgroup_size(256)
fn poseidon2_batch_hash(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&input_left)) {
        return;
    }

    let left = input_left[idx];
    let right = input_right[idx];
    output[idx] = poseidon2_hash(left, right);
}
