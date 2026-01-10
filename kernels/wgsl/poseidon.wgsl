// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Poseidon Hash Function - WGSL Implementation
// ZK-friendly hash function for SNARKs/STARKs
//
// Part of the Lux Network GPU acceleration library

// ============================================================================
// Poseidon Parameters (BN254 scalar field)
// ============================================================================

// BN254 scalar field prime: 21888242871839275222246405745257275088548364400416034343698204186575808495617
// We use Montgomery representation for efficient modular arithmetic

const POSEIDON_T: u32 = 3u;           // State width (t=3 for 2-to-1 hash)
const POSEIDON_FULL_ROUNDS: u32 = 8u; // Full rounds (4 before, 4 after partial)
const POSEIDON_PARTIAL_ROUNDS: u32 = 57u; // Partial rounds

// Field prime limbs (low to high)
const P0: u32 = 0x43E1F593u;
const P1: u32 = 0x79B97091u;
const P2: u32 = 0x2833E848u;
const P3: u32 = 0x8181585Du;
const P4: u32 = 0xB85045B6u;
const P5: u32 = 0xE131A029u;
const P6: u32 = 0x30644E72u;
const P7: u32 = 0x00000000u;

// ============================================================================
// 256-bit Field Element (8 x 32-bit limbs)
// ============================================================================

struct Fe {
    limbs: array<u32, 8>,
}

fn fe_zero() -> Fe {
    var r: Fe;
    for (var i = 0u; i < 8u; i++) {
        r.limbs[i] = 0u;
    }
    return r;
}

fn fe_from_u32(x: u32) -> Fe {
    var r = fe_zero();
    r.limbs[0] = x;
    return r;
}

// Add two field elements (no reduction)
fn fe_add_no_reduce(a: Fe, b: Fe) -> Fe {
    var r: Fe;
    var carry = 0u;
    for (var i = 0u; i < 8u; i++) {
        let sum = a.limbs[i] + b.limbs[i] + carry;
        r.limbs[i] = sum;
        carry = select(0u, 1u, sum < a.limbs[i] || (carry == 1u && sum == a.limbs[i]));
    }
    return r;
}

// Subtract field elements (assumes a >= b)
fn fe_sub(a: Fe, b: Fe) -> Fe {
    var r: Fe;
    var borrow = 0u;
    for (var i = 0u; i < 8u; i++) {
        let diff = a.limbs[i] - b.limbs[i] - borrow;
        r.limbs[i] = diff;
        borrow = select(0u, 1u, a.limbs[i] < b.limbs[i] + borrow);
    }
    return r;
}

// Check if a >= b
fn fe_gte(a: Fe, b: Fe) -> bool {
    for (var i = 7u; i >= 0u; i--) {
        if (a.limbs[i] > b.limbs[i]) { return true; }
        if (a.limbs[i] < b.limbs[i]) { return false; }
        if (i == 0u) { break; }
    }
    return true;
}

// Get the field prime
fn fe_prime() -> Fe {
    var p: Fe;
    p.limbs[0] = P0; p.limbs[1] = P1; p.limbs[2] = P2; p.limbs[3] = P3;
    p.limbs[4] = P4; p.limbs[5] = P5; p.limbs[6] = P6; p.limbs[7] = P7;
    return p;
}

// Reduce modulo prime
fn fe_reduce(a: Fe) -> Fe {
    let p = fe_prime();
    var r = a;
    while (fe_gte(r, p)) {
        r = fe_sub(r, p);
    }
    return r;
}

// Modular addition
fn fe_add(a: Fe, b: Fe) -> Fe {
    return fe_reduce(fe_add_no_reduce(a, b));
}

// ============================================================================
// S-box: x^5 (Poseidon uses quintic S-box)
// ============================================================================

fn fe_square(a: Fe) -> Fe {
    // Simplified squaring (full implementation needs proper 512-bit intermediate)
    // For WGSL, we use a simplified version
    var r = fe_zero();
    
    // Basic schoolbook multiplication (truncated for demonstration)
    for (var i = 0u; i < 4u; i++) {
        var carry = 0u;
        for (var j = 0u; j < 4u; j++) {
            let k = i + j;
            if (k < 8u) {
                let prod = u64_from_u32(a.limbs[i]) * u64_from_u32(a.limbs[j]);
                let sum = u64_from_u32(r.limbs[k]) + prod + u64_from_u32(carry);
                r.limbs[k] = u32(sum.lo);
                carry = sum.hi;
            }
        }
    }
    return fe_reduce(r);
}

fn fe_mul(a: Fe, b: Fe) -> Fe {
    var r = fe_zero();
    for (var i = 0u; i < 4u; i++) {
        var carry = 0u;
        for (var j = 0u; j < 4u; j++) {
            let k = i + j;
            if (k < 8u) {
                let prod = u64_from_u32(a.limbs[i]) * u64_from_u32(b.limbs[j]);
                let sum = u64_from_u32(r.limbs[k]) + prod + u64_from_u32(carry);
                r.limbs[k] = u32(sum.lo);
                carry = sum.hi;
            }
        }
    }
    return fe_reduce(r);
}

// x^5 = x * x^4 = x * (x^2)^2
fn sbox(x: Fe) -> Fe {
    let x2 = fe_square(x);
    let x4 = fe_square(x2);
    return fe_mul(x, x4);
}

// ============================================================================
// Helper: u64 emulation
// ============================================================================

struct U64 {
    lo: u32,
    hi: u32,
}

fn u64_from_u32(x: u32) -> U64 {
    return U64(x, 0u);
}

// ============================================================================
// Kernel Bindings
// ============================================================================

struct PoseidonParams {
    num_hashes: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read_write> state: array<u32>;    // State elements (8 limbs each)
@group(0) @binding(1) var<storage, read> round_constants: array<u32>; // Round constants
@group(0) @binding(2) var<storage, read> mds_matrix: array<u32>;      // MDS matrix
@group(0) @binding(3) var<uniform> params: PoseidonParams;

// ============================================================================
// Poseidon Permutation
// ============================================================================

fn load_fe(base: u32) -> Fe {
    var r: Fe;
    for (var i = 0u; i < 8u; i++) {
        r.limbs[i] = state[base + i];
    }
    return r;
}

fn store_fe(base: u32, x: Fe) {
    for (var i = 0u; i < 8u; i++) {
        state[base + i] = x.limbs[i];
    }
}

fn load_rc(round: u32, idx: u32) -> Fe {
    var r: Fe;
    let base = (round * POSEIDON_T + idx) * 8u;
    for (var i = 0u; i < 8u; i++) {
        r.limbs[i] = round_constants[base + i];
    }
    return r;
}

fn load_mds(i: u32, j: u32) -> Fe {
    var r: Fe;
    let base = (i * POSEIDON_T + j) * 8u;
    for (var k = 0u; k < 8u; k++) {
        r.limbs[k] = mds_matrix[base + k];
    }
    return r;
}

// MDS matrix multiplication
fn mds_mul(s: array<Fe, 3>) -> array<Fe, 3> {
    var result: array<Fe, 3>;
    for (var i = 0u; i < POSEIDON_T; i++) {
        result[i] = fe_zero();
        for (var j = 0u; j < POSEIDON_T; j++) {
            let m = load_mds(i, j);
            result[i] = fe_add(result[i], fe_mul(m, s[j]));
        }
    }
    return result;
}

@compute @workgroup_size(64)
fn poseidon_permutation(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.num_hashes) { return; }
    
    let base = gid.x * POSEIDON_T * 8u;
    
    // Load state
    var s: array<Fe, 3>;
    for (var i = 0u; i < POSEIDON_T; i++) {
        s[i] = load_fe(base + i * 8u);
    }
    
    var round = 0u;
    
    // First half of full rounds
    for (var r = 0u; r < POSEIDON_FULL_ROUNDS / 2u; r++) {
        // Add round constants
        for (var i = 0u; i < POSEIDON_T; i++) {
            s[i] = fe_add(s[i], load_rc(round, i));
        }
        // S-box on all elements
        for (var i = 0u; i < POSEIDON_T; i++) {
            s[i] = sbox(s[i]);
        }
        // MDS
        s = mds_mul(s);
        round++;
    }
    
    // Partial rounds
    for (var r = 0u; r < POSEIDON_PARTIAL_ROUNDS; r++) {
        // Add round constants
        for (var i = 0u; i < POSEIDON_T; i++) {
            s[i] = fe_add(s[i], load_rc(round, i));
        }
        // S-box only on first element
        s[0] = sbox(s[0]);
        // MDS
        s = mds_mul(s);
        round++;
    }
    
    // Second half of full rounds
    for (var r = 0u; r < POSEIDON_FULL_ROUNDS / 2u; r++) {
        // Add round constants
        for (var i = 0u; i < POSEIDON_T; i++) {
            s[i] = fe_add(s[i], load_rc(round, i));
        }
        // S-box on all elements
        for (var i = 0u; i < POSEIDON_T; i++) {
            s[i] = sbox(s[i]);
        }
        // MDS
        s = mds_mul(s);
        round++;
    }
    
    // Store result
    for (var i = 0u; i < POSEIDON_T; i++) {
        store_fe(base + i * 8u, s[i]);
    }
}

// ============================================================================
// 2-to-1 Poseidon Hash
// ============================================================================

@compute @workgroup_size(64)
fn poseidon_hash_2to1(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.num_hashes) { return; }
    
    // Input: two field elements, output: one field element
    // State: [input0, input1, 0] -> permutation -> output = state[0]
    
    let input_base = gid.x * 2u * 8u;  // Two Fe inputs
    let output_base = gid.x * 8u;      // One Fe output (can be same buffer)
    
    var s: array<Fe, 3>;
    s[0] = load_fe(input_base);
    s[1] = load_fe(input_base + 8u);
    s[2] = fe_zero();  // Domain separator / capacity
    
    // Run permutation (inlined for single kernel)
    var round = 0u;
    
    for (var r = 0u; r < POSEIDON_FULL_ROUNDS / 2u; r++) {
        for (var i = 0u; i < POSEIDON_T; i++) { s[i] = fe_add(s[i], load_rc(round, i)); }
        for (var i = 0u; i < POSEIDON_T; i++) { s[i] = sbox(s[i]); }
        s = mds_mul(s);
        round++;
    }
    
    for (var r = 0u; r < POSEIDON_PARTIAL_ROUNDS; r++) {
        for (var i = 0u; i < POSEIDON_T; i++) { s[i] = fe_add(s[i], load_rc(round, i)); }
        s[0] = sbox(s[0]);
        s = mds_mul(s);
        round++;
    }
    
    for (var r = 0u; r < POSEIDON_FULL_ROUNDS / 2u; r++) {
        for (var i = 0u; i < POSEIDON_T; i++) { s[i] = fe_add(s[i], load_rc(round, i)); }
        for (var i = 0u; i < POSEIDON_T; i++) { s[i] = sbox(s[i]); }
        s = mds_mul(s);
        round++;
    }
    
    // Output is state[1] (capacity element)
    store_fe(output_base, s[1]);
}
