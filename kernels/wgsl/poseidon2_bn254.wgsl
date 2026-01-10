// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Ecosystem
//
// Poseidon2 Hash Function over BN254 Scalar Field (Fr)
// WebGPU/WGSL Implementation
//
// This implements the Poseidon2 permutation as specified in:
// "Poseidon2: A Faster Version of the Poseidon Hash Function"
// https://eprint.iacr.org/2023/323
//
// Key differences from Poseidon:
// - Split into external rounds (full S-box) and internal rounds (partial S-box)
// - Different MDS matrices for external (M_E) and internal (M_I) rounds
// - More efficient linear layer in internal rounds
//
// Part of the Lux Network ZK cryptography library

// =============================================================================
// BN254 Scalar Field Parameters
// =============================================================================
//
// Field: Fr of BN254 (alt_bn128)
// Prime p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// Stored as 8 x u32 limbs in little-endian order

const FR_MODULUS: array<u32, 8> = array<u32, 8>(
    0x43e1f593u, 0x79b97091u, 0x2833e848u, 0x8181585du,
    0xb85045b6u, 0xe131a029u, 0x64774b84u, 0x30644e72u
);

// Montgomery constant: R = 2^256 mod p
const FR_R: array<u32, 8> = array<u32, 8>(
    0xd35d438u,  0x0a78eb28u, 0x7178e800u, 0x8a9d3ea6u,
    0x1ba36b62u, 0x31776c89u, 0x08ccf157u, 0x14b85d4u
);

// Montgomery constant: R^2 mod p (for converting to Montgomery form)
const FR_R2: array<u32, 8> = array<u32, 8>(
    0x1bb8e645u, 0xe0a77c19u, 0x07bf34d5u, 0x06d89f71u,
    0xabf46bdeu, 0x5e6186f8u, 0x04d28b6eu, 0x0216d0b1u
);

// Montgomery constant: -p^(-1) mod 2^32 (for CIOS reduction)
const FR_INV: u32 = 0xe4866389u;

// =============================================================================
// Poseidon2 Configuration (t=3, for 2-to-1 hashing)
// =============================================================================

const POSEIDON2_T: u32 = 3u;              // State width
const POSEIDON2_RATE: u32 = 2u;           // Absorption rate
const POSEIDON2_CAPACITY: u32 = 1u;       // Capacity (security)
const POSEIDON2_ROUNDS_F: u32 = 8u;       // External (full) rounds total
const POSEIDON2_ROUNDS_P: u32 = 56u;      // Internal (partial) rounds
const POSEIDON2_ALPHA: u32 = 5u;          // S-box exponent: x^5

// =============================================================================
// Fr256 Type: 256-bit field element as 8 x u32 limbs
// =============================================================================

struct Fr256 {
    limbs: array<u32, 8>
}

fn fr_zero() -> Fr256 {
    var r: Fr256;
    for (var i = 0u; i < 8u; i++) {
        r.limbs[i] = 0u;
    }
    return r;
}

fn fr_one() -> Fr256 {
    // Returns 1 in Montgomery form (= R mod p)
    var r: Fr256;
    for (var i = 0u; i < 8u; i++) {
        r.limbs[i] = FR_R[i];
    }
    return r;
}

fn fr_from_u32(x: u32) -> Fr256 {
    // Convert u32 to Montgomery form: x * R mod p
    var r = fr_zero();
    r.limbs[0] = x;
    return fr_to_mont(r);
}

fn fr_copy(a: Fr256) -> Fr256 {
    var r: Fr256;
    for (var i = 0u; i < 8u; i++) {
        r.limbs[i] = a.limbs[i];
    }
    return r;
}

// =============================================================================
// 256-bit Arithmetic Helpers
// =============================================================================

// Add two u32 with carry, returns (sum, carry)
fn adc(a: u32, b: u32, carry: u32) -> vec2<u32> {
    let sum_lo = a + b;
    let carry1 = select(0u, 1u, sum_lo < a);
    let sum = sum_lo + carry;
    let carry2 = select(0u, 1u, sum < sum_lo);
    return vec2<u32>(sum, carry1 + carry2);
}

// Subtract with borrow, returns (diff, borrow)
fn sbb(a: u32, b: u32, borrow: u32) -> vec2<u32> {
    let diff1 = a - b;
    let borrow1 = select(0u, 1u, a < b);
    let diff = diff1 - borrow;
    let borrow2 = select(0u, 1u, diff1 < borrow);
    return vec2<u32>(diff, borrow1 + borrow2);
}

// Multiply two u32, returns (lo, hi)
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;

    let mid = lh + hl;
    let mid_carry = select(0u, 0x10000u, mid < lh);

    let lo = ll + ((mid & 0xFFFFu) << 16u);
    let lo_carry = select(0u, 1u, lo < ll);

    let hi = hh + (mid >> 16u) + mid_carry + lo_carry;

    return vec2<u32>(lo, hi);
}

// Multiply-accumulate: acc += a * b, returns new carry
fn mac(acc: u32, a: u32, b: u32, carry: u32) -> vec2<u32> {
    let prod = mul32(a, b);
    let sum1 = adc(acc, prod.x, 0u);
    let sum2 = adc(sum1.x, carry, 0u);
    return vec2<u32>(sum2.x, prod.y + sum1.y + sum2.y);
}

// =============================================================================
// Field Element Comparison
// =============================================================================

fn fr_gte(a: Fr256, b: Fr256) -> bool {
    for (var i = 7; i >= 0; i--) {
        let idx = u32(i);
        if (a.limbs[idx] > b.limbs[idx]) { return true; }
        if (a.limbs[idx] < b.limbs[idx]) { return false; }
        if (i == 0) { break; }
    }
    return true; // Equal
}

fn fr_is_zero(a: Fr256) -> bool {
    for (var i = 0u; i < 8u; i++) {
        if (a.limbs[i] != 0u) { return false; }
    }
    return true;
}

// =============================================================================
// Modular Arithmetic
// =============================================================================

// Reduce if >= modulus (constant-time)
fn fr_reduce_once(a: Fr256) -> Fr256 {
    var r: Fr256;
    var borrow: u32 = 0u;

    // Compute a - p
    for (var i = 0u; i < 8u; i++) {
        let sb = sbb(a.limbs[i], FR_MODULUS[i], borrow);
        r.limbs[i] = sb.x;
        borrow = sb.y;
    }

    // If borrow == 0, result is valid (a >= p), else return a
    if (borrow == 0u) {
        return r;
    }
    return a;
}

// Modular addition: (a + b) mod p
fn fr_add(a: Fr256, b: Fr256) -> Fr256 {
    var r: Fr256;
    var carry: u32 = 0u;

    for (var i = 0u; i < 8u; i++) {
        let ac = adc(a.limbs[i], b.limbs[i], carry);
        r.limbs[i] = ac.x;
        carry = ac.y;
    }

    // Reduce if overflow or >= p
    r = fr_reduce_once(r);
    if (carry != 0u) {
        r = fr_reduce_once(r);
    }

    return r;
}

// Modular subtraction: (a - b) mod p
fn fr_sub(a: Fr256, b: Fr256) -> Fr256 {
    var r: Fr256;
    var borrow: u32 = 0u;

    for (var i = 0u; i < 8u; i++) {
        let sb = sbb(a.limbs[i], b.limbs[i], borrow);
        r.limbs[i] = sb.x;
        borrow = sb.y;
    }

    // If underflow, add modulus
    if (borrow != 0u) {
        var carry: u32 = 0u;
        for (var i = 0u; i < 8u; i++) {
            let ac = adc(r.limbs[i], FR_MODULUS[i], carry);
            r.limbs[i] = ac.x;
            carry = ac.y;
        }
    }

    return r;
}

// Montgomery multiplication: (a * b * R^-1) mod p
// Using CIOS (Coarsely Integrated Operand Scanning) algorithm
fn fr_mul_mont(a: Fr256, b: Fr256) -> Fr256 {
    var t: array<u32, 9>; // Extended for carries
    for (var i = 0u; i < 9u; i++) { t[i] = 0u; }

    for (var i = 0u; i < 8u; i++) {
        // First: t += a[i] * b
        var carry: u32 = 0u;
        for (var j = 0u; j < 8u; j++) {
            let mc = mac(t[j], a.limbs[i], b.limbs[j], carry);
            t[j] = mc.x;
            carry = mc.y;
        }
        let ac1 = adc(t[8], carry, 0u);
        t[8] = ac1.x;

        // Second: Montgomery reduction step
        let m = t[0] * FR_INV;
        carry = 0u;
        let mc0 = mac(t[0], m, FR_MODULUS[0], 0u);
        carry = mc0.y;

        for (var j = 1u; j < 8u; j++) {
            let mc = mac(t[j], m, FR_MODULUS[j], carry);
            t[j - 1u] = mc.x;
            carry = mc.y;
        }
        let ac2 = adc(t[8], carry, 0u);
        t[7] = ac2.x;
        t[8] = ac2.y;
    }

    var r: Fr256;
    for (var i = 0u; i < 8u; i++) {
        r.limbs[i] = t[i];
    }

    return fr_reduce_once(r);
}

// Square in Montgomery form (using multiplication for simplicity)
fn fr_square_mont(a: Fr256) -> Fr256 {
    return fr_mul_mont(a, a);
}

// Convert to Montgomery form: a * R mod p
fn fr_to_mont(a: Fr256) -> Fr256 {
    var r2: Fr256;
    for (var i = 0u; i < 8u; i++) {
        r2.limbs[i] = FR_R2[i];
    }
    return fr_mul_mont(a, r2);
}

// Convert from Montgomery form: a * R^-1 mod p
fn fr_from_mont(a: Fr256) -> Fr256 {
    var one = fr_zero();
    one.limbs[0] = 1u;
    return fr_mul_mont(a, one);
}

// =============================================================================
// Poseidon2 S-box: x^5
// =============================================================================

fn poseidon2_sbox(x: Fr256) -> Fr256 {
    // x^5 = x * x^4 = x * (x^2)^2
    let x2 = fr_square_mont(x);
    let x4 = fr_square_mont(x2);
    return fr_mul_mont(x4, x);
}

// =============================================================================
// Poseidon2 MDS Matrices
// =============================================================================

// External rounds MDS: Circulant matrix with [2, 1, 1]
// M_E * [a, b, c] = [2a+b+c, a+2b+c, a+b+2c]
fn poseidon2_mds_external(state: array<Fr256, 3>) -> array<Fr256, 3> {
    var result: array<Fr256, 3>;

    // Compute sum = a + b + c
    let sum = fr_add(fr_add(state[0], state[1]), state[2]);

    // result[i] = sum + state[i] = 2*state[i] + other_elements
    result[0] = fr_add(sum, state[0]);
    result[1] = fr_add(sum, state[1]);
    result[2] = fr_add(sum, state[2]);

    return result;
}

// Internal rounds MDS: Optimized for t=3
// M_I = diag(1, 1, ..., 1) + outer(1, [1, ..., 1, 0])
// More efficient: only modifies first element based on others
fn poseidon2_mds_internal(state: array<Fr256, 3>) -> array<Fr256, 3> {
    var result: array<Fr256, 3>;

    // For internal rounds, use a simpler diffusion:
    // s'[0] = 2*s[0] + s[1] + s[2]
    // s'[1] = s[0] + 2*s[1] + s[2]
    // s'[2] = s[0] + s[1] + 2*s[2]
    // Same as external for t=3, but in general different

    let sum = fr_add(fr_add(state[0], state[1]), state[2]);
    result[0] = fr_add(sum, state[0]);
    result[1] = fr_add(sum, state[1]);
    result[2] = fr_add(sum, state[2]);

    return result;
}

// =============================================================================
// Round Constants Storage
// =============================================================================

// Uniform buffer for round constants (loaded at runtime)
struct Poseidon2Params {
    num_elements: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32
}

@group(0) @binding(0) var<storage, read> round_constants_ext: array<Fr256>; // External round constants
@group(0) @binding(1) var<storage, read> round_constants_int: array<Fr256>; // Internal round constants
@group(0) @binding(2) var<uniform> params: Poseidon2Params;

// Load external round constant
fn load_rc_ext(round: u32, idx: u32) -> Fr256 {
    return round_constants_ext[round * POSEIDON2_T + idx];
}

// Load internal round constant (only first element has non-zero constant)
fn load_rc_int(round: u32) -> Fr256 {
    return round_constants_int[round];
}

// =============================================================================
// Poseidon2 Permutation
// =============================================================================

fn poseidon2_permutation(state: ptr<function, array<Fr256, 3>>) {
    // --- First half of external (full) rounds ---
    for (var r = 0u; r < POSEIDON2_ROUNDS_F / 2u; r++) {
        // Add round constants
        for (var i = 0u; i < POSEIDON2_T; i++) {
            (*state)[i] = fr_add((*state)[i], load_rc_ext(r, i));
        }

        // S-box on ALL elements (external round)
        (*state)[0] = poseidon2_sbox((*state)[0]);
        (*state)[1] = poseidon2_sbox((*state)[1]);
        (*state)[2] = poseidon2_sbox((*state)[2]);

        // External MDS matrix
        *state = poseidon2_mds_external(*state);
    }

    // --- Internal (partial) rounds ---
    for (var r = 0u; r < POSEIDON2_ROUNDS_P; r++) {
        // Add round constant (only to first element)
        (*state)[0] = fr_add((*state)[0], load_rc_int(r));

        // S-box ONLY on first element (internal round)
        (*state)[0] = poseidon2_sbox((*state)[0]);

        // Internal MDS matrix
        *state = poseidon2_mds_internal(*state);
    }

    // --- Second half of external (full) rounds ---
    let ext_offset = POSEIDON2_ROUNDS_F / 2u;
    for (var r = 0u; r < POSEIDON2_ROUNDS_F / 2u; r++) {
        // Add round constants
        for (var i = 0u; i < POSEIDON2_T; i++) {
            (*state)[i] = fr_add((*state)[i], load_rc_ext(ext_offset + r, i));
        }

        // S-box on ALL elements
        (*state)[0] = poseidon2_sbox((*state)[0]);
        (*state)[1] = poseidon2_sbox((*state)[1]);
        (*state)[2] = poseidon2_sbox((*state)[2]);

        // External MDS matrix
        *state = poseidon2_mds_external(*state);
    }
}

// =============================================================================
// Hash Functions
// =============================================================================

// 2-to-1 hash: H(left, right) -> digest
fn poseidon2_hash_2to1(left: Fr256, right: Fr256) -> Fr256 {
    var state: array<Fr256, 3>;
    state[0] = left;
    state[1] = right;
    state[2] = fr_zero(); // Capacity element

    poseidon2_permutation(&state);

    return state[0];
}

// Sponge construction for arbitrary length input
fn poseidon2_hash_sponge(inputs: ptr<function, array<Fr256, 16>>, len: u32) -> Fr256 {
    var state: array<Fr256, 3>;
    state[0] = fr_zero();
    state[1] = fr_zero();
    state[2] = fr_zero();

    var absorbed = 0u;

    // Absorb phase
    while (absorbed < len) {
        // Absorb up to RATE elements
        let to_absorb = min(POSEIDON2_RATE, len - absorbed);

        for (var i = 0u; i < to_absorb; i++) {
            state[i] = fr_add(state[i], (*inputs)[absorbed + i]);
        }
        absorbed += to_absorb;

        poseidon2_permutation(&state);
    }

    // Squeeze: return first element
    return state[0];
}

// =============================================================================
// Storage Buffers for Batch Operations
// =============================================================================

@group(0) @binding(3) var<storage, read> input_left: array<Fr256>;
@group(0) @binding(4) var<storage, read> input_right: array<Fr256>;
@group(0) @binding(5) var<storage, read_write> output: array<Fr256>;

// =============================================================================
// Compute Kernels
// =============================================================================

// Batch 2-to-1 hash
@compute @workgroup_size(256)
fn poseidon2_batch_hash(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_elements) {
        return;
    }

    let left = input_left[idx];
    let right = input_right[idx];
    output[idx] = poseidon2_hash_2to1(left, right);
}

// =============================================================================
// Merkle Tree Operations
// =============================================================================

@group(1) @binding(0) var<storage, read> merkle_nodes: array<Fr256>;
@group(1) @binding(1) var<storage, read_write> merkle_parents: array<Fr256>;

struct MerkleParams {
    layer_size: u32,    // Number of nodes in current layer
    _pad1: u32,
    _pad2: u32,
    _pad3: u32
}
@group(1) @binding(2) var<uniform> merkle_params: MerkleParams;

// Compute one layer of Merkle tree
@compute @workgroup_size(256)
fn merkle_layer(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let parent_count = merkle_params.layer_size / 2u;

    if (idx >= parent_count) {
        return;
    }

    let left_idx = idx * 2u;
    let right_idx = left_idx + 1u;

    let left = merkle_nodes[left_idx];
    let right = merkle_nodes[right_idx];

    merkle_parents[idx] = poseidon2_hash_2to1(left, right);
}

// Compute Merkle root for small tree (fits in single workgroup)
// Tree is stored level by level: [leaf0, leaf1, ..., leafN, parent0, parent1, ..., root]
@compute @workgroup_size(256)
fn merkle_root_small(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let num_leaves = merkle_params.layer_size;

    // Load leaf into shared memory
    var shared_nodes: array<Fr256, 256>;

    if (tid < num_leaves) {
        shared_nodes[tid] = merkle_nodes[tid];
    } else {
        shared_nodes[tid] = fr_zero();
    }

    workgroupBarrier();

    // Reduction loop
    var active = num_leaves / 2u;
    var offset = 0u;

    while (active > 0u) {
        if (tid < active) {
            let left = shared_nodes[offset + tid * 2u];
            let right = shared_nodes[offset + tid * 2u + 1u];
            shared_nodes[tid] = poseidon2_hash_2to1(left, right);
        }

        workgroupBarrier();

        offset = 0u;
        active = active / 2u;
    }

    // Thread 0 writes root
    if (tid == 0u) {
        merkle_parents[0] = shared_nodes[0];
    }
}

// =============================================================================
// Commitment Operations (for ZK privacy)
// =============================================================================

@group(2) @binding(0) var<storage, read> commit_values: array<Fr256>;
@group(2) @binding(1) var<storage, read> commit_blindings: array<Fr256>;
@group(2) @binding(2) var<storage, read> commit_salts: array<Fr256>;
@group(2) @binding(3) var<storage, read_write> commitments: array<Fr256>;

struct CommitParams {
    count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32
}
@group(2) @binding(4) var<uniform> commit_params: CommitParams;

// Compute Pedersen-style commitment: C = H(H(value, blinding), salt)
@compute @workgroup_size(256)
fn batch_commitment(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= commit_params.count) {
        return;
    }

    let value = commit_values[idx];
    let blinding = commit_blindings[idx];
    let salt = commit_salts[idx];

    let inner = poseidon2_hash_2to1(value, blinding);
    commitments[idx] = poseidon2_hash_2to1(inner, salt);
}

// =============================================================================
// Nullifier Operations (for ZK privacy)
// =============================================================================

@group(3) @binding(0) var<storage, read> nullifier_keys: array<Fr256>;
@group(3) @binding(1) var<storage, read> nullifier_commits: array<Fr256>;
@group(3) @binding(2) var<storage, read> nullifier_indices: array<Fr256>;
@group(3) @binding(3) var<storage, read_write> nullifiers: array<Fr256>;

struct NullifierParams {
    count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32
}
@group(3) @binding(4) var<uniform> nullifier_params: NullifierParams;

// Compute nullifier: N = H(H(key, commitment), index)
@compute @workgroup_size(256)
fn batch_nullifier(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= nullifier_params.count) {
        return;
    }

    let key = nullifier_keys[idx];
    let commitment = nullifier_commits[idx];
    let index = nullifier_indices[idx];

    let inner = poseidon2_hash_2to1(key, commitment);
    nullifiers[idx] = poseidon2_hash_2to1(inner, index);
}

// =============================================================================
// Arbitrary Input Hash (Sponge)
// =============================================================================

@group(4) @binding(0) var<storage, read> sponge_input: array<Fr256>;
@group(4) @binding(1) var<storage, read_write> sponge_output: array<Fr256>;

struct SpongeParams {
    batch_size: u32,     // Number of hashes to compute
    input_len: u32,      // Length of each input (in field elements)
    _pad1: u32,
    _pad2: u32
}
@group(4) @binding(2) var<uniform> sponge_params: SpongeParams;

// Hash arbitrary-length input using sponge construction
@compute @workgroup_size(256)
fn poseidon2_sponge_hash(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.x;
    if (batch_idx >= sponge_params.batch_size) {
        return;
    }

    var state: array<Fr256, 3>;
    state[0] = fr_zero();
    state[1] = fr_zero();
    state[2] = fr_zero();

    let base = batch_idx * sponge_params.input_len;
    var absorbed = 0u;

    // Absorb all input elements
    while (absorbed < sponge_params.input_len) {
        let to_absorb = min(POSEIDON2_RATE, sponge_params.input_len - absorbed);

        for (var i = 0u; i < to_absorb; i++) {
            state[i] = fr_add(state[i], sponge_input[base + absorbed + i]);
        }
        absorbed += to_absorb;

        poseidon2_permutation(&state);
    }

    // Output first element as digest
    sponge_output[batch_idx] = state[0];
}

// =============================================================================
// Merkle Path Verification
// =============================================================================

// For verifying inclusion proofs
@group(5) @binding(0) var<storage, read> path_leaves: array<Fr256>;
@group(5) @binding(1) var<storage, read> path_siblings: array<Fr256>;
@group(5) @binding(2) var<storage, read> path_indices: array<u32>;  // 0 = left, 1 = right
@group(5) @binding(3) var<storage, read> expected_roots: array<Fr256>;
@group(5) @binding(4) var<storage, read_write> verification_results: array<u32>; // 1 = valid, 0 = invalid

struct PathParams {
    batch_size: u32,     // Number of paths to verify
    path_length: u32,    // Depth of tree
    _pad1: u32,
    _pad2: u32
}
@group(5) @binding(5) var<uniform> path_params: PathParams;

@compute @workgroup_size(256)
fn verify_merkle_paths(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.x;
    if (batch_idx >= path_params.batch_size) {
        return;
    }

    let path_base = batch_idx * path_params.path_length;
    var current = path_leaves[batch_idx];

    // Walk up the tree
    for (var i = 0u; i < path_params.path_length; i++) {
        let sibling = path_siblings[path_base + i];
        let position = path_indices[path_base + i];

        if (position == 0u) {
            // Current is left child
            current = poseidon2_hash_2to1(current, sibling);
        } else {
            // Current is right child
            current = poseidon2_hash_2to1(sibling, current);
        }
    }

    // Compare computed root with expected
    let expected = expected_roots[batch_idx];
    var match_found = true;
    for (var i = 0u; i < 8u; i++) {
        if (current.limbs[i] != expected.limbs[i]) {
            match_found = false;
            break;
        }
    }

    verification_results[batch_idx] = select(0u, 1u, match_found);
}

// =============================================================================
// Domain Separation Helpers
// =============================================================================

// Hash with domain separator for different use cases
fn poseidon2_hash_with_domain(left: Fr256, right: Fr256, domain: u32) -> Fr256 {
    var state: array<Fr256, 3>;
    state[0] = left;
    state[1] = right;
    state[2] = fr_from_u32(domain); // Domain separator in capacity

    poseidon2_permutation(&state);

    return state[0];
}

// Domain constants
const DOMAIN_MERKLE: u32 = 0u;
const DOMAIN_COMMITMENT: u32 = 1u;
const DOMAIN_NULLIFIER: u32 = 2u;
const DOMAIN_MESSAGE: u32 = 3u;

// =============================================================================
// Standalone Permutation (No Constants Buffer)
// Uses simplified placeholder constants for testing
// =============================================================================

fn poseidon2_permutation_simple(state: ptr<function, array<Fr256, 3>>) {
    // First half external rounds
    for (var r = 0u; r < POSEIDON2_ROUNDS_F / 2u; r++) {
        // Add placeholder round constants
        for (var i = 0u; i < 3u; i++) {
            (*state)[i].limbs[0] ^= (r * 3u + i + 1u);
        }

        (*state)[0] = poseidon2_sbox((*state)[0]);
        (*state)[1] = poseidon2_sbox((*state)[1]);
        (*state)[2] = poseidon2_sbox((*state)[2]);

        *state = poseidon2_mds_external(*state);
    }

    // Internal rounds
    for (var r = 0u; r < POSEIDON2_ROUNDS_P; r++) {
        (*state)[0].limbs[0] ^= (r + 100u);
        (*state)[0] = poseidon2_sbox((*state)[0]);
        *state = poseidon2_mds_internal(*state);
    }

    // Second half external rounds
    for (var r = 0u; r < POSEIDON2_ROUNDS_F / 2u; r++) {
        for (var i = 0u; i < 3u; i++) {
            (*state)[i].limbs[0] ^= (r * 3u + i + 200u);
        }

        (*state)[0] = poseidon2_sbox((*state)[0]);
        (*state)[1] = poseidon2_sbox((*state)[1]);
        (*state)[2] = poseidon2_sbox((*state)[2]);

        *state = poseidon2_mds_external(*state);
    }
}

// Standalone 2-to-1 hash (no external constants)
fn poseidon2_hash_standalone(left: Fr256, right: Fr256) -> Fr256 {
    var state: array<Fr256, 3>;
    state[0] = left;
    state[1] = right;
    state[2] = fr_zero();

    poseidon2_permutation_simple(&state);

    return state[0];
}

// =============================================================================
// Simplified Entry Points (Self-Contained)
// =============================================================================

@group(6) @binding(0) var<storage, read> simple_left: array<Fr256>;
@group(6) @binding(1) var<storage, read> simple_right: array<Fr256>;
@group(6) @binding(2) var<storage, read_write> simple_output: array<Fr256>;

// Self-contained batch hash (no constants buffer needed)
@compute @workgroup_size(256)
fn poseidon2_batch_hash_simple(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&simple_left)) {
        return;
    }

    let left = simple_left[idx];
    let right = simple_right[idx];
    simple_output[idx] = poseidon2_hash_standalone(left, right);
}

// Self-contained Merkle layer (no constants buffer needed)
@compute @workgroup_size(256)
fn merkle_layer_simple(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let num_pairs = arrayLength(&simple_left) / 2u;

    if (idx >= num_pairs) {
        return;
    }

    let left_idx = idx * 2u;
    let right_idx = left_idx + 1u;

    var left: Fr256;
    var right: Fr256;

    // Read from interleaved array
    left = simple_left[left_idx];
    right = simple_left[right_idx];

    simple_output[idx] = poseidon2_hash_standalone(left, right);
}
