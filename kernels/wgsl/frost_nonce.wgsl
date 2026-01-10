// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// FROST Nonce Generation and Commitment Operations
// Batch nonce generation, hash-to-curve, and binding factor computation
// Portable WebGPU implementation

// ============================================================================
// Types
// ============================================================================

struct Scalar256 {
    limbs: array<u32, 8>,
}

struct Ed25519Affine {
    x: Scalar256,
    y: Scalar256,
}

struct Ed25519Extended {
    x: Scalar256,
    y: Scalar256,
    z: Scalar256,
    t: Scalar256,
}

struct NonceCommitment {
    hiding_nonce_d: Scalar256,
    binding_nonce_e: Scalar256,
    commitment_d: Ed25519Affine,
    commitment_e: Ed25519Affine,
}

struct NonceParams {
    num_participants: u32,
    seed_entropy_offset: u32,
    curve_type: u32,
    batch_size: u32,
}

struct ChaCha20State {
    state: array<u32, 16>,
}

// ============================================================================
// Constants
// ============================================================================

const ED25519_L: array<u32, 8> = array<u32, 8>(
    0x5cf5d3edu, 0x5812631au, 0xa2f79cd6u, 0x14def9deu,
    0x00000000u, 0x00000000u, 0x00000000u, 0x10000000u
);

const SECP256K1_N: array<u32, 8> = array<u32, 8>(
    0xd0364141u, 0xbfd25e8cu, 0xaf48a03bu, 0xbaaedce6u,
    0xfffffffeu, 0xffffffffu, 0xffffffffu, 0xffffffffu
);

const ED25519_GY: array<u32, 8> = array<u32, 8>(
    0x58666666u, 0x66666666u, 0x66666666u, 0x66666666u,
    0x66666666u, 0x66666666u, 0x66666666u, 0x66666666u
);

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> seeds: array<Scalar256>;
@group(0) @binding(1) var<storage, read_write> nonces: array<NonceCommitment>;
@group(0) @binding(2) var<uniform> params: NonceParams;
@group(0) @binding(3) var<storage, read> participant_ids: array<u32>;
@group(0) @binding(4) var<storage, read> commitment_list: array<Ed25519Affine>;
@group(0) @binding(5) var<storage, read> message: array<Scalar256>;
@group(0) @binding(6) var<storage, read_write> binding_factors: array<Scalar256>;
@group(0) @binding(7) var<storage, read_write> commitment_shares: array<Ed25519Extended>;
@group(0) @binding(8) var<storage, read_write> group_commitment: array<Ed25519Extended>;
@group(0) @binding(9) var<storage, read_write> valid: array<u32>;

// ============================================================================
// Scalar Arithmetic
// ============================================================================

fn scalar_zero() -> Scalar256 {
    var r: Scalar256;
    for (var i = 0u; i < 8u; i++) {
        r.limbs[i] = 0u;
    }
    return r;
}

fn scalar_one() -> Scalar256 {
    var r = scalar_zero();
    r.limbs[0] = 1u;
    return r;
}

fn scalar_is_zero(a: Scalar256) -> bool {
    for (var i = 0u; i < 8u; i++) {
        if (a.limbs[i] != 0u) { return false; }
    }
    return true;
}

fn get_modulus(curve_type: u32) -> array<u32, 8> {
    if (curve_type == 0u) {
        return ED25519_L;
    } else {
        return SECP256K1_N;
    }
}

fn scalar_gte_mod(a: Scalar256, mod_arr: array<u32, 8>) -> bool {
    for (var i = 7; i >= 0; i--) {
        if (a.limbs[i] > mod_arr[i]) { return true; }
        if (a.limbs[i] < mod_arr[i]) { return false; }
    }
    return true;
}

fn scalar_add(a: Scalar256, b: Scalar256, mod_arr: array<u32, 8>) -> Scalar256 {
    var r: Scalar256;
    var carry = 0u;
    
    for (var i = 0u; i < 8u; i++) {
        let sum = a.limbs[i] + b.limbs[i] + carry;
        carry = select(0u, 1u, sum < a.limbs[i] || (carry == 1u && sum == a.limbs[i]));
        r.limbs[i] = sum;
    }
    
    if (carry == 1u || scalar_gte_mod(r, mod_arr)) {
        var borrow = 0u;
        for (var i = 0u; i < 8u; i++) {
            let diff = r.limbs[i] - mod_arr[i] - borrow;
            borrow = select(0u, 1u, r.limbs[i] < mod_arr[i] + borrow);
            r.limbs[i] = diff;
        }
    }
    
    return r;
}

fn mul32_wide(a: u32, b: u32) -> vec2<u32> {
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
    let lo_carry = select(0u, 1u, lo < p0);
    let hi = p3 + (mid >> 16u) + mid_carry + lo_carry;
    
    return vec2<u32>(lo, hi);
}

fn scalar_mul(a: Scalar256, b: Scalar256, mod_arr: array<u32, 8>) -> Scalar256 {
    var product: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) {
        product[i] = 0u;
    }
    
    for (var i = 0u; i < 8u; i++) {
        var carry = 0u;
        for (var j = 0u; j < 8u; j++) {
            let prod = mul32_wide(a.limbs[i], b.limbs[j]);
            let sum_lo = product[i + j] + prod.x + carry;
            let carry_lo = select(0u, 1u, sum_lo < product[i + j] || (carry > 0u && sum_lo == product[i + j]));
            product[i + j] = sum_lo;
            carry = prod.y + carry_lo;
        }
        product[i + 8u] = carry;
    }
    
    var r: Scalar256;
    for (var i = 0u; i < 8u; i++) {
        r.limbs[i] = product[i];
    }
    
    while (scalar_gte_mod(r, mod_arr)) {
        var borrow = 0u;
        for (var i = 0u; i < 8u; i++) {
            let diff = r.limbs[i] - mod_arr[i] - borrow;
            borrow = select(0u, 1u, r.limbs[i] < mod_arr[i] + borrow);
            r.limbs[i] = diff;
        }
    }
    
    return r;
}

// ============================================================================
// ChaCha20 PRNG
// ============================================================================

fn rotl32(x: u32, n: u32) -> u32 {
    return (x << n) | (x >> (32u - n));
}

fn chacha_quarter_round(s: ptr<function, array<u32, 16>>, a: u32, b: u32, c: u32, d: u32) {
    (*s)[a] = (*s)[a] + (*s)[b]; (*s)[d] = rotl32((*s)[d] ^ (*s)[a], 16u);
    (*s)[c] = (*s)[c] + (*s)[d]; (*s)[b] = rotl32((*s)[b] ^ (*s)[c], 12u);
    (*s)[a] = (*s)[a] + (*s)[b]; (*s)[d] = rotl32((*s)[d] ^ (*s)[a], 8u);
    (*s)[c] = (*s)[c] + (*s)[d]; (*s)[b] = rotl32((*s)[b] ^ (*s)[c], 7u);
}

fn chacha20_init(key: Scalar256, counter: u32, nonce: u32) -> ChaCha20State {
    var s: ChaCha20State;
    
    // "expand 32-byte k"
    s.state[0] = 0x61707865u;
    s.state[1] = 0x3320646eu;
    s.state[2] = 0x79622d32u;
    s.state[3] = 0x6b206574u;
    
    // Key
    for (var i = 0u; i < 8u; i++) {
        s.state[4u + i] = key.limbs[i];
    }
    
    // Counter and nonce
    s.state[12] = counter;
    s.state[13] = 0u;
    s.state[14] = nonce;
    s.state[15] = 0u;
    
    return s;
}

fn chacha20_block(s: ptr<function, ChaCha20State>) {
    var working: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) {
        working[i] = (*s).state[i];
    }
    
    // 20 rounds
    for (var i = 0u; i < 10u; i++) {
        chacha_quarter_round(&working, 0u, 4u, 8u, 12u);
        chacha_quarter_round(&working, 1u, 5u, 9u, 13u);
        chacha_quarter_round(&working, 2u, 6u, 10u, 14u);
        chacha_quarter_round(&working, 3u, 7u, 11u, 15u);
        chacha_quarter_round(&working, 0u, 5u, 10u, 15u);
        chacha_quarter_round(&working, 1u, 6u, 11u, 12u);
        chacha_quarter_round(&working, 2u, 7u, 8u, 13u);
        chacha_quarter_round(&working, 3u, 4u, 9u, 14u);
    }
    
    for (var i = 0u; i < 16u; i++) {
        (*s).state[i] = (*s).state[i] + working[i];
    }
    
    (*s).state[12] = (*s).state[12] + 1u;
}

fn random_scalar(rng: ptr<function, ChaCha20State>, mod_arr: array<u32, 8>) -> Scalar256 {
    var r: Scalar256;
    
    chacha20_block(rng);
    for (var i = 0u; i < 8u; i++) {
        r.limbs[i] = (*rng).state[i];
    }
    
    r.limbs[7] = r.limbs[7] & 0x0FFFFFFFu;
    
    while (scalar_gte_mod(r, mod_arr)) {
        var borrow = 0u;
        for (var i = 0u; i < 8u; i++) {
            let diff = r.limbs[i] - mod_arr[i] - borrow;
            borrow = select(0u, 1u, r.limbs[i] < mod_arr[i] + borrow);
            r.limbs[i] = diff;
        }
    }
    
    if (scalar_is_zero(r)) {
        r.limbs[0] = 1u;
    }
    
    return r;
}

// ============================================================================
// Point Operations
// ============================================================================

fn ed25519_identity() -> Ed25519Extended {
    var r: Ed25519Extended;
    r.x = scalar_zero();
    r.y = scalar_one();
    r.z = scalar_one();
    r.t = scalar_zero();
    return r;
}

fn ed25519_add(p: Ed25519Extended, q: Ed25519Extended, mod_arr: array<u32, 8>) -> Ed25519Extended {
    var r: Ed25519Extended;
    r.x = scalar_add(p.x, q.x, mod_arr);
    r.y = scalar_add(p.y, q.y, mod_arr);
    r.z = scalar_add(p.z, q.z, mod_arr);
    r.t = scalar_add(p.t, q.t, mod_arr);
    return r;
}

fn ed25519_double(p: Ed25519Extended, mod_arr: array<u32, 8>) -> Ed25519Extended {
    var r = p;
    r.z = scalar_add(p.z, p.z, mod_arr);
    return r;
}

fn ed25519_scalar_mul_base(scalar: Scalar256, mod_arr: array<u32, 8>) -> Ed25519Extended {
    var result = ed25519_identity();
    var base: Ed25519Extended;
    
    for (var i = 0u; i < 8u; i++) {
        base.y.limbs[i] = ED25519_GY[i];
    }
    base.x = scalar_zero();
    base.z = scalar_one();
    base.t = scalar_zero();
    
    for (var bit = 0u; bit < 256u; bit++) {
        let limb = bit / 32u;
        let bit_in_limb = bit % 32u;
        
        if ((scalar.limbs[limb] & (1u << bit_in_limb)) != 0u) {
            result = ed25519_add(result, base, mod_arr);
        }
        base = ed25519_double(base, mod_arr);
    }
    
    return result;
}

fn ed25519_to_affine(ext: Ed25519Extended) -> Ed25519Affine {
    var r: Ed25519Affine;
    r.x = ext.x;
    r.y = ext.y;
    return r;
}

// ============================================================================
// Hash Functions
// ============================================================================

fn hash_to_scalar(data: array<u32, 16>, data_len: u32, mod_arr: array<u32, 8>) -> Scalar256 {
    var r = scalar_zero();
    
    for (var i = 0u; i < min(data_len, 8u); i++) {
        r.limbs[i] = data[i];
    }
    
    for (var round = 0u; round < 4u; round++) {
        for (var i = 0u; i < 8u; i++) {
            r.limbs[i] = r.limbs[i] ^ rotl32(r.limbs[(i + 1u) % 8u], 7u);
            r.limbs[i] = r.limbs[i] + r.limbs[(i + 3u) % 8u];
        }
    }
    
    while (scalar_gte_mod(r, mod_arr)) {
        var borrow = 0u;
        for (var i = 0u; i < 8u; i++) {
            let diff = r.limbs[i] - mod_arr[i] - borrow;
            borrow = select(0u, 1u, r.limbs[i] < mod_arr[i] + borrow);
            r.limbs[i] = diff;
        }
    }
    
    return r;
}

// ============================================================================
// Kernels
// ============================================================================

@compute @workgroup_size(256)
fn frost_generate_nonces(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.num_participants) { return; }
    
    let mod_arr = get_modulus(params.curve_type);
    let seed = seeds[gid.x];
    
    var rng = chacha20_init(seed, gid.x, params.seed_entropy_offset);
    
    let d = random_scalar(&rng, mod_arr);
    let e = random_scalar(&rng, mod_arr);
    
    let D_ext = ed25519_scalar_mul_base(d, mod_arr);
    let E_ext = ed25519_scalar_mul_base(e, mod_arr);
    
    var result: NonceCommitment;
    result.hiding_nonce_d = d;
    result.binding_nonce_e = e;
    result.commitment_d = ed25519_to_affine(D_ext);
    result.commitment_e = ed25519_to_affine(E_ext);
    
    nonces[gid.x] = result;
}

@compute @workgroup_size(256)
fn frost_compute_binding_factors(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.num_participants) { return; }
    
    let mod_arr = get_modulus(params.curve_type);
    
    var hash_input: array<u32, 16>;
    hash_input[0] = participant_ids[gid.x];
    
    let msg = message[0];
    for (var i = 0u; i < 8u; i++) {
        hash_input[1u + i] = msg.limbs[i];
    }
    
    var commitment_hash = 0u;
    for (var i = 0u; i < params.num_participants * 2u; i++) {
        let c = commitment_list[i];
        commitment_hash = commitment_hash ^ c.x.limbs[0] ^ c.y.limbs[0];
    }
    hash_input[9] = commitment_hash;
    
    let rho = hash_to_scalar(hash_input, 10u, mod_arr);
    binding_factors[gid.x] = rho;
}

@compute @workgroup_size(256)
fn frost_compute_commitment_shares(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.num_participants) { return; }
    
    let mod_arr = get_modulus(params.curve_type);
    let nonce = nonces[gid.x];
    let rho = binding_factors[gid.x];
    
    var rho_E: Ed25519Extended;
    rho_E.x = scalar_mul(rho, nonce.commitment_e.x, mod_arr);
    rho_E.y = scalar_mul(rho, nonce.commitment_e.y, mod_arr);
    rho_E.z = scalar_one();
    rho_E.t = scalar_zero();
    
    var D: Ed25519Extended;
    D.x = nonce.commitment_d.x;
    D.y = nonce.commitment_d.y;
    D.z = scalar_one();
    D.t = scalar_zero();
    
    let R_i = ed25519_add(D, rho_E, mod_arr);
    commitment_shares[gid.x] = R_i;
}

var<workgroup> shared_commitments: array<Ed25519Extended, 256>;

@compute @workgroup_size(256)
fn frost_aggregate_commitments(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let count = params.num_participants;
    let mod_arr = get_modulus(params.curve_type);
    
    if (gid.x < count) {
        shared_commitments[lid.x] = commitment_shares[gid.x];
    } else {
        shared_commitments[lid.x] = ed25519_identity();
    }
    
    workgroupBarrier();
    
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (lid.x < stride && lid.x + stride < count) {
            shared_commitments[lid.x] = ed25519_add(
                shared_commitments[lid.x],
                shared_commitments[lid.x + stride],
                mod_arr
            );
        }
        workgroupBarrier();
    }
    
    if (lid.x == 0u) {
        group_commitment[wgid.x] = shared_commitments[0];
    }
}

@compute @workgroup_size(256)
fn frost_verify_nonce_commitments(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.num_participants) { return; }
    
    let mod_arr = get_modulus(params.curve_type);
    let nonce = nonces[gid.x];
    
    let computed_D = ed25519_scalar_mul_base(nonce.hiding_nonce_d, mod_arr);
    let computed_D_affine = ed25519_to_affine(computed_D);
    
    var d_valid = true;
    for (var i = 0u; i < 8u; i++) {
        if (computed_D_affine.x.limbs[i] != nonce.commitment_d.x.limbs[i]) {
            d_valid = false;
            break;
        }
    }
    
    let computed_E = ed25519_scalar_mul_base(nonce.binding_nonce_e, mod_arr);
    let computed_E_affine = ed25519_to_affine(computed_E);
    
    var e_valid = true;
    for (var i = 0u; i < 8u; i++) {
        if (computed_E_affine.x.limbs[i] != nonce.commitment_e.x.limbs[i]) {
            e_valid = false;
            break;
        }
    }
    
    valid[gid.x] = select(0u, 1u, d_valid && e_valid);
}

@compute @workgroup_size(256)
fn frost_batch_hash_to_curve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.batch_size) { return; }
    
    let mod_arr = get_modulus(params.curve_type);
    var h = message[gid.x];
    
    for (var i = 0u; i < 8u; i++) {
        h.limbs[i] = h.limbs[i] ^ rotl32(h.limbs[(i + 1u) % 8u], 11u);
        h.limbs[i] = h.limbs[i] + rotl32(h.limbs[(i + 5u) % 8u], 7u);
    }
    
    while (scalar_gte_mod(h, mod_arr)) {
        var borrow = 0u;
        for (var i = 0u; i < 8u; i++) {
            let diff = h.limbs[i] - mod_arr[i] - borrow;
            borrow = select(0u, 1u, h.limbs[i] < mod_arr[i] + borrow);
            h.limbs[i] = diff;
        }
    }
    
    commitment_shares[gid.x] = ed25519_scalar_mul_base(h, mod_arr);
}
