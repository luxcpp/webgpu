// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Shamir Secret Sharing Lagrange Interpolation
// GPU-accelerated field interpolation for t-of-n threshold schemes
// Portable WebGPU implementation

// ============================================================================
// Constants
// ============================================================================

const MAX_PARTICIPANTS: u32 = 256u;

const FIELD_ED25519: u32 = 0u;
const FIELD_SECP256K1: u32 = 1u;
const FIELD_BLS12_381: u32 = 2u;
const FIELD_RINGTAIL: u32 = 3u;

// Ed25519 scalar field l
const ED25519_L: array<u32, 8> = array<u32, 8>(
    0x5cf5d3edu, 0x5812631au, 0xa2f79cd6u, 0x14def9deu,
    0x00000000u, 0x00000000u, 0x00000000u, 0x10000000u
);

// secp256k1 scalar field n
const SECP256K1_N: array<u32, 8> = array<u32, 8>(
    0xd0364141u, 0xbfd25e8cu, 0xaf48a03bu, 0xbaaedce6u,
    0xfffffffeu, 0xffffffffu, 0xffffffffu, 0xffffffffu
);

// BLS12-381 scalar field r
const BLS12_381_R: array<u32, 8> = array<u32, 8>(
    0x00000001u, 0xffffffffu, 0xfffe5bfeu, 0x53bda402u,
    0x09a1d805u, 0x3339d808u, 0x299d7d48u, 0x73eda753u
);

// ============================================================================
// Data Types
// ============================================================================

struct Fe256 {
    limbs: array<u32, 8>,
}

struct Share {
    index: u32,
    value: Fe256,
}

struct InterpolateParams {
    num_shares: u32,
    threshold: u32,
    field_type: u32,
    batch_size: u32,
    eval_point: u32,
}

struct LagrangeCache {
    coefficients: array<Fe256, 256>,
    participant_ids: array<u32, 256>,
    num_participants: u32,
}

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> participant_indices: array<u32>;
@group(0) @binding(1) var<storage, read_write> lagrange_coeffs: array<Fe256>;
@group(0) @binding(2) var<uniform> params: InterpolateParams;

@group(1) @binding(0) var<storage, read> shares: array<Share>;
@group(1) @binding(1) var<storage, read_write> results: array<Fe256>;
@group(1) @binding(2) var<storage, read_write> partial_sums: array<Fe256>;

@group(2) @binding(0) var<storage, read_write> cache: array<LagrangeCache>;
@group(2) @binding(1) var<storage, read> eval_points: array<u32>;
@group(2) @binding(2) var<storage, read_write> valid: array<u32>;

@group(3) @binding(0) var<storage, read> old_shares: array<Share>;
@group(3) @binding(1) var<storage, read> old_indices: array<u32>;
@group(3) @binding(2) var<storage, read> new_indices: array<u32>;
@group(3) @binding(3) var<storage, read_write> new_shares: array<Share>;
@group(3) @binding(4) var<storage, read_write> denominators: array<Fe256>;
@group(3) @binding(5) var<storage, read_write> inverses: array<Fe256>;

// ============================================================================
// Field Arithmetic
// ============================================================================

fn fe_zero() -> Fe256 {
    var r: Fe256;
    for (var i = 0u; i < 8u; i++) {
        r.limbs[i] = 0u;
    }
    return r;
}

fn fe_one() -> Fe256 {
    var r = fe_zero();
    r.limbs[0] = 1u;
    return r;
}

fn fe_is_zero(a: Fe256) -> bool {
    for (var i = 0u; i < 8u; i++) {
        if (a.limbs[i] != 0u) { return false; }
    }
    return true;
}

fn fe_eq(a: Fe256, b: Fe256) -> bool {
    for (var i = 0u; i < 8u; i++) {
        if (a.limbs[i] != b.limbs[i]) { return false; }
    }
    return true;
}

fn get_modulus(field_type: u32) -> array<u32, 8> {
    if (field_type == FIELD_ED25519) {
        return ED25519_L;
    } else if (field_type == FIELD_SECP256K1) {
        return SECP256K1_N;
    } else if (field_type == FIELD_BLS12_381) {
        return BLS12_381_R;
    }
    return ED25519_L;
}

fn fe_gte_mod(a: Fe256, mod_arr: array<u32, 8>) -> bool {
    for (var i = 7; i >= 0; i--) {
        if (a.limbs[i] > mod_arr[i]) { return true; }
        if (a.limbs[i] < mod_arr[i]) { return false; }
    }
    return true;
}

fn fe_add_mod(a: Fe256, b: Fe256, mod_arr: array<u32, 8>) -> Fe256 {
    var r: Fe256;
    var carry = 0u;
    
    for (var i = 0u; i < 8u; i++) {
        let sum = a.limbs[i] + b.limbs[i] + carry;
        carry = select(0u, 1u, sum < a.limbs[i] || (carry == 1u && sum == a.limbs[i]));
        r.limbs[i] = sum;
    }
    
    if (carry == 1u || fe_gte_mod(r, mod_arr)) {
        var borrow = 0u;
        for (var i = 0u; i < 8u; i++) {
            let diff = r.limbs[i] - mod_arr[i] - borrow;
            borrow = select(0u, 1u, r.limbs[i] < mod_arr[i] + borrow);
            r.limbs[i] = diff;
        }
    }
    
    return r;
}

fn fe_sub_mod(a: Fe256, b: Fe256, mod_arr: array<u32, 8>) -> Fe256 {
    var r: Fe256;
    var borrow = 0u;
    
    for (var i = 0u; i < 8u; i++) {
        let diff = a.limbs[i] - b.limbs[i] - borrow;
        borrow = select(0u, 1u, a.limbs[i] < b.limbs[i] + borrow);
        r.limbs[i] = diff;
    }
    
    if (borrow == 1u) {
        var carry = 0u;
        for (var i = 0u; i < 8u; i++) {
            let sum = r.limbs[i] + mod_arr[i] + carry;
            carry = select(0u, 1u, sum < r.limbs[i]);
            r.limbs[i] = sum;
        }
    }
    
    return r;
}

fn fe_neg_mod(a: Fe256, mod_arr: array<u32, 8>) -> Fe256 {
    if (fe_is_zero(a)) { return a; }
    
    var r: Fe256;
    var borrow = 0u;
    for (var i = 0u; i < 8u; i++) {
        let diff = mod_arr[i] - a.limbs[i] - borrow;
        borrow = select(0u, 1u, mod_arr[i] < a.limbs[i] + borrow);
        r.limbs[i] = diff;
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

fn fe_mul_mod(a: Fe256, b: Fe256, mod_arr: array<u32, 8>) -> Fe256 {
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
    
    var r: Fe256;
    for (var i = 0u; i < 8u; i++) {
        r.limbs[i] = product[i];
    }
    
    while (fe_gte_mod(r, mod_arr)) {
        var borrow = 0u;
        for (var i = 0u; i < 8u; i++) {
            let diff = r.limbs[i] - mod_arr[i] - borrow;
            borrow = select(0u, 1u, r.limbs[i] < mod_arr[i] + borrow);
            r.limbs[i] = diff;
        }
    }
    
    return r;
}

fn fe_from_uint(x: u32) -> Fe256 {
    var r = fe_zero();
    r.limbs[0] = x;
    return r;
}

fn fe_inv_mod(a: Fe256, mod_arr: array<u32, 8>) -> Fe256 {
    var exp = fe_zero();
    for (var i = 0u; i < 8u; i++) {
        exp.limbs[i] = mod_arr[i];
    }
    
    // exp = mod - 2
    if (exp.limbs[0] >= 2u) {
        exp.limbs[0] = exp.limbs[0] - 2u;
    } else {
        exp.limbs[0] = 0xFFFFFFFEu;
        var borrow = 1u;
        for (var i = 1u; i < 8u; i++) {
            if (borrow == 0u) { break; }
            if (exp.limbs[i] > 0u) {
                exp.limbs[i] = exp.limbs[i] - 1u;
                borrow = 0u;
            } else {
                exp.limbs[i] = 0xFFFFFFFFu;
            }
        }
    }
    
    var result = fe_one();
    var base = a;
    
    for (var bit = 0u; bit < 256u; bit++) {
        let limb_idx = bit / 32u;
        let bit_in_limb = bit % 32u;
        
        if ((exp.limbs[limb_idx] & (1u << bit_in_limb)) != 0u) {
            result = fe_mul_mod(result, base, mod_arr);
        }
        base = fe_mul_mod(base, base, mod_arr);
    }
    
    return result;
}

// ============================================================================
// Lagrange Interpolation Functions
// ============================================================================

fn lagrange_at_zero(
    i: u32,
    num_participants: u32,
    mod_arr: array<u32, 8>
) -> Fe256 {
    var numerator = fe_one();
    var denominator = fe_one();
    
    for (var j = 0u; j < num_participants; j++) {
        let j_idx = participant_indices[j];
        if (j_idx == i) { continue; }
        
        // numerator *= -j_idx
        let j_fe = fe_from_uint(j_idx);
        let neg_j = fe_neg_mod(j_fe, mod_arr);
        numerator = fe_mul_mod(numerator, neg_j, mod_arr);
        
        // denominator *= (i - j_idx)
        let i_fe = fe_from_uint(i);
        let denom_term = fe_sub_mod(i_fe, j_fe, mod_arr);
        denominator = fe_mul_mod(denominator, denom_term, mod_arr);
    }
    
    let denom_inv = fe_inv_mod(denominator, mod_arr);
    return fe_mul_mod(numerator, denom_inv, mod_arr);
}

fn lagrange_coefficient(
    i: u32,
    eval_point: u32,
    num_participants: u32,
    mod_arr: array<u32, 8>
) -> Fe256 {
    var numerator = fe_one();
    var denominator = fe_one();
    
    let eval_fe = fe_from_uint(eval_point);
    let i_fe = fe_from_uint(i);
    
    for (var j = 0u; j < num_participants; j++) {
        let j_idx = participant_indices[j];
        if (j_idx == i) { continue; }
        
        let j_fe = fe_from_uint(j_idx);
        
        let num_term = fe_sub_mod(eval_fe, j_fe, mod_arr);
        numerator = fe_mul_mod(numerator, num_term, mod_arr);
        
        let denom_term = fe_sub_mod(i_fe, j_fe, mod_arr);
        denominator = fe_mul_mod(denominator, denom_term, mod_arr);
    }
    
    let denom_inv = fe_inv_mod(denominator, mod_arr);
    return fe_mul_mod(numerator, denom_inv, mod_arr);
}

// ============================================================================
// Kernels
// ============================================================================

@compute @workgroup_size(256)
fn shamir_compute_lagrange_coeffs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.num_shares) { return; }
    
    let mod_arr = get_modulus(params.field_type);
    let my_index = participant_indices[gid.x];
    
    var lambda: Fe256;
    if (params.eval_point == 0u) {
        lambda = lagrange_at_zero(my_index, params.num_shares, mod_arr);
    } else {
        lambda = lagrange_coefficient(my_index, params.eval_point, params.num_shares, mod_arr);
    }
    
    lagrange_coeffs[gid.x] = lambda;
}

@compute @workgroup_size(1)
fn shamir_interpolate_single(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    
    let mod_arr = get_modulus(params.field_type);
    var sum = fe_zero();
    
    for (var i = 0u; i < params.num_shares; i++) {
        let term = fe_mul_mod(lagrange_coeffs[i], shares[i].value, mod_arr);
        sum = fe_add_mod(sum, term, mod_arr);
    }
    
    results[0] = sum;
}

@compute @workgroup_size(256)
fn shamir_interpolate_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.batch_size) { return; }
    
    let mod_arr = get_modulus(params.field_type);
    let offset = gid.x * params.num_shares;
    
    var sum = fe_zero();
    for (var i = 0u; i < params.num_shares; i++) {
        let term = fe_mul_mod(lagrange_coeffs[i], shares[offset + i].value, mod_arr);
        sum = fe_add_mod(sum, term, mod_arr);
    }
    
    results[gid.x] = sum;
}

var<workgroup> shared_sums: array<Fe256, 256>;

@compute @workgroup_size(256)
fn shamir_parallel_interpolate(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    if (gid.x >= params.num_shares) { return; }
    
    let mod_arr = get_modulus(params.field_type);
    
    let term = fe_mul_mod(lagrange_coeffs[gid.x], shares[gid.x].value, mod_arr);
    shared_sums[lid.x] = term;
    
    workgroupBarrier();
    
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (lid.x < stride && lid.x + stride < params.num_shares) {
            shared_sums[lid.x] = fe_add_mod(shared_sums[lid.x], shared_sums[lid.x + stride], mod_arr);
        }
        workgroupBarrier();
    }
    
    if (lid.x == 0u) {
        partial_sums[wgid.x] = shared_sums[0];
    }
}

@compute @workgroup_size(256)
fn shamir_evaluate_poly(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.batch_size) { return; }
    
    let mod_arr = get_modulus(params.field_type);
    let eval_point = eval_points[gid.x];
    
    var sum = fe_zero();
    
    for (var i = 0u; i < params.num_shares; i++) {
        let lambda = lagrange_coefficient(participant_indices[i], eval_point, params.num_shares, mod_arr);
        let term = fe_mul_mod(lambda, shares[i].value, mod_arr);
        sum = fe_add_mod(sum, term, mod_arr);
    }
    
    results[gid.x] = sum;
}

@compute @workgroup_size(256)
fn shamir_verify_shares(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.num_shares) { return; }
    
    // Placeholder: mark as valid
    valid[gid.x] = 1u;
}

@compute @workgroup_size(256)
fn shamir_reshare(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.batch_size) { return; }
    
    let mod_arr = get_modulus(params.field_type);
    let new_index = new_indices[gid.x];
    
    var new_value = fe_zero();
    
    for (var i = 0u; i < params.num_shares; i++) {
        let lambda = lagrange_coefficient(old_indices[i], new_index, params.num_shares, mod_arr);
        let term = fe_mul_mod(lambda, old_shares[i].value, mod_arr);
        new_value = fe_add_mod(new_value, term, mod_arr);
    }
    
    new_shares[gid.x].index = new_index;
    new_shares[gid.x].value = new_value;
}

@compute @workgroup_size(256)
fn shamir_batch_denominators(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.num_shares) { return; }
    
    let mod_arr = get_modulus(params.field_type);
    let my_index = participant_indices[gid.x];
    var denom = fe_one();
    
    for (var j = 0u; j < params.num_shares; j++) {
        let j_idx = participant_indices[j];
        if (j_idx == my_index) { continue; }
        
        let i_fe = fe_from_uint(my_index);
        let j_fe = fe_from_uint(j_idx);
        let diff = fe_sub_mod(i_fe, j_fe, mod_arr);
        denom = fe_mul_mod(denom, diff, mod_arr);
    }
    
    denominators[gid.x] = denom;
}

@compute @workgroup_size(1)
fn shamir_batch_invert(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    
    let mod_arr = get_modulus(params.field_type);
    let n = params.num_shares;
    
    // Montgomery's batch inversion
    var products: array<Fe256, 256>;
    products[0] = denominators[0];
    
    for (var i = 1u; i < n; i++) {
        products[i] = fe_mul_mod(products[i - 1u], denominators[i], mod_arr);
    }
    
    var all_inv = fe_inv_mod(products[n - 1u], mod_arr);
    
    for (var i = n - 1u; i > 0u; i--) {
        inverses[i] = fe_mul_mod(all_inv, products[i - 1u], mod_arr);
        all_inv = fe_mul_mod(all_inv, denominators[i], mod_arr);
    }
    inverses[0] = all_inv;
}
