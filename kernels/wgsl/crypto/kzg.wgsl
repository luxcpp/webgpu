// =============================================================================
// KZG Polynomial Commitment WebGPU Compute Shaders
// =============================================================================
//
// GPU-accelerated KZG polynomial commitments for EIP-4844 blobs.
// Uses BLS12-381 curve operations.
//
// KZG Parameters:
//   - Uses BLS12-381 G1/G2 for commitments and proofs
//   - Polynomial degree up to 4096 (blob elements)
//   - Trusted setup from Ethereum KZG ceremony
//
// References:
//   - EIP-4844: Shard Blob Transactions
//   - KZG Commitments paper (Kate, Zaverucha, Goldberg 2010)
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// BLS12-381 Constants
// =============================================================================

// BLS12-381 base field prime p (6 limbs, little-endian)
// p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
const BLS_P_0: u64 = 0xb9feffffffffaaabu64;
const BLS_P_1: u64 = 0x1eabfffeb153ffffu64;
const BLS_P_2: u64 = 0x6730d2a0f6b0f624u64;
const BLS_P_3: u64 = 0x64774b84f38512bfu64;
const BLS_P_4: u64 = 0x4b1ba7b6434bacd7u64;
const BLS_P_5: u64 = 0x1a0111ea397fe69au64;

// Scalar field r (for polynomial coefficients)
// r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
const BLS_R_0: u64 = 0xffffffff00000001u64;
const BLS_R_1: u64 = 0x53bda402fffe5bfeu64;
const BLS_R_2: u64 = 0x3339d80809a1d805u64;
const BLS_R_3: u64 = 0x73eda753299d7d48u64;

// Montgomery constant for scalar field: -r^{-1} mod 2^64
const BLS_R_INV: u64 = 0xfffffffeffffffffu64;

// Montgomery R^2 mod r (for converting to Montgomery form)
const BLS_R2_MONT_0: u64 = 0xc999e990f3f29c6du64;
const BLS_R2_MONT_1: u64 = 0x2b6cedcb87925c23u64;
const BLS_R2_MONT_2: u64 = 0x05d314967254398fu64;
const BLS_R2_MONT_3: u64 = 0x0748d9d99f59ff11u64;

// One in Montgomery form for scalar field
const FR_ONE_MONT_0: u64 = 0xFFFE5BFEFFFFFFFFu64;
const FR_ONE_MONT_1: u64 = 0x09A1D80553BDA402u64;
const FR_ONE_MONT_2: u64 = 0x299D7D483339D808u64;
const FR_ONE_MONT_3: u64 = 0x0073EDA753299D7Du64;

// =============================================================================
// Data Types
// =============================================================================

// Fp384: 384-bit field element (6 x 64-bit limbs)
// WGSL doesn't have u64, so we use vec2<u32> pairs
struct Fp384 {
    // 6 limbs stored as 12 x u32 (low, high pairs)
    l0: vec2<u32>,  // limbs[0]
    l1: vec2<u32>,  // limbs[1]
    l2: vec2<u32>,  // limbs[2]
    l3: vec2<u32>,  // limbs[3]
    l4: vec2<u32>,  // limbs[4]
    l5: vec2<u32>,  // limbs[5]
}

// Fr256: 256-bit scalar field element (4 x 64-bit limbs)
struct Fr256 {
    l0: vec2<u32>,  // limbs[0]
    l1: vec2<u32>,  // limbs[1]
    l2: vec2<u32>,  // limbs[2]
    l3: vec2<u32>,  // limbs[3]
}

// G1 affine point
struct G1Affine {
    x: Fp384,
    y: Fp384,
    infinity: u32,
    _pad: u32,
}

// G1 projective point (Jacobian coordinates)
struct G1Projective {
    x: Fp384,
    y: Fp384,
    z: Fp384,
}

// KZG parameters for compute kernels
struct KZGParams {
    degree: u32,           // Polynomial degree
    num_points: u32,       // Number of points in MSM
    window_size: u32,      // Window size for Pippenger
    window_idx: u32,       // Current window index
    num_proofs: u32,       // Number of proofs in batch verification
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// =============================================================================
// Buffer Bindings
// =============================================================================

// Group 0: Primary data buffers
@group(0) @binding(0) var<storage, read> poly_coeffs: array<Fr256>;
@group(0) @binding(1) var<storage, read_write> result_coeffs: array<Fr256>;
@group(0) @binding(2) var<storage, read> eval_point: Fr256;
@group(0) @binding(3) var<uniform> params: KZGParams;

// Group 1: MSM buffers
@group(1) @binding(0) var<storage, read> msm_bases: array<G1Affine>;
@group(1) @binding(1) var<storage, read> msm_scalars: array<Fr256>;
@group(1) @binding(2) var<storage, read_write> msm_buckets: array<G1Projective>;
@group(1) @binding(3) var<storage, read_write> msm_result: array<G1Projective>;

// Group 2: Batch verification buffers
@group(2) @binding(0) var<storage, read> batch_commitments: array<G1Affine>;
@group(2) @binding(1) var<storage, read> batch_witnesses: array<G1Affine>;
@group(2) @binding(2) var<storage, read> batch_points: array<Fr256>;
@group(2) @binding(3) var<storage, read> batch_values: array<Fr256>;
@group(2) @binding(4) var<storage, read> batch_challenge: Fr256;
@group(2) @binding(5) var<storage, read_write> batch_lhs_accum: array<G1Projective>;
@group(2) @binding(6) var<storage, read_write> batch_rhs_accum: array<G1Projective>;

// =============================================================================
// 64-bit Arithmetic Helpers (using vec2<u32>)
// =============================================================================

// Pack two u32 into logical u64
fn pack_u64(lo: u32, hi: u32) -> vec2<u32> {
    return vec2<u32>(lo, hi);
}

// Unpack vec2 to (low, high)
fn unpack_u64(v: vec2<u32>) -> vec2<u32> {
    return v;
}

// Add with carry: a + b + carry_in -> (result, carry_out)
fn adc_u64(a: vec2<u32>, b: vec2<u32>, carry_in: u32) -> vec2<vec2<u32>> {
    // Add low parts
    let sum_lo = u64(a.x) + u64(b.x) + u64(carry_in);
    let result_lo = u32(sum_lo);
    let carry_mid = u32(sum_lo >> 32u);

    // Add high parts
    let sum_hi = u64(a.y) + u64(b.y) + u64(carry_mid);
    let result_hi = u32(sum_hi);
    let carry_out = u32(sum_hi >> 32u);

    return vec2<vec2<u32>>(
        vec2<u32>(result_lo, result_hi),
        vec2<u32>(carry_out, 0u)
    );
}

// Subtract with borrow: a - b - borrow_in -> (result, borrow_out)
fn sbb_u64(a: vec2<u32>, b: vec2<u32>, borrow_in: u32) -> vec2<vec2<u32>> {
    // Subtract low parts
    let a_lo = u64(a.x);
    let b_lo = u64(b.x) + u64(borrow_in);
    var borrow_mid = 0u;
    var result_lo: u32;

    if a_lo < b_lo {
        result_lo = u32((a_lo + 0x100000000u64) - b_lo);
        borrow_mid = 1u;
    } else {
        result_lo = u32(a_lo - b_lo);
    }

    // Subtract high parts
    let a_hi = u64(a.y);
    let b_hi = u64(b.y) + u64(borrow_mid);
    var borrow_out = 0u;
    var result_hi: u32;

    if a_hi < b_hi {
        result_hi = u32((a_hi + 0x100000000u64) - b_hi);
        borrow_out = 1u;
    } else {
        result_hi = u32(a_hi - b_hi);
    }

    return vec2<vec2<u32>>(
        vec2<u32>(result_lo, result_hi),
        vec2<u32>(borrow_out, 0u)
    );
}

// 64x64 -> 128 bit multiplication (returns low and high 64-bit parts)
fn mul_u64(a: vec2<u32>, b: vec2<u32>) -> vec2<vec2<u32>> {
    // Karatsuba-style: (a_hi * 2^32 + a_lo) * (b_hi * 2^32 + b_lo)
    let a_lo = u64(a.x);
    let a_hi = u64(a.y);
    let b_lo = u64(b.x);
    let b_hi = u64(b.y);

    // z0 = a_lo * b_lo (64-bit result)
    let z0 = a_lo * b_lo;

    // z2 = a_hi * b_hi (64-bit result)
    let z2 = a_hi * b_hi;

    // z1 = a_lo * b_hi + a_hi * b_lo (can overflow to 65 bits)
    let z1a = a_lo * b_hi;
    let z1b = a_hi * b_lo;
    let z1 = z1a + z1b;
    let z1_carry = select(0u64, 1u64, z1 < z1a);  // Overflow detection

    // Combine: result = z0 + (z1 << 32) + (z2 << 64)
    let lo_part = z0 + ((z1 & 0xFFFFFFFFu64) << 32u);
    let lo_carry = select(0u64, 1u64, lo_part < z0);

    let hi_part = z2 + (z1 >> 32u) + (z1_carry << 32u) + lo_carry;

    return vec2<vec2<u32>>(
        vec2<u32>(u32(lo_part), u32(lo_part >> 32u)),
        vec2<u32>(u32(hi_part), u32(hi_part >> 32u))
    );
}

// =============================================================================
// Scalar Field (Fr) Operations
// =============================================================================

fn fr_zero() -> Fr256 {
    var r: Fr256;
    r.l0 = vec2<u32>(0u, 0u);
    r.l1 = vec2<u32>(0u, 0u);
    r.l2 = vec2<u32>(0u, 0u);
    r.l3 = vec2<u32>(0u, 0u);
    return r;
}

fn fr_one() -> Fr256 {
    // One in Montgomery form
    var r: Fr256;
    r.l0 = vec2<u32>(u32(FR_ONE_MONT_0), u32(FR_ONE_MONT_0 >> 32u));
    r.l1 = vec2<u32>(u32(FR_ONE_MONT_1), u32(FR_ONE_MONT_1 >> 32u));
    r.l2 = vec2<u32>(u32(FR_ONE_MONT_2), u32(FR_ONE_MONT_2 >> 32u));
    r.l3 = vec2<u32>(u32(FR_ONE_MONT_3), u32(FR_ONE_MONT_3 >> 32u));
    return r;
}

fn fr_is_zero(a: Fr256) -> bool {
    return a.l0.x == 0u && a.l0.y == 0u &&
           a.l1.x == 0u && a.l1.y == 0u &&
           a.l2.x == 0u && a.l2.y == 0u &&
           a.l3.x == 0u && a.l3.y == 0u;
}

// Compare Fr256 with BLS_R: returns -1 if a < r, 0 if a == r, 1 if a > r
fn fr_cmp_r(a: Fr256) -> i32 {
    // Compare from most significant limb
    let r3 = vec2<u32>(u32(BLS_R_3), u32(BLS_R_3 >> 32u));
    if a.l3.y > r3.y { return 1; }
    if a.l3.y < r3.y { return -1; }
    if a.l3.x > r3.x { return 1; }
    if a.l3.x < r3.x { return -1; }

    let r2 = vec2<u32>(u32(BLS_R_2), u32(BLS_R_2 >> 32u));
    if a.l2.y > r2.y { return 1; }
    if a.l2.y < r2.y { return -1; }
    if a.l2.x > r2.x { return 1; }
    if a.l2.x < r2.x { return -1; }

    let r1 = vec2<u32>(u32(BLS_R_1), u32(BLS_R_1 >> 32u));
    if a.l1.y > r1.y { return 1; }
    if a.l1.y < r1.y { return -1; }
    if a.l1.x > r1.x { return 1; }
    if a.l1.x < r1.x { return -1; }

    let r0 = vec2<u32>(u32(BLS_R_0), u32(BLS_R_0 >> 32u));
    if a.l0.y > r0.y { return 1; }
    if a.l0.y < r0.y { return -1; }
    if a.l0.x > r0.x { return 1; }
    if a.l0.x < r0.x { return -1; }

    return 0;
}

// Reduce mod r if >= r
fn fr_reduce(a: ptr<function, Fr256>) {
    if fr_cmp_r(*a) >= 0 {
        let r0 = vec2<u32>(u32(BLS_R_0), u32(BLS_R_0 >> 32u));
        let r1 = vec2<u32>(u32(BLS_R_1), u32(BLS_R_1 >> 32u));
        let r2 = vec2<u32>(u32(BLS_R_2), u32(BLS_R_2 >> 32u));
        let r3 = vec2<u32>(u32(BLS_R_3), u32(BLS_R_3 >> 32u));

        let sub0 = sbb_u64((*a).l0, r0, 0u);
        let sub1 = sbb_u64((*a).l1, r1, sub0[1].x);
        let sub2 = sbb_u64((*a).l2, r2, sub1[1].x);
        let sub3 = sbb_u64((*a).l3, r3, sub2[1].x);

        (*a).l0 = sub0[0];
        (*a).l1 = sub1[0];
        (*a).l2 = sub2[0];
        (*a).l3 = sub3[0];
    }
}

// Fr addition: c = a + b mod r
fn fr_add(a: Fr256, b: Fr256) -> Fr256 {
    var c: Fr256;

    let add0 = adc_u64(a.l0, b.l0, 0u);
    let add1 = adc_u64(a.l1, b.l1, add0[1].x);
    let add2 = adc_u64(a.l2, b.l2, add1[1].x);
    let add3 = adc_u64(a.l3, b.l3, add2[1].x);

    c.l0 = add0[0];
    c.l1 = add1[0];
    c.l2 = add2[0];
    c.l3 = add3[0];

    fr_reduce(&c);
    return c;
}

// Fr subtraction: c = a - b mod r
fn fr_sub(a: Fr256, b: Fr256) -> Fr256 {
    var c: Fr256;

    let sub0 = sbb_u64(a.l0, b.l0, 0u);
    let sub1 = sbb_u64(a.l1, b.l1, sub0[1].x);
    let sub2 = sbb_u64(a.l2, b.l2, sub1[1].x);
    let sub3 = sbb_u64(a.l3, b.l3, sub2[1].x);

    c.l0 = sub0[0];
    c.l1 = sub1[0];
    c.l2 = sub2[0];
    c.l3 = sub3[0];

    // If underflow (borrow), add r
    if sub3[1].x != 0u {
        let r0 = vec2<u32>(u32(BLS_R_0), u32(BLS_R_0 >> 32u));
        let r1 = vec2<u32>(u32(BLS_R_1), u32(BLS_R_1 >> 32u));
        let r2 = vec2<u32>(u32(BLS_R_2), u32(BLS_R_2 >> 32u));
        let r3 = vec2<u32>(u32(BLS_R_3), u32(BLS_R_3 >> 32u));

        let add0 = adc_u64(c.l0, r0, 0u);
        let add1 = adc_u64(c.l1, r1, add0[1].x);
        let add2 = adc_u64(c.l2, r2, add1[1].x);
        let add3 = adc_u64(c.l3, r3, add2[1].x);

        c.l0 = add0[0];
        c.l1 = add1[0];
        c.l2 = add2[0];
        c.l3 = add3[0];
    }

    return c;
}

// Montgomery multiplication for Fr
// This is a simplified version - production would use CIOS or similar
fn fr_mont_mul(a: Fr256, b: Fr256) -> Fr256 {
    // 8 limb result buffer (512 bits)
    var t: array<vec2<u32>, 8>;
    for (var i = 0u; i < 8u; i++) {
        t[i] = vec2<u32>(0u, 0u);
    }

    // Get limb arrays
    var a_limbs: array<vec2<u32>, 4>;
    a_limbs[0] = a.l0; a_limbs[1] = a.l1; a_limbs[2] = a.l2; a_limbs[3] = a.l3;

    var b_limbs: array<vec2<u32>, 4>;
    b_limbs[0] = b.l0; b_limbs[1] = b.l1; b_limbs[2] = b.l2; b_limbs[3] = b.l3;

    var r_limbs: array<vec2<u32>, 4>;
    r_limbs[0] = vec2<u32>(u32(BLS_R_0), u32(BLS_R_0 >> 32u));
    r_limbs[1] = vec2<u32>(u32(BLS_R_1), u32(BLS_R_1 >> 32u));
    r_limbs[2] = vec2<u32>(u32(BLS_R_2), u32(BLS_R_2 >> 32u));
    r_limbs[3] = vec2<u32>(u32(BLS_R_3), u32(BLS_R_3 >> 32u));

    // Schoolbook multiplication
    for (var i = 0u; i < 4u; i++) {
        var carry = vec2<u32>(0u, 0u);
        for (var j = 0u; j < 4u; j++) {
            let prod = mul_u64(a_limbs[i], b_limbs[j]);
            let add1 = adc_u64(t[i + j], prod[0], 0u);
            let add2 = adc_u64(add1[0], carry, 0u);
            t[i + j] = add2[0];
            carry = adc_u64(prod[1], vec2<u32>(add1[1].x + add2[1].x, 0u), 0u)[0];
        }
        t[i + 4u] = carry;
    }

    // Montgomery reduction
    let r_inv = vec2<u32>(u32(BLS_R_INV), u32(BLS_R_INV >> 32u));

    for (var i = 0u; i < 4u; i++) {
        // k = t[i] * r_inv mod 2^64
        let k = mul_u64(t[i], r_inv)[0];

        var carry = vec2<u32>(0u, 0u);
        for (var j = 0u; j < 4u; j++) {
            let prod = mul_u64(k, r_limbs[j]);
            let add1 = adc_u64(t[i + j], prod[0], 0u);
            let add2 = adc_u64(add1[0], carry, 0u);
            t[i + j] = add2[0];
            carry = adc_u64(prod[1], vec2<u32>(add1[1].x + add2[1].x, 0u), 0u)[0];
        }

        // Propagate carry
        for (var j = i + 4u; j < 8u; j++) {
            let add_result = adc_u64(t[j], carry, 0u);
            t[j] = add_result[0];
            carry = add_result[1];
            if carry.x == 0u { break; }
        }
    }

    // Result is in upper 4 limbs
    var c: Fr256;
    c.l0 = t[4];
    c.l1 = t[5];
    c.l2 = t[6];
    c.l3 = t[7];

    fr_reduce(&c);
    return c;
}

// =============================================================================
// Polynomial Operations - Horner's Method
// =============================================================================

// Evaluate polynomial at point using Horner's method
// p(x) = c_n * x^n + c_{n-1} * x^{n-1} + ... + c_1 * x + c_0
// Horner: p(x) = (...((c_n * x + c_{n-1}) * x + c_{n-2}) * x + ...) * x + c_0
@compute @workgroup_size(256)
fn poly_evaluate_horner(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    // Only thread 0 performs the sequential evaluation
    if idx != 0u { return; }

    let degree = params.degree;
    let point = eval_point;

    var result = fr_zero();

    // Horner's method: start from highest coefficient
    for (var i = i32(degree); i >= 0; i--) {
        result = fr_mont_mul(result, point);
        result = fr_add(result, poly_coeffs[u32(i)]);
    }

    result_coeffs[0] = result;
}

// Parallel polynomial evaluation at multiple points
// Each thread evaluates the polynomial at a different point
@compute @workgroup_size(256)
fn poly_evaluate_batch(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if idx >= params.num_points { return; }

    let degree = params.degree;
    let point = msm_scalars[idx];  // Using scalars buffer for eval points

    var result = fr_zero();

    // Horner's method
    for (var i = i32(degree); i >= 0; i--) {
        result = fr_mont_mul(result, point);
        result = fr_add(result, poly_coeffs[u32(i)]);
    }

    result_coeffs[idx] = result;
}

// =============================================================================
// Polynomial Quotient - Synthetic Division
// =============================================================================

// Compute polynomial quotient: q(x) = (p(x) - p(z)) / (x - z)
// Used for KZG proof generation
// q[i] = p[i+1] + z * q[i+1]
@compute @workgroup_size(256)
fn poly_quotient(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    // Only thread 0 performs synthetic division (sequential dependency)
    if idx != 0u { return; }

    let degree = params.degree;
    let z = eval_point;

    // First, evaluate p(z) using Horner's method
    var p_z = fr_zero();
    for (var i = i32(degree); i >= 0; i--) {
        p_z = fr_mont_mul(p_z, z);
        p_z = fr_add(p_z, poly_coeffs[u32(i)]);
    }

    // Synthetic division: q[i] = p[i+1] + z * q[i+1]
    // Working backwards from highest degree
    if degree > 0u {
        result_coeffs[degree - 1u] = poly_coeffs[degree];

        for (var i = i32(degree) - 2; i >= 0; i--) {
            let zi = u32(i);
            let prev = fr_mont_mul(z, result_coeffs[zi + 1u]);
            result_coeffs[zi] = fr_add(poly_coeffs[zi + 1u], prev);
        }
    }
}

// =============================================================================
// G1 Point Operations (Simplified for MSM)
// =============================================================================

fn g1_identity() -> G1Projective {
    var p: G1Projective;
    // x = 0, y = 1, z = 0 (point at infinity)
    p.x.l0 = vec2<u32>(0u, 0u);
    p.x.l1 = vec2<u32>(0u, 0u);
    p.x.l2 = vec2<u32>(0u, 0u);
    p.x.l3 = vec2<u32>(0u, 0u);
    p.x.l4 = vec2<u32>(0u, 0u);
    p.x.l5 = vec2<u32>(0u, 0u);

    p.y.l0 = vec2<u32>(1u, 0u);
    p.y.l1 = vec2<u32>(0u, 0u);
    p.y.l2 = vec2<u32>(0u, 0u);
    p.y.l3 = vec2<u32>(0u, 0u);
    p.y.l4 = vec2<u32>(0u, 0u);
    p.y.l5 = vec2<u32>(0u, 0u);

    p.z.l0 = vec2<u32>(0u, 0u);
    p.z.l1 = vec2<u32>(0u, 0u);
    p.z.l2 = vec2<u32>(0u, 0u);
    p.z.l3 = vec2<u32>(0u, 0u);
    p.z.l4 = vec2<u32>(0u, 0u);
    p.z.l5 = vec2<u32>(0u, 0u);

    return p;
}

fn g1_is_identity(p: G1Projective) -> bool {
    return p.z.l0.x == 0u && p.z.l0.y == 0u &&
           p.z.l1.x == 0u && p.z.l1.y == 0u &&
           p.z.l2.x == 0u && p.z.l2.y == 0u &&
           p.z.l3.x == 0u && p.z.l3.y == 0u &&
           p.z.l4.x == 0u && p.z.l4.y == 0u &&
           p.z.l5.x == 0u && p.z.l5.y == 0u;
}

fn affine_to_projective(a: G1Affine) -> G1Projective {
    var p: G1Projective;
    if a.infinity != 0u {
        return g1_identity();
    }
    p.x = a.x;
    p.y = a.y;
    // Z = 1 (in actual impl, should be Montgomery form)
    p.z.l0 = vec2<u32>(1u, 0u);
    p.z.l1 = vec2<u32>(0u, 0u);
    p.z.l2 = vec2<u32>(0u, 0u);
    p.z.l3 = vec2<u32>(0u, 0u);
    p.z.l4 = vec2<u32>(0u, 0u);
    p.z.l5 = vec2<u32>(0u, 0u);
    return p;
}

// =============================================================================
// Multi-Scalar Multiplication (MSM) - Bucket Method
// =============================================================================

// Extract window bits from scalar
fn extract_window_bits(scalar: Fr256, window_idx: u32, window_size: u32) -> u32 {
    let bit_offset = window_idx * window_size;
    let limb_idx = bit_offset / 64u;
    let bit_idx = bit_offset % 64u;

    var limb: vec2<u32>;
    switch limb_idx {
        case 0u: { limb = scalar.l0; }
        case 1u: { limb = scalar.l1; }
        case 2u: { limb = scalar.l2; }
        case 3u: { limb = scalar.l3; }
        default: { limb = vec2<u32>(0u, 0u); }
    }

    // Convert to single 64-bit value for extraction
    let val64 = u64(limb.x) | (u64(limb.y) << 32u);
    var window_val = u32(val64 >> bit_idx);

    // Handle case where window spans two limbs
    if bit_idx + window_size > 64u && limb_idx + 1u < 4u {
        var next_limb: vec2<u32>;
        switch limb_idx + 1u {
            case 1u: { next_limb = scalar.l1; }
            case 2u: { next_limb = scalar.l2; }
            case 3u: { next_limb = scalar.l3; }
            default: { next_limb = vec2<u32>(0u, 0u); }
        }
        let next_val64 = u64(next_limb.x) | (u64(next_limb.y) << 32u);
        window_val |= u32(next_val64 << (64u - bit_idx));
    }

    let mask = (1u << window_size) - 1u;
    return window_val & mask;
}

// Pippenger bucket accumulation phase
// Each thread processes one point and adds it to the appropriate bucket
@compute @workgroup_size(256)
fn msm_bucket_accumulate(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if idx >= params.num_points { return; }

    let scalar = msm_scalars[idx];
    let base = msm_bases[idx];

    // Skip if base is infinity
    if base.infinity != 0u { return; }

    // Extract window bits for current window
    let window_val = extract_window_bits(scalar, params.window_idx, params.window_size);

    // Skip zero bucket (handled separately)
    if window_val == 0u { return; }

    let bucket_idx = window_val - 1u;

    // Convert affine to projective
    let point = affine_to_projective(base);

    // Atomic accumulation would be needed here
    // For now, just store the point in the bucket
    // Production impl would use atomics or separate accumulation passes
    msm_buckets[bucket_idx] = point;
}

// Bucket reduction phase - combines buckets for final MSM result
// Uses the formula: sum = sum_{i=1}^{2^w-1} i * B_i = sum_{i=1}^{2^w-1} sum_{j=i}^{2^w-1} B_j
@compute @workgroup_size(256)
fn msm_bucket_reduce(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    // Only thread 0 performs reduction
    if idx != 0u { return; }

    let num_buckets = (1u << params.window_size) - 1u;

    var sum = g1_identity();
    var running = g1_identity();

    // Process buckets from highest to lowest
    for (var i = i32(num_buckets) - 1; i >= 0; i--) {
        let bucket = msm_buckets[u32(i)];
        if !g1_is_identity(bucket) {
            // running = running + bucket (simplified - needs proper G1 add)
            running = bucket;  // Placeholder
        }
        // sum = sum + running (simplified)
        sum = running;  // Placeholder
    }

    msm_result[0] = sum;
}

// =============================================================================
// Batch KZG Verification
// =============================================================================

// Precompute powers of random challenge for batch verification
// r^0, r^1, r^2, ..., r^{n-1}
@compute @workgroup_size(256)
fn batch_verify_compute_powers(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if idx >= params.num_proofs { return; }

    let r = batch_challenge;

    // Compute r^idx
    var r_power = fr_one();
    for (var i = 0u; i < idx; i++) {
        r_power = fr_mont_mul(r_power, r);
    }

    // Store in result buffer for later use
    result_coeffs[idx] = r_power;
}

// Batch verification linear combination computation
// Computes: LHS = sum_i(r^i * C_i)
//           RHS = sum_i(r^i * (z_i * W_i + P_i))
// Where C_i = commitment, W_i = witness, z_i = point, P_i = value point
@compute @workgroup_size(256)
fn batch_verify_precompute(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if idx >= params.num_proofs { return; }

    // Get precomputed r^idx
    let r_power = result_coeffs[idx];

    // Get proof components
    let commitment = batch_commitments[idx];
    let witness = batch_witnesses[idx];
    let point = batch_points[idx];
    let value = batch_values[idx];

    // LHS contribution: r^i * C_i (scalar mul of commitment)
    // RHS contribution: r^i * (z_i * W_i + P_i)

    // Note: Actual implementation would perform:
    // 1. Convert commitment/witness to projective
    // 2. Scalar multiplication by r_power
    // 3. Accumulate into LHS/RHS sums

    // Placeholder for illustration - full G1 scalar mul needed
    let lhs_point = affine_to_projective(commitment);
    let rhs_point = affine_to_projective(witness);

    batch_lhs_accum[idx] = lhs_point;
    batch_rhs_accum[idx] = rhs_point;
}

// Final reduction for batch verification
// Combines all LHS and RHS contributions
@compute @workgroup_size(256)
fn batch_verify_reduce(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let idx = global_id.x;
    let local_idx = local_id.x;

    // Shared memory for reduction
    var shared_lhs: array<G1Projective, 256>;
    var shared_rhs: array<G1Projective, 256>;

    // Load into shared memory
    if idx < params.num_proofs {
        shared_lhs[local_idx] = batch_lhs_accum[idx];
        shared_rhs[local_idx] = batch_rhs_accum[idx];
    } else {
        shared_lhs[local_idx] = g1_identity();
        shared_rhs[local_idx] = g1_identity();
    }

    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if local_idx < stride {
            // Simplified - would need proper G1 addition
            // shared_lhs[local_idx] = g1_add(shared_lhs[local_idx], shared_lhs[local_idx + stride]);
            // shared_rhs[local_idx] = g1_add(shared_rhs[local_idx], shared_rhs[local_idx + stride]);
        }
        workgroupBarrier();
    }

    // Thread 0 writes final result
    if local_idx == 0u {
        msm_result[workgroup_id.x * 2u] = shared_lhs[0];
        msm_result[workgroup_id.x * 2u + 1u] = shared_rhs[0];
    }
}

// =============================================================================
// FFT for Polynomial Operations (Cooley-Tukey)
// =============================================================================

// Bit-reversal permutation for FFT
@compute @workgroup_size(256)
fn fft_bit_reverse(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    let n = 1u << params.degree;  // degree here represents log2(n)

    if idx >= n / 2u { return; }

    let log_n = params.degree;

    // Compute bit-reversed index
    var rev = 0u;
    var temp = idx;
    for (var i = 0u; i < log_n; i++) {
        rev = (rev << 1u) | (temp & 1u);
        temp >>= 1u;
    }

    // Swap if idx < rev (to avoid double-swapping)
    if idx < rev {
        let tmp = result_coeffs[idx];
        result_coeffs[idx] = result_coeffs[rev];
        result_coeffs[rev] = tmp;
    }
}

// FFT butterfly operation (single stage)
@compute @workgroup_size(256)
fn fft_butterfly(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    // params.window_idx represents the current stage
    let stage = params.window_idx;
    let n = params.num_points;

    let m = 1u << (stage + 1u);
    let k = idx % (m / 2u);
    let j = (idx / (m / 2u)) * m + k;

    if j + m / 2u >= n { return; }

    // Get omega (primitive root of unity) from params
    // In practice, would compute or lookup twiddle factors
    let omega = eval_point;  // Placeholder

    // Compute twiddle factor: omega^(k * n/m)
    let exponent = k * (n / m);
    var w = fr_one();
    for (var i = 0u; i < exponent; i++) {
        w = fr_mont_mul(w, omega);
    }

    // Butterfly: u = coeffs[j], t = w * coeffs[j + m/2]
    let u = result_coeffs[j];
    let t = fr_mont_mul(w, result_coeffs[j + m / 2u]);

    result_coeffs[j] = fr_add(u, t);
    result_coeffs[j + m / 2u] = fr_sub(u, t);
}

// =============================================================================
// Blob to Field Elements (EIP-4844)
// =============================================================================

// Convert 32-byte blob chunks to field elements
// Used for encoding blobs before KZG commitment
@compute @workgroup_size(256)
fn blob_to_field_elements(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    // Each field element comes from 32 bytes
    let byte_offset = idx * 32u;

    if byte_offset >= params.num_points * 32u { return; }

    // Note: In actual implementation, would read from a byte buffer
    // and convert to Fr256, then reduce mod r

    // Placeholder: just copy from coeffs
    let elem = poly_coeffs[idx];

    // Reduce to ensure valid field element
    var reduced = elem;
    fr_reduce(&reduced);

    result_coeffs[idx] = reduced;
}

// =============================================================================
// Commitment Computation (Combined Kernel)
// =============================================================================

// Compute KZG commitment: C = sum_i(c_i * G_i)
// Where c_i are polynomial coefficients and G_i are trusted setup points
// This is essentially an MSM operation
@compute @workgroup_size(256)
fn compute_commitment(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if idx >= params.degree + 1u { return; }

    // Get coefficient and trusted setup point
    let coeff = poly_coeffs[idx];
    let setup_point = msm_bases[idx];

    // Skip if coefficient is zero
    if fr_is_zero(coeff) { return; }

    // Compute scalar multiplication: coeff * setup_point
    // This would feed into MSM bucket accumulation

    let point = affine_to_projective(setup_point);

    // Extract window bits for bucketing
    let window_val = extract_window_bits(coeff, params.window_idx, params.window_size);

    if window_val != 0u {
        let bucket_idx = window_val - 1u;
        // Accumulate into bucket (needs atomic or reduction)
        msm_buckets[bucket_idx] = point;
    }
}

// =============================================================================
// KZG Opening Proof
// =============================================================================

// Compute opening proof: pi = [q(s)]_1 where q(x) = (p(x) - p(z)) / (x - z)
// Requires: 1. Compute quotient polynomial
//           2. Commit to quotient using MSM
@compute @workgroup_size(256)
fn compute_opening_proof(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    // Phase 1: Compute quotient (done by poly_quotient kernel)
    // Phase 2: MSM on quotient coefficients (done by msm_bucket_* kernels)

    // This kernel orchestrates the computation
    let idx = global_id.x;
    if idx != 0u { return; }

    // The actual work is done by calling:
    // 1. poly_quotient - compute q(x) = (p(x) - y) / (x - z)
    // 2. compute_commitment - MSM on q(x) coefficients

    // Result is stored in msm_result[0]
}

// =============================================================================
// Utility Kernels
// =============================================================================

// Initialize buckets to identity
@compute @workgroup_size(256)
fn init_buckets(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    let num_buckets = (1u << params.window_size) - 1u;

    if idx >= num_buckets { return; }

    msm_buckets[idx] = g1_identity();
}

// Copy polynomial coefficients for in-place operations
@compute @workgroup_size(256)
fn copy_coeffs(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if idx > params.degree { return; }

    result_coeffs[idx] = poly_coeffs[idx];
}
