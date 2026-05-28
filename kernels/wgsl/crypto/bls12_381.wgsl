// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// BLS12-381 G1 elliptic curve compute kernels.
//
// Bit-exact mirror of metal/src/shaders/crypto/bls12_381.metal:
//   - Same Montgomery-form representation (R = 2^384 mod p).
//   - Same Jacobian projective coordinates (Y² = X³ + b · Z⁶).
//   - Same arithmetic algorithms (CIOS Montgomery, two-stage carry adc).
//   - Same point doubling / addition formulas
//     (http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html).
//   - Same scalar-mul ladder (4 × u64 little-endian scalar, LSB-first).
//
// Storage contract:
//   The host writes LuxG1Projective381 (3 × LuxFp384 = 18 × u64 = 144 bytes)
//   little-endian. On a little-endian device that maps lane-for-lane onto
//   the WGSL G1Projective struct below, which holds 3 × Fp = 3 × 12 × u32 =
//   3 × 48 = 144 bytes. Each u64 limb is split into (lo_u32, hi_u32) in
//   the natural little-endian way.
//
// Helpers (Fp384 / mul / inv) below are bit-exact copies of the canonical
// reference fragments in common/fp384.wgsl. WGSL has no `#include`; if you
// change one, change the other. Reviewing common/fp384.wgsl is equivalent
// to reviewing the helpers here.

// ============================================================================
// Fp384 — 12 × u32 little-endian Montgomery field element.
// ============================================================================

struct Fp { limbs: array<u32, 12> }

// BLS12-381 base field prime p (little-endian limbs).
const BLS_P: array<u32, 12> = array<u32, 12>(
    0xffffaaabu, 0xb9feffffu,
    0xb153ffffu, 0x1eabfffeu,
    0xf6b0f624u, 0x6730d2a0u,
    0xf38512bfu, 0x64774b84u,
    0x434bacd7u, 0x4b1ba7b6u,
    0x397fe69au, 0x1a0111eau
);

// R mod p — Montgomery form of 1. Matches metal::BLS_R (12 × u32 split).
const BLS_R: array<u32, 12> = array<u32, 12>(
    0x0002fffdu, 0x76090000u,
    0xc40c0002u, 0xebf4000bu,
    0x53c758bau, 0x5f489857u,
    0x70525745u, 0x77ce5853u,
    0xa256ec6du, 0x5c071a97u,
    0xfa80e493u, 0x15f65ec3u
);

// R^2 mod p — for canonical → Montgomery lifting via fp_mul(x, R2).
const BLS_R2: array<u32, 12> = array<u32, 12>(
    0x1c341746u, 0xf4df1f34u,
    0x09d104f1u, 0x0a76e6a6u,
    0x4c95b6d5u, 0x8de5476cu,
    0x939d83c0u, 0x67eb88a9u,
    0xb519952du, 0x9a793e85u,
    0x92cae3aau, 0x11988fe5u
);

// -p^{-1} mod 2^32 — Montgomery INV constant.
const BLS_INV: u32 = 0xfffcfffdu;

// p - 2 — for Fermat inversion.
const BLS_P_MINUS_2: array<u32, 12> = array<u32, 12>(
    0xffffaaa9u, 0xb9feffffu,
    0xb153ffffu, 0x1eabfffeu,
    0xf6b0f624u, 0x6730d2a0u,
    0xf38512bfu, 0x64774b84u,
    0x434bacd7u, 0x4b1ba7b6u,
    0x397fe69au, 0x1a0111eau
);

// ============================================================================
// 32×32 → 64 wide multiply (16-bit decomposition).
// ============================================================================
fn fp_mul32(a: u32, b: u32) -> vec2<u32> {
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

// ============================================================================
// Fp384 basic ops (mirror of common/fp384.wgsl).
// ============================================================================

fn fp_zero() -> Fp {
    var r: Fp;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) { r.limbs[i] = 0u; }
    return r;
}

fn fp_one() -> Fp {
    var r: Fp;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) { r.limbs[i] = BLS_R[i]; }
    return r;
}

fn fp_is_zero(a: Fp) -> bool {
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        if (a.limbs[i] != 0u) { return false; }
    }
    return true;
}

fn fp_eq(a: Fp, b: Fp) -> bool {
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        if (a.limbs[i] != b.limbs[i]) { return false; }
    }
    return true;
}

fn fp_gte_p(a: Fp) -> bool {
    for (var i: i32 = 11; i >= 0; i = i - 1) {
        let ai = a.limbs[i];
        let pi = BLS_P[i];
        if (ai > pi) { return true; }
        if (ai < pi) { return false; }
    }
    return true;
}

fn fp_reduce(a: Fp) -> Fp {
    if (!fp_gte_p(a)) { return a; }
    var r: Fp;
    var borrow: u32 = 0u;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        let ai = a.limbs[i];
        let pi = BLS_P[i];
        let s1 = ai - borrow;
        let b1 = select(0u, 1u, ai < borrow);
        let s2 = s1 - pi;
        let b2 = select(0u, 1u, s1 < pi);
        r.limbs[i] = s2;
        borrow = b1 + b2;
    }
    return r;
}

fn fp_add(a: Fp, b: Fp) -> Fp {
    var r: Fp;
    var carry: u32 = 0u;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        let ai = a.limbs[i];
        let bi = b.limbs[i];
        let s1 = ai + bi;
        let c1 = select(0u, 1u, s1 < bi);
        let s2 = s1 + carry;
        let c2 = select(0u, 1u, s2 < carry);
        r.limbs[i] = s2;
        carry = c1 + c2;
    }
    return fp_reduce(r);
}

fn fp_sub(a: Fp, b: Fp) -> Fp {
    var r: Fp;
    var borrow: u32 = 0u;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        let ai = a.limbs[i];
        let bi = b.limbs[i];
        let s1 = ai - borrow;
        let b1 = select(0u, 1u, ai < borrow);
        let s2 = s1 - bi;
        let b2 = select(0u, 1u, s1 < bi);
        r.limbs[i] = s2;
        borrow = b1 + b2;
    }
    if (borrow != 0u) {
        var carry: u32 = 0u;
        for (var i: u32 = 0u; i < 12u; i = i + 1u) {
            let pi = BLS_P[i];
            let s1 = r.limbs[i] + pi;
            let c1 = select(0u, 1u, s1 < pi);
            let s2 = s1 + carry;
            let c2 = select(0u, 1u, s2 < carry);
            r.limbs[i] = s2;
            carry = c1 + c2;
        }
    }
    return r;
}

fn fp_double(a: Fp) -> Fp { return fp_add(a, a); }

fn fp_neg(a: Fp) -> Fp {
    if (fp_is_zero(a)) { return a; }
    var r: Fp;
    var borrow: u32 = 0u;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        let pi = BLS_P[i];
        let ai = a.limbs[i];
        let s1 = pi - borrow;
        let b1 = select(0u, 1u, pi < borrow);
        let s2 = s1 - ai;
        let b2 = select(0u, 1u, s1 < ai);
        r.limbs[i] = s2;
        borrow = b1 + b2;
    }
    return r;
}

// ============================================================================
// Montgomery multiplication: c = a · b · R^{-1} mod p (CIOS 12 × u32).
// Bit-exact mirror of metal::fp384_mul (which uses 6 × u64 limbs).
// ============================================================================
fn fp_mul(a: Fp, b: Fp) -> Fp {
    var t: array<u32, 25>;
    for (var i: u32 = 0u; i < 25u; i = i + 1u) { t[i] = 0u; }

    // Schoolbook a × b.
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        var carry: u32 = 0u;
        for (var j: u32 = 0u; j < 12u; j = j + 1u) {
            let prod = fp_mul32(a.limbs[i], b.limbs[j]);
            let s1 = t[i + j] + prod.x;
            let c1 = select(0u, 1u, s1 < prod.x);
            let s2 = s1 + carry;
            let c2 = select(0u, 1u, s2 < carry);
            t[i + j] = s2;
            carry = prod.y + c1 + c2;
        }
        let s_hi = t[i + 12u] + carry;
        let c_hi = select(0u, 1u, s_hi < carry);
        t[i + 12u] = s_hi;
        var k: u32 = i + 13u;
        var rem: u32 = c_hi;
        loop {
            if (rem == 0u || k >= 25u) { break; }
            let s = t[k] + rem;
            rem = select(0u, 1u, s < rem);
            t[k] = s;
            k = k + 1u;
        }
    }

    // CIOS Montgomery reduction.
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        let m = t[i] * BLS_INV;
        var carry: u32 = 0u;
        for (var j: u32 = 0u; j < 12u; j = j + 1u) {
            let prod = fp_mul32(m, BLS_P[j]);
            let s1 = t[i + j] + prod.x;
            let c1 = select(0u, 1u, s1 < prod.x);
            let s2 = s1 + carry;
            let c2 = select(0u, 1u, s2 < carry);
            t[i + j] = s2;
            carry = prod.y + c1 + c2;
        }
        var k: u32 = i + 12u;
        var rem: u32 = carry;
        loop {
            if (rem == 0u || k >= 25u) { break; }
            let s = t[k] + rem;
            rem = select(0u, 1u, s < rem);
            t[k] = s;
            k = k + 1u;
        }
    }

    var r: Fp;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) { r.limbs[i] = t[i + 12u]; }
    return fp_reduce(r);
}

fn fp_sqr(a: Fp) -> Fp { return fp_mul(a, a); }

// Fermat inversion: a^{-1} = a^{p-2} mod p.
fn fp_pow_p_minus_2(base: Fp) -> Fp {
    var result = fp_one();
    var b = base;
    for (var limb: u32 = 0u; limb < 12u; limb = limb + 1u) {
        let e = BLS_P_MINUS_2[limb];
        for (var bit: u32 = 0u; bit < 32u; bit = bit + 1u) {
            if (((e >> bit) & 1u) == 1u) {
                result = fp_mul(result, b);
            }
            b = fp_sqr(b);
        }
    }
    return result;
}

fn fp_inv(a: Fp) -> Fp { return fp_pow_p_minus_2(a); }

// ============================================================================
// G1 point types (byte-identical to LuxG1Projective381 from
// include/lux/gpu/crypto.h on a little-endian host).
// ============================================================================

struct G1Projective {
    x: Fp,
    y: Fp,
    z: Fp,
}

// G1 affine (with infinity flag) — matches LuxG1Affine381.
struct G1Affine {
    x: Fp,
    y: Fp,
    infinity: u32,
    _pad: u32,
}

fn g1_identity() -> G1Projective {
    return G1Projective(fp_zero(), fp_one(), fp_zero());
}

fn g1_is_identity(p: G1Projective) -> bool {
    return fp_is_zero(p.z);
}

// b = 4 in Montgomery form for the curve check Y² = X³ + b · Z⁶.
fn bls_g1_b_mont() -> Fp {
    var four: Fp;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) { four.limbs[i] = 0u; }
    four.limbs[0] = 4u;
    var r2: Fp;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) { r2.limbs[i] = BLS_R2[i]; }
    return fp_mul(four, r2);
}

// On-curve check for Jacobian (X, Y, Z).
//   Y² == X³ + b · Z⁶  (with the identity Z = 0 trivially on-curve).
fn bls_g1_is_on_curve_jacobian(p: G1Projective) -> bool {
    if (fp_is_zero(p.z)) { return true; }
    let y2  = fp_sqr(p.y);
    let x2  = fp_sqr(p.x);
    let x3  = fp_mul(x2, p.x);
    let z2  = fp_sqr(p.z);
    let z4  = fp_sqr(z2);
    let z6  = fp_mul(z4, z2);
    let b   = bls_g1_b_mont();
    let bz6 = fp_mul(b, z6);
    let rhs = fp_add(x3, bz6);
    return fp_eq(y2, rhs);
}

// ----------------------------------------------------------------------------
// Point doubling: result = 2P in Jacobian.
//   http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
// On-curve gate is enforced at the perimeter; this body trusts induction.
// ----------------------------------------------------------------------------
fn g1_double(p: G1Projective) -> G1Projective {
    if (g1_is_identity(p)) { return p; }

    let a = fp_sqr(p.x);            // A = X1²
    let b = fp_sqr(p.y);            // B = Y1²
    let c = fp_sqr(b);              // C = B²

    // D = 2 · ((X1 + B)² - A - C)
    let xb = fp_add(p.x, b);
    let xb2 = fp_sqr(xb);
    var d = fp_sub(xb2, a);
    d = fp_sub(d, c);
    d = fp_double(d);

    // E = 3 · A
    var e = fp_add(a, a);
    e = fp_add(e, a);

    let f = fp_sqr(e);              // F = E²

    let d2 = fp_double(d);
    var r: G1Projective;
    r.x = fp_sub(f, d2);            // X3 = F - 2D

    let dx3 = fp_sub(d, r.x);
    let edx3 = fp_mul(e, dx3);
    var c8 = fp_double(c);
    c8 = fp_double(c8);
    c8 = fp_double(c8);
    r.y = fp_sub(edx3, c8);         // Y3 = E·(D - X3) - 8C

    let yz = fp_mul(p.y, p.z);
    r.z = fp_double(yz);            // Z3 = 2·Y1·Z1

    return r;
}

// ----------------------------------------------------------------------------
// Point addition: R = P + Q (both projective Jacobian).
//   http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
// ----------------------------------------------------------------------------
fn g1_add(p: G1Projective, q: G1Projective) -> G1Projective {
    if (g1_is_identity(p)) { return q; }
    if (g1_is_identity(q)) { return p; }

    let z1z1 = fp_sqr(p.z);
    let z2z2 = fp_sqr(q.z);

    let u1 = fp_mul(p.x, z2z2);     // U1 = X1·Z2²
    let u2 = fp_mul(q.x, z1z1);     // U2 = X2·Z1²

    var s1 = fp_mul(p.y, q.z);      // S1 = Y1·Z2·Z2² = Y1·Z2³
    s1 = fp_mul(s1, z2z2);
    var s2 = fp_mul(q.y, p.z);      // S2 = Y2·Z1·Z1² = Y2·Z1³
    s2 = fp_mul(s2, z1z1);

    let h = fp_sub(u2, u1);         // H = U2 - U1

    let h_zero = fp_is_zero(h);
    let s_diff = fp_sub(s2, s1);
    let s_zero = fp_is_zero(s_diff);

    if (h_zero && s_zero) { return g1_double(p); }
    if (h_zero)            { return g1_identity(); }

    let h2 = fp_double(h);
    let i  = fp_sqr(h2);            // I = (2H)²
    let j  = fp_mul(h, i);          // J = H·I

    let r  = fp_double(s_diff);     // r = 2·(S2 - S1)
    let v  = fp_mul(u1, i);         // V = U1·I

    let r2 = fp_sqr(r);
    let v2 = fp_double(v);
    var result: G1Projective;
    result.x = fp_sub(fp_sub(r2, j), v2);   // X3 = r² - J - 2V

    let vx3 = fp_sub(v, result.x);
    let rvx3 = fp_mul(r, vx3);
    var s1j = fp_mul(s1, j);
    s1j = fp_double(s1j);
    result.y = fp_sub(rvx3, s1j);           // Y3 = r·(V - X3) - 2·S1·J

    let z12 = fp_add(p.z, q.z);
    let z12_2 = fp_sqr(z12);
    var z3 = fp_sub(z12_2, z1z1);
    z3 = fp_sub(z3, z2z2);
    result.z = fp_mul(z3, h);               // Z3 = ((Z1+Z2)² - Z1Z1 - Z2Z2)·H

    return result;
}

// ----------------------------------------------------------------------------
// Scalar multiplication: result = [s]P. Double-and-add, MSB-first, so the
// scalar=0 short-circuit yields canonical infinity (z = 0) — matches the
// Metal/CPU semantics.
//
// scalar is a 4 × u64 little-endian buffer (256 bits). On the WGSL side
// we read it as 8 × u32 with `(lo_u32, hi_u32)` for each u64 limb.
// On-curve gate at the head: off-curve P → return identity, so [k]∞ = ∞.
// ----------------------------------------------------------------------------
fn scalar_bit(s: array<u32, 8>, bit_idx: u32) -> u32 {
    let limb = bit_idx / 32u;
    let bit_in_limb = bit_idx % 32u;
    if (limb >= 8u) { return 0u; }
    return (s[limb] >> bit_in_limb) & 1u;
}

fn g1_scalar_mul(p: G1Projective, s: array<u32, 8>) -> G1Projective {
    if (!bls_g1_is_on_curve_jacobian(p)) { return g1_identity(); }

    var result = g1_identity();
    // MSB → LSB iteration: 256 bits.
    for (var bit: i32 = 255; bit >= 0; bit = bit - 1) {
        result = g1_double(result);
        if (scalar_bit(s, u32(bit)) == 1u) {
            result = g1_add(result, p);
        }
    }
    return result;
}

// ============================================================================
// LuxBackend ABI bridges — operate directly on G1Projective buffers.
//
// Layout match: LuxG1Projective381 = 3 × LuxFp384 = 3 × 6 × u64 = 144 bytes
// per point; on a little-endian host that maps lane-for-lane onto WGSL
// G1Projective { x,y,z: Fp { limbs: array<u32, 12> } } = 3 × 48 = 144 bytes.
//
// Scalar layout: LuxScalar256 = 4 × u64 = 32 bytes; on the WGSL side a
// "scalar" is `array<u32, 8>` (matching little-endian split).
// ============================================================================

// G1Projective buffers (input A, input B for add; input A only for mul).
@group(0) @binding(0) var<storage, read>       proj_a: array<G1Projective>;
@group(0) @binding(1) var<storage, read>       proj_b: array<G1Projective>;
// 4 × u64 = 8 × u32 per scalar; we declare as a flat array of u32 indexed
// by 8·tid + j.
@group(0) @binding(2) var<storage, read>       scalars: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_points: array<G1Projective>;
@group(0) @binding(4) var<uniform>             g1_count: u32;

// ----------------------------------------------------------------------------
// Batch G1 add: out[i] = proj_a[i] + proj_b[i]. Mirrors metal `g1_batch_add`.
// ----------------------------------------------------------------------------
@compute @workgroup_size(64)
fn g1_batch_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= g1_count) { return; }
    let pa = proj_a[idx];
    let pb = proj_b[idx];
    out_points[idx] = g1_add(pa, pb);
}

// ----------------------------------------------------------------------------
// Batch G1 double: out[i] = 2 · proj_a[i].
// ----------------------------------------------------------------------------
@compute @workgroup_size(64)
fn g1_batch_double(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= g1_count) { return; }
    out_points[idx] = g1_double(proj_a[idx]);
}

// ----------------------------------------------------------------------------
// Batch scalar multiplication: out[i] = [scalars[i]] · proj_a[i]. Mirrors
// metal `g1_batch_scalar_mul`. Scalars are flat `array<u32>` with 8 u32 per
// scalar in little-endian (lo, hi) pairs for each of 4 u64 limbs.
// ----------------------------------------------------------------------------
@compute @workgroup_size(64)
fn g1_batch_scalar_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= g1_count) { return; }

    var s: array<u32, 8>;
    let base = idx * 8u;
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        s[j] = scalars[base + j];
    }

    out_points[idx] = g1_scalar_mul(proj_a[idx], s);
}

// ============================================================================
// Z-normalization (Jacobian → affine-as-projective with z = 1).
//
// Jacobian (X, Y, Z) is non-unique; the same affine point has infinitely
// many projective representations differing by a Z scalar. To get bit-exact
// cross-backend output the host must run this normalize pass before
// comparing or persisting points.
//
//   non-identity: (X / Z²,  Y / Z³,  R mod p)
//   identity:     (0, 0, 0)
//
// Mirrors metal `bls12_381_g1_normalize_projective`.
// Cost per point: 1 inv + 1 sqr + 2 muls.
// ============================================================================
@group(0) @binding(5) var<storage, read_write> norm_points: array<G1Projective>;
@group(0) @binding(6) var<uniform>             norm_count: u32;

@compute @workgroup_size(32)
fn g1_normalize_projective(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= norm_count) { return; }

    let p = norm_points[idx];
    if (fp_is_zero(p.z)) {
        // Identity → all-zero limbs (every backend agrees).
        var z: G1Projective;
        for (var i: u32 = 0u; i < 12u; i = i + 1u) {
            z.x.limbs[i] = 0u;
            z.y.limbs[i] = 0u;
            z.z.limbs[i] = 0u;
        }
        norm_points[idx] = z;
        return;
    }

    let z_inv      = fp_inv(p.z);
    let z_inv_sq   = fp_sqr(z_inv);
    let z_inv_cube = fp_mul(z_inv_sq, z_inv);

    var r: G1Projective;
    r.x = fp_mul(p.x, z_inv_sq);
    r.y = fp_mul(p.y, z_inv_cube);
    r.z = fp_one();
    norm_points[idx] = r;
}
