// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// BN254 Elliptic Curve Operations
// Supports G1 group operations for zkSNARK applications
// Optimized for GPU-based proof verification (Groth16, PLONK)

// BN254 (alt_bn128) Field Parameters:
// p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
// p = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
// r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// r = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
// Embedding degree k = 12 (for pairing-friendly curve)

// ============================================================================
// Field Elements (256 bits = 8 x 32-bit limbs, little-endian)
// ============================================================================

// Base field Fp (256 bits = 8 x 32-bit limbs)
struct Fp {
    limbs: array<u32, 8>,
}

// Scalar field Fr (256 bits = 8 x 32-bit limbs)
struct Fr {
    limbs: array<u32, 8>,
}

// Extension field Fp2 = Fp[u] / (u^2 + 1)
struct Fp2 {
    c0: Fp,  // Real part
    c1: Fp,  // Imaginary part (coefficient of u)
}

// ============================================================================
// Curve Points
// ============================================================================

// G1: E(Fp) : y^2 = x^3 + 3
struct G1Affine {
    x: Fp,
    y: Fp,
}

struct G1Projective {
    x: Fp,
    y: Fp,
    z: Fp,
}

// G2: E'(Fp2) : y^2 = x^3 + 3/(u+9)
struct G2Affine {
    x: Fp2,
    y: Fp2,
}

struct G2Projective {
    x: Fp2,
    y: Fp2,
    z: Fp2,
}

// ============================================================================
// Constants (little-endian limbs)
// ============================================================================

// BN254 base field modulus p
// p = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
const BN254_P: array<u32, 8> = array<u32, 8>(
    0xd87cfd47u, 0x3c208c16u, 0x6871ca8du, 0x97816a91u,
    0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u
);

// BN254 scalar field modulus r
// r = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
const BN254_R: array<u32, 8> = array<u32, 8>(
    0xf0000001u, 0x43e1f593u, 0x79b97091u, 0x2833e848u,
    0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u
);

// Montgomery R^2 mod p (for Montgomery multiplication)
// R = 2^256, R^2 mod p
const BN254_R2: array<u32, 8> = array<u32, 8>(
    0x1bb8e645u, 0xe0a77c19u, 0x59aa22d6u, 0xe6d06691u,
    0xa4cd1ca7u, 0x6d6d17abu, 0xe01c77c6u, 0x06d89f71u
);

// Montgomery constant: -p^(-1) mod 2^32
const BN254_INV: u32 = 0xe4866389u;

// Generator point G1 (Montgomery form)
// x = 1
const G1_GENERATOR_X: array<u32, 8> = array<u32, 8>(
    0xd35d438au, 0x0a0f3f0cu, 0x33b5db3du, 0x7c536ff4u,
    0x9d8a9fa0u, 0x85dde1a0u, 0xc9a2b714u, 0x0e0a77c1u
);

// y = 2
const G1_GENERATOR_Y: array<u32, 8> = array<u32, 8>(
    0xa6ba871bu, 0x141e5e18u, 0x677b9c7fu, 0xf87f0c7eu,
    0x3b1465f2u, 0x0bb8e0e0u, 0x93515ccbu, 0x1c14ef83u
);

// b coefficient for BN254: b = 3
const BN254_B: array<u32, 8> = array<u32, 8>(
    0x7a17caa9u, 0x1e2d9e0eu, 0x9b195d23u, 0x74f95f2eu,
    0xd8d8f6aeu, 0x91a83e6bu, 0xb5927515u, 0x2a1f6744u
);

// Buffer bindings
@group(0) @binding(0) var<storage, read> input_points: array<G1Affine>;
@group(0) @binding(1) var<storage, read> input_scalars: array<Fr>;
@group(0) @binding(2) var<storage, read_write> output_points: array<G1Projective>;
@group(0) @binding(3) var<uniform> num_elements: u32;

// ============================================================================
// Fp Arithmetic (256-bit field elements)
// ============================================================================

fn fp_zero() -> Fp {
    var r: Fp;
    for (var i = 0u; i < 8u; i++) {
        r.limbs[i] = 0u;
    }
    return r;
}

fn fp_one() -> Fp {
    // Montgomery form of 1: R mod p
    var r: Fp;
    r.limbs[0] = 0xd35d438au;
    r.limbs[1] = 0x0a0f3f0cu;
    r.limbs[2] = 0x33b5db3du;
    r.limbs[3] = 0x7c536ff4u;
    r.limbs[4] = 0x9d8a9fa0u;
    r.limbs[5] = 0x85dde1a0u;
    r.limbs[6] = 0xc9a2b714u;
    r.limbs[7] = 0x0e0a77c1u;
    return r;
}

fn fp_is_zero(a: Fp) -> bool {
    for (var i = 0u; i < 8u; i++) {
        if (a.limbs[i] != 0u) { return false; }
    }
    return true;
}

fn fp_eq(a: Fp, b: Fp) -> bool {
    for (var i = 0u; i < 8u; i++) {
        if (a.limbs[i] != b.limbs[i]) { return false; }
    }
    return true;
}

// Compare a >= p
fn fp_gte_p(a: Fp) -> bool {
    for (var i = 7; i >= 0; i--) {
        if (a.limbs[i] > BN254_P[i]) { return true; }
        if (a.limbs[i] < BN254_P[i]) { return false; }
    }
    return true; // Equal means >= p
}

// Add without reduction
fn fp_add_no_reduce(a: Fp, b: Fp) -> Fp {
    var r: Fp;
    var carry = 0u;

    for (var i = 0u; i < 8u; i++) {
        let sum = a.limbs[i] + b.limbs[i] + carry;
        r.limbs[i] = sum;
        // Check for overflow: sum < a.limbs[i] means overflow occurred
        carry = select(0u, 1u, sum < a.limbs[i] || (carry == 1u && sum == a.limbs[i]));
    }

    return r;
}

// Subtract p from a (assumes a >= p)
fn fp_sub_p(a: Fp) -> Fp {
    var r: Fp;
    var borrow = 0u;

    for (var i = 0u; i < 8u; i++) {
        let diff = a.limbs[i] - BN254_P[i] - borrow;
        borrow = select(0u, 1u, a.limbs[i] < BN254_P[i] + borrow);
        r.limbs[i] = diff;
    }

    return r;
}

// Modular reduction: a mod p
fn fp_reduce(a: Fp) -> Fp {
    if (fp_gte_p(a)) {
        return fp_sub_p(a);
    }
    return a;
}

// Modular addition: (a + b) mod p
fn fp_add(a: Fp, b: Fp) -> Fp {
    var r = fp_add_no_reduce(a, b);

    // If carry out or result >= p, subtract p
    if (fp_gte_p(r)) {
        r = fp_sub_p(r);
    }

    return r;
}

// Subtract (a - b) mod p
fn fp_sub(a: Fp, b: Fp) -> Fp {
    var r: Fp;
    var borrow = 0u;

    for (var i = 0u; i < 8u; i++) {
        let diff = a.limbs[i] - b.limbs[i] - borrow;
        borrow = select(0u, 1u, a.limbs[i] < b.limbs[i] + borrow);
        r.limbs[i] = diff;
    }

    // If borrow, add p back
    if (borrow != 0u) {
        var carry = 0u;
        for (var i = 0u; i < 8u; i++) {
            let sum = r.limbs[i] + BN254_P[i] + carry;
            r.limbs[i] = sum;
            carry = select(0u, 1u, sum < BN254_P[i]);
        }
    }

    return r;
}

// Double: 2a mod p
fn fp_double(a: Fp) -> Fp {
    return fp_add(a, a);
}

// Negate: -a mod p = p - a
fn fp_neg(a: Fp) -> Fp {
    if (fp_is_zero(a)) {
        return a;
    }

    var r: Fp;
    var borrow = 0u;
    for (var i = 0u; i < 8u; i++) {
        let diff = BN254_P[i] - a.limbs[i] - borrow;
        borrow = select(0u, 1u, BN254_P[i] < a.limbs[i] + borrow);
        r.limbs[i] = diff;
    }
    return r;
}

// Multiply two 32-bit values, return (lo, hi)
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

    let lo = p0 + (mid << 16u);
    let lo_carry = select(0u, 1u, lo < p0);

    let hi = p3 + (mid >> 16u) + mid_carry + lo_carry;

    return vec2<u32>(lo, hi);
}

// Montgomery multiplication: (a * b * R^(-1)) mod p
// Simplified version - full implementation needs CIOS algorithm
fn fp_mul(a: Fp, b: Fp) -> Fp {
    // Placeholder: schoolbook multiplication with reduction
    // Production code should use Montgomery multiplication (CIOS)
    var t: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) {
        t[i] = 0u;
    }

    // Schoolbook multiplication
    for (var i = 0u; i < 8u; i++) {
        var carry = 0u;
        for (var j = 0u; j < 8u; j++) {
            let prod = mul32(a.limbs[i], b.limbs[j]);
            let sum1 = t[i + j] + prod.x + carry;
            let c1 = select(0u, 1u, sum1 < t[i + j] || (carry > 0u && sum1 == t[i + j]));
            t[i + j] = sum1;
            carry = prod.y + c1;
        }
        t[i + 8u] = carry;
    }

    // Barrett reduction (simplified)
    // Full implementation should use Montgomery reduction
    var r: Fp;
    for (var i = 0u; i < 8u; i++) {
        r.limbs[i] = t[i];
    }

    // Multiple subtractions of p until r < p
    for (var iter = 0u; iter < 3u; iter++) {
        if (fp_gte_p(r)) {
            r = fp_sub_p(r);
        }
    }

    return r;
}

// Square: a^2 mod p
fn fp_square(a: Fp) -> Fp {
    return fp_mul(a, a);
}

// ============================================================================
// Fp2 Arithmetic (Extension field)
// ============================================================================

fn fp2_zero() -> Fp2 {
    return Fp2(fp_zero(), fp_zero());
}

fn fp2_one() -> Fp2 {
    return Fp2(fp_one(), fp_zero());
}

fn fp2_is_zero(a: Fp2) -> bool {
    return fp_is_zero(a.c0) && fp_is_zero(a.c1);
}

fn fp2_eq(a: Fp2, b: Fp2) -> bool {
    return fp_eq(a.c0, b.c0) && fp_eq(a.c1, b.c1);
}

fn fp2_add(a: Fp2, b: Fp2) -> Fp2 {
    return Fp2(fp_add(a.c0, b.c0), fp_add(a.c1, b.c1));
}

fn fp2_sub(a: Fp2, b: Fp2) -> Fp2 {
    return Fp2(fp_sub(a.c0, b.c0), fp_sub(a.c1, b.c1));
}

fn fp2_neg(a: Fp2) -> Fp2 {
    return Fp2(fp_neg(a.c0), fp_neg(a.c1));
}

fn fp2_double(a: Fp2) -> Fp2 {
    return Fp2(fp_double(a.c0), fp_double(a.c1));
}

// Conjugate: (a + bu) -> (a - bu)
fn fp2_conjugate(a: Fp2) -> Fp2 {
    return Fp2(a.c0, fp_neg(a.c1));
}

// Fp2 multiplication: (a0 + a1*u)(b0 + b1*u) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*u
// where u^2 = -1
fn fp2_mul(a: Fp2, b: Fp2) -> Fp2 {
    let t0 = fp_mul(a.c0, b.c0);
    let t1 = fp_mul(a.c1, b.c1);
    let t2 = fp_add(a.c0, a.c1);
    let t3 = fp_add(b.c0, b.c1);
    let t4 = fp_mul(t2, t3);

    // c0 = a0*b0 - a1*b1
    let c0 = fp_sub(t0, t1);
    // c1 = (a0+a1)*(b0+b1) - a0*b0 - a1*b1 = a0*b1 + a1*b0
    let c1 = fp_sub(fp_sub(t4, t0), t1);

    return Fp2(c0, c1);
}

// Fp2 square: (a + bu)^2 = (a^2 - b^2) + 2ab*u
fn fp2_square(a: Fp2) -> Fp2 {
    let t0 = fp_add(a.c0, a.c1);
    let t1 = fp_sub(a.c0, a.c1);
    let c0 = fp_mul(t0, t1);
    let c1 = fp_mul(fp_double(a.c0), a.c1);
    return Fp2(c0, c1);
}

// ============================================================================
// G1 Operations (Projective Coordinates)
// ============================================================================

fn g1_identity() -> G1Projective {
    return G1Projective(fp_zero(), fp_one(), fp_zero());
}

fn g1_is_identity(p: G1Projective) -> bool {
    return fp_is_zero(p.z);
}

fn g1_affine_to_projective(a: G1Affine) -> G1Projective {
    // Check if affine point is at infinity (both coords zero)
    if (fp_is_zero(a.x) && fp_is_zero(a.y)) {
        return g1_identity();
    }
    return G1Projective(a.x, a.y, fp_one());
}

// Point doubling in projective coordinates: 2P
// Formula for y^2 = x^3 + b (a = 0):
// XX = X1^2
// YY = Y1^2
// YYYY = YY^2
// ZZ = Z1^2
// S = 2*((X1+YY)^2 - XX - YYYY)
// M = 3*XX (since a=0)
// X3 = M^2 - 2*S
// Y3 = M*(S - X3) - 8*YYYY
// Z3 = 2*Y1*Z1
fn g1_double(p: G1Projective) -> G1Projective {
    if (g1_is_identity(p)) {
        return p;
    }

    // Check if Y = 0 (point of order 2)
    if (fp_is_zero(p.y)) {
        return g1_identity();
    }

    let xx = fp_square(p.x);                    // XX = X^2
    let yy = fp_square(p.y);                    // YY = Y^2
    let yyyy = fp_square(yy);                   // YYYY = YY^2

    // S = 2*((X+YY)^2 - XX - YYYY)
    let t1 = fp_add(p.x, yy);
    let t2 = fp_square(t1);
    let t3 = fp_sub(t2, xx);
    let t4 = fp_sub(t3, yyyy);
    let s = fp_double(t4);

    // M = 3*XX (since a=0)
    let m = fp_add(fp_double(xx), xx);

    // X3 = M^2 - 2*S
    let m2 = fp_square(m);
    let s2 = fp_double(s);
    let x3 = fp_sub(m2, s2);

    // Y3 = M*(S - X3) - 8*YYYY
    let t5 = fp_sub(s, x3);
    let t6 = fp_mul(m, t5);
    let yyyy8 = fp_double(fp_double(fp_double(yyyy)));
    let y3 = fp_sub(t6, yyyy8);

    // Z3 = 2*Y1*Z1
    let z3 = fp_mul(fp_double(p.y), p.z);

    return G1Projective(x3, y3, z3);
}

// Mixed addition: P (projective) + Q (affine) = R (projective)
// When Q.Z = 1, we save multiplications
// Using "madd-2008-s" formulas
fn g1_add_mixed(p: G1Projective, q: G1Affine) -> G1Projective {
    if (g1_is_identity(p)) {
        return g1_affine_to_projective(q);
    }

    // Check if q is identity
    if (fp_is_zero(q.x) && fp_is_zero(q.y)) {
        return p;
    }

    let zz = fp_square(p.z);                    // ZZ = Z1^2
    let u2 = fp_mul(q.x, zz);                   // U2 = X2*ZZ
    let zzz = fp_mul(zz, p.z);                  // ZZZ = Z1^3
    let s2 = fp_mul(q.y, zzz);                  // S2 = Y2*ZZZ

    // H = U2 - X1
    let h = fp_sub(u2, p.x);

    // R = S2 - Y1
    let r = fp_sub(s2, p.y);

    // Check if P == Q (should double instead)
    if (fp_is_zero(h) && fp_is_zero(r)) {
        return g1_double(p);
    }

    // Check if P == -Q (result is identity)
    if (fp_is_zero(h)) {
        return g1_identity();
    }

    let hh = fp_square(h);                      // HH = H^2
    let hhh = fp_mul(hh, h);                    // HHH = H^3
    let v = fp_mul(p.x, hh);                    // V = X1*HH

    // X3 = R^2 - HHH - 2*V
    let r2 = fp_square(r);
    let v2 = fp_double(v);
    let x3 = fp_sub(fp_sub(r2, hhh), v2);

    // Y3 = R*(V - X3) - Y1*HHH
    let t1 = fp_sub(v, x3);
    let t2 = fp_mul(r, t1);
    let t3 = fp_mul(p.y, hhh);
    let y3 = fp_sub(t2, t3);

    // Z3 = Z1*H
    let z3 = fp_mul(p.z, h);

    return G1Projective(x3, y3, z3);
}

// Full addition: P + Q (both projective)
// Using "add-2008-bbjlp" formulas
fn g1_add(p: G1Projective, q: G1Projective) -> G1Projective {
    if (g1_is_identity(p)) { return q; }
    if (g1_is_identity(q)) { return p; }

    let z1z1 = fp_square(p.z);                  // Z1Z1 = Z1^2
    let z2z2 = fp_square(q.z);                  // Z2Z2 = Z2^2
    let u1 = fp_mul(p.x, z2z2);                 // U1 = X1*Z2Z2
    let u2 = fp_mul(q.x, z1z1);                 // U2 = X2*Z1Z1
    let z1z1z1 = fp_mul(z1z1, p.z);             // Z1^3
    let z2z2z2 = fp_mul(z2z2, q.z);             // Z2^3
    let s1 = fp_mul(p.y, z2z2z2);               // S1 = Y1*Z2^3
    let s2 = fp_mul(q.y, z1z1z1);               // S2 = Y2*Z1^3

    let h = fp_sub(u2, u1);                     // H = U2 - U1
    let i = fp_square(fp_double(h));            // I = (2*H)^2
    let j = fp_mul(h, i);                       // J = H*I
    let r = fp_double(fp_sub(s2, s1));          // r = 2*(S2 - S1)
    let v = fp_mul(u1, i);                      // V = U1*I

    // Check for special cases
    if (fp_is_zero(h)) {
        if (fp_is_zero(r)) {
            // P == Q, double instead
            return g1_double(p);
        } else {
            // P == -Q, return identity
            return g1_identity();
        }
    }

    // X3 = r^2 - J - 2*V
    let r2 = fp_square(r);
    let v2 = fp_double(v);
    let x3 = fp_sub(fp_sub(r2, j), v2);

    // Y3 = r*(V - X3) - 2*S1*J
    let t1 = fp_sub(v, x3);
    let t2 = fp_mul(r, t1);
    let t3 = fp_double(fp_mul(s1, j));
    let y3 = fp_sub(t2, t3);

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    let t4 = fp_add(p.z, q.z);
    let t5 = fp_square(t4);
    let t6 = fp_sub(fp_sub(t5, z1z1), z2z2);
    let z3 = fp_mul(t6, h);

    return G1Projective(x3, y3, z3);
}

// Negation: -P = (X, -Y, Z)
fn g1_neg(p: G1Projective) -> G1Projective {
    return G1Projective(p.x, fp_neg(p.y), p.z);
}

// ============================================================================
// G2 Operations
// ============================================================================

fn g2_identity() -> G2Projective {
    return G2Projective(fp2_zero(), fp2_one(), fp2_zero());
}

fn g2_is_identity(p: G2Projective) -> bool {
    return fp2_is_zero(p.z);
}

fn g2_affine_to_projective(a: G2Affine) -> G2Projective {
    if (fp2_is_zero(a.x) && fp2_is_zero(a.y)) {
        return g2_identity();
    }
    return G2Projective(a.x, a.y, fp2_one());
}

fn g2_double(p: G2Projective) -> G2Projective {
    if (g2_is_identity(p)) {
        return p;
    }

    if (fp2_is_zero(p.y)) {
        return g2_identity();
    }

    let xx = fp2_square(p.x);
    let yy = fp2_square(p.y);
    let yyyy = fp2_square(yy);

    let t1 = fp2_add(p.x, yy);
    let t2 = fp2_square(t1);
    let t3 = fp2_sub(t2, xx);
    let t4 = fp2_sub(t3, yyyy);
    let s = fp2_double(t4);

    let m = fp2_add(fp2_double(xx), xx);

    let m2 = fp2_square(m);
    let s2 = fp2_double(s);
    let x3 = fp2_sub(m2, s2);

    let t5 = fp2_sub(s, x3);
    let t6 = fp2_mul(m, t5);
    let yyyy8 = fp2_double(fp2_double(fp2_double(yyyy)));
    let y3 = fp2_sub(t6, yyyy8);

    let z3 = fp2_mul(fp2_double(p.y), p.z);

    return G2Projective(x3, y3, z3);
}

fn g2_add(p: G2Projective, q: G2Projective) -> G2Projective {
    if (g2_is_identity(p)) { return q; }
    if (g2_is_identity(q)) { return p; }

    let z1z1 = fp2_square(p.z);
    let z2z2 = fp2_square(q.z);
    let u1 = fp2_mul(p.x, z2z2);
    let u2 = fp2_mul(q.x, z1z1);
    let z1z1z1 = fp2_mul(z1z1, p.z);
    let z2z2z2 = fp2_mul(z2z2, q.z);
    let s1 = fp2_mul(p.y, z2z2z2);
    let s2 = fp2_mul(q.y, z1z1z1);

    let h = fp2_sub(u2, u1);
    let i = fp2_square(fp2_double(h));
    let j = fp2_mul(h, i);
    let r = fp2_double(fp2_sub(s2, s1));
    let v = fp2_mul(u1, i);

    if (fp2_is_zero(h)) {
        if (fp2_is_zero(r)) {
            return g2_double(p);
        } else {
            return g2_identity();
        }
    }

    let r2 = fp2_square(r);
    let v2 = fp2_double(v);
    let x3 = fp2_sub(fp2_sub(r2, j), v2);

    let t1 = fp2_sub(v, x3);
    let t2 = fp2_mul(r, t1);
    let t3 = fp2_double(fp2_mul(s1, j));
    let y3 = fp2_sub(t2, t3);

    let t4 = fp2_add(p.z, q.z);
    let t5 = fp2_square(t4);
    let t6 = fp2_sub(fp2_sub(t5, z1z1), z2z2);
    let z3 = fp2_mul(t6, h);

    return G2Projective(x3, y3, z3);
}

fn g2_neg(p: G2Projective) -> G2Projective {
    return G2Projective(p.x, fp2_neg(p.y), p.z);
}

// ============================================================================
// Scalar Multiplication
// ============================================================================

// Get bit at position i from scalar (little-endian)
fn scalar_get_bit(s: Fr, bit_idx: u32) -> u32 {
    let limb_idx = bit_idx / 32u;
    let bit_in_limb = bit_idx % 32u;

    if (limb_idx >= 8u) {
        return 0u;
    }

    return (s.limbs[limb_idx] >> bit_in_limb) & 1u;
}

// Scalar multiplication using double-and-add
// Computes [s]P for scalar s and point P
fn g1_scalar_mul(p: G1Projective, s: Fr) -> G1Projective {
    var result = g1_identity();
    var temp = p;

    // Double-and-add from LSB to MSB
    for (var i = 0u; i < 256u; i++) {
        if (scalar_get_bit(s, i) == 1u) {
            result = g1_add(result, temp);
        }
        temp = g1_double(temp);
    }

    return result;
}

// Scalar multiplication with affine input
fn g1_scalar_mul_affine(p: G1Affine, s: Fr) -> G1Projective {
    return g1_scalar_mul(g1_affine_to_projective(p), s);
}

// ============================================================================
// Compute Kernels
// ============================================================================

// Batch G1 point doubling
@compute @workgroup_size(256)
fn bn254_g1_batch_double(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= num_elements) { return; }

    output_points[idx] = g1_double(output_points[idx]);
}

// Batch G1 point addition (mixed: projective + affine)
@compute @workgroup_size(256)
fn bn254_g1_batch_add_mixed(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= num_elements) { return; }

    let p = output_points[idx];
    let q = input_points[idx];
    output_points[idx] = g1_add_mixed(p, q);
}

// Convert affine points to projective (batch)
@compute @workgroup_size(256)
fn bn254_g1_affine_to_projective(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= num_elements) { return; }

    output_points[idx] = g1_affine_to_projective(input_points[idx]);
}

// Batch G1 negation
@compute @workgroup_size(256)
fn bn254_g1_batch_neg(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= num_elements) { return; }

    output_points[idx] = g1_neg(output_points[idx]);
}

// Batch scalar multiplication
@compute @workgroup_size(256)
fn bn254_g1_batch_scalar_mul(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= num_elements) { return; }

    let p = input_points[idx];
    let s = input_scalars[idx];
    output_points[idx] = g1_scalar_mul_affine(p, s);
}

// Single scalar multiplication (for small workloads)
@compute @workgroup_size(1)
fn bn254_g1_scalar_mul_single(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x != 0u) { return; }

    let p = input_points[0];
    let s = input_scalars[0];
    output_points[0] = g1_scalar_mul_affine(p, s);
}

// Batch point equality check (writes 1 or 0 to Z coordinate)
@compute @workgroup_size(256)
fn bn254_g1_batch_eq(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= num_elements) { return; }

    // Compare output_points[2*idx] with output_points[2*idx+1]
    let i1 = 2u * idx;
    let i2 = 2u * idx + 1u;

    if (i2 >= num_elements) { return; }

    let p = output_points[i1];
    let q = output_points[i2];

    // Two projective points P=(X1,Y1,Z1) and Q=(X2,Y2,Z2) are equal iff:
    // X1*Z2 == X2*Z1 and Y1*Z2 == Y2*Z1
    let lhs_x = fp_mul(p.x, q.z);
    let rhs_x = fp_mul(q.x, p.z);
    let lhs_y = fp_mul(p.y, q.z);
    let rhs_y = fp_mul(q.y, p.z);

    let eq = fp_eq(lhs_x, rhs_x) && fp_eq(lhs_y, rhs_y);

    // Store result in Z limb[0] of first point
    var result = output_points[i1];
    result.z.limbs[0] = select(0u, 1u, eq);
    output_points[i1] = result;
}
