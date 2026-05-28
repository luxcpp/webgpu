// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Fp384 — BLS12-381 base field Montgomery arithmetic on 12×u32 limbs
// (little-endian). Bit-exact mirror of metal/src/shaders/crypto/bls12_381.metal.
//
// **This file is a documentation-and-canonical-source fragment**: WGSL has
// no module system; consumers (crypto/bls12_381.wgsl) embed bit-exact
// copies of the helpers below. Reviewing here is equivalent to reviewing
// the inline copies in every consumer.
//
// Layout convention:
//   The host buffer encodes a single Fp384 as 6 × u64 little-endian
//   (LuxFp384 from include/lux/gpu/crypto.h). On a little-endian device
//   that maps lane-for-lane onto 12 × u32, where:
//     u32 index 2·i + 0 = low 32 bits of u64 limb i
//     u32 index 2·i + 1 = high 32 bits of u64 limb i
//   So when the Metal code reads `a.limbs[i]` (u64), the WGSL code reads
//   `(a.limbs[2*i], a.limbs[2*i+1])` (u32 pair = u64p).
//
// Reduction invariant:
//   Every public Fp384 returned by fp384_add / fp384_sub / fp384_mont_mul /
//   fp384_sqr is fully reduced (< p). The Montgomery representation means
//   the "value" of an Fp384 is (limbs as int) · R^{-1} mod p with R = 2^384.
//
// Bit-exact correctness:
//   - fp384_add  : ≡ Metal fp384_add   (two-stage carry; final fp384_reduce).
//   - fp384_sub  : ≡ Metal fp384_sub   (sbb; if borrow, add p back).
//   - fp384_mont_mul : ≡ Metal fp384_mul (schoolbook + CIOS Montgomery).
//   - fp384_sqr  : ≡ fp384_mont_mul(a, a).
//   - fp384_inv  : Fermat, a^{p-2} mod p; uses BLS_P_MINUS_2.
//
// Side channels:
//   The Montgomery reduction loop is straight-line; the only branches are
//   the final canonical subtraction (fp384_reduce). Branches are on the
//   carry-out bit of the addition chain, which leaks at most one bit of
//   magnitude information about the random Montgomery-form value — that
//   bit is non-secret given the public p.

// Limb count: 12 × u32 = 384 bits.
struct Fp384 { limbs: array<u32, 12> }

// ----------------------------------------------------------------------------
// BLS12-381 constants (12 × u32 little-endian).
// p  = 0x1a0111ea397fe69a 4b1ba7b6434bacd7 64774b84f38512bf
//      6730d2a0f6b0f624 1eabfffeb153ffff b9feffffffffaaab
// ----------------------------------------------------------------------------
const FP384_P: array<u32, 12> = array<u32, 12>(
    0xffffaaabu, 0xb9feffffu,
    0xb153ffffu, 0x1eabfffeu,
    0xf6b0f624u, 0x6730d2a0u,
    0xf38512bfu, 0x64774b84u,
    0x434bacd7u, 0x4b1ba7b6u,
    0x397fe69au, 0x1a0111eau
);

// R mod p — Montgomery form of 1.
// Matches metal::BLS_R = { 0x760900000002fffd, 0xebf4000bc40c0002, ... }.
const FP384_R: array<u32, 12> = array<u32, 12>(
    0x0002fffdu, 0x76090000u,
    0xc40c0002u, 0xebf4000bu,
    0x53c758bau, 0x5f489857u,
    0x70525745u, 0x77ce5853u,
    0xa256ec6du, 0x5c071a97u,
    0xfa80e493u, 0x15f65ec3u
);

// R^2 mod p — for canonical → Montgomery lifting via fp384_mont_mul(x, R2).
// Matches metal::BLS_R2 = { 0xf4df1f341c341746, 0x0a76e6a609d104f1, ... }.
const FP384_R2: array<u32, 12> = array<u32, 12>(
    0x1c341746u, 0xf4df1f34u,
    0x09d104f1u, 0x0a76e6a6u,
    0x4c95b6d5u, 0x8de5476cu,
    0x939d83c0u, 0x67eb88a9u,
    0xb519952du, 0x9a793e85u,
    0x92cae3aau, 0x11988fe5u
);

// -p^{-1} mod 2^32 — Montgomery INV constant.
// Derived from metal::BLS_INV = 0x89f3fffcfffcfffd; we use only its low 32
// bits because our CIOS step uses 32-bit limbs.
const FP384_INV: u32 = 0xfffcfffdu;

// p - 2 for Fermat inversion (12 × u32).
const FP384_P_MINUS_2: array<u32, 12> = array<u32, 12>(
    0xffffaaa9u, 0xb9feffffu,
    0xb153ffffu, 0x1eabfffeu,
    0xf6b0f624u, 0x6730d2a0u,
    0xf38512bfu, 0x64774b84u,
    0x434bacd7u, 0x4b1ba7b6u,
    0x397fe69au, 0x1a0111eau
);

// ----------------------------------------------------------------------------
// 32×32 → 64 wide multiply via 16-bit half-decomposition.
// Returns vec2<u32>(lo, hi). Same shape as u64_pair.wgsl::u32_mul_wide.
// ----------------------------------------------------------------------------
fn fp384_mul32(a: u32, b: u32) -> vec2<u32> {
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

// ----------------------------------------------------------------------------
// Compare against P (constant). Returns true if a >= p.
// ----------------------------------------------------------------------------
fn fp384_gte_p(a: Fp384) -> bool {
    for (var i: i32 = 11; i >= 0; i = i - 1) {
        let ai = a.limbs[i];
        let pi = FP384_P[i];
        if (ai > pi) { return true; }
        if (ai < pi) { return false; }
    }
    return true;
}

// Conditional subtraction of P if a >= P.
fn fp384_reduce(a: Fp384) -> Fp384 {
    if (!fp384_gte_p(a)) { return a; }
    var r: Fp384;
    var borrow: u32 = 0u;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        let ai = a.limbs[i];
        let pi = FP384_P[i];
        // sbb: stage 1 a - borrow; stage 2 result - pi.
        let s1 = ai - borrow;
        let b1 = select(0u, 1u, ai < borrow);
        let s2 = s1 - pi;
        let b2 = select(0u, 1u, s1 < pi);
        r.limbs[i] = s2;
        borrow = b1 + b2;
    }
    return r;
}

// ----------------------------------------------------------------------------
// fp384_zero / fp384_one (Montgomery form).
// ----------------------------------------------------------------------------
fn fp384_zero() -> Fp384 {
    var r: Fp384;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) { r.limbs[i] = 0u; }
    return r;
}

fn fp384_one() -> Fp384 {
    var r: Fp384;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) { r.limbs[i] = FP384_R[i]; }
    return r;
}

fn fp384_is_zero(a: Fp384) -> bool {
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        if (a.limbs[i] != 0u) { return false; }
    }
    return true;
}

fn fp384_eq(a: Fp384, b: Fp384) -> bool {
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        if (a.limbs[i] != b.limbs[i]) { return false; }
    }
    return true;
}

// ----------------------------------------------------------------------------
// fp384_add — (a + b) mod p. Two-stage carry detection per limb.
// ----------------------------------------------------------------------------
fn fp384_add(a: Fp384, b: Fp384) -> Fp384 {
    var r: Fp384;
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
    return fp384_reduce(r);
}

// ----------------------------------------------------------------------------
// fp384_sub — (a - b) mod p. If borrow, add p back (two-stage carry).
// ----------------------------------------------------------------------------
fn fp384_sub(a: Fp384, b: Fp384) -> Fp384 {
    var r: Fp384;
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
            let pi = FP384_P[i];
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

// Doubling: c = 2a mod p.
fn fp384_double(a: Fp384) -> Fp384 { return fp384_add(a, a); }

// Negate: c = p - a (or 0 if a == 0).
fn fp384_neg(a: Fp384) -> Fp384 {
    if (fp384_is_zero(a)) { return a; }
    var r: Fp384;
    var borrow: u32 = 0u;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        let pi = FP384_P[i];
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

// ----------------------------------------------------------------------------
// fp384_mont_mul — Montgomery multiplication c = a · b · R^{-1} mod p.
//
// CIOS (Coarsely-Integrated-Operand-Scanning) Montgomery on 12 × u32 limbs.
// Bit-exact mirror of metal::fp384_mul which uses 6 × u64 limbs. The 32-bit
// version walks 12 outer steps; each outer step reduces one limb by
// multiplying by m = t[i] · INV mod 2^32 and adding m · p.
//
// t is a working buffer of 13 u32s (12 limbs + 1 high overflow). Carry
// handling uses the canonical two-stage adc pattern.
// ----------------------------------------------------------------------------
fn fp384_mont_mul(a: Fp384, b: Fp384) -> Fp384 {
    // 24-limb intermediate (full product can be 768 bits).
    var t: array<u32, 25>;
    for (var i: u32 = 0u; i < 25u; i = i + 1u) { t[i] = 0u; }

    // Phase 1: schoolbook a × b → t[0..23].
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        var carry: u32 = 0u;
        for (var j: u32 = 0u; j < 12u; j = j + 1u) {
            let prod = fp384_mul32(a.limbs[i], b.limbs[j]);
            // t[i+j] + prod.x + carry, capture both carry bits.
            let s1 = t[i + j] + prod.x;
            let c1 = select(0u, 1u, s1 < prod.x);
            let s2 = s1 + carry;
            let c2 = select(0u, 1u, s2 < carry);
            t[i + j] = s2;
            carry = prod.y + c1 + c2;
        }
        // Propagate final carry into the high overflow lane.
        let s_hi = t[i + 12u] + carry;
        let c_hi = select(0u, 1u, s_hi < carry);
        t[i + 12u] = s_hi;
        // Propagate further if needed.
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

    // Phase 2: CIOS Montgomery reduction.
    // For i in 0..12: m = t[i] · INV mod 2^32; t += m · p · 2^(32·i).
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        let m = t[i] * FP384_INV;
        var carry: u32 = 0u;
        for (var j: u32 = 0u; j < 12u; j = j + 1u) {
            let prod = fp384_mul32(m, FP384_P[j]);
            let s1 = t[i + j] + prod.x;
            let c1 = select(0u, 1u, s1 < prod.x);
            let s2 = s1 + carry;
            let c2 = select(0u, 1u, s2 < carry);
            t[i + j] = s2;
            carry = prod.y + c1 + c2;
        }
        // Propagate carry through high limbs.
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

    // Top half (t[12..23]) is the reduced result.
    var r: Fp384;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        r.limbs[i] = t[i + 12u];
    }
    return fp384_reduce(r);
}

// Square is just multiplication with self in this reference implementation.
// Consumers may inline a squaring-optimized variant if profiling demands it.
fn fp384_sqr(a: Fp384) -> Fp384 { return fp384_mont_mul(a, a); }

// ----------------------------------------------------------------------------
// fp384_inv — Fermat inversion: a^{-1} = a^{p-2} mod p.
//
// Square-and-multiply LSB-first. Cost: ≤ 384 squarings + ≤ 381 multiplies
// (p-2 has Hamming weight ≈ 192). All in Montgomery form: result is
// Montgomery-form inverse.
//
// Mirrors metal::fp384_inv -> fp384_pow(a, BLS_P_MINUS_2).
// ----------------------------------------------------------------------------
fn fp384_pow(base: Fp384, exp: array<u32, 12>) -> Fp384 {
    var result = fp384_one();
    var b = base;
    for (var limb: u32 = 0u; limb < 12u; limb = limb + 1u) {
        let e = exp[limb];
        for (var bit: u32 = 0u; bit < 32u; bit = bit + 1u) {
            if (((e >> bit) & 1u) == 1u) {
                result = fp384_mont_mul(result, b);
            }
            b = fp384_sqr(b);
        }
    }
    return result;
}

fn fp384_inv(a: Fp384) -> Fp384 { return fp384_pow(a, FP384_P_MINUS_2); }
