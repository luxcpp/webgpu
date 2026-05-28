// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// u64 emulation via u32-pair `vec2<u32>` — canonical reference for every
// WGSL kernel that needs 64-bit arithmetic. WGSL has no native 64-bit
// integer type; we encode u64 as `vec2<u32>(lo, hi)` where lo = bits 0..31
// and hi = bits 32..63.
//
// **This file is a documentation-and-canonical-source fragment**: WGSL has
// no module system, so kernels that need these helpers embed bit-exact
// copies of the same functions inline. The contract is that the inline
// copies match what is written here byte-for-byte (modulo whitespace and
// trivial renames). Reviewing helpers here is equivalent to reviewing the
// inline copies in every consumer.
//
// Bit-exact semantics:
//   - u64_add(a, b) → (sum, carry) where `sum` is the low 64 bits of a+b
//     and `carry` ∈ {0,1} signals whether the addition exceeded 2^64-1.
//   - u64_sub(a, b) → (diff, borrow) where `diff` is the wrap-around
//     subtraction a-b modulo 2^64, and `borrow` ∈ {0,1} signals a < b.
//   - u64_mul_lo_hi(a, b) → vec4<u32>(lo_lo, lo_hi, hi_lo, hi_hi),
//     i.e. the full 128-bit product (a.lo + a.hi·2^32)·(b.lo + b.hi·2^32).
//   - u64_shl(a, s) → low 64 bits of a · 2^s, s ∈ [0, 64].
//   - u64_shr(a, s) → ⌊a / 2^s⌋, s ∈ [0, 64].
//   - u64_cmp(a, b) → -1 / 0 / +1 for a<b / a==b / a>b (unsigned).
//
// Consumers (must keep their inline copies in sync with this file):
//   fhe/ntt_fused_extprod.wgsl        — u64 mod-q arithmetic
//   crypto/bls12_381.wgsl             — 12×u32 == 6× u64 limb stride helpers
//   crypto/goldilocks.wgsl            — pre-existing vec2<u32> = `GL` (older
//                                       naming; semantically identical).
//
// All functions are pure (no global state, no `var<workgroup>` access).

// ----------------------------------------------------------------------------
// 64-bit value type — u32 pair.
// vec2<u32>: x = lo (bits 0..31), y = hi (bits 32..63).
// ----------------------------------------------------------------------------
alias u64p = vec2<u32>;

fn u64_zero() -> u64p { return u64p(0u, 0u); }
fn u64_one()  -> u64p { return u64p(1u, 0u); }

fn u64_from_u32(x: u32) -> u64p { return u64p(x, 0u); }
fn u64_eq(a: u64p, b: u64p) -> bool { return a.x == b.x && a.y == b.y; }
fn u64_is_zero(a: u64p) -> bool { return a.x == 0u && a.y == 0u; }

// ----------------------------------------------------------------------------
// u64_add — unsigned 64-bit addition.
//   Returns vec3<u32>(lo, hi, carry) where carry ∈ {0, 1}.
//
// Canonical two-stage detection (matches the adc pattern in
// metal/src/shaders/crypto/bls12_381.metal::adc):
//   stage 1: a.x + b.x -> s1; c1 = (s1 < b.x)
//   stage 2: a.y + b.y + c1 -> s2; c2 = (s2 < b.y) || (c1==1 && s2==b.y)
// Each addend is independent; a triple-add never double-overflows in the
// 2^32 domain because max sum is 3·(2^32-1) < 2^34, so c1 + c2 ∈ {0,1,2}
// and the next adc up the chain consumes that via its own carry operand.
// ----------------------------------------------------------------------------
fn u64_add(a: u64p, b: u64p) -> vec3<u32> {
    let s1 = a.x + b.x;
    let c1 = select(0u, 1u, s1 < b.x);
    let s2_lo = a.y + b.y;
    let oc1 = select(0u, 1u, s2_lo < b.y);
    let s2 = s2_lo + c1;
    let oc2 = select(0u, 1u, s2 < c1);
    return vec3<u32>(s1, s2, oc1 + oc2);
}

// ----------------------------------------------------------------------------
// u64_sub — unsigned 64-bit subtraction with borrow.
//   Returns vec3<u32>(lo, hi, borrow) where borrow ∈ {0, 1}.
//
// Two-stage mirror of u64_add. Borrow bit is set when a < b unsigned.
// ----------------------------------------------------------------------------
fn u64_sub(a: u64p, b: u64p) -> vec3<u32> {
    let lo_borrow = select(0u, 1u, a.x < b.x);
    let lo = a.x - b.x;
    let hi_inter = a.y - lo_borrow;
    let hi_inter_borrow = select(0u, 1u, a.y < lo_borrow);
    let hi = hi_inter - b.y;
    let hi_borrow = select(0u, 1u, hi_inter < b.y);
    return vec3<u32>(lo, hi, hi_inter_borrow + hi_borrow);
}

// ----------------------------------------------------------------------------
// Helper: 32×32 → 64-bit multiply.  Returns vec2<u32>(lo, hi).
// Splits each input into hi/lo 16-bit halves and accumulates four partial
// products in 32-bit lanes with explicit carry. Safe for the WGSL "u32 wraps
// silently on overflow" semantics — no input pair can overflow a single
// partial product (each pair is ≤ (2^16-1)^2 < 2^32).
// ----------------------------------------------------------------------------
fn u32_mul_wide(a: u32, b: u32) -> u64p {
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
    return u64p(lo, hi);
}

// ----------------------------------------------------------------------------
// u64_mul_lo_hi — full 128-bit unsigned product of two u64s.
//   Returns vec4<u32>(w0, w1, w2, w3) where the product equals
//     w0 + w1·2^32 + w2·2^64 + w3·2^96.
//
// Schoolbook decomposition into four 32×32→64 partials, accumulated with
// canonical two-stage carry detection. This is the slow but bit-exact path;
// optimized callers (e.g. Goldilocks reduction) may bypass it.
// ----------------------------------------------------------------------------
fn u64_mul_lo_hi(a: u64p, b: u64p) -> vec4<u32> {
    let p00 = u32_mul_wide(a.x, b.x);   // bits   0..63
    let p01 = u32_mul_wide(a.x, b.y);   // bits  32..95
    let p10 = u32_mul_wide(a.y, b.x);   // bits  32..95
    let p11 = u32_mul_wide(a.y, b.y);   // bits  64..127

    var w0 = p00.x;
    var w1 = p00.y;
    var w2 = p11.x;
    var w3 = p11.y;

    // Add p01 at offset 32 (lanes w1, w2).
    let s_w1_a = w1 + p01.x;
    let c_w1_a = select(0u, 1u, s_w1_a < p01.x);
    w1 = s_w1_a;
    let s_w2_a = w2 + p01.y;
    let c_w2_a = select(0u, 1u, s_w2_a < p01.y);
    let s_w2_a2 = s_w2_a + c_w1_a;
    let c_w2_a2 = select(0u, 1u, s_w2_a2 < c_w1_a);
    w2 = s_w2_a2;
    w3 = w3 + c_w2_a + c_w2_a2;

    // Add p10 at offset 32 (lanes w1, w2).
    let s_w1_b = w1 + p10.x;
    let c_w1_b = select(0u, 1u, s_w1_b < p10.x);
    w1 = s_w1_b;
    let s_w2_b = w2 + p10.y;
    let c_w2_b = select(0u, 1u, s_w2_b < p10.y);
    let s_w2_b2 = s_w2_b + c_w1_b;
    let c_w2_b2 = select(0u, 1u, s_w2_b2 < c_w1_b);
    w2 = s_w2_b2;
    w3 = w3 + c_w2_b + c_w2_b2;

    return vec4<u32>(w0, w1, w2, w3);
}

// ----------------------------------------------------------------------------
// u64_shl — left shift, s in [0, 64].
// Returns the low 64 bits of a · 2^s.
// ----------------------------------------------------------------------------
fn u64_shl(a: u64p, s: u32) -> u64p {
    if (s == 0u) { return a; }
    if (s >= 64u) { return u64p(0u, 0u); }
    if (s >= 32u) {
        return u64p(0u, a.x << (s - 32u));
    }
    let lo = a.x << s;
    let hi = (a.y << s) | (a.x >> (32u - s));
    return u64p(lo, hi);
}

// ----------------------------------------------------------------------------
// u64_shr — logical right shift, s in [0, 64].
// Returns ⌊a / 2^s⌋.
// ----------------------------------------------------------------------------
fn u64_shr(a: u64p, s: u32) -> u64p {
    if (s == 0u) { return a; }
    if (s >= 64u) { return u64p(0u, 0u); }
    if (s >= 32u) {
        return u64p(a.y >> (s - 32u), 0u);
    }
    let lo = (a.x >> s) | (a.y << (32u - s));
    let hi = a.y >> s;
    return u64p(lo, hi);
}

// ----------------------------------------------------------------------------
// u64_cmp — unsigned 64-bit compare. Returns -1, 0, +1 encoded as i32.
// ----------------------------------------------------------------------------
fn u64_cmp(a: u64p, b: u64p) -> i32 {
    if (a.y < b.y) { return -1; }
    if (a.y > b.y) { return  1; }
    if (a.x < b.x) { return -1; }
    if (a.x > b.x) { return  1; }
    return 0;
}

fn u64_lt(a: u64p, b: u64p) -> bool { return u64_cmp(a, b) < 0; }
fn u64_gte(a: u64p, b: u64p) -> bool { return u64_cmp(a, b) >= 0; }
