// Copyright (c) 2025-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// WGSL port of dex/gpu/metal/amm_xyk.metal — batched xy=k AMM curve
// evaluation. One thread per (reserve_x, reserve_y, amount) tuple computes
//
//     out = (amount * reserve_y) / (reserve_x + amount)
//
// All values are u64 (Metal ulong). WGSL has no native u64, so we encode
// every 64-bit value as vec2<u32>(lo, hi). The numerator amount*reserve_y
// can overflow 64 bits; we keep it as a 128-bit intermediate (vec4<u32>)
// and divide by the 64-bit denominator via the standard shift-subtract
// loop.
//
// Bit-exact parity with the Metal kernel: same name (`amm_xyk_eval`),
// same buffer ordering (reserves, amounts, outs, n), same workgroup
// dimensions (the host launches `n` threads with workgroup_size 64).

// =============================================================================
// Storage bindings — exact buffer-index parity with the Metal driver.
// Metal: reserves @ buffer(0), amounts @ buffer(1), outs @ buffer(2),
//        n @ buffer(3) (constant uint&).
// =============================================================================

struct ReservePair {
    // ulong reserve_x  → vec2<u32>(lo, hi)
    reserve_x_lo: u32,
    reserve_x_hi: u32,
    // ulong reserve_y  → vec2<u32>(lo, hi)
    reserve_y_lo: u32,
    reserve_y_hi: u32,
}

struct AmmParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read>       reserves: array<ReservePair>;
@group(0) @binding(1) var<storage, read>       amounts:  array<vec2<u32>>;     // u64 amount (lo, hi)
@group(0) @binding(2) var<storage, read_write> outs:     array<vec2<u32>>;     // u64 out   (lo, hi)
@group(0) @binding(3) var<uniform>             params:   AmmParams;

// =============================================================================
// 64-bit helpers — vec2<u32>(lo, hi). Same shape as webgpu/kernels/wgsl/
// common/u64_pair.wgsl. WGSL has no module system, so we inline the bit-
// exact subset this kernel needs.
// =============================================================================

fn u64_eq_zero(a: vec2<u32>) -> bool {
    return a.x == 0u && a.y == 0u;
}

// Returns (sum_lo, sum_hi, carry). carry ∈ {0, 1}.
fn u64_add_carry(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let s_lo = a.x + b.x;
    let c1: u32 = select(0u, 1u, s_lo < a.x);
    let s_hi_inter = a.y + b.y;
    let oc1: u32 = select(0u, 1u, s_hi_inter < a.y);
    let s_hi = s_hi_inter + c1;
    let oc2: u32 = select(0u, 1u, s_hi < s_hi_inter);
    return vec3<u32>(s_lo, s_hi, oc1 + oc2);
}

// Returns vec2<u32>(lo, hi) for the low 64 bits of a + b (carry discarded).
fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let r = u64_add_carry(a, b);
    return vec2<u32>(r.x, r.y);
}

// a >= b, unsigned 64-bit.
fn u64_gte(a: vec2<u32>, b: vec2<u32>) -> bool {
    if (a.y > b.y) { return true; }
    if (a.y < b.y) { return false; }
    return a.x >= b.x;
}

// a - b (mod 2^64), unsigned.
fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo_borrow: u32 = select(0u, 1u, a.x < b.x);
    let lo = a.x - b.x;
    // hi = a.y - b.y - lo_borrow
    let hi_inter = a.y - lo_borrow;
    let hi = hi_inter - b.y;
    return vec2<u32>(lo, hi);
}

// 32×32 → 64-bit unsigned multiply. Returns vec2<u32>(lo, hi).
fn u32_mul_wide(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;

    let mid = lh + hl;
    let mid_carry: u32 = select(0u, 0x10000u, mid < lh);

    let lo = ll + ((mid & 0xFFFFu) << 16u);
    let lo_carry: u32 = select(0u, 1u, lo < ll);

    let hi = hh + (mid >> 16u) + mid_carry + lo_carry;
    return vec2<u32>(lo, hi);
}

// 64×64 → 128-bit unsigned multiply. Returns vec4<u32>(w0, w1, w2, w3)
// where product = w0 + w1·2^32 + w2·2^64 + w3·2^96.
//
// Same schoolbook decomposition as Metal's mul_u64_u64 and the canonical
// helper in webgpu/kernels/wgsl/common/u64_pair.wgsl::u64_mul_lo_hi.
fn u64_mul_128(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
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
    let c_w1_a: u32 = select(0u, 1u, s_w1_a < p01.x);
    w1 = s_w1_a;
    let s_w2_a = w2 + p01.y;
    let c_w2_a: u32 = select(0u, 1u, s_w2_a < p01.y);
    let s_w2_a2 = s_w2_a + c_w1_a;
    let c_w2_a2: u32 = select(0u, 1u, s_w2_a2 < c_w1_a);
    w2 = s_w2_a2;
    w3 = w3 + c_w2_a + c_w2_a2;

    // Add p10 at offset 32 (lanes w1, w2).
    let s_w1_b = w1 + p10.x;
    let c_w1_b: u32 = select(0u, 1u, s_w1_b < p10.x);
    w1 = s_w1_b;
    let s_w2_b = w2 + p10.y;
    let c_w2_b: u32 = select(0u, 1u, s_w2_b < p10.y);
    let s_w2_b2 = s_w2_b + c_w1_b;
    let c_w2_b2: u32 = select(0u, 1u, s_w2_b2 < c_w1_b);
    w2 = s_w2_b2;
    w3 = w3 + c_w2_b + c_w2_b2;

    return vec4<u32>(w0, w1, w2, w3);
}

// =============================================================================
// 128/64 → 64-bit unsigned division via shift-subtract.
//
// Mirrors the Metal `div_u128_by_u64` loop: when the numerator's top half is
// zero, fall through to the simple u64/u64 path. Otherwise process the 128
// bits MSB-first, tracking remainder modulo denom and OR-ing 1 into the
// quotient at each bit position that fits in the low 64 bits.
//
// `denom` MUST be non-zero — caller guarantees this (matches Metal contract).
// Caller MUST ensure the true quotient fits in 64 bits (for AMM workload it
// does: amount * reserve_y / (reserve_x + amount) < reserve_y < 2^64).
//
// num is vec4<u32>(lo_lo, lo_hi, hi_lo, hi_hi). low 64 = (lo_lo, lo_hi),
// high 64 = (hi_lo, hi_hi).
// =============================================================================

fn div_u128_by_u64(num: vec4<u32>, denom: vec2<u32>) -> vec2<u32> {
    // Trivial path: numerator already fits in 64 bits.
    if (num.z == 0u && num.w == 0u) {
        // lo / denom — full long division on u64/u64.
        return div_u64_by_u64(vec2<u32>(num.x, num.y), denom);
    }

    var quotient: vec2<u32> = vec2<u32>(0u, 0u);
    var remainder: vec2<u32> = vec2<u32>(0u, 0u);

    // 128-bit MSB-first. Each iteration shifts remainder left by 1, brings
    // in the next bit from num, and subtracts denom if remainder >= denom.
    for (var i: i32 = 127; i >= 0; i = i - 1) {
        // remainder <<= 1
        let r_lo_new = remainder.x << 1u;
        let r_hi_new = (remainder.y << 1u) | (remainder.x >> 31u);
        remainder = vec2<u32>(r_lo_new, r_hi_new);

        // Pull bit i out of num. num layout: bits 0..31 = num.x, 32..63 =
        // num.y, 64..95 = num.z, 96..127 = num.w.
        let u_i = u32(i);
        var bit: u32 = 0u;
        if (u_i < 32u) {
            bit = (num.x >> u_i) & 1u;
        } else if (u_i < 64u) {
            bit = (num.y >> (u_i - 32u)) & 1u;
        } else if (u_i < 96u) {
            bit = (num.z >> (u_i - 64u)) & 1u;
        } else {
            bit = (num.w >> (u_i - 96u)) & 1u;
        }
        remainder.x = remainder.x | bit;

        if (u64_gte(remainder, denom)) {
            remainder = u64_sub(remainder, denom);
            // Set bit i of quotient, but only the low 64 bits matter
            // (caller contract). Bits >= 64 are silently discarded.
            if (u_i < 32u) {
                quotient.x = quotient.x | (1u << u_i);
            } else if (u_i < 64u) {
                quotient.y = quotient.y | (1u << (u_i - 32u));
            }
        }
    }
    return quotient;
}

// 64/64 → 64 unsigned division via shift-subtract. Used only on the fast
// path of div_u128_by_u64; same bit-exact recipe.
fn div_u64_by_u64(num: vec2<u32>, denom: vec2<u32>) -> vec2<u32> {
    var quotient: vec2<u32> = vec2<u32>(0u, 0u);
    var remainder: vec2<u32> = vec2<u32>(0u, 0u);

    for (var i: i32 = 63; i >= 0; i = i - 1) {
        let r_lo_new = remainder.x << 1u;
        let r_hi_new = (remainder.y << 1u) | (remainder.x >> 31u);
        remainder = vec2<u32>(r_lo_new, r_hi_new);

        let u_i = u32(i);
        var bit: u32 = 0u;
        if (u_i < 32u) {
            bit = (num.x >> u_i) & 1u;
        } else {
            bit = (num.y >> (u_i - 32u)) & 1u;
        }
        remainder.x = remainder.x | bit;

        if (u64_gte(remainder, denom)) {
            remainder = u64_sub(remainder, denom);
            if (u_i < 32u) {
                quotient.x = quotient.x | (1u << u_i);
            } else {
                quotient.y = quotient.y | (1u << (u_i - 32u));
            }
        }
    }
    return quotient;
}

// =============================================================================
// Main kernel — function name matches Metal kernel `amm_xyk_eval` so the
// host dispatcher uses one symbol across all three backends.
// =============================================================================

@compute @workgroup_size(64)
fn amm_xyk_eval(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let tid = gid.x;
    if (tid >= params.n) { return; }

    let r = reserves[tid];
    let rx = vec2<u32>(r.reserve_x_lo, r.reserve_x_hi);
    let ry = vec2<u32>(r.reserve_y_lo, r.reserve_y_hi);
    let a  = amounts[tid];

    // denom = reserve_x + amount. Carry discarded — Metal kernel allows
    // 64-bit wrap because the next compare detects denom == 0.
    let denom = u64_add(rx, a);
    if (u64_eq_zero(denom)) {
        outs[tid] = vec2<u32>(0u, 0u);
        return;
    }

    // 128-bit product a * reserve_y.
    let prod = u64_mul_128(a, ry);

    outs[tid] = div_u128_by_u64(prod, denom);
}
