// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// EVM256 — uint256 modular arithmetic primitives for FHE-EVM precompiles
// (WebGPU/WGSL). 256-bit values are stored as 4 little-endian u64 limbs;
// each u64 is two u32s (lo, hi). Each uint256 thus occupies 8 contiguous u32
// slots in storage buffers.
//
// Kernel/function names match the CUDA reference at
// cuda/kernels/fhe/evm256.cu and the Metal port at
// metal/src/shaders/fhe/evm256.metal so the dispatcher can route by name.
//
// 64-bit helpers follow the canonical pattern in
// webgpu/kernels/wgsl/common/u64_pair.wgsl (bit-exact inline copy, since WGSL
// has no module system).

// ============================================================================
// 64-bit Integer Emulation (canonical u64_pair pattern)
// ============================================================================

struct U64 {
    lo: u32,
    hi: u32,
}

fn u64_zero() -> U64 { return U64(0u, 0u); }
fn u64_one()  -> U64 { return U64(1u, 0u); }

fn u64_is_zero(a: U64) -> bool { return a.lo == 0u && a.hi == 0u; }

// 64-bit add. Returns sum.lo, sum.hi, carry-out in vec3.
fn u64_add_c(a: U64, b: U64) -> vec3<u32> {
    let s1 = a.lo + b.lo;
    let c1 = select(0u, 1u, s1 < b.lo);
    let s2_pre = a.hi + b.hi;
    let oc1 = select(0u, 1u, s2_pre < b.hi);
    let s2 = s2_pre + c1;
    let oc2 = select(0u, 1u, s2 < c1);
    return vec3<u32>(s1, s2, oc1 + oc2);
}

// 64-bit sub. Returns diff.lo, diff.hi, borrow-out in vec3.
fn u64_sub_c(a: U64, b: U64) -> vec3<u32> {
    let lo_borrow = select(0u, 1u, a.lo < b.lo);
    let lo = a.lo - b.lo;
    let hi_inter = a.hi - lo_borrow;
    let hi_inter_borrow = select(0u, 1u, a.hi < lo_borrow);
    let hi = hi_inter - b.hi;
    let hi_borrow = select(0u, 1u, hi_inter < b.hi);
    return vec3<u32>(lo, hi, hi_inter_borrow + hi_borrow);
}

fn u64_cmp(a: U64, b: U64) -> i32 {
    if (a.hi < b.hi) { return -1; }
    if (a.hi > b.hi) { return  1; }
    if (a.lo < b.lo) { return -1; }
    if (a.lo > b.lo) { return  1; }
    return 0;
}

fn u64_gte(a: U64, b: U64) -> bool { return u64_cmp(a, b) >= 0; }

// 32×32 → 64 multiply.
fn u32_mul_wide(a: u32, b: u32) -> U64 {
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
    return U64(lo, hi);
}

// 64×64 → 128 multiply, returns vec4<u32>(w0, w1, w2, w3)
// product = w0 + w1·2^32 + w2·2^64 + w3·2^96
fn u64_mul_128(a: U64, b: U64) -> vec4<u32> {
    let p00 = u32_mul_wide(a.lo, b.lo);
    let p01 = u32_mul_wide(a.lo, b.hi);
    let p10 = u32_mul_wide(a.hi, b.lo);
    let p11 = u32_mul_wide(a.hi, b.hi);

    var w0 = p00.lo;
    var w1 = p00.hi;
    var w2 = p11.lo;
    var w3 = p11.hi;

    // Add p01 at lane offset 1
    let s_w1_a = w1 + p01.lo;
    let c_w1_a = select(0u, 1u, s_w1_a < p01.lo);
    w1 = s_w1_a;
    let s_w2_a = w2 + p01.hi;
    let c_w2_a = select(0u, 1u, s_w2_a < p01.hi);
    let s_w2_a2 = s_w2_a + c_w1_a;
    let c_w2_a2 = select(0u, 1u, s_w2_a2 < c_w1_a);
    w2 = s_w2_a2;
    w3 = w3 + c_w2_a + c_w2_a2;

    // Add p10 at lane offset 1
    let s_w1_b = w1 + p10.lo;
    let c_w1_b = select(0u, 1u, s_w1_b < p10.lo);
    w1 = s_w1_b;
    let s_w2_b = w2 + p10.hi;
    let c_w2_b = select(0u, 1u, s_w2_b < p10.hi);
    let s_w2_b2 = s_w2_b + c_w1_b;
    let c_w2_b2 = select(0u, 1u, s_w2_b2 < c_w1_b);
    w2 = s_w2_b2;
    w3 = w3 + c_w2_b + c_w2_b2;

    return vec4<u32>(w0, w1, w2, w3);
}

// ============================================================================
// uint256 layout
//   Each element occupies 8 contiguous u32s: [l0.lo, l0.hi, l1.lo, l1.hi, ...]
//   where (l0, l1, l2, l3) are 64-bit little-endian limbs.
// ============================================================================

const LIMBS: u32 = 4u;
const U256_U32: u32 = 8u;  // 4 limbs × 2 u32

// Load limb `i` of uint256 element `idx` from a storage buffer.
fn load_limb(buf: ptr<storage, array<u32>, read>, idx: u32, i: u32) -> U64 {
    let base = idx * U256_U32 + i * 2u;
    return U64((*buf)[base], (*buf)[base + 1u]);
}

fn store_limb(buf: ptr<storage, array<u32>, read_write>, idx: u32, i: u32, v: U64) {
    let base = idx * U256_U32 + i * 2u;
    (*buf)[base]      = v.lo;
    (*buf)[base + 1u] = v.hi;
}

// ============================================================================
// Buffer bindings (shared for the simple add/sub/mul/div/mod kernels)
// ============================================================================

struct Evm256Params {
    n: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read>       evm_a: array<u32>;
@group(0) @binding(1) var<storage, read>       evm_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> evm_out: array<u32>;
@group(0) @binding(3) var<uniform>             evm_params: Evm256Params;

// ============================================================================
// kernel_add256: out = a + b (mod 2^256)
// ============================================================================
@compute @workgroup_size(64)
fn kernel_add256(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= evm_params.n) { return; }

    var carry: u32 = 0u;
    for (var i: u32 = 0u; i < LIMBS; i++) {
        let a_i = load_limb(&evm_a, idx, i);
        let b_i = load_limb(&evm_b, idx, i);

        // (a + b) with incoming carry, returning fresh carry.
        let s_ab = u64_add_c(a_i, b_i);
        let sum_ab = U64(s_ab.x, s_ab.y);
        let c_ab = s_ab.z;

        let s_total = u64_add_c(sum_ab, U64(carry, 0u));
        carry = c_ab + s_total.z;
        store_limb(&evm_out, idx, i, U64(s_total.x, s_total.y));
    }
}

// ============================================================================
// kernel_sub256: out = a - b (mod 2^256)
// ============================================================================
@compute @workgroup_size(64)
fn kernel_sub256(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= evm_params.n) { return; }

    var borrow: u32 = 0u;
    for (var i: u32 = 0u; i < LIMBS; i++) {
        let a_i = load_limb(&evm_a, idx, i);
        let b_i = load_limb(&evm_b, idx, i);

        let s_ab = u64_sub_c(a_i, b_i);
        let diff_ab = U64(s_ab.x, s_ab.y);
        let b_ab = s_ab.z;

        let s_total = u64_sub_c(diff_ab, U64(borrow, 0u));
        borrow = b_ab + s_total.z;
        store_limb(&evm_out, idx, i, U64(s_total.x, s_total.y));
    }
}

// ============================================================================
// kernel_mul256: out = lower 256 bits of (a * b)
// Schoolbook 4×4 limbs with explicit 64×64 → 128 partials.
// ============================================================================
@compute @workgroup_size(64)
fn kernel_mul256(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= evm_params.n) { return; }

    // product[k] holds bits 64*k..64*k+63 in a U64.
    var product: array<U64, 8>;
    for (var k: u32 = 0u; k < 8u; k++) { product[k] = u64_zero(); }

    for (var i: u32 = 0u; i < LIMBS; i++) {
        let a_i = load_limb(&evm_a, idx, i);
        var carry: U64 = u64_zero();
        for (var j: u32 = 0u; j < LIMBS; j++) {
            let b_j = load_limb(&evm_b, idx, j);

            // 64×64 → 128 product, w_lo + w_hi*2^64.
            let p = u64_mul_128(a_i, b_j);
            let p_lo = U64(p.x, p.y);
            let p_hi = U64(p.z, p.w);

            // sum = product[i+j] + p_lo + carry, carry := overflow + p_hi
            let s1 = u64_add_c(product[i + j], p_lo);
            let sum_a = U64(s1.x, s1.y);
            let c1 = s1.z;

            let s2 = u64_add_c(sum_a, carry);
            product[i + j] = U64(s2.x, s2.y);
            let c2 = s2.z;

            // new carry = p_hi + (c1 + c2) as 64-bit
            let c_total = c1 + c2;
            let cc = u64_add_c(p_hi, U64(c_total, 0u));
            carry = U64(cc.x, cc.y);  // u64×u64 → 128 carry never overflows u64 here
        }
        // Add carry into product[i + LIMBS]
        let scarry = u64_add_c(product[i + LIMBS], carry);
        product[i + LIMBS] = U64(scarry.x, scarry.y);
    }

    for (var k: u32 = 0u; k < LIMBS; k++) {
        store_limb(&evm_out, idx, k, product[k]);
    }
}

// ============================================================================
// uint256 compare (thread-local helper).
// ============================================================================
fn cmp256(a: array<U64, 4>, b: array<U64, 4>) -> i32 {
    for (var i: i32 = i32(LIMBS) - 1; i >= 0; i--) {
        let ai = a[u32(i)];
        let bi = b[u32(i)];
        let c = u64_cmp(ai, bi);
        if (c != 0) { return c; }
    }
    return 0;
}

fn load_u256(buf: ptr<storage, array<u32>, read>, idx: u32) -> array<U64, 4> {
    var r: array<U64, 4>;
    for (var i: u32 = 0u; i < LIMBS; i++) { r[i] = load_limb(buf, idx, i); }
    return r;
}

fn store_u256(buf: ptr<storage, array<u32>, read_write>, idx: u32, v: array<U64, 4>) {
    for (var i: u32 = 0u; i < LIMBS; i++) { store_limb(buf, idx, i, v[i]); }
}

fn u256_is_zero(v: array<U64, 4>) -> bool {
    for (var i: u32 = 0u; i < LIMBS; i++) {
        if (!u64_is_zero(v[i])) { return false; }
    }
    return true;
}

// ============================================================================
// Long division on thread-private uint256s.
// q = numerator / denominator, r = numerator mod denominator (bit-wise).
// Matches the bitwise long-division in evm256.cu / evm256.metal exactly.
// ============================================================================
fn div256_impl(
    numerator: array<U64, 4>,
    denominator: array<U64, 4>,
) -> array<U64, 8> {  // [q0..q3, r0..r3]
    var q: array<U64, 4>;
    var r: array<U64, 4>;
    for (var i: u32 = 0u; i < LIMBS; i++) {
        q[i] = u64_zero();
        r[i] = u64_zero();
    }

    var out: array<U64, 8>;
    if (u256_is_zero(denominator)) {
        for (var i: u32 = 0u; i < 8u; i++) { out[i] = u64_zero(); }
        return out;
    }

    for (var i: i32 = 255; i >= 0; i--) {
        // r <<= 1 (cross-limb shift)
        var carry_bit: u32 = 0u;
        for (var j: u32 = 0u; j < LIMBS; j++) {
            let lo = (r[j].lo << 1u) | carry_bit;
            let next_carry_lo = r[j].lo >> 31u;
            let hi = (r[j].hi << 1u) | next_carry_lo;
            carry_bit = r[j].hi >> 31u;
            r[j] = U64(lo, hi);
        }

        // bit i of numerator → r[0] LSB
        let limb_idx = u32(i) / 64u;
        let bit_idx = u32(i) % 64u;
        var bit: u32 = 0u;
        if (bit_idx < 32u) {
            bit = (numerator[limb_idx].lo >> bit_idx) & 1u;
        } else {
            bit = (numerator[limb_idx].hi >> (bit_idx - 32u)) & 1u;
        }
        r[0] = U64(r[0].lo | bit, r[0].hi);

        // if r >= denominator: r -= denominator, set quotient bit
        if (cmp256(r, denominator) >= 0) {
            var borrow: u32 = 0u;
            for (var j: u32 = 0u; j < LIMBS; j++) {
                let s_ab = u64_sub_c(r[j], denominator[j]);
                let diff_ab = U64(s_ab.x, s_ab.y);
                let b_ab = s_ab.z;
                let s_total = u64_sub_c(diff_ab, U64(borrow, 0u));
                borrow = b_ab + s_total.z;
                r[j] = U64(s_total.x, s_total.y);
            }
            // q[limb_idx] |= (1 << bit_idx)
            if (bit_idx < 32u) {
                q[limb_idx] = U64(q[limb_idx].lo | (1u << bit_idx), q[limb_idx].hi);
            } else {
                q[limb_idx] = U64(q[limb_idx].lo, q[limb_idx].hi | (1u << (bit_idx - 32u)));
            }
        }
    }

    for (var i: u32 = 0u; i < LIMBS; i++) { out[i] = q[i]; }
    for (var i: u32 = 0u; i < LIMBS; i++) { out[LIMBS + i] = r[i]; }
    return out;
}

// ============================================================================
// kernel_div256: quotient = a / b
// ============================================================================
@compute @workgroup_size(64)
fn kernel_div256(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= evm_params.n) { return; }

    let num = load_u256(&evm_a, idx);
    let den = load_u256(&evm_b, idx);
    let qr = div256_impl(num, den);

    var q: array<U64, 4>;
    for (var i: u32 = 0u; i < LIMBS; i++) { q[i] = qr[i]; }
    store_u256(&evm_out, idx, q);
}

// ============================================================================
// kernel_mod256: remainder = a mod b
// ============================================================================
@compute @workgroup_size(64)
fn kernel_mod256(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= evm_params.n) { return; }

    let num = load_u256(&evm_a, idx);
    let den = load_u256(&evm_b, idx);
    let qr = div256_impl(num, den);

    var r: array<U64, 4>;
    for (var i: u32 = 0u; i < LIMBS; i++) { r[i] = qr[LIMBS + i]; }
    store_u256(&evm_out, idx, r);
}

// ============================================================================
// Modular exponentiation (separate binding group so we have a 4th input).
//   modexp_a   = base
//   modexp_b   = exponent
//   modexp_m   = modulus
//   modexp_out = result
// ============================================================================

@group(0) @binding(4) var<storage, read>       modexp_m: array<u32>;

@compute @workgroup_size(64)
fn kernel_modexp256(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= evm_params.n) { return; }

    let base = load_u256(&evm_a, idx);
    let expo = load_u256(&evm_b, idx);
    let mod_v = load_u256(&modexp_m, idx);

    // res = 1
    var res: array<U64, 4>;
    res[0] = u64_one(); res[1] = u64_zero(); res[2] = u64_zero(); res[3] = u64_zero();

    // b = base
    var b: array<U64, 4> = base;

    for (var i: u32 = 0u; i < 256u; i++) {
        let limb_idx = i / 64u;
        let bit_idx = i % 64u;
        var bit: u32 = 0u;
        if (bit_idx < 32u) {
            bit = (expo[limb_idx].lo >> bit_idx) & 1u;
        } else {
            bit = (expo[limb_idx].hi >> (bit_idx - 32u)) & 1u;
        }

        if (bit == 1u) {
            // product = res * b (truncated to 256 bits, matches CUDA simplification)
            var product: array<U64, 8>;
            for (var k: u32 = 0u; k < 8u; k++) { product[k] = u64_zero(); }
            for (var j: u32 = 0u; j < LIMBS; j++) {
                var carry: U64 = u64_zero();
                for (var kk: u32 = 0u; kk < LIMBS; kk++) {
                    let p = u64_mul_128(res[j], b[kk]);
                    let p_lo = U64(p.x, p.y);
                    let p_hi = U64(p.z, p.w);

                    let s1 = u64_add_c(product[j + kk], p_lo);
                    let sum_a = U64(s1.x, s1.y);
                    let c1 = s1.z;
                    let s2 = u64_add_c(sum_a, carry);
                    product[j + kk] = U64(s2.x, s2.y);
                    let c2 = s2.z;

                    let c_total = c1 + c2;
                    let cc = u64_add_c(p_hi, U64(c_total, 0u));
                    carry = U64(cc.x, cc.y);
                }
                let scarry = u64_add_c(product[j + LIMBS], carry);
                product[j + LIMBS] = U64(scarry.x, scarry.y);
            }
            var temp: array<U64, 4>;
            for (var k: u32 = 0u; k < LIMBS; k++) { temp[k] = product[k]; }
            let qr = div256_impl(temp, mod_v);
            for (var k: u32 = 0u; k < LIMBS; k++) { res[k] = qr[LIMBS + k]; }
        }

        // b = (b * b) mod modulus
        {
            var product: array<U64, 8>;
            for (var k: u32 = 0u; k < 8u; k++) { product[k] = u64_zero(); }
            for (var j: u32 = 0u; j < LIMBS; j++) {
                var carry: U64 = u64_zero();
                for (var kk: u32 = 0u; kk < LIMBS; kk++) {
                    let p = u64_mul_128(b[j], b[kk]);
                    let p_lo = U64(p.x, p.y);
                    let p_hi = U64(p.z, p.w);

                    let s1 = u64_add_c(product[j + kk], p_lo);
                    let sum_a = U64(s1.x, s1.y);
                    let c1 = s1.z;
                    let s2 = u64_add_c(sum_a, carry);
                    product[j + kk] = U64(s2.x, s2.y);
                    let c2 = s2.z;

                    let c_total = c1 + c2;
                    let cc = u64_add_c(p_hi, U64(c_total, 0u));
                    carry = U64(cc.x, cc.y);
                }
                let scarry = u64_add_c(product[j + LIMBS], carry);
                product[j + LIMBS] = U64(scarry.x, scarry.y);
            }
            var temp: array<U64, 4>;
            for (var k: u32 = 0u; k < LIMBS; k++) { temp[k] = product[k]; }
            let qr = div256_impl(temp, mod_v);
            for (var k: u32 = 0u; k < LIMBS; k++) { b[k] = qr[LIMBS + k]; }
        }
    }

    store_u256(&evm_out, idx, res);
}

// ============================================================================
// Montgomery multiplication batch.
//   modexp_m used as the modulus buffer.
//   `mont_m_inv` is the precomputed -m^{-1} mod 2^64 scalar.
// ============================================================================

struct MontParams {
    n: u32,
    pad0: u32,
    m_inv_lo: u32,
    m_inv_hi: u32,
}

@group(0) @binding(5) var<uniform> mont_params: MontParams;

@compute @workgroup_size(64)
fn kernel_mont_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= mont_params.n) { return; }

    let a = load_u256(&evm_a, idx);
    let b = load_u256(&evm_b, idx);
    let m = load_u256(&modexp_m, idx);
    let m_inv = U64(mont_params.m_inv_lo, mont_params.m_inv_hi);

    // 512-bit product
    var product: array<U64, 8>;
    for (var k: u32 = 0u; k < 8u; k++) { product[k] = u64_zero(); }

    for (var i: u32 = 0u; i < LIMBS; i++) {
        var carry: U64 = u64_zero();
        for (var j: u32 = 0u; j < LIMBS; j++) {
            let p = u64_mul_128(a[i], b[j]);
            let p_lo = U64(p.x, p.y);
            let p_hi = U64(p.z, p.w);

            let s1 = u64_add_c(product[i + j], p_lo);
            let sum_a = U64(s1.x, s1.y);
            let c1 = s1.z;
            let s2 = u64_add_c(sum_a, carry);
            product[i + j] = U64(s2.x, s2.y);
            let c2 = s2.z;

            let c_total = c1 + c2;
            let cc = u64_add_c(p_hi, U64(c_total, 0u));
            carry = U64(cc.x, cc.y);
        }
        let scarry = u64_add_c(product[i + LIMBS], carry);
        product[i + LIMBS] = U64(scarry.x, scarry.y);
    }

    // Montgomery reduction
    for (var i: u32 = 0u; i < LIMBS; i++) {
        // u = product[i] * m_inv (low 64 bits)
        let mul = u64_mul_128(product[i], m_inv);
        let u = U64(mul.x, mul.y);

        var carry: U64 = u64_zero();
        for (var j: u32 = 0u; j < LIMBS; j++) {
            let p = u64_mul_128(u, m[j]);
            let p_lo = U64(p.x, p.y);
            let p_hi = U64(p.z, p.w);

            let s1 = u64_add_c(product[i + j], p_lo);
            let sum_a = U64(s1.x, s1.y);
            let c1 = s1.z;
            let s2 = u64_add_c(sum_a, carry);
            product[i + j] = U64(s2.x, s2.y);
            let c2 = s2.z;

            let c_total = c1 + c2;
            let cc = u64_add_c(p_hi, U64(c_total, 0u));
            carry = U64(cc.x, cc.y);
        }
        // Propagate remaining carry upward.
        var j: u32 = LIMBS;
        loop {
            if (j >= (8u - i) || u64_is_zero(carry)) { break; }
            let s = u64_add_c(product[i + j], carry);
            product[i + j] = U64(s.x, s.y);
            carry = U64(s.z, 0u);
            j = j + 1u;
        }
    }

    // Upper half is candidate result; conditionally subtract m.
    var upper: array<U64, 4>;
    for (var i: u32 = 0u; i < LIMBS; i++) { upper[i] = product[LIMBS + i]; }

    if (cmp256(upper, m) >= 0) {
        var borrow: u32 = 0u;
        for (var i: u32 = 0u; i < LIMBS; i++) {
            let s_ab = u64_sub_c(upper[i], m[i]);
            let diff_ab = U64(s_ab.x, s_ab.y);
            let b_ab = s_ab.z;
            let s_total = u64_sub_c(diff_ab, U64(borrow, 0u));
            borrow = b_ab + s_total.z;
            upper[i] = U64(s_total.x, s_total.y);
        }
    }

    store_u256(&evm_out, idx, upper);
}
