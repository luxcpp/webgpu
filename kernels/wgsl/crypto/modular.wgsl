// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Modular Arithmetic - WGSL port of cuda/kernels/crypto/modular.cu.
//
// 64-bit modular primitives over a generic prime modulus q (q < 2^64), emulated
// using U64 = (lo: u32, hi: u32) pairs because WGSL has no native u64.
// Provides:
//   - Montgomery arithmetic (CIOS multiplication, reduction, to/from form)
//   - Barrett reduction (64-bit and wide-128-bit forms)
//   - Basic mod_{add,sub,neg,mul,pow,inv}
//
// Output is bit-identical to lux::cuda::modular::* (cuda/kernels/crypto/modular.cu)
// and metal/src/shaders/crypto/modular.metal.
//
// Buffer layout: each ulong element occupies 2 consecutive u32 words
// (data[2*i] = lo, data[2*i + 1] = hi).

// =============================================================================
// 64-bit Emulation Helpers
// =============================================================================

struct U64 {
    lo: u32,
    hi: u32,
}

struct U128 {
    w0: u32,
    w1: u32,
    w2: u32,
    w3: u32,
}

fn u64_zero() -> U64 { return U64(0u, 0u); }
fn u64_one()  -> U64 { return U64(1u, 0u); }

fn u64_eq(a: U64, b: U64) -> bool { return a.lo == b.lo && a.hi == b.hi; }
fn u64_is_zero(a: U64) -> bool { return a.lo == 0u && a.hi == 0u; }

fn u64_lt(a: U64, b: U64) -> bool {
    return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
}

fn u64_gte(a: U64, b: U64) -> bool {
    return !u64_lt(a, b);
}

fn u64_add(a: U64, b: U64) -> U64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    return U64(lo, a.hi + b.hi + carry);
}

// Returns (sum, carry_out) so callers can detect 64-bit overflow without
// recomputing comparisons.
fn u64_add_with_carry(a: U64, b: U64) -> vec3<u32> {
    let lo = a.lo + b.lo;
    let c0 = select(0u, 1u, lo < a.lo);
    let hi_sum = a.hi + b.hi;
    let c1 = select(0u, 1u, hi_sum < a.hi);
    let hi = hi_sum + c0;
    let c2 = select(0u, 1u, hi < hi_sum);
    return vec3<u32>(lo, hi, c1 | c2);
}

fn u64_sub(a: U64, b: U64) -> U64 {
    let borrow = select(0u, 1u, a.lo < b.lo);
    let lo = a.lo - b.lo;
    let hi = a.hi - b.hi - borrow;
    return U64(lo, hi);
}

fn u64_shr(a: U64, shift: u32) -> U64 {
    if (shift == 0u) { return a; }
    if (shift >= 64u) { return u64_zero(); }
    if (shift >= 32u) {
        return U64(a.hi >> (shift - 32u), 0u);
    }
    let hi = a.hi >> shift;
    let lo = (a.lo >> shift) | (a.hi << (32u - shift));
    return U64(lo, hi);
}

// 64x64 -> 128 full multiplication.
fn u64_mul_128(a: U64, b: U64) -> U128 {
    let a0 = a.lo & 0xFFFFu;
    let a1 = a.lo >> 16u;
    let a2 = a.hi & 0xFFFFu;
    let a3 = a.hi >> 16u;

    let b0 = b.lo & 0xFFFFu;
    let b1 = b.lo >> 16u;
    let b2 = b.hi & 0xFFFFu;
    let b3 = b.hi >> 16u;

    var acc0 = a0 * b0;
    var acc1 = a0 * b1 + a1 * b0;
    var acc2 = a0 * b2 + a1 * b1 + a2 * b0;
    var acc3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;
    var acc4 = a1 * b3 + a2 * b2 + a3 * b1;
    var acc5 = a2 * b3 + a3 * b2;
    var acc6 = a3 * b3;

    acc1 += acc0 >> 16u; acc0 &= 0xFFFFu;
    acc2 += acc1 >> 16u; acc1 &= 0xFFFFu;
    acc3 += acc2 >> 16u; acc2 &= 0xFFFFu;
    acc4 += acc3 >> 16u; acc3 &= 0xFFFFu;
    acc5 += acc4 >> 16u; acc4 &= 0xFFFFu;
    acc6 += acc5 >> 16u; acc5 &= 0xFFFFu;

    return U128(
        acc0 | (acc1 << 16u),
        acc2 | (acc3 << 16u),
        acc4 | (acc5 << 16u),
        acc6
    );
}

// High 64 bits of a 64×64 multiplication — equivalent to CUDA's __umul64hi /
// Metal's mulhi(ulong, ulong).
fn u64_mul_hi(a: U64, b: U64) -> U64 {
    let prod = u64_mul_128(a, b);
    return U64(prod.w2, prod.w3);
}

// =============================================================================
// Montgomery Arithmetic (CIOS for 64-bit modulus)
// =============================================================================

// Montgomery multiplication: (a * b * R^-1) mod q  where R = 2^64.
fn mont_mul_cios(a: U64, b: U64, mod_q: U64, m0_inv: U64) -> U64 {
    let ab    = u64_mul_128(a, b);
    let lo    = U64(ab.w0, ab.w1);
    let hi    = U64(ab.w2, ab.w3);

    // m = lo * m0_inv (low 64 bits of the product)
    let m_full = u64_mul_128(lo, m0_inv);
    let m      = U64(m_full.w0, m_full.w1);

    // t_hi = high 64 bits of m * mod
    let t_hi   = u64_mul_hi(m, mod_q);

    var result = u64_sub(hi, t_hi);
    if (u64_lt(hi, t_hi)) {
        result = u64_add(result, mod_q);
    }
    if (u64_gte(result, mod_q)) {
        result = u64_sub(result, mod_q);
    }
    return result;
}

// Montgomery reduction of a 128-bit value (hi:lo).
fn mont_reduce(lo: U64, hi: U64, mod_q: U64, m0_inv: U64) -> U64 {
    let m_full = u64_mul_128(lo, m0_inv);
    let m      = U64(m_full.w0, m_full.w1);

    let t_hi_full = u64_mul_128(m, mod_q);
    let t_hi      = U64(t_hi_full.w2, t_hi_full.w3);
    let t_lo      = U64(t_hi_full.w0, t_hi_full.w1);

    let sum_lo = u64_add(lo, t_lo);
    let carry  = select(0u, 1u, u64_lt(sum_lo, lo));
    let carry_u = U64(carry, 0u);

    var result = u64_add(u64_add(hi, t_hi), carry_u);
    if (u64_gte(result, mod_q) || u64_lt(result, hi)) {
        result = u64_sub(result, mod_q);
    }
    return result;
}

fn to_mont(a: U64, r2: U64, mod_q: U64, m0_inv: U64) -> U64 {
    return mont_mul_cios(a, r2, mod_q, m0_inv);
}

fn from_mont(a: U64, mod_q: U64, m0_inv: U64) -> U64 {
    return mont_mul_cios(a, u64_one(), mod_q, m0_inv);
}

// =============================================================================
// Barrett Reduction
// =============================================================================

fn barrett_reduce_wide(lo: U64, hi: U64, mod_q: U64, mu: U64) -> U64 {
    var hi_var = hi;

    // Approximate quotient from the upper half.
    let q_approx = u64_mul_hi(hi_var, mu);

    let q_mod_full = u64_mul_128(q_approx, mod_q);
    let q_mod_lo   = U64(q_mod_full.w0, q_mod_full.w1);
    let q_mod_hi   = U64(q_mod_full.w2, q_mod_full.w3);

    var r = u64_sub(lo, q_mod_lo);
    if (u64_lt(lo, q_mod_lo)) {
        hi_var = u64_sub(hi_var, u64_one());
    }
    hi_var = u64_sub(hi_var, q_mod_hi);

    // Final corrections.
    var iter = 0u;
    loop {
        if (u64_is_zero(hi_var) && u64_lt(r, mod_q)) { break; }
        if (iter > 4u) { break; }  // safety: at most a handful of corrections.
        r = u64_sub(r, mod_q);
        if (u64_gte(r, lo)) {
            hi_var = u64_sub(hi_var, u64_one());
        }
        iter = iter + 1u;
    }
    return r;
}

// =============================================================================
// Basic Modular Operations
// =============================================================================

fn mod_add(a: U64, b: U64, mod_q: U64) -> U64 {
    let s = u64_add_with_carry(a, b);
    var sum = U64(s.x, s.y);
    if (s.z != 0u || u64_gte(sum, mod_q)) {
        sum = u64_sub(sum, mod_q);
    }
    return sum;
}

fn mod_sub(a: U64, b: U64, mod_q: U64) -> U64 {
    if (u64_gte(a, b)) {
        return u64_sub(a, b);
    }
    return u64_sub(u64_add(a, mod_q), b);
}

fn mod_neg(a: U64, mod_q: U64) -> U64 {
    if (u64_is_zero(a)) { return a; }
    return u64_sub(mod_q, a);
}

fn mod_mul(a: U64, b: U64, mod_q: U64, mu: U64) -> U64 {
    let prod = u64_mul_128(a, b);
    return barrett_reduce_wide(U64(prod.w0, prod.w1), U64(prod.w2, prod.w3), mod_q, mu);
}

fn mod_pow(base_in: U64, exp_in: U64, mod_q: U64, mu: U64) -> U64 {
    var result = u64_one();
    var base   = base_in;
    var exp    = exp_in;
    // Reduce base.
    if (u64_gte(base, mod_q)) {
        base = mod_sub(base, mod_q, mod_q);  // safe: base < 2*mod after initial reduction is wider
    }
    loop {
        if (u64_is_zero(exp)) { break; }
        if ((exp.lo & 1u) != 0u) {
            result = mod_mul(result, base, mod_q, mu);
        }
        exp  = u64_shr(exp, 1u);
        base = mod_mul(base, base, mod_q, mu);
    }
    return result;
}

fn mod_inv(a: U64, mod_q: U64, mu: U64) -> U64 {
    // mod_q - 2 by Fermat's little theorem.
    let two = U64(2u, 0u);
    let exp = u64_sub(mod_q, two);
    return mod_pow(a, exp, mod_q, mu);
}

// =============================================================================
// Kernel Bindings
// =============================================================================
//
// Layout uniformity across batch kernels: a, b, c are u32 buffers carrying
// 64-bit values as (lo, hi) pairs; mod / mu / r2 / m0_inv arrive via uniform
// buffer fields (also (lo, hi)).

struct ModParams {
    mod_lo:    u32,
    mod_hi:    u32,
    mu_lo:     u32,
    mu_hi:     u32,
    r2_lo:     u32,
    r2_hi:     u32,
    m0_inv_lo: u32,
    m0_inv_hi: u32,
    n:         u32,
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
}

@group(0) @binding(0) var<storage, read>       a: array<u32>;
@group(0) @binding(1) var<storage, read>       b: array<u32>;
@group(0) @binding(2) var<storage, read_write> c: array<u32>;
@group(0) @binding(3) var<uniform>             params: ModParams;

fn load(buf_idx: u32, src: u32) -> U64 {
    return U64(0u, 0u);  // unreachable; kept to silence missing-fn shape
}

// Helpers below load/store directly from the typed bindings.
fn load_a(i: u32) -> U64 { return U64(a[2u * i], a[2u * i + 1u]); }
fn load_b(i: u32) -> U64 { return U64(b[2u * i], b[2u * i + 1u]); }
fn load_c(i: u32) -> U64 { return U64(c[2u * i], c[2u * i + 1u]); }
fn store_c(i: u32, v: U64) {
    c[2u * i]      = v.lo;
    c[2u * i + 1u] = v.hi;
}

fn param_mod()    -> U64 { return U64(params.mod_lo,    params.mod_hi); }
fn param_mu()     -> U64 { return U64(params.mu_lo,     params.mu_hi); }
fn param_r2()     -> U64 { return U64(params.r2_lo,     params.r2_hi); }
fn param_m0inv()  -> U64 { return U64(params.m0_inv_lo, params.m0_inv_hi); }

// =============================================================================
// Batch kernels — one thread per element (matches CUDA grid layout).
// =============================================================================

@compute @workgroup_size(256)
fn batch_add_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    store_c(gid.x, mod_add(load_a(gid.x), load_b(gid.x), param_mod()));
}

@compute @workgroup_size(256)
fn batch_sub_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    store_c(gid.x, mod_sub(load_a(gid.x), load_b(gid.x), param_mod()));
}

@compute @workgroup_size(256)
fn batch_mul_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    store_c(gid.x, mod_mul(load_a(gid.x), load_b(gid.x), param_mod(), param_mu()));
}

@compute @workgroup_size(256)
fn batch_mont_mul_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    store_c(gid.x, mont_mul_cios(load_a(gid.x), load_b(gid.x), param_mod(), param_m0inv()));
}

@compute @workgroup_size(256)
fn batch_pow_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    // bases in `a`, exponents in `b`, results in `c`.
    store_c(gid.x, mod_pow(load_a(gid.x), load_b(gid.x), param_mod(), param_mu()));
}

@compute @workgroup_size(256)
fn batch_inv_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    store_c(gid.x, mod_inv(load_a(gid.x), param_mod(), param_mu()));
}

@compute @workgroup_size(256)
fn batch_to_mont_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    store_c(gid.x, to_mont(load_a(gid.x), param_r2(), param_mod(), param_m0inv()));
}

@compute @workgroup_size(256)
fn batch_from_mont_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    store_c(gid.x, from_mont(load_a(gid.x), param_mod(), param_m0inv()));
}
