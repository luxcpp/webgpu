// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Polynomial Arithmetic - WGSL port of cuda/kernels/lattice/poly_arithmetic.cu.
//
// 64-bit polynomial arithmetic for RLWE/RGSW pipelines, with u64 emulated
// using U64 = (lo: u32, hi: u32) pairs. Output is bit-identical to the CUDA
// and Metal counterparts.
//
// Buffer convention: each u64 coefficient occupies 2 consecutive u32 words
// (buf[2*i] = lo, buf[2*i + 1] = hi).

// =============================================================================
// 64-bit Emulation
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
fn u64_is_zero(a: U64) -> bool { return a.lo == 0u && a.hi == 0u; }

fn u64_lt(a: U64, b: U64) -> bool {
    return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
}

fn u64_gte(a: U64, b: U64) -> bool { return !u64_lt(a, b); }

fn u64_add(a: U64, b: U64) -> U64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    return U64(lo, a.hi + b.hi + carry);
}

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

fn u64_shl1(a: U64) -> U64 {
    let new_hi = (a.hi << 1u) | (a.lo >> 31u);
    let new_lo = a.lo << 1u;
    return U64(new_lo, new_hi);
}

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

fn u64_mul_hi(a: U64, b: U64) -> U64 {
    let p = u64_mul_128(a, b);
    return U64(p.w2, p.w3);
}

// =============================================================================
// Modular Arithmetic
// =============================================================================

fn mod_add(a: U64, b: U64, q: U64) -> U64 {
    let s = u64_add_with_carry(a, b);
    var sum = U64(s.x, s.y);
    if (s.z != 0u || u64_gte(sum, q)) {
        sum = u64_sub(sum, q);
    }
    return sum;
}

fn mod_sub(a: U64, b: U64, q: U64) -> U64 {
    if (u64_gte(a, b)) { return u64_sub(a, b); }
    return u64_sub(u64_add(a, q), b);
}

fn mod_neg(a: U64, q: U64) -> U64 {
    if (u64_is_zero(a)) { return a; }
    return u64_sub(q, a);
}

// Montgomery reduction of a 128-bit value (hi:lo).
fn mont_reduce(lo: U64, hi: U64, q: U64, q_inv: U64) -> U64 {
    let m_full = u64_mul_128(lo, q_inv);
    let m      = U64(m_full.w0, m_full.w1);
    let t_full = u64_mul_128(m, q);
    let t_hi   = U64(t_full.w2, t_full.w3);
    var result = u64_sub(hi, t_hi);
    if (u64_lt(hi, t_hi)) { result = u64_add(result, q); }
    if (u64_gte(result, q)) { result = u64_sub(result, q); }
    return result;
}

fn mont_mul(a: U64, b: U64, q: U64, q_inv: U64) -> U64 {
    let p  = u64_mul_128(a, b);
    let lo = U64(p.w0, p.w1);
    let hi = U64(p.w2, p.w3);
    return mont_reduce(lo, hi, q, q_inv);
}

// =============================================================================
// Bindings
// =============================================================================

struct PolyParams {
    q_lo:      u32,
    q_hi:      u32,
    q_inv_lo:  u32,
    q_inv_hi:  u32,
    mu_lo:     u32,
    mu_hi:     u32,
    r2_lo:     u32,
    r2_hi:     u32,
    n:         u32,
    batch:     u32,
    rows:      u32,
    cols:      u32,
}

@group(0) @binding(0) var<storage, read>       a_buf: array<u32>;
@group(0) @binding(1) var<storage, read>       b_buf: array<u32>;
@group(0) @binding(2) var<storage, read_write> c_buf: array<u32>;
@group(0) @binding(3) var<uniform>             params: PolyParams;

fn load_a(i: u32) -> U64 { return U64(a_buf[2u * i], a_buf[2u * i + 1u]); }
fn load_b(i: u32) -> U64 { return U64(b_buf[2u * i], b_buf[2u * i + 1u]); }
fn load_c(i: u32) -> U64 { return U64(c_buf[2u * i], c_buf[2u * i + 1u]); }
fn store_c(i: u32, v: U64) {
    c_buf[2u * i]      = v.lo;
    c_buf[2u * i + 1u] = v.hi;
}

fn param_q()     -> U64 { return U64(params.q_lo,     params.q_hi); }
fn param_q_inv() -> U64 { return U64(params.q_inv_lo, params.q_inv_hi); }
fn param_mu()    -> U64 { return U64(params.mu_lo,    params.mu_hi); }
fn param_r2()    -> U64 { return U64(params.r2_lo,    params.r2_hi); }

// =============================================================================
// Coefficient-wise operations
// =============================================================================

@compute @workgroup_size(256)
fn poly_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    store_c(gid.x, mod_add(load_a(gid.x), load_b(gid.x), param_q()));
}

@compute @workgroup_size(256)
fn poly_sub(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    store_c(gid.x, mod_sub(load_a(gid.x), load_b(gid.x), param_q()));
}

@compute @workgroup_size(256)
fn poly_neg(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    store_c(gid.x, mod_neg(load_a(gid.x), param_q()));
}

// Scalar lives in r2 slot of params for binding density.
@compute @workgroup_size(256)
fn poly_scalar_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    store_c(gid.x, mont_mul(load_a(gid.x), param_r2(), param_q(), param_q_inv()));
}

// =============================================================================
// NTT-domain multiplication
// =============================================================================

@compute @workgroup_size(256)
fn poly_mul_ntt(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    store_c(gid.x, mont_mul(load_a(gid.x), load_b(gid.x), param_q(), param_q_inv()));
}

@compute @workgroup_size(256)
fn poly_mul_add_ntt(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    let prod = mont_mul(load_a(gid.x), load_b(gid.x), param_q(), param_q_inv());
    store_c(gid.x, mod_add(load_c(gid.x), prod, param_q()));
}

@compute @workgroup_size(256)
fn poly_mul_sub_ntt(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    let prod = mont_mul(load_a(gid.x), load_b(gid.x), param_q(), param_q_inv());
    store_c(gid.x, mod_sub(load_c(gid.x), prod, param_q()));
}

// =============================================================================
// Batch variants
// =============================================================================

@compute @workgroup_size(256)
fn poly_batch_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.n * params.batch;
    if (gid.x >= total) { return; }
    store_c(gid.x, mod_add(load_a(gid.x), load_b(gid.x), param_q()));
}

@compute @workgroup_size(256)
fn poly_batch_mul_ntt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.n * params.batch;
    if (gid.x >= total) { return; }
    store_c(gid.x, mont_mul(load_a(gid.x), load_b(gid.x), param_q(), param_q_inv()));
}

// =============================================================================
// Reduction / Montgomery conversion.
// =============================================================================

@compute @workgroup_size(256)
fn poly_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    var v = load_c(gid.x);
    let q = param_q();
    loop {
        if (u64_lt(v, q)) { break; }
        v = u64_sub(v, q);
    }
    store_c(gid.x, v);
}

@compute @workgroup_size(256)
fn poly_to_mont(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    store_c(gid.x, mont_mul(load_a(gid.x), param_r2(), param_q(), param_q_inv()));
}

@compute @workgroup_size(256)
fn poly_from_mont(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    store_c(gid.x, mont_reduce(load_a(gid.x), u64_zero(), param_q(), param_q_inv()));
}

// =============================================================================
// Matrix-vector multiplication (NTT domain).
//
// One workgroup per row; threads stream over coefficients. The intermediate
// accumulator lives in workgroup memory.
// =============================================================================

var<workgroup> mvm_shmem: array<U64, 4096>;

@compute @workgroup_size(64)
fn poly_matrix_vec_mul(
    @builtin(workgroup_id) wgid:        vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let row    = wgid.x;
    let stride = 64u;
    if (row >= params.rows) { return; }

    for (var i = tid; i < params.n; i = i + stride) {
        mvm_shmem[i] = u64_zero();
    }
    workgroupBarrier();

    for (var col = 0u; col < params.cols; col = col + 1u) {
        for (var i = tid; i < params.n; i = i + stride) {
            let a_idx = (row * params.cols + col) * params.n + i;
            let v_idx = col * params.n + i;
            let a_val = U64(a_buf[2u * a_idx], a_buf[2u * a_idx + 1u]);
            let v_val = U64(b_buf[2u * v_idx], b_buf[2u * v_idx + 1u]);
            let prod  = mont_mul(a_val, v_val, param_q(), param_q_inv());
            mvm_shmem[i] = mod_add(mvm_shmem[i], prod, param_q());
        }
        workgroupBarrier();
    }

    for (var i = tid; i < params.n; i = i + stride) {
        let out_idx = row * params.n + i;
        c_buf[2u * out_idx]      = mvm_shmem[i].lo;
        c_buf[2u * out_idx + 1u] = mvm_shmem[i].hi;
    }
}

// =============================================================================
// Automorphism: a(X) -> a(X^k) over Z_q[X]/(X^n + 1).
//
// k is passed through r2_lo/r2_hi to avoid a separate uniform buffer.
// =============================================================================

@compute @workgroup_size(256)
fn poly_automorphism(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    let k       = params.r2_lo;
    let two_n   = 2u * params.n;
    var src     = (gid.x * k) % two_n;
    var negate  = false;
    if (src >= params.n) {
        src = src - params.n;
        negate = true;
    }
    let val = U64(a_buf[2u * src], a_buf[2u * src + 1u]);
    let out = select(val, mod_neg(val, param_q()), negate);
    store_c(gid.x, out);
}
