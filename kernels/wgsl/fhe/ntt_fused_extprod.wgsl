// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// NTT-fused TFHE external product — WGSL port of
// metal/src/shaders/fhe/ntt_fused_extprod.metal.
//
// Bit-exact mirror of the Metal kernels: every modular operation
// (mod_add_q / mod_sub_q / mod_mul_q / mod_neg_q) walks the same
// arithmetic on the same input. The differences vs Metal:
//   - WGSL has no native u64. We represent each u64 mod-q value as
//     `vec2<u32>(lo, hi)` (= `u64p` from common/u64_pair.wgsl). The host
//     buffer is still 64-bit per element; on a little-endian device this
//     matches lane-for-lane.
//   - Multiply is done by full 128-bit schoolbook then shift-and-subtract
//     mod-q reduction (matching the Metal "bit-by-bit" loop).
//   - All seven entry points keep their Metal names so the host plugin
//     can fetch pipelines by literal string.
//   - WGSL doesn't accept `ptr<storage>` function parameters by default
//     (that's an extension), so the per-buffer helpers are inlined in
//     each entry point.
//
// Bit-exact contract:
//   For identical (decomp[], BSK[], psi_pow[], q, N, rows, cols) inputs,
//   the WGSL `fhe_untwist_accumulate` writes the same acc[c][j] bytes as
//   Metal `fhe_untwist_accumulate`. Validated end-to-end by decrypting
//   PBS outputs from both backends and checking they decode to the same
//   plaintext (test_fhe_ntt_parity in gpu/test/).
//
// Helpers below are bit-exact copies of common/u64_pair.wgsl. WGSL has
// no `#include`; keep the two in sync.

// ============================================================================
// u64 emulation (vec2<u32>(lo, hi)) — bit-exact mirror of common/u64_pair.wgsl.
// ============================================================================
alias u64p = vec2<u32>;

fn u64_add(a: u64p, b: u64p) -> vec3<u32> {
    let s1 = a.x + b.x;
    let c1 = select(0u, 1u, s1 < b.x);
    let s2_lo = a.y + b.y;
    let oc1 = select(0u, 1u, s2_lo < b.y);
    let s2 = s2_lo + c1;
    let oc2 = select(0u, 1u, s2 < c1);
    return vec3<u32>(s1, s2, oc1 + oc2);
}

fn u64_sub(a: u64p, b: u64p) -> vec3<u32> {
    let lo_borrow = select(0u, 1u, a.x < b.x);
    let lo = a.x - b.x;
    let hi_inter = a.y - lo_borrow;
    let hi_inter_borrow = select(0u, 1u, a.y < lo_borrow);
    let hi = hi_inter - b.y;
    let hi_borrow = select(0u, 1u, hi_inter < b.y);
    return vec3<u32>(lo, hi, hi_inter_borrow + hi_borrow);
}

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

// 64×64 → 128 bit unsigned product as vec4<u32>(w0, w1, w2, w3) where
// product = w0 + w1·2^32 + w2·2^64 + w3·2^96.
fn u64_mul_128(a: u64p, b: u64p) -> vec4<u32> {
    let p00 = u32_mul_wide(a.x, b.x);
    let p01 = u32_mul_wide(a.x, b.y);
    let p10 = u32_mul_wide(a.y, b.x);
    let p11 = u32_mul_wide(a.y, b.y);

    var w0 = p00.x;
    var w1 = p00.y;
    var w2 = p11.x;
    var w3 = p11.y;

    let s_w1_a = w1 + p01.x;
    let c_w1_a = select(0u, 1u, s_w1_a < p01.x);
    w1 = s_w1_a;
    let s_w2_a = w2 + p01.y;
    let c_w2_a = select(0u, 1u, s_w2_a < p01.y);
    let s_w2_a2 = s_w2_a + c_w1_a;
    let c_w2_a2 = select(0u, 1u, s_w2_a2 < c_w1_a);
    w2 = s_w2_a2;
    w3 = w3 + c_w2_a + c_w2_a2;

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

fn u64_lt(a: u64p, b: u64p) -> bool {
    if (a.y < b.y) { return true; }
    if (a.y > b.y) { return false; }
    return a.x < b.x;
}

fn u64_gte(a: u64p, b: u64p) -> bool { return !u64_lt(a, b); }

// 64-bit msb (bit 63).
fn u64_msb(a: u64p) -> u32 { return (a.y >> 31u) & 1u; }

// Logical left-shift by one, returning the low 64 bits.
fn u64_shl1(a: u64p) -> u64p {
    let new_hi = (a.y << 1u) | (a.x >> 31u);
    let new_lo = a.x << 1u;
    return u64p(new_lo, new_hi);
}

// ============================================================================
// Modular arithmetic mod q (q is u64 fits in u64p).
// ============================================================================

// mod_neg_q: a is assumed already < q.
fn mod_neg_q(a: u64p, q: u64p) -> u64p {
    if (a.x == 0u && a.y == 0u) { return a; }
    let s = u64_sub(q, a);
    return u64p(s.x, s.y);
}

// Reduce a < 2q once.
fn mod_reduce_once(a: u64p, q: u64p) -> u64p {
    if (u64_gte(a, q)) {
        let s = u64_sub(a, q);
        return u64p(s.x, s.y);
    }
    return a;
}

// Reduce an arbitrary u64 modulo q (shift-and-subtract).
fn mod_reduce_q(a: u64p, q: u64p) -> u64p {
    var r = a;
    while (u64_gte(r, q)) {
        let s = u64_sub(r, q);
        r = u64p(s.x, s.y);
    }
    return r;
}

// Compute -q in u64 (two's-complement). Used to fold overflow back into q.
fn neg_q_u64(q: u64p) -> u64p {
    let neg_lo = 0u - q.x;
    let neg_hi = 0u - q.y - select(0u, 1u, q.x != 0u);
    return u64p(neg_lo, neg_hi);
}

// (a + b) mod q.
fn mod_add_q(a: u64p, b: u64p, q: u64p) -> u64p {
    let a_red = mod_reduce_once(a, q);
    let b_red = mod_reduce_once(b, q);
    let s = u64_add(a_red, b_red);
    if (s.z != 0u) {
        // 64-bit overflow: add -q (i.e. subtract q).
        let nq = neg_q_u64(q);
        let adj = u64_add(u64p(s.x, s.y), nq);
        return mod_reduce_once(u64p(adj.x, adj.y), q);
    }
    return mod_reduce_once(u64p(s.x, s.y), q);
}

// (a - b) mod q.
fn mod_sub_q(a: u64p, b: u64p, q: u64p) -> u64p {
    let a_red = mod_reduce_once(a, q);
    let b_red = mod_reduce_once(b, q);
    if (u64_gte(a_red, b_red)) {
        let s = u64_sub(a_red, b_red);
        return u64p(s.x, s.y);
    }
    let s1 = u64_sub(q, b_red);
    let s2 = u64_add(u64p(s1.x, s1.y), a_red);
    return mod_reduce_once(u64p(s2.x, s2.y), q);
}

// (a * b) mod q via 128-bit schoolbook + shift+sub reduction over 64 bits.
fn mod_mul_q(a: u64p, b: u64p, q: u64p) -> u64p {
    let p = u64_mul_128(a, b);
    let lo  = u64p(p.x, p.y);
    let hi  = u64p(p.z, p.w);

    var r = mod_reduce_q(hi, q);
    var lo_var = lo;
    let nq = neg_q_u64(q);
    for (var bit: u32 = 0u; bit < 64u; bit = bit + 1u) {
        let msb_in = u64_msb(lo_var);
        lo_var = u64_shl1(lo_var);
        let r_top = u64_msb(r);
        var r_next = u64_shl1(r);
        r_next.x = r_next.x | msb_in;
        if (r_top != 0u) {
            let adj = u64_add(r_next, nq);
            r = mod_reduce_once(u64p(adj.x, adj.y), q);
        } else if (u64_gte(r_next, q)) {
            let s = u64_sub(r_next, q);
            r = u64p(s.x, s.y);
        } else {
            r = r_next;
        }
    }
    return r;
}

// ============================================================================
// Param structs — match Metal's struct layout. The single ulong fields
// (q, n_inv) are split into two u32s (lo, hi) so the host can use a packed
// little-endian write.
// ============================================================================

struct DecompTwistParams {
    N: u32,
    rows: u32,
    q_lo: u32,
    q_hi: u32,
}

struct BskTwistParams {
    N: u32,
    rows: u32,
    cols: u32,
    _pad: u32,
    q_lo: u32,
    q_hi: u32,
    _pad2: u32,
    _pad3: u32,
}

struct InnerProductParams {
    N: u32,
    rows: u32,
    cols: u32,
    _pad: u32,
    q_lo: u32,
    q_hi: u32,
    _pad2: u32,
    _pad3: u32,
}

struct UntwistAccumulateParams {
    N: u32,
    cols: u32,
    _pad: u32,
    _pad2: u32,
    q_lo: u32,
    q_hi: u32,
    n_inv_lo: u32,
    n_inv_hi: u32,
}

struct NttBatchParams {
    N: u32,
    M: u32,
    len: u32,
    halfn: u32,
    q_lo: u32,
    q_hi: u32,
    direction: u32,
    _pad: u32,
}

struct BitReverseBatchParams {
    N: u32,
    M: u32,
    _pad0: u32,
    _pad1: u32,
}

struct NttZeroParams {
    total_len_lo: u32,
    total_len_hi: u32,
    _pad0: u32,
    _pad1: u32,
}

// ============================================================================
// Bindings — a "kitchen-sink" group that satisfies every entry point.
// All u64 arrays are flattened to `array<u32>`; an element at logical index
// k lives at u32 indices (2k, 2k+1) — little-endian (lo, hi) split that
// matches the host buffer layout exactly.
// ============================================================================

// Signed int64 decomp (also reused as decomp-NTT input for inner product).
@group(0) @binding(0) var<storage, read>       i_decomp: array<u32>;
// Output buffer (twist output OR accumulator).
@group(0) @binding(1) var<storage, read_write> o_twist_or_acc: array<u32>;
// ψ^j powers (precomputed by the host).
@group(0) @binding(2) var<storage, read>       i_psi_pow: array<u32>;
// Decomp-twist params.
@group(0) @binding(3) var<uniform>             p_decomp_twist: DecompTwistParams;
// BSK-twist params.
@group(0) @binding(4) var<uniform>             p_bsk_twist: BskTwistParams;
// BSK source (rebound to twiddles for the butterfly kernel).
@group(0) @binding(5) var<storage, read>       i_bsk_or_twiddles: array<u32>;
// BSK-NTT (for inner product).
@group(0) @binding(6) var<storage, read>       i_bsk_ntt: array<u32>;
// Result-NTT (inner-product output / untwist-accumulate input).
@group(0) @binding(7) var<storage, read_write> o_result_ntt: array<u32>;
@group(0) @binding(8) var<uniform>             p_ip: InnerProductParams;
@group(0) @binding(9) var<uniform>             p_ntt: NttBatchParams;
@group(0) @binding(10) var<uniform>            p_bitrev: BitReverseBatchParams;
@group(0) @binding(11) var<uniform>            p_untwist: UntwistAccumulateParams;
@group(0) @binding(12) var<storage, read>      i_psi_inv_pow: array<u32>;
@group(0) @binding(13) var<uniform>            p_zero: NttZeroParams;

// ============================================================================
// 1. Decomp twist: out[r][j] = ψ^j · |decomp[r][j]|_q
//    Grid: (N, rows)
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn fhe_decomp_twist(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let r = gid.y;
    let N = p_decomp_twist.N;
    let rows = p_decomp_twist.rows;
    if (j >= N || r >= rows) { return; }
    let q = u64p(p_decomp_twist.q_lo, p_decomp_twist.q_hi);

    // Signed decode of i_decomp[r*N + j].
    let idx = r * N + j;
    let lo = i_decomp[idx * 2u + 0u];
    let hi = i_decomp[idx * 2u + 1u];
    var dq: u64p;
    if (((hi >> 31u) & 1u) == 1u) {
        let neg_lo = 0u - lo;
        let neg_hi = 0u - hi - select(0u, 1u, lo != 0u);
        let abs_red = mod_reduce_q(u64p(neg_lo, neg_hi), q);
        dq = mod_neg_q(abs_red, q);
    } else {
        dq = mod_reduce_q(u64p(lo, hi), q);
    }

    let psi = u64p(i_psi_pow[j * 2u + 0u], i_psi_pow[j * 2u + 1u]);
    let v = mod_mul_q(dq, psi, q);
    o_twist_or_acc[idx * 2u + 0u] = v.x;
    o_twist_or_acc[idx * 2u + 1u] = v.y;
}

// ============================================================================
// 2. BSK twist: out[r][c][j] = ψ^j · bsk[r][c][j] mod q
//    Grid: (N, cols, rows)
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn fhe_bsk_twist(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let c = gid.y;
    let r = gid.z;
    let N = p_bsk_twist.N;
    let cols = p_bsk_twist.cols;
    let rows = p_bsk_twist.rows;
    if (j >= N || c >= cols || r >= rows) { return; }
    let q = u64p(p_bsk_twist.q_lo, p_bsk_twist.q_hi);

    let idx = (r * cols + c) * N + j;
    let v_raw = u64p(i_bsk_or_twiddles[idx * 2u + 0u], i_bsk_or_twiddles[idx * 2u + 1u]);
    let v_red = mod_reduce_q(v_raw, q);
    let psi = u64p(i_psi_pow[j * 2u + 0u], i_psi_pow[j * 2u + 1u]);
    let result = mod_mul_q(v_red, psi, q);
    o_twist_or_acc[idx * 2u + 0u] = result.x;
    o_twist_or_acc[idx * 2u + 1u] = result.y;
}

// ============================================================================
// 3. Batched bit-reverse over [M, N]. In-place swap when j < rev.
//    Grid: (N, M)
// ============================================================================
@compute @workgroup_size(64, 1, 1)
fn fhe_ntt_batch_bit_reverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let m = gid.y;
    let N = p_bitrev.N;
    let M = p_bitrev.M;
    if (j >= N || m >= M) { return; }

    var log_n: u32 = 0u;
    var t = N;
    while (t > 1u) { t = t >> 1u; log_n = log_n + 1u; }

    var i = j;
    var rev: u32 = 0u;
    for (var b: u32 = 0u; b < log_n; b = b + 1u) {
        rev = (rev << 1u) | (i & 1u);
        i = i >> 1u;
    }
    if (j < rev) {
        let idx_j   = m * N + j;
        let idx_rev = m * N + rev;
        let a_lo = o_twist_or_acc[idx_j * 2u + 0u];
        let a_hi = o_twist_or_acc[idx_j * 2u + 1u];
        let b_lo = o_twist_or_acc[idx_rev * 2u + 0u];
        let b_hi = o_twist_or_acc[idx_rev * 2u + 1u];
        o_twist_or_acc[idx_j * 2u + 0u]   = b_lo;
        o_twist_or_acc[idx_j * 2u + 1u]   = b_hi;
        o_twist_or_acc[idx_rev * 2u + 0u] = a_lo;
        o_twist_or_acc[idx_rev * 2u + 1u] = a_hi;
    }
}

// ============================================================================
// 4. Batched NTT butterfly stage. Twiddles read from i_bsk_or_twiddles
//    (host rebinds the BSK slot to the per-stage twiddle slice).
//    Grid: (N/2, M)
// ============================================================================
@compute @workgroup_size(64, 1, 1)
fn fhe_ntt_batch_butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let m   = gid.y;
    let N = p_ntt.N;
    let M = p_ntt.M;
    let len = p_ntt.len;
    let halfn = p_ntt.halfn;
    let total = N >> 1u;
    if (tid >= total || m >= M) { return; }

    let q = u64p(p_ntt.q_lo, p_ntt.q_hi);
    let group  = tid / halfn;
    let inside = tid - group * halfn;
    let i_lo   = group * len + inside;
    let i_hi   = i_lo + halfn;
    let base   = m * N;

    let u_v = u64p(o_twist_or_acc[(base + i_lo) * 2u + 0u],
                   o_twist_or_acc[(base + i_lo) * 2u + 1u]);
    let v_v = u64p(o_twist_or_acc[(base + i_hi) * 2u + 0u],
                   o_twist_or_acc[(base + i_hi) * 2u + 1u]);
    let w   = u64p(i_bsk_or_twiddles[inside * 2u + 0u],
                   i_bsk_or_twiddles[inside * 2u + 1u]);

    var out_lo: u64p;
    var out_hi: u64p;
    if (p_ntt.direction == 0u) {
        let t = mod_mul_q(v_v, w, q);
        out_lo = mod_add_q(u_v, t, q);
        out_hi = mod_sub_q(u_v, t, q);
    } else {
        out_lo = mod_add_q(u_v, v_v, q);
        let diff = mod_sub_q(u_v, v_v, q);
        out_hi = mod_mul_q(diff, w, q);
    }
    o_twist_or_acc[(base + i_lo) * 2u + 0u] = out_lo.x;
    o_twist_or_acc[(base + i_lo) * 2u + 1u] = out_lo.y;
    o_twist_or_acc[(base + i_hi) * 2u + 0u] = out_hi.x;
    o_twist_or_acc[(base + i_hi) * 2u + 1u] = out_hi.y;
}

// ============================================================================
// 5. Pointwise inner product across rows:
//      result_ntt[c][i] = Σ_r decomp_ntt[r][i] · bsk_ntt[r][c][i]  mod q
//    Grid: (N, cols)
//    decomp_ntt source = i_decomp (host writes NTT-domain decomp here).
// ============================================================================
@compute @workgroup_size(64, 1, 1)
fn fhe_ntt_inner_product(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let c = gid.y;
    let N = p_ip.N;
    let rows = p_ip.rows;
    let cols = p_ip.cols;
    if (i >= N || c >= cols) { return; }
    let q = u64p(p_ip.q_lo, p_ip.q_hi);

    var acc = u64p(0u, 0u);
    for (var r: u32 = 0u; r < rows; r = r + 1u) {
        let a_idx = (r * N + i) * 2u;
        let b_idx = ((r * cols + c) * N + i) * 2u;
        let aa = u64p(i_decomp[a_idx + 0u], i_decomp[a_idx + 1u]);
        let bb = u64p(i_bsk_ntt[b_idx + 0u], i_bsk_ntt[b_idx + 1u]);
        acc = mod_add_q(acc, mod_mul_q(aa, bb, q), q);
    }
    let out_idx = (c * N + i) * 2u;
    o_result_ntt[out_idx + 0u] = acc.x;
    o_result_ntt[out_idx + 1u] = acc.y;
}

// ============================================================================
// 6. Inverse twist + scale + accumulate:
//      acc[c][j] += ψ^{-j} · N^{-1} · result_ntt[c][j]   mod q
//    Grid: (N, cols)
// ============================================================================
@compute @workgroup_size(64, 1, 1)
fn fhe_untwist_accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let c = gid.y;
    let N = p_untwist.N;
    let cols = p_untwist.cols;
    if (j >= N || c >= cols) { return; }
    let q     = u64p(p_untwist.q_lo, p_untwist.q_hi);
    let n_inv = u64p(p_untwist.n_inv_lo, p_untwist.n_inv_hi);

    let idx = (c * N + j) * 2u;
    let v_in = u64p(o_result_ntt[idx + 0u], o_result_ntt[idx + 1u]);
    let psi  = u64p(i_psi_inv_pow[j * 2u + 0u], i_psi_inv_pow[j * 2u + 1u]);
    let v1   = mod_mul_q(v_in, n_inv, q);
    let v2   = mod_mul_q(v1, psi, q);
    let prev = u64p(o_twist_or_acc[idx + 0u], o_twist_or_acc[idx + 1u]);
    let out  = mod_add_q(prev, v2, q);
    o_twist_or_acc[idx + 0u] = out.x;
    o_twist_or_acc[idx + 1u] = out.y;
}

// ============================================================================
// 7. Zero an array (used to clear result_ntt between AP steps).
//    Grid: total_len   (one thread per u64 slot)
// ============================================================================
@compute @workgroup_size(64, 1, 1)
fn fhe_ntt_zero(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    let total_lo = p_zero.total_len_lo;
    if (id >= total_lo) { return; }
    o_twist_or_acc[id * 2u + 0u] = 0u;
    o_twist_or_acc[id * 2u + 1u] = 0u;
}
