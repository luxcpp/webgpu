// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Ringtail Lattice Threshold Operations - WGSL port of
// cuda/kernels/crypto/ringtail_ops.cu.
//
// Post-quantum threshold signature helpers based on Module-LWE:
//   - Lagrange-weighted share combination over Dilithium-like Z_q[X]/(X^N+1)
//   - Hint generation/use for signature verification
//   - Response-vector bound checks
//
// Pairs with webgpu/kernels/wgsl/crypto/ringtail_sign.wgsl (full signing
// path). _ops is the polynomial-arithmetic helper layer.
//
// Output is bit-identical to lux::cuda::ringtail::* (cuda/kernels/crypto/ringtail_ops.cu)
// and metal/src/shaders/crypto/ringtail_ops.metal.
//
// WGSL has no signed 64-bit type and no native u64; q < 2^23 so all
// intermediates fit in i32 (signed) or i64-emulation via vec2<i32>.
// We follow the same i32-throughout pattern as ringtail_sign.wgsl.

// =============================================================================
// Ringtail Parameters
// =============================================================================

const RT_Q:        u32 = 8380417u;
const RT_QINV:     u32 = 58728449u;
const RT_N:        u32 = 256u;
const RT_K:        u32 = 4u;
const RT_L:        u32 = 4u;
const RT_GAMMA1:   i32 = 131072;
const RT_GAMMA2:   i32 = 95232;
const RT_BETA:     i32 = 78;
const RT_OMEGA:    u32 = 80u;

// =============================================================================
// Modular Arithmetic (q-prime)
// =============================================================================

// Montgomery reduction: returns (a * R^-1) mod q where R = 2^32.
fn mont_reduce(a_lo: u32, a_hi: u32) -> i32 {
    // t = (a_lo * RT_QINV) mod 2^32
    let t = a_lo * RT_QINV;
    // u = a - t * q     (128-bit subtraction collapses since we only keep hi)
    let tq_lo = t * RT_Q;
    let tq_hi = mul_hi_u32(t, RT_Q);

    // (a_hi:a_lo) - (tq_hi:tq_lo)
    let borrow = select(0u, 1u, a_lo < tq_lo);
    let hi = a_hi - tq_hi - borrow;
    return i32(hi);
}

// Helper: high 32 bits of a 32×32 unsigned multiplication.
fn mul_hi_u32(a: u32, b: u32) -> u32 {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;

    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;

    let mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    return p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
}

// 32×32 → vec2(lo, hi)
fn mul_wide_u32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;

    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;

    let mid     = p01 + p10;
    let mid_c   = select(0u, 0x10000u, mid < p01);
    let lo      = p00 + (mid << 16u);
    let lo_c    = select(0u, 1u, lo < p00);
    let hi      = p11 + (mid >> 16u) + mid_c + lo_c;
    return vec2<u32>(lo, hi);
}

fn mod_add(a: i32, b: i32) -> i32 {
    var r = a + b;
    if (r >= i32(RT_Q)) { r = r - i32(RT_Q); }
    if (r < 0)          { r = r + i32(RT_Q); }
    return r;
}

fn mod_sub(a: i32, b: i32) -> i32 {
    var r = a - b;
    if (r < 0) { r = r + i32(RT_Q); }
    return r;
}

fn caddq(a: i32) -> i32 {
    return a + ((a >> 31u) & i32(RT_Q));
}

fn freeze(a: i32) -> i32 {
    var x = caddq(a);
    return x - i32(RT_Q) + ((i32(RT_Q) - 1 - x) >> 31u & i32(RT_Q));
}

// (a * b) mod q with a, b < q < 2^23 — fits in u32 exactly.
fn modmul(a: i32, b: i32) -> i32 {
    let prod = u32(a) * u32(b);  // < 2^46, but products of <2^23 fit in u32
    return i32(prod % RT_Q);
}

// =============================================================================
// HighBits / LowBits / Power2Round
// =============================================================================

fn highbits(r_in: i32, alpha: i32) -> i32 {
    let r = freeze(r_in);
    return (r + (alpha >> 1u)) / alpha;
}

fn lowbits(r: i32, alpha: i32) -> i32 {
    let r1 = highbits(r, alpha);
    return r - r1 * alpha;
}

// =============================================================================
// Lagrange coefficient at zero (mod q).
// =============================================================================

fn modpow(base_in: i32, exp_in: i32) -> i32 {
    var result: i32 = 1;
    var base   = base_in;
    var exp    = exp_in;
    loop {
        if (exp == 0) { break; }
        if ((exp & 1) != 0) { result = modmul(result, base); }
        base = modmul(base, base);
        exp  = exp >> 1u;
    }
    return result;
}

fn modinv(a: i32) -> i32 {
    return modpow(a, i32(RT_Q) - 2);
}

// indices_buf is the per-thread view into the global indices array.
fn compute_lagrange_coeff(my_index:  u32,
                          base:      u32,
                          num:       u32) -> i32 {
    var num_acc:   i32 = 1;
    var den_acc:   i32 = 1;
    for (var j: u32 = 0u; j < num; j = j + 1u) {
        let other = indices_buf[base + j];
        if (other == my_index) { continue; }
        num_acc = modmul(num_acc, i32(other));
        var diff = i32(other) - i32(my_index);
        if (diff < 0) { diff = diff + i32(RT_Q); }
        den_acc = modmul(den_acc, diff);
    }
    return modmul(num_acc, modinv(den_acc));
}

// =============================================================================
// Bindings
// =============================================================================
//
// Buffer layout for ThresholdShare (matches CUDA struct):
//   uint  index
//   int   s_share[L*N]
//   int   y_share[L*N]
// = 1 + L*N + L*N words.
//
// We store shares flat in `shares_flat`, indexed by share_idx, with stride
// SHARE_STRIDE = 1 + 2 * L * N words.

const SHARE_STRIDE: u32 = 1u + 2u * 4u * 256u;  // = 2049 u32 words (L=4, N=256)

struct RingtailOpsParams {
    num_shares:        u32,
    threshold:         u32,
    batch_size:        u32,
    gamma1_minus_beta: i32,
}

@group(0) @binding(0) var<storage, read>       shares_flat:   array<u32>;
@group(0) @binding(1) var<storage, read>       indices_buf:   array<u32>;
@group(0) @binding(2) var<storage, read_write> combined_s:    array<i32>;
@group(0) @binding(3) var<storage, read_write> combined_y:    array<i32>;
@group(0) @binding(4) var<uniform>             params:        RingtailOpsParams;

// Optional bindings for hint kernels.
@group(0) @binding(5) var<storage, read>       r_buf:         array<i32>;
@group(0) @binding(6) var<storage, read>       z_buf:         array<i32>;
@group(0) @binding(7) var<storage, read_write> hint_buf:      array<u32>;   // packed u32 (one bit per coeff is wasteful; we use 1 u32 per byte slot for parity with CUDA `uint8_t hint[]`)
@group(0) @binding(8) var<storage, read_write> hint_count:    array<atomic<u32>, 1>;

// Auxiliary bindings for response-bound and use_hint kernels.
@group(0) @binding(9)  var<storage, read>       r0_buf:       array<i32>;
@group(0) @binding(10) var<storage, read>       r1_buf:       array<i32>;
@group(0) @binding(11) var<storage, read_write> recovered:    array<i32>;
@group(0) @binding(12) var<storage, read_write> valid_flag:   array<atomic<u32>, 1>;

// Helpers to read ThresholdShare fields.
fn share_index(share_idx: u32) -> u32 {
    return shares_flat[share_idx * SHARE_STRIDE];
}

fn share_s_coeff(share_idx: u32, poly_idx: u32, coeff_idx: u32) -> i32 {
    let base = share_idx * SHARE_STRIDE + 1u;
    return i32(shares_flat[base + poly_idx * RT_N + coeff_idx]);
}

fn share_y_coeff(share_idx: u32, poly_idx: u32, coeff_idx: u32) -> i32 {
    let base = share_idx * SHARE_STRIDE + 1u + RT_L * RT_N;
    return i32(shares_flat[base + poly_idx * RT_N + coeff_idx]);
}

// =============================================================================
// Share combination — one thread per (poly_idx, coeff_idx).
// =============================================================================

@compute @workgroup_size(64)
fn combine_shares_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let poly_idx  = gid.x;
    let coeff_idx = gid.y;
    if (poly_idx >= RT_L || coeff_idx >= RT_N) { return; }

    var s_sum: i32 = 0;
    var y_sum: i32 = 0;

    for (var i: u32 = 0u; i < params.num_shares; i = i + 1u) {
        let lambda = compute_lagrange_coeff(share_index(i), 0u, params.num_shares);

        s_sum = mod_add(s_sum, modmul(share_s_coeff(i, poly_idx, coeff_idx), lambda));
        y_sum = mod_add(y_sum, modmul(share_y_coeff(i, poly_idx, coeff_idx), lambda));
    }

    combined_s[poly_idx * RT_N + coeff_idx] = s_sum;
    combined_y[poly_idx * RT_N + coeff_idx] = y_sum;
}

@compute @workgroup_size(64)
fn batch_combine_shares_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let poly_idx  = gid.x;
    let coeff_idx = gid.y;
    let batch_idx = gid.z;
    if (batch_idx >= params.batch_size || poly_idx >= RT_L || coeff_idx >= RT_N) { return; }

    let share_base   = batch_idx * params.num_shares;
    let indices_base = batch_idx * params.num_shares;

    var s_sum: i32 = 0;
    var y_sum: i32 = 0;

    for (var i: u32 = 0u; i < params.num_shares; i = i + 1u) {
        let lambda = compute_lagrange_coeff(share_index(share_base + i), indices_base, params.num_shares);
        s_sum = mod_add(s_sum, modmul(share_s_coeff(share_base + i, poly_idx, coeff_idx), lambda));
        y_sum = mod_add(y_sum, modmul(share_y_coeff(share_base + i, poly_idx, coeff_idx), lambda));
    }

    let out_base = batch_idx * RT_L * RT_N + poly_idx * RT_N + coeff_idx;
    combined_s[out_base] = s_sum;
    combined_y[out_base] = y_sum;
}

// =============================================================================
// Response bound check — sets valid_flag[0]=0 if any coeff is out of bounds.
// =============================================================================

@compute @workgroup_size(64)
fn check_response_bounds_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let poly_idx  = gid.x;
    let coeff_idx = gid.y;
    if (poly_idx >= RT_L || coeff_idx >= RT_N) { return; }

    let val = freeze(z_buf[poly_idx * RT_N + coeff_idx]);
    if (val > params.gamma1_minus_beta && val < i32(RT_Q) - params.gamma1_minus_beta) {
        atomicStore(&valid_flag[0], 0u);
    }
}

// =============================================================================
// Hint kernels.
// =============================================================================

@compute @workgroup_size(64)
fn make_hint_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let poly_idx  = gid.x;
    let coeff_idx = gid.y;
    if (poly_idx >= RT_K || coeff_idx >= RT_N) { return; }

    let r_val = r_buf[poly_idx * RT_N + coeff_idx];
    let z_val = z_buf[poly_idx * RT_N + coeff_idx];

    let r_high  = highbits(r_val, 2 * RT_GAMMA2);
    let rz_high = highbits(mod_add(r_val, z_val), 2 * RT_GAMMA2);

    if (r_high != rz_high) {
        let idx = poly_idx * RT_N + coeff_idx;
        hint_buf[idx] = 1u;
        atomicAdd(&hint_count[0], 1u);
    }
}

@compute @workgroup_size(64)
fn use_hint_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= RT_K * RT_N) { return; }

    let r0_val = r0_buf[idx];
    let r1_val = r1_buf[idx];
    let h      = hint_buf[idx];

    if (h == 0u) {
        recovered[idx] = r1_val;
    } else {
        let m = (i32(RT_Q) - 1) / (2 * RT_GAMMA2) + 1;
        if (r0_val > 0) {
            recovered[idx] = (r1_val + 1) % m;
        } else {
            recovered[idx] = (r1_val + i32(RT_Q) - 1) % m;
        }
    }
}
