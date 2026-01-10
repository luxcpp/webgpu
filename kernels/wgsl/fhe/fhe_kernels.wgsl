// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Common FHE Utility Kernels for WebGPU
// Collection of frequently used operations in TFHE/CKKS
// Compatible with Metal/Vulkan/D3D12 via Dawn/wgpu

// ============================================================================
// 64-bit Integer Emulation (Shared across all FHE kernels)
// ============================================================================

struct u64 {
    lo: u32,
    hi: u32,
}

fn u64_zero() -> u64 { return u64(0u, 0u); }
fn u64_one() -> u64 { return u64(1u, 0u); }
fn u64_from_u32(v: u32) -> u64 { return u64(v, 0u); }
fn u64_from_i32(v: i32) -> u64 {
    if (v >= 0) { return u64(u32(v), 0u); }
    return u64(u32(-v), 0xFFFFFFFFu);  // Two's complement
}

fn u64_eq(a: u64, b: u64) -> bool { return a.lo == b.lo && a.hi == b.hi; }
fn u64_neq(a: u64, b: u64) -> bool { return a.lo != b.lo || a.hi != b.hi; }

fn u64_lt(a: u64, b: u64) -> bool {
    if (a.hi != b.hi) { return a.hi < b.hi; }
    return a.lo < b.lo;
}

fn u64_le(a: u64, b: u64) -> bool { return !u64_lt(b, a); }
fn u64_gt(a: u64, b: u64) -> bool { return u64_lt(b, a); }
fn u64_ge(a: u64, b: u64) -> bool { return !u64_lt(a, b); }

fn u64_is_zero(a: u64) -> bool { return a.lo == 0u && a.hi == 0u; }

fn u64_add(a: u64, b: u64) -> u64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    let hi = a.hi + b.hi + carry;
    return u64(lo, hi);
}

fn u64_sub(a: u64, b: u64) -> u64 {
    let borrow = select(0u, 1u, a.lo < b.lo);
    let lo = a.lo - b.lo;
    let hi = a.hi - b.hi - borrow;
    return u64(lo, hi);
}

fn u64_neg(a: u64) -> u64 {
    let not_lo = ~a.lo;
    let not_hi = ~a.hi;
    return u64_add(u64(not_lo, not_hi), u64_one());
}

fn u64_and(a: u64, b: u64) -> u64 {
    return u64(a.lo & b.lo, a.hi & b.hi);
}

fn u64_or(a: u64, b: u64) -> u64 {
    return u64(a.lo | b.lo, a.hi | b.hi);
}

fn u64_xor(a: u64, b: u64) -> u64 {
    return u64(a.lo ^ b.lo, a.hi ^ b.hi);
}

fn u64_shl(a: u64, shift: u32) -> u64 {
    if (shift == 0u) { return a; }
    if (shift >= 64u) { return u64_zero(); }
    if (shift >= 32u) {
        return u64(0u, a.lo << (shift - 32u));
    }
    let lo = a.lo << shift;
    let hi = (a.hi << shift) | (a.lo >> (32u - shift));
    return u64(lo, hi);
}

fn u64_shr(a: u64, shift: u32) -> u64 {
    if (shift == 0u) { return a; }
    if (shift >= 64u) { return u64_zero(); }
    if (shift >= 32u) {
        return u64(a.hi >> (shift - 32u), 0u);
    }
    let lo = (a.lo >> shift) | (a.hi << (32u - shift));
    let hi = a.hi >> shift;
    return u64(lo, hi);
}

fn u32_mul_wide(a: u32, b: u32) -> u64 {
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

    let lo = p0 + ((mid & 0xFFFFu) << 16u);
    let lo_carry = select(0u, 1u, lo < p0);

    let hi = p3 + (mid >> 16u) + mid_carry + lo_carry;

    return u64(lo, hi);
}

fn u64_mul(a: u64, b: u64) -> u64 {
    let p0 = u32_mul_wide(a.lo, b.lo);
    let cross1 = a.lo * b.hi;
    let cross2 = a.hi * b.lo;
    let hi = p0.hi + cross1 + cross2;
    return u64(p0.lo, hi);
}

fn u64_mulhi(a: u64, b: u64) -> u64 {
    let p0 = u32_mul_wide(a.lo, b.lo);
    let p1 = u32_mul_wide(a.lo, b.hi);
    let p2 = u32_mul_wide(a.hi, b.lo);
    let p3 = u32_mul_wide(a.hi, b.hi);

    let mid = u64_add(p1, p2);
    let mid2 = u64_add(mid, u64(p0.hi, 0u));

    var result = p3;
    result = u64_add(result, u64(mid2.hi, 0u));

    if (u64_lt(mid, p1)) { result = u64_add(result, u64(0u, 1u)); }
    if (u64_lt(mid2, mid)) { result = u64_add(result, u64(1u, 0u)); }

    return result;
}

// ============================================================================
// Modular Arithmetic
// ============================================================================

fn mod_add(a: u64, b: u64, Q: u64) -> u64 {
    let sum = u64_add(a, b);
    if (u64_ge(sum, Q)) { return u64_sub(sum, Q); }
    return sum;
}

fn mod_sub(a: u64, b: u64, Q: u64) -> u64 {
    if (u64_ge(a, b)) { return u64_sub(a, b); }
    return u64_sub(u64_add(a, Q), b);
}

fn mod_neg(a: u64, Q: u64) -> u64 {
    if (u64_is_zero(a)) { return a; }
    return u64_sub(Q, a);
}

fn mod_double(a: u64, Q: u64) -> u64 {
    return mod_add(a, a, Q);
}

// Barrett reduction: (a * b) mod Q with precomputed mu
fn barrett_mul(a: u64, b: u64, Q: u64, mu: u64) -> u64 {
    let q_approx = u64_mulhi(a, mu);
    let product = u64_mul(a, b);
    let qQ = u64_mul(q_approx, Q);
    var result = u64_sub(product, qQ);
    if (u64_ge(result, Q)) { result = u64_sub(result, Q); }
    return result;
}

// Montgomery multiplication: (a * b * R^{-1}) mod Q
fn montgomery_mul(a: u64, b: u64, Q: u64, Q_inv: u64) -> u64 {
    let t = u64_mul(a, b);
    let m = u64_mul(u64_mul(t, Q_inv), Q);
    var result = u64_sub(u64_add(t, m), m);
    if (u64_ge(result, Q)) { result = u64_sub(result, Q); }
    return result;
}

// ============================================================================
// Common FHE Parameters
// ============================================================================

struct FHEParams {
    // Polynomial degree
    N: u32,
    log_N: u32,

    // Modulus (64-bit split)
    Q_lo: u32,
    Q_hi: u32,

    // Barrett constant
    mu_lo: u32,
    mu_hi: u32,

    // Montgomery constant
    Q_inv_lo: u32,
    Q_inv_hi: u32,

    // Decomposition
    l: u32,           // Number of levels
    base_log: u32,    // Log of decomposition base

    // Batch
    batch_size: u32,
    _pad: u32,
}

fn params_Q(p: FHEParams) -> u64 { return u64(p.Q_lo, p.Q_hi); }
fn params_mu(p: FHEParams) -> u64 { return u64(p.mu_lo, p.mu_hi); }
fn params_Q_inv(p: FHEParams) -> u64 { return u64(p.Q_inv_lo, p.Q_inv_hi); }

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read_write> data_a: array<u32>;
@group(0) @binding(1) var<storage, read_write> data_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> data_c: array<u32>;
@group(0) @binding(3) var<uniform> params: FHEParams;

// ============================================================================
// Helper Functions
// ============================================================================

fn read_u64(arr: ptr<storage, array<u32>, read_write>, idx: u32) -> u64 {
    return u64((*arr)[idx * 2u], (*arr)[idx * 2u + 1u]);
}

fn write_u64(arr: ptr<storage, array<u32>, read_write>, idx: u32, val: u64) {
    (*arr)[idx * 2u] = val.lo;
    (*arr)[idx * 2u + 1u] = val.hi;
}

// ============================================================================
// Polynomial Operations
// ============================================================================

// Element-wise polynomial addition: c = a + b mod Q
@compute @workgroup_size(256)
fn poly_add(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N = params.N;
    let Q = params_Q(params);
    let idx = gid.x;

    if (idx >= N * params.batch_size) { return; }

    let a = read_u64(&data_a, idx);
    let b = read_u64(&data_b, idx);
    let c = mod_add(a, b, Q);
    write_u64(&data_c, idx, c);
}

// Element-wise polynomial subtraction: c = a - b mod Q
@compute @workgroup_size(256)
fn poly_sub(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N = params.N;
    let Q = params_Q(params);
    let idx = gid.x;

    if (idx >= N * params.batch_size) { return; }

    let a = read_u64(&data_a, idx);
    let b = read_u64(&data_b, idx);
    let c = mod_sub(a, b, Q);
    write_u64(&data_c, idx, c);
}

// Element-wise polynomial negation: b = -a mod Q
@compute @workgroup_size(256)
fn poly_neg(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N = params.N;
    let Q = params_Q(params);
    let idx = gid.x;

    if (idx >= N * params.batch_size) { return; }

    let a = read_u64(&data_a, idx);
    let b = mod_neg(a, Q);
    write_u64(&data_b, idx, b);
}

// Element-wise polynomial multiplication (NTT domain): c = a * b mod Q
@compute @workgroup_size(256)
fn poly_mul_pointwise(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N = params.N;
    let Q = params_Q(params);
    let mu = params_mu(params);
    let idx = gid.x;

    if (idx >= N * params.batch_size) { return; }

    let a = read_u64(&data_a, idx);
    let b = read_u64(&data_b, idx);
    let c = barrett_mul(a, b, Q, mu);
    write_u64(&data_c, idx, c);
}

// Polynomial scaling: b = a * scalar mod Q
@group(1) @binding(0) var<uniform> scalar: u64;

@compute @workgroup_size(256)
fn poly_scale(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N = params.N;
    let Q = params_Q(params);
    let mu = params_mu(params);
    let idx = gid.x;

    if (idx >= N * params.batch_size) { return; }

    let a = read_u64(&data_a, idx);
    let b = barrett_mul(a, scalar, Q, mu);
    write_u64(&data_b, idx, b);
}

// ============================================================================
// Polynomial Copy and Zero
// ============================================================================

// Copy polynomial: b = a
@compute @workgroup_size(256)
fn poly_copy(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N = params.N;
    let idx = gid.x;

    if (idx >= N * params.batch_size) { return; }

    let a = read_u64(&data_a, idx);
    write_u64(&data_b, idx, a);
}

// Zero polynomial: a = 0
@compute @workgroup_size(256)
fn poly_zero(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N = params.N;
    let idx = gid.x;

    if (idx >= N * params.batch_size) { return; }

    write_u64(&data_a, idx, u64_zero());
}

// Set polynomial to constant: a = c for all coefficients
@compute @workgroup_size(256)
fn poly_set_constant(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N = params.N;
    let idx = gid.x;

    if (idx >= N * params.batch_size) { return; }

    write_u64(&data_a, idx, scalar);
}

// ============================================================================
// Gadget Decomposition
// ============================================================================

// Signed decomposition digit extraction
fn signed_decompose_digit(val: u64, level: u32, base_log: u32) -> i32 {
    let Bg = 1u << base_log;
    let half_Bg = Bg >> 1u;
    let mask = Bg - 1u;

    let shift = 64u - (level + 1u) * base_log;

    var digit: u32;
    if (shift >= 32u) {
        digit = (val.hi >> (shift - 32u)) & mask;
    } else if (shift == 0u) {
        digit = val.lo & mask;
    } else {
        let lo_part = val.lo >> shift;
        let hi_part = val.hi << (32u - shift);
        digit = (lo_part | hi_part) & mask;
    }

    if (digit >= half_Bg) {
        return i32(digit) - i32(Bg);
    }
    return i32(digit);
}

// Full gadget decomposition output
@group(2) @binding(0) var<storage, read_write> decomposed: array<i32>;

@compute @workgroup_size(256)
fn gadget_decompose(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N = params.N;
    let l = params.l;
    let base_log = params.base_log;
    let idx = gid.x;

    if (idx >= N * params.batch_size) { return; }

    let val = read_u64(&data_a, idx);

    for (var level = 0u; level < l; level++) {
        let digit = signed_decompose_digit(val, level, base_log);
        let out_idx = idx * l + level;
        decomposed[out_idx] = digit;
    }
}

// ============================================================================
// Noise Sampling (Discrete Gaussian)
// ============================================================================

// PRNG state for noise generation
var<workgroup> prng_state: array<u32, 256>;

// Xorshift128+ PRNG
fn xorshift128plus(idx: u32) -> u32 {
    var s0 = prng_state[idx * 2u];
    var s1 = prng_state[idx * 2u + 1u];
    let result = s0 + s1;

    s1 ^= s0;
    prng_state[idx * 2u] = ((s0 << 23u) | (s0 >> 9u)) ^ s1 ^ (s1 << 18u);
    prng_state[idx * 2u + 1u] = (s1 << 5u) | (s1 >> 27u);

    return result;
}

// Box-Muller approximation for Gaussian
fn sample_gaussian_approx(tid: u32, sigma: f32) -> i32 {
    // Sum of uniforms approximation
    var sum = 0;
    for (var i = 0u; i < 12u; i++) {
        sum += i32(xorshift128plus(tid) & 0xFFFFu);
    }
    // Center around 0 (subtract mean of 12 * 0x8000)
    sum -= 393216;
    // Scale by sigma
    let scaled = f32(sum) * sigma / 65536.0;
    return i32(scaled);
}

@group(3) @binding(0) var<storage, read> prng_seeds: array<u32>;
@group(3) @binding(1) var<uniform> noise_sigma: f32;

@compute @workgroup_size(256)
fn sample_noise(
    @builtin(local_invocation_index) tid: u32,
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N = params.N;
    let Q = params_Q(params);
    let idx = gid.x;

    if (idx >= N * params.batch_size) { return; }

    // Initialize PRNG
    prng_state[tid * 2u] = prng_seeds[idx * 2u];
    prng_state[tid * 2u + 1u] = prng_seeds[idx * 2u + 1u];
    workgroupBarrier();

    // Sample Gaussian noise
    let noise = sample_gaussian_approx(tid, noise_sigma);

    // Convert to modular representation
    var noise_mod: u64;
    if (noise >= 0) {
        noise_mod = u64_from_u32(u32(noise));
    } else {
        noise_mod = u64_sub(Q, u64_from_u32(u32(-noise)));
    }

    write_u64(&data_a, idx, noise_mod);
}

// ============================================================================
// LWE Operations
// ============================================================================

struct LWEParams {
    n: u32,           // LWE dimension
    Q_lo: u32,
    Q_hi: u32,
    batch_size: u32,
}

@group(4) @binding(0) var<storage, read> lwe_a: array<u32>;      // LWE masks [batch][n]
@group(4) @binding(1) var<storage, read> lwe_s: array<u32>;      // LWE secret key [n]
@group(4) @binding(2) var<storage, read_write> lwe_b: array<u32>; // LWE body [batch]
@group(4) @binding(3) var<uniform> lwe_params: LWEParams;

// LWE encryption: b = <a, s> + m + e mod Q
@compute @workgroup_size(256)
fn lwe_inner_product(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let n = lwe_params.n;
    let Q = u64(lwe_params.Q_lo, lwe_params.Q_hi);
    let batch_idx = wgid.x;
    let tpg = 256u;

    if (batch_idx >= lwe_params.batch_size) { return; }

    // Parallel reduction for inner product
    var partial_sum = u64_zero();
    let base = batch_idx * n;

    for (var i = tid; i < n; i += tpg) {
        let a = u64(lwe_a[(base + i) * 2u], lwe_a[(base + i) * 2u + 1u]);
        let s = u64(lwe_s[i * 2u], lwe_s[i * 2u + 1u]);
        let product = u64_mul(a, s);
        partial_sum = mod_add(partial_sum, product, Q);
    }

    // Store partial sum for reduction (simplified - full version needs shared memory reduction)
    if (tid == 0u) {
        lwe_b[batch_idx * 2u] = partial_sum.lo;
        lwe_b[batch_idx * 2u + 1u] = partial_sum.hi;
    }
}

// ============================================================================
// Rounding and Modular Switch
// ============================================================================

// Modulus switching: round(a * Q_new / Q_old)
struct ModSwitchParams {
    Q_old_lo: u32,
    Q_old_hi: u32,
    Q_new_lo: u32,
    Q_new_hi: u32,
    N: u32,
    batch_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(5) @binding(0) var<uniform> ms_params: ModSwitchParams;

@compute @workgroup_size(256)
fn mod_switch(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N = ms_params.N;
    let Q_old = u64(ms_params.Q_old_lo, ms_params.Q_old_hi);
    let Q_new = u64(ms_params.Q_new_lo, ms_params.Q_new_hi);
    let idx = gid.x;

    if (idx >= N * ms_params.batch_size) { return; }

    let a = read_u64(&data_a, idx);

    // Compute a * Q_new / Q_old with rounding
    // Simplified: assumes Q_new < Q_old and both fit in 64 bits
    let scaled = u64_mul(a, Q_new);
    // Division approximation (would need proper implementation)
    let result = u64(scaled.lo / Q_old.lo, 0u);

    write_u64(&data_b, idx, result);
}

// ============================================================================
// Bit Extraction
// ============================================================================

// Extract specific bit from encrypted value
@group(6) @binding(0) var<uniform> bit_position: u32;

@compute @workgroup_size(256)
fn extract_bit(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N = params.N;
    let Q = params_Q(params);
    let idx = gid.x;

    if (idx >= N * params.batch_size) { return; }

    let a = read_u64(&data_a, idx);

    // Extract bit at position
    let bit = (u64_shr(a, bit_position).lo) & 1u;

    // Scale to Q/2 for bit representation
    let half_Q = u64_shr(Q, 1u);
    let result = select(u64_zero(), half_Q, bit == 1u);

    write_u64(&data_b, idx, result);
}

// ============================================================================
// GLWE Operations
// ============================================================================

// Zero GLWE ciphertext: (0, ..., 0)
@compute @workgroup_size(256)
fn glwe_zero(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = wgid.y;  // GLWE dimension index
    let idx = gid.x;

    if (idx >= N) { return; }

    let out_idx = k * N + idx;
    write_u64(&data_a, out_idx, u64_zero());
}

// GLWE addition: c = a + b
@compute @workgroup_size(256)
fn glwe_add(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let Q = params_Q(params);
    let k = wgid.y;
    let idx = gid.x;

    if (idx >= N) { return; }

    let offset = k * N + idx;
    let a = read_u64(&data_a, offset);
    let b = read_u64(&data_b, offset);
    let c = mod_add(a, b, Q);
    write_u64(&data_c, offset, c);
}

// ============================================================================
// Utility: Check Equality
// ============================================================================

@group(7) @binding(0) var<storage, read_write> equality_result: atomic<u32>;

@compute @workgroup_size(256)
fn poly_equal(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N = params.N;
    let idx = gid.x;

    if (idx >= N * params.batch_size) { return; }

    let a = read_u64(&data_a, idx);
    let b = read_u64(&data_b, idx);

    if (!u64_eq(a, b)) {
        atomicStore(&equality_result, 0u);
    }
}

// Initialize equality check (call before poly_equal)
@compute @workgroup_size(1)
fn init_equality() {
    atomicStore(&equality_result, 1u);
}
