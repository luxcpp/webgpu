// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// twiddle_cache.wgsl - Cached twiddle factors for NTT
//
// Implements hierarchical twiddle factor caching for NTT operations:
// - L1 cache in workgroup shared memory
// - L2 cache in storage buffer
// - On-demand generation with caching
// - Precomputation kernels for common sizes

// ============================================================================
// 64-bit integer emulation
// ============================================================================

struct u64 {
    lo: u32,
    hi: u32,
}

fn u64_zero() -> u64 {
    return u64(0u, 0u);
}

fn u64_from_u32(x: u32) -> u64 {
    return u64(x, 0u);
}

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

fn u64_mul(a: u64, b: u64) -> u64 {
    let a_lo = a.lo & 0xFFFFu;
    let a_hi = a.lo >> 16u;
    let b_lo = b.lo & 0xFFFFu;
    let b_hi = b.lo >> 16u;

    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;

    let mid = p1 + p2;
    let lo = p0 + ((mid & 0xFFFFu) << 16u);
    let carry = select(0u, 1u, lo < p0);
    let hi = p3 + (mid >> 16u) + carry + a.lo * b.hi + a.hi * b.lo;

    return u64(lo, hi);
}

fn u64_ge(a: u64, b: u64) -> bool {
    if (a.hi != b.hi) {
        return a.hi > b.hi;
    }
    return a.lo >= b.lo;
}

// ============================================================================
// Modular arithmetic
// ============================================================================

fn mod_add(a: u64, b: u64, Q: u64) -> u64 {
    var result = u64_add(a, b);
    if (u64_ge(result, Q)) {
        result = u64_sub(result, Q);
    }
    return result;
}

fn mod_sub(a: u64, b: u64, Q: u64) -> u64 {
    if (u64_ge(a, b)) {
        return u64_sub(a, b);
    }
    return u64_sub(u64_add(a, Q), b);
}

fn u64_mulhi(a: u64, b: u64) -> u64 {
    let a0 = a.lo & 0xFFFFu;
    let a1 = a.lo >> 16u;
    let a2 = a.hi & 0xFFFFu;
    let a3 = a.hi >> 16u;
    let b0 = b.lo & 0xFFFFu;
    let b1 = b.lo >> 16u;
    let b2 = b.hi & 0xFFFFu;
    let b3 = b.hi >> 16u;

    var acc = u64_zero();
    acc = u64_add(acc, u64_from_u32(a2 * b2));
    acc = u64_add(acc, u64_from_u32(a3 * b1));
    acc = u64_add(acc, u64_from_u32(a1 * b3));
    let hi_lo = acc.lo + a3 * b2 + a2 * b3;
    let hi_hi = acc.hi + a3 * b3;

    return u64(hi_lo, hi_hi);
}

fn barrett_mul(a: u64, b: u64, Q: u64, mu: u64) -> u64 {
    let product = u64_mul(a, b);
    let q_approx = u64_mulhi(product, mu);
    let qQ = u64_mul(q_approx, Q);
    var result = u64_sub(product, qQ);
    if (u64_ge(result, Q)) {
        result = u64_sub(result, Q);
    }
    if (u64_ge(result, Q)) {
        result = u64_sub(result, Q);
    }
    return result;
}

// ============================================================================
// Twiddle factor computation
// ============================================================================

// Compute w^k mod Q using repeated squaring
fn mod_pow(base: u64, exp: u32, Q: u64, mu: u64) -> u64 {
    var result = u64_from_u32(1u);
    var b = base;
    var e = exp;

    while (e > 0u) {
        if ((e & 1u) != 0u) {
            result = barrett_mul(result, b, Q, mu);
        }
        b = barrett_mul(b, b, Q, mu);
        e = e >> 1u;
    }

    return result;
}

// Compute twiddle factor: w^(i * (N / (2^stage)))
fn compute_twiddle(i: u32, stage: u32, N: u32, omega: u64, Q: u64, mu: u64) -> u64 {
    let step = N >> stage;
    let exp = i * step;
    return mod_pow(omega, exp, Q, mu);
}

// Compute inverse twiddle factor
fn compute_inv_twiddle(i: u32, stage: u32, N: u32, omega_inv: u64, Q: u64, mu: u64) -> u64 {
    let step = N >> stage;
    let exp = i * step;
    return mod_pow(omega_inv, exp, Q, mu);
}

// ============================================================================
// Shared memory cache
// ============================================================================

const CACHE_SIZE: u32 = 512u;  // Fits in workgroup memory

var<workgroup> twiddle_cache_fwd: array<u64, 512>;
var<workgroup> twiddle_cache_inv: array<u64, 512>;
var<workgroup> cache_valid: array<u32, 16>;  // Bitmask for cached stages

// ============================================================================
// Storage buffers
// ============================================================================

struct TwiddleParams {
    N: u32,
    log_N: u32,
    num_primes: u32,
    cache_mode: u32,  // 0=compute, 1=load, 2=hybrid
}

@group(0) @binding(0) var<uniform> params: TwiddleParams;

// Precomputed twiddle factors (L2 cache)
@group(0) @binding(1) var<storage, read> twiddles_fwd: array<u64>;
@group(0) @binding(2) var<storage, read> twiddles_inv: array<u64>;

// Output for precomputation
@group(0) @binding(3) var<storage, read_write> twiddles_out: array<u64>;

// Prime moduli and roots
@group(0) @binding(4) var<storage, read> primes: array<u64>;
@group(0) @binding(5) var<storage, read> roots: array<u64>;      // Primitive N-th roots
@group(0) @binding(6) var<storage, read> roots_inv: array<u64>;  // Inverse roots
@group(0) @binding(7) var<storage, read> barrett_mu: array<u64>; // Barrett parameters

// ============================================================================
// Kernel: Precompute all twiddle factors
// ============================================================================

@compute @workgroup_size(256)
fn precompute_twiddles(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let idx = gid.x;
    let N = params.N;
    let half_N = N >> 1u;

    if (idx >= half_N * params.num_primes) {
        return;
    }

    let prime_idx = idx / half_N;
    let twiddle_idx = idx % half_N;

    let Q = primes[prime_idx];
    let omega = roots[prime_idx];
    let omega_inv = roots_inv[prime_idx];
    let mu = barrett_mu[prime_idx];

    // Forward twiddle: omega^(bit_reverse(i))
    var rev_i = 0u;
    var temp = twiddle_idx;
    let log_half_N = params.log_N - 1u;
    for (var b = 0u; b < log_half_N; b++) {
        rev_i = (rev_i << 1u) | (temp & 1u);
        temp = temp >> 1u;
    }

    let fwd_twiddle = mod_pow(omega, rev_i, Q, mu);
    let inv_twiddle = mod_pow(omega_inv, rev_i, Q, mu);

    let out_idx = prime_idx * half_N + twiddle_idx;
    twiddles_out[out_idx * 2u] = fwd_twiddle;
    twiddles_out[out_idx * 2u + 1u] = inv_twiddle;
}

// ============================================================================
// Kernel: Precompute twiddles for specific stages (Cooley-Tukey layout)
// ============================================================================

@compute @workgroup_size(256)
fn precompute_stage_twiddles(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let N = params.N;
    let log_N = params.log_N;

    // Total twiddles = N/2 + N/4 + N/8 + ... + 1 = N - 1
    if (idx >= (N - 1u) * params.num_primes) {
        return;
    }

    let prime_idx = idx / (N - 1u);
    let local_idx = idx % (N - 1u);

    let Q = primes[prime_idx];
    let omega = roots[prime_idx];
    let mu = barrett_mu[prime_idx];

    // Find which stage this index belongs to
    var stage = 0u;
    var offset = 0u;
    var stage_size = N >> 1u;

    while (local_idx >= offset + stage_size && stage < log_N) {
        offset += stage_size;
        stage += 1u;
        stage_size = stage_size >> 1u;
    }

    let stage_idx = local_idx - offset;
    let step = 1u << stage;
    let exp = stage_idx * step;

    let twiddle = mod_pow(omega, exp, Q, mu);
    twiddles_out[idx] = twiddle;
}

// ============================================================================
// Kernel: Load twiddles into shared memory cache
// ============================================================================

@compute @workgroup_size(256)
fn load_twiddle_cache(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_id = lid.x;
    let prime_idx = wid.y;
    let N = params.N;
    let half_N = N >> 1u;

    // Load subset of twiddles into shared memory
    let cache_entries = min(CACHE_SIZE, half_N);

    if (local_id < cache_entries) {
        let src_idx = prime_idx * half_N + local_id;
        twiddle_cache_fwd[local_id] = twiddles_fwd[src_idx];
        twiddle_cache_inv[local_id] = twiddles_inv[src_idx];
    }

    // Mark cache as valid
    if (local_id < 16u) {
        cache_valid[local_id] = select(0u, 0xFFFFFFFFu, local_id * 32u < cache_entries);
    }

    workgroupBarrier();
}

// ============================================================================
// Kernel: Generate twiddles on-demand with caching
// ============================================================================

struct OnDemandParams {
    stage: u32,
    butterfly_idx: u32,
    prime_idx: u32,
    _pad: u32,
}

@group(1) @binding(0) var<uniform> od_params: OnDemandParams;
@group(1) @binding(1) var<storage, read_write> twiddle_out: array<u64>;

@compute @workgroup_size(256)
fn generate_twiddles_on_demand(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let idx = gid.x;
    let stage = od_params.stage;
    let prime_idx = od_params.prime_idx;
    let N = params.N;

    let num_butterflies = N >> (stage + 1u);
    if (idx >= num_butterflies) {
        return;
    }

    // Check L1 cache first
    let cache_idx = idx & (CACHE_SIZE - 1u);
    let cache_slot = cache_idx >> 5u;
    let cache_bit = cache_idx & 31u;

    var twiddle: u64;

    if ((cache_valid[cache_slot] & (1u << cache_bit)) != 0u) {
        // Cache hit
        twiddle = twiddle_cache_fwd[cache_idx];
    } else {
        // Cache miss - compute or load from L2
        let Q = primes[prime_idx];
        let omega = roots[prime_idx];
        let mu = barrett_mu[prime_idx];

        if (params.cache_mode == 1u) {
            // Load from storage
            let storage_idx = prime_idx * (N >> 1u) + idx;
            twiddle = twiddles_fwd[storage_idx];
        } else {
            // Compute on-demand
            twiddle = compute_twiddle(idx, stage, N, omega, Q, mu);
        }

        // Update cache (best effort, no synchronization needed for read-only data)
        if (cache_idx < CACHE_SIZE) {
            twiddle_cache_fwd[cache_idx] = twiddle;
        }
    }

    twiddle_out[idx] = twiddle;
}

// ============================================================================
// Kernel: Compute twiddles for four-step NTT
// ============================================================================

struct FourStepParams {
    N1: u32,  // First dimension
    N2: u32,  // Second dimension
    prime_idx: u32,
    is_inverse: u32,
}

@group(1) @binding(0) var<uniform> fs_params: FourStepParams;
@group(1) @binding(1) var<storage, read_write> row_twiddles: array<u64>;
@group(1) @binding(2) var<storage, read_write> col_twiddles: array<u64>;
@group(1) @binding(3) var<storage, read_write> inter_twiddles: array<u64>;  // omega^(i*j)

@compute @workgroup_size(256)
fn compute_four_step_twiddles(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let N1 = fs_params.N1;
    let N2 = fs_params.N2;
    let N = N1 * N2;
    let prime_idx = fs_params.prime_idx;

    let Q = primes[prime_idx];
    let mu = barrett_mu[prime_idx];
    let omega = select(roots[prime_idx], roots_inv[prime_idx], fs_params.is_inverse != 0u);

    // Compute row twiddles (N2 values)
    if (idx < N2) {
        let omega_N2 = mod_pow(omega, N1, Q, mu);  // omega^N1 is N2-th root
        row_twiddles[idx] = mod_pow(omega_N2, idx, Q, mu);
    }

    // Compute column twiddles (N1 values)
    if (idx < N1) {
        let omega_N1 = mod_pow(omega, N2, Q, mu);  // omega^N2 is N1-th root
        col_twiddles[idx] = mod_pow(omega_N1, idx, Q, mu);
    }

    // Compute inter-stage twiddles (N1 * N2 values for full matrix)
    if (idx < N) {
        let i = idx / N2;
        let j = idx % N2;
        inter_twiddles[idx] = mod_pow(omega, i * j, Q, mu);
    }
}

// ============================================================================
// Kernel: Precompute NTT twiddles in bit-reversed order
// ============================================================================

@compute @workgroup_size(256)
fn precompute_bitrev_twiddles(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let N = params.N;
    let log_N = params.log_N;

    if (idx >= N * params.num_primes) {
        return;
    }

    let prime_idx = idx / N;
    let local_idx = idx % N;

    let Q = primes[prime_idx];
    let omega = roots[prime_idx];
    let mu = barrett_mu[prime_idx];

    // Bit-reverse the index
    var rev_idx = 0u;
    var temp = local_idx;
    for (var b = 0u; b < log_N; b++) {
        rev_idx = (rev_idx << 1u) | (temp & 1u);
        temp = temp >> 1u;
    }

    let twiddle = mod_pow(omega, rev_idx, Q, mu);
    twiddles_out[idx] = twiddle;
}

// ============================================================================
// Kernel: Generate twiddle lookup table for radix-4 NTT
// ============================================================================

@compute @workgroup_size(256)
fn generate_radix4_twiddles(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let N = params.N;
    let log_N = params.log_N;

    // Radix-4 needs 3 twiddles per butterfly
    let num_stages = log_N >> 1u;  // log4(N) stages
    let twiddles_per_stage = N >> 2u;

    if (idx >= twiddles_per_stage * 3u * num_stages * params.num_primes) {
        return;
    }

    let prime_idx = idx / (twiddles_per_stage * 3u * num_stages);
    let local_idx = idx % (twiddles_per_stage * 3u * num_stages);

    let stage = local_idx / (twiddles_per_stage * 3u);
    let within_stage = local_idx % (twiddles_per_stage * 3u);
    let twiddle_type = within_stage / twiddles_per_stage;  // 0, 1, or 2
    let butterfly_idx = within_stage % twiddles_per_stage;

    let Q = primes[prime_idx];
    let omega = roots[prime_idx];
    let mu = barrett_mu[prime_idx];

    let step = 1u << (stage * 2u);
    let base_exp = butterfly_idx * step;

    // Radix-4 butterfly uses w^k, w^(2k), w^(3k)
    let exp = base_exp * (twiddle_type + 1u);

    let twiddle = mod_pow(omega, exp, Q, mu);
    twiddles_out[idx] = twiddle;
}

// ============================================================================
// Kernel: Validate cached twiddles against computed values
// ============================================================================

@group(1) @binding(0) var<storage, read_write> validation_result: array<u32>;

@compute @workgroup_size(256)
fn validate_twiddle_cache(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let N = params.N;
    let half_N = N >> 1u;

    if (idx >= half_N * params.num_primes) {
        return;
    }

    let prime_idx = idx / half_N;
    let twiddle_idx = idx % half_N;

    let Q = primes[prime_idx];
    let omega = roots[prime_idx];
    let mu = barrett_mu[prime_idx];

    // Compute expected twiddle
    let expected = mod_pow(omega, twiddle_idx, Q, mu);

    // Load cached twiddle
    let cached = twiddles_fwd[idx];

    // Check if they match
    let match_lo = expected.lo == cached.lo;
    let match_hi = expected.hi == cached.hi;

    if (!match_lo || !match_hi) {
        atomicAdd(&validation_result[0], 1u);  // Count mismatches
    }
}
