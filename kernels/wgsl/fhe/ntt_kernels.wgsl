// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// NTT (Number Theoretic Transform) WebGPU Kernels for FHE
// Ported from Metal implementation with 64-bit integer emulation
//
// WebGPU Constraints:
// - No native u64, emulated with struct { lo: u32, hi: u32 }
// - Max workgroup size: 256 threads
// - Max shared memory: 16KB (2048 u64 values)
// - Max N for fused kernel: 2048

// ============================================================================
// 64-bit Integer Emulation
// ============================================================================

struct u64 {
    lo: u32,
    hi: u32,
}

fn u64_from_u32(v: u32) -> u64 {
    return u64(v, 0u);
}

fn u64_zero() -> u64 {
    return u64(0u, 0u);
}

fn u64_eq(a: u64, b: u64) -> bool {
    return a.lo == b.lo && a.hi == b.hi;
}

fn u64_lt(a: u64, b: u64) -> bool {
    if (a.hi != b.hi) {
        return a.hi < b.hi;
    }
    return a.lo < b.lo;
}

fn u64_ge(a: u64, b: u64) -> bool {
    return !u64_lt(a, b);
}

// Addition with carry
fn u64_add(a: u64, b: u64) -> u64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    let hi = a.hi + b.hi + carry;
    return u64(lo, hi);
}

// Subtraction with borrow
fn u64_sub(a: u64, b: u64) -> u64 {
    let borrow = select(0u, 1u, a.lo < b.lo);
    let lo = a.lo - b.lo;
    let hi = a.hi - b.hi - borrow;
    return u64(lo, hi);
}

// Multiply two u32 -> u64
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

    let hi = p3 + (mid >> 16u) + mid_carry + lo_carry + ((mid & 0xFFFFu) >> 16u);

    return u64(lo, hi);
}

// Full 64x64 -> 128 bit multiply, return high 64 bits
fn u64_mulhi(a: u64, b: u64) -> u64 {
    // Decompose into 32-bit parts
    // a = a_hi * 2^32 + a_lo
    // b = b_hi * 2^32 + b_lo
    // a*b = a_hi*b_hi*2^64 + (a_hi*b_lo + a_lo*b_hi)*2^32 + a_lo*b_lo

    let p0 = u32_mul_wide(a.lo, b.lo);  // a_lo * b_lo
    let p1 = u32_mul_wide(a.lo, b.hi);  // a_lo * b_hi
    let p2 = u32_mul_wide(a.hi, b.lo);  // a_hi * b_lo
    let p3 = u32_mul_wide(a.hi, b.hi);  // a_hi * b_hi

    // Sum the middle terms (bits 32-95)
    let mid = u64_add(p1, p2);

    // Add carry from p0.hi to mid
    let mid2 = u64_add(mid, u64(p0.hi, 0u));

    // High 64 bits = p3 + mid2.hi + (mid2.lo >> 0, but shifted)
    var result = p3;
    result = u64_add(result, u64(mid2.hi, 0u));

    // Handle the overflow from mid into high bits
    if (u64_lt(mid, p1)) {
        result = u64_add(result, u64(0u, 1u));
    }
    if (u64_lt(mid2, mid)) {
        result = u64_add(result, u64(1u, 0u));
    }

    return result;
}

// 64x64 -> low 64 bits
fn u64_mul(a: u64, b: u64) -> u64 {
    let p0 = u32_mul_wide(a.lo, b.lo);
    let cross1 = a.lo * b.hi;
    let cross2 = a.hi * b.lo;
    let hi = p0.hi + cross1 + cross2;
    return u64(p0.lo, hi);
}

// ============================================================================
// NTT Parameters (Uniform Buffer)
// ============================================================================

struct NTTParams {
    Q_lo: u32,
    Q_hi: u32,
    mu_lo: u32,
    mu_hi: u32,
    N_inv_lo: u32,
    N_inv_hi: u32,
    N_inv_precon_lo: u32,
    N_inv_precon_hi: u32,
    N: u32,
    log_N: u32,
    stage: u32,
    batch: u32,
}

fn params_Q(params: NTTParams) -> u64 {
    return u64(params.Q_lo, params.Q_hi);
}

fn params_mu(params: NTTParams) -> u64 {
    return u64(params.mu_lo, params.mu_hi);
}

fn params_N_inv(params: NTTParams) -> u64 {
    return u64(params.N_inv_lo, params.N_inv_hi);
}

fn params_N_inv_precon(params: NTTParams) -> u64 {
    return u64(params.N_inv_precon_lo, params.N_inv_precon_hi);
}

// ============================================================================
// Modular Arithmetic
// ============================================================================

fn mod_add(a: u64, b: u64, Q: u64) -> u64 {
    let sum = u64_add(a, b);
    if (u64_ge(sum, Q)) {
        return u64_sub(sum, Q);
    }
    return sum;
}

fn mod_sub(a: u64, b: u64, Q: u64) -> u64 {
    if (u64_ge(a, b)) {
        return u64_sub(a, b);
    }
    return u64_sub(u64_add(a, Q), b);
}

// Barrett reduction: (a * omega) mod Q
// precon = floor(2^64 * omega / Q)
fn barrett_mul(a: u64, omega: u64, Q: u64, precon: u64) -> u64 {
    // q_approx = high 64 bits of (a * precon)
    let q_approx = u64_mulhi(a, precon);

    // product = a * omega (low 64 bits)
    let product = u64_mul(a, omega);

    // q_approx * Q (low 64 bits)
    let qQ = u64_mul(q_approx, Q);

    // result = product - qQ
    var result = u64_sub(product, qQ);

    // Final reduction
    if (u64_ge(result, Q)) {
        result = u64_sub(result, Q);
    }

    return result;
}

// ============================================================================
// Butterfly Operations
// ============================================================================

// Cooley-Tukey butterfly (forward NTT)
fn ct_butterfly(lo: ptr<function, u64>, hi: ptr<function, u64>, omega: u64, Q: u64, precon: u64) {
    let u = *lo;
    let v = barrett_mul(*hi, omega, Q, precon);
    *lo = mod_add(u, v, Q);
    *hi = mod_sub(u, v, Q);
}

// Gentleman-Sande butterfly (inverse NTT)
fn gs_butterfly(lo: ptr<function, u64>, hi: ptr<function, u64>, omega: u64, Q: u64, precon: u64) {
    let u = *lo;
    let v = *hi;
    *lo = mod_add(u, v, Q);
    *hi = barrett_mul(mod_sub(u, v, Q), omega, Q, precon);
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read> twiddles: array<u32>;
@group(0) @binding(2) var<storage, read> precons: array<u32>;
@group(0) @binding(3) var<uniform> params: NTTParams;

// Shared memory (workgroup) - stores as pairs of u32
var<workgroup> shared_poly: array<u32, 4096>;  // 2048 u64 values
var<workgroup> shared_tw: array<u32, 4096>;    // 2048 u64 values

// Helper to read u64 from storage
fn read_u64(arr: ptr<storage, array<u32>, read>, idx: u32) -> u64 {
    return u64((*arr)[idx * 2u], (*arr)[idx * 2u + 1u]);
}

fn read_u64_rw(arr: ptr<storage, array<u32>, read_write>, idx: u32) -> u64 {
    return u64((*arr)[idx * 2u], (*arr)[idx * 2u + 1u]);
}

fn write_u64(arr: ptr<storage, array<u32>, read_write>, idx: u32, val: u64) {
    (*arr)[idx * 2u] = val.lo;
    (*arr)[idx * 2u + 1u] = val.hi;
}

fn read_shared(idx: u32) -> u64 {
    return u64(shared_poly[idx * 2u], shared_poly[idx * 2u + 1u]);
}

fn write_shared(idx: u32, val: u64) {
    shared_poly[idx * 2u] = val.lo;
    shared_poly[idx * 2u + 1u] = val.hi;
}

fn read_shared_tw(idx: u32) -> u64 {
    return u64(shared_tw[idx * 2u], shared_tw[idx * 2u + 1u]);
}

fn write_shared_tw(idx: u32, val: u64) {
    shared_tw[idx * 2u] = val.lo;
    shared_tw[idx * 2u + 1u] = val.hi;
}

// ============================================================================
// Staged NTT Kernel (for large N)
// ============================================================================

@compute @workgroup_size(256)
fn ntt_forward_stage(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let N = params.N;
    let Q = params_Q(params);
    let stage = params.stage;
    let batch_idx = workgroup_id.y;

    let butterfly_idx = global_id.x;
    let total_butterflies = N >> 1u;

    if (butterfly_idx >= total_butterflies) {
        return;
    }

    let m = 1u << stage;
    let t = N >> (stage + 1u);

    let group = butterfly_idx / t;
    let j = butterfly_idx % t;

    let idx_lo = group * (2u * t) + j;
    let idx_hi = idx_lo + t;

    let base = batch_idx * N;
    let tw_idx = m + group;

    let omega = read_u64(&twiddles, tw_idx);
    let precon = read_u64(&precons, tw_idx);

    var lo = read_u64_rw(&data, base + idx_lo);
    var hi = read_u64_rw(&data, base + idx_hi);

    ct_butterfly(&lo, &hi, omega, Q, precon);

    write_u64(&data, base + idx_lo, lo);
    write_u64(&data, base + idx_hi, hi);
}

@compute @workgroup_size(256)
fn ntt_inverse_stage(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let N = params.N;
    let Q = params_Q(params);
    let stage = params.stage;
    let batch_idx = workgroup_id.y;

    let butterfly_idx = global_id.x;
    let total_butterflies = N >> 1u;

    if (butterfly_idx >= total_butterflies) {
        return;
    }

    let m = N >> (stage + 1u);
    let t = 1u << stage;

    let group = butterfly_idx / t;
    let j = butterfly_idx % t;

    let idx_lo = group * (2u * t) + j;
    let idx_hi = idx_lo + t;

    let base = batch_idx * N;
    let tw_idx = m + group;

    let omega = read_u64(&twiddles, tw_idx);
    let precon = read_u64(&precons, tw_idx);

    var lo = read_u64_rw(&data, base + idx_lo);
    var hi = read_u64_rw(&data, base + idx_hi);

    gs_butterfly(&lo, &hi, omega, Q, precon);

    write_u64(&data, base + idx_lo, lo);
    write_u64(&data, base + idx_hi, hi);
}

// ============================================================================
// Fused NTT Kernel (all stages in workgroup memory, N <= 2048)
// ============================================================================

@compute @workgroup_size(256)
fn ntt_forward_fused(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let N = params.N;
    let log_N = params.log_N;
    let Q = params_Q(params);
    let batch_idx = workgroup_id.x;
    let tpg = 256u;

    let base = batch_idx * N;

    // Cooperative load: polynomial to shared memory
    for (var i = lid; i < N; i += tpg) {
        let val = read_u64_rw(&data, base + i);
        write_shared(i, val);
    }

    // Cooperative load: twiddles to shared memory
    for (var i = lid; i < N; i += tpg) {
        let val = read_u64(&twiddles, i);
        write_shared_tw(i, val);
    }

    workgroupBarrier();

    // Process all log_N stages
    for (var stage = 0u; stage < log_N; stage += 1u) {
        let m = 1u << stage;
        let t = N >> (stage + 1u);
        let butterflies_per_thread = (N / 2u + tpg - 1u) / tpg;

        for (var b = 0u; b < butterflies_per_thread; b += 1u) {
            let butterfly_idx = lid + b * tpg;
            if (butterfly_idx >= N / 2u) {
                break;
            }

            let group = butterfly_idx / t;
            let j = butterfly_idx % t;
            let idx_lo = group * (2u * t) + j;
            let idx_hi = idx_lo + t;

            let tw_idx = m + group;
            let omega = read_shared_tw(tw_idx);
            let precon = read_u64(&precons, tw_idx);

            var lo = read_shared(idx_lo);
            var hi = read_shared(idx_hi);

            ct_butterfly(&lo, &hi, omega, Q, precon);

            write_shared(idx_lo, lo);
            write_shared(idx_hi, hi);
        }

        workgroupBarrier();
    }

    // Write back to global memory
    for (var i = lid; i < N; i += tpg) {
        let val = read_shared(i);
        write_u64(&data, base + i, val);
    }
}

@compute @workgroup_size(256)
fn ntt_inverse_fused(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let N = params.N;
    let log_N = params.log_N;
    let Q = params_Q(params);
    let N_inv = params_N_inv(params);
    let N_inv_precon = params_N_inv_precon(params);
    let batch_idx = workgroup_id.x;
    let tpg = 256u;

    let base = batch_idx * N;

    // Cooperative load
    for (var i = lid; i < N; i += tpg) {
        let val = read_u64_rw(&data, base + i);
        write_shared(i, val);
    }

    for (var i = lid; i < N; i += tpg) {
        let val = read_u64(&twiddles, i);
        write_shared_tw(i, val);
    }

    workgroupBarrier();

    // Process all log_N stages (reverse direction)
    for (var stage = 0u; stage < log_N; stage += 1u) {
        let m = N >> (stage + 1u);
        let t = 1u << stage;
        let butterflies_per_thread = (N / 2u + tpg - 1u) / tpg;

        for (var b = 0u; b < butterflies_per_thread; b += 1u) {
            let butterfly_idx = lid + b * tpg;
            if (butterfly_idx >= N / 2u) {
                break;
            }

            let group = butterfly_idx / t;
            let j = butterfly_idx % t;
            let idx_lo = group * (2u * t) + j;
            let idx_hi = idx_lo + t;

            let tw_idx = m + group;
            let omega = read_shared_tw(tw_idx);
            let precon = read_u64(&precons, tw_idx);

            var lo = read_shared(idx_lo);
            var hi = read_shared(idx_hi);

            gs_butterfly(&lo, &hi, omega, Q, precon);

            write_shared(idx_lo, lo);
            write_shared(idx_hi, hi);
        }

        workgroupBarrier();
    }

    // Scale by N^{-1} and write back
    for (var i = lid; i < N; i += tpg) {
        var val = read_shared(i);
        val = barrett_mul(val, N_inv, Q, N_inv_precon);
        write_u64(&data, base + i, val);
    }
}

// ============================================================================
// Utility Kernels
// ============================================================================

@compute @workgroup_size(256)
fn ntt_pointwise_mul(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let N = params.N;
    let Q = params_Q(params);
    let batch_size = params.batch;
    let total = N * batch_size;

    let idx = global_id.x;
    if (idx >= total) {
        return;
    }

    // For pointwise mul, assume second operand in offset position
    // This kernel needs separate binding for second array
    // Simplified: square the element (self-multiply)
    let a = read_u64_rw(&data, idx);
    let result = barrett_mul(a, a, Q, params_mu(params));
    write_u64(&data, idx, result);
}

@compute @workgroup_size(256)
fn ntt_scale(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let N = params.N;
    let Q = params_Q(params);
    let batch_size = params.batch;
    let total = N * batch_size;
    let scale = params_N_inv(params);
    let scale_precon = params_N_inv_precon(params);

    let idx = global_id.x;
    if (idx >= total) {
        return;
    }

    let val = read_u64_rw(&data, idx);
    let result = barrett_mul(val, scale, Q, scale_precon);
    write_u64(&data, idx, result);
}
