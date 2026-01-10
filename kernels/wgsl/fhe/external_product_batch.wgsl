// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// TFHE Batched External Product Kernel for WebGPU
// High-throughput GGSW x GLWE batch operations with memory coalescing
// Compatible with Metal/Vulkan/D3D12 via Dawn/wgpu

// ============================================================================
// 64-bit Integer Emulation
// ============================================================================

struct u64 {
    lo: u32,
    hi: u32,
}

fn u64_zero() -> u64 { return u64(0u, 0u); }

fn u64_from_u32(v: u32) -> u64 { return u64(v, 0u); }

fn u64_eq(a: u64, b: u64) -> bool {
    return a.lo == b.lo && a.hi == b.hi;
}

fn u64_lt(a: u64, b: u64) -> bool {
    if (a.hi != b.hi) { return a.hi < b.hi; }
    return a.lo < b.lo;
}

fn u64_ge(a: u64, b: u64) -> bool { return !u64_lt(a, b); }

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

    if (u64_lt(mid, p1)) {
        result = u64_add(result, u64(0u, 1u));
    }
    if (u64_lt(mid2, mid)) {
        result = u64_add(result, u64(1u, 0u));
    }

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

// Barrett reduction: (a * b) mod Q
fn barrett_mul(a: u64, b: u64, Q: u64, mu: u64) -> u64 {
    let q_approx = u64_mulhi(a, mu);
    let product = u64_mul(a, b);
    let qQ = u64_mul(q_approx, Q);
    var result = u64_sub(product, qQ);
    if (u64_ge(result, Q)) { result = u64_sub(result, Q); }
    return result;
}

// Simplified modular multiplication for 32-bit effective moduli
fn mod_mul_simple(a: u64, b: u64, Q: u64) -> u64 {
    let prod = u64_mul(a, b);
    if (Q.hi == 0u && Q.lo != 0u) {
        // For 32-bit modulus, can use hardware division
        let result = prod.lo % Q.lo;
        return u64(result, 0u);
    }
    // For larger moduli, need proper Barrett reduction
    return prod;
}

// ============================================================================
// Batch External Product Parameters
// ============================================================================

struct BatchExtProdParams {
    // Dimensions
    N: u32,              // Polynomial degree (1024, 2048)
    k: u32,              // GLWE dimension (1)
    l: u32,              // Decomposition levels (3-4)
    base_log: u32,       // Base log (8)

    // Modulus (split for u64)
    Q_lo: u32,
    Q_hi: u32,

    // Barrett constant
    mu_lo: u32,
    mu_hi: u32,

    // Batch control
    batch_size: u32,     // Number of external products in batch
    num_ggsw: u32,       // Number of GGSW ciphertexts (for CMux tree)

    // Memory layout
    glwe_stride: u32,    // Elements between GLWE ciphertexts
    ggsw_stride: u32,    // Elements between GGSW ciphertexts

    // Processing mode
    ntt_domain: u32,     // 1 if inputs are in NTT domain
    accumulate: u32,     // 1 to accumulate into output
}

fn params_Q(p: BatchExtProdParams) -> u64 { return u64(p.Q_lo, p.Q_hi); }
fn params_mu(p: BatchExtProdParams) -> u64 { return u64(p.mu_lo, p.mu_hi); }

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> glwe_batch: array<u32>;     // Batch of GLWE inputs
@group(0) @binding(1) var<storage, read> ggsw_batch: array<u32>;     // Batch of GGSW keys
@group(0) @binding(2) var<storage, read_write> output_batch: array<u32>; // Batch of outputs
@group(0) @binding(3) var<uniform> params: BatchExtProdParams;

// Shared memory for coefficient caching
var<workgroup> glwe_shared: array<u32, 2048>;  // GLWE coefficients
var<workgroup> decomp_shared: array<i32, 1024>; // Decomposed digits

// ============================================================================
// Signed Decomposition
// ============================================================================

fn signed_decompose_digit(val: u64, level: u32, base_log: u32) -> i32 {
    let Bg = 1u << base_log;
    let half_Bg = Bg >> 1u;
    let mask = Bg - 1u;

    // Extract digit from position: 64 - (level + 1) * base_log
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

    // Centered representation
    if (digit >= half_Bg) {
        return i32(digit) - i32(Bg);
    }
    return i32(digit);
}

// Batch decomposition with carry propagation
fn decompose_with_carry(val: u64, l: u32, base_log: u32, out_digits: ptr<function, array<i32, 8>>) {
    let Bg = 1u << base_log;
    let half_Bg = Bg >> 1u;
    let mask = Bg - 1u;

    var carry: u32 = 0u;

    for (var level = 0u; level < l && level < 8u; level++) {
        let shift = 64u - (level + 1u) * base_log;

        var digit: u32;
        if (shift >= 32u) {
            digit = ((val.hi >> (shift - 32u)) + carry) & mask;
        } else if (shift == 0u) {
            digit = (val.lo + carry) & mask;
        } else {
            let lo_part = val.lo >> shift;
            let hi_part = val.hi << (32u - shift);
            digit = ((lo_part | hi_part) + carry) & mask;
        }
        carry = 0u;

        if (digit >= half_Bg) {
            (*out_digits)[level] = i32(digit) - i32(Bg);
            carry = 1u;
        } else {
            (*out_digits)[level] = i32(digit);
        }
    }
}

// ============================================================================
// Single External Product (within batch)
// ============================================================================

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

// ============================================================================
// Batch External Product - Coefficient Domain
// ============================================================================

@compute @workgroup_size(256)
fn external_product_batch_coeff(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let l = params.l;
    let Q = params_Q(params);
    let tpg = 256u;

    let coeff_idx = gid.x % N;
    let out_poly = (gid.x / N) % (k + 1u);
    let batch_idx = wgid.z;

    if (coeff_idx >= N || out_poly > k || batch_idx >= params.batch_size) {
        return;
    }

    // Calculate offsets
    let glwe_offset = batch_idx * params.glwe_stride;
    let ggsw_offset = batch_idx * params.ggsw_stride;
    let out_offset = batch_idx * params.glwe_stride;

    var acc = u64_zero();

    // For each input polynomial
    for (var in_poly = 0u; in_poly <= k; in_poly++) {
        // Load GLWE coefficient
        let glwe_idx = glwe_offset + in_poly * N + coeff_idx;
        let val = read_u64(&glwe_batch, glwe_idx);

        // Decompose
        var digits: array<i32, 8>;
        decompose_with_carry(val, l, params.base_log, &digits);

        // For each decomposition level
        for (var level = 0u; level < l; level++) {
            let digit = digits[level];

            if (digit == 0) { continue; }

            // GGSW coefficient index
            let ggsw_idx = ggsw_offset + ((in_poly * l + level) * (k + 1u) + out_poly) * N + coeff_idx;
            let ggsw_val = read_u64(&ggsw_batch, ggsw_idx);

            // Multiply
            let abs_digit = u32(select(-digit, digit, digit >= 0));
            let prod = mod_mul_simple(u64_from_u32(abs_digit), ggsw_val, Q);

            if (digit > 0) {
                acc = mod_add(acc, prod, Q);
            } else {
                acc = mod_sub(acc, prod, Q);
            }
        }
    }

    // Store result
    let result_idx = out_offset + out_poly * N + coeff_idx;

    if (params.accumulate != 0u) {
        let existing = read_u64_rw(&output_batch, result_idx);
        acc = mod_add(existing, acc, Q);
    }

    write_u64(&output_batch, result_idx, acc);
}

// ============================================================================
// Batch External Product - NTT Domain (Production)
// ============================================================================

@compute @workgroup_size(256)
fn external_product_batch_ntt(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let l = params.l;
    let Q = params_Q(params);
    let mu = params_mu(params);
    let tpg = 256u;

    let coeff_idx = gid.x;
    let out_poly = wgid.y;
    let batch_idx = wgid.z;

    if (coeff_idx >= N || out_poly > k || batch_idx >= params.batch_size) {
        return;
    }

    // Load GLWE to shared memory cooperatively
    let glwe_offset = batch_idx * params.glwe_stride;

    for (var i = tid; i < (k + 1u) * N * 2u && i < 2048u; i += tpg) {
        glwe_shared[i] = glwe_batch[(glwe_offset * 2u) + i];
    }
    workgroupBarrier();

    // Calculate offsets
    let ggsw_offset = batch_idx * params.ggsw_stride;
    let out_offset = batch_idx * params.glwe_stride;

    var acc = u64_zero();

    for (var in_poly = 0u; in_poly <= k; in_poly++) {
        // Read from shared memory
        let shared_idx = in_poly * N * 2u + coeff_idx * 2u;
        var val: u64;
        if (shared_idx + 1u < 2048u) {
            val = u64(glwe_shared[shared_idx], glwe_shared[shared_idx + 1u]);
        } else {
            val = read_u64(&glwe_batch, glwe_offset + in_poly * N + coeff_idx);
        }

        for (var level = 0u; level < l; level++) {
            let digit = signed_decompose_digit(val, level, params.base_log);

            if (digit == 0) { continue; }

            let ggsw_idx = ggsw_offset + ((in_poly * l + level) * (k + 1u) + out_poly) * N + coeff_idx;
            let ggsw_val = read_u64(&ggsw_batch, ggsw_idx);

            // NTT domain: pointwise multiply
            let abs_digit = u32(select(-digit, digit, digit >= 0));
            let prod = barrett_mul(u64_from_u32(abs_digit), ggsw_val, Q, mu);

            if (digit > 0) {
                acc = mod_add(acc, prod, Q);
            } else {
                acc = mod_sub(acc, prod, Q);
            }
        }
    }

    let result_idx = out_offset + out_poly * N + coeff_idx;

    if (params.accumulate != 0u) {
        let existing = read_u64_rw(&output_batch, result_idx);
        acc = mod_add(existing, acc, Q);
    }

    write_u64(&output_batch, result_idx, acc);
}

// ============================================================================
// Parallel Decomposition Kernel
// ============================================================================

@group(1) @binding(0) var<storage, read_write> decomposed: array<i32>;

@compute @workgroup_size(256)
fn batch_decompose(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let l = params.l;

    let coeff_idx = gid.x;
    let poly_idx = wgid.y;
    let batch_idx = wgid.z;

    if (coeff_idx >= N || poly_idx > k || batch_idx >= params.batch_size) {
        return;
    }

    let glwe_offset = batch_idx * params.glwe_stride;
    let glwe_idx = glwe_offset + poly_idx * N + coeff_idx;
    let val = read_u64(&glwe_batch, glwe_idx);

    // Decompose and store all levels
    for (var level = 0u; level < l; level++) {
        let digit = signed_decompose_digit(val, level, params.base_log);
        let out_idx = ((batch_idx * (k + 1u) + poly_idx) * l + level) * N + coeff_idx;
        decomposed[out_idx] = digit;
    }
}

// ============================================================================
// Parallel Multiply-Accumulate with Pre-decomposed Input
// ============================================================================

@compute @workgroup_size(256)
fn batch_multiply_accumulate(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let l = params.l;
    let Q = params_Q(params);
    let mu = params_mu(params);

    let coeff_idx = gid.x;
    let out_poly = wgid.y;
    let batch_idx = wgid.z;

    if (coeff_idx >= N || out_poly > k || batch_idx >= params.batch_size) {
        return;
    }

    let ggsw_offset = batch_idx * params.ggsw_stride;

    var acc = u64_zero();

    for (var in_poly = 0u; in_poly <= k; in_poly++) {
        for (var level = 0u; level < l; level++) {
            // Read pre-decomposed digit
            let decomp_idx = ((batch_idx * (k + 1u) + in_poly) * l + level) * N + coeff_idx;
            let digit = decomposed[decomp_idx];

            if (digit == 0) { continue; }

            let ggsw_idx = ggsw_offset + ((in_poly * l + level) * (k + 1u) + out_poly) * N + coeff_idx;
            let ggsw_val = read_u64(&ggsw_batch, ggsw_idx);

            let abs_digit = u32(select(-digit, digit, digit >= 0));
            let prod = barrett_mul(u64_from_u32(abs_digit), ggsw_val, Q, mu);

            if (digit > 0) {
                acc = mod_add(acc, prod, Q);
            } else {
                acc = mod_sub(acc, prod, Q);
            }
        }
    }

    let out_offset = batch_idx * params.glwe_stride;
    let result_idx = out_offset + out_poly * N + coeff_idx;
    write_u64(&output_batch, result_idx, acc);
}

// ============================================================================
// Fused Batch CMux (for bootstrapping tree)
// ============================================================================

@group(2) @binding(0) var<storage, read> glwe_d0_batch: array<u32>;  // GLWE batch 0
@group(2) @binding(1) var<storage, read> glwe_d1_batch: array<u32>;  // GLWE batch 1
@group(2) @binding(2) var<storage, read_write> cmux_output: array<u32>;

@compute @workgroup_size(256)
fn batch_cmux(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let l = params.l;
    let Q = params_Q(params);
    let mu = params_mu(params);
    let tpg = 256u;

    let coeff_idx = gid.x;
    let out_poly = wgid.y;
    let batch_idx = wgid.z;

    if (coeff_idx >= N || out_poly > k || batch_idx >= params.batch_size) {
        return;
    }

    let glwe_stride = params.glwe_stride;
    let ggsw_offset = batch_idx * params.ggsw_stride;
    let d0_offset = batch_idx * glwe_stride;
    let d1_offset = batch_idx * glwe_stride;

    // Load d0 and d1 coefficients
    // Compute diff = d1 - d0 in shared memory
    for (var p = 0u; p <= k; p++) {
        let idx = p * N + coeff_idx;
        let d0 = read_u64(&glwe_d0_batch, d0_offset + idx);
        let d1 = read_u64(&glwe_d1_batch, d1_offset + idx);
        let diff = mod_sub(d1, d0, Q);

        let shared_idx = p * N * 2u + coeff_idx * 2u;
        if (shared_idx + 1u < 2048u) {
            glwe_shared[shared_idx] = diff.lo;
            glwe_shared[shared_idx + 1u] = diff.hi;
        }
    }
    workgroupBarrier();

    // External product with diff
    var ext_prod = u64_zero();

    for (var in_poly = 0u; in_poly <= k; in_poly++) {
        let shared_idx = in_poly * N * 2u + coeff_idx * 2u;
        var diff: u64;
        if (shared_idx + 1u < 2048u) {
            diff = u64(glwe_shared[shared_idx], glwe_shared[shared_idx + 1u]);
        } else {
            let d0 = read_u64(&glwe_d0_batch, d0_offset + in_poly * N + coeff_idx);
            let d1 = read_u64(&glwe_d1_batch, d1_offset + in_poly * N + coeff_idx);
            diff = mod_sub(d1, d0, Q);
        }

        for (var level = 0u; level < l; level++) {
            let digit = signed_decompose_digit(diff, level, params.base_log);

            if (digit == 0) { continue; }

            let ggsw_idx = ggsw_offset + ((in_poly * l + level) * (k + 1u) + out_poly) * N + coeff_idx;
            let ggsw_val = read_u64(&ggsw_batch, ggsw_idx);

            let abs_digit = u32(select(-digit, digit, digit >= 0));
            let prod = barrett_mul(u64_from_u32(abs_digit), ggsw_val, Q, mu);

            if (digit > 0) {
                ext_prod = mod_add(ext_prod, prod, Q);
            } else {
                ext_prod = mod_sub(ext_prod, prod, Q);
            }
        }
    }

    // Result = d0 + external_product
    let d0_val = read_u64(&glwe_d0_batch, d0_offset + out_poly * N + coeff_idx);
    let result = mod_add(d0_val, ext_prod, Q);

    let out_idx = batch_idx * glwe_stride + out_poly * N + coeff_idx;
    write_u64(&cmux_output, out_idx, result);
}

// ============================================================================
// Reduction Kernel for Parallel External Products
// ============================================================================

@group(3) @binding(0) var<storage, read> partial_results: array<u32>;
@group(3) @binding(1) var<storage, read_write> final_result: array<u32>;

struct ReductionParams {
    N: u32,
    k: u32,
    num_partials: u32,
    Q_lo: u32,
    Q_hi: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(3) @binding(2) var<uniform> reduce_params: ReductionParams;

@compute @workgroup_size(256)
fn reduce_external_products(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = reduce_params.N;
    let k = reduce_params.k;
    let Q = u64(reduce_params.Q_lo, reduce_params.Q_hi);

    let coeff_idx = gid.x;
    let out_poly = wgid.y;

    if (coeff_idx >= N || out_poly > k) {
        return;
    }

    let stride = (k + 1u) * N;

    var sum = u64_zero();

    for (var p = 0u; p < reduce_params.num_partials; p++) {
        let idx = p * stride + out_poly * N + coeff_idx;
        let val = u64(partial_results[idx * 2u], partial_results[idx * 2u + 1u]);
        sum = mod_add(sum, val, Q);
    }

    let out_idx = out_poly * N + coeff_idx;
    final_result[out_idx * 2u] = sum.lo;
    final_result[out_idx * 2u + 1u] = sum.hi;
}
