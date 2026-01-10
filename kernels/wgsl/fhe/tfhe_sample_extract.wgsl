// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// TFHE Sample Extraction Kernels for WebGPU
// Extracts LWE ciphertext from GLWE ciphertext
// Critical operation in TFHE bootstrapping pipeline
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

// ============================================================================
// Modular Arithmetic
// ============================================================================

fn mod_neg(a: u64, Q: u64) -> u64 {
    if (a.lo == 0u && a.hi == 0u) { return a; }
    return u64_sub(Q, a);
}

fn mod_add(a: u64, b: u64, Q: u64) -> u64 {
    let sum = u64_add(a, b);
    if (u64_ge(sum, Q)) { return u64_sub(sum, Q); }
    return sum;
}

fn mod_sub(a: u64, b: u64, Q: u64) -> u64 {
    if (u64_ge(a, b)) { return u64_sub(a, b); }
    return u64_sub(u64_add(a, Q), b);
}

// ============================================================================
// Sample Extraction Parameters
// ============================================================================

struct SampleExtractParams {
    N: u32,              // GLWE polynomial degree
    k: u32,              // GLWE dimension (typically 1)
    n_out: u32,          // Output LWE dimension (N or custom)
    extract_idx: u32,    // Which coefficient to extract (0 to N-1)

    // Modulus
    Q_lo: u32,
    Q_hi: u32,

    // Batch
    batch_size: u32,
    _pad: u32,
}

fn params_Q(p: SampleExtractParams) -> u64 { return u64(p.Q_lo, p.Q_hi); }

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> glwe_in: array<u32>;        // Input GLWE [batch][(k+1)][N]
@group(0) @binding(1) var<storage, read_write> lwe_out: array<u32>;  // Output LWE [batch][kN + 1]
@group(0) @binding(2) var<uniform> params: SampleExtractParams;

// Shared memory for GLWE
var<workgroup> glwe_shared: array<u32, 4096>;

// ============================================================================
// Helper Functions
// ============================================================================

fn read_u64_storage(arr: ptr<storage, array<u32>, read>, idx: u32) -> u64 {
    return u64((*arr)[idx * 2u], (*arr)[idx * 2u + 1u]);
}

fn read_u64_storage_rw(arr: ptr<storage, array<u32>, read_write>, idx: u32) -> u64 {
    return u64((*arr)[idx * 2u], (*arr)[idx * 2u + 1u]);
}

fn write_u64_storage(arr: ptr<storage, array<u32>, read_write>, idx: u32, val: u64) {
    (*arr)[idx * 2u] = val.lo;
    (*arr)[idx * 2u + 1u] = val.hi;
}

fn read_shared(idx: u32) -> u64 {
    return u64(glwe_shared[idx * 2u], glwe_shared[idx * 2u + 1u]);
}

fn write_shared(idx: u32, val: u64) {
    glwe_shared[idx * 2u] = val.lo;
    glwe_shared[idx * 2u + 1u] = val.hi;
}

// ============================================================================
// Standard Sample Extract (Index 0)
// ============================================================================

// Extracts LWE ciphertext from coefficient 0 of GLWE
// LWE mask: a[i] = -s_{k}[N-1-i] for i > 0, a[0] = s_{k}[0]
// LWE body: b = body[0]
@compute @workgroup_size(256)
fn sample_extract_index_0(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let Q = params_Q(params);
    let tpg = 256u;

    let batch_idx = wgid.x;

    if (batch_idx >= params.batch_size) { return; }

    let glwe_base = batch_idx * (k + 1u) * N;
    let lwe_base = batch_idx * (k * N + 1u);

    // For each mask polynomial in GLWE
    for (var poly_idx = 0u; poly_idx < k; poly_idx++) {
        let poly_offset = poly_idx * N;

        // Load polynomial to shared memory
        for (var i = tid; i < N; i += tpg) {
            let val = read_u64_storage(&glwe_in, glwe_base + poly_offset + i);
            write_shared(i, val);
        }
        workgroupBarrier();

        // Extract with index reversal and negation
        for (var i = tid; i < N; i += tpg) {
            var lwe_coeff: u64;

            if (i == 0u) {
                // a[0] = s[0]
                lwe_coeff = read_shared(0u);
            } else {
                // a[i] = -s[N-i] for i > 0
                let src_idx = N - i;
                lwe_coeff = mod_neg(read_shared(src_idx), Q);
            }

            let out_idx = lwe_base + poly_idx * N + i;
            write_u64_storage(&lwe_out, out_idx, lwe_coeff);
        }
        workgroupBarrier();
    }

    // Extract body coefficient (constant term)
    if (tid == 0u) {
        let body = read_u64_storage(&glwe_in, glwe_base + k * N);
        write_u64_storage(&lwe_out, lwe_base + k * N, body);
    }
}

// ============================================================================
// General Sample Extract (Any Index)
// ============================================================================

// Extracts LWE ciphertext from coefficient at extract_idx
@compute @workgroup_size(256)
fn sample_extract_general(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let h = params.extract_idx;  // Index to extract
    let Q = params_Q(params);
    let tpg = 256u;

    let batch_idx = wgid.x;

    if (batch_idx >= params.batch_size || h >= N) { return; }

    let glwe_base = batch_idx * (k + 1u) * N;
    let lwe_base = batch_idx * (k * N + 1u);

    // For each mask polynomial in GLWE
    for (var poly_idx = 0u; poly_idx < k; poly_idx++) {
        let poly_offset = poly_idx * N;

        // Load polynomial to shared memory
        for (var i = tid; i < N; i += tpg) {
            let val = read_u64_storage(&glwe_in, glwe_base + poly_offset + i);
            write_shared(i, val);
        }
        workgroupBarrier();

        // Extract with rotation by h
        // a[i] = s_{poly}[(h - i) mod N] with negation for wrap-around
        for (var i = tid; i < N; i += tpg) {
            var lwe_coeff: u64;

            if (i <= h) {
                // No wrap-around: a[i] = s[h - i]
                let src_idx = h - i;
                lwe_coeff = read_shared(src_idx);
            } else {
                // Wrap-around: a[i] = -s[N + h - i]
                let src_idx = N + h - i;
                lwe_coeff = mod_neg(read_shared(src_idx), Q);
            }

            let out_idx = lwe_base + poly_idx * N + i;
            write_u64_storage(&lwe_out, out_idx, lwe_coeff);
        }
        workgroupBarrier();
    }

    // Extract body coefficient at index h
    if (tid == 0u) {
        let body = read_u64_storage(&glwe_in, glwe_base + k * N + h);
        write_u64_storage(&lwe_out, lwe_base + k * N, body);
    }
}

// ============================================================================
// Batch Sample Extract (Multiple Indices)
// ============================================================================

@group(1) @binding(0) var<storage, read> extract_indices: array<u32>;  // Indices to extract
@group(1) @binding(1) var<storage, read_write> lwe_batch_out: array<u32>;
@group(1) @binding(2) var<uniform> num_extractions: u32;

@compute @workgroup_size(256)
fn sample_extract_batch(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let Q = params_Q(params);
    let tpg = 256u;

    let batch_idx = wgid.x;
    let extract_num = wgid.y;

    if (batch_idx >= params.batch_size || extract_num >= num_extractions) { return; }

    let h = extract_indices[extract_num];

    if (h >= N) { return; }

    let glwe_base = batch_idx * (k + 1u) * N;
    let lwe_size = k * N + 1u;
    let lwe_base = (batch_idx * num_extractions + extract_num) * lwe_size;

    // For each mask polynomial
    for (var poly_idx = 0u; poly_idx < k; poly_idx++) {
        let poly_offset = poly_idx * N;

        // Load to shared
        for (var i = tid; i < N; i += tpg) {
            let val = read_u64_storage(&glwe_in, glwe_base + poly_offset + i);
            write_shared(i, val);
        }
        workgroupBarrier();

        // Extract
        for (var i = tid; i < N; i += tpg) {
            var lwe_coeff: u64;

            if (i <= h) {
                lwe_coeff = read_shared(h - i);
            } else {
                lwe_coeff = mod_neg(read_shared(N + h - i), Q);
            }

            let out_idx = lwe_base + poly_idx * N + i;
            lwe_batch_out[out_idx * 2u] = lwe_coeff.lo;
            lwe_batch_out[out_idx * 2u + 1u] = lwe_coeff.hi;
        }
        workgroupBarrier();
    }

    // Body
    if (tid == 0u) {
        let body = read_u64_storage(&glwe_in, glwe_base + k * N + h);
        lwe_batch_out[(lwe_base + k * N) * 2u] = body.lo;
        lwe_batch_out[(lwe_base + k * N) * 2u + 1u] = body.hi;
    }
}

// ============================================================================
// Trace (Sum of Rotations) for Packing
// ============================================================================

// Computes trace: sum_{i=0}^{N-1} X^i * GLWE to pack multiple LWEs
@compute @workgroup_size(256)
fn glwe_trace(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let Q = params_Q(params);
    let tpg = 256u;

    let batch_idx = wgid.x;
    let out_poly = wgid.y;  // 0 to k

    if (batch_idx >= params.batch_size || out_poly > k) { return; }

    let glwe_base = batch_idx * (k + 1u) * N;

    // Load polynomial to shared
    for (var i = tid; i < N; i += tpg) {
        let val = read_u64_storage(&glwe_in, glwe_base + out_poly * N + i);
        write_shared(i, val);
    }
    workgroupBarrier();

    // Compute trace: sum of all rotations
    // For coefficient 0: just sum all coefficients with appropriate signs
    // For negacyclic: trace[0] = sum_{i=0}^{N-1} coeff[i]
    // Other coefficients sum to 0 in trace

    var sum = u64_zero();

    // Parallel reduction
    for (var i = tid; i < N; i += tpg) {
        let coeff = read_shared(i);
        sum = mod_add(sum, coeff, Q);
    }

    // Store partial sum (needs workgroup reduction)
    write_shared(tid + N, sum);
    workgroupBarrier();

    // Simple reduction (could be optimized)
    if (tid == 0u) {
        var total = u64_zero();
        for (var i = 0u; i < tpg; i++) {
            total = mod_add(total, read_shared(i + N), Q);
        }

        // Write trace result (only coefficient 0 is non-zero)
        write_u64_storage(&lwe_out, batch_idx * (k + 1u) * N + out_poly * N, total);

        // Zero other coefficients
        for (var i = 1u; i < N; i++) {
            write_u64_storage(&lwe_out, batch_idx * (k + 1u) * N + out_poly * N + i, u64_zero());
        }
    }
}

// ============================================================================
// Partial Trace (for Packing Subset)
// ============================================================================

@group(2) @binding(0) var<uniform> trace_power: u32;  // Trace T_{N/2^k}

@compute @workgroup_size(256)
fn glwe_partial_trace(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let Q = params_Q(params);
    let tpg = 256u;

    let batch_idx = wgid.x;
    let out_poly = wgid.y;

    if (batch_idx >= params.batch_size || out_poly > k) { return; }

    let step = 1u << trace_power;  // 2^trace_power
    let num_terms = N / step;

    let glwe_base = batch_idx * (k + 1u) * N;

    // Load polynomial to shared
    for (var i = tid; i < N; i += tpg) {
        let val = read_u64_storage(&glwe_in, glwe_base + out_poly * N + i);
        write_shared(i, val);
    }
    workgroupBarrier();

    // Compute partial trace
    for (var out_idx = tid; out_idx < step; out_idx += tpg) {
        var sum = u64_zero();

        for (var t = 0u; t < num_terms; t++) {
            let src_idx = out_idx + t * step;
            let coeff = read_shared(src_idx);

            // Apply appropriate sign for negacyclic
            if (t % 2u == 0u) {
                sum = mod_add(sum, coeff, Q);
            } else {
                sum = mod_sub(sum, coeff, Q);
            }
        }

        let result_idx = batch_idx * (k + 1u) * N + out_poly * N + out_idx;
        write_u64_storage(&lwe_out, result_idx, sum);
    }
}

// ============================================================================
// Sample Extract with Keyswitch
// ============================================================================

@group(3) @binding(0) var<storage, read> ksk: array<u32>;  // Keyswitch key for dimension reduction

struct KSParams {
    n_out: u32,      // Output LWE dimension
    l_ks: u32,       // Keyswitch decomposition levels
    base_log_ks: u32,// Keyswitch base log
    _pad: u32,
}

@group(3) @binding(1) var<uniform> ks_params: KSParams;

@compute @workgroup_size(256)
fn sample_extract_and_keyswitch(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let n_out = ks_params.n_out;
    let l_ks = ks_params.l_ks;
    let base_log = ks_params.base_log_ks;
    let Q = params_Q(params);
    let tpg = 256u;

    let batch_idx = wgid.x;
    let out_idx = wgid.y * tpg + tid;  // Output LWE coefficient index

    if (batch_idx >= params.batch_size || out_idx > n_out) { return; }

    let glwe_base = batch_idx * (k + 1u) * N;
    let h = params.extract_idx;

    // First: extract to full-dimension LWE (in shared memory)
    for (var poly_idx = 0u; poly_idx < k; poly_idx++) {
        for (var i = tid; i < N; i += tpg) {
            let glwe_idx = glwe_base + poly_idx * N + i;
            let val = read_u64_storage(&glwe_in, glwe_idx);
            write_shared(poly_idx * N + i, val);
        }
    }
    workgroupBarrier();

    // Second: keyswitch to reduce dimension
    // For each output coefficient, accumulate from full LWE via KSK

    if (out_idx == n_out) {
        // Body: just extract from GLWE body
        let body = read_u64_storage(&glwe_in, glwe_base + k * N + h);
        let lwe_base = batch_idx * (n_out + 1u);
        write_u64_storage(&lwe_out, lwe_base + n_out, body);
        return;
    }

    var acc = u64_zero();
    let Bg = 1u << base_log;
    let half_Bg = Bg >> 1u;
    let mask = Bg - 1u;

    // For each full LWE coefficient
    for (var i = 0u; i < k * N; i++) {
        // Get extracted LWE coefficient
        var extracted: u64;
        let poly_idx = i / N;
        let coeff_idx = i % N;

        if (coeff_idx <= h) {
            extracted = read_shared(poly_idx * N + (h - coeff_idx));
        } else {
            extracted = mod_neg(read_shared(poly_idx * N + (N + h - coeff_idx)), Q);
        }

        // Decompose and apply KSK
        for (var level = 0u; level < l_ks; level++) {
            let shift = 64u - (level + 1u) * base_log;

            var digit: u32;
            if (shift >= 32u) {
                digit = (extracted.hi >> (shift - 32u)) & mask;
            } else if (shift == 0u) {
                digit = extracted.lo & mask;
            } else {
                digit = ((extracted.lo >> shift) | (extracted.hi << (32u - shift))) & mask;
            }

            if (digit == 0u) { continue; }

            // Get KSK coefficient
            let ksk_idx = (i * l_ks + level) * (n_out + 1u) + out_idx;
            let ksk_val = read_u64_storage(&ksk, ksk_idx);

            // Centered representation
            if (digit >= half_Bg) {
                let neg_digit = Bg - digit;
                let prod = u64_mul(u64_from_u32(neg_digit), ksk_val);
                acc = mod_add(acc, prod, Q);
            } else {
                let prod = u64_mul(u64_from_u32(digit), ksk_val);
                acc = mod_sub(acc, prod, Q);
            }
        }
    }

    let lwe_base = batch_idx * (n_out + 1u);
    write_u64_storage(&lwe_out, lwe_base + out_idx, acc);
}

// ============================================================================
// Multi-Value Sample Extract (for MFHE)
// ============================================================================

@group(4) @binding(0) var<storage, read_write> multi_lwe_out: array<u32>;
@group(4) @binding(1) var<uniform> num_values: u32;

// Extract multiple values from GLWE (for multi-value FHE)
@compute @workgroup_size(256)
fn sample_extract_multi(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let Q = params_Q(params);
    let tpg = 256u;

    let batch_idx = wgid.x;
    let value_idx = wgid.y;

    if (batch_idx >= params.batch_size || value_idx >= num_values) { return; }

    // Extract at index value_idx * (N / num_values)
    let spacing = N / num_values;
    let h = value_idx * spacing;

    let glwe_base = batch_idx * (k + 1u) * N;
    let lwe_size = k * N + 1u;
    let lwe_base = (batch_idx * num_values + value_idx) * lwe_size;

    // Process each mask polynomial
    for (var poly_idx = 0u; poly_idx < k; poly_idx++) {
        // Load polynomial
        for (var i = tid; i < N; i += tpg) {
            let val = read_u64_storage(&glwe_in, glwe_base + poly_idx * N + i);
            write_shared(i, val);
        }
        workgroupBarrier();

        // Extract with rotation
        for (var i = tid; i < N; i += tpg) {
            var lwe_coeff: u64;

            if (i <= h) {
                lwe_coeff = read_shared(h - i);
            } else {
                lwe_coeff = mod_neg(read_shared(N + h - i), Q);
            }

            multi_lwe_out[(lwe_base + poly_idx * N + i) * 2u] = lwe_coeff.lo;
            multi_lwe_out[(lwe_base + poly_idx * N + i) * 2u + 1u] = lwe_coeff.hi;
        }
        workgroupBarrier();
    }

    // Body
    if (tid == 0u) {
        let body = read_u64_storage(&glwe_in, glwe_base + k * N + h);
        multi_lwe_out[(lwe_base + k * N) * 2u] = body.lo;
        multi_lwe_out[(lwe_base + k * N) * 2u + 1u] = body.hi;
    }
}
