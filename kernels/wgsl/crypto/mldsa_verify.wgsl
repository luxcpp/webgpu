// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// ML-DSA Batch Verification - WebGPU WGSL Kernel
// GPU-accelerated ML-DSA (FIPS 204) signature verification
//
// ML-DSA Parameters (65 = NIST Level 3):
//   n = 256 (polynomial degree)
//   q = 8380417 (prime modulus)
//   k = 6 (rows in A matrix)
//   l = 5 (columns in A matrix)
//
// Key sizes:
//   Public key: 1952 bytes
//   Signature: 3309 bytes

// ============================================================================
// ML-DSA-65 Parameters
// ============================================================================

const MLDSA_N: u32 = 256u;         // Polynomial degree
const MLDSA_Q: u32 = 8380417u;     // Prime modulus
const MLDSA_K: u32 = 6u;           // A matrix rows
const MLDSA_L: u32 = 5u;           // A matrix columns
const MLDSA_LOG_N: u32 = 8u;       // log2(N)

// Montgomery constants for q = 8380417
const MLDSA_QINV: u32 = 58728449u; // -q^{-1} mod 2^32

// ============================================================================
// Modular Arithmetic
// ============================================================================

// Barrett reduction: compute a mod q where a < q^2
fn barrett_reduce(a: u32) -> u32 {
    // For q = 8380417, mu = floor(2^46 / q) approx
    let mu = 8396807u;
    let q_approx = ((u64(a) * u64(mu)) >> 46u);
    var result = a - u32(q_approx) * MLDSA_Q;

    if (result >= MLDSA_Q) {
        result = result - MLDSA_Q;
    }
    return result;
}

// Montgomery multiplication for ML-DSA
fn mont_mul(a: u32, b: u32) -> u32 {
    let prod = u64(a) * u64(b);
    let m = u32(prod) * MLDSA_QINV;
    let t = prod + u64(m) * u64(MLDSA_Q);
    var result = u32(t >> 32u);
    if (result >= MLDSA_Q) {
        result = result - MLDSA_Q;
    }
    return result;
}

// Modular addition
fn mod_add(a: u32, b: u32) -> u32 {
    var sum = a + b;
    if (sum >= MLDSA_Q) {
        sum = sum - MLDSA_Q;
    }
    return sum;
}

// Modular subtraction
fn mod_sub(a: u32, b: u32) -> u32 {
    if (a >= b) {
        return a - b;
    }
    return a + MLDSA_Q - b;
}

// ============================================================================
// NTT Butterfly Operations
// ============================================================================

// Cooley-Tukey butterfly for forward NTT
fn ct_butterfly(lo: u32, hi: u32, omega: u32) -> vec2<u32> {
    let omega_hi = mont_mul(hi, omega);
    let new_lo = mod_add(lo, omega_hi);
    let new_hi = mod_sub(lo, omega_hi);
    return vec2<u32>(new_lo, new_hi);
}

// Gentleman-Sande butterfly for inverse NTT
fn gs_butterfly(lo: u32, hi: u32, omega: u32) -> vec2<u32> {
    let sum = mod_add(lo, hi);
    let diff = mod_sub(lo, hi);
    let new_hi = mont_mul(diff, omega);
    return vec2<u32>(sum, new_hi);
}

// ============================================================================
// Bindings
// ============================================================================

struct MLDSAParams {
    batch_size: u32,
    stage: u32,
    mode: u32,  // 0=forward NTT, 1=inverse NTT
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> polys: array<u32>;
@group(0) @binding(1) var<storage, read> twiddles: array<u32>;
@group(0) @binding(2) var<uniform> params: MLDSAParams;

// ============================================================================
// NTT Kernels
// ============================================================================

// Batch NTT forward transform (staged execution)
@compute @workgroup_size(128)
fn mldsa_ntt_forward(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_idx = gid.y;
    let butterfly_idx = gid.x;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let stage = params.stage;
    let m = 1u << stage;
    let t = MLDSA_N >> (stage + 1u);
    let num_butterflies = MLDSA_N >> 1u;

    if (butterfly_idx >= num_butterflies) {
        return;
    }

    let i = butterfly_idx / t;
    let j = butterfly_idx % t;
    let idx_lo = (i << (MLDSA_LOG_N - stage)) + j;
    let idx_hi = idx_lo + t;

    let poly_offset = batch_idx * MLDSA_N;
    let omega = twiddles[m + i];

    let lo = polys[poly_offset + idx_lo];
    let hi = polys[poly_offset + idx_hi];
    let result = ct_butterfly(lo, hi, omega);

    polys[poly_offset + idx_lo] = result.x;
    polys[poly_offset + idx_hi] = result.y;
}

// Batch NTT inverse transform (staged execution)
@compute @workgroup_size(128)
fn mldsa_ntt_inverse(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_idx = gid.y;
    let butterfly_idx = gid.x;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let stage = params.stage;
    let m = MLDSA_N >> (stage + 1u);
    let t = 1u << stage;
    let num_butterflies = MLDSA_N >> 1u;

    if (butterfly_idx >= num_butterflies) {
        return;
    }

    let i = butterfly_idx / t;
    let j = butterfly_idx % t;
    let idx_lo = (i << (stage + 1u)) + j;
    let idx_hi = idx_lo + t;

    let poly_offset = batch_idx * MLDSA_N;
    let omega = twiddles[MLDSA_N + m + i];  // Inverse twiddles in second half

    let lo = polys[poly_offset + idx_lo];
    let hi = polys[poly_offset + idx_hi];
    let result = gs_butterfly(lo, hi, omega);

    polys[poly_offset + idx_lo] = result.x;
    polys[poly_offset + idx_hi] = result.y;
}

// ============================================================================
// Polynomial Arithmetic Kernels
// ============================================================================

@group(0) @binding(3) var<storage, read> poly_a: array<u32>;
@group(0) @binding(4) var<storage, read> poly_b: array<u32>;
@group(0) @binding(5) var<storage, read_write> poly_result: array<u32>;

// Pointwise multiplication in NTT domain
@compute @workgroup_size(256)
fn mldsa_poly_mul_ntt(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_idx = gid.y;
    let coeff_idx = gid.x;

    if (batch_idx >= params.batch_size || coeff_idx >= MLDSA_N) {
        return;
    }

    let idx = batch_idx * MLDSA_N + coeff_idx;
    poly_result[idx] = mont_mul(poly_a[idx], poly_b[idx]);
}

// Polynomial addition
@compute @workgroup_size(256)
fn mldsa_poly_add(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_idx = gid.y;
    let coeff_idx = gid.x;

    if (batch_idx >= params.batch_size || coeff_idx >= MLDSA_N) {
        return;
    }

    let idx = batch_idx * MLDSA_N + coeff_idx;
    poly_result[idx] = mod_add(poly_a[idx], poly_b[idx]);
}

// Polynomial subtraction
@compute @workgroup_size(256)
fn mldsa_poly_sub(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_idx = gid.y;
    let coeff_idx = gid.x;

    if (batch_idx >= params.batch_size || coeff_idx >= MLDSA_N) {
        return;
    }

    let idx = batch_idx * MLDSA_N + coeff_idx;
    poly_result[idx] = mod_sub(poly_a[idx], poly_b[idx]);
}

// ============================================================================
// ML-DSA Verification Core Kernels
// ============================================================================

@group(0) @binding(6) var<storage, read> w_approx: array<u32>;
@group(0) @binding(7) var<storage, read> hints: array<u32>;
@group(0) @binding(8) var<storage, read_write> w1_prime: array<u32>;

// Hint expansion: use hint to recover w1
@compute @workgroup_size(256)
fn mldsa_verify_hint(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_idx = gid.y;
    let coeff_idx = gid.x;

    if (batch_idx >= params.batch_size || coeff_idx >= MLDSA_N * MLDSA_K) {
        return;
    }

    let idx = batch_idx * MLDSA_N * MLDSA_K + coeff_idx;
    let hint_word_idx = idx / 32u;
    let hint_bit_idx = idx % 32u;

    let hint = (hints[hint_word_idx] >> hint_bit_idx) & 1u;
    let w = w_approx[idx];

    // UseHint algorithm from FIPS 204
    var w1 = i32(w) >> 19;  // HighBits
    if (hint != 0u) {
        w1 = (w1 + 1) % 16;  // Adjust based on hint
    }

    w1_prime[idx] = u32(w1 & 0xF);
}

// ============================================================================
// Verification Result Checking
// ============================================================================

@group(0) @binding(9) var<storage, read> c_computed: array<u32>;
@group(0) @binding(10) var<storage, read> c_expected: array<u32>;
@group(0) @binding(11) var<storage, read_write> verify_results: array<u32>;

// Check if verification passed for a signature
@compute @workgroup_size(256)
fn mldsa_verify_check(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let tid = gid.x;
    if (tid >= params.batch_size) {
        return;
    }

    // Compare challenge hashes (32 coefficients)
    var match_val = 1u;
    for (var i = 0u; i < 32u; i++) {
        if (c_computed[tid * 32u + i] != c_expected[tid * 32u + i]) {
            match_val = 0u;
            break;
        }
    }

    verify_results[tid] = match_val;
}

// ============================================================================
// Matrix-Vector Multiplication for ML-DSA
// ============================================================================

@group(0) @binding(12) var<storage, read> matrix_A: array<u32>;
@group(0) @binding(13) var<storage, read> vector_s: array<u32>;
@group(0) @binding(14) var<storage, read_write> mat_vec_result: array<u32>;

// Compute A * s where A is k x l matrix of polynomials in NTT domain
@compute @workgroup_size(64, 1, 1)
fn mldsa_matrix_vec_mul(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let coeff_idx = gid.x;
    let row_idx = gid.y;
    let batch_idx = gid.z;

    if (batch_idx >= params.batch_size || row_idx >= MLDSA_K || coeff_idx >= MLDSA_N) {
        return;
    }

    // Accumulate A[row, col] * s[col] for all columns
    var acc = 0u;
    for (var col = 0u; col < MLDSA_L; col++) {
        let A_idx = (row_idx * MLDSA_L + col) * MLDSA_N + coeff_idx;
        let s_idx = batch_idx * MLDSA_L * MLDSA_N + col * MLDSA_N + coeff_idx;

        let prod = mont_mul(matrix_A[A_idx], vector_s[s_idx]);
        acc = mod_add(acc, prod);
    }

    let result_idx = batch_idx * MLDSA_K * MLDSA_N + row_idx * MLDSA_N + coeff_idx;
    mat_vec_result[result_idx] = acc;
}

// ============================================================================
// Fused NTT for Small Batches (workgroup-local)
// ============================================================================

var<workgroup> shared_poly: array<u32, 256>;

@compute @workgroup_size(128)
fn mldsa_ntt_forward_fused(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let batch_idx = wgid.x;
    if (batch_idx >= params.batch_size) {
        return;
    }

    let poly_offset = batch_idx * MLDSA_N;

    // Load polynomial to shared memory
    if (tid < MLDSA_N) {
        shared_poly[tid] = polys[poly_offset + tid];
    }
    if (tid + 128u < MLDSA_N) {
        shared_poly[tid + 128u] = polys[poly_offset + tid + 128u];
    }
    workgroupBarrier();

    // Perform all NTT stages
    for (var s = 0u; s < MLDSA_LOG_N; s++) {
        let m = 1u << s;
        let t = MLDSA_N >> (s + 1u);

        for (var butterfly = tid; butterfly < MLDSA_N / 2u; butterfly += 128u) {
            let i = butterfly / t;
            let j = butterfly % t;
            let idx_lo = (i << (MLDSA_LOG_N - s)) + j;
            let idx_hi = idx_lo + t;

            let omega = twiddles[m + i];
            let lo = shared_poly[idx_lo];
            let hi = shared_poly[idx_hi];

            let result = ct_butterfly(lo, hi, omega);
            shared_poly[idx_lo] = result.x;
            shared_poly[idx_hi] = result.y;
        }
        workgroupBarrier();
    }

    // Write back
    if (tid < MLDSA_N) {
        polys[poly_offset + tid] = shared_poly[tid];
    }
    if (tid + 128u < MLDSA_N) {
        polys[poly_offset + tid + 128u] = shared_poly[tid + 128u];
    }
}

// ============================================================================
// Batch Verification Reduction
// ============================================================================

var<workgroup> shared_count: array<u32, 256>;

@compute @workgroup_size(256)
fn mldsa_reduce_results(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    // Load verification results
    var local_count = 0u;
    for (var i = tid; i < params.batch_size; i += 256u) {
        local_count += verify_results[i];
    }
    shared_count[tid] = local_count;
    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_count[tid] += shared_count[tid + stride];
        }
        workgroupBarrier();
    }

    // Write final count
    if (tid == 0u) {
        verify_results[0] = shared_count[0];  // Store valid count in first element
    }
}
