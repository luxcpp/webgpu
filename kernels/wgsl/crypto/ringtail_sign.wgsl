// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Ringtail Lattice-Based Threshold Signatures - WebGPU WGSL Kernel
// GPU-accelerated MLWE-based threshold signing operations
//
// Parameters from Ringtail specification:
// - Ring dimension N = 256 (power of 2 for NTT)
// - Modulus Q = 8380417 (23-bit prime, NTT-friendly)
// - Vector dimensions: M = 8, N_vec = 7
// - Gaussian parameter sigma for rejection sampling

// ============================================================================
// Ringtail Parameters
// ============================================================================

const RING_N: u32 = 256u;          // Polynomial degree
const RING_Q: u32 = 8380417u;      // Modulus (23-bit prime)
const LOG_N: u32 = 8u;             // log2(RING_N)
const VEC_M: u32 = 8u;             // Public key rows
const VEC_N: u32 = 7u;             // Secret key / signature dimension

// NTT parameters
const OMEGA: u32 = 1753u;          // Primitive 512th root of unity mod Q
const N_INV: u32 = 8347649u;       // Inverse of N=256 mod Q

// Rejection sampling bound
const REJECTION_BOUND: i32 = 262144;  // 2^18

// Montgomery constant: -Q^{-1} mod 2^32
const RING_Q_INV: u32 = 58728449u;

// ============================================================================
// Modular Arithmetic
// ============================================================================

// Montgomery reduction
fn mont_reduce(a: u32) -> u32 {
    let t = (a * RING_Q_INV) & 0xFFFFFFFFu;
    let u = a + t * RING_Q;
    var result = u >> 32u;
    if (result >= RING_Q) {
        result = result - RING_Q;
    }
    return result;
}

// Modular addition
fn mod_add(a: u32, b: u32) -> u32 {
    var sum = a + b;
    if (sum >= RING_Q) {
        sum = sum - RING_Q;
    }
    return sum;
}

// Modular subtraction
fn mod_sub(a: u32, b: u32) -> u32 {
    if (a >= b) {
        return a - b;
    }
    return a + RING_Q - b;
}

// Modular multiplication with Montgomery
fn mod_mul(a: u32, b: u32) -> u32 {
    let prod = u64(a) * u64(b);
    let t = u32(prod) * RING_Q_INV;
    let u = prod + u64(t) * u64(RING_Q);
    var result = u32(u >> 32u);
    if (result >= RING_Q) {
        result = result - RING_Q;
    }
    return result;
}

// Modular negation
fn mod_neg(a: u32) -> u32 {
    if (a == 0u) {
        return 0u;
    }
    return RING_Q - a;
}

// Center reduction: map [0, Q) to [-(Q-1)/2, (Q-1)/2]
fn center_reduce(a: u32) -> i32 {
    let t = i32(a);
    let half_q = i32(RING_Q >> 1u);
    if (t > half_q) {
        return t - i32(RING_Q);
    }
    return t;
}

// ============================================================================
// NTT Butterfly Operations
// ============================================================================

// Cooley-Tukey butterfly for forward NTT
fn ct_butterfly(lo: u32, hi: u32, omega: u32) -> vec2<u32> {
    let omega_hi = mod_mul(hi, omega);
    let new_lo = mod_add(lo, omega_hi);
    let new_hi = mod_sub(lo, omega_hi);
    return vec2<u32>(new_lo, new_hi);
}

// Gentleman-Sande butterfly for inverse NTT
fn gs_butterfly(lo: u32, hi: u32, omega: u32) -> vec2<u32> {
    let sum = mod_add(lo, hi);
    let diff = mod_sub(lo, hi);
    let new_hi = mod_mul(diff, omega);
    return vec2<u32>(sum, new_hi);
}

// ============================================================================
// Polynomial Types (stored as arrays of coefficients)
// ============================================================================

struct RingtailParams {
    num_participants: u32,
    threshold: u32,
    batch_size: u32,
    ntt_stage: u32,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> secret_shares: array<u32>;
@group(0) @binding(1) var<storage, read> public_key_A: array<u32>;
@group(0) @binding(2) var<storage, read> commitment_y: array<u32>;
@group(0) @binding(3) var<storage, read> challenge: array<u32>;
@group(0) @binding(4) var<storage, read> randomness: array<u32>;
@group(0) @binding(5) var<storage, read> ntt_twiddles: array<u32>;
@group(0) @binding(6) var<storage, read_write> shares_out: array<u32>;
@group(0) @binding(7) var<uniform> params: RingtailParams;

// ============================================================================
// NTT Kernels
// ============================================================================

@group(0) @binding(8) var<storage, read_write> polys: array<u32>;

// Batch NTT forward transform
@compute @workgroup_size(128)
fn ringtail_batch_ntt_forward(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let poly_idx = gid.y;
    let butterfly_idx = gid.x;

    if (poly_idx >= params.batch_size) {
        return;
    }

    let stage = params.ntt_stage;
    let m = 1u << stage;
    let t = RING_N >> (stage + 1u);
    let num_butterflies = RING_N >> 1u;

    if (butterfly_idx >= num_butterflies) {
        return;
    }

    let i = butterfly_idx / t;
    let j = butterfly_idx % t;
    let idx_lo = (i << (LOG_N - stage)) + j;
    let idx_hi = idx_lo + t;

    let poly_offset = poly_idx * RING_N;
    let omega = ntt_twiddles[m + i];

    let lo = polys[poly_offset + idx_lo];
    let hi = polys[poly_offset + idx_hi];
    let result = ct_butterfly(lo, hi, omega);

    polys[poly_offset + idx_lo] = result.x;
    polys[poly_offset + idx_hi] = result.y;
}

// Batch NTT inverse transform
@compute @workgroup_size(128)
fn ringtail_batch_ntt_inverse(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let poly_idx = gid.y;
    let butterfly_idx = gid.x;

    if (poly_idx >= params.batch_size) {
        return;
    }

    let stage = params.ntt_stage;
    let m = RING_N >> (stage + 1u);
    let t = 1u << stage;
    let num_butterflies = RING_N >> 1u;

    if (butterfly_idx >= num_butterflies) {
        return;
    }

    let i = butterfly_idx / t;
    let j = butterfly_idx % t;
    let idx_lo = (i << (stage + 1u)) + j;
    let idx_hi = idx_lo + t;

    let poly_offset = poly_idx * RING_N;
    let omega = ntt_twiddles[RING_N + m + i];  // Inverse twiddles in second half

    let lo = polys[poly_offset + idx_lo];
    let hi = polys[poly_offset + idx_hi];
    let result = gs_butterfly(lo, hi, omega);

    polys[poly_offset + idx_lo] = result.x;
    polys[poly_offset + idx_hi] = result.y;
}

// Scale by N^-1 after inverse NTT
@compute @workgroup_size(256)
fn ringtail_ntt_scale(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= params.batch_size * RING_N) {
        return;
    }

    polys[idx] = mod_mul(polys[idx], N_INV);
}

// ============================================================================
// Polynomial Arithmetic
// ============================================================================

@group(0) @binding(9) var<storage, read> poly_a: array<u32>;
@group(0) @binding(10) var<storage, read> poly_b: array<u32>;
@group(0) @binding(11) var<storage, read_write> poly_result: array<u32>;

// Pointwise multiplication in NTT domain
@compute @workgroup_size(256)
fn ringtail_poly_mul_ntt(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= params.batch_size * RING_N) {
        return;
    }

    poly_result[idx] = mod_mul(poly_a[idx], poly_b[idx]);
}

// Polynomial addition
@compute @workgroup_size(256)
fn ringtail_poly_add(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= params.batch_size * RING_N) {
        return;
    }

    poly_result[idx] = mod_add(poly_a[idx], poly_b[idx]);
}

// Polynomial subtraction
@compute @workgroup_size(256)
fn ringtail_poly_sub(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= params.batch_size * RING_N) {
        return;
    }

    poly_result[idx] = mod_sub(poly_a[idx], poly_b[idx]);
}

// ============================================================================
// Signature Share Generation
// ============================================================================

// Generate signature share: z_i = r_i + c * s_i
@compute @workgroup_size(256)
fn ringtail_generate_share(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let participant_idx = gid.y;
    let coeff_idx = gid.x;

    if (participant_idx >= params.num_participants || coeff_idx >= RING_N) {
        return;
    }

    // For each component j in VEC_N
    for (var j = 0u; j < VEC_N; j++) {
        let s_i = secret_shares[participant_idx * RING_N + coeff_idx];
        let c = challenge[coeff_idx];
        let r_ij = randomness[(participant_idx * VEC_N + j) * RING_N + coeff_idx];

        // z_ij = r_ij + c * s_i (in NTT domain)
        let cs = mod_mul(c, s_i);
        let z_ij = mod_add(r_ij, cs);

        // Store in output
        let out_idx = (participant_idx * VEC_N + j) * RING_N + coeff_idx;
        shares_out[out_idx] = z_ij;
    }
}

// ============================================================================
// Share Aggregation with Lagrange Interpolation
// ============================================================================

@group(0) @binding(12) var<storage, read> lagrange_coeffs: array<u32>;
@group(0) @binding(13) var<storage, read_write> aggregated_z: array<u32>;
@group(0) @binding(14) var<storage, read_write> aggregated_delta: array<u32>;

// Aggregate signature shares using Lagrange coefficients
@compute @workgroup_size(256)
fn ringtail_aggregate_shares(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let coeff_idx = gid.x;
    if (coeff_idx >= RING_N) {
        return;
    }

    // Aggregate z = sum(lambda_i * z_i) for each component
    for (var j = 0u; j < VEC_N; j++) {
        var acc = 0u;

        for (var p = 0u; p < params.num_participants; p++) {
            let lambda = lagrange_coeffs[p];
            let z_pi = shares_out[(p * VEC_N + j) * RING_N + coeff_idx];
            let weighted = mod_mul(lambda, z_pi);
            acc = mod_add(acc, weighted);
        }

        aggregated_z[j * RING_N + coeff_idx] = acc;
    }
}

// ============================================================================
// Rejection Sampling Check
// ============================================================================

@group(0) @binding(15) var<storage, read> z_vectors: array<u32>;
@group(0) @binding(16) var<storage, read_write> valid_flags: array<u32>;

// Check if z vector passes rejection sampling bound
@compute @workgroup_size(256)
fn ringtail_check_rejection(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let sig_idx = gid.x;
    if (sig_idx >= params.batch_size) {
        return;
    }

    var valid = 1u;

    // Check each polynomial in the z vector
    for (var i = 0u; i < VEC_N && valid != 0u; i++) {
        for (var j = 0u; j < RING_N && valid != 0u; j++) {
            let idx = (sig_idx * VEC_N + i) * RING_N + j;
            let coeff = center_reduce(z_vectors[idx]);

            if (coeff > REJECTION_BOUND || coeff < -REJECTION_BOUND) {
                valid = 0u;
            }
        }
    }

    valid_flags[sig_idx] = valid;
}

// ============================================================================
// Polynomial Norm Computation
// ============================================================================

@group(0) @binding(17) var<storage, read_write> norms: array<i32>;

// Compute infinity norm of polynomials
@compute @workgroup_size(256)
fn ringtail_compute_norms(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let poly_idx = gid.x;
    if (poly_idx >= params.batch_size) {
        return;
    }

    var max_val: i32 = 0;

    for (var i = 0u; i < RING_N; i++) {
        let coeff = center_reduce(polys[poly_idx * RING_N + i]);
        let abs_coeff = select(-coeff, coeff, coeff >= 0);
        if (abs_coeff > max_val) {
            max_val = abs_coeff;
        }
    }

    norms[poly_idx] = max_val;
}

// ============================================================================
// Matrix-Vector Multiplication (A * z)
// ============================================================================

// Compute A * z where A is M x N matrix of polynomials
@compute @workgroup_size(64)
fn ringtail_matrix_vec_mul(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let row_idx = gid.y;
    let coeff_idx = gid.x;

    if (row_idx >= VEC_M || coeff_idx >= RING_N) {
        return;
    }

    // Accumulate A[row, col] * z[col] for all columns
    var acc = 0u;
    for (var col = 0u; col < VEC_N; col++) {
        let A_idx = (row_idx * VEC_N + col) * RING_N + coeff_idx;
        let z_idx = col * RING_N + coeff_idx;

        let prod = mod_mul(public_key_A[A_idx], aggregated_z[z_idx]);
        acc = mod_add(acc, prod);
    }

    poly_result[row_idx * RING_N + coeff_idx] = acc;
}

// ============================================================================
// Lagrange Coefficient Combination (per coefficient)
// ============================================================================

@group(0) @binding(18) var<storage, read> participant_ids: array<u32>;
@group(0) @binding(19) var<storage, read_write> combined_secret: array<u32>;

// Combine shares using Lagrange interpolation at x=0
@compute @workgroup_size(256)
fn ringtail_lagrange_combine(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let coeff_idx = gid.x;
    if (coeff_idx >= RING_N) {
        return;
    }

    var sum = 0u;

    for (var p = 0u; p < params.num_participants; p++) {
        let lambda = lagrange_coeffs[p];
        // Get p-th participant's first z polynomial coefficient
        let z_coeff = shares_out[p * VEC_N * RING_N + coeff_idx];
        sum = mod_add(sum, mod_mul(lambda, z_coeff));
    }

    combined_secret[coeff_idx] = sum;
}

// ============================================================================
// Fused NTT for Small Batches
// ============================================================================

var<workgroup> shared_poly: array<u32, 256>;

@compute @workgroup_size(128)
fn ringtail_ntt_forward_fused(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let poly_idx = wgid.x;
    if (poly_idx >= params.batch_size) {
        return;
    }

    let poly_offset = poly_idx * RING_N;

    // Load polynomial to shared memory
    if (tid < RING_N) {
        shared_poly[tid] = polys[poly_offset + tid];
    }
    if (tid + 128u < RING_N) {
        shared_poly[tid + 128u] = polys[poly_offset + tid + 128u];
    }
    workgroupBarrier();

    // Perform all NTT stages
    for (var s = 0u; s < LOG_N; s++) {
        let m = 1u << s;
        let t = RING_N >> (s + 1u);

        for (var butterfly = tid; butterfly < RING_N / 2u; butterfly += 128u) {
            let i = butterfly / t;
            let j = butterfly % t;
            let idx_lo = (i << (LOG_N - s)) + j;
            let idx_hi = idx_lo + t;

            let omega = ntt_twiddles[m + i];
            let lo = shared_poly[idx_lo];
            let hi = shared_poly[idx_hi];

            let result = ct_butterfly(lo, hi, omega);
            shared_poly[idx_lo] = result.x;
            shared_poly[idx_hi] = result.y;
        }
        workgroupBarrier();
    }

    // Write back
    if (tid < RING_N) {
        polys[poly_offset + tid] = shared_poly[tid];
    }
    if (tid + 128u < RING_N) {
        polys[poly_offset + tid + 128u] = shared_poly[tid + 128u];
    }
}
