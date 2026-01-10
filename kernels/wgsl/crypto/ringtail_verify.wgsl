// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Ringtail Lattice-Based Threshold Signature Verification - WebGPU WGSL Kernel
// Batch verification of Ringtail signatures with polynomial norm checks
//
// Ringtail Parameters:
//   n = 256 (polynomial degree)
//   q = 8380417 (prime modulus)
//   m = 8 (public key matrix rows)
//   l = 7 (public key matrix columns)

// ============================================================================
// Ringtail Parameters
// ============================================================================

const RING_N: u32 = 256u;
const RING_Q: u32 = 8380417u;
const RING_Q_INV: u32 = 58728449u;   // -q^{-1} mod 2^32
const VEC_M: u32 = 8u;               // Matrix rows
const VEC_N: u32 = 7u;               // Matrix columns
const N_INV: u32 = 8347649u;         // N^{-1} mod q
const CHALLENGE_WEIGHT: u32 = 60u;   // Hamming weight of challenge

// ============================================================================
// Modular Arithmetic
// ============================================================================

// Montgomery reduction: compute (a * b * R^{-1}) mod q
fn mont_reduce(a: u64) -> u32 {
    let t = (u32(a) * RING_Q_INV);
    let u = a + u64(t) * u64(RING_Q);
    var result = u32(u >> 32u);
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

// Montgomery multiplication
fn mod_mul(a: u32, b: u32) -> u32 {
    return mont_reduce(u64(a) * u64(b));
}

// Center-reduce coefficient to signed representation
fn center_reduce(a: u32) -> i32 {
    let t = i32(a);
    let half_q = i32(RING_Q >> 1u);
    if (t > half_q) {
        return t - i32(RING_Q);
    }
    return t;
}

// Absolute value for i32
fn abs_i32(x: i32) -> i32 {
    if (x >= 0) {
        return x;
    }
    return -x;
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
// Bindings
// ============================================================================

struct VerifyParams {
    batch_size: u32,
    beta_bound: i32,      // Bound for z vector norm
    delta_bound: i32,     // Bound for Delta vector norm
    num_shares: u32,
    round_bits: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

// Signature storage: c (challenge), z (response), Delta (correction)
// Layout: [batch][component][coeff]
@group(0) @binding(0) var<storage, read> signatures_c: array<u32>;      // batch * RING_N
@group(0) @binding(1) var<storage, read> signatures_z: array<u32>;      // batch * VEC_N * RING_N
@group(0) @binding(2) var<storage, read> signatures_delta: array<u32>;  // batch * VEC_M * RING_N

// Public key storage: A matrix and bTilde
@group(0) @binding(3) var<storage, read> pk_A: array<u32>;              // VEC_M * VEC_N * RING_N
@group(0) @binding(4) var<storage, read> pk_btilde: array<u32>;         // batch * VEC_M * RING_N

// NTT twiddle factors
@group(0) @binding(5) var<storage, read> ntt_twiddles: array<u32>;      // RING_N
@group(0) @binding(6) var<storage, read> inv_twiddles: array<u32>;      // RING_N

// Verification results
@group(0) @binding(7) var<storage, read_write> verify_results: array<u32>;

// Parameters
@group(0) @binding(8) var<uniform> params: VerifyParams;

// ============================================================================
// Workgroup Shared Memory
// ============================================================================

var<workgroup> shared_poly: array<u32, 256>;
var<workgroup> shared_temp: array<u32, 256>;

// ============================================================================
// Kernel 1: Check Z Vector Norm Bounds
// ============================================================================

@compute @workgroup_size(256)
fn ringtail_check_z_norm(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_idx = gid.y;
    let vec_idx = gid.x / RING_N;
    let coeff_idx = gid.x % RING_N;

    if (batch_idx >= params.batch_size || vec_idx >= VEC_N) {
        return;
    }

    let idx = batch_idx * VEC_N * RING_N + vec_idx * RING_N + coeff_idx;
    let coeff = center_reduce(signatures_z[idx]);
    let abs_coeff = abs_i32(coeff);

    // Use atomic to track if any coefficient exceeds bound
    if (abs_coeff > params.beta_bound) {
        // Mark as invalid - atomicMin would be ideal but use store for simplicity
        verify_results[batch_idx] = 0u;
    }
}

// ============================================================================
// Kernel 2: Check Delta Vector Norm Bounds
// ============================================================================

@compute @workgroup_size(256)
fn ringtail_check_delta_norm(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_idx = gid.y;
    let vec_idx = gid.x / RING_N;
    let coeff_idx = gid.x % RING_N;

    if (batch_idx >= params.batch_size || vec_idx >= VEC_M) {
        return;
    }

    let idx = batch_idx * VEC_M * RING_N + vec_idx * RING_N + coeff_idx;
    let coeff = center_reduce(signatures_delta[idx]);
    let abs_coeff = abs_i32(coeff);

    if (abs_coeff > params.delta_bound) {
        verify_results[batch_idx] = 0u;
    }
}

// ============================================================================
// Kernel 3: Check Challenge Format (sparse with +/-1 coefficients)
// ============================================================================

@group(1) @binding(0) var<storage, read_write> challenge_weight_count: array<atomic<u32>>;

@compute @workgroup_size(256)
fn ringtail_check_challenge_format(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32
) {
    let batch_idx = gid.y;
    let coeff_idx = gid.x;

    if (batch_idx >= params.batch_size || coeff_idx >= RING_N) {
        return;
    }

    let c_idx = batch_idx * RING_N + coeff_idx;
    let coeff = signatures_c[c_idx];

    // Count non-zero coefficients and check they are +/-1
    if (coeff != 0u) {
        // Valid coefficients: 1 or q-1 (representing -1)
        if (coeff != 1u && coeff != RING_Q - 1u) {
            verify_results[batch_idx] = 0u;
        } else {
            // Increment weight counter
            atomicAdd(&challenge_weight_count[batch_idx], 1u);
        }
    }
}

// ============================================================================
// Kernel 4: Forward NTT for Z Vector (Staged)
// ============================================================================

@group(1) @binding(1) var<storage, read_write> z_ntt: array<u32>;
@group(1) @binding(2) var<uniform> ntt_stage: u32;

@compute @workgroup_size(128)
fn ringtail_ntt_forward_z(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_idx = gid.z;
    let vec_idx = gid.y;
    let butterfly_idx = gid.x;

    if (batch_idx >= params.batch_size || vec_idx >= VEC_N) {
        return;
    }

    let stage = ntt_stage;
    let m = 1u << stage;
    let t = RING_N >> (stage + 1u);
    let num_butterflies = RING_N >> 1u;

    if (butterfly_idx >= num_butterflies) {
        return;
    }

    let i = butterfly_idx / t;
    let j = butterfly_idx % t;
    let idx_lo = (i << (8u - stage)) + j;  // 8 = log2(256)
    let idx_hi = idx_lo + t;

    let base_offset = batch_idx * VEC_N * RING_N + vec_idx * RING_N;
    let omega = ntt_twiddles[m + i];

    let lo = z_ntt[base_offset + idx_lo];
    let hi = z_ntt[base_offset + idx_hi];
    let result = ct_butterfly(lo, hi, omega);

    z_ntt[base_offset + idx_lo] = result.x;
    z_ntt[base_offset + idx_hi] = result.y;
}

// ============================================================================
// Kernel 5: Matrix-Vector Multiplication A * z (in NTT domain)
// ============================================================================

@group(1) @binding(3) var<storage, read_write> Az_result: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn ringtail_matrix_vec_mul(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let coeff_idx = gid.x;
    let row_idx = gid.y;
    let batch_idx = gid.z;

    if (batch_idx >= params.batch_size || row_idx >= VEC_M || coeff_idx >= RING_N) {
        return;
    }

    // Accumulate A[row, col] * z[col] for all columns
    var acc = 0u;
    for (var col = 0u; col < VEC_N; col++) {
        let A_idx = (row_idx * VEC_N + col) * RING_N + coeff_idx;
        let z_idx = batch_idx * VEC_N * RING_N + col * RING_N + coeff_idx;

        let prod = mod_mul(pk_A[A_idx], z_ntt[z_idx]);
        acc = mod_add(acc, prod);
    }

    let result_idx = batch_idx * VEC_M * RING_N + row_idx * RING_N + coeff_idx;
    Az_result[result_idx] = acc;
}

// ============================================================================
// Kernel 6: Inverse NTT for Az Result (Staged)
// ============================================================================

@compute @workgroup_size(128)
fn ringtail_ntt_inverse_Az(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_idx = gid.z;
    let vec_idx = gid.y;
    let butterfly_idx = gid.x;

    if (batch_idx >= params.batch_size || vec_idx >= VEC_M) {
        return;
    }

    let stage = ntt_stage;
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

    let base_offset = batch_idx * VEC_M * RING_N + vec_idx * RING_N;
    let omega = inv_twiddles[m + i];

    let lo = Az_result[base_offset + idx_lo];
    let hi = Az_result[base_offset + idx_hi];
    let result = gs_butterfly(lo, hi, omega);

    Az_result[base_offset + idx_lo] = result.x;
    Az_result[base_offset + idx_hi] = result.y;
}

// ============================================================================
// Kernel 7: Scale by N^{-1} after Inverse NTT
// ============================================================================

@compute @workgroup_size(256)
fn ringtail_scale_ninv(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_idx = gid.y;
    let idx = gid.x;

    if (batch_idx >= params.batch_size || idx >= VEC_M * RING_N) {
        return;
    }

    let offset = batch_idx * VEC_M * RING_N + idx;
    Az_result[offset] = mod_mul(Az_result[offset], N_INV);
}

// ============================================================================
// Kernel 8: Compute c * bTilde
// ============================================================================

@group(1) @binding(4) var<storage, read> c_ntt: array<u32>;
@group(1) @binding(5) var<storage, read_write> c_btilde_result: array<u32>;

@compute @workgroup_size(256)
fn ringtail_mul_c_btilde(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_idx = gid.z;
    let vec_idx = gid.y;
    let coeff_idx = gid.x;

    if (batch_idx >= params.batch_size || vec_idx >= VEC_M || coeff_idx >= RING_N) {
        return;
    }

    let c_idx = batch_idx * RING_N + coeff_idx;
    let btilde_idx = batch_idx * VEC_M * RING_N + vec_idx * RING_N + coeff_idx;
    let result_idx = batch_idx * VEC_M * RING_N + vec_idx * RING_N + coeff_idx;

    // Both c and btilde should be in NTT domain
    c_btilde_result[result_idx] = mod_mul(c_ntt[c_idx], pk_btilde[btilde_idx]);
}

// ============================================================================
// Kernel 9: Verify Equation Az = c*bTilde + Delta
// ============================================================================

@compute @workgroup_size(256)
fn ringtail_verify_equation(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_idx = gid.y;
    let idx = gid.x;

    let vec_idx = idx / RING_N;
    let coeff_idx = idx % RING_N;

    if (batch_idx >= params.batch_size || vec_idx >= VEC_M) {
        return;
    }

    let offset = batch_idx * VEC_M * RING_N + vec_idx * RING_N + coeff_idx;

    // Compute difference: Az - c*bTilde - Delta
    let Az_val = Az_result[offset];
    let c_btilde_val = c_btilde_result[offset];
    let delta_val = signatures_delta[offset];

    // diff = Az - c*bTilde - Delta
    var diff = mod_sub(Az_val, c_btilde_val);
    diff = mod_sub(diff, delta_val);

    // Check if residual is within rounding error bound
    let centered = center_reduce(diff);
    let abs_val = abs_i32(centered);

    if (abs_val > params.delta_bound) {
        verify_results[batch_idx] = 0u;
    }
}

// ============================================================================
// Kernel 10: Compute Polynomial Infinity Norm
// ============================================================================

@group(2) @binding(0) var<storage, read> norm_input_polys: array<u32>;
@group(2) @binding(1) var<storage, read_write> inf_norms: array<i32>;

var<workgroup> shared_max: array<i32, 256>;

@compute @workgroup_size(256)
fn ringtail_compute_inf_norm(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let poly_idx = wgid.x;
    let coeff_idx = tid;

    // Load and compute absolute value of centered coefficient
    let idx = poly_idx * RING_N + coeff_idx;
    let coeff = center_reduce(norm_input_polys[idx]);
    shared_max[tid] = abs_i32(coeff);
    workgroupBarrier();

    // Parallel reduction for maximum
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            if (shared_max[tid + stride] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + stride];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        inf_norms[poly_idx] = shared_max[0];
    }
}

// ============================================================================
// Kernel 11: Compute Polynomial L2 Norm Squared
// ============================================================================

@group(2) @binding(2) var<storage, read_write> l2_norms: array<u64>;

var<workgroup> shared_sum: array<u64, 256>;

@compute @workgroup_size(256)
fn ringtail_compute_l2_norm(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let poly_idx = wgid.x;
    let coeff_idx = tid;

    // Load and compute squared value
    let idx = poly_idx * RING_N + coeff_idx;
    let coeff = center_reduce(norm_input_polys[idx]);
    shared_sum[tid] = u64(coeff) * u64(coeff);
    workgroupBarrier();

    // Parallel reduction for sum
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        l2_norms[poly_idx] = shared_sum[0];
    }
}

// ============================================================================
// Kernel 12: Reconstruct Public Key from Shares (Lagrange Interpolation)
// ============================================================================

@group(2) @binding(3) var<storage, read> pk_shares: array<u32>;
@group(2) @binding(4) var<storage, read> lagrange_coeffs: array<u32>;
@group(2) @binding(5) var<storage, read_write> reconstructed_pk: array<u32>;

@compute @workgroup_size(256)
fn ringtail_reconstruct_pk(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= VEC_M * RING_N) {
        return;
    }

    let poly_idx = idx / RING_N;
    let coeff_idx = idx % RING_N;

    // Sum lambda_i * share_i for all shares
    var sum = 0u;
    for (var s = 0u; s < params.num_shares; s++) {
        let lambda = lagrange_coeffs[s];
        let share_idx = s * VEC_M * RING_N + poly_idx * RING_N + coeff_idx;
        let share_coeff = pk_shares[share_idx];
        sum = mod_add(sum, mod_mul(lambda, share_coeff));
    }

    reconstructed_pk[idx] = sum;
}

// ============================================================================
// Kernel 13: Batch Apply Rounding
// ============================================================================

@group(2) @binding(6) var<storage, read_write> round_polys: array<u32>;

@compute @workgroup_size(256)
fn ringtail_batch_round(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;

    // round_bits comes from params
    let d = params.round_bits;
    let mask = (1u << d) - 1u;
    let half_val = 1u << (d - 1u);

    let a = round_polys[idx];
    let rounded = (a + half_val) & ~mask;
    round_polys[idx] = rounded % RING_Q;
}

// ============================================================================
// Kernel 14: Challenge Comparison
// ============================================================================

@group(2) @binding(7) var<storage, read> expected_challenges: array<u32>;
@group(2) @binding(8) var<storage, read> computed_challenges: array<u32>;

@compute @workgroup_size(256)
fn ringtail_verify_challenge(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_idx = gid.y;
    let coeff_idx = gid.x;

    if (batch_idx >= params.batch_size || coeff_idx >= RING_N) {
        return;
    }

    let idx = batch_idx * RING_N + coeff_idx;

    if (expected_challenges[idx] != computed_challenges[idx]) {
        verify_results[batch_idx] = 0u;
    }
}

// ============================================================================
// Kernel 15: Initialize Verification Results
// ============================================================================

@compute @workgroup_size(256)
fn ringtail_init_verify_results(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= params.batch_size) {
        return;
    }
    // Initialize all as valid (1), kernels will set to 0 if invalid
    verify_results[idx] = 1u;
}

// ============================================================================
// Kernel 16: Count Valid Signatures (Reduction)
// ============================================================================

var<workgroup> shared_count: array<u32, 256>;

@compute @workgroup_size(256)
fn ringtail_count_valid(
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

    // Write final count to first element
    if (tid == 0u) {
        verify_results[0] = shared_count[0];
    }
}
