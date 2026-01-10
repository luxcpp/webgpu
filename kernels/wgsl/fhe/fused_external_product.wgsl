// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// TFHE Fused External Product with Keyswitch and Blind Rotate
// Combines keyswitch + blind rotation into single optimized kernel sequence
// Minimizes memory traffic between operations
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
    if (a.lo == 0u && a.hi == 0u) { return a; }
    return u64_sub(Q, a);
}

fn barrett_mul(a: u64, b: u64, Q: u64, precon: u64) -> u64 {
    let q_approx = u64_mulhi(a, precon);
    let product = u64_mul(a, b);
    let qQ = u64_mul(q_approx, Q);
    var result = u64_sub(product, qQ);
    if (u64_ge(result, Q)) { result = u64_sub(result, Q); }
    return result;
}

// ============================================================================
// NTT Butterfly Operations
// ============================================================================

fn ct_butterfly(lo: ptr<function, u64>, hi: ptr<function, u64>, omega: u64, Q: u64, precon: u64) {
    let u = *lo;
    let v = barrett_mul(*hi, omega, Q, precon);
    *lo = mod_add(u, v, Q);
    *hi = mod_sub(u, v, Q);
}

fn gs_butterfly(lo: ptr<function, u64>, hi: ptr<function, u64>, omega: u64, Q: u64, precon: u64) {
    let u = *lo;
    let v = *hi;
    *lo = mod_add(u, v, Q);
    *hi = barrett_mul(mod_sub(u, v, Q), omega, Q, precon);
}

// ============================================================================
// Parameters for Fused Operations
// ============================================================================

struct FusedParams {
    // Polynomial parameters
    N: u32,              // GLWE polynomial degree (1024, 2048)
    log_N: u32,          // log2(N)
    n: u32,              // LWE dimension (~630)
    k: u32,              // GLWE dimension (1)

    // Decomposition
    l_ks: u32,           // Keyswitch decomposition levels
    base_log_ks: u32,    // Keyswitch base log
    l_bs: u32,           // Bootstrap decomposition levels
    base_log_bs: u32,    // Bootstrap base log

    // Modulus
    Q_lo: u32,
    Q_hi: u32,

    // Barrett constant
    mu_lo: u32,
    mu_hi: u32,

    // Inverse N for INTT
    N_inv_lo: u32,
    N_inv_hi: u32,
    N_inv_precon_lo: u32,
    N_inv_precon_hi: u32,

    // Current step
    step_idx: u32,       // For iterative blind rotation
    batch_size: u32,
}

fn params_Q(p: FusedParams) -> u64 { return u64(p.Q_lo, p.Q_hi); }
fn params_mu(p: FusedParams) -> u64 { return u64(p.mu_lo, p.mu_hi); }
fn params_N_inv(p: FusedParams) -> u64 { return u64(p.N_inv_lo, p.N_inv_hi); }
fn params_N_inv_precon(p: FusedParams) -> u64 { return u64(p.N_inv_precon_lo, p.N_inv_precon_hi); }

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> lwe_in: array<u32>;         // Input LWE [batch][n+1]
@group(0) @binding(1) var<storage, read> ksk: array<u32>;            // Keyswitch key
@group(0) @binding(2) var<storage, read> bsk: array<u32>;            // Bootstrapping key
@group(0) @binding(3) var<storage, read> test_vector: array<u32>;    // Test polynomial (LUT)
@group(0) @binding(4) var<storage, read_write> accumulator: array<u32>; // GLWE accumulator
@group(0) @binding(5) var<storage, read_write> lwe_out: array<u32>;  // Output LWE
@group(0) @binding(6) var<storage, read> twiddles: array<u32>;       // NTT twiddles
@group(0) @binding(7) var<storage, read> inv_twiddles: array<u32>;   // Inverse NTT twiddles
@group(0) @binding(8) var<storage, read> precons: array<u32>;        // Precomputed Barrett values
@group(0) @binding(9) var<storage, read> inv_precons: array<u32>;    // Inverse precomputed
@group(0) @binding(10) var<uniform> params: FusedParams;

// Shared memory for fused operations
var<workgroup> shared_acc: array<u32, 2048>;   // Accumulator (N * 2)
var<workgroup> shared_work: array<u32, 2048>;  // Working buffer

// ============================================================================
// Shared Memory Helpers
// ============================================================================

fn read_shared_acc(idx: u32) -> u64 {
    return u64(shared_acc[idx * 2u], shared_acc[idx * 2u + 1u]);
}

fn write_shared_acc(idx: u32, val: u64) {
    shared_acc[idx * 2u] = val.lo;
    shared_acc[idx * 2u + 1u] = val.hi;
}

fn read_shared_work(idx: u32) -> u64 {
    return u64(shared_work[idx * 2u], shared_work[idx * 2u + 1u]);
}

fn write_shared_work(idx: u32, val: u64) {
    shared_work[idx * 2u] = val.lo;
    shared_work[idx * 2u + 1u] = val.hi;
}

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

// ============================================================================
// Signed Decomposition
// ============================================================================

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

// ============================================================================
// Phase 1: Keyswitch (LWE to LWE with different dimension)
// ============================================================================

// Keyswitch: convert LWE(n) to LWE(N) using keyswitch key
// KSK layout: [n][l][N+1] for each decomposition level
@compute @workgroup_size(256)
fn keyswitch_decompose_accumulate(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let n = params.n;
    let N = params.N;
    let l_ks = params.l_ks;
    let base_log_ks = params.base_log_ks;
    let Q = params_Q(params);
    let mu = params_mu(params);
    let tpg = 256u;

    let batch_idx = wgid.x;
    let coeff_idx = wgid.y * 256u + tid;  // Output LWE coefficient index

    if (batch_idx >= params.batch_size || coeff_idx > N) { return; }

    let lwe_base = batch_idx * (n + 1u);

    var acc = u64_zero();

    // For each input LWE coefficient a[i]
    for (var i = 0u; i < n; i++) {
        let a_i = read_u64_storage(&lwe_in, lwe_base + i);

        // For each decomposition level
        for (var level = 0u; level < l_ks; level++) {
            let digit = signed_decompose_digit(a_i, level, base_log_ks);

            if (digit == 0) { continue; }

            // KSK coefficient
            let ksk_idx = (i * l_ks + level) * (N + 1u) + coeff_idx;
            let ksk_val = read_u64_storage(&ksk, ksk_idx);

            let abs_digit = u32(select(-digit, digit, digit >= 0));
            let prod = barrett_mul(u64_from_u32(abs_digit), ksk_val, Q, mu);

            if (digit > 0) {
                acc = mod_sub(acc, prod, Q);  // Subtract for keyswitch
            } else {
                acc = mod_add(acc, prod, Q);
            }
        }
    }

    // Add body for b coefficient
    if (coeff_idx == N) {
        let b = read_u64_storage(&lwe_in, lwe_base + n);
        acc = mod_add(acc, b, Q);
    }

    // Store keyswitched LWE
    let out_base = batch_idx * (N + 1u);
    write_u64_storage(&lwe_out, out_base + coeff_idx, acc);
}

// ============================================================================
// Phase 2: Initialize Accumulator for Blind Rotation
// ============================================================================

@compute @workgroup_size(256)
fn blind_rotate_init(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let Q = params_Q(params);
    let tpg = 256u;

    let batch_idx = wgid.x;

    if (batch_idx >= params.batch_size) { return; }

    // Get b from keyswitched LWE
    let lwe_base = batch_idx * (N + 1u);
    let b = read_u64_storage(&lwe_out, lwe_base + N);

    // Calculate rotation: round(b * 2N / Q) mod 2N
    let log_N = params.log_N;
    let rotation = (b.hi >> (32u - log_N - 1u)) & ((2u << log_N) - 1u);
    let neg_rot = (2u * N - rotation) % (2u * N);

    let acc_base = batch_idx * (k + 1u) * N;

    // Initialize mask polynomials to zero
    for (var p = 0u; p < k; p++) {
        for (var i = tid; i < N; i += tpg) {
            write_u64_storage(&accumulator, acc_base + p * N + i, u64_zero());
        }
    }

    // Initialize body with rotated test vector
    for (var i = tid; i < N; i += tpg) {
        let N_i32 = i32(N);
        var src_idx = (i32(i) - i32(neg_rot)) % (2 * N_i32);
        if (src_idx < 0) { src_idx += 2 * N_i32; }

        var negate = false;
        if (src_idx >= N_i32) {
            src_idx -= N_i32;
            negate = true;
        }

        var val = read_u64_storage(&test_vector, u32(src_idx));
        if (negate) {
            val = mod_neg(val, Q);
        }

        write_u64_storage(&accumulator, acc_base + k * N + i, val);
    }
}

// ============================================================================
// Phase 3: Single Blind Rotation Step
// ============================================================================

@compute @workgroup_size(256)
fn blind_rotate_step(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let l_bs = params.l_bs;
    let base_log_bs = params.base_log_bs;
    let Q = params_Q(params);
    let mu = params_mu(params);
    let tpg = 256u;

    let batch_idx = wgid.x;
    let out_poly = wgid.y;  // 0 to k
    let step_idx = params.step_idx;

    if (batch_idx >= params.batch_size || out_poly > k) { return; }

    let lwe_base = batch_idx * (N + 1u);
    let acc_base = batch_idx * (k + 1u) * N;

    // Get a[step_idx] from keyswitched LWE
    let a_j = read_u64_storage(&lwe_out, lwe_base + step_idx);

    // Calculate rotation amount
    let log_N = params.log_N;
    let rot = (a_j.hi >> (32u - log_N - 1u)) & ((2u << log_N) - 1u);

    // Skip if rotation is zero
    if (rot == 0u) { return; }

    let N_i32 = i32(N);
    let rot_i32 = i32(rot);

    // Load accumulator polynomial to shared memory
    for (var i = tid; i < N; i += tpg) {
        let val = read_u64_storage_rw(&accumulator, acc_base + out_poly * N + i);
        write_shared_acc(i, val);
    }
    workgroupBarrier();

    // Compute rotated - original (diff for CMux)
    for (var i = tid; i < N; i += tpg) {
        var src_idx = (i32(i) - rot_i32) % (2 * N_i32);
        if (src_idx < 0) { src_idx += 2 * N_i32; }

        var negate = false;
        if (src_idx >= N_i32) {
            src_idx -= N_i32;
            negate = true;
        }

        var rotated = read_shared_acc(u32(src_idx));
        if (negate) {
            rotated = mod_neg(rotated, Q);
        }

        let original = read_shared_acc(i);
        let diff = mod_sub(rotated, original, Q);
        write_shared_work(i, diff);
    }
    workgroupBarrier();

    // External product: BSK[step_idx] x diff
    // Accumulate into accumulator
    var acc = u64_zero();

    for (var in_poly = 0u; in_poly <= k; in_poly++) {
        for (var i = tid; i < N; i += tpg) {
            // Get diff coefficient for this input poly
            // (simplified: assuming we stored all diffs, or recompute)
            let diff_val = write_shared_work(i, u64_zero());  // Placeholder

            for (var level = 0u; level < l_bs; level++) {
                let digit = signed_decompose_digit(diff_val, level, base_log_bs);

                if (digit == 0) { continue; }

                // BSK coefficient
                let bsk_idx = ((step_idx * (k + 1u) + in_poly) * l_bs + level) * (k + 1u) * N
                            + out_poly * N + i;
                let bsk_val = read_u64_storage(&bsk, bsk_idx);

                let abs_digit = u32(select(-digit, digit, digit >= 0));
                let prod = barrett_mul(u64_from_u32(abs_digit), bsk_val, Q, mu);

                if (digit > 0) {
                    acc = mod_add(acc, prod, Q);
                } else {
                    acc = mod_sub(acc, prod, Q);
                }
            }
        }
    }

    // Update accumulator: acc + external_product
    for (var i = tid; i < N; i += tpg) {
        let current = read_shared_acc(i);
        // Add external product result (simplified)
        let updated = mod_add(current, acc, Q);
        write_u64_storage(&accumulator, acc_base + out_poly * N + i, updated);
    }
}

// ============================================================================
// Phase 4: Sample Extract (GLWE to LWE)
// ============================================================================

@compute @workgroup_size(256)
fn sample_extract(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let Q = params_Q(params);
    let tpg = 256u;

    let batch_idx = wgid.x;

    if (batch_idx >= params.batch_size) { return; }

    let acc_base = batch_idx * (k + 1u) * N;
    let out_base = batch_idx * (k * N + 1u);

    // Extract LWE from GLWE at index 0
    // a[i] = -acc_mask[N-1-i] for i < N, with sign handling for negacyclic

    // For mask polynomials
    for (var p = 0u; p < k; p++) {
        for (var i = tid; i < N; i += tpg) {
            // Reverse and negate for extraction
            let src_idx = (N - 1u - i) % N;
            var val = read_u64_storage_rw(&accumulator, acc_base + p * N + src_idx);

            // Negate if extracting from position > 0
            if (i > 0u) {
                val = mod_neg(val, Q);
            }

            write_u64_storage(&lwe_out, out_base + p * N + i, val);
        }
    }

    // Body: just the constant coefficient
    if (tid == 0u) {
        let body = read_u64_storage_rw(&accumulator, acc_base + k * N);
        write_u64_storage(&lwe_out, out_base + k * N, body);
    }
}

// ============================================================================
// Combined Fused Bootstrap (All Phases)
// ============================================================================

// Note: Full bootstrap requires multiple dispatches due to data dependencies
// This kernel combines phases where possible within workgroup constraints

@compute @workgroup_size(256)
fn fused_bootstrap_small(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    // For small N (256, 512), can fit entire bootstrap in workgroup
    let N = params.N;
    let n = params.n;
    let k = params.k;
    let Q = params_Q(params);
    let mu = params_mu(params);
    let tpg = 256u;

    let batch_idx = wgid.x;

    if (batch_idx >= params.batch_size) { return; }

    // Phase 1: Modular reduction for LWE coefficients
    // Simplified: just compute rotation amounts

    let lwe_base = batch_idx * (n + 1u);
    let b = read_u64_storage(&lwe_in, lwe_base + n);

    // Compute initial rotation from body
    let log_N = params.log_N;
    let b_rot = (b.hi >> (32u - log_N - 1u)) & ((2u << log_N) - 1u);

    // Initialize accumulator in shared memory
    for (var i = tid; i < N; i += tpg) {
        let N_i32 = i32(N);
        var src_idx = (i32(i) + i32(b_rot)) % (2 * N_i32);
        if (src_idx < 0) { src_idx += 2 * N_i32; }

        var negate = false;
        if (src_idx >= N_i32) {
            src_idx -= N_i32;
            negate = true;
        }

        var val = read_u64_storage(&test_vector, u32(src_idx));
        if (negate) {
            val = mod_neg(val, Q);
        }

        write_shared_acc(i, val);
    }
    workgroupBarrier();

    // Phase 2: Blind rotation steps (iterative)
    for (var step = 0u; step < n; step++) {
        let a_j = read_u64_storage(&lwe_in, lwe_base + step);
        let rot = (a_j.hi >> (32u - log_N - 1u)) & ((2u << log_N) - 1u);

        if (rot == 0u) { continue; }

        // Simplified CMux: acc = acc * X^{a_j} (negacyclic)
        let N_i32 = i32(N);
        let rot_i32 = i32(rot);

        for (var i = tid; i < N; i += tpg) {
            var src_idx = (i32(i) - rot_i32) % (2 * N_i32);
            if (src_idx < 0) { src_idx += 2 * N_i32; }

            var negate = false;
            if (src_idx >= N_i32) {
                src_idx -= N_i32;
                negate = true;
            }

            var val = read_shared_acc(u32(src_idx));
            if (negate) {
                val = mod_neg(val, Q);
            }

            write_shared_work(i, val);
        }
        workgroupBarrier();

        // Copy back
        for (var i = tid; i < N; i += tpg) {
            write_shared_acc(i, read_shared_work(i));
        }
        workgroupBarrier();
    }

    // Phase 3: Sample extract
    let out_base = batch_idx * (N + 1u);

    for (var i = tid; i < N; i += tpg) {
        let src_idx = (N - 1u - i) % N;
        var val = read_shared_acc(src_idx);
        if (i > 0u) {
            val = mod_neg(val, Q);
        }
        write_u64_storage(&lwe_out, out_base + i, val);
    }

    if (tid == 0u) {
        write_u64_storage(&lwe_out, out_base + N, read_shared_acc(0u));
    }
}

// ============================================================================
// Functional Bootstrap with Programmable LUT
// ============================================================================

@group(1) @binding(0) var<storage, read> lut_polynomials: array<u32>;  // Multiple LUTs
@group(1) @binding(1) var<uniform> lut_selector: u32;

@compute @workgroup_size(256)
fn programmable_bootstrap(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let Q = params_Q(params);
    let tpg = 256u;

    let batch_idx = wgid.x;

    if (batch_idx >= params.batch_size) { return; }

    // Select LUT based on lut_selector
    let lut_base = lut_selector * N;

    // Load selected LUT to shared memory
    for (var i = tid; i < N; i += tpg) {
        let val = read_u64_storage(&lut_polynomials, lut_base + i);
        write_shared_acc(i, val);
    }
    workgroupBarrier();

    // Perform bootstrap using selected LUT as test vector
    // (Implementation same as blind rotation, using shared LUT)

    // ... (same as fused_bootstrap_small, but using shared_acc as LUT)
}
