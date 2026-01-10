// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// FHE Scheme Switching Kernels for WebGPU
// Implements conversion between TFHE and CKKS schemes
// Enables hybrid computations leveraging strengths of each scheme
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
// Fixed-Point Representation for CKKS
// ============================================================================

struct Complex64 {
    real_lo: u32,
    real_hi: u32,
    imag_lo: u32,
    imag_hi: u32,
}

fn complex_zero() -> Complex64 {
    return Complex64(0u, 0u, 0u, 0u);
}

fn complex_real(c: Complex64) -> u64 {
    return u64(c.real_lo, c.real_hi);
}

fn complex_imag(c: Complex64) -> u64 {
    return u64(c.imag_lo, c.imag_hi);
}

fn make_complex(real: u64, imag: u64) -> Complex64 {
    return Complex64(real.lo, real.hi, imag.lo, imag.hi);
}

// ============================================================================
// Scheme Switching Parameters
// ============================================================================

struct SchemeSwitchParams {
    // TFHE parameters
    N_tfhe: u32,          // TFHE polynomial degree
    Q_tfhe_lo: u32,       // TFHE modulus low
    Q_tfhe_hi: u32,       // TFHE modulus high

    // CKKS parameters
    N_ckks: u32,          // CKKS polynomial degree
    Q_ckks_lo: u32,       // CKKS modulus low
    Q_ckks_hi: u32,       // CKKS modulus high
    scale_log: u32,       // Log of CKKS scale (e.g., 40)

    // Conversion parameters
    slots: u32,           // Number of CKKS slots
    precision_bits: u32,  // Bit precision for conversion

    // Barrett constants
    mu_tfhe_lo: u32,
    mu_tfhe_hi: u32,
    mu_ckks_lo: u32,
    mu_ckks_hi: u32,

    batch_size: u32,
    _pad: u32,
}

fn params_Q_tfhe(p: SchemeSwitchParams) -> u64 { return u64(p.Q_tfhe_lo, p.Q_tfhe_hi); }
fn params_Q_ckks(p: SchemeSwitchParams) -> u64 { return u64(p.Q_ckks_lo, p.Q_ckks_hi); }
fn params_mu_tfhe(p: SchemeSwitchParams) -> u64 { return u64(p.mu_tfhe_lo, p.mu_tfhe_hi); }
fn params_mu_ckks(p: SchemeSwitchParams) -> u64 { return u64(p.mu_ckks_lo, p.mu_ckks_hi); }

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> tfhe_input: array<u32>;      // TFHE ciphertext
@group(0) @binding(1) var<storage, read_write> ckks_output: array<u32>; // CKKS ciphertext
@group(0) @binding(2) var<storage, read> ckks_input: array<u32>;      // CKKS ciphertext (for reverse)
@group(0) @binding(3) var<storage, read_write> tfhe_output: array<u32>; // TFHE ciphertext (for reverse)
@group(0) @binding(4) var<storage, read> switch_key: array<u32>;      // Scheme switching key
@group(0) @binding(5) var<storage, read> rotation_factors: array<u32>; // Roots of unity
@group(0) @binding(6) var<uniform> params: SchemeSwitchParams;

// Shared memory
var<workgroup> shared_buffer: array<u32, 4096>;

// ============================================================================
// TFHE to CKKS Conversion
// ============================================================================

// Step 1: Extract TFHE message bits to polynomial coefficients
@compute @workgroup_size(256)
fn tfhe_extract_to_poly(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N_tfhe = params.N_tfhe;
    let Q_tfhe = params_Q_tfhe(params);
    let tpg = 256u;

    let coeff_idx = gid.x;
    let batch_idx = wgid.y;

    if (coeff_idx >= N_tfhe || batch_idx >= params.batch_size) { return; }

    // Read TFHE coefficient
    let tfhe_idx = batch_idx * N_tfhe + coeff_idx;
    let tfhe_val = u64(tfhe_input[tfhe_idx * 2u], tfhe_input[tfhe_idx * 2u + 1u]);

    // Rescale from Q_tfhe to precision_bits
    // value_scaled = round(tfhe_val * 2^precision / Q_tfhe)
    let precision = params.precision_bits;
    let scaled = u64_shr(u64_mul(tfhe_val, u64_shl(u64_from_u32(1u), precision)), 32u);

    // Store as CKKS coefficient (will need encoding later)
    let ckks_idx = batch_idx * N_tfhe + coeff_idx;
    ckks_output[ckks_idx * 2u] = scaled.lo;
    ckks_output[ckks_idx * 2u + 1u] = scaled.hi;
}

// Step 2: Apply CKKS encoding (canonical embedding)
@compute @workgroup_size(256)
fn apply_ckks_encoding(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N_ckks = params.N_ckks;
    let slots = params.slots;
    let Q_ckks = params_Q_ckks(params);
    let mu_ckks = params_mu_ckks(params);
    let tpg = 256u;

    let slot_idx = gid.x;
    let batch_idx = wgid.y;

    if (slot_idx >= slots || batch_idx >= params.batch_size) { return; }

    let input_base = batch_idx * N_ckks;

    // Load coefficients to shared memory
    for (var i = tid; i < N_ckks * 2u && i < 4096u; i += tpg) {
        shared_buffer[i] = ckks_output[(input_base * 2u) + i];
    }
    workgroupBarrier();

    // Apply FFT-based encoding (simplified inverse DFT)
    // For slot k: value_k = sum_{j=0}^{N-1} coeff_j * omega^{jk}
    var sum_real = u64_zero();
    var sum_imag = u64_zero();

    let omega_base = slot_idx * 2u + 1u;  // Primitive root index

    for (var j = 0u; j < N_ckks && j < 256u; j++) {
        let coeff = u64(shared_buffer[j * 2u], shared_buffer[j * 2u + 1u]);

        // Get rotation factor omega^{j * omega_base}
        let rot_idx = ((j * omega_base) % (2u * N_ckks)) * 2u;
        let omega_real = u64(rotation_factors[rot_idx], rotation_factors[rot_idx + 1u]);
        let omega_imag = u64(rotation_factors[rot_idx + 2u], rotation_factors[rot_idx + 3u]);

        // Multiply and accumulate
        let prod_real = barrett_mul(coeff, omega_real, Q_ckks, mu_ckks);
        let prod_imag = barrett_mul(coeff, omega_imag, Q_ckks, mu_ckks);

        sum_real = mod_add(sum_real, prod_real, Q_ckks);
        sum_imag = mod_add(sum_imag, prod_imag, Q_ckks);
    }

    // Store encoded value
    let out_idx = batch_idx * slots * 4u + slot_idx * 4u;
    ckks_output[out_idx] = sum_real.lo;
    ckks_output[out_idx + 1u] = sum_real.hi;
    ckks_output[out_idx + 2u] = sum_imag.lo;
    ckks_output[out_idx + 3u] = sum_imag.hi;
}

// Step 3: Scale and add noise for CKKS
@compute @workgroup_size(256)
fn ckks_scale_and_noise(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N_ckks = params.N_ckks;
    let Q_ckks = params_Q_ckks(params);
    let scale_log = params.scale_log;

    let idx = gid.x;
    let batch_idx = gid.y;

    if (idx >= N_ckks || batch_idx >= params.batch_size) { return; }

    let in_idx = batch_idx * N_ckks + idx;
    let val = u64(ckks_output[in_idx * 2u], ckks_output[in_idx * 2u + 1u]);

    // Apply CKKS scale: value * 2^scale_log
    let scale = u64_shl(u64_from_u32(1u), scale_log);
    let scaled = barrett_mul(val, scale, Q_ckks, params_mu_ckks(params));

    ckks_output[in_idx * 2u] = scaled.lo;
    ckks_output[in_idx * 2u + 1u] = scaled.hi;
}

// ============================================================================
// CKKS to TFHE Conversion
// ============================================================================

// Step 1: CKKS decoding (remove encoding)
@compute @workgroup_size(256)
fn apply_ckks_decoding(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N_ckks = params.N_ckks;
    let slots = params.slots;
    let Q_ckks = params_Q_ckks(params);
    let mu_ckks = params_mu_ckks(params);
    let tpg = 256u;

    let coeff_idx = gid.x;
    let batch_idx = wgid.y;

    if (coeff_idx >= N_ckks || batch_idx >= params.batch_size) { return; }

    let input_base = batch_idx * slots * 4u;

    // Load slot values to shared memory
    for (var i = tid; i < slots * 4u && i < 4096u; i += tpg) {
        shared_buffer[i] = ckks_input[input_base + i];
    }
    workgroupBarrier();

    // Apply DFT to convert slots to coefficients
    // coeff_j = (1/N) * sum_{k=0}^{slots-1} value_k * omega^{-jk}
    var sum = u64_zero();

    for (var k = 0u; k < slots && k < 256u; k++) {
        let val_real = u64(shared_buffer[k * 4u], shared_buffer[k * 4u + 1u]);
        let val_imag = u64(shared_buffer[k * 4u + 2u], shared_buffer[k * 4u + 3u]);

        // Get inverse rotation factor
        let omega_base = k * 2u + 1u;
        let inv_rot_idx = ((2u * N_ckks - (coeff_idx * omega_base) % (2u * N_ckks)) % (2u * N_ckks)) * 2u;
        let omega_real = u64(rotation_factors[inv_rot_idx], rotation_factors[inv_rot_idx + 1u]);

        // Multiply (only real part for simplicity)
        let prod = barrett_mul(val_real, omega_real, Q_ckks, mu_ckks);
        sum = mod_add(sum, prod, Q_ckks);
    }

    // Scale by 1/slots
    let inv_slots = u64_from_u32(1u);  // Placeholder - need proper inverse
    sum = barrett_mul(sum, inv_slots, Q_ckks, mu_ckks);

    let out_idx = batch_idx * N_ckks + coeff_idx;
    tfhe_output[out_idx * 2u] = sum.lo;
    tfhe_output[out_idx * 2u + 1u] = sum.hi;
}

// Step 2: Modulus switching from Q_ckks to Q_tfhe
@compute @workgroup_size(256)
fn mod_switch_ckks_to_tfhe(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N = params.N_ckks;  // Assuming same N for simplicity
    let Q_ckks = params_Q_ckks(params);
    let Q_tfhe = params_Q_tfhe(params);
    let scale_log = params.scale_log;

    let idx = gid.x;
    let batch_idx = gid.y;

    if (idx >= N || batch_idx >= params.batch_size) { return; }

    let in_idx = batch_idx * N + idx;
    let val = u64(tfhe_output[in_idx * 2u], tfhe_output[in_idx * 2u + 1u]);

    // Remove CKKS scale
    let descaled = u64_shr(val, scale_log);

    // Rescale modulus: round(val * Q_tfhe / Q_ckks)
    // Simplified approximation
    let ratio = u64_mul(descaled, Q_tfhe);
    let switched = u64_shr(ratio, 32u);  // Approximate division

    tfhe_output[in_idx * 2u] = switched.lo;
    tfhe_output[in_idx * 2u + 1u] = switched.hi;
}

// Step 3: Discretize for TFHE (add rounding)
@compute @workgroup_size(256)
fn discretize_for_tfhe(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N = params.N_tfhe;
    let Q_tfhe = params_Q_tfhe(params);
    let precision = params.precision_bits;

    let idx = gid.x;
    let batch_idx = gid.y;

    if (idx >= N || batch_idx >= params.batch_size) { return; }

    let in_idx = batch_idx * N + idx;
    let val = u64(tfhe_output[in_idx * 2u], tfhe_output[in_idx * 2u + 1u]);

    // Round to nearest precision level
    let half = u64_shl(u64_from_u32(1u), 64u - precision - 1u);
    let rounded = u64_add(val, half);
    let discretized = u64_shr(u64_shl(rounded, precision), precision);

    // Ensure within modulus
    var result = discretized;
    if (u64_ge(result, Q_tfhe)) {
        result = u64_sub(result, Q_tfhe);
    }

    tfhe_output[in_idx * 2u] = result.lo;
    tfhe_output[in_idx * 2u + 1u] = result.hi;
}

// ============================================================================
// Hybrid Operations (Operations between schemes)
// ============================================================================

// Add TFHE ciphertext to CKKS ciphertext (after conversion)
@compute @workgroup_size(256)
fn hybrid_add(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N = params.N_ckks;
    let Q_ckks = params_Q_ckks(params);

    let idx = gid.x;
    let batch_idx = gid.y;

    if (idx >= N || batch_idx >= params.batch_size) { return; }

    let in_idx = batch_idx * N + idx;

    // Both should be in CKKS domain after conversion
    let a = u64(ckks_input[in_idx * 2u], ckks_input[in_idx * 2u + 1u]);
    let b = u64(ckks_output[in_idx * 2u], ckks_output[in_idx * 2u + 1u]);
    let sum = mod_add(a, b, Q_ckks);

    ckks_output[in_idx * 2u] = sum.lo;
    ckks_output[in_idx * 2u + 1u] = sum.hi;
}

// ============================================================================
// Functional Bootstrapping with Scheme Switch
// ============================================================================

@group(1) @binding(0) var<storage, read> lut_ckks: array<u32>;  // LUT in CKKS form

// Apply arbitrary function via TFHE bootstrap then CKKS evaluation
@compute @workgroup_size(256)
fn functional_scheme_switch(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N_tfhe = params.N_tfhe;
    let N_ckks = params.N_ckks;
    let Q_tfhe = params_Q_tfhe(params);
    let Q_ckks = params_Q_ckks(params);
    let tpg = 256u;

    let batch_idx = wgid.x;

    if (batch_idx >= params.batch_size) { return; }

    // Load TFHE input to shared memory
    let tfhe_base = batch_idx * N_tfhe;
    for (var i = tid; i < N_tfhe * 2u && i < 4096u; i += tpg) {
        shared_buffer[i] = tfhe_input[(tfhe_base * 2u) + i];
    }
    workgroupBarrier();

    // Convert and evaluate
    // This is a simplified placeholder - full implementation would:
    // 1. Extract TFHE bits via PBS
    // 2. Convert to CKKS representation
    // 3. Evaluate polynomial function in CKKS
    // 4. Convert back to TFHE

    for (var i = tid; i < N_ckks; i += tpg) {
        // Placeholder: copy with modulus rescale
        let tfhe_idx = i % N_tfhe;
        let tfhe_val = u64(shared_buffer[tfhe_idx * 2u], shared_buffer[tfhe_idx * 2u + 1u]);

        // Scale to CKKS modulus
        let scaled = barrett_mul(tfhe_val, Q_ckks, Q_tfhe, params_mu_tfhe(params));

        let out_idx = batch_idx * N_ckks + i;
        ckks_output[out_idx * 2u] = scaled.lo;
        ckks_output[out_idx * 2u + 1u] = scaled.hi;
    }
}

// ============================================================================
// Bit Decomposition for Scheme Switch
// ============================================================================

@group(2) @binding(0) var<storage, read_write> bit_decomposed: array<u32>;

// Decompose CKKS value into bits for TFHE processing
@compute @workgroup_size(256)
fn decompose_to_bits(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N = params.N_ckks;
    let precision = params.precision_bits;
    let Q_ckks = params_Q_ckks(params);

    let coeff_idx = gid.x;
    let batch_idx = gid.y;

    if (coeff_idx >= N || batch_idx >= params.batch_size) { return; }

    let in_idx = batch_idx * N + coeff_idx;
    let val = u64(ckks_input[in_idx * 2u], ckks_input[in_idx * 2u + 1u]);

    // Extract bits (MSB first)
    let out_base = (batch_idx * N + coeff_idx) * precision;

    for (var bit = 0u; bit < precision; bit++) {
        let shift = precision - 1u - bit;
        var bit_val: u32;
        if (shift >= 32u) {
            bit_val = (val.hi >> (shift - 32u)) & 1u;
        } else {
            bit_val = (val.lo >> shift) & 1u;
        }

        // Store bit as TFHE-compatible value (Q/2 for 1, 0 for 0)
        if (bit_val == 1u) {
            let half_Q = u64_shr(params_Q_tfhe(params), 1u);
            bit_decomposed[(out_base + bit) * 2u] = half_Q.lo;
            bit_decomposed[(out_base + bit) * 2u + 1u] = half_Q.hi;
        } else {
            bit_decomposed[(out_base + bit) * 2u] = 0u;
            bit_decomposed[(out_base + bit) * 2u + 1u] = 0u;
        }
    }
}

// Reconstruct value from TFHE bit ciphertexts
@compute @workgroup_size(256)
fn reconstruct_from_bits(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let N = params.N_ckks;
    let precision = params.precision_bits;
    let Q_ckks = params_Q_ckks(params);

    let coeff_idx = gid.x;
    let batch_idx = gid.y;

    if (coeff_idx >= N || batch_idx >= params.batch_size) { return; }

    let in_base = (batch_idx * N + coeff_idx) * precision;

    var result = u64_zero();

    for (var bit = 0u; bit < precision; bit++) {
        let bit_val = u64(bit_decomposed[(in_base + bit) * 2u],
                         bit_decomposed[(in_base + bit) * 2u + 1u]);

        // Check if bit is set (value near Q/2)
        let half_Q = u64_shr(Q_ckks, 1u);
        let threshold = u64_shr(half_Q, 1u);

        let shift = precision - 1u - bit;
        if (u64_ge(bit_val, threshold)) {
            result = u64_add(result, u64_shl(u64_from_u32(1u), shift));
        }
    }

    // Scale to CKKS modulus
    let scale_log = params.scale_log;
    result = u64_shl(result, scale_log - precision);

    let out_idx = batch_idx * N + coeff_idx;
    ckks_output[out_idx * 2u] = result.lo;
    ckks_output[out_idx * 2u + 1u] = result.hi;
}
