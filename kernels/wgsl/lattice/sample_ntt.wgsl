// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Sampling Kernels for ML-DSA (Dilithium) and ML-KEM (Kyber)
// - Sample uniform polynomials in NTT domain
// - Rejection sampling from SHAKE128/256 output
// - CBD (centered binomial distribution) sampling
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

// ============================================================================
// Constants
// ============================================================================

const DILITHIUM_Q: u32 = 8380417u;
const KYBER_Q: u32 = 3329u;

const DILITHIUM_MASK: u32 = 0x7FFFFFu;  // 23 bits
const KYBER_MASK: u32 = 0xFFFu;          // 12 bits

// ============================================================================
// Parameter Structures
// ============================================================================

struct SampleParams {
    n: u32,
    buf_len: u32,
    batch: u32,
    eta: u32,
    gamma1_bits: u32,
    tau: u32,
    rows: u32,
    cols: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read_write> output: array<i32>;
@group(0) @binding(1) var<storage, read> buf: array<u32>;  // Packed bytes as u32
@group(0) @binding(2) var<storage, read_write> valid_count: atomic<u32>;
@group(0) @binding(3) var<uniform> params: SampleParams;

// ============================================================================
// Helper Functions
// ============================================================================

fn popcount(x: u32) -> u32 {
    var v = x;
    v = v - ((v >> 1u) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    v = (v + (v >> 4u)) & 0x0F0F0F0Fu;
    v = v + (v >> 8u);
    v = v + (v >> 16u);
    return v & 0x3Fu;
}

fn get_byte(buf_idx: u32) -> u32 {
    let word_idx = buf_idx / 4u;
    let byte_offset = buf_idx % 4u;
    return (buf[word_idx] >> (byte_offset * 8u)) & 0xFFu;
}

fn bit_reverse(x: u32, log_n: u32) -> u32 {
    var result = 0u;
    var val = x;
    for (var i = 0u; i < log_n; i = i + 1u) {
        result = (result << 1u) | (val & 1u);
        val = val >> 1u;
    }
    return result;
}

// ============================================================================
// Rejection Sampling: Uniform in [0, q) from SHAKE output
// ============================================================================

@compute @workgroup_size(256)
fn sample_uniform_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let buf_len = params.buf_len;
    let batch = params.batch;
    
    let sample_idx = gid.x;
    let byte_idx = sample_idx * 3u;
    
    if (byte_idx + 2u >= buf_len) { return; }
    
    // Read 3 bytes and mask to 23 bits
    let b0 = get_byte(byte_idx);
    let b1 = get_byte(byte_idx + 1u);
    let b2 = get_byte(byte_idx + 2u);
    
    var val = b0 | (b1 << 8u) | (b2 << 16u);
    val = val & DILITHIUM_MASK;
    
    // Rejection sampling
    if (val < DILITHIUM_Q) {
        let slot = atomicAdd(&valid_count, 1u);
        if (slot < n) {
            output[batch * n + slot] = i32(val);
        }
    }
}

@compute @workgroup_size(256)
fn sample_uniform_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let buf_len = params.buf_len;
    let batch = params.batch;
    
    let byte_idx = gid.x * 3u;
    
    if (byte_idx + 2u >= buf_len) { return; }
    
    let b0 = get_byte(byte_idx);
    let b1 = get_byte(byte_idx + 1u);
    let b2 = get_byte(byte_idx + 2u);
    
    // Extract two 12-bit values
    let d1 = b0 | ((b1 & 0x0Fu) << 8u);
    let d2 = (b1 >> 4u) | (b2 << 4u);
    
    // First sample
    if (d1 < KYBER_Q) {
        let slot = atomicAdd(&valid_count, 1u);
        if (slot < n) {
            output[batch * n + slot] = i32(d1);
        }
    }
    
    // Second sample
    if (d2 < KYBER_Q) {
        let slot = atomicAdd(&valid_count, 1u);
        if (slot < n) {
            output[batch * n + slot] = i32(d2);
        }
    }
}

// ============================================================================
// Centered Binomial Distribution (CBD) Sampling
// ============================================================================

@compute @workgroup_size(256)
fn sample_cbd2_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let batch = params.batch;
    
    let total = batch * n;
    if (gid.x >= total) { return; }
    
    // 4 bits for a, 4 bits for b
    let byte_val = get_byte(gid.x);
    
    let a_bits = byte_val & 0x0Fu;
    let a = i32(popcount(a_bits));
    
    let b_bits = byte_val >> 4u;
    let b = i32(popcount(b_bits));
    
    output[gid.x] = a - b;
}

@compute @workgroup_size(64)
fn sample_cbd3_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let batch = params.batch;
    
    // Each thread processes 3 bytes -> 4 coefficients
    let group_idx = gid.x;
    let byte_idx = group_idx * 3u;
    let coef_idx = group_idx * 4u;
    
    let total_coefs = batch * n;
    if (coef_idx + 3u >= total_coefs) { return; }
    
    // Read 24 bits
    let b0 = get_byte(byte_idx);
    let b1 = get_byte(byte_idx + 1u);
    let b2 = get_byte(byte_idx + 2u);
    let bits = b0 | (b1 << 8u) | (b2 << 16u);
    
    // Process 4 coefficients, 6 bits each
    for (var i = 0u; i < 4u; i = i + 1u) {
        let a_bits = (bits >> (i * 6u)) & 0x07u;
        let b_bits = (bits >> (i * 6u + 3u)) & 0x07u;
        
        let a = i32(popcount(a_bits));
        let b = i32(popcount(b_bits));
        
        output[coef_idx + i] = a - b;
    }
}

@compute @workgroup_size(256)
fn sample_cbd_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let eta = params.eta;
    let batch = params.batch;
    
    let total = batch * n;
    if (gid.x >= total) { return; }
    
    let byte_val = get_byte(gid.x);
    
    if (eta == 2u) {
        // eta = 2: use 4 bits
        let a_bits = byte_val & 0x0Fu;
        let b_bits = byte_val >> 4u;
        let a = i32(popcount(a_bits & 0x03u)) + i32(popcount(a_bits >> 2u));
        let b = i32(popcount(b_bits & 0x03u)) + i32(popcount(b_bits >> 2u));
        output[gid.x] = a - b;
    } else {
        // eta = 4: use 8 bits
        let a = i32(popcount(byte_val & 0x0Fu));
        let b = i32(popcount(byte_val >> 4u));
        output[gid.x] = a - b;
    }
}

// ============================================================================
// Sample Bounded Coefficients for Dilithium Signing
// ============================================================================

@compute @workgroup_size(256)
fn sample_gamma1_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let gamma1_bits = params.gamma1_bits;
    let batch = params.batch;
    
    let total = batch * n;
    if (gid.x >= total) { return; }
    
    let bytes_per_coef = 3u;
    let byte_idx = gid.x * bytes_per_coef;
    
    let gamma1 = 1u << gamma1_bits;
    let mask = (gamma1 << 1u) - 1u;
    
    let b0 = get_byte(byte_idx);
    let b1 = get_byte(byte_idx + 1u);
    let b2 = get_byte(byte_idx + 2u);
    
    var val = b0 | (b1 << 8u) | (b2 << 16u);
    val = val & mask;
    
    // Center around 0
    output[gid.x] = i32(gamma1) - i32(val);
}

// ============================================================================
// Sample Challenge Polynomial for Dilithium
// ============================================================================

@compute @workgroup_size(256)
fn sample_challenge_init_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    if (gid.x >= n) { return; }
    
    output[gid.x] = 0;
}

// Additional bindings for challenge
@group(0) @binding(4) var<storage, read> signs: array<u32>;
@group(0) @binding(5) var<storage, read> positions: array<u32>;

@compute @workgroup_size(64)
fn sample_challenge_set_coeffs_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tau = params.tau;
    if (gid.x >= tau) { return; }
    
    // Get position for this non-zero coefficient
    let pos = positions[gid.x] & 0xFFu;
    
    // Get sign bit
    let sign_word = signs[gid.x / 32u];
    let sign_bit = (sign_word >> (gid.x % 32u)) & 1u;
    
    // Set coefficient to +1 or -1
    if (sign_bit == 1u) {
        output[pos] = -1;
    } else {
        output[pos] = 1;
    }
}

// ============================================================================
// Expand Matrix A from Seed
// ============================================================================

struct ExpandParams {
    n: u32,
    rows: u32,
    cols: u32,
    batch: u32,
}

@group(0) @binding(6) var<uniform> expand_params: ExpandParams;
@group(0) @binding(7) var<storage, read> shake_output: array<u32>;

@compute @workgroup_size(256)
fn expand_matrix_entry_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = expand_params.n;
    let rows = expand_params.rows;
    let cols = expand_params.cols;
    
    let row = gid.y;
    let col = gid.z;
    
    if (row >= rows || col >= cols) { return; }
    
    let poly_idx = row * cols + col;
    let coef_idx = gid.x;
    
    if (coef_idx >= n) { return; }
    
    // 3 bytes per sample attempt
    let buf_offset = poly_idx * n * 3u;
    let byte_idx = buf_offset + coef_idx * 3u;
    
    let b0 = get_byte(byte_idx);
    let b1 = get_byte(byte_idx + 1u);
    let b2 = get_byte(byte_idx + 2u);
    
    var val = b0 | (b1 << 8u) | (b2 << 16u);
    val = val & DILITHIUM_MASK;
    
    // Fallback for rejection
    if (val >= DILITHIUM_Q) {
        val = val % DILITHIUM_Q;
    }
    
    output[poly_idx * n + coef_idx] = i32(val);
}

@compute @workgroup_size(128)
fn expand_matrix_entry_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = expand_params.n;
    let rows = expand_params.rows;
    let cols = expand_params.cols;
    
    let row = gid.y;
    let col = gid.z;
    
    if (row >= rows || col >= cols) { return; }
    
    let poly_idx = row * cols + col;
    let coef_pair = gid.x;
    
    if (coef_pair * 2u >= n) { return; }
    
    let buf_offset = poly_idx * n * 2u;
    let byte_idx = buf_offset + coef_pair * 3u;
    
    let b0 = get_byte(byte_idx);
    let b1 = get_byte(byte_idx + 1u);
    let b2 = get_byte(byte_idx + 2u);
    
    var d1 = b0 | ((b1 & 0x0Fu) << 8u);
    var d2 = (b1 >> 4u) | (b2 << 4u);
    
    if (d1 >= KYBER_Q) { d1 = d1 % KYBER_Q; }
    if (d2 >= KYBER_Q) { d2 = d2 % KYBER_Q; }
    
    output[poly_idx * n + coef_pair * 2u] = i32(d1);
    output[poly_idx * n + coef_pair * 2u + 1u] = i32(d2);
}

// ============================================================================
// Convert to NTT Order (Bit-Reversal)
// ============================================================================

@compute @workgroup_size(256)
fn sample_to_ntt_order(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let batch = params.batch;
    
    let batch_idx = gid.x / n;
    let i = gid.x % n;
    
    if (batch_idx >= batch) { return; }
    
    // Calculate log_n
    var log_n = 0u;
    var temp = n;
    loop {
        if (temp <= 1u) { break; }
        temp = temp >> 1u;
        log_n = log_n + 1u;
    }
    
    let j = bit_reverse(i, log_n);
    
    if (i < j) {
        let base = batch_idx * n;
        let tmp = output[base + i];
        output[base + i] = output[base + j];
        output[base + j] = tmp;
    }
}
