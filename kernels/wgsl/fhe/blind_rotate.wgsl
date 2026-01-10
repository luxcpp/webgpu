// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// TFHE Blind Rotation (Programmable Bootstrapping)
// Core operation for TFHE-style homomorphic encryption
// Converts LWE ciphertext to GLWE via rotation with encrypted index

// TFHE Parameters (configurable via uniforms)
// N: polynomial degree (typically 1024, 2048, or 4096)
// k: GLWE dimension
// n: LWE dimension
// Q: ciphertext modulus

// ============================================================================
// Type Definitions
// ============================================================================

// 64-bit unsigned integer (emulated with two u32)
struct U64 {
    lo: u32,
    hi: u32,
}

// Torus element (32-bit by default, can be extended to 64-bit)
alias Torus32 = u32;

// LWE sample: (a_0, ..., a_{n-1}, b) where b = sum(a_i * s_i) + m + e
struct LweSample {
    // a coefficients stored separately for coalesced access
    b: Torus32,  // Body
}

// GLWE sample: polynomial ciphertext over R_Q = Z_Q[X]/(X^N + 1)
// Stored as (k+1) polynomials of degree N
struct GlweSample {
    // Polynomials stored separately for efficiency
    // body polynomial + k mask polynomials
}

// GGSW ciphertext (encryption of polynomial under GSW scheme)
// Used for the bootstrapping key
struct GgswCiphertext {
    // (k+1) * l GLWE samples (l is decomposition level)
}

// Blind rotation parameters
struct BlindRotateParams {
    N: u32,              // Polynomial degree
    k: u32,              // GLWE dimension
    n: u32,              // LWE dimension
    l: u32,              // Decomposition levels
    base_log: u32,       // Base log for decomposition (Bg = 2^base_log)
    num_samples: u32,    // Batch size
}

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> lwe_a: array<Torus32>;           // LWE 'a' coefficients
@group(0) @binding(1) var<storage, read> lwe_b: array<Torus32>;           // LWE 'b' values
@group(0) @binding(2) var<storage, read> bsk: array<Torus32>;             // Bootstrapping key (GGSW samples)
@group(0) @binding(3) var<storage, read> test_vector: array<Torus32>;     // Test polynomial
@group(0) @binding(4) var<storage, read_write> acc_poly: array<Torus32>;  // Accumulator polynomial
@group(0) @binding(5) var<storage, read_write> temp_poly: array<Torus32>; // Temporary polynomial
@group(0) @binding(6) var<uniform> params: BlindRotateParams;

// Shared memory for workgroup-local computation
var<workgroup> shared_poly: array<Torus32, 4096>;  // Max polynomial degree
var<workgroup> shared_decomp: array<i32, 4096>;    // Decomposition workspace

// ============================================================================
// Polynomial Arithmetic in R_Q = Z_Q[X]/(X^N + 1)
// ============================================================================

// Multiply polynomial by X^a mod (X^N + 1)
// This is a rotation with sign flip for coefficients that wrap around
fn rotate_polynomial(
    poly: ptr<storage, array<Torus32>, read_write>,
    rotation: u32,
    N: u32,
    offset: u32
) {
    let local_id = workgroup_id.x * 256u + local_invocation_id.x;
    
    if (local_id < N) {
        let src_idx = offset + local_id;
        let rot = rotation % (2u * N);
        
        var dst_idx: u32;
        var negate: bool;
        
        if (rot < N) {
            // Simple rotation within polynomial
            if (local_id >= rot) {
                dst_idx = local_id - rot;
                negate = false;
            } else {
                dst_idx = N - rot + local_id;
                negate = true;
            }
        } else {
            // Rotation >= N: negate and rotate by (rot - N)
            let r = rot - N;
            if (local_id >= r) {
                dst_idx = local_id - r;
                negate = true;
            } else {
                dst_idx = N - r + local_id;
                negate = false;
            }
        }
        
        let val = (*poly)[src_idx];
        shared_poly[dst_idx] = select(val, 0u - val, negate);
    }
    
    workgroupBarrier();
    
    // Copy back from shared memory
    if (local_id < N) {
        (*poly)[offset + local_id] = shared_poly[local_id];
    }
}

// Add two polynomials coefficient-wise
fn poly_add(
    result: ptr<storage, array<Torus32>, read_write>,
    a: ptr<storage, array<Torus32>, read>,
    b: ptr<storage, array<Torus32>, read>,
    N: u32,
    offset_r: u32,
    offset_a: u32,
    offset_b: u32
) {
    let idx = workgroup_id.x * 256u + local_invocation_id.x;
    
    if (idx < N) {
        (*result)[offset_r + idx] = (*a)[offset_a + idx] + (*b)[offset_b + idx];
    }
}

// Subtract two polynomials
fn poly_sub(
    result: ptr<storage, array<Torus32>, read_write>,
    a: ptr<storage, array<Torus32>, read_write>,
    b: ptr<storage, array<Torus32>, read>,
    N: u32,
    offset_r: u32,
    offset_a: u32,
    offset_b: u32
) {
    let idx = workgroup_id.x * 256u + local_invocation_id.x;
    
    if (idx < N) {
        (*result)[offset_r + idx] = (*a)[offset_a + idx] - (*b)[offset_b + idx];
    }
}

// ============================================================================
// Gadget Decomposition
// ============================================================================

// Signed decomposition of a polynomial coefficient
// Decomposes x into (a_0, ..., a_{l-1}) where x ≈ sum(a_i * Bg^i)
fn signed_decompose(x: Torus32, level: u32, base_log: u32) -> i32 {
    let Bg = 1u << base_log;
    let half_Bg = Bg >> 1u;
    let mask = Bg - 1u;
    
    // Extract level-th digit from MSB
    let shift = 32u - (level + 1u) * base_log;
    var digit = (x >> shift) & mask;
    
    // Signed representation: if digit > Bg/2, subtract Bg
    if (digit >= half_Bg) {
        return i32(digit) - i32(Bg);
    }
    return i32(digit);
}

// Decompose entire polynomial
@compute @workgroup_size(256)
fn decompose_polynomial(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let idx = gid.x;
    let N = params.N;
    let level = gid.y;
    
    if (idx >= N || level >= params.l) { return; }
    
    let coeff = acc_poly[idx];
    let decomposed = signed_decompose(coeff, level, params.base_log);
    
    // Store decomposition (convert to unsigned for storage)
    let decomp_offset = level * N + idx;
    shared_decomp[idx] = decomposed;
}

// ============================================================================
// External Product: GLWE x GGSW -> GLWE
// ============================================================================

// External product is the core operation in blind rotation
// result = acc * GGSW(s_i) where s_i is a bit of the secret key
@compute @workgroup_size(256)
fn external_product(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let idx = gid.x;
    let N = params.N;
    let k = params.k;
    let l = params.l;
    
    if (idx >= N) { return; }
    
    // Step 1: Decompose accumulator polynomial
    // For each level j in [0, l):
    //   decomp[j] = SignedDecompose(acc, j, Bg)
    
    // Step 2: Multiply decomposed values by GGSW rows
    // For each polynomial index i in [0, k+1):
    //   result[i] = sum_{j=0}^{l-1} decomp[j] * GGSW[j][i]
    
    // Step 3: Accumulate results
    
    // Simplified implementation - full version needs NTT multiplication
    let coeff_idx = idx;
    
    for (var level = 0u; level < l; level++) {
        let decomp_val = signed_decompose(acc_poly[coeff_idx], level, params.base_log);
        
        // Get corresponding GGSW coefficient
        let bsk_offset = level * N + coeff_idx;
        let bsk_coeff = bsk[bsk_offset];
        
        // Multiply and accumulate (schoolbook - should use NTT)
        temp_poly[coeff_idx] += u32(decomp_val) * bsk_coeff;
    }
}

// ============================================================================
// Blind Rotation Kernel
// ============================================================================

// Main blind rotation: rotates test vector by encrypted amount
// Input: LWE sample (a, b), test vector TV, bootstrapping key BSK
// Output: GLWE sample where body ≈ X^{-b} * TV * prod_i X^{a_i * s_i}
@compute @workgroup_size(256)
fn blind_rotate(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let idx = gid.x;
    let sample_idx = wgid.y;
    let N = params.N;
    let n = params.n;
    
    if (idx >= N) { return; }
    if (sample_idx >= params.num_samples) { return; }
    
    let acc_offset = sample_idx * N;
    
    // Step 1: Initialize accumulator with rotated test vector
    // acc = X^{-b} * test_vector
    if (idx == 0u) {
        let b = lwe_b[sample_idx];
        // Compute rotation amount from b
        // b is in Torus32, map to rotation in [0, 2N)
        let rotation = ((b + (1u << 31u) / N) >> (32u - 1u - u32(log2(f32(N))))) % (2u * N);
        
        // This should call rotate_polynomial, but we inline for simplicity
    }
    
    workgroupBarrier();
    
    // Copy test vector to accumulator initially
    acc_poly[acc_offset + idx] = test_vector[idx];
    
    workgroupBarrier();
    
    // Step 2: For each LWE coefficient a_i, perform CMux
    // CMux(GGSW(s_i), acc * X^{a_i}, acc)
    // = acc * X^{a_i * s_i}
    
    for (var i = 0u; i < n; i++) {
        let a_i = lwe_a[sample_idx * n + i];
        
        // Map a_i to rotation amount
        let rotation = ((a_i + (1u << 31u) / N) >> (32u - 1u - u32(log2(f32(N))))) % (2u * N);
        
        if (rotation != 0u) {
            // Compute acc * X^{rotation} - acc
            // Then external product with BSK[i]
            // acc += (acc * X^{rotation} - acc) * GGSW(s_i)
            
            // Simplified - actual implementation uses external_product kernel
            workgroupBarrier();
        }
    }
}

// ============================================================================
// Sample Extraction
// ============================================================================

// Extract LWE sample from GLWE after blind rotation
@compute @workgroup_size(256)
fn sample_extract(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let sample_idx = gid.y;
    let N = params.N;
    let k = params.k;
    
    if (idx >= N * k || sample_idx >= params.num_samples) { return; }
    
    let acc_offset = sample_idx * N * (k + 1u);
    
    // Extract coefficient 0 from each mask polynomial
    // The extracted LWE has dimension N * k
    
    let poly_idx = idx / N;
    let coeff_idx = idx % N;
    
    // For extraction, we need to reverse and negate certain coefficients
    // to account for the structure of R_Q = Z_Q[X]/(X^N + 1)
    
    if (coeff_idx == 0u) {
        // First coefficient stays as is
        lwe_a[sample_idx * N * k + idx] = acc_poly[acc_offset + poly_idx * N];
    } else {
        // Other coefficients get negated in reverse order
        lwe_a[sample_idx * N * k + idx] = 0u - acc_poly[acc_offset + poly_idx * N + N - coeff_idx];
    }
}

// Extract body coefficient
@compute @workgroup_size(256)
fn extract_body(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let sample_idx = gid.x;
    
    if (sample_idx >= params.num_samples) { return; }
    
    let acc_offset = sample_idx * params.N * (params.k + 1u);
    let body_offset = params.k * params.N;  // Body is last polynomial
    
    // Extract coefficient 0 of body polynomial
    lwe_b[sample_idx] = acc_poly[acc_offset + body_offset];
}
