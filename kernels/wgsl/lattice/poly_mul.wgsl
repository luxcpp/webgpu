// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Polynomial Multiplication for ML-DSA (Dilithium) and ML-KEM (Kyber)
// - Coefficient-wise multiplication in NTT domain
// - Batch operations for vector polynomials
// - Fused NTT multiply for polynomial convolution
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

// ============================================================================
// Constants
// ============================================================================

const DILITHIUM_Q: u32 = 8380417u;
const DILITHIUM_QINV: u32 = 58728449u;

const KYBER_Q: u32 = 3329u;
const KYBER_QINV: u32 = 62209u;

// ============================================================================
// Parameter Structures
// ============================================================================

struct PolyMulParams {
    size: u32,
    n: u32,           // polynomial degree
    k: u32,           // vector dimension
    batch: u32,
    m: u32,           // matrix dimension m
    p: u32,           // matrix dimension p
    d: u32,           // power2round parameter
    scalar: i32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read_write> result: array<i32>;
@group(0) @binding(1) var<storage, read> a: array<i32>;
@group(0) @binding(2) var<storage, read> b: array<i32>;
@group(0) @binding(3) var<uniform> params: PolyMulParams;

// ============================================================================
// Montgomery Multiplication
// ============================================================================

fn mont_mul_dilithium(x: i32, y: i32) -> i32 {
    // 32x32 -> 64 bit multiply emulation
    let x_u = bitcast<u32>(x);
    let y_u = bitcast<u32>(y);
    
    let x_lo = x_u & 0xFFFFu;
    let x_hi = x_u >> 16u;
    let y_lo = y_u & 0xFFFFu;
    let y_hi = y_u >> 16u;
    
    let p0 = x_lo * y_lo;
    let p1 = x_lo * y_hi;
    let p2 = x_hi * y_lo;
    let p3 = x_hi * y_hi;
    
    let mid = p1 + p2;
    let lo = p0 + (mid << 16u);
    let hi = p3 + (mid >> 16u) + select(0u, 1u, lo < p0);
    
    // Montgomery reduction
    let t = bitcast<i32>(lo) * i32(DILITHIUM_QINV);
    let tq_lo = bitcast<u32>(t) * DILITHIUM_Q;
    
    // Result is approximately (prod - tq) >> 32
    return bitcast<i32>(hi) - bitcast<i32>(tq_lo >> 31u);
}

fn mont_mul_kyber(x: i32, y: i32) -> i32 {
    let prod = x * y;
    let t = (prod & 0xFFFF) * i32(KYBER_QINV);
    let reduced = (prod - (t & 0xFFFF) * i32(KYBER_Q)) >> 16;
    return reduced;
}

fn barrett_reduce_kyber(a: i32) -> i32 {
    let v: i32 = 20159;
    let t = ((v * a + (1 << 25)) >> 26);
    return a - t * i32(KYBER_Q);
}

fn cond_sub_dilithium(x: i32) -> i32 {
    var a = x + ((x >> 31) & i32(DILITHIUM_Q));
    a = a - i32(DILITHIUM_Q);
    a = a + ((a >> 31) & i32(DILITHIUM_Q));
    return a;
}

fn cond_sub_kyber(x: i32) -> i32 {
    var a = x + ((x >> 15) & i32(KYBER_Q));
    a = a - i32(KYBER_Q);
    a = a + ((a >> 15) & i32(KYBER_Q));
    return a;
}

// ============================================================================
// Coefficient-wise Multiplication in NTT Domain
// ============================================================================

@compute @workgroup_size(256)
fn poly_pointwise_mul_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    result[gid.x] = mont_mul_dilithium(a[gid.x], b[gid.x]);
}

@compute @workgroup_size(256)
fn poly_pointwise_mul_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    result[gid.x] = mont_mul_kyber(a[gid.x], b[gid.x]);
}

// ============================================================================
// Coefficient-wise Multiply-Accumulate (MAC) in NTT Domain
// ============================================================================

@compute @workgroup_size(256)
fn poly_pointwise_mac_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    let prod = mont_mul_dilithium(a[gid.x], b[gid.x]);
    result[gid.x] = result[gid.x] + prod;
}

@compute @workgroup_size(256)
fn poly_pointwise_mac_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    let prod = mont_mul_kyber(a[gid.x], b[gid.x]);
    result[gid.x] = result[gid.x] + prod;
}

// ============================================================================
// Polynomial Addition and Subtraction
// ============================================================================

@compute @workgroup_size(256)
fn poly_add_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    result[gid.x] = a[gid.x] + b[gid.x];
}

@compute @workgroup_size(256)
fn poly_add_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    result[gid.x] = a[gid.x] + b[gid.x];
}

@compute @workgroup_size(256)
fn poly_sub_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    result[gid.x] = a[gid.x] - b[gid.x];
}

@compute @workgroup_size(256)
fn poly_sub_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    result[gid.x] = a[gid.x] - b[gid.x];
}

// ============================================================================
// Polynomial Reduction
// ============================================================================

@compute @workgroup_size(256)
fn poly_reduce_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    result[gid.x] = cond_sub_dilithium(result[gid.x]);
}

@compute @workgroup_size(256)
fn poly_reduce_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    result[gid.x] = cond_sub_kyber(result[gid.x]);
}

// ============================================================================
// Scalar Multiplication
// ============================================================================

@compute @workgroup_size(256)
fn poly_scalar_mul_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    let scalar = params.scalar;
    result[gid.x] = mont_mul_dilithium(a[gid.x], scalar);
}

@compute @workgroup_size(256)
fn poly_scalar_mul_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    let scalar = params.scalar;
    result[gid.x] = mont_mul_kyber(a[gid.x], scalar);
}

// ============================================================================
// Vector Polynomial Multiplication (Batch Operations)
// ============================================================================

@compute @workgroup_size(256)
fn poly_vector_mul_acc_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let k = params.k;
    let batch = params.batch;
    
    let total = batch * n;
    if (gid.x >= total) { return; }
    
    let batch_idx = gid.x / n;
    let coef_idx = gid.x % n;
    
    var acc = 0i;
    
    for (var i = 0u; i < k; i = i + 1u) {
        let a_idx = batch_idx * k * n + i * n + coef_idx;
        let b_idx = i * n + coef_idx;
        acc = acc + mont_mul_dilithium(a[a_idx], b[b_idx]);
    }
    
    result[gid.x] = acc;
}

@compute @workgroup_size(256)
fn poly_vector_mul_acc_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let k = params.k;
    let batch = params.batch;
    
    let total = batch * n;
    if (gid.x >= total) { return; }
    
    let batch_idx = gid.x / n;
    let coef_idx = gid.x % n;
    
    var acc = 0i;
    
    for (var i = 0u; i < k; i = i + 1u) {
        let a_idx = batch_idx * k * n + i * n + coef_idx;
        let b_idx = i * n + coef_idx;
        acc = acc + mont_mul_kyber(a[a_idx], b[b_idx]);
    }
    
    result[gid.x] = barrett_reduce_kyber(acc);
}

// ============================================================================
// Kyber-Specific: Basemul (NTT multiplication with special structure)
// ============================================================================

// Additional binding for zetas
@group(0) @binding(4) var<storage, read> zetas: array<i32>;

@compute @workgroup_size(128)
fn kyber_basemul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let pair_count = n / 2u;
    
    if (gid.x >= pair_count) { return; }
    
    let idx = gid.x * 2u;
    
    let a0 = a[idx];
    let a1 = a[idx + 1u];
    let b0 = b[idx];
    let b1 = b[idx + 1u];
    let zeta = zetas[64u + gid.x];
    
    // Compute (a0 + a1*X) * (b0 + b1*X) mod (X^2 - zeta)
    // = (a0*b0 + a1*b1*zeta) + (a0*b1 + a1*b0)*X
    
    var r0 = mont_mul_kyber(a0, b0);
    var t = mont_mul_kyber(a1, b1);
    t = mont_mul_kyber(t, zeta);
    r0 = r0 + t;
    
    var r1 = mont_mul_kyber(a0, b1);
    t = mont_mul_kyber(a1, b0);
    r1 = r1 + t;
    
    result[idx] = r0;
    result[idx + 1u] = r1;
}

// ============================================================================
// Dilithium-Specific: Power2Round
// ============================================================================

// For Power2Round, we need separate output buffers
@group(0) @binding(5) var<storage, read_write> a0_out: array<i32>;

@compute @workgroup_size(256)
fn dilithium_power2round(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size) { return; }
    
    let d = params.d;
    
    var val = a[gid.x];
    
    // Ensure val is in [0, q)
    val = cond_sub_dilithium(val);
    
    // a0 = centered remainder of a mod 2^D
    let mask = (1i << d) - 1i;
    var a0_val = val & mask;
    
    // Center a0 to [-2^(D-1), 2^(D-1)]
    if (a0_val > (1i << (d - 1u))) {
        a0_val = a0_val - (1i << d);
    }
    
    // a1 = (a - a0) / 2^D
    let a1_val = (val - a0_val) >> d;
    
    result[gid.x] = a1_val;
    a0_out[gid.x] = a0_val;
}

// ============================================================================
// Matrix Polynomial Multiplication
// ============================================================================

@compute @workgroup_size(256)
fn poly_matrix_mul_ntt_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let m = params.m;
    let k_dim = params.k;
    let p = params.p;
    
    // gid.x = coefficient index (0 to n-1)
    // gid.y = output matrix index (flattened i*p + j)
    
    if (gid.x >= n) { return; }
    
    let out_idx = gid.y;
    if (out_idx >= m * p) { return; }
    
    let i = out_idx / p;
    let j = out_idx % p;
    let coef = gid.x;
    
    var acc = 0i;
    
    for (var k = 0u; k < k_dim; k = k + 1u) {
        let a_poly_idx = i * k_dim + k;
        let b_poly_idx = k * p + j;
        
        let a_coef = a[a_poly_idx * n + coef];
        let b_coef = b[b_poly_idx * n + coef];
        
        acc = acc + mont_mul_dilithium(a_coef, b_coef);
    }
    
    result[out_idx * n + coef] = acc;
}

@compute @workgroup_size(256)
fn poly_matrix_mul_ntt_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let m = params.m;
    let k_dim = params.k;
    let p = params.p;
    
    if (gid.x >= n) { return; }
    
    let out_idx = gid.y;
    if (out_idx >= m * p) { return; }
    
    let i = out_idx / p;
    let j = out_idx % p;
    let coef = gid.x;
    
    var acc = 0i;
    
    for (var k = 0u; k < k_dim; k = k + 1u) {
        let a_poly_idx = i * k_dim + k;
        let b_poly_idx = k * p + j;
        
        let a_coef = a[a_poly_idx * n + coef];
        let b_coef = b[b_poly_idx * n + coef];
        
        acc = acc + mont_mul_kyber(a_coef, b_coef);
    }
    
    result[out_idx * n + coef] = barrett_reduce_kyber(acc);
}
