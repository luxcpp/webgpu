// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Portable Number Theoretic Transform (NTT) kernel in WGSL
// Works on WebGPU (Metal/Vulkan/D3D12 via Dawn/wgpu)
//
// Part of the Lux Network GPU acceleration library

// ============================================================================
// Modular Arithmetic Primitives
// ============================================================================

// Note: WGSL doesn't have native u64, so we use u32 pairs for 64-bit math
// For full 64-bit support, we emulate with (low, high) pairs

struct U64 {
    lo: u32,
    hi: u32,
}

// Add two 64-bit numbers
fn u64_add(a: U64, b: U64) -> U64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    let hi = a.hi + b.hi + carry;
    return U64(lo, hi);
}

// Subtract two 64-bit numbers (a - b)
fn u64_sub(a: U64, b: U64) -> U64 {
    let borrow = select(0u, 1u, a.lo < b.lo);
    let lo = a.lo - b.lo;
    let hi = a.hi - b.hi - borrow;
    return U64(lo, hi);
}

// Compare a >= b
fn u64_gte(a: U64, b: U64) -> bool {
    if (a.hi > b.hi) { return true; }
    if (a.hi < b.hi) { return false; }
    return a.lo >= b.lo;
}

// Multiply two u32 -> U64
fn u32_mul_wide(a: u32, b: u32) -> U64 {
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
    
    let lo = p0 + (mid << 16u);
    let lo_carry = select(0u, 1u, lo < p0);
    let hi = p3 + (mid >> 16u) + mid_carry + lo_carry;
    
    return U64(lo, hi);
}

// Modular addition: (a + b) mod Q
fn mod_add(a: U64, b: U64, Q: U64) -> U64 {
    var sum = u64_add(a, b);
    if (u64_gte(sum, Q)) {
        sum = u64_sub(sum, Q);
    }
    return sum;
}

// Modular subtraction: (a - b) mod Q  
fn mod_sub(a: U64, b: U64, Q: U64) -> U64 {
    if (u64_gte(a, b)) {
        return u64_sub(a, b);
    } else {
        return u64_sub(u64_add(a, Q), b);
    }
}

// ============================================================================
// Butterfly Operations
// ============================================================================

// Cooley-Tukey butterfly for forward NTT
// X = x0 + w * x1
// Y = x0 - w * x1
fn ct_butterfly(x0: ptr<function, U64>, x1: ptr<function, U64>, w: U64, Q: U64) {
    // Simplified: for demonstration, using 32-bit modmul
    // Full implementation needs 128-bit intermediate
    let t = mod_mul_approx(*x1, w, Q);
    let new_x1 = mod_sub(*x0, t, Q);
    let new_x0 = mod_add(*x0, t, Q);
    *x0 = new_x0;
    *x1 = new_x1;
}

// Gentleman-Sande butterfly for inverse NTT
// X = x0 + x1
// Y = (x0 - x1) * w
fn gs_butterfly(x0: ptr<function, U64>, x1: ptr<function, U64>, w: U64, Q: U64) {
    let t = mod_sub(*x0, *x1, Q);
    let new_x0 = mod_add(*x0, *x1, Q);
    let new_x1 = mod_mul_approx(t, w, Q);
    *x0 = new_x0;
    *x1 = new_x1;
}

// Approximate modular multiplication (32-bit for now)
fn mod_mul_approx(a: U64, b: U64, Q: U64) -> U64 {
    // For 32-bit moduli, use lo parts only
    let prod = u32_mul_wide(a.lo, b.lo);
    // Barrett reduction approximation
    let q_lo = Q.lo;
    if (q_lo == 0u) { return U64(0u, 0u); }
    let result = prod.lo % q_lo;
    return U64(result, 0u);
}

// ============================================================================
// Kernel Bindings
// ============================================================================

struct NTTParams {
    Q_lo: u32,
    Q_hi: u32,
    mu_lo: u32,
    mu_hi: u32,
    N: u32,
    batch: u32,
    stage: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read> twiddles: array<u32>;
@group(0) @binding(2) var<uniform> params: NTTParams;

var<workgroup> shared_data: array<u32, 4096>;  // Max N = 2048 * 2 for U64

// ============================================================================
// Forward NTT Kernel (staged)
// ============================================================================

@compute @workgroup_size(256)
fn ntt_forward_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params.N;
    let batch = params.batch;
    let stage = params.stage;
    let Q = U64(params.Q_lo, params.Q_hi);
    
    let total = batch * (N / 2u);
    if (gid.x >= total) { return; }
    
    let batch_idx = gid.x / (N / 2u);
    let k = gid.x % (N / 2u);
    
    let m = 1u << (stage + 1u);
    let half_m = m >> 1u;
    let j = k / half_m;
    let i = k % half_m;
    let idx0 = j * m + i;
    let idx1 = idx0 + half_m;
    
    let base = batch_idx * N * 2u;  // *2 for U64 storage
    
    // Load values
    var x0 = U64(data[base + idx0 * 2u], data[base + idx0 * 2u + 1u]);
    var x1 = U64(data[base + idx1 * 2u], data[base + idx1 * 2u + 1u]);
    
    // Load twiddle
    let tw_idx = (half_m + i) * 2u;
    let w = U64(twiddles[tw_idx], twiddles[tw_idx + 1u]);
    
    // Butterfly
    ct_butterfly(&x0, &x1, w, Q);
    
    // Store results
    data[base + idx0 * 2u] = x0.lo;
    data[base + idx0 * 2u + 1u] = x0.hi;
    data[base + idx1 * 2u] = x1.lo;
    data[base + idx1 * 2u + 1u] = x1.hi;
}

// ============================================================================
// Inverse NTT Kernel (staged)
// ============================================================================

@compute @workgroup_size(256)
fn ntt_inverse_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params.N;
    let batch = params.batch;
    let stage = params.stage;
    let Q = U64(params.Q_lo, params.Q_hi);
    
    let total = batch * (N / 2u);
    if (gid.x >= total) { return; }
    
    let batch_idx = gid.x / (N / 2u);
    let k = gid.x % (N / 2u);
    
    let m = 1u << stage;
    let half_m = m >> 1u;
    let j = k / half_m;
    let i = k % half_m;
    let idx0 = j * m + i;
    let idx1 = idx0 + half_m;
    
    let base = batch_idx * N * 2u;
    
    // Load values
    var x0 = U64(data[base + idx0 * 2u], data[base + idx0 * 2u + 1u]);
    var x1 = U64(data[base + idx1 * 2u], data[base + idx1 * 2u + 1u]);
    
    // Load inverse twiddle
    let tw_idx = (half_m + i) * 2u;
    let w = U64(twiddles[tw_idx], twiddles[tw_idx + 1u]);
    
    // Butterfly
    gs_butterfly(&x0, &x1, w, Q);
    
    // Store results
    data[base + idx0 * 2u] = x0.lo;
    data[base + idx0 * 2u + 1u] = x0.hi;
    data[base + idx1 * 2u] = x1.lo;
    data[base + idx1 * 2u + 1u] = x1.hi;
}

// ============================================================================
// Pointwise Modular Multiplication
// ============================================================================

@compute @workgroup_size(256)
fn ntt_pointwise_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.N * params.batch;
    if (gid.x >= size) { return; }
    
    let Q = U64(params.Q_lo, params.Q_hi);
    let idx = gid.x * 2u;
    
    // Load a and b (interleaved in data buffer for this kernel)
    // Assumes data layout: [a0, a1, ..., b0, b1, ...]
    let a = U64(data[idx], data[idx + 1u]);
    let b = U64(twiddles[idx], twiddles[idx + 1u]);  // b in twiddles buffer
    
    let result = mod_mul_approx(a, b, Q);
    
    data[idx] = result.lo;
    data[idx + 1u] = result.hi;
}

// ============================================================================
// Inverse N Scaling
// ============================================================================

@compute @workgroup_size(256)
fn ntt_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.N * params.batch;
    if (gid.x >= size) { return; }
    
    let Q = U64(params.Q_lo, params.Q_hi);
    let inv_N = U64(params.mu_lo, params.mu_hi);  // Reusing mu for inv_N
    let idx = gid.x * 2u;
    
    let val = U64(data[idx], data[idx + 1u]);
    let result = mod_mul_approx(val, inv_N, Q);
    
    data[idx] = result.lo;
    data[idx + 1u] = result.hi;
}
