// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Portable Modular Arithmetic Kernels for FHE/Lattice Cryptography
// Works on WebGPU (Metal/Vulkan/D3D12 via Dawn/wgpu)
//
// Part of the Lux Network GPU acceleration library

// ============================================================================
// 64-bit Emulation Helpers (WGSL only has u32/i32)
// ============================================================================

struct U64 {
    lo: u32,
    hi: u32,
}

struct U128 {
    w0: u32,  // bits 0-31
    w1: u32,  // bits 32-63
    w2: u32,  // bits 64-95
    w3: u32,  // bits 96-127
}

fn u64_from_u32(x: u32) -> U64 {
    return U64(x, 0u);
}

fn u64_add(a: U64, b: U64) -> U64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    return U64(lo, a.hi + b.hi + carry);
}

fn u64_sub(a: U64, b: U64) -> U64 {
    let borrow = select(0u, 1u, a.lo < b.lo);
    return U64(a.lo - b.lo, a.hi - b.hi - borrow);
}

fn u64_gte(a: U64, b: U64) -> bool {
    return (a.hi > b.hi) || (a.hi == b.hi && a.lo >= b.lo);
}

fn u64_lt(a: U64, b: U64) -> bool {
    return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
}

fn u64_eq(a: U64, b: U64) -> bool {
    return a.lo == b.lo && a.hi == b.hi;
}

fn u64_is_zero(a: U64) -> bool {
    return a.lo == 0u && a.hi == 0u;
}

fn u64_shl(a: U64, shift: u32) -> U64 {
    if (shift == 0u) { return a; }
    if (shift >= 64u) { return U64(0u, 0u); }
    if (shift >= 32u) {
        return U64(0u, a.lo << (shift - 32u));
    }
    let lo = a.lo << shift;
    let hi = (a.hi << shift) | (a.lo >> (32u - shift));
    return U64(lo, hi);
}

fn u64_shr(a: U64, shift: u32) -> U64 {
    if (shift == 0u) { return a; }
    if (shift >= 64u) { return U64(0u, 0u); }
    if (shift >= 32u) {
        return U64(a.hi >> (shift - 32u), 0u);
    }
    let hi = a.hi >> shift;
    let lo = (a.lo >> shift) | (a.hi << (32u - shift));
    return U64(lo, hi);
}

// ============================================================================
// 64x64 -> 128 Multiplication
// ============================================================================

fn u64_mul_128(a: U64, b: U64) -> U128 {
    // Karatsuba-style decomposition
    let a0 = a.lo & 0xFFFFu;
    let a1 = a.lo >> 16u;
    let a2 = a.hi & 0xFFFFu;
    let a3 = a.hi >> 16u;
    
    let b0 = b.lo & 0xFFFFu;
    let b1 = b.lo >> 16u;
    let b2 = b.hi & 0xFFFFu;
    let b3 = b.hi >> 16u;
    
    // Partial products
    var acc0 = a0 * b0;
    var acc1 = a0 * b1 + a1 * b0;
    var acc2 = a0 * b2 + a1 * b1 + a2 * b0;
    var acc3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;
    var acc4 = a1 * b3 + a2 * b2 + a3 * b1;
    var acc5 = a2 * b3 + a3 * b2;
    var acc6 = a3 * b3;
    
    // Propagate carries
    acc1 += acc0 >> 16u; acc0 &= 0xFFFFu;
    acc2 += acc1 >> 16u; acc1 &= 0xFFFFu;
    acc3 += acc2 >> 16u; acc2 &= 0xFFFFu;
    acc4 += acc3 >> 16u; acc3 &= 0xFFFFu;
    acc5 += acc4 >> 16u; acc4 &= 0xFFFFu;
    acc6 += acc5 >> 16u; acc5 &= 0xFFFFu;
    
    let w0 = acc0 | (acc1 << 16u);
    let w1 = acc2 | (acc3 << 16u);
    let w2 = acc4 | (acc5 << 16u);
    let w3 = acc6;
    
    return U128(w0, w1, w2, w3);
}

// ============================================================================
// Barrett Reduction
// ============================================================================

// Barrett reduction: compute x mod Q using precomputed mu = floor(2^128 / Q)
fn barrett_reduce_128(x: U128, Q: U64, mu: U64) -> U64 {
    // Approximate quotient: q = floor(x * mu / 2^128)
    // We use the high 64 bits of x for approximation
    let x_hi = U64(x.w2, x.w3);
    
    // q_approx = (x_hi * mu) >> 64
    let prod = u64_mul_128(x_hi, mu);
    let q_approx = U64(prod.w2, prod.w3);
    
    // r = x - q_approx * Q
    let qQ = u64_mul_128(q_approx, Q);
    var r = U64(x.w0, x.w1);
    
    // Subtract q*Q from lower 64 bits of x
    let sub_lo = U64(qQ.w0, qQ.w1);
    if (u64_gte(r, sub_lo)) {
        r = u64_sub(r, sub_lo);
    } else {
        // Handle borrow from higher words
        r = u64_add(r, Q);
        r = u64_sub(r, sub_lo);
    }
    
    // Correction: r might still be >= Q (at most 2 corrections needed)
    if (u64_gte(r, Q)) { r = u64_sub(r, Q); }
    if (u64_gte(r, Q)) { r = u64_sub(r, Q); }
    
    return r;
}

// ============================================================================
// Kernel Bindings
// ============================================================================

struct ModParams {
    Q_lo: u32,
    Q_hi: u32,
    mu_lo: u32,
    mu_hi: u32,
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read_write> result: array<u32>;
@group(0) @binding(1) var<storage, read> input_a: array<u32>;
@group(0) @binding(2) var<storage, read> input_b: array<u32>;
@group(0) @binding(3) var<uniform> params: ModParams;

// ============================================================================
// Modular Addition: result = (a + b) mod Q
// ============================================================================

@compute @workgroup_size(256)
fn mod_add_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }
    
    let Q = U64(params.Q_lo, params.Q_hi);
    let idx = gid.x * 2u;
    
    let a = U64(input_a[idx], input_a[idx + 1u]);
    let b = U64(input_b[idx], input_b[idx + 1u]);
    
    var sum = u64_add(a, b);
    if (u64_gte(sum, Q)) {
        sum = u64_sub(sum, Q);
    }
    // Handle overflow case
    if (u64_lt(sum, a) && u64_lt(sum, b)) {
        sum = u64_sub(sum, Q);
    }
    
    result[idx] = sum.lo;
    result[idx + 1u] = sum.hi;
}

// ============================================================================
// Modular Subtraction: result = (a - b) mod Q
// ============================================================================

@compute @workgroup_size(256)
fn mod_sub_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }
    
    let Q = U64(params.Q_lo, params.Q_hi);
    let idx = gid.x * 2u;
    
    let a = U64(input_a[idx], input_a[idx + 1u]);
    let b = U64(input_b[idx], input_b[idx + 1u]);
    
    var diff: U64;
    if (u64_gte(a, b)) {
        diff = u64_sub(a, b);
    } else {
        diff = u64_sub(u64_add(a, Q), b);
    }
    
    result[idx] = diff.lo;
    result[idx + 1u] = diff.hi;
}

// ============================================================================
// Modular Multiplication: result = (a * b) mod Q
// ============================================================================

@compute @workgroup_size(256)
fn mod_mul_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }
    
    let Q = U64(params.Q_lo, params.Q_hi);
    let mu = U64(params.mu_lo, params.mu_hi);
    let idx = gid.x * 2u;
    
    let a = U64(input_a[idx], input_a[idx + 1u]);
    let b = U64(input_b[idx], input_b[idx + 1u]);
    
    // 64x64 -> 128-bit product
    let prod = u64_mul_128(a, b);
    
    // Barrett reduction
    let r = barrett_reduce_128(prod, Q, mu);
    
    result[idx] = r.lo;
    result[idx + 1u] = r.hi;
}

// ============================================================================
// Modular Negation: result = -a mod Q = Q - a
// ============================================================================

@compute @workgroup_size(256)
fn mod_neg_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }
    
    let Q = U64(params.Q_lo, params.Q_hi);
    let idx = gid.x * 2u;
    
    let a = U64(input_a[idx], input_a[idx + 1u]);
    
    var neg: U64;
    if (u64_is_zero(a)) {
        neg = a;
    } else {
        neg = u64_sub(Q, a);
    }
    
    result[idx] = neg.lo;
    result[idx + 1u] = neg.hi;
}

// ============================================================================
// Fused Multiply-Add: result = (a * b + c) mod Q
// ============================================================================

@compute @workgroup_size(256)
fn mod_mul_add_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }
    
    let Q = U64(params.Q_lo, params.Q_hi);
    let mu = U64(params.mu_lo, params.mu_hi);
    let idx = gid.x * 2u;
    
    let a = U64(input_a[idx], input_a[idx + 1u]);
    let b = U64(input_b[idx], input_b[idx + 1u]);
    // c comes from result buffer (in-place accumulation)
    let c = U64(result[idx], result[idx + 1u]);
    
    // Compute a*b mod Q
    let prod = u64_mul_128(a, b);
    let ab = barrett_reduce_128(prod, Q, mu);
    
    // Add c
    var sum = u64_add(ab, c);
    if (u64_gte(sum, Q)) {
        sum = u64_sub(sum, Q);
    }
    
    result[idx] = sum.lo;
    result[idx + 1u] = sum.hi;
}
