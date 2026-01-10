// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// TFHE External Product: GGSW x GLWE -> GLWE for WebGPU
// Core operation for blind rotation in programmable bootstrapping
// Compatible with Metal/Vulkan/D3D12 via Dawn/wgpu

// ============================================================================
// 64-bit Integer Emulation (WGSL only has u32)
// ============================================================================

struct U64 {
    lo: u32,
    hi: u32,
}

fn u64_zero() -> U64 {
    return U64(0u, 0u);
}

fn u64_from_u32(x: u32) -> U64 {
    return U64(x, 0u);
}

fn u64_add(a: U64, b: U64) -> U64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    let hi = a.hi + b.hi + carry;
    return U64(lo, hi);
}

fn u64_sub(a: U64, b: U64) -> U64 {
    let borrow = select(0u, 1u, a.lo < b.lo);
    let lo = a.lo - b.lo;
    let hi = a.hi - b.hi - borrow;
    return U64(lo, hi);
}

fn u64_gte(a: U64, b: U64) -> bool {
    if (a.hi > b.hi) { return true; }
    if (a.hi < b.hi) { return false; }
    return a.lo >= b.lo;
}

fn u64_is_zero(a: U64) -> bool {
    return a.lo == 0u && a.hi == 0u;
}

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

// ============================================================================
// Modular Arithmetic
// ============================================================================

fn mod_add(a: U64, b: U64, Q: U64) -> U64 {
    var sum = u64_add(a, b);
    // Check for overflow
    let overflow = sum.hi < a.hi || (sum.hi == a.hi && sum.lo < a.lo);
    if (overflow || u64_gte(sum, Q)) {
        sum = u64_sub(sum, Q);
    }
    return sum;
}

fn mod_sub(a: U64, b: U64, Q: U64) -> U64 {
    if (u64_gte(a, b)) {
        return u64_sub(a, b);
    }
    return u64_sub(u64_add(a, Q), b);
}

fn mod_neg(a: U64, Q: U64) -> U64 {
    if (u64_is_zero(a)) { return a; }
    return u64_sub(Q, a);
}

// Modular multiplication (simplified for 32-bit effective moduli)
fn mod_mul(a: U64, b: U64, Q: U64) -> U64 {
    // For production, implement full 128-bit reduction
    let prod = u32_mul_wide(a.lo, b.lo);
    if (Q.hi == 0u && Q.lo != 0u) {
        let result = prod.lo % Q.lo;
        return U64(result, 0u);
    }
    return prod;
}

// ============================================================================
// External Product Parameters
// ============================================================================

struct ExternalProductParams {
    N: u32,           // Polynomial degree (1024)
    k: u32,           // GLWE dimension (1)
    l: u32,           // Decomposition levels (3)
    base_log: u32,    // Base log (8)
    batch: u32,       // Batch size
    Q_lo: u32,        // Modulus low bits
    Q_hi: u32,        // Modulus high bits
    ntt_mode: u32,    // 0 = coefficient, 1 = NTT domain
}

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> glwe_in: array<u32>;        // Input GLWE [(k+1)][N][2]
@group(0) @binding(1) var<storage, read> ggsw: array<u32>;           // GGSW [(k+1)][l][(k+1)][N][2]
@group(0) @binding(2) var<storage, read_write> glwe_out: array<u32>; // Output GLWE [(k+1)][N][2]
@group(0) @binding(3) var<uniform> params: ExternalProductParams;

var<workgroup> glwe_cache: array<u32, 4096>;  // Cache for GLWE (N * 2 * (k+1))

// ============================================================================
// Signed Gadget Decomposition
// ============================================================================

fn signed_decompose_digit(val_lo: u32, val_hi: u32, level: u32, base_log: u32) -> i32 {
    let Bg = 1u << base_log;
    let half_Bg = Bg >> 1u;
    let mask = Bg - 1u;
    
    let shift = 64u - (level + 1u) * base_log;
    
    var digit: u32;
    if (shift >= 32u) {
        digit = (val_hi >> (shift - 32u)) & mask;
    } else if (shift == 0u) {
        digit = val_lo & mask;
    } else {
        let lo_part = val_lo >> shift;
        let hi_part = val_hi << (32u - shift);
        digit = (lo_part | hi_part) & mask;
    }
    
    if (digit >= half_Bg) {
        return i32(digit) - i32(Bg);
    }
    return i32(digit);
}

// Full decomposition with carry propagation
struct DecompResult {
    d0: i32,
    d1: i32,
    d2: i32,
    d3: i32,
    d4: i32,
    d5: i32,
    d6: i32,
    d7: i32,
}

fn signed_decompose_full(val_lo: u32, val_hi: u32, l: u32, base_log: u32) -> DecompResult {
    var result: DecompResult;
    let Bg = 1u << base_log;
    let half_Bg = Bg >> 1u;
    let mask = Bg - 1u;
    
    var carry: u32 = 0u;
    
    for (var level = 0u; level < l && level < 8u; level++) {
        let shift = 64u - (level + 1u) * base_log;
        
        var digit: u32;
        if (shift >= 32u) {
            digit = ((val_hi >> (shift - 32u)) + carry) & mask;
        } else if (shift == 0u) {
            digit = (val_lo + carry) & mask;
        } else {
            let lo_part = val_lo >> shift;
            let hi_part = val_hi << (32u - shift);
            digit = ((lo_part | hi_part) + carry) & mask;
        }
        carry = 0u;
        
        var signed_digit: i32;
        if (digit >= half_Bg) {
            signed_digit = i32(digit) - i32(Bg);
            carry = 1u;
        } else {
            signed_digit = i32(digit);
        }
        
        switch (level) {
            case 0u: { result.d0 = signed_digit; }
            case 1u: { result.d1 = signed_digit; }
            case 2u: { result.d2 = signed_digit; }
            case 3u: { result.d3 = signed_digit; }
            case 4u: { result.d4 = signed_digit; }
            case 5u: { result.d5 = signed_digit; }
            case 6u: { result.d6 = signed_digit; }
            case 7u: { result.d7 = signed_digit; }
            default: {}
        }
    }
    
    return result;
}

fn get_decomp_digit(decomp: DecompResult, level: u32) -> i32 {
    switch (level) {
        case 0u: { return decomp.d0; }
        case 1u: { return decomp.d1; }
        case 2u: { return decomp.d2; }
        case 3u: { return decomp.d3; }
        case 4u: { return decomp.d4; }
        case 5u: { return decomp.d5; }
        case 6u: { return decomp.d6; }
        case 7u: { return decomp.d7; }
        default: { return 0; }
    }
}

// ============================================================================
// External Product in Coefficient Domain
// ============================================================================

@compute @workgroup_size(256)
fn external_product_coefficient(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let coeff_idx = gid.x;
    let out_poly = wgid.y;
    
    let N = params.N;
    let k = params.k;
    let l = params.l;
    let Q = U64(params.Q_lo, params.Q_hi);
    
    if (coeff_idx >= N || out_poly > k) { return; }
    
    var acc = u64_zero();
    
    // For each input polynomial
    for (var in_poly = 0u; in_poly <= k; in_poly++) {
        // Load GLWE coefficient
        let glwe_idx = in_poly * N * 2u + coeff_idx * 2u;
        let val_lo = glwe_in[glwe_idx];
        let val_hi = glwe_in[glwe_idx + 1u];
        
        // Decompose
        let decomp = signed_decompose_full(val_lo, val_hi, l, params.base_log);
        
        // For each decomposition level
        for (var level = 0u; level < l; level++) {
            let digit = get_decomp_digit(decomp, level);
            
            if (digit == 0) { continue; }
            
            // GGSW coefficient
            let ggsw_idx = ((in_poly * l + level) * (k + 1u) + out_poly) * N * 2u + coeff_idx * 2u;
            let ggsw_lo = ggsw[ggsw_idx];
            let ggsw_hi = ggsw[ggsw_idx + 1u];
            let ggsw_val = U64(ggsw_lo, ggsw_hi);
            
            // Multiply
            let abs_digit = u32(select(-digit, digit, digit >= 0));
            let prod = mod_mul(u64_from_u32(abs_digit), ggsw_val, Q);
            
            if (digit > 0) {
                acc = mod_add(acc, prod, Q);
            } else {
                acc = mod_sub(acc, prod, Q);
            }
        }
    }
    
    // Store result
    let out_idx = out_poly * N * 2u + coeff_idx * 2u;
    glwe_out[out_idx] = acc.lo;
    glwe_out[out_idx + 1u] = acc.hi;
}

// ============================================================================
// External Product in NTT Domain (Production)
// ============================================================================

@compute @workgroup_size(256)
fn external_product_ntt(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let coeff_idx = gid.x;
    let out_poly = wgid.y;
    
    let N = params.N;
    let k = params.k;
    let l = params.l;
    let Q = U64(params.Q_lo, params.Q_hi);
    
    if (coeff_idx >= N || out_poly > k) { return; }
    
    // Cache input GLWE in shared memory
    let thread_idx = lid.x;
    let threads = 256u;
    
    for (var i = thread_idx; i < (k + 1u) * N * 2u; i += threads) {
        glwe_cache[i] = glwe_in[i];
    }
    workgroupBarrier();
    
    var acc = u64_zero();
    
    for (var in_poly = 0u; in_poly <= k; in_poly++) {
        let cache_idx = in_poly * N * 2u + coeff_idx * 2u;
        let val_lo = glwe_cache[cache_idx];
        let val_hi = glwe_cache[cache_idx + 1u];
        
        for (var level = 0u; level < l; level++) {
            let digit = signed_decompose_digit(val_lo, val_hi, level, params.base_log);
            
            if (digit == 0) { continue; }
            
            let ggsw_idx = ((in_poly * l + level) * (k + 1u) + out_poly) * N * 2u + coeff_idx * 2u;
            let ggsw_val = U64(ggsw[ggsw_idx], ggsw[ggsw_idx + 1u]);
            
            let abs_digit = u32(select(-digit, digit, digit >= 0));
            let prod = mod_mul(u64_from_u32(abs_digit), ggsw_val, Q);
            
            if (digit > 0) {
                acc = mod_add(acc, prod, Q);
            } else {
                acc = mod_sub(acc, prod, Q);
            }
        }
    }
    
    let out_idx = out_poly * N * 2u + coeff_idx * 2u;
    glwe_out[out_idx] = acc.lo;
    glwe_out[out_idx + 1u] = acc.hi;
}

// ============================================================================
// Batched External Product
// ============================================================================

@group(0) @binding(4) var<storage, read> glwe_batch: array<u32>;
@group(0) @binding(5) var<storage, read_write> output_batch: array<u32>;

@compute @workgroup_size(256)
fn external_product_batch(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let coeff_idx = gid.x;
    let out_poly = wgid.y;
    let batch_idx = wgid.z;
    
    let N = params.N;
    let k = params.k;
    let l = params.l;
    let batch = params.batch;
    let Q = U64(params.Q_lo, params.Q_hi);
    
    if (coeff_idx >= N || out_poly > k || batch_idx >= batch) { return; }
    
    let glwe_stride = (k + 1u) * N * 2u;
    let glwe_offset = batch_idx * glwe_stride;
    
    var acc = u64_zero();
    
    for (var in_poly = 0u; in_poly <= k; in_poly++) {
        let idx = glwe_offset + in_poly * N * 2u + coeff_idx * 2u;
        let val_lo = glwe_batch[idx];
        let val_hi = glwe_batch[idx + 1u];
        
        for (var level = 0u; level < l; level++) {
            let digit = signed_decompose_digit(val_lo, val_hi, level, params.base_log);
            
            if (digit == 0) { continue; }
            
            let ggsw_idx = ((in_poly * l + level) * (k + 1u) + out_poly) * N * 2u + coeff_idx * 2u;
            let ggsw_val = U64(ggsw[ggsw_idx], ggsw[ggsw_idx + 1u]);
            
            let abs_digit = u32(select(-digit, digit, digit >= 0));
            let prod = mod_mul(u64_from_u32(abs_digit), ggsw_val, Q);
            
            if (digit > 0) {
                acc = mod_add(acc, prod, Q);
            } else {
                acc = mod_sub(acc, prod, Q);
            }
        }
    }
    
    let out_idx = glwe_offset + out_poly * N * 2u + coeff_idx * 2u;
    output_batch[out_idx] = acc.lo;
    output_batch[out_idx + 1u] = acc.hi;
}

// ============================================================================
// CMux Operation
// ============================================================================

@group(0) @binding(6) var<storage, read> glwe_d0: array<u32>;
@group(0) @binding(7) var<storage, read> glwe_d1: array<u32>;
@group(0) @binding(8) var<storage, read> ggsw_bit: array<u32>;
@group(0) @binding(9) var<storage, read_write> cmux_result: array<u32>;

@compute @workgroup_size(256)
fn cmux(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let coeff_idx = gid.x;
    let out_poly = wgid.y;
    
    let N = params.N;
    let k = params.k;
    let l = params.l;
    let Q = U64(params.Q_lo, params.Q_hi);
    
    if (coeff_idx >= N || out_poly > k) { return; }
    
    // Compute d1 - d0 in shared memory
    let thread_idx = lid.x;
    let threads = 256u;
    
    for (var p = 0u; p <= k; p++) {
        let idx = p * N * 2u + coeff_idx * 2u;
        let d0 = U64(glwe_d0[idx], glwe_d0[idx + 1u]);
        let d1 = U64(glwe_d1[idx], glwe_d1[idx + 1u]);
        let diff = mod_sub(d1, d0, Q);
        glwe_cache[idx] = diff.lo;
        glwe_cache[idx + 1u] = diff.hi;
    }
    workgroupBarrier();
    
    // External product: GGSW(bit) x (d1 - d0)
    var ext_prod = u64_zero();
    
    for (var in_poly = 0u; in_poly <= k; in_poly++) {
        let cache_idx = in_poly * N * 2u + coeff_idx * 2u;
        let diff_lo = glwe_cache[cache_idx];
        let diff_hi = glwe_cache[cache_idx + 1u];
        
        for (var level = 0u; level < l; level++) {
            let digit = signed_decompose_digit(diff_lo, diff_hi, level, params.base_log);
            
            if (digit == 0) { continue; }
            
            let ggsw_idx = ((in_poly * l + level) * (k + 1u) + out_poly) * N * 2u + coeff_idx * 2u;
            let ggsw_val = U64(ggsw_bit[ggsw_idx], ggsw_bit[ggsw_idx + 1u]);
            
            let abs_digit = u32(select(-digit, digit, digit >= 0));
            let prod = mod_mul(u64_from_u32(abs_digit), ggsw_val, Q);
            
            if (digit > 0) {
                ext_prod = mod_add(ext_prod, prod, Q);
            } else {
                ext_prod = mod_sub(ext_prod, prod, Q);
            }
        }
    }
    
    // result = d0 + external_product
    let d0_idx = out_poly * N * 2u + coeff_idx * 2u;
    let d0_val = U64(glwe_d0[d0_idx], glwe_d0[d0_idx + 1u]);
    let result = mod_add(d0_val, ext_prod, Q);
    
    cmux_result[d0_idx] = result.lo;
    cmux_result[d0_idx + 1u] = result.hi;
}
