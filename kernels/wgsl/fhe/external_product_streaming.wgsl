// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// External Product Streaming: Level-by-Level Processing (WebGPU/WGSL)
// Memory-efficient external product when GGSW is too large to cache
// Processes one decomposition level per kernel dispatch
// Compatible with Metal/Vulkan/D3D12 via Dawn/wgpu

// ============================================================================
// 64-bit Integer Emulation
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

fn mod_mul(a: U64, b: U64, Q: U64) -> U64 {
    // Simplified for 32-bit effective moduli
    let prod = u32_mul_wide(a.lo, b.lo);
    if (Q.hi == 0u && Q.lo != 0u) {
        let result = prod.lo % Q.lo;
        return U64(result, 0u);
    }
    return prod;
}

// ============================================================================
// Signed Decomposition
// ============================================================================

fn signed_decomp_digit(val_lo: u32, val_hi: u32, level: u32, base_log: u32) -> i32 {
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

    // Add half_Bg for rounding
    digit = (digit + half_Bg) & mask;

    return i32(digit) - i32(half_Bg);
}

// ============================================================================
// Parameters
// ============================================================================

struct StreamingParams {
    N: u32,           // Polynomial degree
    k: u32,           // GLWE dimension
    l: u32,           // Decomposition levels
    base_log: u32,    // Bg = 2^base_log
    level: u32,       // Current level being processed
    batch_size: u32,  // Batch size
    Q_lo: u32,        // Modulus low
    Q_hi: u32,        // Modulus high
}

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> glwe_in: array<u32>;             // [(k+1)][N][2]
@group(0) @binding(1) var<storage, read> ggsw: array<u32>;                // [(k+1)][l][(k+1)][N][2]
@group(0) @binding(2) var<storage, read_write> glwe_out: array<u32>;      // [(k+1)][N][2]
@group(0) @binding(3) var<storage, read_write> temp_acc: array<u32>;      // [(k+1)][N][2]
@group(0) @binding(4) var<uniform> params: StreamingParams;

// ============================================================================
// External Product Streaming Kernel
// ============================================================================

@compute @workgroup_size(256)
fn external_product_streaming(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let coeff_idx = gid.x;
    let out_poly = wgid.y;

    let N = params.N;
    let k = params.k;
    let l = params.l;
    let level = params.level;
    let Q = U64(params.Q_lo, params.Q_hi);

    if (coeff_idx >= N || out_poly > k) { return; }

    // Load current accumulator (or initialize to zero for level 0)
    var acc: U64;
    if (level == 0u) {
        acc = u64_zero();
    } else {
        let acc_idx = out_poly * N * 2u + coeff_idx * 2u;
        acc = U64(temp_acc[acc_idx], temp_acc[acc_idx + 1u]);
    }

    // Process this level for all input polynomials
    for (var in_poly = 0u; in_poly <= k; in_poly++) {
        let glwe_idx = in_poly * N * 2u + coeff_idx * 2u;
        let glwe_lo = glwe_in[glwe_idx];
        let glwe_hi = glwe_in[glwe_idx + 1u];

        let digit = signed_decomp_digit(glwe_lo, glwe_hi, level, params.base_log);

        if (digit != 0) {
            // GGSW layout: [in_poly][level][out_poly][coeff_idx][2]
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

    // Store to temp or final output
    let out_idx = out_poly * N * 2u + coeff_idx * 2u;
    if (level == l - 1u) {
        glwe_out[out_idx] = acc.lo;
        glwe_out[out_idx + 1u] = acc.hi;
    } else {
        temp_acc[out_idx] = acc.lo;
        temp_acc[out_idx + 1u] = acc.hi;
    }
}

// ============================================================================
// Batched External Product Streaming
// ============================================================================

@group(0) @binding(5) var<storage, read> glwe_batch: array<u32>;           // [batch][(k+1)][N][2]
@group(0) @binding(6) var<storage, read_write> output_batch: array<u32>;   // [batch][(k+1)][N][2]
@group(0) @binding(7) var<storage, read_write> temp_acc_batch: array<u32>; // [batch][(k+1)][N][2]

@compute @workgroup_size(256)
fn external_product_streaming_batch(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let coeff_idx = gid.x;
    let out_poly = wgid.y;
    let batch_idx = wgid.z;

    let N = params.N;
    let k = params.k;
    let l = params.l;
    let level = params.level;
    let batch_size = params.batch_size;
    let Q = U64(params.Q_lo, params.Q_hi);

    if (coeff_idx >= N || out_poly > k || batch_idx >= batch_size) { return; }

    let glwe_stride = (k + 1u) * N * 2u;
    let batch_offset = batch_idx * glwe_stride;

    var acc: U64;
    if (level == 0u) {
        acc = u64_zero();
    } else {
        let acc_idx = batch_offset + out_poly * N * 2u + coeff_idx * 2u;
        acc = U64(temp_acc_batch[acc_idx], temp_acc_batch[acc_idx + 1u]);
    }

    for (var in_poly = 0u; in_poly <= k; in_poly++) {
        let glwe_idx = batch_offset + in_poly * N * 2u + coeff_idx * 2u;
        let glwe_lo = glwe_batch[glwe_idx];
        let glwe_hi = glwe_batch[glwe_idx + 1u];

        let digit = signed_decomp_digit(glwe_lo, glwe_hi, level, params.base_log);

        if (digit != 0) {
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

    let out_idx = batch_offset + out_poly * N * 2u + coeff_idx * 2u;
    if (level == l - 1u) {
        output_batch[out_idx] = acc.lo;
        output_batch[out_idx + 1u] = acc.hi;
    } else {
        temp_acc_batch[out_idx] = acc.lo;
        temp_acc_batch[out_idx + 1u] = acc.hi;
    }
}

// ============================================================================
// CMux Streaming
// ============================================================================

@group(0) @binding(8) var<storage, read> glwe_d0: array<u32>;              // [(k+1)][N][2]
@group(0) @binding(9) var<storage, read> diff: array<u32>;                 // [(k+1)][N][2] (d1 - d0)
@group(0) @binding(10) var<storage, read> ggsw_bit: array<u32>;            // [(k+1)][l][(k+1)][N][2]
@group(0) @binding(11) var<storage, read_write> cmux_result: array<u32>;   // [(k+1)][N][2]

@compute @workgroup_size(256)
fn cmux_streaming(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let coeff_idx = gid.x;
    let out_poly = wgid.y;

    let N = params.N;
    let k = params.k;
    let l = params.l;
    let level = params.level;
    let Q = U64(params.Q_lo, params.Q_hi);

    if (coeff_idx >= N || out_poly > k) { return; }

    var acc: U64;
    if (level == 0u) {
        acc = u64_zero();
    } else {
        let acc_idx = out_poly * N * 2u + coeff_idx * 2u;
        acc = U64(temp_acc[acc_idx], temp_acc[acc_idx + 1u]);
    }

    for (var in_poly = 0u; in_poly <= k; in_poly++) {
        let diff_idx = in_poly * N * 2u + coeff_idx * 2u;
        let diff_lo = diff[diff_idx];
        let diff_hi = diff[diff_idx + 1u];

        let digit = signed_decomp_digit(diff_lo, diff_hi, level, params.base_log);

        if (digit != 0) {
            let ggsw_idx = ((in_poly * l + level) * (k + 1u) + out_poly) * N * 2u + coeff_idx * 2u;
            let ggsw_val = U64(ggsw_bit[ggsw_idx], ggsw_bit[ggsw_idx + 1u]);

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
    if (level == l - 1u) {
        // Final level: add d0 to external product
        let d0_idx = out_poly * N * 2u + coeff_idx * 2u;
        let d0_val = U64(glwe_d0[d0_idx], glwe_d0[d0_idx + 1u]);
        let result = mod_add(d0_val, acc, Q);
        cmux_result[out_idx] = result.lo;
        cmux_result[out_idx + 1u] = result.hi;
    } else {
        temp_acc[out_idx] = acc.lo;
        temp_acc[out_idx + 1u] = acc.hi;
    }
}

// ============================================================================
// Compute Difference (d1 - d0)
// ============================================================================

@group(0) @binding(12) var<storage, read> glwe_d1: array<u32>;             // [(k+1)][N][2]
@group(0) @binding(13) var<storage, read_write> diff_out: array<u32>;      // [(k+1)][N][2]

@compute @workgroup_size(256)
fn compute_diff(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let total = (params.k + 1u) * params.N;
    let Q = U64(params.Q_lo, params.Q_hi);

    if (idx >= total) { return; }

    let v0 = U64(glwe_d0[idx * 2u], glwe_d0[idx * 2u + 1u]);
    let v1 = U64(glwe_d1[idx * 2u], glwe_d1[idx * 2u + 1u]);
    let result = mod_sub(v1, v0, Q);

    diff_out[idx * 2u] = result.lo;
    diff_out[idx * 2u + 1u] = result.hi;
}
