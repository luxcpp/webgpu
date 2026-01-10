// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// TFHE LWE Key Switching Kernel for WebGPU
// Converts LWE ciphertext from one key to another
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

fn mod_neg(a: U64, Q: U64) -> U64 {
    if (u64_is_zero(a)) { return a; }
    return u64_sub(Q, a);
}

fn mod_mul(a: U64, b: U64, Q: U64) -> U64 {
    let prod = u32_mul_wide(a.lo, b.lo);
    if (Q.hi == 0u && Q.lo != 0u) {
        let result = prod.lo % Q.lo;
        return U64(result, 0u);
    }
    return prod;
}

// ============================================================================
// Key Switch Parameters
// ============================================================================

struct KeySwitchParams {
    N_in: u32,         // Input LWE dimension (large, e.g., 1024)
    N_out: u32,        // Output LWE dimension (small, e.g., 630)
    ks_l: u32,         // Key switch decomposition levels
    ks_base_log: u32,  // Bg = 2^ks_base_log
    batch: u32,        // Batch size
    Q_lo: u32,         // Modulus low bits
    Q_hi: u32,         // Modulus high bits
    _pad: u32,
}

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> lwe_in_a: array<u32>;       // Input mask [batch][N_in][2]
@group(0) @binding(1) var<storage, read> lwe_in_b: array<u32>;       // Input body [batch][2]
@group(0) @binding(2) var<storage, read> ksk: array<u32>;            // Key switch key [N_in][ks_l][N_out+1][2]
@group(0) @binding(3) var<storage, read_write> lwe_out: array<u32>;  // Output [batch][N_out+1][2]
@group(0) @binding(4) var<uniform> params: KeySwitchParams;

var<workgroup> decomp_cache: array<i32, 4096>;  // Cache for decomposed digits

// ============================================================================
// Key Switch Decomposition
// ============================================================================

fn ks_signed_decomp_digit(val_lo: u32, val_hi: u32, level: u32, base_log: u32) -> i32 {
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

// ============================================================================
// LWE Key Switching - Main Kernel
// ============================================================================

@compute @workgroup_size(256)
fn keyswitch_lwe(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let output_idx = gid.x;  // Output coefficient [0, N_out]
    let batch_idx = wgid.y;
    
    let N_in = params.N_in;
    let N_out = params.N_out;
    let ks_l = params.ks_l;
    let ks_base_log = params.ks_base_log;
    let Q = U64(params.Q_lo, params.Q_hi);
    
    if (output_idx > N_out || batch_idx >= params.batch) { return; }
    
    let is_body = output_idx == N_out;
    
    // Initialize accumulator
    var acc: U64;
    if (is_body) {
        // Body starts with input body
        let b_idx = batch_idx * 2u;
        acc = U64(lwe_in_b[b_idx], lwe_in_b[b_idx + 1u]);
    } else {
        acc = u64_zero();
    }
    
    // KSK stride: each input coefficient has ks_l LWE ciphertexts of dimension N_out+1
    let ksk_stride = ks_l * (N_out + 1u) * 2u;
    
    // Process each input coefficient
    for (var i = 0u; i < N_in; i++) {
        // Load input coefficient
        let in_idx = batch_idx * N_in * 2u + i * 2u;
        let a_lo = lwe_in_a[in_idx];
        let a_hi = lwe_in_a[in_idx + 1u];
        
        // Process each decomposition level
        for (var level = 0u; level < ks_l; level++) {
            let digit = ks_signed_decomp_digit(a_lo, a_hi, level, ks_base_log);
            
            if (digit == 0) { continue; }
            
            // KSK coefficient: ksk[i][level][output_idx]
            let ksk_offset = i * ksk_stride + level * (N_out + 1u) * 2u + output_idx * 2u;
            let ksk_lo = ksk[ksk_offset];
            let ksk_hi = ksk[ksk_offset + 1u];
            let ksk_val = U64(ksk_lo, ksk_hi);
            
            // Multiply and accumulate
            let abs_digit = u32(select(-digit, digit, digit >= 0));
            let prod = mod_mul(u64_from_u32(abs_digit), ksk_val, Q);
            
            if (is_body) {
                // Body: subtract contribution
                if (digit > 0) {
                    acc = mod_sub(acc, prod, Q);
                } else {
                    acc = mod_add(acc, prod, Q);
                }
            } else {
                // Mask: accumulate (will negate at end)
                if (digit > 0) {
                    acc = mod_add(acc, prod, Q);
                } else {
                    acc = mod_sub(acc, prod, Q);
                }
            }
        }
    }
    
    // Negate mask coefficients
    if (!is_body) {
        acc = mod_neg(acc, Q);
    }
    
    // Store result
    let out_idx = batch_idx * (N_out + 1u) * 2u + output_idx * 2u;
    lwe_out[out_idx] = acc.lo;
    lwe_out[out_idx + 1u] = acc.hi;
}

// ============================================================================
// Fused Key Switching with Caching
// ============================================================================

@compute @workgroup_size(256)
fn keyswitch_lwe_cached(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let output_idx = gid.x;
    let batch_idx = wgid.y;
    
    let N_in = params.N_in;
    let N_out = params.N_out;
    let ks_l = params.ks_l;
    let ks_base_log = params.ks_base_log;
    let Q = U64(params.Q_lo, params.Q_hi);
    
    if (output_idx > N_out || batch_idx >= params.batch) { return; }
    
    let thread_idx = lid.x;
    let threads = 256u;
    
    // Cache decomposition for all input coefficients
    for (var i = thread_idx; i < N_in; i += threads) {
        let in_idx = batch_idx * N_in * 2u + i * 2u;
        let a_lo = lwe_in_a[in_idx];
        let a_hi = lwe_in_a[in_idx + 1u];
        
        for (var level = 0u; level < ks_l && level < 8u; level++) {
            decomp_cache[i * ks_l + level] = ks_signed_decomp_digit(a_lo, a_hi, level, ks_base_log);
        }
    }
    workgroupBarrier();
    
    let is_body = output_idx == N_out;
    
    var acc: U64;
    if (is_body) {
        let b_idx = batch_idx * 2u;
        acc = U64(lwe_in_b[b_idx], lwe_in_b[b_idx + 1u]);
    } else {
        acc = u64_zero();
    }
    
    let ksk_stride = ks_l * (N_out + 1u) * 2u;
    
    for (var i = 0u; i < N_in; i++) {
        for (var level = 0u; level < ks_l; level++) {
            let digit = decomp_cache[i * ks_l + level];
            
            if (digit == 0) { continue; }
            
            let ksk_offset = i * ksk_stride + level * (N_out + 1u) * 2u + output_idx * 2u;
            let ksk_val = U64(ksk[ksk_offset], ksk[ksk_offset + 1u]);
            
            let abs_digit = u32(select(-digit, digit, digit >= 0));
            let prod = mod_mul(u64_from_u32(abs_digit), ksk_val, Q);
            
            if (is_body) {
                if (digit > 0) {
                    acc = mod_sub(acc, prod, Q);
                } else {
                    acc = mod_add(acc, prod, Q);
                }
            } else {
                if (digit > 0) {
                    acc = mod_add(acc, prod, Q);
                } else {
                    acc = mod_sub(acc, prod, Q);
                }
            }
        }
    }
    
    if (!is_body) {
        acc = mod_neg(acc, Q);
    }
    
    let out_idx = batch_idx * (N_out + 1u) * 2u + output_idx * 2u;
    lwe_out[out_idx] = acc.lo;
    lwe_out[out_idx + 1u] = acc.hi;
}

// ============================================================================
// Tiled Key Switching (Memory-Efficient for Large N_in)
// ============================================================================

@group(0) @binding(5) var<uniform> tile_info: vec2<u32>;  // [tile_size, tile_idx]

@compute @workgroup_size(256)
fn keyswitch_tiled(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let output_idx = gid.x;
    let batch_idx = wgid.y;
    
    let N_in = params.N_in;
    let N_out = params.N_out;
    let ks_l = params.ks_l;
    let ks_base_log = params.ks_base_log;
    let Q = U64(params.Q_lo, params.Q_hi);
    
    let tile_size = tile_info.x;
    let tile_idx = tile_info.y;
    
    if (output_idx > N_out || batch_idx >= params.batch) { return; }
    
    let tile_start = tile_idx * tile_size;
    let tile_end = min(tile_start + tile_size, N_in);
    let is_body = output_idx == N_out;
    let is_first_tile = tile_idx == 0u;
    let is_last_tile = tile_end == N_in;
    
    // Load or initialize accumulator
    var acc: U64;
    if (is_first_tile) {
        if (is_body) {
            let b_idx = batch_idx * 2u;
            acc = U64(lwe_in_b[b_idx], lwe_in_b[b_idx + 1u]);
        } else {
            acc = u64_zero();
        }
    } else {
        // Load intermediate result
        let out_idx = batch_idx * (N_out + 1u) * 2u + output_idx * 2u;
        acc = U64(lwe_out[out_idx], lwe_out[out_idx + 1u]);
    }
    
    // Cache input coefficients for this tile
    let thread_idx = lid.x;
    let threads = 256u;
    
    for (var i = thread_idx; i < tile_end - tile_start; i += threads) {
        let in_idx = batch_idx * N_in * 2u + (tile_start + i) * 2u;
        decomp_cache[i * 2u] = i32(lwe_in_a[in_idx]);
        decomp_cache[i * 2u + 1u] = i32(lwe_in_a[in_idx + 1u]);
    }
    workgroupBarrier();
    
    let ksk_stride = ks_l * (N_out + 1u) * 2u;
    
    // Process tile
    for (var local_i = 0u; local_i < tile_end - tile_start; local_i++) {
        let a_lo = u32(decomp_cache[local_i * 2u]);
        let a_hi = u32(decomp_cache[local_i * 2u + 1u]);
        let global_i = tile_start + local_i;
        
        for (var level = 0u; level < ks_l; level++) {
            let digit = ks_signed_decomp_digit(a_lo, a_hi, level, ks_base_log);
            
            if (digit == 0) { continue; }
            
            let ksk_offset = global_i * ksk_stride + level * (N_out + 1u) * 2u + output_idx * 2u;
            let ksk_val = U64(ksk[ksk_offset], ksk[ksk_offset + 1u]);
            
            let abs_digit = u32(select(-digit, digit, digit >= 0));
            let prod = mod_mul(u64_from_u32(abs_digit), ksk_val, Q);
            
            if (is_body) {
                if (digit > 0) {
                    acc = mod_sub(acc, prod, Q);
                } else {
                    acc = mod_add(acc, prod, Q);
                }
            } else {
                if (digit > 0) {
                    acc = mod_add(acc, prod, Q);
                } else {
                    acc = mod_sub(acc, prod, Q);
                }
            }
        }
    }
    
    // Finalize on last tile
    if (is_last_tile && !is_body) {
        acc = mod_neg(acc, Q);
    }
    
    let out_idx = batch_idx * (N_out + 1u) * 2u + output_idx * 2u;
    lwe_out[out_idx] = acc.lo;
    lwe_out[out_idx + 1u] = acc.hi;
}

// ============================================================================
// Key Switching with Modulus Switching
// ============================================================================

@group(0) @binding(6) var<storage, read_write> lwe_out_32: array<u32>;  // 32-bit output
@group(0) @binding(7) var<uniform> Q_out: u32;  // Output modulus (32-bit)

@compute @workgroup_size(256)
fn keyswitch_modswitch(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let output_idx = gid.x;
    let batch_idx = wgid.y;
    
    let N_in = params.N_in;
    let N_out = params.N_out;
    let ks_l = params.ks_l;
    let ks_base_log = params.ks_base_log;
    let Q = U64(params.Q_lo, params.Q_hi);
    
    if (output_idx > N_out || batch_idx >= params.batch) { return; }
    
    let is_body = output_idx == N_out;
    
    var acc: U64;
    if (is_body) {
        let b_idx = batch_idx * 2u;
        acc = U64(lwe_in_b[b_idx], lwe_in_b[b_idx + 1u]);
    } else {
        acc = u64_zero();
    }
    
    let ksk_stride = ks_l * (N_out + 1u) * 2u;
    
    for (var i = 0u; i < N_in; i++) {
        let in_idx = batch_idx * N_in * 2u + i * 2u;
        let a_lo = lwe_in_a[in_idx];
        let a_hi = lwe_in_a[in_idx + 1u];
        
        for (var level = 0u; level < ks_l; level++) {
            let digit = ks_signed_decomp_digit(a_lo, a_hi, level, ks_base_log);
            
            if (digit == 0) { continue; }
            
            let ksk_offset = i * ksk_stride + level * (N_out + 1u) * 2u + output_idx * 2u;
            let ksk_val = U64(ksk[ksk_offset], ksk[ksk_offset + 1u]);
            
            let abs_digit = u32(select(-digit, digit, digit >= 0));
            let prod = mod_mul(u64_from_u32(abs_digit), ksk_val, Q);
            
            if (is_body) {
                if (digit > 0) {
                    acc = mod_sub(acc, prod, Q);
                } else {
                    acc = mod_add(acc, prod, Q);
                }
            } else {
                if (digit > 0) {
                    acc = mod_add(acc, prod, Q);
                } else {
                    acc = mod_sub(acc, prod, Q);
                }
            }
        }
    }
    
    if (!is_body) {
        acc = mod_neg(acc, Q);
    }
    
    // Modulus switch: scale from Q to Q_out
    // out = round(acc * Q_out / Q)
    // Simplified: for 32-bit Q_out, scale the 64-bit result
    let scaled = u32_mul_wide(acc.lo, Q_out);
    // Add Q.lo/2 for rounding
    let rounded_lo = scaled.lo + (params.Q_lo >> 1u);
    let rounded_hi = scaled.hi + select(0u, 1u, rounded_lo < scaled.lo);
    
    // Divide by Q (simplified for 32-bit moduli)
    var result: u32;
    if (params.Q_hi == 0u) {
        result = rounded_lo / params.Q_lo;
    } else {
        result = rounded_lo;  // Approximation
    }
    
    let out_idx = batch_idx * (N_out + 1u) + output_idx;
    lwe_out_32[out_idx] = result;
}
