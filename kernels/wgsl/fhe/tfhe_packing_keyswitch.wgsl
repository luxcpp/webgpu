// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// TFHE Packing Key Switch: Multiple LWE -> Single RLWE (WebGPU/WGSL)
// Converts a batch of LWE ciphertexts into a single RLWE ciphertext
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
    // Simplified for 32-bit effective moduli
    let prod = u32_mul_wide(a.lo, b.lo);
    if (Q.hi == 0u && Q.lo != 0u) {
        let result = prod.lo % Q.lo;
        return U64(result, 0u);
    }
    return prod;
}

// ============================================================================
// Parameters
// ============================================================================

struct PackingKeySwitchParams {
    N: u32,           // RLWE polynomial degree
    n_lwe: u32,       // LWE dimension
    l: u32,           // Decomposition levels
    base_log: u32,    // Bg = 2^base_log
    batch_size: u32,  // Number of LWE to pack
    Q_lo: u32,        // Modulus low
    Q_hi: u32,        // Modulus high
    pad: u32,         // Padding for alignment
}

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> lwe_batch: array<u32>;           // [batch_size][n_lwe+1][2]
@group(0) @binding(1) var<storage, read> pksk: array<u32>;                // [n_lwe][l][2][N][2]
@group(0) @binding(2) var<storage, read_write> rlwe_out: array<u32>;      // [2][N][2]
@group(0) @binding(3) var<uniform> params: PackingKeySwitchParams;

var<workgroup> acc_cache: array<u32, 8192>;  // [2][N][2] workspace

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

    // Add half_Bg for rounding, then center
    digit = (digit + half_Bg) & mask;

    return i32(digit) - i32(half_Bg);
}

// ============================================================================
// Negacyclic Rotation Helper
// ============================================================================

fn get_rotated_index(coeff_idx: u32, rotation: u32, N: u32) -> vec2<u32> {
    let rot = rotation % (2u * N);

    var src_idx: u32;
    var negate: u32;  // 0 = no negate, 1 = negate

    if (rot < N) {
        if (coeff_idx >= rot) {
            src_idx = coeff_idx - rot;
            negate = 0u;
        } else {
            src_idx = N - rot + coeff_idx;
            negate = 1u;
        }
    } else {
        let r = rot - N;
        if (coeff_idx >= r) {
            src_idx = coeff_idx - r;
            negate = 1u;
        } else {
            src_idx = N - r + coeff_idx;
            negate = 0u;
        }
    }

    return vec2<u32>(src_idx, negate);
}

// ============================================================================
// Packing Key Switch Kernel
// ============================================================================

@compute @workgroup_size(256)
fn packing_keyswitch(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let coeff_idx = gid.x;
    let poly_idx = wgid.y;  // 0 or 1 (mask or body)

    let N = params.N;
    let n_lwe = params.n_lwe;
    let l = params.l;
    let batch_size = params.batch_size;
    let Q = U64(params.Q_lo, params.Q_hi);

    if (coeff_idx >= N || poly_idx > 1u) { return; }

    // Initialize accumulator
    var acc = u64_zero();

    // Process each LWE in the batch
    for (var batch_idx = 0u; batch_idx < batch_size; batch_idx++) {
        // LWE layout: [batch_size][n_lwe+1][2] (u64 as 2xu32)
        let lwe_base = batch_idx * (n_lwe + 1u) * 2u;

        // Rotation for this slot
        let slot_rotation = batch_idx * (N / batch_size);

        // For each LWE coefficient
        for (var in_idx = 0u; in_idx < n_lwe; in_idx++) {
            let a_lo = lwe_batch[lwe_base + in_idx * 2u];
            let a_hi = lwe_batch[lwe_base + in_idx * 2u + 1u];

            // For each decomposition level
            for (var level = 0u; level < l; level++) {
                let digit = signed_decompose_digit(a_lo, a_hi, level, params.base_log);

                if (digit == 0) { continue; }

                // Get rotated PKSK coefficient
                let rot_info = get_rotated_index(coeff_idx, slot_rotation, N);
                let src_idx = rot_info.x;
                let negate = rot_info.y == 1u;

                // PKSK layout: [n_lwe][l][2][N][2]
                let pksk_offset = ((in_idx * l + level) * 2u + poly_idx) * N * 2u + src_idx * 2u;
                var pksk_val = U64(pksk[pksk_offset], pksk[pksk_offset + 1u]);

                // Apply rotation negation
                if (negate) {
                    pksk_val = mod_neg(pksk_val, Q);
                }

                // Multiply by signed digit
                let abs_digit = u32(select(-digit, digit, digit >= 0));
                let prod = mod_mul(u64_from_u32(abs_digit), pksk_val, Q);

                if (digit > 0) {
                    acc = mod_add(acc, prod, Q);
                } else {
                    acc = mod_sub(acc, prod, Q);
                }
            }
        }

        // Add body contribution for body polynomial
        if (poly_idx == 1u && coeff_idx == slot_rotation % N) {
            let body_lo = lwe_batch[lwe_base + n_lwe * 2u];
            let body_hi = lwe_batch[lwe_base + n_lwe * 2u + 1u];
            let body = U64(body_lo, body_hi);
            acc = mod_add(acc, body, Q);
        }
    }

    // Write output
    let out_idx = poly_idx * N * 2u + coeff_idx * 2u;
    rlwe_out[out_idx] = acc.lo;
    rlwe_out[out_idx + 1u] = acc.hi;
}

// ============================================================================
// Batched Packing Key Switch
// ============================================================================

@group(0) @binding(4) var<storage, read> lwe_batches: array<u32>;        // [num_packs][batch_size][n_lwe+1][2]
@group(0) @binding(5) var<storage, read_write> rlwe_batch: array<u32>;   // [num_packs][2][N][2]
@group(0) @binding(6) var<uniform> num_packs: u32;

@compute @workgroup_size(256)
fn packing_keyswitch_batch(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let coeff_idx = gid.x;
    let poly_idx = wgid.y;
    let pack_idx = wgid.z;

    let N = params.N;
    let n_lwe = params.n_lwe;
    let l = params.l;
    let batch_size = params.batch_size;
    let Q = U64(params.Q_lo, params.Q_hi);

    if (coeff_idx >= N || poly_idx > 1u || pack_idx >= num_packs) { return; }

    let lwe_stride = batch_size * (n_lwe + 1u) * 2u;
    let lwe_pack_base = pack_idx * lwe_stride;

    var acc = u64_zero();

    for (var batch_idx = 0u; batch_idx < batch_size; batch_idx++) {
        let lwe_base = lwe_pack_base + batch_idx * (n_lwe + 1u) * 2u;
        let slot_rotation = batch_idx * (N / batch_size);

        for (var in_idx = 0u; in_idx < n_lwe; in_idx++) {
            let a_lo = lwe_batches[lwe_base + in_idx * 2u];
            let a_hi = lwe_batches[lwe_base + in_idx * 2u + 1u];

            for (var level = 0u; level < l; level++) {
                let digit = signed_decompose_digit(a_lo, a_hi, level, params.base_log);

                if (digit == 0) { continue; }

                let rot_info = get_rotated_index(coeff_idx, slot_rotation, N);
                let src_idx = rot_info.x;
                let negate = rot_info.y == 1u;

                let pksk_offset = ((in_idx * l + level) * 2u + poly_idx) * N * 2u + src_idx * 2u;
                var pksk_val = U64(pksk[pksk_offset], pksk[pksk_offset + 1u]);

                if (negate) {
                    pksk_val = mod_neg(pksk_val, Q);
                }

                let abs_digit = u32(select(-digit, digit, digit >= 0));
                let prod = mod_mul(u64_from_u32(abs_digit), pksk_val, Q);

                if (digit > 0) {
                    acc = mod_add(acc, prod, Q);
                } else {
                    acc = mod_sub(acc, prod, Q);
                }
            }
        }

        if (poly_idx == 1u && coeff_idx == slot_rotation % N) {
            let body_lo = lwe_batches[lwe_base + n_lwe * 2u];
            let body_hi = lwe_batches[lwe_base + n_lwe * 2u + 1u];
            let body = U64(body_lo, body_hi);
            acc = mod_add(acc, body, Q);
        }
    }

    let rlwe_stride = 2u * N * 2u;
    let out_idx = pack_idx * rlwe_stride + poly_idx * N * 2u + coeff_idx * 2u;
    rlwe_batch[out_idx] = acc.lo;
    rlwe_batch[out_idx + 1u] = acc.hi;
}
