// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// TFHE Functional Key Switch (WebGPU/WGSL)
// Evaluates a linear function f(x) = sum(c_i * x_i) homomorphically
// during the key switching operation
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

struct FunctionalKeySwitchParams {
    n_in: u32,        // Input LWE dimension
    n_out: u32,       // Output LWE dimension
    l: u32,           // Decomposition levels
    base_log: u32,    // Bg = 2^base_log
    batch_size: u32,  // Number of functions to evaluate
    Q_lo: u32,        // Modulus low
    Q_hi: u32,        // Modulus high
    pad: u32,         // Padding for alignment
}

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> lwe_in: array<u32>;              // [n_in+1][2]
@group(0) @binding(1) var<storage, read> function_coeffs: array<i32>;     // [n_in]
@group(0) @binding(2) var<storage, read> fksk: array<u32>;                // [n_in][l][n_out+1][2]
@group(0) @binding(3) var<storage, read_write> lwe_out: array<u32>;       // [n_out+1][2]
@group(0) @binding(4) var<uniform> params: FunctionalKeySwitchParams;

var<workgroup> shared_decomp: array<i32, 4096>;  // [n_in][l]

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
// Functional Key Switch Kernel
// ============================================================================

@compute @workgroup_size(256)
fn functional_keyswitch(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let out_idx = gid.x;
    let thread_idx = lid.x;
    let threads = 256u;

    let n_in = params.n_in;
    let n_out = params.n_out;
    let l = params.l;
    let Q = U64(params.Q_lo, params.Q_hi);

    if (out_idx > n_out) { return; }

    // Step 1: Collaboratively decompose scaled inputs
    for (var i = thread_idx; i < n_in; i += threads) {
        let f_i = function_coeffs[i];

        // Load LWE coefficient
        let a_lo = lwe_in[i * 2u];
        let a_hi = lwe_in[i * 2u + 1u];
        let a_val = U64(a_lo, a_hi);

        // Scale by function coefficient
        var scaled: U64;
        if (f_i >= 0) {
            scaled = mod_mul(a_val, u64_from_u32(u32(f_i)), Q);
        } else {
            scaled = mod_neg(mod_mul(a_val, u64_from_u32(u32(-f_i)), Q), Q);
        }

        // Decompose and store
        for (var level = 0u; level < l; level++) {
            shared_decomp[i * l + level] = signed_decompose_digit(scaled.lo, scaled.hi, level, params.base_log);
        }
    }

    workgroupBarrier();

    // Step 2: Accumulate using FKSK
    var acc = u64_zero();

    // Body passthrough
    if (out_idx == n_out) {
        let body_lo = lwe_in[n_in * 2u];
        let body_hi = lwe_in[n_in * 2u + 1u];
        acc = U64(body_lo, body_hi);
    }

    // Accumulate over all inputs
    for (var in_idx = 0u; in_idx < n_in; in_idx++) {
        for (var level = 0u; level < l; level++) {
            let digit = shared_decomp[in_idx * l + level];

            if (digit != 0) {
                // FKSK layout: [n_in][l][n_out+1][2]
                let fksk_offset = ((in_idx * l + level) * (n_out + 1u) + out_idx) * 2u;
                let fksk_val = U64(fksk[fksk_offset], fksk[fksk_offset + 1u]);

                let abs_digit = u32(select(-digit, digit, digit >= 0));
                let prod = mod_mul(u64_from_u32(abs_digit), fksk_val, Q);

                if (digit > 0) {
                    acc = mod_add(acc, prod, Q);
                } else {
                    acc = mod_sub(acc, prod, Q);
                }
            }
        }
    }

    lwe_out[out_idx * 2u] = acc.lo;
    lwe_out[out_idx * 2u + 1u] = acc.hi;
}

// ============================================================================
// Functional Key Switch - Batched
// ============================================================================

@group(0) @binding(5) var<storage, read> function_batch: array<i32>;       // [batch_size][n_in]
@group(0) @binding(6) var<storage, read_write> lwe_out_batch: array<u32>;  // [batch_size][n_out+1][2]

@compute @workgroup_size(256)
fn functional_keyswitch_batch(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let out_idx = gid.x;
    let batch_idx = wgid.y;
    let thread_idx = lid.x;
    let threads = 256u;

    let n_in = params.n_in;
    let n_out = params.n_out;
    let l = params.l;
    let batch_size = params.batch_size;
    let Q = U64(params.Q_lo, params.Q_hi);

    if (out_idx > n_out || batch_idx >= batch_size) { return; }

    let func_base = batch_idx * n_in;

    // Decompose scaled inputs for this function
    for (var i = thread_idx; i < n_in; i += threads) {
        let f_i = function_batch[func_base + i];

        let a_lo = lwe_in[i * 2u];
        let a_hi = lwe_in[i * 2u + 1u];
        let a_val = U64(a_lo, a_hi);

        var scaled: U64;
        if (f_i >= 0) {
            scaled = mod_mul(a_val, u64_from_u32(u32(f_i)), Q);
        } else {
            scaled = mod_neg(mod_mul(a_val, u64_from_u32(u32(-f_i)), Q), Q);
        }

        for (var level = 0u; level < l; level++) {
            shared_decomp[i * l + level] = signed_decompose_digit(scaled.lo, scaled.hi, level, params.base_log);
        }
    }

    workgroupBarrier();

    var acc = u64_zero();

    if (out_idx == n_out) {
        let body_lo = lwe_in[n_in * 2u];
        let body_hi = lwe_in[n_in * 2u + 1u];
        acc = U64(body_lo, body_hi);
    }

    for (var in_idx = 0u; in_idx < n_in; in_idx++) {
        for (var level = 0u; level < l; level++) {
            let digit = shared_decomp[in_idx * l + level];

            if (digit != 0) {
                let fksk_offset = ((in_idx * l + level) * (n_out + 1u) + out_idx) * 2u;
                let fksk_val = U64(fksk[fksk_offset], fksk[fksk_offset + 1u]);

                let abs_digit = u32(select(-digit, digit, digit >= 0));
                let prod = mod_mul(u64_from_u32(abs_digit), fksk_val, Q);

                if (digit > 0) {
                    acc = mod_add(acc, prod, Q);
                } else {
                    acc = mod_sub(acc, prod, Q);
                }
            }
        }
    }

    let out_base = batch_idx * (n_out + 1u) * 2u;
    lwe_out_batch[out_base + out_idx * 2u] = acc.lo;
    lwe_out_batch[out_base + out_idx * 2u + 1u] = acc.hi;
}

// ============================================================================
// Functional Key Switch - Multi-Input
// ============================================================================

@group(0) @binding(7) var<storage, read> lwe_inputs: array<u32>;          // [num_inputs][n_in+1][2]
@group(0) @binding(8) var<storage, read> coefficients: array<i32>;        // [num_inputs][n_in]
@group(0) @binding(9) var<uniform> num_inputs: u32;

@compute @workgroup_size(256)
fn functional_keyswitch_multi(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let out_idx = gid.x;
    let thread_idx = lid.x;
    let threads = 256u;

    let n_in = params.n_in;
    let n_out = params.n_out;
    let l = params.l;
    let Q = U64(params.Q_lo, params.Q_hi);

    if (out_idx > n_out) { return; }

    var acc = u64_zero();

    // Process each input LWE
    for (var input_idx = 0u; input_idx < num_inputs; input_idx++) {
        let lwe_base = input_idx * (n_in + 1u) * 2u;
        let coeff_base = input_idx * n_in;

        // Decompose scaled inputs
        for (var i = thread_idx; i < n_in; i += threads) {
            let c_i = coefficients[coeff_base + i];

            let a_lo = lwe_inputs[lwe_base + i * 2u];
            let a_hi = lwe_inputs[lwe_base + i * 2u + 1u];
            let a_val = U64(a_lo, a_hi);

            var scaled: U64;
            if (c_i >= 0) {
                scaled = mod_mul(a_val, u64_from_u32(u32(c_i)), Q);
            } else {
                scaled = mod_neg(mod_mul(a_val, u64_from_u32(u32(-c_i)), Q), Q);
            }

            for (var level = 0u; level < l; level++) {
                shared_decomp[i * l + level] = signed_decompose_digit(scaled.lo, scaled.hi, level, params.base_log);
            }
        }

        workgroupBarrier();

        // Accumulate body contribution
        if (out_idx == n_out) {
            var body_coeff: i32 = 0;
            for (var i = 0u; i < n_in; i++) {
                body_coeff += coefficients[coeff_base + i];
            }

            let body_lo = lwe_inputs[lwe_base + n_in * 2u];
            let body_hi = lwe_inputs[lwe_base + n_in * 2u + 1u];
            let body = U64(body_lo, body_hi);

            if (body_coeff >= 0) {
                let scaled_body = mod_mul(body, u64_from_u32(u32(body_coeff)), Q);
                acc = mod_add(acc, scaled_body, Q);
            } else {
                let scaled_body = mod_mul(body, u64_from_u32(u32(-body_coeff)), Q);
                acc = mod_sub(acc, scaled_body, Q);
            }
        }

        // Accumulate using FKSK
        for (var in_idx = 0u; in_idx < n_in; in_idx++) {
            for (var level = 0u; level < l; level++) {
                let digit = shared_decomp[in_idx * l + level];

                if (digit != 0) {
                    let fksk_offset = ((in_idx * l + level) * (n_out + 1u) + out_idx) * 2u;
                    let fksk_val = U64(fksk[fksk_offset], fksk[fksk_offset + 1u]);

                    let abs_digit = u32(select(-digit, digit, digit >= 0));
                    let prod = mod_mul(u64_from_u32(abs_digit), fksk_val, Q);

                    if (digit > 0) {
                        acc = mod_add(acc, prod, Q);
                    } else {
                        acc = mod_sub(acc, prod, Q);
                    }
                }
            }
        }

        workgroupBarrier();
    }

    lwe_out[out_idx * 2u] = acc.lo;
    lwe_out[out_idx * 2u + 1u] = acc.hi;
}
