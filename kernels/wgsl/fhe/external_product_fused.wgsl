// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// TFHE Fused External Product Kernel for WebGPU
// Combines decomposition, NTT, multiply, INTT in single kernel dispatch
// Optimized for workgroup-local memory to minimize global memory traffic
// Compatible with Metal/Vulkan/D3D12 via Dawn/wgpu

// ============================================================================
// 64-bit Integer Emulation
// ============================================================================

struct u64 {
    lo: u32,
    hi: u32,
}

fn u64_zero() -> u64 { return u64(0u, 0u); }
fn u64_from_u32(v: u32) -> u64 { return u64(v, 0u); }

fn u64_eq(a: u64, b: u64) -> bool {
    return a.lo == b.lo && a.hi == b.hi;
}

fn u64_lt(a: u64, b: u64) -> bool {
    if (a.hi != b.hi) { return a.hi < b.hi; }
    return a.lo < b.lo;
}

fn u64_ge(a: u64, b: u64) -> bool { return !u64_lt(a, b); }

fn u64_add(a: u64, b: u64) -> u64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    let hi = a.hi + b.hi + carry;
    return u64(lo, hi);
}

fn u64_sub(a: u64, b: u64) -> u64 {
    let borrow = select(0u, 1u, a.lo < b.lo);
    let lo = a.lo - b.lo;
    let hi = a.hi - b.hi - borrow;
    return u64(lo, hi);
}

fn u32_mul_wide(a: u32, b: u32) -> u64 {
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

    let lo = p0 + ((mid & 0xFFFFu) << 16u);
    let lo_carry = select(0u, 1u, lo < p0);

    let hi = p3 + (mid >> 16u) + mid_carry + lo_carry;

    return u64(lo, hi);
}

fn u64_mul(a: u64, b: u64) -> u64 {
    let p0 = u32_mul_wide(a.lo, b.lo);
    let cross1 = a.lo * b.hi;
    let cross2 = a.hi * b.lo;
    let hi = p0.hi + cross1 + cross2;
    return u64(p0.lo, hi);
}

fn u64_mulhi(a: u64, b: u64) -> u64 {
    let p0 = u32_mul_wide(a.lo, b.lo);
    let p1 = u32_mul_wide(a.lo, b.hi);
    let p2 = u32_mul_wide(a.hi, b.lo);
    let p3 = u32_mul_wide(a.hi, b.hi);

    let mid = u64_add(p1, p2);
    let mid2 = u64_add(mid, u64(p0.hi, 0u));

    var result = p3;
    result = u64_add(result, u64(mid2.hi, 0u));

    if (u64_lt(mid, p1)) { result = u64_add(result, u64(0u, 1u)); }
    if (u64_lt(mid2, mid)) { result = u64_add(result, u64(1u, 0u)); }

    return result;
}

// ============================================================================
// Modular Arithmetic
// ============================================================================

fn mod_add(a: u64, b: u64, Q: u64) -> u64 {
    let sum = u64_add(a, b);
    if (u64_ge(sum, Q)) { return u64_sub(sum, Q); }
    return sum;
}

fn mod_sub(a: u64, b: u64, Q: u64) -> u64 {
    if (u64_ge(a, b)) { return u64_sub(a, b); }
    return u64_sub(u64_add(a, Q), b);
}

fn barrett_mul(a: u64, b: u64, Q: u64, precon: u64) -> u64 {
    let q_approx = u64_mulhi(a, precon);
    let product = u64_mul(a, b);
    let qQ = u64_mul(q_approx, Q);
    var result = u64_sub(product, qQ);
    if (u64_ge(result, Q)) { result = u64_sub(result, Q); }
    return result;
}

// ============================================================================
// NTT Butterfly Operations
// ============================================================================

fn ct_butterfly(lo: ptr<function, u64>, hi: ptr<function, u64>, omega: u64, Q: u64, precon: u64) {
    let u = *lo;
    let v = barrett_mul(*hi, omega, Q, precon);
    *lo = mod_add(u, v, Q);
    *hi = mod_sub(u, v, Q);
}

fn gs_butterfly(lo: ptr<function, u64>, hi: ptr<function, u64>, omega: u64, Q: u64, precon: u64) {
    let u = *lo;
    let v = *hi;
    *lo = mod_add(u, v, Q);
    *hi = barrett_mul(mod_sub(u, v, Q), omega, Q, precon);
}

// ============================================================================
// Parameters
// ============================================================================

struct FusedExtProdParams {
    // Polynomial parameters
    N: u32,              // Polynomial degree (256, 512 for fused)
    log_N: u32,          // log2(N)
    k: u32,              // GLWE dimension (1)
    l: u32,              // Decomposition levels (3-4)
    base_log: u32,       // Base log (8)

    // Modulus
    Q_lo: u32,
    Q_hi: u32,

    // Barrett constant
    mu_lo: u32,
    mu_hi: u32,

    // Inverse N for INTT
    N_inv_lo: u32,
    N_inv_hi: u32,
    N_inv_precon_lo: u32,
    N_inv_precon_hi: u32,

    // Batch
    batch_idx: u32,
    _pad: u32,
}

fn params_Q(p: FusedExtProdParams) -> u64 { return u64(p.Q_lo, p.Q_hi); }
fn params_mu(p: FusedExtProdParams) -> u64 { return u64(p.mu_lo, p.mu_hi); }
fn params_N_inv(p: FusedExtProdParams) -> u64 { return u64(p.N_inv_lo, p.N_inv_hi); }
fn params_N_inv_precon(p: FusedExtProdParams) -> u64 { return u64(p.N_inv_precon_lo, p.N_inv_precon_hi); }

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> glwe_in: array<u32>;        // Input GLWE (coefficient domain)
@group(0) @binding(1) var<storage, read> ggsw_ntt: array<u32>;       // GGSW in NTT domain
@group(0) @binding(2) var<storage, read_write> glwe_out: array<u32>; // Output GLWE (coefficient domain)
@group(0) @binding(3) var<storage, read> twiddles: array<u32>;       // Forward NTT twiddles
@group(0) @binding(4) var<storage, read> inv_twiddles: array<u32>;   // Inverse NTT twiddles
@group(0) @binding(5) var<storage, read> precons: array<u32>;        // Barrett precomputed values
@group(0) @binding(6) var<storage, read> inv_precons: array<u32>;    // Inverse twiddle precomputed
@group(0) @binding(7) var<uniform> params: FusedExtProdParams;

// Shared memory for fused operation (max N=512 for 16KB limit)
// Layout: [2 polys (in/out)][N coefficients][2 u32 per u64] = 2 * 512 * 2 = 2048 u32 = 8KB
var<workgroup> poly_a: array<u32, 2048>;  // First polynomial workspace
var<workgroup> poly_b: array<u32, 2048>;  // Second polynomial workspace
var<workgroup> accum: array<u32, 2048>;   // Accumulator

// ============================================================================
// Shared Memory Helpers
// ============================================================================

fn read_poly_a(idx: u32) -> u64 {
    return u64(poly_a[idx * 2u], poly_a[idx * 2u + 1u]);
}

fn write_poly_a(idx: u32, val: u64) {
    poly_a[idx * 2u] = val.lo;
    poly_a[idx * 2u + 1u] = val.hi;
}

fn read_poly_b(idx: u32) -> u64 {
    return u64(poly_b[idx * 2u], poly_b[idx * 2u + 1u]);
}

fn write_poly_b(idx: u32, val: u64) {
    poly_b[idx * 2u] = val.lo;
    poly_b[idx * 2u + 1u] = val.hi;
}

fn read_accum(idx: u32) -> u64 {
    return u64(accum[idx * 2u], accum[idx * 2u + 1u]);
}

fn write_accum(idx: u32, val: u64) {
    accum[idx * 2u] = val.lo;
    accum[idx * 2u + 1u] = val.hi;
}

fn read_u64_storage(arr: ptr<storage, array<u32>, read>, idx: u32) -> u64 {
    return u64((*arr)[idx * 2u], (*arr)[idx * 2u + 1u]);
}

fn write_u64_storage(arr: ptr<storage, array<u32>, read_write>, idx: u32, val: u64) {
    (*arr)[idx * 2u] = val.lo;
    (*arr)[idx * 2u + 1u] = val.hi;
}

// ============================================================================
// Signed Decomposition
// ============================================================================

fn signed_decompose_digit(val: u64, level: u32, base_log: u32) -> i32 {
    let Bg = 1u << base_log;
    let half_Bg = Bg >> 1u;
    let mask = Bg - 1u;

    let shift = 64u - (level + 1u) * base_log;

    var digit: u32;
    if (shift >= 32u) {
        digit = (val.hi >> (shift - 32u)) & mask;
    } else if (shift == 0u) {
        digit = val.lo & mask;
    } else {
        let lo_part = val.lo >> shift;
        let hi_part = val.hi << (32u - shift);
        digit = (lo_part | hi_part) & mask;
    }

    if (digit >= half_Bg) {
        return i32(digit) - i32(Bg);
    }
    return i32(digit);
}

// ============================================================================
// In-Place NTT Forward (in shared memory)
// ============================================================================

fn ntt_forward_shared(N: u32, log_N: u32, Q: u64, tid: u32, tpg: u32) {
    for (var stage = 0u; stage < log_N; stage++) {
        let m = 1u << stage;
        let t = N >> (stage + 1u);
        let butterflies_per_thread = (N / 2u + tpg - 1u) / tpg;

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_idx = tid + b * tpg;
            if (butterfly_idx >= N / 2u) { break; }

            let group = butterfly_idx / t;
            let j = butterfly_idx % t;
            let idx_lo = group * (2u * t) + j;
            let idx_hi = idx_lo + t;

            let tw_idx = m + group;
            let omega = read_u64_storage(&twiddles, tw_idx);
            let precon = read_u64_storage(&precons, tw_idx);

            var lo = read_poly_a(idx_lo);
            var hi = read_poly_a(idx_hi);

            ct_butterfly(&lo, &hi, omega, Q, precon);

            write_poly_a(idx_lo, lo);
            write_poly_a(idx_hi, hi);
        }

        workgroupBarrier();
    }
}

// ============================================================================
// In-Place NTT Inverse (in shared memory)
// ============================================================================

fn ntt_inverse_shared(N: u32, log_N: u32, Q: u64, N_inv: u64, N_inv_precon: u64, tid: u32, tpg: u32) {
    for (var stage = 0u; stage < log_N; stage++) {
        let m = N >> (stage + 1u);
        let t = 1u << stage;
        let butterflies_per_thread = (N / 2u + tpg - 1u) / tpg;

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_idx = tid + b * tpg;
            if (butterfly_idx >= N / 2u) { break; }

            let group = butterfly_idx / t;
            let j = butterfly_idx % t;
            let idx_lo = group * (2u * t) + j;
            let idx_hi = idx_lo + t;

            let tw_idx = m + group;
            let omega = read_u64_storage(&inv_twiddles, tw_idx);
            let precon = read_u64_storage(&inv_precons, tw_idx);

            var lo = read_poly_a(idx_lo);
            var hi = read_poly_a(idx_hi);

            gs_butterfly(&lo, &hi, omega, Q, precon);

            write_poly_a(idx_lo, lo);
            write_poly_a(idx_hi, hi);
        }

        workgroupBarrier();
    }

    // Scale by N^{-1}
    let elements_per_thread = (N + tpg - 1u) / tpg;
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid + i * tpg;
        if (idx < N) {
            var val = read_poly_a(idx);
            val = barrett_mul(val, N_inv, Q, N_inv_precon);
            write_poly_a(idx, val);
        }
    }
    workgroupBarrier();
}

// ============================================================================
// Fused External Product Kernel
// ============================================================================

@compute @workgroup_size(256)
fn external_product_fused(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let log_N = params.log_N;
    let k = params.k;
    let l = params.l;
    let Q = params_Q(params);
    let mu = params_mu(params);
    let N_inv = params_N_inv(params);
    let N_inv_precon = params_N_inv_precon(params);
    let tpg = 256u;

    let out_poly = wgid.x;  // Which output polynomial (0 to k)

    if (out_poly > k) { return; }

    // Initialize accumulator to zero
    let elems_per_thread = (N + tpg - 1u) / tpg;
    for (var i = 0u; i < elems_per_thread; i++) {
        let idx = tid + i * tpg;
        if (idx < N) {
            write_accum(idx, u64_zero());
        }
    }
    workgroupBarrier();

    // Process each input polynomial and decomposition level
    for (var in_poly = 0u; in_poly <= k; in_poly++) {
        // Load GLWE input polynomial to shared memory
        for (var i = 0u; i < elems_per_thread; i++) {
            let idx = tid + i * tpg;
            if (idx < N) {
                let glwe_idx = in_poly * N + idx;
                let val = read_u64_storage(&glwe_in, glwe_idx);
                write_poly_a(idx, val);
            }
        }
        workgroupBarrier();

        // Forward NTT on input polynomial
        ntt_forward_shared(N, log_N, Q, tid, tpg);

        // For each decomposition level
        for (var level = 0u; level < l; level++) {
            // Load GGSW coefficient (already in NTT domain)
            // GGSW layout: [in_poly][level][out_poly][N]
            let ggsw_base = ((in_poly * l + level) * (k + 1u) + out_poly) * N;

            // Pointwise multiply: decomposed_digit * GGSW_coeff
            for (var i = 0u; i < elems_per_thread; i++) {
                let idx = tid + i * tpg;
                if (idx < N) {
                    let input_ntt = read_poly_a(idx);

                    // Get digit for this level
                    let digit = signed_decompose_digit(input_ntt, level, params.base_log);

                    if (digit != 0) {
                        let ggsw_val = read_u64_storage(&ggsw_ntt, ggsw_base + idx);
                        let abs_digit = u32(select(-digit, digit, digit >= 0));
                        let prod = barrett_mul(u64_from_u32(abs_digit), ggsw_val, Q, mu);

                        var acc_val = read_accum(idx);
                        if (digit > 0) {
                            acc_val = mod_add(acc_val, prod, Q);
                        } else {
                            acc_val = mod_sub(acc_val, prod, Q);
                        }
                        write_accum(idx, acc_val);
                    }
                }
            }
            workgroupBarrier();
        }
    }

    // Copy accumulator to poly_a for INTT
    for (var i = 0u; i < elems_per_thread; i++) {
        let idx = tid + i * tpg;
        if (idx < N) {
            write_poly_a(idx, read_accum(idx));
        }
    }
    workgroupBarrier();

    // Inverse NTT
    ntt_inverse_shared(N, log_N, Q, N_inv, N_inv_precon, tid, tpg);

    // Write result to global memory
    for (var i = 0u; i < elems_per_thread; i++) {
        let idx = tid + i * tpg;
        if (idx < N) {
            let out_idx = out_poly * N + idx;
            write_u64_storage(&glwe_out, out_idx, read_poly_a(idx));
        }
    }
}

// ============================================================================
// Fused CMux Operation
// ============================================================================

@group(1) @binding(0) var<storage, read> glwe_d0: array<u32>;     // GLWE_0
@group(1) @binding(1) var<storage, read> glwe_d1: array<u32>;     // GLWE_1
@group(1) @binding(2) var<storage, read_write> cmux_out: array<u32>;

@compute @workgroup_size(256)
fn cmux_fused(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let log_N = params.log_N;
    let k = params.k;
    let l = params.l;
    let Q = params_Q(params);
    let mu = params_mu(params);
    let N_inv = params_N_inv(params);
    let N_inv_precon = params_N_inv_precon(params);
    let tpg = 256u;

    let out_poly = wgid.x;

    if (out_poly > k) { return; }

    let elems_per_thread = (N + tpg - 1u) / tpg;

    // Initialize accumulator
    for (var i = 0u; i < elems_per_thread; i++) {
        let idx = tid + i * tpg;
        if (idx < N) {
            write_accum(idx, u64_zero());
        }
    }
    workgroupBarrier();

    // Process each input polynomial
    for (var in_poly = 0u; in_poly <= k; in_poly++) {
        // Load diff = d1 - d0 to shared memory
        for (var i = 0u; i < elems_per_thread; i++) {
            let idx = tid + i * tpg;
            if (idx < N) {
                let glwe_idx = in_poly * N + idx;
                let d0 = read_u64_storage(&glwe_d0, glwe_idx);
                let d1 = read_u64_storage(&glwe_d1, glwe_idx);
                let diff = mod_sub(d1, d0, Q);
                write_poly_a(idx, diff);
            }
        }
        workgroupBarrier();

        // Forward NTT on diff
        ntt_forward_shared(N, log_N, Q, tid, tpg);

        // External product accumulation
        for (var level = 0u; level < l; level++) {
            let ggsw_base = ((in_poly * l + level) * (k + 1u) + out_poly) * N;

            for (var i = 0u; i < elems_per_thread; i++) {
                let idx = tid + i * tpg;
                if (idx < N) {
                    let diff_ntt = read_poly_a(idx);
                    let digit = signed_decompose_digit(diff_ntt, level, params.base_log);

                    if (digit != 0) {
                        let ggsw_val = read_u64_storage(&ggsw_ntt, ggsw_base + idx);
                        let abs_digit = u32(select(-digit, digit, digit >= 0));
                        let prod = barrett_mul(u64_from_u32(abs_digit), ggsw_val, Q, mu);

                        var acc_val = read_accum(idx);
                        if (digit > 0) {
                            acc_val = mod_add(acc_val, prod, Q);
                        } else {
                            acc_val = mod_sub(acc_val, prod, Q);
                        }
                        write_accum(idx, acc_val);
                    }
                }
            }
            workgroupBarrier();
        }
    }

    // INTT on accumulator
    for (var i = 0u; i < elems_per_thread; i++) {
        let idx = tid + i * tpg;
        if (idx < N) {
            write_poly_a(idx, read_accum(idx));
        }
    }
    workgroupBarrier();

    ntt_inverse_shared(N, log_N, Q, N_inv, N_inv_precon, tid, tpg);

    // Result = d0 + external_product
    for (var i = 0u; i < elems_per_thread; i++) {
        let idx = tid + i * tpg;
        if (idx < N) {
            let out_idx = out_poly * N + idx;
            let ext_prod = read_poly_a(idx);
            let d0 = read_u64_storage(&glwe_d0, out_idx);
            let result = mod_add(d0, ext_prod, Q);
            write_u64_storage(&cmux_out, out_idx, result);
        }
    }
}

// ============================================================================
// Fused External Product with Rotation (for Blind Rotate)
// ============================================================================

@group(2) @binding(0) var<uniform> rotation_amount: i32;

@compute @workgroup_size(256)
fn external_product_rotate_fused(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let log_N = params.log_N;
    let k = params.k;
    let l = params.l;
    let Q = params_Q(params);
    let mu = params_mu(params);
    let N_inv = params_N_inv(params);
    let N_inv_precon = params_N_inv_precon(params);
    let tpg = 256u;

    let out_poly = wgid.x;

    if (out_poly > k) { return; }

    let elems_per_thread = (N + tpg - 1u) / tpg;

    // Initialize accumulator
    for (var i = 0u; i < elems_per_thread; i++) {
        let idx = tid + i * tpg;
        if (idx < N) {
            write_accum(idx, u64_zero());
        }
    }
    workgroupBarrier();

    let rot = rotation_amount;
    let N_i32 = i32(N);

    for (var in_poly = 0u; in_poly <= k; in_poly++) {
        // Load with negacyclic rotation
        for (var i = 0u; i < elems_per_thread; i++) {
            let idx = tid + i * tpg;
            if (idx < N) {
                // Compute source index for X^rot rotation
                var src_idx = (i32(idx) - rot) % (2 * N_i32);
                if (src_idx < 0) { src_idx += 2 * N_i32; }

                var negate = false;
                if (src_idx >= N_i32) {
                    src_idx -= N_i32;
                    negate = true;
                }

                let glwe_idx = in_poly * N + u32(src_idx);
                var val = read_u64_storage(&glwe_in, glwe_idx);
                if (negate) {
                    val = mod_sub(u64_zero(), val, Q);
                }
                write_poly_a(idx, val);
            }
        }
        workgroupBarrier();

        // Forward NTT
        ntt_forward_shared(N, log_N, Q, tid, tpg);

        // External product
        for (var level = 0u; level < l; level++) {
            let ggsw_base = ((in_poly * l + level) * (k + 1u) + out_poly) * N;

            for (var i = 0u; i < elems_per_thread; i++) {
                let idx = tid + i * tpg;
                if (idx < N) {
                    let val_ntt = read_poly_a(idx);
                    let digit = signed_decompose_digit(val_ntt, level, params.base_log);

                    if (digit != 0) {
                        let ggsw_val = read_u64_storage(&ggsw_ntt, ggsw_base + idx);
                        let abs_digit = u32(select(-digit, digit, digit >= 0));
                        let prod = barrett_mul(u64_from_u32(abs_digit), ggsw_val, Q, mu);

                        var acc_val = read_accum(idx);
                        if (digit > 0) {
                            acc_val = mod_add(acc_val, prod, Q);
                        } else {
                            acc_val = mod_sub(acc_val, prod, Q);
                        }
                        write_accum(idx, acc_val);
                    }
                }
            }
            workgroupBarrier();
        }
    }

    // INTT
    for (var i = 0u; i < elems_per_thread; i++) {
        let idx = tid + i * tpg;
        if (idx < N) {
            write_poly_a(idx, read_accum(idx));
        }
    }
    workgroupBarrier();

    ntt_inverse_shared(N, log_N, Q, N_inv, N_inv_precon, tid, tpg);

    // Write result
    for (var i = 0u; i < elems_per_thread; i++) {
        let idx = tid + i * tpg;
        if (idx < N) {
            let out_idx = out_poly * N + idx;
            write_u64_storage(&glwe_out, out_idx, read_poly_a(idx));
        }
    }
}
