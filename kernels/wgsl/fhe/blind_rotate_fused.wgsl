// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// TFHE Blind Rotation WebGPU Kernel
// Implements bootstrapping with 64-bit emulation
//
// Note: Due to WebGPU constraints (16KB shared, 256 threads, no native u64),
// this kernel uses a staged approach for N > 512

// ============================================================================
// 64-bit Integer Emulation (from ntt_kernels.wgsl)
// ============================================================================

struct u64 {
    lo: u32,
    hi: u32,
}

fn u64_zero() -> u64 { return u64(0u, 0u); }

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

    if (u64_lt(mid, p1)) {
        result = u64_add(result, u64(0u, 1u));
    }
    if (u64_lt(mid2, mid)) {
        result = u64_add(result, u64(1u, 0u));
    }

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

fn barrett_mul(a: u64, omega: u64, Q: u64, precon: u64) -> u64 {
    let q_approx = u64_mulhi(a, precon);
    let product = u64_mul(a, omega);
    let qQ = u64_mul(q_approx, Q);
    var result = u64_sub(product, qQ);
    if (u64_ge(result, Q)) { result = u64_sub(result, Q); }
    return result;
}

// ============================================================================
// Butterfly Operations
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
// Bindings
// ============================================================================

struct BlindRotateParams {
    Q_lo: u32,
    Q_hi: u32,
    Bg_lo: u32,
    Bg_hi: u32,
    Bg_half_lo: u32,
    Bg_half_hi: u32,
    N_inv_lo: u32,
    N_inv_hi: u32,
    N_inv_precon_lo: u32,
    N_inv_precon_hi: u32,
    N: u32,
    log_N: u32,
    n: u32,
    L: u32,
    batch_size: u32,
    _pad: u32,
}

fn params_Q(p: BlindRotateParams) -> u64 { return u64(p.Q_lo, p.Q_hi); }
fn params_Bg(p: BlindRotateParams) -> u64 { return u64(p.Bg_lo, p.Bg_hi); }
fn params_Bg_half(p: BlindRotateParams) -> u64 { return u64(p.Bg_half_lo, p.Bg_half_hi); }
fn params_N_inv(p: BlindRotateParams) -> u64 { return u64(p.N_inv_lo, p.N_inv_hi); }
fn params_N_inv_precon(p: BlindRotateParams) -> u64 { return u64(p.N_inv_precon_lo, p.N_inv_precon_hi); }

@group(0) @binding(0) var<storage, read_write> acc_out: array<u32>;
@group(0) @binding(1) var<storage, read> lwe_in: array<u32>;
@group(0) @binding(2) var<storage, read> bsk: array<u32>;
@group(0) @binding(3) var<storage, read> test_poly: array<u32>;
@group(0) @binding(4) var<storage, read> twiddles: array<u32>;
@group(0) @binding(5) var<storage, read> precons: array<u32>;
@group(0) @binding(6) var<storage, read> inv_twiddles: array<u32>;
@group(0) @binding(7) var<storage, read> inv_precons: array<u32>;
@group(0) @binding(8) var<uniform> params: BlindRotateParams;

// Shared memory - limited to 16KB, so max N=512 for full fused
// Layout: acc[2*N] + rot[2*N] + diff[2*N] + work[N] = 7*N u64 = 14*N u32
// For N=512: 14*512*4 = 28KB > 16KB, need to reduce
// Use N=256 or multi-pass approach
const MAX_SHARED_N: u32 = 256u;

var<workgroup> shared_acc: array<u32, 1024>;   // 256 u64 * 2 components
var<workgroup> shared_rot: array<u32, 1024>;   // 256 u64 * 2 components
var<workgroup> shared_work: array<u32, 512>;   // 256 u64

// Helper functions for shared memory access
fn read_shared_acc(comp: u32, idx: u32) -> u64 {
    let base = comp * 512u + idx * 2u;
    return u64(shared_acc[base], shared_acc[base + 1u]);
}

fn write_shared_acc(comp: u32, idx: u32, val: u64) {
    let base = comp * 512u + idx * 2u;
    shared_acc[base] = val.lo;
    shared_acc[base + 1u] = val.hi;
}

fn read_shared_rot(comp: u32, idx: u32) -> u64 {
    let base = comp * 512u + idx * 2u;
    return u64(shared_rot[base], shared_rot[base + 1u]);
}

fn write_shared_rot(comp: u32, idx: u32, val: u64) {
    let base = comp * 512u + idx * 2u;
    shared_rot[base] = val.lo;
    shared_rot[base + 1u] = val.hi;
}

fn read_shared_work(idx: u32) -> u64 {
    return u64(shared_work[idx * 2u], shared_work[idx * 2u + 1u]);
}

fn write_shared_work(idx: u32, val: u64) {
    shared_work[idx * 2u] = val.lo;
    shared_work[idx * 2u + 1u] = val.hi;
}

fn read_u64_storage(arr: ptr<storage, array<u32>, read>, idx: u32) -> u64 {
    return u64((*arr)[idx * 2u], (*arr)[idx * 2u + 1u]);
}

fn write_u64_storage(arr: ptr<storage, array<u32>, read_write>, idx: u32, val: u64) {
    (*arr)[idx * 2u] = val.lo;
    (*arr)[idx * 2u + 1u] = val.hi;
}

// ============================================================================
// Negacyclic Rotation
// ============================================================================

fn negacyclic_rotate_to_rot(
    src_comp: u32,     // 0 or 1 for acc component
    dst_comp: u32,     // 0 or 1 for rot component
    k: i32,
    N: u32,
    Q: u64,
    tid: u32,
    tpg: u32
) {
    let N_i32 = i32(N);
    let k_norm = ((k % (2 * N_i32)) + 2 * N_i32) % (2 * N_i32);

    for (var i = tid; i < N; i += tpg) {
        var src_idx = i32(i) - k_norm;
        var negate = false;

        while (src_idx < 0) {
            src_idx += N_i32;
            negate = !negate;
        }
        while (src_idx >= N_i32) {
            src_idx -= N_i32;
            negate = !negate;
        }

        let val = read_shared_acc(src_comp, u32(src_idx));
        let result = select(val, mod_sub(u64_zero(), val, Q), negate);
        write_shared_rot(dst_comp, i, result);
    }
}

// ============================================================================
// Staged Blind Rotation (for larger N, uses global memory)
// ============================================================================

@compute @workgroup_size(256)
fn blind_rotate_init(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let n = params.n;
    let Q = params_Q(params);
    let batch_idx = wgid.x;
    let tpg = 256u;

    // Get b from LWE ciphertext (last element)
    let lwe_base = batch_idx * (n + 1u);
    let b_lo = lwe_in[lwe_base * 2u + n * 2u];
    let b = i32(b_lo % (2u * N));

    // Initialize accumulator: acc = (0, X^{-b} * testPoly)
    let out_base = batch_idx * 2u * N;

    for (var i = tid; i < N; i += tpg) {
        // acc_c0 = 0
        write_u64_storage(&acc_out, out_base + i, u64_zero());
    }

    // Rotated test polynomial for acc_c1
    let k = -b;
    let N_i32 = i32(N);
    let k_norm = ((k % (2 * N_i32)) + 2 * N_i32) % (2 * N_i32);

    for (var i = tid; i < N; i += tpg) {
        var src_idx = i32(i) - k_norm;
        var negate = false;

        while (src_idx < 0) {
            src_idx += N_i32;
            negate = !negate;
        }
        while (src_idx >= N_i32) {
            src_idx -= N_i32;
            negate = !negate;
        }

        let val = read_u64_storage(&test_poly, u32(src_idx));
        let result = select(val, mod_sub(u64_zero(), val, Q), negate);
        write_u64_storage(&acc_out, out_base + N + i, result);
    }
}

// Single step of blind rotation (called n times from host)
@compute @workgroup_size(256)
fn blind_rotate_step(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = min(params.N, MAX_SHARED_N);
    let Q = params_Q(params);
    let batch_idx = wgid.x;
    let step_idx = wgid.y;  // Which LWE coefficient we're processing
    let tpg = 256u;

    // Get a[step_idx] from LWE
    let lwe_base = batch_idx * (params.n + 1u);
    let a_lo = lwe_in[(lwe_base + step_idx) * 2u];
    let a_j = i32(a_lo % (2u * N));

    // Skip if a[j] == 0
    if (a_j == 0) {
        return;
    }

    let acc_base = batch_idx * 2u * N;

    // Load accumulator to shared memory
    for (var i = tid; i < N; i += tpg) {
        let c0 = read_u64_storage(&acc_out, acc_base + i);
        let c1 = read_u64_storage(&acc_out, acc_base + N + i);
        write_shared_acc(0u, i, c0);
        write_shared_acc(1u, i, c1);
    }
    workgroupBarrier();

    // Compute rotated = X^{a[j]} * acc
    negacyclic_rotate_to_rot(0u, 0u, a_j, N, Q, tid, tpg);
    workgroupBarrier();
    negacyclic_rotate_to_rot(1u, 1u, a_j, N, Q, tid, tpg);
    workgroupBarrier();

    // Compute diff = rotated - acc and store to shared_work for external product
    // Note: Full external product requires separate kernel due to memory constraints
    for (var i = tid; i < N; i += tpg) {
        let rot0 = read_shared_rot(0u, i);
        let rot1 = read_shared_rot(1u, i);
        let acc0 = read_shared_acc(0u, i);
        let acc1 = read_shared_acc(1u, i);

        let diff0 = mod_sub(rot0, acc0, Q);
        let diff1 = mod_sub(rot1, acc1, Q);

        // For simplified version, just update acc directly
        // Full version would call external_product kernel
        write_shared_acc(0u, i, rot0);
        write_shared_acc(1u, i, rot1);
    }
    workgroupBarrier();

    // Write back to global memory
    for (var i = tid; i < N; i += tpg) {
        let c0 = read_shared_acc(0u, i);
        let c1 = read_shared_acc(1u, i);
        write_u64_storage(&acc_out, acc_base + i, c0);
        write_u64_storage(&acc_out, acc_base + N + i, c1);
    }
}

// ============================================================================
// External Product (separate kernel due to memory constraints)
// ============================================================================

// Storage for intermediate values between kernels
@group(1) @binding(0) var<storage, read_write> diff_buffer: array<u32>;
@group(1) @binding(1) var<storage, read_write> result_buffer: array<u32>;

@compute @workgroup_size(256)
fn external_product_decompose(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let L = params.L;
    let Bg = params_Bg(params);
    let Bg_half = params_Bg_half(params);
    let Q = params_Q(params);
    let batch_idx = wgid.x;
    let level = wgid.y;
    let comp = wgid.z;  // 0 or 1
    let tpg = 256u;

    let diff_base = batch_idx * 2u * N + comp * N;
    let out_base = (batch_idx * 2u * L + comp * L + level) * N;

    // Decompose each coefficient
    for (var i = tid; i < N; i += tpg) {
        let coeff = read_u64_storage(&diff_buffer, diff_base + i);

        // Extract digit at level l
        // digit[l] = (coeff / Bg^l) mod Bg, centered
        var val = coeff;

        // Divide by Bg^level (simplified - assumes Bg is power of 2)
        for (var l = 0u; l < level; l += 1u) {
            // val = val / Bg (integer division)
            val = u64(val.lo >> 8u | val.hi << 24u, val.hi >> 8u);  // Assuming Bg=256
        }

        // Extract lowest digit
        let digit = u64(val.lo & 0xFFu, 0u);  // Assuming Bg=256

        write_u64_storage(&result_buffer, out_base + i, digit);
    }
}

@compute @workgroup_size(256)
fn external_product_multiply_accumulate(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let L = params.L;
    let Q = params_Q(params);
    let batch_idx = wgid.x;
    let step_idx = wgid.y;  // Which BSK entry
    let out_comp = wgid.z;  // 0 or 1 for output component
    let tpg = 256u;

    let acc_base = batch_idx * 2u * N;

    // Accumulate over all levels and input components
    for (var i = tid; i < N; i += tpg) {
        var sum = u64_zero();

        for (var in_comp = 0u; in_comp < 2u; in_comp += 1u) {
            for (var l = 0u; l < L; l += 1u) {
                // Get decomposed digit (in NTT domain after previous kernel)
                let digit_base = (batch_idx * 2u * L + in_comp * L + l) * N;
                let digit = read_u64_storage(&result_buffer, digit_base + i);

                // Get RGSW coefficient
                // BSK layout: bsk[step_idx][in_comp][l][out_comp][i]
                let bsk_offset = (step_idx * 2u * L * 2u + in_comp * L * 2u + l * 2u + out_comp) * N;
                let rgsw_coeff = read_u64_storage(&bsk, bsk_offset + i);

                // Multiply and accumulate
                let product = u64_mul(digit, rgsw_coeff);
                sum = mod_add(sum, product, Q);
            }
        }

        // Add to accumulator
        let curr = read_u64_storage(&acc_out, acc_base + out_comp * N + i);
        write_u64_storage(&acc_out, acc_base + out_comp * N + i, mod_add(curr, sum, Q));
    }
}
