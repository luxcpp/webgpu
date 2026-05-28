// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Negacyclic Rotation for TFHE Blind Rotation (WebGPU/WGSL)
// Computes X^k * poly in Z_Q[X]/(X^N + 1)
// Essential operation for TFHE programmable bootstrapping
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

fn u64_is_zero(a: U64) -> bool {
    return a.lo == 0u && a.hi == 0u;
}

fn u64_sub(a: U64, b: U64) -> U64 {
    let borrow = select(0u, 1u, a.lo < b.lo);
    let lo = a.lo - b.lo;
    let hi = a.hi - b.hi - borrow;
    return U64(lo, hi);
}

fn u64_add(a: U64, b: U64) -> U64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    let hi = a.hi + b.hi + carry;
    return U64(lo, hi);
}

fn u64_gte(a: U64, b: U64) -> bool {
    if (a.hi > b.hi) { return true; }
    if (a.hi < b.hi) { return false; }
    return a.lo >= b.lo;
}

fn mod_neg(a: U64, Q: U64) -> U64 {
    if (u64_is_zero(a)) { return a; }
    return u64_sub(Q, a);
}

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

// ============================================================================
// Parameters
// ============================================================================

struct NegacyclicParams {
    N: u32,           // Polynomial degree
    k: u32,           // GLWE dimension (for GLWE rotation)
    batch_size: u32,  // Batch size
    Q_lo: u32,        // Modulus low
    Q_hi: u32,        // Modulus high
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> input: array<u32>;               // [batch][N][2]
@group(0) @binding(1) var<storage, read> rotations: array<i32>;           // [batch]
@group(0) @binding(2) var<storage, read_write> output: array<u32>;        // [batch][N][2]
@group(0) @binding(3) var<uniform> params: NegacyclicParams;

var<workgroup> shared_poly: array<u32, 8192>;  // [N][2] workspace

// ============================================================================
// Rotation Index Helper
// ============================================================================

// Compute source index and negation flag for X^rotation in Z_Q[X]/(X^N + 1)
fn get_rotation_source(coeff_idx: u32, rotation: i32, N: u32) -> vec2<i32> {
    let two_N = i32(2u * N);
    let norm_rot = ((rotation % two_N) + two_N) % two_N;

    var src_idx = i32(coeff_idx) - norm_rot;
    var negate: i32 = 0;

    // Handle wraparound
    if (src_idx < 0) {
        src_idx = src_idx + i32(N);
        negate = 1 - negate;
        if (src_idx < 0) {
            src_idx = src_idx + i32(N);
            negate = 1 - negate;
        }
    }
    if (src_idx >= i32(N)) {
        src_idx = src_idx - i32(N);
        negate = 1 - negate;
        if (src_idx >= i32(N)) {
            src_idx = src_idx - i32(N);
            negate = 1 - negate;
        }
    }

    return vec2<i32>(src_idx, negate);
}

// ============================================================================
// Basic Negacyclic Rotation
// ============================================================================

@compute @workgroup_size(256)
fn negacyclic_rotate(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let coeff_idx = gid.x;
    let batch_idx = wgid.y;

    let N = params.N;
    let batch_size = params.batch_size;
    let Q = U64(params.Q_lo, params.Q_hi);

    if (coeff_idx >= N || batch_idx >= batch_size) { return; }

    let rotation = rotations[batch_idx];
    let src_info = get_rotation_source(coeff_idx, rotation, N);
    let src_idx = u32(src_info.x);
    let negate = src_info.y == 1;

    // Read source coefficient
    let in_base = batch_idx * N * 2u;
    let src_lo = input[in_base + src_idx * 2u];
    let src_hi = input[in_base + src_idx * 2u + 1u];
    var val = U64(src_lo, src_hi);

    // Apply negation if needed
    if (negate) {
        val = mod_neg(val, Q);
    }

    // Write output
    let out_base = batch_idx * N * 2u;
    output[out_base + coeff_idx * 2u] = val.lo;
    output[out_base + coeff_idx * 2u + 1u] = val.hi;
}

// ============================================================================
// In-Place Negacyclic Rotation
// ============================================================================

@group(0) @binding(4) var<storage, read_write> poly_inplace: array<u32>;  // [batch][N][2]
@group(0) @binding(5) var<uniform> single_rotation: i32;

@compute @workgroup_size(256)
fn negacyclic_rotate_inplace(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let thread_idx = lid.x;
    let threads = 256u;
    let batch_idx = wgid.x;

    let N = params.N;
    let batch_size = params.batch_size;
    let Q = U64(params.Q_lo, params.Q_hi);

    if (batch_idx >= batch_size) { return; }

    let rotation = single_rotation;
    let poly_base = batch_idx * N * 2u;

    // Load with rotation into shared memory
    for (var i = thread_idx; i < N; i += threads) {
        let src_info = get_rotation_source(i, rotation, N);
        let src_idx = u32(src_info.x);
        let negate = src_info.y == 1;

        let src_lo = poly_inplace[poly_base + src_idx * 2u];
        let src_hi = poly_inplace[poly_base + src_idx * 2u + 1u];
        var val = U64(src_lo, src_hi);

        if (negate) {
            val = mod_neg(val, Q);
        }

        shared_poly[i * 2u] = val.lo;
        shared_poly[i * 2u + 1u] = val.hi;
    }

    workgroupBarrier();

    // Write back
    for (var i = thread_idx; i < N; i += threads) {
        poly_inplace[poly_base + i * 2u] = shared_poly[i * 2u];
        poly_inplace[poly_base + i * 2u + 1u] = shared_poly[i * 2u + 1u];
    }
}

// ============================================================================
// Rotation with Accumulation
// ============================================================================

@group(0) @binding(6) var<storage, read_write> acc: array<u32>;           // [batch][N][2]
@group(0) @binding(7) var<storage, read> poly_src: array<u32>;            // [batch][N][2]

@compute @workgroup_size(256)
fn negacyclic_rotate_accumulate(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let coeff_idx = gid.x;
    let batch_idx = wgid.y;

    let N = params.N;
    let batch_size = params.batch_size;
    let Q = U64(params.Q_lo, params.Q_hi);

    if (coeff_idx >= N || batch_idx >= batch_size) { return; }

    let rotation = rotations[batch_idx];
    let src_info = get_rotation_source(coeff_idx, rotation, N);
    let src_idx = u32(src_info.x);
    let negate = src_info.y == 1;

    let base = batch_idx * N * 2u;

    // Read and rotate source
    let src_lo = poly_src[base + src_idx * 2u];
    let src_hi = poly_src[base + src_idx * 2u + 1u];
    var rotated = U64(src_lo, src_hi);

    if (negate) {
        rotated = mod_neg(rotated, Q);
    }

    // Accumulate
    let acc_lo = acc[base + coeff_idx * 2u];
    let acc_hi = acc[base + coeff_idx * 2u + 1u];
    let acc_val = U64(acc_lo, acc_hi);

    let result = mod_add(acc_val, rotated, Q);

    acc[base + coeff_idx * 2u] = result.lo;
    acc[base + coeff_idx * 2u + 1u] = result.hi;
}

// ============================================================================
// Rotation Difference for CMux
// ============================================================================

@group(0) @binding(8) var<storage, read_write> diff: array<u32>;          // [batch][N][2]

@compute @workgroup_size(256)
fn negacyclic_rotate_diff(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let coeff_idx = gid.x;
    let batch_idx = wgid.y;

    let N = params.N;
    let batch_size = params.batch_size;
    let Q = U64(params.Q_lo, params.Q_hi);

    if (coeff_idx >= N || batch_idx >= batch_size) { return; }

    let rotation = single_rotation;
    let src_info = get_rotation_source(coeff_idx, rotation, N);
    let src_idx = u32(src_info.x);
    let negate = src_info.y == 1;

    let base = batch_idx * N * 2u;

    // Original value
    let orig_lo = poly_src[base + coeff_idx * 2u];
    let orig_hi = poly_src[base + coeff_idx * 2u + 1u];
    let original = U64(orig_lo, orig_hi);

    // Rotated value
    let rot_lo = poly_src[base + src_idx * 2u];
    let rot_hi = poly_src[base + src_idx * 2u + 1u];
    var rotated = U64(rot_lo, rot_hi);

    if (negate) {
        rotated = mod_neg(rotated, Q);
    }

    // Compute difference: rotated - original
    let result = mod_sub(rotated, original, Q);

    diff[base + coeff_idx * 2u] = result.lo;
    diff[base + coeff_idx * 2u + 1u] = result.hi;
}

// ============================================================================
// GLWE Rotation (all k+1 polynomials)
// ============================================================================

@group(0) @binding(9) var<storage, read> glwe_in: array<u32>;             // [batch][(k+1)][N][2]
@group(0) @binding(10) var<storage, read_write> glwe_out: array<u32>;     // [batch][(k+1)][N][2]

@compute @workgroup_size(256)
fn negacyclic_rotate_glwe(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let coeff_idx = gid.x;
    let poly_idx = wgid.y;
    let batch_idx = wgid.z;

    let N = params.N;
    let k = params.k;
    let batch_size = params.batch_size;
    let Q = U64(params.Q_lo, params.Q_hi);

    if (coeff_idx >= N || poly_idx > k || batch_idx >= batch_size) { return; }

    let rotation = single_rotation;
    let src_info = get_rotation_source(coeff_idx, rotation, N);
    let src_idx = u32(src_info.x);
    let negate = src_info.y == 1;

    let glwe_stride = (k + 1u) * N * 2u;
    let poly_stride = N * 2u;

    let in_offset = batch_idx * glwe_stride + poly_idx * poly_stride + src_idx * 2u;
    let out_offset = batch_idx * glwe_stride + poly_idx * poly_stride + coeff_idx * 2u;

    let src_lo = glwe_in[in_offset];
    let src_hi = glwe_in[in_offset + 1u];
    var val = U64(src_lo, src_hi);

    if (negate) {
        val = mod_neg(val, Q);
    }

    glwe_out[out_offset] = val.lo;
    glwe_out[out_offset + 1u] = val.hi;
}
