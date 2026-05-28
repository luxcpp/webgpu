// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel GEMM WGSL (umbrella for cuda/kernels/steel/gemm/gemm.cu and
// metal/src/shaders/steel/gemm/gemm.metal).
//
//   steel_gemm                 - C = alpha * A @ B + beta * C (FP32 tiled)
//   steel_gemm_tn              - C = alpha * A^T @ B + beta * C
//   steel_gemv                 - matrix-vector multiply
//   steel_gemm_strided_batched - batched GEMM with per-batch strides
//
// Tile sizes: TILE_M=128, TILE_N=128, TILE_K=8 (matches CUDA file)
// 256-thread workgroup; tx = tid%16, ty = tid/16.

const TILE_M_G: u32 = 128u;
const TILE_N_G: u32 = 128u;
const TILE_K_G: u32 = 8u;
const WG_SIZE_G: u32 = 256u;

struct GemmParams {
    alpha: f32,
    beta:  f32,
    M:     u32,
    N:     u32,
    K:     u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct GemmBatchedParams {
    alpha:    f32,
    beta:     f32,
    M:        u32,
    N:        u32,
    K:        u32,
    batch_size: u32,
    stride_a: u32,
    stride_b: u32,
    stride_c: u32,
    _pad0:    u32,
    _pad1:    u32,
    _pad2:    u32,
}

@group(0) @binding(0) var<storage, read_write> C: array<f32>;
@group(0) @binding(1) var<storage, read>       A: array<f32>;
@group(0) @binding(2) var<storage, read>       B: array<f32>;
@group(0) @binding(3) var<uniform>             params: GemmParams;
@group(0) @binding(4) var<uniform>             batched_params: GemmBatchedParams;
@group(0) @binding(5) var<storage, read>       Avec: array<f32>;     // for GEMV: A flattened
@group(0) @binding(6) var<storage, read>       Xvec: array<f32>;     // for GEMV: x
@group(0) @binding(7) var<storage, read_write> Yvec: array<f32>;     // for GEMV: y

var<workgroup> As: array<f32, 1024>;   // TILE_M * TILE_K = 128 * 8
var<workgroup> Bs: array<f32, 1024>;   // TILE_K * TILE_N = 8 * 128
var<workgroup> gemv_partial: array<f32, 256>;

// ============================================================================
// FP32 GEMM: C = alpha * A @ B + beta * C
// Grid: (ceil(N/TILE_N), ceil(M/TILE_M), 1); 256 threads
// ============================================================================
@compute @workgroup_size(256)
fn steel_gemm(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let M = params.M;
    let N = params.N;
    let K = params.K;
    let bx = wid.x;
    let by = wid.y;
    let tx = tid % 16u;
    let ty = tid / 16u;
    let row = by * TILE_M_G + ty;
    let col = bx * TILE_N_G + tx;
    var acc: f32 = 0.0;

    let num_k_tiles = (K + TILE_K_G - 1u) / TILE_K_G;
    for (var t: u32 = 0u; t < num_k_tiles; t = t + 1u) {
        for (var i = tid; i < TILE_M_G * TILE_K_G; i = i + WG_SIZE_G) {
            let ti = i / TILE_K_G;
            let tk = i % TILE_K_G;
            let a_row = by * TILE_M_G + ti;
            let a_col = t * TILE_K_G + tk;
            if (a_row < M && a_col < K) {
                As[ti * TILE_K_G + tk] = A[a_row * K + a_col];
            } else {
                As[ti * TILE_K_G + tk] = 0.0;
            }
        }
        for (var i = tid; i < TILE_K_G * TILE_N_G; i = i + WG_SIZE_G) {
            let tk = i / TILE_N_G;
            let tj = i % TILE_N_G;
            let b_row = t * TILE_K_G + tk;
            let b_col = bx * TILE_N_G + tj;
            if (b_row < K && b_col < N) {
                Bs[tk * TILE_N_G + tj] = B[b_row * N + b_col];
            } else {
                Bs[tk * TILE_N_G + tj] = 0.0;
            }
        }
        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_K_G; k = k + 1u) {
            acc = acc + As[ty * TILE_K_G + k] * Bs[k * TILE_N_G + tx];
        }
        workgroupBarrier();
    }

    if (row < M && col < N) {
        if (params.beta != 0.0) {
            C[row * N + col] = params.alpha * acc + params.beta * C[row * N + col];
        } else {
            C[row * N + col] = params.alpha * acc;
        }
    }
}

// ============================================================================
// FP32 GEMM TN: A stored [K, M], accessed as A^T
// ============================================================================
@compute @workgroup_size(256)
fn steel_gemm_tn(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let M = params.M;
    let N = params.N;
    let K = params.K;
    let bx = wid.x;
    let by = wid.y;
    let tx = tid % 16u;
    let ty = tid / 16u;
    let row = by * TILE_M_G + ty;
    let col = bx * TILE_N_G + tx;
    var acc: f32 = 0.0;

    for (var t: u32 = 0u; t < (K + TILE_K_G - 1u) / TILE_K_G; t = t + 1u) {
        for (var i = tid; i < TILE_K_G * TILE_M_G; i = i + WG_SIZE_G) {
            let tk = i / TILE_M_G;
            let ti = i % TILE_M_G;
            let a_row = t * TILE_K_G + tk;
            let a_col = by * TILE_M_G + ti;
            if (a_row < K && a_col < M) {
                // store As[tk][ti] flattened as tk*TILE_M + ti
                As[tk * TILE_M_G + ti] = A[a_row * M + a_col];
            } else {
                As[tk * TILE_M_G + ti] = 0.0;
            }
        }
        for (var i = tid; i < TILE_K_G * TILE_N_G; i = i + WG_SIZE_G) {
            let tk = i / TILE_N_G;
            let tj = i % TILE_N_G;
            let b_row = t * TILE_K_G + tk;
            let b_col = bx * TILE_N_G + tj;
            if (b_row < K && b_col < N) {
                Bs[tk * TILE_N_G + tj] = B[b_row * N + b_col];
            } else {
                Bs[tk * TILE_N_G + tj] = 0.0;
            }
        }
        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_K_G; k = k + 1u) {
            acc = acc + As[k * TILE_M_G + ty] * Bs[k * TILE_N_G + tx];
        }
        workgroupBarrier();
    }

    if (row < M && col < N) {
        if (params.beta != 0.0) {
            C[row * N + col] = params.alpha * acc + params.beta * C[row * N + col];
        } else {
            C[row * N + col] = params.alpha * acc;
        }
    }
}

// ============================================================================
// GEMV: y = alpha * A @ x + beta * y
// Grid: (M); 256 threads with workgroup-array reduction.
// ============================================================================
@compute @workgroup_size(256)
fn steel_gemv(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wid.x;
    let tid = lid.x;
    let M = params.M;
    let N = params.N;
    if (row >= M) { return; }

    var partial: f32 = 0.0;
    for (var j = tid; j < N; j = j + WG_SIZE_G) {
        partial = partial + Avec[row * N + j] * Xvec[j];
    }
    gemv_partial[tid] = partial;
    workgroupBarrier();

    var s: u32 = WG_SIZE_G / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) {
            gemv_partial[tid] = gemv_partial[tid] + gemv_partial[tid + s];
        }
        workgroupBarrier();
        s = s / 2u;
    }
    if (tid == 0u) {
        if (params.beta != 0.0) {
            Yvec[row] = params.alpha * gemv_partial[0] + params.beta * Yvec[row];
        } else {
            Yvec[row] = params.alpha * gemv_partial[0];
        }
    }
}

// ============================================================================
// Strided Batched GEMM
// Grid: (ceil(N/TILE_N), ceil(M/TILE_M), batch_size); 256 threads
// ============================================================================
@compute @workgroup_size(256)
fn steel_gemm_strided_batched(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let M = batched_params.M;
    let N = batched_params.N;
    let K = batched_params.K;
    let batch = wid.z;
    if (batch >= batched_params.batch_size) { return; }

    let A_off = batch * batched_params.stride_a;
    let B_off = batch * batched_params.stride_b;
    let C_off = batch * batched_params.stride_c;

    let bx = wid.x;
    let by = wid.y;
    let tx = tid % 16u;
    let ty = tid / 16u;
    let row = by * TILE_M_G + ty;
    let col = bx * TILE_N_G + tx;
    var acc: f32 = 0.0;

    for (var t: u32 = 0u; t < (K + TILE_K_G - 1u) / TILE_K_G; t = t + 1u) {
        for (var i = tid; i < TILE_M_G * TILE_K_G; i = i + WG_SIZE_G) {
            let ti = i / TILE_K_G;
            let tk = i % TILE_K_G;
            let a_row = by * TILE_M_G + ti;
            let a_col = t * TILE_K_G + tk;
            if (a_row < M && a_col < K) {
                As[ti * TILE_K_G + tk] = A[A_off + a_row * K + a_col];
            } else {
                As[ti * TILE_K_G + tk] = 0.0;
            }
        }
        for (var i = tid; i < TILE_K_G * TILE_N_G; i = i + WG_SIZE_G) {
            let tk = i / TILE_N_G;
            let tj = i % TILE_N_G;
            let b_row = t * TILE_K_G + tk;
            let b_col = bx * TILE_N_G + tj;
            if (b_row < K && b_col < N) {
                Bs[tk * TILE_N_G + tj] = B[B_off + b_row * N + b_col];
            } else {
                Bs[tk * TILE_N_G + tj] = 0.0;
            }
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < TILE_K_G; k = k + 1u) {
            acc = acc + As[ty * TILE_K_G + k] * Bs[k * TILE_N_G + tx];
        }
        workgroupBarrier();
    }

    if (row < M && col < N) {
        if (batched_params.beta != 0.0) {
            C[C_off + row * N + col] = batched_params.alpha * acc
                                     + batched_params.beta * C[C_off + row * N + col];
        } else {
            C[C_off + row * N + col] = batched_params.alpha * acc;
        }
    }
}
