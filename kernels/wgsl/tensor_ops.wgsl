// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// WGSL Tensor Operations — umbrella entry point mirroring
// cuda/kernels/tensor_ops.cu and metal/src/shaders/tensor_ops.metal.
//
// Provides the small-grained ML primitives expected at the backend root:
//   - Elementwise binary: add/sub/mul/div
//   - Unary: exp/log/sqrt/neg/abs/tanh/sigmoid/relu/gelu
//   - Copy
//   - Tiled matmul (TILE_SIZE=16)
//   - Tiled transpose
//   - LayerNorm and RMSNorm (per-row)
//   - Softmax / log-softmax (per-row)

const TILE_SIZE_TOPS: u32 = 16u;
const WG_SIZE_TOPS:   u32 = 256u;

// ============================================================================
// Common bindings (separate kernels rebind dynamically; CUDA/Metal use
// per-call buffer index; WGSL needs explicit bindings, so we split groups.)
// ============================================================================

struct ElemParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct MatMulParams {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
}

struct TransposeParams {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

struct RowParams {
    batch_size: u32,
    dim: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read>       a:   array<f32>;
@group(0) @binding(2) var<storage, read>       b:   array<f32>;
@group(0) @binding(3) var<uniform>             elem_p: ElemParams;
@group(0) @binding(4) var<uniform>             mm_p:   MatMulParams;
@group(0) @binding(5) var<uniform>             tr_p:   TransposeParams;
@group(0) @binding(6) var<uniform>             row_p:  RowParams;
@group(0) @binding(7) var<storage, read>       gamma:  array<f32>;
@group(0) @binding(8) var<storage, read>       beta_:  array<f32>;

// Workgroup scratch
var<workgroup> tile_a: array<f32, 256>;  // TILE_SIZE * TILE_SIZE
var<workgroup> tile_b: array<f32, 256>;
var<workgroup> tr_tile: array<f32, 272>; // (TILE_SIZE+1) * TILE_SIZE pad-avoid
var<workgroup> row_reduce: array<f32, 256>;
var<workgroup> wg_scalar:  f32;

// ============================================================================
// Elementwise binary
// ============================================================================
@compute @workgroup_size(256)
fn lux_add_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < elem_p.n) { out[gid.x] = a[gid.x] + b[gid.x]; }
}

@compute @workgroup_size(256)
fn lux_sub_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < elem_p.n) { out[gid.x] = a[gid.x] - b[gid.x]; }
}

@compute @workgroup_size(256)
fn lux_mul_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < elem_p.n) { out[gid.x] = a[gid.x] * b[gid.x]; }
}

@compute @workgroup_size(256)
fn lux_div_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < elem_p.n) { out[gid.x] = a[gid.x] / b[gid.x]; }
}

// ============================================================================
// Unary
// ============================================================================
@compute @workgroup_size(256)
fn lux_exp_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < elem_p.n) { out[gid.x] = exp(a[gid.x]); }
}

@compute @workgroup_size(256)
fn lux_log_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < elem_p.n) { out[gid.x] = log(a[gid.x]); }
}

@compute @workgroup_size(256)
fn lux_sqrt_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < elem_p.n) { out[gid.x] = sqrt(a[gid.x]); }
}

@compute @workgroup_size(256)
fn lux_neg_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < elem_p.n) { out[gid.x] = -a[gid.x]; }
}

@compute @workgroup_size(256)
fn lux_abs_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < elem_p.n) { out[gid.x] = abs(a[gid.x]); }
}

@compute @workgroup_size(256)
fn lux_tanh_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < elem_p.n) { out[gid.x] = tanh(a[gid.x]); }
}

@compute @workgroup_size(256)
fn lux_sigmoid_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < elem_p.n) {
        out[gid.x] = 1.0 / (1.0 + exp(-a[gid.x]));
    }
}

@compute @workgroup_size(256)
fn lux_relu_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < elem_p.n) { out[gid.x] = max(0.0, a[gid.x]); }
}

@compute @workgroup_size(256)
fn lux_gelu_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < elem_p.n) {
        let x = a[gid.x];
        let x3 = x * x * x;
        let inner = 0.7978845608 * (x + 0.044715 * x3);
        out[gid.x] = 0.5 * x * (1.0 + tanh(inner));
    }
}

@compute @workgroup_size(256)
fn lux_copy_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < elem_p.n) { out[gid.x] = a[gid.x]; }
}

// ============================================================================
// Tiled MatMul: C[M,N] = A[M,K] @ B[K,N]
// Grid: (ceil(N/16), ceil(M/16)); 16x16 workgroup
// ============================================================================
@compute @workgroup_size(16, 16)
fn lux_matmul_tiled_f32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let M = mm_p.M;
    let K = mm_p.K;
    let N = mm_p.N;
    let bx = wid.x;
    let by = wid.y;
    let tx = lid.x;
    let ty = lid.y;
    let row = by * TILE_SIZE_TOPS + ty;
    let col = bx * TILE_SIZE_TOPS + tx;
    var sum: f32 = 0.0;

    let num_tiles = (K + TILE_SIZE_TOPS - 1u) / TILE_SIZE_TOPS;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE_SIZE_TOPS + tx;
        if (row < M && a_col < K) {
            tile_a[ty * TILE_SIZE_TOPS + tx] = a[row * K + a_col];
        } else {
            tile_a[ty * TILE_SIZE_TOPS + tx] = 0.0;
        }
        let b_row = t * TILE_SIZE_TOPS + ty;
        if (b_row < K && col < N) {
            tile_b[ty * TILE_SIZE_TOPS + tx] = b[b_row * N + col];
        } else {
            tile_b[ty * TILE_SIZE_TOPS + tx] = 0.0;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < TILE_SIZE_TOPS; k = k + 1u) {
            sum = sum + tile_a[ty * TILE_SIZE_TOPS + k]
                      * tile_b[k * TILE_SIZE_TOPS + tx];
        }
        workgroupBarrier();
    }
    if (row < M && col < N) { out[row * N + col] = sum; }
}

// ============================================================================
// Tiled Transpose
// ============================================================================
@compute @workgroup_size(16, 16)
fn lux_transpose_f32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let rows = tr_p.rows;
    let cols = tr_p.cols;
    var x = wid.x * TILE_SIZE_TOPS + lid.x;
    var y = wid.y * TILE_SIZE_TOPS + lid.y;

    if (x < cols && y < rows) {
        tr_tile[lid.y * (TILE_SIZE_TOPS + 1u) + lid.x] = a[y * cols + x];
    }
    workgroupBarrier();

    x = wid.y * TILE_SIZE_TOPS + lid.x;
    y = wid.x * TILE_SIZE_TOPS + lid.y;
    if (x < rows && y < cols) {
        out[y * rows + x] = tr_tile[lid.x * (TILE_SIZE_TOPS + 1u) + lid.y];
    }
}

// ============================================================================
// LayerNorm (per-row): y = ((x - mean) / sqrt(var + eps)) * gamma + beta
// Grid: (batch_size); 256 threads
// ============================================================================
@compute @workgroup_size(256)
fn lux_layer_norm_f32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let batch_idx = wid.x;
    let tid = lid.x;
    if (batch_idx >= row_p.batch_size) { return; }
    let dim = row_p.dim;
    let row = batch_idx * dim;

    var local_sum: f32 = 0.0;
    for (var i = tid; i < dim; i = i + WG_SIZE_TOPS) {
        local_sum = local_sum + a[row + i];
    }
    row_reduce[tid] = local_sum;
    workgroupBarrier();
    var s: u32 = WG_SIZE_TOPS / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { row_reduce[tid] = row_reduce[tid] + row_reduce[tid + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    if (tid == 0u) { wg_scalar = row_reduce[0] / f32(dim); }
    workgroupBarrier();
    let mean = wg_scalar;

    var local_var: f32 = 0.0;
    for (var i = tid; i < dim; i = i + WG_SIZE_TOPS) {
        let diff = a[row + i] - mean;
        local_var = local_var + diff * diff;
    }
    row_reduce[tid] = local_var;
    workgroupBarrier();
    s = WG_SIZE_TOPS / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { row_reduce[tid] = row_reduce[tid] + row_reduce[tid + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    if (tid == 0u) { wg_scalar = row_reduce[0] / f32(dim); }
    workgroupBarrier();
    let var_v = wg_scalar;
    let inv_std = inverseSqrt(var_v + row_p.eps);

    for (var i = tid; i < dim; i = i + WG_SIZE_TOPS) {
        let normalized = (a[row + i] - mean) * inv_std;
        out[row + i] = normalized * gamma[i] + beta_[i];
    }
}

// ============================================================================
// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight (weight in gamma binding)
// ============================================================================
@compute @workgroup_size(256)
fn lux_rms_norm_f32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let batch_idx = wid.x;
    let tid = lid.x;
    if (batch_idx >= row_p.batch_size) { return; }
    let dim = row_p.dim;
    let row = batch_idx * dim;

    var local_sq: f32 = 0.0;
    for (var i = tid; i < dim; i = i + WG_SIZE_TOPS) {
        let v = a[row + i];
        local_sq = local_sq + v * v;
    }
    row_reduce[tid] = local_sq;
    workgroupBarrier();
    var s: u32 = WG_SIZE_TOPS / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { row_reduce[tid] = row_reduce[tid] + row_reduce[tid + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    if (tid == 0u) { wg_scalar = inverseSqrt(row_reduce[0] / f32(dim) + row_p.eps); }
    workgroupBarrier();
    let rms_scale = wg_scalar;

    for (var i = tid; i < dim; i = i + WG_SIZE_TOPS) {
        out[row + i] = a[row + i] * rms_scale * gamma[i];
    }
}

// ============================================================================
// Softmax along last dim (numerically stable two-pass)
// ============================================================================
@compute @workgroup_size(256)
fn lux_softmax_f32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let batch_idx = wid.x;
    let tid = lid.x;
    if (batch_idx >= row_p.batch_size) { return; }
    let dim = row_p.dim;
    let row = batch_idx * dim;

    var local_max: f32 = -3.4028235e+38;
    for (var i = tid; i < dim; i = i + WG_SIZE_TOPS) {
        local_max = max(local_max, a[row + i]);
    }
    row_reduce[tid] = local_max;
    workgroupBarrier();
    var s: u32 = WG_SIZE_TOPS / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { row_reduce[tid] = max(row_reduce[tid], row_reduce[tid + s]); }
        workgroupBarrier();
        s = s / 2u;
    }
    if (tid == 0u) { wg_scalar = row_reduce[0]; }
    workgroupBarrier();
    let max_val = wg_scalar;

    var local_sum: f32 = 0.0;
    for (var i = tid; i < dim; i = i + WG_SIZE_TOPS) {
        let e = exp(a[row + i] - max_val);
        out[row + i] = e;
        local_sum = local_sum + e;
    }
    row_reduce[tid] = local_sum;
    workgroupBarrier();
    s = WG_SIZE_TOPS / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { row_reduce[tid] = row_reduce[tid] + row_reduce[tid + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    if (tid == 0u) { wg_scalar = row_reduce[0]; }
    workgroupBarrier();
    let inv_sum = 1.0 / wg_scalar;
    for (var i = tid; i < dim; i = i + WG_SIZE_TOPS) {
        out[row + i] = out[row + i] * inv_sum;
    }
}

@compute @workgroup_size(256)
fn lux_log_softmax_f32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let batch_idx = wid.x;
    let tid = lid.x;
    if (batch_idx >= row_p.batch_size) { return; }
    let dim = row_p.dim;
    let row = batch_idx * dim;

    var local_max: f32 = -3.4028235e+38;
    for (var i = tid; i < dim; i = i + WG_SIZE_TOPS) {
        local_max = max(local_max, a[row + i]);
    }
    row_reduce[tid] = local_max;
    workgroupBarrier();
    var s: u32 = WG_SIZE_TOPS / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { row_reduce[tid] = max(row_reduce[tid], row_reduce[tid + s]); }
        workgroupBarrier();
        s = s / 2u;
    }
    if (tid == 0u) { wg_scalar = row_reduce[0]; }
    workgroupBarrier();
    let max_val = wg_scalar;

    var local_sum: f32 = 0.0;
    for (var i = tid; i < dim; i = i + WG_SIZE_TOPS) {
        local_sum = local_sum + exp(a[row + i] - max_val);
    }
    row_reduce[tid] = local_sum;
    workgroupBarrier();
    s = WG_SIZE_TOPS / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { row_reduce[tid] = row_reduce[tid] + row_reduce[tid + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    if (tid == 0u) { wg_scalar = log(row_reduce[0]); }
    workgroupBarrier();
    let log_sum = wg_scalar;
    for (var i = tid; i < dim; i = i + WG_SIZE_TOPS) {
        out[row + i] = a[row + i] - max_val - log_sum;
    }
}
