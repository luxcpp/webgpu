// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Tiled Convolution in WGSL
// Works on WebGPU (Metal/Vulkan/D3D12 via Dawn/wgpu)
//
// Part of the Lux Network GPU acceleration library
// Implements im2col + GEMM style tiled convolution for optimal performance

// ============================================================================
// Configuration Constants
// ============================================================================

const TILE_M: u32 = 64u;       // Output tile height
const TILE_N: u32 = 64u;       // Output tile width
const TILE_K: u32 = 16u;       // Reduction tile (C_in * Kh * Kw chunk)
const BLOCK_SIZE: u32 = 256u;

// ============================================================================
// Kernel Bindings
// ============================================================================

struct ConvParams {
    batch_size: u32,           // N
    in_channels: u32,          // C_in
    out_channels: u32,         // C_out
    in_height: u32,            // H_in
    in_width: u32,             // W_in
    out_height: u32,           // H_out
    out_width: u32,            // W_out
    kernel_h: u32,             // Kh
    kernel_w: u32,             // Kw
    stride_h: u32,             // Stride height
    stride_w: u32,             // Stride width
    pad_h: u32,                // Padding height
    pad_w: u32,                // Padding width
    dilation_h: u32,           // Dilation height
    dilation_w: u32,           // Dilation width
    groups: u32,               // Group convolution
}

// Input: [N, C_in, H_in, W_in]
// Weight: [C_out, C_in/groups, Kh, Kw]
// Bias: [C_out]
// Output: [N, C_out, H_out, W_out]
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: ConvParams;

// Shared memory for tiled computation
var<workgroup> tile_input: array<f32, 4096>;   // Unrolled input patch
var<workgroup> tile_weight: array<f32, 4096>;  // Weight tile
var<workgroup> tile_output: array<f32, 4096>;  // Output accumulator

// ============================================================================
// Index Helpers
// ============================================================================

// Input index: [N, C, H, W]
fn idx_nchw(n: u32, c: u32, h: u32, w: u32, C: u32, H: u32, W: u32) -> u32 {
    return ((n * C + c) * H + h) * W + w;
}

// Weight index: [C_out, C_in/groups, Kh, Kw]
fn idx_weight(co: u32, ci: u32, kh: u32, kw: u32, CI: u32, Kh: u32, Kw: u32) -> u32 {
    return ((co * CI + ci) * Kh + kh) * Kw + kw;
}

// ============================================================================
// Direct Convolution (for small kernels)
// ============================================================================

@compute @workgroup_size(256)
fn steel_conv_direct(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_idx = wid.z / params.out_channels;
    let out_c = wid.z % params.out_channels;

    if (batch_idx >= params.batch_size) { return; }

    let out_h = params.out_height;
    let out_w = params.out_width;
    let in_c = params.in_channels;
    let in_h = params.in_height;
    let in_w = params.in_width;
    let kh = params.kernel_h;
    let kw = params.kernel_w;
    let stride_h = params.stride_h;
    let stride_w = params.stride_w;
    let pad_h = params.pad_h;
    let pad_w = params.pad_w;
    let dilation_h = params.dilation_h;
    let dilation_w = params.dilation_w;
    let groups = params.groups;

    // Group convolution parameters
    let channels_per_group_in = in_c / groups;
    let channels_per_group_out = params.out_channels / groups;
    let group_idx = out_c / channels_per_group_out;
    let in_c_start = group_idx * channels_per_group_in;

    // Each thread computes one output pixel
    let out_spatial = wid.x * BLOCK_SIZE + lid.x;
    let out_row = out_spatial / out_w;
    let out_col = out_spatial % out_w;

    if (out_row >= out_h || out_col >= out_w) { return; }

    // Compute convolution
    var sum: f32 = 0.0;

    for (var ic = 0u; ic < channels_per_group_in; ic++) {
        for (var ky = 0u; ky < kh; ky++) {
            for (var kx = 0u; kx < kw; kx++) {
                let in_row = i32(out_row * stride_h + ky * dilation_h) - i32(pad_h);
                let in_col = i32(out_col * stride_w + kx * dilation_w) - i32(pad_w);

                if (in_row >= 0 && in_row < i32(in_h) && in_col >= 0 && in_col < i32(in_w)) {
                    let in_idx = idx_nchw(batch_idx, in_c_start + ic,
                                         u32(in_row), u32(in_col), in_c, in_h, in_w);
                    let w_idx = idx_weight(out_c, ic, ky, kx,
                                          channels_per_group_in, kh, kw);

                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    // Add bias
    sum += bias[out_c];

    // Store output
    let out_idx = idx_nchw(batch_idx, out_c, out_row, out_col,
                          params.out_channels, out_h, out_w);
    output[out_idx] = sum;
}

// ============================================================================
// Im2Col + GEMM Convolution (for larger kernels)
// ============================================================================

@compute @workgroup_size(256)
fn steel_conv_im2col(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // This kernel reformulates conv as matrix multiplication:
    // Output[N, C_out, H_out*W_out] = Weight[C_out, K] @ Im2Col[K, H_out*W_out]
    // where K = C_in * Kh * Kw

    let tid = lid.x;
    let batch_idx = wid.z;

    if (batch_idx >= params.batch_size) { return; }

    let out_c = params.out_channels;
    let in_c = params.in_channels;
    let out_h = params.out_height;
    let out_w = params.out_width;
    let in_h = params.in_height;
    let in_w = params.in_width;
    let kh = params.kernel_h;
    let kw = params.kernel_w;
    let stride_h = params.stride_h;
    let stride_w = params.stride_w;
    let pad_h = params.pad_h;
    let pad_w = params.pad_w;

    let out_spatial = out_h * out_w;
    let K = in_c * kh * kw;

    // Tile dimensions for output
    let out_tile_row = wid.x * TILE_M;  // C_out dimension
    let out_tile_col = wid.y * TILE_N;  // Spatial dimension

    let num_out_rows = min(TILE_M, out_c - out_tile_row);
    let num_out_cols = min(TILE_N, out_spatial - out_tile_col);

    // Initialize output tile
    for (var i = tid; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        tile_output[i] = 0.0;
    }
    workgroupBarrier();

    // Iterate over K dimension in tiles
    let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var k_tile = 0u; k_tile < num_k_tiles; k_tile++) {
        let k_start = k_tile * TILE_K;
        let num_k = min(TILE_K, K - k_start);

        // Load weight tile [num_out_rows, num_k]
        for (var i = tid; i < num_out_rows * num_k; i += BLOCK_SIZE) {
            let row = i / num_k;
            let col = i % num_k;
            let oc = out_tile_row + row;
            let k_idx = k_start + col;

            // Decode k_idx into (ic, ky, kx)
            let ic = k_idx / (kh * kw);
            let rem = k_idx % (kh * kw);
            let ky = rem / kw;
            let kx = rem % kw;

            let w_idx = idx_weight(oc, ic, ky, kx, in_c, kh, kw);
            tile_weight[row * TILE_K + col] = weight[w_idx];
        }

        // Load im2col tile [num_k, num_out_cols]
        for (var i = tid; i < num_k * num_out_cols; i += BLOCK_SIZE) {
            let row = i / num_out_cols;
            let col = i % num_out_cols;
            let k_idx = k_start + row;
            let spatial_idx = out_tile_col + col;

            // Decode k_idx and spatial_idx
            let ic = k_idx / (kh * kw);
            let rem = k_idx % (kh * kw);
            let ky = rem / kw;
            let kx = rem % kw;

            let out_row = spatial_idx / out_w;
            let out_col = spatial_idx % out_w;

            let in_row = i32(out_row * stride_h + ky) - i32(pad_h);
            let in_col = i32(out_col * stride_w + kx) - i32(pad_w);

            var val: f32 = 0.0;
            if (in_row >= 0 && in_row < i32(in_h) && in_col >= 0 && in_col < i32(in_w)) {
                let in_idx = idx_nchw(batch_idx, ic, u32(in_row), u32(in_col),
                                     in_c, in_h, in_w);
                val = input[in_idx];
            }
            tile_input[row * TILE_N + col] = val;
        }
        workgroupBarrier();

        // Compute tile of output: C = A @ B
        for (var i = tid; i < num_out_rows * num_out_cols; i += BLOCK_SIZE) {
            let row = i / num_out_cols;
            let col = i % num_out_cols;

            var sum: f32 = 0.0;
            for (var k = 0u; k < num_k; k++) {
                let a_val = tile_weight[row * TILE_K + k];
                let b_val = tile_input[k * TILE_N + col];
                sum += a_val * b_val;
            }
            tile_output[row * TILE_N + col] += sum;
        }
        workgroupBarrier();
    }

    // Store output with bias
    for (var i = tid; i < num_out_rows * num_out_cols; i += BLOCK_SIZE) {
        let row = i / num_out_cols;
        let col = i % num_out_cols;
        let oc = out_tile_row + row;
        let spatial_idx = out_tile_col + col;

        let out_row = spatial_idx / out_w;
        let out_col = spatial_idx % out_w;

        let out_idx = idx_nchw(batch_idx, oc, out_row, out_col,
                              out_c, out_h, out_w);
        output[out_idx] = tile_output[row * TILE_N + col] + bias[oc];
    }
}

// ============================================================================
// Winograd Convolution (for 3x3 kernels)
// ============================================================================

@compute @workgroup_size(256)
fn steel_conv_winograd_3x3(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Winograd F(2x2, 3x3) - computes 2x2 output with 3x3 kernel
    // Uses 4x4 tiles to compute 2x2 output tiles
    // Reduces multiplications from 36 to 16 per 2x2 output

    let tid = lid.x;
    let batch_idx = wid.z / params.out_channels;
    let out_c = wid.z % params.out_channels;

    if (batch_idx >= params.batch_size) { return; }
    if (params.kernel_h != 3u || params.kernel_w != 3u) { return; }  // Only for 3x3
    if (params.stride_h != 1u || params.stride_w != 1u) { return; }  // Only for stride=1

    let in_c = params.in_channels;
    let out_h = params.out_height;
    let out_w = params.out_width;
    let in_h = params.in_height;
    let in_w = params.in_width;
    let groups = params.groups;

    let channels_per_group_in = in_c / groups;
    let channels_per_group_out = params.out_channels / groups;
    let group_idx = out_c / channels_per_group_out;
    let in_c_start = group_idx * channels_per_group_in;

    // Each workgroup handles multiple 2x2 output tiles
    let tiles_per_row = (out_w + 1u) / 2u;
    let tiles_per_col = (out_h + 1u) / 2u;
    let total_tiles = tiles_per_row * tiles_per_col;

    let tile_idx = wid.x * (BLOCK_SIZE / 4u) + tid / 4u;
    let elem_in_tile = tid % 4u;

    if (tile_idx >= total_tiles) { return; }

    let tile_row = tile_idx / tiles_per_row;
    let tile_col = tile_idx % tiles_per_row;
    let out_row_base = tile_row * 2u;
    let out_col_base = tile_col * 2u;

    // Winograd transform matrices for F(2x2, 3x3)
    // B^T = [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]
    // G   = [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]]
    // A^T = [[1, 1, 1, 0], [0, 1, -1, -1]]

    // Accumulator for this 2x2 tile
    var acc: array<f32, 4>;
    for (var i = 0u; i < 4u; i++) { acc[i] = 0.0; }

    // Loop over input channels
    for (var ic = 0u; ic < channels_per_group_in; ic++) {
        // Load 4x4 input tile
        var d: array<f32, 16>;
        for (var i = 0u; i < 4u; i++) {
            for (var j = 0u; j < 4u; j++) {
                let in_row = i32(out_row_base + i) - 1i;  // -1 for 3x3 kernel
                let in_col = i32(out_col_base + j) - 1i;

                if (in_row >= 0 && in_row < i32(in_h) && in_col >= 0 && in_col < i32(in_w)) {
                    let idx = idx_nchw(batch_idx, in_c_start + ic,
                                      u32(in_row), u32(in_col), in_c, in_h, in_w);
                    d[i * 4u + j] = input[idx];
                } else {
                    d[i * 4u + j] = 0.0;
                }
            }
        }

        // Load 3x3 weight and transform to 4x4
        var g: array<f32, 9>;
        for (var i = 0u; i < 3u; i++) {
            for (var j = 0u; j < 3u; j++) {
                let idx = idx_weight(out_c, ic, i, j, channels_per_group_in, 3u, 3u);
                g[i * 3u + j] = weight[idx];
            }
        }

        // Transform input: BT @ d @ B
        var BT_d: array<f32, 16>;
        // Row 0: d[0] - d[2]
        BT_d[0] = d[0] - d[8];
        BT_d[1] = d[1] - d[9];
        BT_d[2] = d[2] - d[10];
        BT_d[3] = d[3] - d[11];
        // Row 1: d[1] + d[2]
        BT_d[4] = d[4] + d[8];
        BT_d[5] = d[5] + d[9];
        BT_d[6] = d[6] + d[10];
        BT_d[7] = d[7] + d[11];
        // Row 2: -d[1] + d[2]
        BT_d[8] = -d[4] + d[8];
        BT_d[9] = -d[5] + d[9];
        BT_d[10] = -d[6] + d[10];
        BT_d[11] = -d[7] + d[11];
        // Row 3: d[1] - d[3]
        BT_d[12] = d[4] - d[12];
        BT_d[13] = d[5] - d[13];
        BT_d[14] = d[6] - d[14];
        BT_d[15] = d[7] - d[15];

        // BT_d @ B (column transform)
        var V: array<f32, 16>;
        for (var i = 0u; i < 4u; i++) {
            V[i * 4u + 0u] = BT_d[i * 4u + 0u] - BT_d[i * 4u + 2u];
            V[i * 4u + 1u] = BT_d[i * 4u + 1u] + BT_d[i * 4u + 2u];
            V[i * 4u + 2u] = -BT_d[i * 4u + 1u] + BT_d[i * 4u + 2u];
            V[i * 4u + 3u] = BT_d[i * 4u + 1u] - BT_d[i * 4u + 3u];
        }

        // Transform weight: G @ g @ GT (simplified for 3x3)
        var U: array<f32, 16>;
        // G transform rows
        U[0] = g[0];
        U[1] = g[1];
        U[2] = g[2];
        U[3] = 0.0;
        U[4] = 0.5 * (g[0] + g[3] + g[6]);
        U[5] = 0.5 * (g[1] + g[4] + g[7]);
        U[6] = 0.5 * (g[2] + g[5] + g[8]);
        U[7] = 0.0;
        U[8] = 0.5 * (g[0] - g[3] + g[6]);
        U[9] = 0.5 * (g[1] - g[4] + g[7]);
        U[10] = 0.5 * (g[2] - g[5] + g[8]);
        U[11] = 0.0;
        U[12] = g[6];
        U[13] = g[7];
        U[14] = g[8];
        U[15] = 0.0;

        // Element-wise multiply: M = U * V
        var M: array<f32, 16>;
        for (var i = 0u; i < 16u; i++) {
            M[i] = U[i] * V[i];
        }

        // Inverse transform: AT @ M @ A
        var AT_M: array<f32, 8>;
        for (var j = 0u; j < 4u; j++) {
            AT_M[0 * 4u + j] = M[0 * 4u + j] + M[1 * 4u + j] + M[2 * 4u + j];
            AT_M[1 * 4u + j] = M[1 * 4u + j] - M[2 * 4u + j] - M[3 * 4u + j];
        }

        // AT_M @ A to get 2x2 output
        acc[0] += AT_M[0] + AT_M[1] + AT_M[2];
        acc[1] += AT_M[1] - AT_M[2] - AT_M[3];
        acc[2] += AT_M[4] + AT_M[5] + AT_M[6];
        acc[3] += AT_M[5] - AT_M[6] - AT_M[7];
    }

    // Store 2x2 output tile
    let b = bias[out_c];
    for (var i = 0u; i < 2u; i++) {
        for (var j = 0u; j < 2u; j++) {
            let out_row = out_row_base + i;
            let out_col = out_col_base + j;
            if (out_row < out_h && out_col < out_w) {
                let out_idx = idx_nchw(batch_idx, out_c, out_row, out_col,
                                      params.out_channels, out_h, out_w);
                output[out_idx] = acc[i * 2u + j] + b;
            }
        }
    }
}

// ============================================================================
// Depthwise Convolution
// ============================================================================

@compute @workgroup_size(256)
fn steel_conv_depthwise(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Depthwise convolution: each input channel is convolved independently
    // Weight shape: [C, 1, Kh, Kw] (C_out = C_in)

    let batch_idx = wid.z / params.in_channels;
    let channel = wid.z % params.in_channels;

    if (batch_idx >= params.batch_size) { return; }

    let out_h = params.out_height;
    let out_w = params.out_width;
    let in_h = params.in_height;
    let in_w = params.in_width;
    let in_c = params.in_channels;
    let kh = params.kernel_h;
    let kw = params.kernel_w;
    let stride_h = params.stride_h;
    let stride_w = params.stride_w;
    let pad_h = params.pad_h;
    let pad_w = params.pad_w;

    let out_spatial = wid.x * BLOCK_SIZE + lid.x;
    let out_row = out_spatial / out_w;
    let out_col = out_spatial % out_w;

    if (out_row >= out_h || out_col >= out_w) { return; }

    var sum: f32 = 0.0;

    for (var ky = 0u; ky < kh; ky++) {
        for (var kx = 0u; kx < kw; kx++) {
            let in_row = i32(out_row * stride_h + ky) - i32(pad_h);
            let in_col = i32(out_col * stride_w + kx) - i32(pad_w);

            if (in_row >= 0 && in_row < i32(in_h) && in_col >= 0 && in_col < i32(in_w)) {
                let in_idx = idx_nchw(batch_idx, channel, u32(in_row), u32(in_col),
                                     in_c, in_h, in_w);
                let w_idx = (channel * kh + ky) * kw + kx;
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    sum += bias[channel];

    let out_idx = idx_nchw(batch_idx, channel, out_row, out_col, in_c, out_h, out_w);
    output[out_idx] = sum;
}
