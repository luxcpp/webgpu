// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel General/Strided Convolution in WGSL
// Works on WebGPU (Metal/Vulkan/D3D12 via Dawn/wgpu)
//
// Part of the Lux Network GPU acceleration library
// Supports arbitrary strides, dilations, padding, and group convolutions

// ============================================================================
// Configuration Constants
// ============================================================================

const TILE_M: u32 = 32u;       // Output channels tile
const TILE_N: u32 = 32u;       // Spatial output tile
const TILE_K: u32 = 32u;       // Reduction dimension tile
const BLOCK_SIZE: u32 = 256u;
const VECTOR_SIZE: u32 = 4u;   // Vectorized loads

// ============================================================================
// Kernel Bindings
// ============================================================================

struct ConvGeneralParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    in_dims: vec4<u32>,        // [D, H, W, 1] for 3D/2D/1D
    out_dims: vec4<u32>,       // [D_out, H_out, W_out, 1]
    kernel_dims: vec4<u32>,    // [Kd, Kh, Kw, 1]
    stride: vec4<u32>,         // [stride_d, stride_h, stride_w, 1]
    padding: vec4<u32>,        // [pad_d, pad_h, pad_w, 1]
    dilation: vec4<u32>,       // [dil_d, dil_h, dil_w, 1]
    groups: u32,
    ndim: u32,                 // 1, 2, or 3
    transposed: u32,           // 1 for transposed conv
    output_padding: vec4<u32>, // For transposed conv
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: ConvGeneralParams;

// Shared memory
var<workgroup> smem_input: array<f32, 4096>;
var<workgroup> smem_weight: array<f32, 2048>;
var<workgroup> smem_output: array<f32, 1024>;

// ============================================================================
// Index Helpers for N-dimensional Convolution
// ============================================================================

// 3D input index: [N, C, D, H, W]
fn idx_3d(n: u32, c: u32, d: u32, h: u32, w: u32, C: u32, D: u32, H: u32, W: u32) -> u32 {
    return (((n * C + c) * D + d) * H + h) * W + w;
}

// 3D weight index: [C_out, C_in/groups, Kd, Kh, Kw]
fn idx_weight_3d(co: u32, ci: u32, kd: u32, kh: u32, kw: u32,
                 CI: u32, Kd: u32, Kh: u32, Kw: u32) -> u32 {
    return (((co * CI + ci) * Kd + kd) * Kh + kh) * Kw + kw;
}

// 1D convolution (used as base case)
fn conv1d_element(batch: u32, out_c: u32, out_x: u32, in_c_start: u32, in_c_count: u32) -> f32 {
    let in_w = params.in_dims.z;
    let out_w = params.out_dims.z;
    let kw = params.kernel_dims.z;
    let stride_w = params.stride.z;
    let pad_w = params.padding.z;
    let dil_w = params.dilation.z;
    let in_c = params.in_channels;

    var sum: f32 = 0.0;

    for (var ic = 0u; ic < in_c_count; ic++) {
        for (var kx = 0u; kx < kw; kx++) {
            let in_x = i32(out_x * stride_w + kx * dil_w) - i32(pad_w);

            if (in_x >= 0 && in_x < i32(in_w)) {
                let in_idx = (batch * in_c + in_c_start + ic) * in_w + u32(in_x);
                let w_idx = (out_c * in_c_count + ic) * kw + kx;
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    return sum;
}

// ============================================================================
// General N-D Convolution Kernel
// ============================================================================

@compute @workgroup_size(256)
fn steel_conv_general(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z / params.out_channels;
    let out_c = wid.z % params.out_channels;

    if (batch_idx >= params.batch_size) { return; }

    let groups = params.groups;
    let in_c = params.in_channels;
    let out_channels = params.out_channels;
    let channels_per_group_in = in_c / groups;
    let channels_per_group_out = out_channels / groups;
    let group_idx = out_c / channels_per_group_out;
    let in_c_start = group_idx * channels_per_group_in;

    let ndim = params.ndim;

    // Calculate total output elements for this batch/channel
    var out_spatial: u32 = 1u;
    if (ndim >= 1u) { out_spatial *= params.out_dims.z; }  // W
    if (ndim >= 2u) { out_spatial *= params.out_dims.y; }  // H
    if (ndim >= 3u) { out_spatial *= params.out_dims.x; }  // D

    let out_idx_base = wid.x * BLOCK_SIZE + tid;
    if (out_idx_base >= out_spatial) { return; }

    // Decode spatial index
    var out_w_idx = out_idx_base;
    var out_h_idx = 0u;
    var out_d_idx = 0u;

    if (ndim >= 2u) {
        out_h_idx = out_idx_base / params.out_dims.z;
        out_w_idx = out_idx_base % params.out_dims.z;
    }
    if (ndim >= 3u) {
        out_d_idx = out_h_idx / params.out_dims.y;
        out_h_idx = out_h_idx % params.out_dims.y;
    }

    // Compute convolution
    var sum: f32 = 0.0;

    let kd = select(1u, params.kernel_dims.x, ndim >= 3u);
    let kh = select(1u, params.kernel_dims.y, ndim >= 2u);
    let kw = params.kernel_dims.z;

    let stride_d = select(1u, params.stride.x, ndim >= 3u);
    let stride_h = select(1u, params.stride.y, ndim >= 2u);
    let stride_w = params.stride.z;

    let pad_d = select(0u, params.padding.x, ndim >= 3u);
    let pad_h = select(0u, params.padding.y, ndim >= 2u);
    let pad_w = params.padding.z;

    let dil_d = select(1u, params.dilation.x, ndim >= 3u);
    let dil_h = select(1u, params.dilation.y, ndim >= 2u);
    let dil_w = params.dilation.z;

    let in_d = select(1u, params.in_dims.x, ndim >= 3u);
    let in_h = select(1u, params.in_dims.y, ndim >= 2u);
    let in_w = params.in_dims.z;

    for (var ic = 0u; ic < channels_per_group_in; ic++) {
        for (var kd_i = 0u; kd_i < kd; kd_i++) {
            for (var kh_i = 0u; kh_i < kh; kh_i++) {
                for (var kw_i = 0u; kw_i < kw; kw_i++) {
                    let in_d_idx = i32(out_d_idx * stride_d + kd_i * dil_d) - i32(pad_d);
                    let in_h_idx = i32(out_h_idx * stride_h + kh_i * dil_h) - i32(pad_h);
                    let in_w_idx = i32(out_w_idx * stride_w + kw_i * dil_w) - i32(pad_w);

                    var valid = in_w_idx >= 0 && in_w_idx < i32(in_w);
                    if (ndim >= 2u) { valid = valid && in_h_idx >= 0 && in_h_idx < i32(in_h); }
                    if (ndim >= 3u) { valid = valid && in_d_idx >= 0 && in_d_idx < i32(in_d); }

                    if (valid) {
                        let in_idx = idx_3d(batch_idx, in_c_start + ic,
                                           u32(in_d_idx), u32(in_h_idx), u32(in_w_idx),
                                           in_c, in_d, in_h, in_w);
                        let w_idx = idx_weight_3d(out_c, ic, kd_i, kh_i, kw_i,
                                                 channels_per_group_in, kd, kh, kw);
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
    }

    sum += bias[out_c];

    // Store output
    let out_idx = idx_3d(batch_idx, out_c, out_d_idx, out_h_idx, out_w_idx,
                        out_channels, params.out_dims.x, params.out_dims.y, params.out_dims.z);
    output[out_idx] = sum;
}

// ============================================================================
// Transposed Convolution (Deconvolution)
// ============================================================================

@compute @workgroup_size(256)
fn steel_conv_transpose(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Transposed convolution: output_size = (input_size - 1) * stride - 2 * pad + dilation * (kernel - 1) + output_padding + 1

    let tid = lid.x;
    let batch_idx = wid.z / params.out_channels;
    let out_c = wid.z % params.out_channels;

    if (batch_idx >= params.batch_size || params.transposed == 0u) { return; }

    let groups = params.groups;
    let in_c = params.in_channels;
    let out_channels = params.out_channels;
    let channels_per_group_in = in_c / groups;
    let channels_per_group_out = out_channels / groups;
    let group_idx = out_c / channels_per_group_out;
    let in_c_start = group_idx * channels_per_group_in;

    // Output spatial dimensions
    let out_h = params.out_dims.y;
    let out_w = params.out_dims.z;
    let in_h = params.in_dims.y;
    let in_w = params.in_dims.z;
    let kh = params.kernel_dims.y;
    let kw = params.kernel_dims.z;
    let stride_h = params.stride.y;
    let stride_w = params.stride.z;
    let pad_h = params.padding.y;
    let pad_w = params.padding.z;
    let dil_h = params.dilation.y;
    let dil_w = params.dilation.z;

    let out_spatial = out_h * out_w;
    let out_idx_base = wid.x * BLOCK_SIZE + tid;
    if (out_idx_base >= out_spatial) { return; }

    let out_row = out_idx_base / out_w;
    let out_col = out_idx_base % out_w;

    var sum: f32 = 0.0;

    // For transposed conv, we iterate over input positions that contribute to this output
    for (var ic = 0u; ic < channels_per_group_in; ic++) {
        for (var ky = 0u; ky < kh; ky++) {
            for (var kx = 0u; kx < kw; kx++) {
                // Inverse relationship: output_pos = input_pos * stride - pad + kernel_pos * dilation
                // So: input_pos = (output_pos + pad - kernel_pos * dilation) / stride
                let h_offset = i32(out_row) + i32(pad_h) - i32(ky * dil_h);
                let w_offset = i32(out_col) + i32(pad_w) - i32(kx * dil_w);

                // Check if divisible by stride
                if (h_offset % i32(stride_h) == 0 && w_offset % i32(stride_w) == 0) {
                    let in_row = h_offset / i32(stride_h);
                    let in_col = w_offset / i32(stride_w);

                    if (in_row >= 0 && in_row < i32(in_h) && in_col >= 0 && in_col < i32(in_w)) {
                        let in_idx = ((batch_idx * in_c + in_c_start + ic) * in_h + u32(in_row)) * in_w + u32(in_col);
                        // Weight is transposed: [C_in, C_out/groups, Kh, Kw]
                        let w_idx = ((ic * channels_per_group_out + (out_c % channels_per_group_out)) * kh + ky) * kw + kx;
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
    }

    sum += bias[out_c];

    let out_idx = ((batch_idx * out_channels + out_c) * out_h + out_row) * out_w + out_col;
    output[out_idx] = sum;
}

// ============================================================================
// Strided Convolution with Dilation (Atrous Convolution)
// ============================================================================

@compute @workgroup_size(256)
fn steel_conv_atrous(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Atrous (dilated) convolution for semantic segmentation
    // Uses dilation to increase receptive field without increasing parameters

    let tid = lid.x;
    let batch_idx = wid.z / params.out_channels;
    let out_c = wid.z % params.out_channels;

    if (batch_idx >= params.batch_size) { return; }

    let in_c = params.in_channels;
    let out_h = params.out_dims.y;
    let out_w = params.out_dims.z;
    let in_h = params.in_dims.y;
    let in_w = params.in_dims.z;
    let kh = params.kernel_dims.y;
    let kw = params.kernel_dims.z;
    let dil_h = params.dilation.y;
    let dil_w = params.dilation.z;
    let pad_h = params.padding.y;
    let pad_w = params.padding.z;

    // Effective kernel size with dilation
    let eff_kh = dil_h * (kh - 1u) + 1u;
    let eff_kw = dil_w * (kw - 1u) + 1u;

    let out_spatial = out_h * out_w;
    let out_idx_base = wid.x * BLOCK_SIZE + tid;
    if (out_idx_base >= out_spatial) { return; }

    let out_row = out_idx_base / out_w;
    let out_col = out_idx_base % out_w;

    var sum: f32 = 0.0;

    for (var ic = 0u; ic < in_c; ic++) {
        for (var ky = 0u; ky < kh; ky++) {
            for (var kx = 0u; kx < kw; kx++) {
                // Dilated kernel position
                let in_row = i32(out_row + ky * dil_h) - i32(pad_h);
                let in_col = i32(out_col + kx * dil_w) - i32(pad_w);

                if (in_row >= 0 && in_row < i32(in_h) && in_col >= 0 && in_col < i32(in_w)) {
                    let in_idx = ((batch_idx * in_c + ic) * in_h + u32(in_row)) * in_w + u32(in_col);
                    let w_idx = ((out_c * in_c + ic) * kh + ky) * kw + kx;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    sum += bias[out_c];

    let out_idx = ((batch_idx * params.out_channels + out_c) * out_h + out_row) * out_w + out_col;
    output[out_idx] = sum;
}

// ============================================================================
// Grouped Convolution with Channel Shuffle
// ============================================================================

@compute @workgroup_size(256)
fn steel_conv_grouped_shuffle(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // ShuffleNet-style grouped convolution with channel shuffle
    // First performs grouped conv, then shuffles channels between groups

    let tid = lid.x;
    let batch_idx = wid.z;

    if (batch_idx >= params.batch_size) { return; }

    let groups = params.groups;
    let out_channels = params.out_channels;
    let out_h = params.out_dims.y;
    let out_w = params.out_dims.z;
    let channels_per_group = out_channels / groups;

    // After grouped convolution, shuffle channels
    // Reshape: [N, g, c, H, W] -> [N, c, g, H, W]
    // where g = groups, c = channels_per_group

    let total = out_channels * out_h * out_w;
    let idx = wid.x * BLOCK_SIZE + tid;
    if (idx >= total) { return; }

    let c_out = idx / (out_h * out_w);
    let spatial = idx % (out_h * out_w);

    // Original position: channel c_out
    // group = c_out / channels_per_group
    // position_in_group = c_out % channels_per_group

    let group = c_out / channels_per_group;
    let pos_in_group = c_out % channels_per_group;

    // New position after shuffle: pos_in_group * groups + group
    let new_channel = pos_in_group * groups + group;

    let src_idx = (batch_idx * out_channels + c_out) * out_h * out_w + spatial;
    let dst_idx = (batch_idx * out_channels + new_channel) * out_h * out_w + spatial;

    // Load to shared memory
    smem_output[tid] = output[src_idx];
    workgroupBarrier();

    // Store shuffled
    output[dst_idx] = smem_output[tid];
}

// ============================================================================
// Separable Convolution (Depthwise + Pointwise)
// ============================================================================

@compute @workgroup_size(256)
fn steel_conv_separable(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // MobileNet-style separable convolution
    // 1. Depthwise: [N, C, H, W] * [C, 1, Kh, Kw] -> [N, C, H', W']
    // 2. Pointwise: [N, C, H', W'] * [C_out, C, 1, 1] -> [N, C_out, H', W']
    // This kernel performs both in one pass for small channel counts

    let tid = lid.x;
    let batch_idx = wid.z / params.out_channels;
    let out_c = wid.z % params.out_channels;

    if (batch_idx >= params.batch_size) { return; }

    let in_c = params.in_channels;
    let out_h = params.out_dims.y;
    let out_w = params.out_dims.z;
    let in_h = params.in_dims.y;
    let in_w = params.in_dims.z;
    let kh = params.kernel_dims.y;
    let kw = params.kernel_dims.z;
    let stride_h = params.stride.y;
    let stride_w = params.stride.z;
    let pad_h = params.padding.y;
    let pad_w = params.padding.z;

    let out_spatial = out_h * out_w;
    let out_idx_base = wid.x * BLOCK_SIZE + tid;
    if (out_idx_base >= out_spatial) { return; }

    let out_row = out_idx_base / out_w;
    let out_col = out_idx_base % out_w;

    // Depthwise intermediate + pointwise in one pass
    var sum: f32 = 0.0;

    for (var ic = 0u; ic < in_c; ic++) {
        // Depthwise convolution for channel ic
        var dw_sum: f32 = 0.0;

        for (var ky = 0u; ky < kh; ky++) {
            for (var kx = 0u; kx < kw; kx++) {
                let in_row = i32(out_row * stride_h + ky) - i32(pad_h);
                let in_col = i32(out_col * stride_w + kx) - i32(pad_w);

                if (in_row >= 0 && in_row < i32(in_h) && in_col >= 0 && in_col < i32(in_w)) {
                    let in_idx = ((batch_idx * in_c + ic) * in_h + u32(in_row)) * in_w + u32(in_col);
                    // Depthwise weight: [C, 1, Kh, Kw]
                    let dw_idx = (ic * kh + ky) * kw + kx;
                    dw_sum += input[in_idx] * weight[dw_idx];
                }
            }
        }

        // Pointwise weight: [C_out, C, 1, 1] stored after depthwise weights
        let pw_offset = in_c * kh * kw;
        let pw_idx = out_c * in_c + ic;
        sum += dw_sum * weight[pw_offset + pw_idx];
    }

    sum += bias[out_c];

    let out_idx = ((batch_idx * params.out_channels + out_c) * out_h + out_row) * out_w + out_col;
    output[out_idx] = sum;
}
