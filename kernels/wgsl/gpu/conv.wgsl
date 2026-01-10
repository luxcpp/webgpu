// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Convolution Operations Kernels
//
// Supports:
// - 1D Convolution (for sequence models)
// - 2D Convolution (for vision)
// - Depthwise Convolution
// - Pointwise (1x1) Convolution
// - Transposed Convolution
//
// Part of the Lux Network GPU acceleration library

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;
const TILE_SIZE: u32 = 16u;

// ============================================================================
// Parameter Structures
// ============================================================================

// 1D Convolution parameters
struct Conv1DParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    seq_len: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    dilation: u32,
    groups: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// 2D Convolution parameters
struct Conv2DParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    in_height: u32,
    in_width: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    padding_h: u32,
    padding_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    groups: u32,
    _pad0: u32,
    _pad1: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

// Input: [batch, channels, height, width] or [batch, channels, seq_len]
@group(0) @binding(0) var<storage, read> input: array<f32>;
// Weight: [out_channels, in_channels/groups, kernel_h, kernel_w]
@group(0) @binding(1) var<storage, read> weight: array<f32>;
// Bias: [out_channels] (optional)
@group(0) @binding(2) var<storage, read> bias_buf: array<f32>;
// Output: [batch, out_channels, out_height, out_width]
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params_1d: Conv1DParams;

@group(1) @binding(0) var<uniform> params_2d: Conv2DParams;

// Shared memory for tiling
var<workgroup> shared_input: array<f32, 1024>;
var<workgroup> shared_weight: array<f32, 1024>;

// ============================================================================
// 1D Convolution (for sequences, audio, NLP)
// ============================================================================

fn get_output_len_1d(in_len: u32, kernel: u32, stride: u32, padding: u32, dilation: u32) -> u32 {
    let effective_kernel = (kernel - 1u) * dilation + 1u;
    return (in_len + 2u * padding - effective_kernel) / stride + 1u;
}

@compute @workgroup_size(256)
fn conv1d(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_size = params_1d.batch_size;
    let in_channels = params_1d.in_channels;
    let out_channels = params_1d.out_channels;
    let seq_len = params_1d.seq_len;
    let kernel_size = params_1d.kernel_size;
    let stride = params_1d.stride;
    let padding = params_1d.padding;
    let dilation = params_1d.dilation;
    let groups = params_1d.groups;

    let out_len = get_output_len_1d(seq_len, kernel_size, stride, padding, dilation);
    let total = batch_size * out_channels * out_len;

    if (gid.x >= total) {
        return;
    }

    // Decode indices
    let out_pos = gid.x % out_len;
    let temp = gid.x / out_len;
    let out_ch = temp % out_channels;
    let batch = temp / out_channels;

    // Group convolution
    let group_out_channels = out_channels / groups;
    let group_in_channels = in_channels / groups;
    let group_idx = out_ch / group_out_channels;
    let out_ch_in_group = out_ch % group_out_channels;

    var sum: f32 = 0.0;

    // Convolution
    for (var in_ch = 0u; in_ch < group_in_channels; in_ch++) {
        let actual_in_ch = group_idx * group_in_channels + in_ch;

        for (var k = 0u; k < kernel_size; k++) {
            let in_pos_raw = i32(out_pos * stride) - i32(padding) + i32(k * dilation);

            if (in_pos_raw >= 0 && u32(in_pos_raw) < seq_len) {
                let in_pos = u32(in_pos_raw);
                let in_idx = (batch * in_channels + actual_in_ch) * seq_len + in_pos;
                let w_idx = (out_ch * group_in_channels + in_ch) * kernel_size + k;
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    // Add bias
    sum += bias_buf[out_ch];

    let out_idx = (batch * out_channels + out_ch) * out_len + out_pos;
    output[out_idx] = sum;
}

// ============================================================================
// 2D Convolution (for images)
// ============================================================================

fn get_output_size_2d(in_size: u32, kernel: u32, stride: u32, padding: u32, dilation: u32) -> u32 {
    let effective_kernel = (kernel - 1u) * dilation + 1u;
    return (in_size + 2u * padding - effective_kernel) / stride + 1u;
}

@compute @workgroup_size(256)
fn conv2d(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_size = params_2d.batch_size;
    let in_channels = params_2d.in_channels;
    let out_channels = params_2d.out_channels;
    let in_height = params_2d.in_height;
    let in_width = params_2d.in_width;
    let kernel_h = params_2d.kernel_h;
    let kernel_w = params_2d.kernel_w;
    let stride_h = params_2d.stride_h;
    let stride_w = params_2d.stride_w;
    let padding_h = params_2d.padding_h;
    let padding_w = params_2d.padding_w;
    let dilation_h = params_2d.dilation_h;
    let dilation_w = params_2d.dilation_w;
    let groups = params_2d.groups;

    let out_height = get_output_size_2d(in_height, kernel_h, stride_h, padding_h, dilation_h);
    let out_width = get_output_size_2d(in_width, kernel_w, stride_w, padding_w, dilation_w);
    let total = batch_size * out_channels * out_height * out_width;

    if (gid.x >= total) {
        return;
    }

    // Decode indices
    let out_w = gid.x % out_width;
    let temp1 = gid.x / out_width;
    let out_h = temp1 % out_height;
    let temp2 = temp1 / out_height;
    let out_ch = temp2 % out_channels;
    let batch = temp2 / out_channels;

    // Group convolution
    let group_out_channels = out_channels / groups;
    let group_in_channels = in_channels / groups;
    let group_idx = out_ch / group_out_channels;

    var sum: f32 = 0.0;

    // Convolution
    for (var in_ch = 0u; in_ch < group_in_channels; in_ch++) {
        let actual_in_ch = group_idx * group_in_channels + in_ch;

        for (var kh = 0u; kh < kernel_h; kh++) {
            for (var kw = 0u; kw < kernel_w; kw++) {
                let in_h_raw = i32(out_h * stride_h) - i32(padding_h) + i32(kh * dilation_h);
                let in_w_raw = i32(out_w * stride_w) - i32(padding_w) + i32(kw * dilation_w);

                if (in_h_raw >= 0 && u32(in_h_raw) < in_height &&
                    in_w_raw >= 0 && u32(in_w_raw) < in_width) {

                    let in_h = u32(in_h_raw);
                    let in_w = u32(in_w_raw);

                    let in_idx = ((batch * in_channels + actual_in_ch) * in_height + in_h) * in_width + in_w;
                    let w_idx = ((out_ch * group_in_channels + in_ch) * kernel_h + kh) * kernel_w + kw;

                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    // Add bias
    sum += bias_buf[out_ch];

    let out_idx = ((batch * out_channels + out_ch) * out_height + out_h) * out_width + out_w;
    output[out_idx] = sum;
}

// ============================================================================
// Depthwise Convolution (groups = in_channels = out_channels)
// ============================================================================

@compute @workgroup_size(256)
fn conv2d_depthwise(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_size = params_2d.batch_size;
    let channels = params_2d.in_channels;  // = out_channels for depthwise
    let in_height = params_2d.in_height;
    let in_width = params_2d.in_width;
    let kernel_h = params_2d.kernel_h;
    let kernel_w = params_2d.kernel_w;
    let stride_h = params_2d.stride_h;
    let stride_w = params_2d.stride_w;
    let padding_h = params_2d.padding_h;
    let padding_w = params_2d.padding_w;

    let out_height = get_output_size_2d(in_height, kernel_h, stride_h, padding_h, 1u);
    let out_width = get_output_size_2d(in_width, kernel_w, stride_w, padding_w, 1u);
    let total = batch_size * channels * out_height * out_width;

    if (gid.x >= total) {
        return;
    }

    let out_w = gid.x % out_width;
    let temp1 = gid.x / out_width;
    let out_h = temp1 % out_height;
    let temp2 = temp1 / out_height;
    let channel = temp2 % channels;
    let batch = temp2 / channels;

    var sum: f32 = 0.0;

    for (var kh = 0u; kh < kernel_h; kh++) {
        for (var kw = 0u; kw < kernel_w; kw++) {
            let in_h_raw = i32(out_h * stride_h) - i32(padding_h) + i32(kh);
            let in_w_raw = i32(out_w * stride_w) - i32(padding_w) + i32(kw);

            if (in_h_raw >= 0 && u32(in_h_raw) < in_height &&
                in_w_raw >= 0 && u32(in_w_raw) < in_width) {

                let in_idx = ((batch * channels + channel) * in_height + u32(in_h_raw)) * in_width + u32(in_w_raw);
                let w_idx = (channel * kernel_h + kh) * kernel_w + kw;

                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    sum += bias_buf[channel];

    let out_idx = ((batch * channels + channel) * out_height + out_h) * out_width + out_w;
    output[out_idx] = sum;
}

// ============================================================================
// Pointwise (1x1) Convolution
// ============================================================================

@compute @workgroup_size(256)
fn conv2d_pointwise(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_size = params_2d.batch_size;
    let in_channels = params_2d.in_channels;
    let out_channels = params_2d.out_channels;
    let height = params_2d.in_height;
    let width = params_2d.in_width;

    let total = batch_size * out_channels * height * width;

    if (gid.x >= total) {
        return;
    }

    let w = gid.x % width;
    let temp1 = gid.x / width;
    let h = temp1 % height;
    let temp2 = temp1 / height;
    let out_ch = temp2 % out_channels;
    let batch = temp2 / out_channels;

    var sum: f32 = 0.0;

    for (var in_ch = 0u; in_ch < in_channels; in_ch++) {
        let in_idx = ((batch * in_channels + in_ch) * height + h) * width + w;
        let w_idx = out_ch * in_channels + in_ch;
        sum += input[in_idx] * weight[w_idx];
    }

    sum += bias_buf[out_ch];

    let out_idx = ((batch * out_channels + out_ch) * height + h) * width + w;
    output[out_idx] = sum;
}

// ============================================================================
// Transposed Convolution (Deconvolution)
// ============================================================================

struct ConvTranspose2DParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    in_height: u32,
    in_width: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    padding_h: u32,
    padding_w: u32,
    output_padding_h: u32,
    output_padding_w: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(2) @binding(0) var<uniform> trans_params: ConvTranspose2DParams;

@compute @workgroup_size(256)
fn conv_transpose2d(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_size = trans_params.batch_size;
    let in_channels = trans_params.in_channels;
    let out_channels = trans_params.out_channels;
    let in_height = trans_params.in_height;
    let in_width = trans_params.in_width;
    let kernel_h = trans_params.kernel_h;
    let kernel_w = trans_params.kernel_w;
    let stride_h = trans_params.stride_h;
    let stride_w = trans_params.stride_w;
    let padding_h = trans_params.padding_h;
    let padding_w = trans_params.padding_w;
    let output_padding_h = trans_params.output_padding_h;
    let output_padding_w = trans_params.output_padding_w;

    let out_height = (in_height - 1u) * stride_h - 2u * padding_h + kernel_h + output_padding_h;
    let out_width = (in_width - 1u) * stride_w - 2u * padding_w + kernel_w + output_padding_w;
    let total = batch_size * out_channels * out_height * out_width;

    if (gid.x >= total) {
        return;
    }

    let out_w = gid.x % out_width;
    let temp1 = gid.x / out_width;
    let out_h = temp1 % out_height;
    let temp2 = temp1 / out_height;
    let out_ch = temp2 % out_channels;
    let batch = temp2 / out_channels;

    var sum: f32 = 0.0;

    for (var in_ch = 0u; in_ch < in_channels; in_ch++) {
        for (var kh = 0u; kh < kernel_h; kh++) {
            for (var kw = 0u; kw < kernel_w; kw++) {
                // Transposed conv: map output position to input
                let in_h_raw = i32(out_h + padding_h) - i32(kh);
                let in_w_raw = i32(out_w + padding_w) - i32(kw);

                // Check if divisible by stride
                if (in_h_raw % i32(stride_h) == 0 && in_w_raw % i32(stride_w) == 0) {
                    let in_h = in_h_raw / i32(stride_h);
                    let in_w = in_w_raw / i32(stride_w);

                    if (in_h >= 0 && u32(in_h) < in_height &&
                        in_w >= 0 && u32(in_w) < in_width) {

                        let in_idx = ((batch * in_channels + in_ch) * in_height + u32(in_h)) * in_width + u32(in_w);
                        // Weight is [in_ch, out_ch, kh, kw] for transposed
                        let w_idx = ((in_ch * out_channels + out_ch) * kernel_h + kh) * kernel_w + kw;

                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
    }

    sum += bias_buf[out_ch];

    let out_idx = ((batch * out_channels + out_ch) * out_height + out_h) * out_width + out_w;
    output[out_idx] = sum;
}

// ============================================================================
// Im2Col (for efficient convolution via GEMM)
// ============================================================================

struct Im2ColParams {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    padding_h: u32,
    padding_w: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(3) @binding(0) var<uniform> im2col_params: Im2ColParams;
@group(3) @binding(1) var<storage, read_write> col_output: array<f32>;

@compute @workgroup_size(256)
fn im2col(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_size = im2col_params.batch_size;
    let channels = im2col_params.channels;
    let height = im2col_params.height;
    let width = im2col_params.width;
    let kernel_h = im2col_params.kernel_h;
    let kernel_w = im2col_params.kernel_w;
    let stride_h = im2col_params.stride_h;
    let stride_w = im2col_params.stride_w;
    let padding_h = im2col_params.padding_h;
    let padding_w = im2col_params.padding_w;

    let out_height = (height + 2u * padding_h - kernel_h) / stride_h + 1u;
    let out_width = (width + 2u * padding_w - kernel_w) / stride_w + 1u;
    let col_height = channels * kernel_h * kernel_w;
    let col_width = out_height * out_width;

    let total = batch_size * col_height * col_width;

    if (gid.x >= total) {
        return;
    }

    // Decode indices
    let col_w = gid.x % col_width;
    let temp = gid.x / col_width;
    let col_h = temp % col_height;
    let batch = temp / col_height;

    // Map col indices to image indices
    let out_h = col_w / out_width;
    let out_w = col_w % out_width;
    let k_idx = col_h;
    let channel = k_idx / (kernel_h * kernel_w);
    let k_offset = k_idx % (kernel_h * kernel_w);
    let kh = k_offset / kernel_w;
    let kw = k_offset % kernel_w;

    let in_h_raw = i32(out_h * stride_h) - i32(padding_h) + i32(kh);
    let in_w_raw = i32(out_w * stride_w) - i32(padding_w) + i32(kw);

    var val: f32 = 0.0;
    if (in_h_raw >= 0 && u32(in_h_raw) < height &&
        in_w_raw >= 0 && u32(in_w_raw) < width) {
        let in_idx = ((batch * channels + channel) * height + u32(in_h_raw)) * width + u32(in_w_raw);
        val = input[in_idx];
    }

    let col_idx = (batch * col_height + col_h) * col_width + col_w;
    col_output[col_idx] = val;
}

// ============================================================================
// Col2Im (reverse of Im2Col)
// ============================================================================

@compute @workgroup_size(256)
fn col2im(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_size = im2col_params.batch_size;
    let channels = im2col_params.channels;
    let height = im2col_params.height;
    let width = im2col_params.width;
    let kernel_h = im2col_params.kernel_h;
    let kernel_w = im2col_params.kernel_w;
    let stride_h = im2col_params.stride_h;
    let stride_w = im2col_params.stride_w;
    let padding_h = im2col_params.padding_h;
    let padding_w = im2col_params.padding_w;

    let out_height = (height + 2u * padding_h - kernel_h) / stride_h + 1u;
    let out_width = (width + 2u * padding_w - kernel_w) / stride_w + 1u;

    let total = batch_size * channels * height * width;

    if (gid.x >= total) {
        return;
    }

    // Decode output image indices
    let w = gid.x % width;
    let temp1 = gid.x / width;
    let h = temp1 % height;
    let temp2 = temp1 / height;
    let c = temp2 % channels;
    let batch = temp2 / channels;

    var sum: f32 = 0.0;

    // Iterate over all column positions that contributed to this image position
    for (var kh = 0u; kh < kernel_h; kh++) {
        for (var kw = 0u; kw < kernel_w; kw++) {
            let out_h_raw = i32(h + padding_h) - i32(kh);
            let out_w_raw = i32(w + padding_w) - i32(kw);

            if (out_h_raw % i32(stride_h) == 0 && out_w_raw % i32(stride_w) == 0) {
                let out_h = out_h_raw / i32(stride_h);
                let out_w = out_w_raw / i32(stride_w);

                if (out_h >= 0 && u32(out_h) < out_height &&
                    out_w >= 0 && u32(out_w) < out_width) {

                    let col_h = (c * kernel_h + kh) * kernel_w + kw;
                    let col_w = u32(out_h) * out_width + u32(out_w);
                    let col_idx = (batch * channels * kernel_h * kernel_w + col_h) * out_height * out_width + col_w;

                    sum += col_output[col_idx];
                }
            }
        }
    }

    output[gid.x] = sum;
}
