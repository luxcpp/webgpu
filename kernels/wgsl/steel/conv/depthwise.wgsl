// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Depthwise Convolution in WGSL
// Efficient depthwise and separable convolution for MobileNet-style architectures
// Works on WebGPU (Metal/Vulkan/D3D12 via Dawn/wgpu)

// ============================================================================
// Configuration Constants
// ============================================================================

const TILE_H: u32 = 4u;
const TILE_W: u32 = 4u;
const BLOCK_SIZE: u32 = 256u;

// ============================================================================
// Bindings
// ============================================================================

struct DepthwiseConvParams {
    batch_size: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    depth_multiplier: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

// NCHW layout by default
// Input: [N, C, H_in, W_in]
// Weight: [C * depth_multiplier, 1, K_h, K_w] or [C, depth_multiplier, K_h, K_w]
// Output: [N, C * depth_multiplier, H_out, W_out]

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: DepthwiseConvParams;

// Shared memory
var<workgroup> input_tile: array<f32, 1024>;
var<workgroup> weight_tile: array<f32, 256>;  // Up to 16x16 kernel

// ============================================================================
// Helper Functions
// ============================================================================

fn compute_output_size(in_size: u32, kernel: u32, pad: u32, stride: u32, dilation: u32) -> u32 {
    return (in_size + 2u * pad - dilation * (kernel - 1u) - 1u) / stride + 1u;
}

fn idx_nchw(n: u32, c: u32, h: u32, w: u32, C: u32, H: u32, W: u32) -> u32 {
    return ((n * C + c) * H + h) * W + w;
}

fn idx_nhwc(n: u32, h: u32, w: u32, c: u32, H: u32, W: u32, C: u32) -> u32 {
    return ((n * H + h) * W + w) * C + c;
}

// ============================================================================
// Depthwise Conv2d NCHW (channels first)
// ============================================================================

@compute @workgroup_size(256)
fn depthwise_conv2d_nchw(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_channel_idx = wid.z;
    let batch_idx = batch_channel_idx / params.channels;
    let channel_idx = batch_channel_idx % params.channels;
    let spatial_idx = wid.x * BLOCK_SIZE + tid;

    if (batch_idx >= params.batch_size) { return; }

    let in_height = params.in_height;
    let in_width = params.in_width;
    let kernel_h = params.kernel_h;
    let kernel_w = params.kernel_w;
    let stride_h = params.stride_h;
    let stride_w = params.stride_w;
    let pad_h = params.pad_h;
    let pad_w = params.pad_w;
    let dilation_h = params.dilation_h;
    let dilation_w = params.dilation_w;
    let channels = params.channels;

    let out_height = compute_output_size(in_height, kernel_h, pad_h, stride_h, dilation_h);
    let out_width = compute_output_size(in_width, kernel_w, pad_w, stride_w, dilation_w);

    if (spatial_idx >= out_height * out_width) { return; }

    let h_out = spatial_idx / out_width;
    let w_out = spatial_idx % out_width;

    // Load weight for this channel
    let weight_size = kernel_h * kernel_w;
    for (var i = tid; i < weight_size; i += BLOCK_SIZE) {
        let kh = i / kernel_w;
        let kw = i % kernel_w;
        // Weight layout: [C, 1, K_h, K_w]
        let w_idx = (channel_idx * kernel_h + kh) * kernel_w + kw;
        weight_tile[i] = weight[w_idx];
    }
    workgroupBarrier();

    // Compute depthwise convolution
    var sum: f32 = 0.0;

    for (var kh = 0u; kh < kernel_h; kh++) {
        for (var kw = 0u; kw < kernel_w; kw++) {
            let h_in = i32(h_out * stride_h + kh * dilation_h) - i32(pad_h);
            let w_in = i32(w_out * stride_w + kw * dilation_w) - i32(pad_w);

            if (h_in >= 0 && h_in < i32(in_height) &&
                w_in >= 0 && w_in < i32(in_width)) {

                let in_idx = idx_nchw(batch_idx, channel_idx, u32(h_in), u32(w_in),
                                      channels, in_height, in_width);
                let w_tile_idx = kh * kernel_w + kw;
                sum += input[in_idx] * weight_tile[w_tile_idx];
            }
        }
    }

    sum += bias[channel_idx];
    let out_idx = idx_nchw(batch_idx, channel_idx, h_out, w_out,
                           channels, out_height, out_width);
    output[out_idx] = sum;
}

// ============================================================================
// Depthwise Conv2d NHWC (channels last)
// ============================================================================

@compute @workgroup_size(256)
fn depthwise_conv2d_nhwc(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z;
    let spatial_idx = wid.x * BLOCK_SIZE + tid;

    if (batch_idx >= params.batch_size) { return; }

    let in_height = params.in_height;
    let in_width = params.in_width;
    let kernel_h = params.kernel_h;
    let kernel_w = params.kernel_w;
    let stride_h = params.stride_h;
    let stride_w = params.stride_w;
    let pad_h = params.pad_h;
    let pad_w = params.pad_w;
    let dilation_h = params.dilation_h;
    let dilation_w = params.dilation_w;
    let channels = params.channels;

    let out_height = compute_output_size(in_height, kernel_h, pad_h, stride_h, dilation_h);
    let out_width = compute_output_size(in_width, kernel_w, pad_w, stride_w, dilation_w);

    // Each thread handles one spatial position, all channels
    let spatial_size = out_height * out_width;
    if (spatial_idx >= spatial_size) { return; }

    let h_out = spatial_idx / out_width;
    let w_out = spatial_idx % out_width;

    // Process all channels
    for (var c = 0u; c < channels; c++) {
        var sum: f32 = 0.0;

        for (var kh = 0u; kh < kernel_h; kh++) {
            for (var kw = 0u; kw < kernel_w; kw++) {
                let h_in = i32(h_out * stride_h + kh * dilation_h) - i32(pad_h);
                let w_in = i32(w_out * stride_w + kw * dilation_w) - i32(pad_w);

                if (h_in >= 0 && h_in < i32(in_height) &&
                    w_in >= 0 && w_in < i32(in_width)) {

                    let in_idx = idx_nhwc(batch_idx, u32(h_in), u32(w_in), c,
                                          in_height, in_width, channels);
                    // Weight layout: [K_h, K_w, C, 1]
                    let w_idx = (kh * kernel_w + kw) * channels + c;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }

        sum += bias[c];
        let out_idx = idx_nhwc(batch_idx, h_out, w_out, c,
                               out_height, out_width, channels);
        output[out_idx] = sum;
    }
}

// ============================================================================
// Depthwise Conv2d with Depth Multiplier (NCHW)
// Each input channel produces depth_multiplier output channels
// ============================================================================

@compute @workgroup_size(256)
fn depthwise_conv2d_multiplier_nchw(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_oc_idx = wid.z;
    let out_channels = params.channels * params.depth_multiplier;
    let batch_idx = batch_oc_idx / out_channels;
    let oc = batch_oc_idx % out_channels;
    let spatial_idx = wid.x * BLOCK_SIZE + tid;

    if (batch_idx >= params.batch_size) { return; }

    let in_height = params.in_height;
    let in_width = params.in_width;
    let kernel_h = params.kernel_h;
    let kernel_w = params.kernel_w;
    let stride_h = params.stride_h;
    let stride_w = params.stride_w;
    let pad_h = params.pad_h;
    let pad_w = params.pad_w;
    let dilation_h = params.dilation_h;
    let dilation_w = params.dilation_w;
    let channels = params.channels;
    let depth_multiplier = params.depth_multiplier;

    let out_height = compute_output_size(in_height, kernel_h, pad_h, stride_h, dilation_h);
    let out_width = compute_output_size(in_width, kernel_w, pad_w, stride_w, dilation_w);

    if (spatial_idx >= out_height * out_width) { return; }

    let h_out = spatial_idx / out_width;
    let w_out = spatial_idx % out_width;

    // Map output channel to input channel
    let ic = oc / depth_multiplier;
    let dm = oc % depth_multiplier;

    var sum: f32 = 0.0;

    for (var kh = 0u; kh < kernel_h; kh++) {
        for (var kw = 0u; kw < kernel_w; kw++) {
            let h_in = i32(h_out * stride_h + kh * dilation_h) - i32(pad_h);
            let w_in = i32(w_out * stride_w + kw * dilation_w) - i32(pad_w);

            if (h_in >= 0 && h_in < i32(in_height) &&
                w_in >= 0 && w_in < i32(in_width)) {

                let in_idx = idx_nchw(batch_idx, ic, u32(h_in), u32(w_in),
                                      channels, in_height, in_width);
                // Weight layout: [C, depth_multiplier, K_h, K_w]
                let w_idx = ((ic * depth_multiplier + dm) * kernel_h + kh) * kernel_w + kw;
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    sum += bias[oc];
    let out_idx = idx_nchw(batch_idx, oc, h_out, w_out,
                           out_channels, out_height, out_width);
    output[out_idx] = sum;
}

// ============================================================================
// Depthwise Conv2d 3x3 Optimized (NCHW)
// ============================================================================

@compute @workgroup_size(256)
fn depthwise_conv2d_3x3_nchw(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_channel_idx = wid.z;
    let batch_idx = batch_channel_idx / params.channels;
    let channel_idx = batch_channel_idx % params.channels;
    let spatial_idx = wid.x * BLOCK_SIZE + tid;

    if (batch_idx >= params.batch_size) { return; }

    let in_height = params.in_height;
    let in_width = params.in_width;
    let stride_h = params.stride_h;
    let stride_w = params.stride_w;
    let pad_h = params.pad_h;
    let pad_w = params.pad_w;
    let channels = params.channels;

    let out_height = compute_output_size(in_height, 3u, pad_h, stride_h, 1u);
    let out_width = compute_output_size(in_width, 3u, pad_w, stride_w, 1u);

    if (spatial_idx >= out_height * out_width) { return; }

    let h_out = spatial_idx / out_width;
    let w_out = spatial_idx % out_width;

    // Load 3x3 weights for this channel
    var w: array<f32, 9>;
    let w_base = channel_idx * 9u;
    for (var i = 0u; i < 9u; i++) {
        w[i] = weight[w_base + i];
    }

    // Compute with unrolled 3x3 kernel
    var sum: f32 = 0.0;

    let h_base = i32(h_out * stride_h) - i32(pad_h);
    let w_base_in = i32(w_out * stride_w) - i32(pad_w);

    // Row 0
    if (h_base >= 0 && h_base < i32(in_height)) {
        if (w_base_in >= 0 && w_base_in < i32(in_width)) {
            sum += input[idx_nchw(batch_idx, channel_idx, u32(h_base), u32(w_base_in), channels, in_height, in_width)] * w[0];
        }
        if (w_base_in + 1 >= 0 && w_base_in + 1 < i32(in_width)) {
            sum += input[idx_nchw(batch_idx, channel_idx, u32(h_base), u32(w_base_in + 1), channels, in_height, in_width)] * w[1];
        }
        if (w_base_in + 2 >= 0 && w_base_in + 2 < i32(in_width)) {
            sum += input[idx_nchw(batch_idx, channel_idx, u32(h_base), u32(w_base_in + 2), channels, in_height, in_width)] * w[2];
        }
    }

    // Row 1
    if (h_base + 1 >= 0 && h_base + 1 < i32(in_height)) {
        if (w_base_in >= 0 && w_base_in < i32(in_width)) {
            sum += input[idx_nchw(batch_idx, channel_idx, u32(h_base + 1), u32(w_base_in), channels, in_height, in_width)] * w[3];
        }
        if (w_base_in + 1 >= 0 && w_base_in + 1 < i32(in_width)) {
            sum += input[idx_nchw(batch_idx, channel_idx, u32(h_base + 1), u32(w_base_in + 1), channels, in_height, in_width)] * w[4];
        }
        if (w_base_in + 2 >= 0 && w_base_in + 2 < i32(in_width)) {
            sum += input[idx_nchw(batch_idx, channel_idx, u32(h_base + 1), u32(w_base_in + 2), channels, in_height, in_width)] * w[5];
        }
    }

    // Row 2
    if (h_base + 2 >= 0 && h_base + 2 < i32(in_height)) {
        if (w_base_in >= 0 && w_base_in < i32(in_width)) {
            sum += input[idx_nchw(batch_idx, channel_idx, u32(h_base + 2), u32(w_base_in), channels, in_height, in_width)] * w[6];
        }
        if (w_base_in + 1 >= 0 && w_base_in + 1 < i32(in_width)) {
            sum += input[idx_nchw(batch_idx, channel_idx, u32(h_base + 2), u32(w_base_in + 1), channels, in_height, in_width)] * w[7];
        }
        if (w_base_in + 2 >= 0 && w_base_in + 2 < i32(in_width)) {
            sum += input[idx_nchw(batch_idx, channel_idx, u32(h_base + 2), u32(w_base_in + 2), channels, in_height, in_width)] * w[8];
        }
    }

    sum += bias[channel_idx];
    let out_idx = idx_nchw(batch_idx, channel_idx, h_out, w_out,
                           channels, out_height, out_width);
    output[out_idx] = sum;
}

// ============================================================================
// Depthwise Conv2d with Fused ReLU (NCHW)
// ============================================================================

@compute @workgroup_size(256)
fn depthwise_conv2d_relu_nchw(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_channel_idx = wid.z;
    let batch_idx = batch_channel_idx / params.channels;
    let channel_idx = batch_channel_idx % params.channels;
    let spatial_idx = wid.x * BLOCK_SIZE + tid;

    if (batch_idx >= params.batch_size) { return; }

    let in_height = params.in_height;
    let in_width = params.in_width;
    let kernel_h = params.kernel_h;
    let kernel_w = params.kernel_w;
    let stride_h = params.stride_h;
    let stride_w = params.stride_w;
    let pad_h = params.pad_h;
    let pad_w = params.pad_w;
    let dilation_h = params.dilation_h;
    let dilation_w = params.dilation_w;
    let channels = params.channels;

    let out_height = compute_output_size(in_height, kernel_h, pad_h, stride_h, dilation_h);
    let out_width = compute_output_size(in_width, kernel_w, pad_w, stride_w, dilation_w);

    if (spatial_idx >= out_height * out_width) { return; }

    let h_out = spatial_idx / out_width;
    let w_out = spatial_idx % out_width;

    var sum: f32 = 0.0;

    for (var kh = 0u; kh < kernel_h; kh++) {
        for (var kw = 0u; kw < kernel_w; kw++) {
            let h_in = i32(h_out * stride_h + kh * dilation_h) - i32(pad_h);
            let w_in = i32(w_out * stride_w + kw * dilation_w) - i32(pad_w);

            if (h_in >= 0 && h_in < i32(in_height) &&
                w_in >= 0 && w_in < i32(in_width)) {

                let in_idx = idx_nchw(batch_idx, channel_idx, u32(h_in), u32(w_in),
                                      channels, in_height, in_width);
                let w_idx = (channel_idx * kernel_h + kh) * kernel_w + kw;
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    sum += bias[channel_idx];
    sum = max(sum, 0.0);  // ReLU

    let out_idx = idx_nchw(batch_idx, channel_idx, h_out, w_out,
                           channels, out_height, out_width);
    output[out_idx] = sum;
}

// ============================================================================
// Depthwise Conv2d with Fused ReLU6 (NCHW)
// ============================================================================

@compute @workgroup_size(256)
fn depthwise_conv2d_relu6_nchw(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_channel_idx = wid.z;
    let batch_idx = batch_channel_idx / params.channels;
    let channel_idx = batch_channel_idx % params.channels;
    let spatial_idx = wid.x * BLOCK_SIZE + tid;

    if (batch_idx >= params.batch_size) { return; }

    let in_height = params.in_height;
    let in_width = params.in_width;
    let kernel_h = params.kernel_h;
    let kernel_w = params.kernel_w;
    let stride_h = params.stride_h;
    let stride_w = params.stride_w;
    let pad_h = params.pad_h;
    let pad_w = params.pad_w;
    let dilation_h = params.dilation_h;
    let dilation_w = params.dilation_w;
    let channels = params.channels;

    let out_height = compute_output_size(in_height, kernel_h, pad_h, stride_h, dilation_h);
    let out_width = compute_output_size(in_width, kernel_w, pad_w, stride_w, dilation_w);

    if (spatial_idx >= out_height * out_width) { return; }

    let h_out = spatial_idx / out_width;
    let w_out = spatial_idx % out_width;

    var sum: f32 = 0.0;

    for (var kh = 0u; kh < kernel_h; kh++) {
        for (var kw = 0u; kw < kernel_w; kw++) {
            let h_in = i32(h_out * stride_h + kh * dilation_h) - i32(pad_h);
            let w_in = i32(w_out * stride_w + kw * dilation_w) - i32(pad_w);

            if (h_in >= 0 && h_in < i32(in_height) &&
                w_in >= 0 && w_in < i32(in_width)) {

                let in_idx = idx_nchw(batch_idx, channel_idx, u32(h_in), u32(w_in),
                                      channels, in_height, in_width);
                let w_idx = (channel_idx * kernel_h + kh) * kernel_w + kw;
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    sum += bias[channel_idx];
    sum = clamp(sum, 0.0, 6.0);  // ReLU6

    let out_idx = idx_nchw(batch_idx, channel_idx, h_out, w_out,
                           channels, out_height, out_width);
    output[out_idx] = sum;
}

// ============================================================================
// Separable Conv2d (Depthwise + Pointwise) NCHW
// Combines depthwise and pointwise in single kernel for fusion
// ============================================================================

struct SeparableConvParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    in_height: u32,
    in_width: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    _pad1: u32,
}

@group(0) @binding(5) var<storage, read> pointwise_weight: array<f32>;  // [C_out, C_in]
@group(0) @binding(6) var<storage, read> pointwise_bias: array<f32>;    // [C_out]
@group(0) @binding(7) var<uniform> sep_params: SeparableConvParams;

var<workgroup> depthwise_out: array<f32, 1024>;  // Intermediate depthwise output

@compute @workgroup_size(256)
fn separable_conv2d_nchw(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z;
    let oc_tile = wid.y;
    let spatial_idx = wid.x;

    if (batch_idx >= sep_params.batch_size) { return; }

    let in_height = sep_params.in_height;
    let in_width = sep_params.in_width;
    let kernel_h = sep_params.kernel_h;
    let kernel_w = sep_params.kernel_w;
    let stride_h = sep_params.stride_h;
    let stride_w = sep_params.stride_w;
    let pad_h = sep_params.pad_h;
    let pad_w = sep_params.pad_w;
    let in_channels = sep_params.in_channels;
    let out_channels = sep_params.out_channels;

    let out_height = compute_output_size(in_height, kernel_h, pad_h, stride_h, 1u);
    let out_width = compute_output_size(in_width, kernel_w, pad_w, stride_w, 1u);

    let h_out = spatial_idx / out_width;
    let w_out = spatial_idx % out_width;

    if (h_out >= out_height || w_out >= out_width) { return; }

    // Step 1: Depthwise convolution for all input channels
    for (var c = tid; c < in_channels; c += BLOCK_SIZE) {
        var sum: f32 = 0.0;

        for (var kh = 0u; kh < kernel_h; kh++) {
            for (var kw = 0u; kw < kernel_w; kw++) {
                let h_in = i32(h_out * stride_h + kh) - i32(pad_h);
                let w_in = i32(w_out * stride_w + kw) - i32(pad_w);

                if (h_in >= 0 && h_in < i32(in_height) &&
                    w_in >= 0 && w_in < i32(in_width)) {

                    let in_idx = idx_nchw(batch_idx, c, u32(h_in), u32(w_in),
                                          in_channels, in_height, in_width);
                    let w_idx = (c * kernel_h + kh) * kernel_w + kw;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }

        sum += bias[c];
        depthwise_out[c] = sum;
    }
    workgroupBarrier();

    // Step 2: Pointwise convolution (1x1)
    let oc_start = oc_tile * 16u;

    for (var oc_local = tid; oc_local < 16u; oc_local += BLOCK_SIZE) {
        let oc = oc_start + oc_local;
        if (oc >= out_channels) { continue; }

        var sum: f32 = 0.0;

        for (var ic = 0u; ic < in_channels; ic++) {
            let pw_idx = oc * in_channels + ic;
            sum += depthwise_out[ic] * pointwise_weight[pw_idx];
        }

        sum += pointwise_bias[oc];

        let out_idx = idx_nchw(batch_idx, oc, h_out, w_out,
                               out_channels, out_height, out_width);
        output[out_idx] = sum;
    }
}
