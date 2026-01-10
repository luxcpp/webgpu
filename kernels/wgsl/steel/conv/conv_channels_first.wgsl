// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Convolution NCHW (Channels First) in WGSL
// Optimized for PyTorch/ONNX tensor format
// Works on WebGPU (Metal/Vulkan/D3D12 via Dawn/wgpu)

// ============================================================================
// Configuration Constants
// ============================================================================

const TILE_OUT_H: u32 = 4u;
const TILE_OUT_W: u32 = 4u;
const TILE_OC: u32 = 32u;
const TILE_IC: u32 = 16u;
const BLOCK_SIZE: u32 = 256u;

// ============================================================================
// Bindings
// ============================================================================

struct Conv2dParams {
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
    dilation_h: u32,
    dilation_w: u32,
    groups: u32,
    _pad1: u32,
    _pad2: u32,
}

// Input: [N, C_in, H_in, W_in] - PyTorch/ONNX format
// Weight: [C_out, C_in/groups, K_h, K_w]
// Output: [N, C_out, H_out, W_out]

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Conv2dParams;

// Shared memory
var<workgroup> weight_tile: array<f32, 4096>;  // TILE_OC * TILE_IC * 9 (3x3 max)
var<workgroup> input_tile: array<f32, 2048>;   // Padded input region

// ============================================================================
// Helper Functions
// ============================================================================

fn compute_output_size(in_size: u32, kernel: u32, pad: u32, stride: u32, dilation: u32) -> u32 {
    return (in_size + 2u * pad - dilation * (kernel - 1u) - 1u) / stride + 1u;
}

fn idx_nchw(n: u32, c: u32, h: u32, w: u32, C: u32, H: u32, W: u32) -> u32 {
    return ((n * C + c) * H + h) * W + w;
}

// ============================================================================
// NCHW Conv2d Forward
// ============================================================================

@compute @workgroup_size(256)
fn conv2d_nchw_forward(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z;
    let oc_tile = wid.y;
    let spatial_tile = wid.x;

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
    let in_channels = params.in_channels;
    let out_channels = params.out_channels;

    let out_height = compute_output_size(in_height, kernel_h, pad_h, stride_h, dilation_h);
    let out_width = compute_output_size(in_width, kernel_w, pad_w, stride_w, dilation_w);

    let out_tiles_w = (out_width + TILE_OUT_W - 1u) / TILE_OUT_W;
    let tile_h = spatial_tile / out_tiles_w;
    let tile_w = spatial_tile % out_tiles_w;

    let h_out_start = tile_h * TILE_OUT_H;
    let w_out_start = tile_w * TILE_OUT_W;
    let oc_start = oc_tile * TILE_OC;

    // Accumulator for output tile
    var acc: array<f32, 512>;  // TILE_OUT_H * TILE_OUT_W * TILE_OC
    for (var i = 0u; i < 512u; i++) {
        acc[i] = 0.0;
    }

    // Process input channels in tiles
    let num_ic_tiles = (in_channels + TILE_IC - 1u) / TILE_IC;

    for (var ic_tile = 0u; ic_tile < num_ic_tiles; ic_tile++) {
        let ic_start = ic_tile * TILE_IC;
        let ic_end = min(ic_start + TILE_IC, in_channels);
        let ic_count = ic_end - ic_start;

        // Load weight tile cooperatively
        let weight_size = min(TILE_OC, out_channels - oc_start) * ic_count * kernel_h * kernel_w;
        for (var i = tid; i < weight_size; i += BLOCK_SIZE) {
            let oc_local = i / (ic_count * kernel_h * kernel_w);
            let rem = i % (ic_count * kernel_h * kernel_w);
            let ic_local = rem / (kernel_h * kernel_w);
            let k_idx = rem % (kernel_h * kernel_w);
            let kh = k_idx / kernel_w;
            let kw = k_idx % kernel_w;

            let oc = oc_start + oc_local;
            let ic = ic_start + ic_local;

            if (oc < out_channels && ic < in_channels) {
                // Weight layout: [C_out, C_in, K_h, K_w]
                let w_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                weight_tile[i] = weight[w_idx];
            }
        }
        workgroupBarrier();

        // Compute convolution for this IC tile
        for (var oh = 0u; oh < TILE_OUT_H; oh++) {
            let h_out = h_out_start + oh;
            if (h_out >= out_height) { continue; }

            for (var ow = 0u; ow < TILE_OUT_W; ow++) {
                let w_out = w_out_start + ow;
                if (w_out >= out_width) { continue; }

                for (var kh = 0u; kh < kernel_h; kh++) {
                    for (var kw = 0u; kw < kernel_w; kw++) {
                        let h_in = i32(h_out * stride_h + kh * dilation_h) - i32(pad_h);
                        let w_in = i32(w_out * stride_w + kw * dilation_w) - i32(pad_w);

                        if (h_in >= 0 && h_in < i32(in_height) &&
                            w_in >= 0 && w_in < i32(in_width)) {

                            for (var ic_local = tid; ic_local < ic_count; ic_local += BLOCK_SIZE) {
                                let ic = ic_start + ic_local;

                                // NCHW layout
                                let in_idx = idx_nchw(batch_idx, ic, u32(h_in), u32(w_in),
                                                      in_channels, in_height, in_width);
                                let in_val = input[in_idx];

                                for (var oc_local = 0u; oc_local < min(TILE_OC, out_channels - oc_start); oc_local++) {
                                    let w_tile_idx = (oc_local * ic_count + ic_local) * kernel_h * kernel_w +
                                                     kh * kernel_w + kw;
                                    let w_val = weight_tile[w_tile_idx];

                                    let acc_idx = (oh * TILE_OUT_W + ow) * TILE_OC + oc_local;
                                    acc[acc_idx] += in_val * w_val;
                                }
                            }
                        }
                    }
                }
            }
        }
        workgroupBarrier();
    }

    // Write output with bias
    for (var i = tid; i < TILE_OUT_H * TILE_OUT_W * TILE_OC; i += BLOCK_SIZE) {
        let oh = i / (TILE_OUT_W * TILE_OC);
        let rem = i % (TILE_OUT_W * TILE_OC);
        let ow = rem / TILE_OC;
        let oc_local = rem % TILE_OC;

        let h_out = h_out_start + oh;
        let w_out = w_out_start + ow;
        let oc = oc_start + oc_local;

        if (h_out < out_height && w_out < out_width && oc < out_channels) {
            var result = acc[i];
            result += bias[oc];

            let out_idx = idx_nchw(batch_idx, oc, h_out, w_out,
                                   out_channels, out_height, out_width);
            output[out_idx] = result;
        }
    }
}

// ============================================================================
// NCHW Conv2d 1x1 (Pointwise)
// ============================================================================

@compute @workgroup_size(256)
fn conv2d_nchw_1x1(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z;
    let oc_tile = wid.y;
    let spatial_idx = wid.x * BLOCK_SIZE + tid;

    if (batch_idx >= params.batch_size) { return; }

    let height = params.in_height;
    let width = params.in_width;
    let in_channels = params.in_channels;
    let out_channels = params.out_channels;

    let spatial_size = height * width;
    if (spatial_idx >= spatial_size) { return; }

    let h = spatial_idx / width;
    let w = spatial_idx % width;
    let oc_start = oc_tile * 16u;
    let num_oc = min(16u, out_channels - oc_start);

    // Accumulate
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) {
        acc[i] = 0.0;
    }

    for (var ic = 0u; ic < in_channels; ic++) {
        let in_idx = idx_nchw(batch_idx, ic, h, w, in_channels, height, width);
        let in_val = input[in_idx];

        for (var oc_local = 0u; oc_local < num_oc; oc_local++) {
            let oc = oc_start + oc_local;
            // Weight layout for 1x1: [C_out, C_in, 1, 1] -> effectively [C_out, C_in]
            let w_idx = oc * in_channels + ic;
            acc[oc_local] += in_val * weight[w_idx];
        }
    }

    // Write output
    for (var oc_local = 0u; oc_local < num_oc; oc_local++) {
        let oc = oc_start + oc_local;
        var result = acc[oc_local] + bias[oc];
        let out_idx = idx_nchw(batch_idx, oc, h, w, out_channels, height, width);
        output[out_idx] = result;
    }
}

// ============================================================================
// NCHW Conv2d 3x3 Optimized
// ============================================================================

@compute @workgroup_size(256)
fn conv2d_nchw_3x3(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z;
    let oc_tile = wid.y;
    let spatial_tile = wid.x;

    if (batch_idx >= params.batch_size) { return; }

    let in_height = params.in_height;
    let in_width = params.in_width;
    let stride_h = params.stride_h;
    let stride_w = params.stride_w;
    let pad_h = params.pad_h;
    let pad_w = params.pad_w;
    let in_channels = params.in_channels;
    let out_channels = params.out_channels;

    let out_height = compute_output_size(in_height, 3u, pad_h, stride_h, 1u);
    let out_width = compute_output_size(in_width, 3u, pad_w, stride_w, 1u);

    let out_tiles_w = (out_width + TILE_OUT_W - 1u) / TILE_OUT_W;
    let tile_h = spatial_tile / out_tiles_w;
    let tile_w = spatial_tile % out_tiles_w;

    let h_out_start = tile_h * TILE_OUT_H;
    let w_out_start = tile_w * TILE_OUT_W;
    let oc_start = oc_tile * TILE_OC;

    // Thread computes one output position
    let thread_h = tid / (TILE_OUT_W * (TILE_OC / 8u));
    let rem = tid % (TILE_OUT_W * (TILE_OC / 8u));
    let thread_w = rem / (TILE_OC / 8u);
    let oc_group = rem % (TILE_OC / 8u);

    let h_out = h_out_start + thread_h;
    let w_out = w_out_start + thread_w;

    if (h_out >= out_height || w_out >= out_width) { return; }

    // Each thread computes 8 output channels
    var acc: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        acc[i] = 0.0;
    }

    let oc_base = oc_start + oc_group * 8u;

    for (var ic = 0u; ic < in_channels; ic++) {
        // 3x3 kernel unrolled
        for (var kh = 0u; kh < 3u; kh++) {
            for (var kw = 0u; kw < 3u; kw++) {
                let h_in = i32(h_out * stride_h + kh) - i32(pad_h);
                let w_in = i32(w_out * stride_w + kw) - i32(pad_w);

                if (h_in >= 0 && h_in < i32(in_height) &&
                    w_in >= 0 && w_in < i32(in_width)) {

                    let in_idx = idx_nchw(batch_idx, ic, u32(h_in), u32(w_in),
                                          in_channels, in_height, in_width);
                    let in_val = input[in_idx];

                    for (var oc_local = 0u; oc_local < 8u; oc_local++) {
                        let oc = oc_base + oc_local;
                        if (oc < out_channels) {
                            let w_idx = ((oc * in_channels + ic) * 3u + kh) * 3u + kw;
                            acc[oc_local] += in_val * weight[w_idx];
                        }
                    }
                }
            }
        }
    }

    // Write output
    for (var oc_local = 0u; oc_local < 8u; oc_local++) {
        let oc = oc_base + oc_local;
        if (oc < out_channels) {
            var result = acc[oc_local] + bias[oc];
            let out_idx = idx_nchw(batch_idx, oc, h_out, w_out,
                                   out_channels, out_height, out_width);
            output[out_idx] = result;
        }
    }
}

// ============================================================================
// NCHW Transposed Convolution (ConvTranspose2d)
// ============================================================================

struct ConvTranspose2dParams {
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
    output_pad_h: u32,
    output_pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    groups: u32,
}

@group(0) @binding(5) var<uniform> transpose_params: ConvTranspose2dParams;

@compute @workgroup_size(256)
fn conv_transpose2d_nchw(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z;
    let oc_tile = wid.y;
    let spatial_idx = wid.x * BLOCK_SIZE + tid;

    if (batch_idx >= transpose_params.batch_size) { return; }

    let in_height = transpose_params.in_height;
    let in_width = transpose_params.in_width;
    let kernel_h = transpose_params.kernel_h;
    let kernel_w = transpose_params.kernel_w;
    let stride_h = transpose_params.stride_h;
    let stride_w = transpose_params.stride_w;
    let pad_h = transpose_params.pad_h;
    let pad_w = transpose_params.pad_w;
    let output_pad_h = transpose_params.output_pad_h;
    let output_pad_w = transpose_params.output_pad_w;
    let in_channels = transpose_params.in_channels;
    let out_channels = transpose_params.out_channels;

    // Output size for transposed convolution
    let out_height = (in_height - 1u) * stride_h - 2u * pad_h + kernel_h + output_pad_h;
    let out_width = (in_width - 1u) * stride_w - 2u * pad_w + kernel_w + output_pad_w;

    if (spatial_idx >= out_height * out_width) { return; }

    let h_out = spatial_idx / out_width;
    let w_out = spatial_idx % out_width;
    let oc_start = oc_tile * 16u;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) {
        acc[i] = 0.0;
    }

    // For transposed conv, we iterate over input and scatter to output
    // Equivalently, for each output position, find contributing inputs
    for (var ic = 0u; ic < in_channels; ic++) {
        for (var kh = 0u; kh < kernel_h; kh++) {
            for (var kw = 0u; kw < kernel_w; kw++) {
                // Check if this kernel position contributes to (h_out, w_out)
                let h_in_offset = i32(h_out + pad_h) - i32(kh);
                let w_in_offset = i32(w_out + pad_w) - i32(kw);

                if (h_in_offset >= 0 && h_in_offset % i32(stride_h) == 0 &&
                    w_in_offset >= 0 && w_in_offset % i32(stride_w) == 0) {

                    let h_in = u32(h_in_offset) / stride_h;
                    let w_in = u32(w_in_offset) / stride_w;

                    if (h_in < in_height && w_in < in_width) {
                        let in_idx = idx_nchw(batch_idx, ic, h_in, w_in,
                                              in_channels, in_height, in_width);
                        let in_val = input[in_idx];

                        for (var oc_local = 0u; oc_local < 16u; oc_local++) {
                            let oc = oc_start + oc_local;
                            if (oc < out_channels) {
                                // Weight layout: [C_in, C_out, K_h, K_w] for transposed
                                let w_idx = ((ic * out_channels + oc) * kernel_h + kh) * kernel_w + kw;
                                acc[oc_local] += in_val * weight[w_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // Write output
    for (var oc_local = 0u; oc_local < 16u; oc_local++) {
        let oc = oc_start + oc_local;
        if (oc < out_channels) {
            var result = acc[oc_local] + bias[oc];
            let out_idx = idx_nchw(batch_idx, oc, h_out, w_out,
                                   out_channels, out_height, out_width);
            output[out_idx] = result;
        }
    }
}

// ============================================================================
// NCHW Grouped Convolution
// ============================================================================

@compute @workgroup_size(256)
fn conv2d_nchw_grouped(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_group_idx = wid.z;
    let batch_idx = batch_group_idx / params.groups;
    let group_idx = batch_group_idx % params.groups;
    let spatial_idx = wid.x * BLOCK_SIZE + tid;
    let oc_local_tile = wid.y;

    if (batch_idx >= params.batch_size) { return; }

    let in_height = params.in_height;
    let in_width = params.in_width;
    let kernel_h = params.kernel_h;
    let kernel_w = params.kernel_w;
    let stride_h = params.stride_h;
    let stride_w = params.stride_w;
    let pad_h = params.pad_h;
    let pad_w = params.pad_w;
    let in_channels = params.in_channels;
    let out_channels = params.out_channels;
    let groups = params.groups;

    let in_channels_per_group = in_channels / groups;
    let out_channels_per_group = out_channels / groups;

    let out_height = compute_output_size(in_height, kernel_h, pad_h, stride_h, 1u);
    let out_width = compute_output_size(in_width, kernel_w, pad_w, stride_w, 1u);

    if (spatial_idx >= out_height * out_width) { return; }

    let h_out = spatial_idx / out_width;
    let w_out = spatial_idx % out_width;

    let oc_local = oc_local_tile;
    if (oc_local >= out_channels_per_group) { return; }

    let oc = group_idx * out_channels_per_group + oc_local;

    var sum: f32 = 0.0;

    for (var ic_local = 0u; ic_local < in_channels_per_group; ic_local++) {
        let ic = group_idx * in_channels_per_group + ic_local;

        for (var kh = 0u; kh < kernel_h; kh++) {
            for (var kw = 0u; kw < kernel_w; kw++) {
                let h_in = i32(h_out * stride_h + kh) - i32(pad_h);
                let w_in = i32(w_out * stride_w + kw) - i32(pad_w);

                if (h_in >= 0 && h_in < i32(in_height) &&
                    w_in >= 0 && w_in < i32(in_width)) {

                    let in_idx = idx_nchw(batch_idx, ic, u32(h_in), u32(w_in),
                                          in_channels, in_height, in_width);
                    // Weight: [C_out, C_in/groups, K_h, K_w]
                    let w_idx = ((oc * in_channels_per_group + ic_local) * kernel_h + kh) * kernel_w + kw;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    sum += bias[oc];
    let out_idx = idx_nchw(batch_idx, oc, h_out, w_out, out_channels, out_height, out_width);
    output[out_idx] = sum;
}

// ============================================================================
// NCHW Conv2d with Fused Activation
// ============================================================================

@compute @workgroup_size(256)
fn conv2d_nchw_relu(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.z;
    let oc_tile = wid.y;
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
    let in_channels = params.in_channels;
    let out_channels = params.out_channels;

    let out_height = compute_output_size(in_height, kernel_h, pad_h, stride_h, dilation_h);
    let out_width = compute_output_size(in_width, kernel_w, pad_w, stride_w, dilation_w);

    if (spatial_idx >= out_height * out_width) { return; }

    let h_out = spatial_idx / out_width;
    let w_out = spatial_idx % out_width;
    let oc_start = oc_tile * 16u;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) {
        acc[i] = 0.0;
    }

    for (var ic = 0u; ic < in_channels; ic++) {
        for (var kh = 0u; kh < kernel_h; kh++) {
            for (var kw = 0u; kw < kernel_w; kw++) {
                let h_in = i32(h_out * stride_h + kh * dilation_h) - i32(pad_h);
                let w_in = i32(w_out * stride_w + kw * dilation_w) - i32(pad_w);

                if (h_in >= 0 && h_in < i32(in_height) &&
                    w_in >= 0 && w_in < i32(in_width)) {

                    let in_idx = idx_nchw(batch_idx, ic, u32(h_in), u32(w_in),
                                          in_channels, in_height, in_width);
                    let in_val = input[in_idx];

                    for (var oc_local = 0u; oc_local < 16u; oc_local++) {
                        let oc = oc_start + oc_local;
                        if (oc < out_channels) {
                            let w_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                            acc[oc_local] += in_val * weight[w_idx];
                        }
                    }
                }
            }
        }
    }

    // Write with ReLU
    for (var oc_local = 0u; oc_local < 16u; oc_local++) {
        let oc = oc_start + oc_local;
        if (oc < out_channels) {
            var result = acc[oc_local] + bias[oc];
            result = max(result, 0.0);  // ReLU
            let out_idx = idx_nchw(batch_idx, oc, h_out, w_out,
                                   out_channels, out_height, out_width);
            output[out_idx] = result;
        }
    }
}
