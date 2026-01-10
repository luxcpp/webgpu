// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Convolution NHWC (Channels Last) in WGSL
// Optimized for memory coalescing with channels-last tensor format
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

struct Conv2dNHWCParams {
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

// Input: [N, H_in, W_in, C_in] - TensorFlow/Keras format
// Weight: [K_h, K_w, C_in, C_out] or [C_out, K_h, K_w, C_in]
// Output: [N, H_out, W_out, C_out]

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Conv2dNHWCParams;

// Shared memory
var<workgroup> weight_tile: array<f32, 4096>;

// ============================================================================
// Helper Functions
// ============================================================================

fn compute_output_size(in_size: u32, kernel: u32, pad: u32, stride: u32, dilation: u32) -> u32 {
    return (in_size + 2u * pad - dilation * (kernel - 1u) - 1u) / stride + 1u;
}

fn idx_nhwc(n: u32, h: u32, w: u32, c: u32, H: u32, W: u32, C: u32) -> u32 {
    return ((n * H + h) * W + w) * C + c;
}

// ============================================================================
// NHWC Conv2d Forward
// ============================================================================

@compute @workgroup_size(256)
fn conv2d_nhwc_forward(
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
    let oc_start = oc_tile * TILE_OC;

    // Load weight tile cooperatively
    let weight_size = kernel_h * kernel_w * in_channels * min(TILE_OC, out_channels - oc_start);
    for (var i = tid; i < weight_size; i += BLOCK_SIZE) {
        let kh_kw_ic = i / TILE_OC;
        let oc_local = i % TILE_OC;

        let kh_kw = kh_kw_ic / in_channels;
        let ic = kh_kw_ic % in_channels;
        let kh = kh_kw / kernel_w;
        let kw = kh_kw % kernel_w;

        let oc = oc_start + oc_local;
        if (oc < out_channels) {
            // Weight layout: [K_h, K_w, C_in, C_out]
            let w_idx = ((kh * kernel_w + kw) * in_channels + ic) * out_channels + oc;
            weight_tile[i] = weight[w_idx];
        }
    }
    workgroupBarrier();

    // Compute convolution
    var acc: array<f32, 32>;  // Max TILE_OC
    for (var i = 0u; i < 32u; i++) {
        acc[i] = 0.0;
    }

    for (var kh = 0u; kh < kernel_h; kh++) {
        for (var kw = 0u; kw < kernel_w; kw++) {
            let h_in = i32(h_out * stride_h + kh * dilation_h) - i32(pad_h);
            let w_in = i32(w_out * stride_w + kw * dilation_w) - i32(pad_w);

            if (h_in >= 0 && h_in < i32(in_height) &&
                w_in >= 0 && w_in < i32(in_width)) {

                for (var ic = 0u; ic < in_channels; ic++) {
                    // NHWC layout: input[n, h, w, c]
                    let in_idx = idx_nhwc(batch_idx, u32(h_in), u32(w_in), ic,
                                          in_height, in_width, in_channels);
                    let in_val = input[in_idx];

                    let w_base = (kh * kernel_w + kw) * in_channels * TILE_OC + ic * TILE_OC;

                    for (var oc_local = 0u; oc_local < TILE_OC; oc_local++) {
                        if (oc_start + oc_local < out_channels) {
                            acc[oc_local] += in_val * weight_tile[w_base + oc_local];
                        }
                    }
                }
            }
        }
    }

    // Write output with bias (NHWC layout)
    for (var oc_local = 0u; oc_local < TILE_OC; oc_local++) {
        let oc = oc_start + oc_local;
        if (oc >= out_channels) { break; }

        var result = acc[oc_local] + bias[oc];
        let out_idx = idx_nhwc(batch_idx, h_out, w_out, oc,
                               out_height, out_width, out_channels);
        output[out_idx] = result;
    }
}

// ============================================================================
// NHWC 1x1 Convolution (Pointwise)
// ============================================================================

@compute @workgroup_size(256)
fn conv2d_nhwc_1x1(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let oc_tile = wid.y;
    let spatial_idx = wid.x * BLOCK_SIZE + tid;

    let height = params.in_height;
    let width = params.in_width;
    let in_channels = params.in_channels;
    let out_channels = params.out_channels;
    let batch_size = params.batch_size;

    let total_elements = batch_size * height * width;
    if (spatial_idx >= total_elements) { return; }

    let n = spatial_idx / (height * width);
    let hw = spatial_idx % (height * width);
    let h = hw / width;
    let w = hw % width;

    let oc_start = oc_tile * 16u;
    let num_oc = min(16u, out_channels - oc_start);

    // Load weight tile
    for (var i = tid; i < in_channels * num_oc; i += BLOCK_SIZE) {
        let ic = i / num_oc;
        let oc_local = i % num_oc;
        // Weight: [C_in, C_out]
        weight_tile[i] = weight[ic * out_channels + oc_start + oc_local];
    }
    workgroupBarrier();

    // Compute
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) {
        acc[i] = 0.0;
    }

    for (var ic = 0u; ic < in_channels; ic++) {
        let in_idx = idx_nhwc(n, h, w, ic, height, width, in_channels);
        let in_val = input[in_idx];

        for (var oc_local = 0u; oc_local < num_oc; oc_local++) {
            acc[oc_local] += in_val * weight_tile[ic * num_oc + oc_local];
        }
    }

    // Write output
    for (var oc_local = 0u; oc_local < num_oc; oc_local++) {
        var result = acc[oc_local] + bias[oc_start + oc_local];
        let out_idx = idx_nhwc(n, h, w, oc_start + oc_local, height, width, out_channels);
        output[out_idx] = result;
    }
}

// ============================================================================
// NHWC Grouped Convolution
// ============================================================================

@compute @workgroup_size(256)
fn conv2d_nhwc_grouped(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_group_idx = wid.z;
    let batch_idx = batch_group_idx / params.groups;
    let group_idx = batch_group_idx % params.groups;
    let spatial_idx = wid.x * BLOCK_SIZE + tid;
    let oc_local = wid.y;

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
    if (oc_local >= out_channels_per_group) { return; }

    let h_out = spatial_idx / out_width;
    let w_out = spatial_idx % out_width;
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

                    let in_idx = idx_nhwc(batch_idx, u32(h_in), u32(w_in), ic,
                                          in_height, in_width, in_channels);
                    // Weight layout for grouped: [K_h, K_w, C_in/groups, C_out]
                    let w_idx = ((kh * kernel_w + kw) * in_channels_per_group + ic_local) * out_channels + oc;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    sum += bias[oc];
    let out_idx = idx_nhwc(batch_idx, h_out, w_out, oc,
                           out_height, out_width, out_channels);
    output[out_idx] = sum;
}

// ============================================================================
// NHWC Conv2d with Fused ReLU
// ============================================================================

@compute @workgroup_size(256)
fn conv2d_nhwc_relu(
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

    for (var kh = 0u; kh < kernel_h; kh++) {
        for (var kw = 0u; kw < kernel_w; kw++) {
            let h_in = i32(h_out * stride_h + kh * dilation_h) - i32(pad_h);
            let w_in = i32(w_out * stride_w + kw * dilation_w) - i32(pad_w);

            if (h_in >= 0 && h_in < i32(in_height) &&
                w_in >= 0 && w_in < i32(in_width)) {

                for (var ic = 0u; ic < in_channels; ic++) {
                    let in_idx = idx_nhwc(batch_idx, u32(h_in), u32(w_in), ic,
                                          in_height, in_width, in_channels);
                    let in_val = input[in_idx];

                    for (var oc_local = 0u; oc_local < 16u; oc_local++) {
                        let oc = oc_start + oc_local;
                        if (oc < out_channels) {
                            let w_idx = ((kh * kernel_w + kw) * in_channels + ic) * out_channels + oc;
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
            let out_idx = idx_nhwc(batch_idx, h_out, w_out, oc,
                                   out_height, out_width, out_channels);
            output[out_idx] = result;
        }
    }
}

// ============================================================================
// NHWC Conv2d with Fused ReLU6
// ============================================================================

@compute @workgroup_size(256)
fn conv2d_nhwc_relu6(
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

    for (var kh = 0u; kh < kernel_h; kh++) {
        for (var kw = 0u; kw < kernel_w; kw++) {
            let h_in = i32(h_out * stride_h + kh * dilation_h) - i32(pad_h);
            let w_in = i32(w_out * stride_w + kw * dilation_w) - i32(pad_w);

            if (h_in >= 0 && h_in < i32(in_height) &&
                w_in >= 0 && w_in < i32(in_width)) {

                for (var ic = 0u; ic < in_channels; ic++) {
                    let in_idx = idx_nhwc(batch_idx, u32(h_in), u32(w_in), ic,
                                          in_height, in_width, in_channels);
                    let in_val = input[in_idx];

                    for (var oc_local = 0u; oc_local < 16u; oc_local++) {
                        let oc = oc_start + oc_local;
                        if (oc < out_channels) {
                            let w_idx = ((kh * kernel_w + kw) * in_channels + ic) * out_channels + oc;
                            acc[oc_local] += in_val * weight[w_idx];
                        }
                    }
                }
            }
        }
    }

    // Write with ReLU6: clamp(x, 0, 6)
    for (var oc_local = 0u; oc_local < 16u; oc_local++) {
        let oc = oc_start + oc_local;
        if (oc < out_channels) {
            var result = acc[oc_local] + bias[oc];
            result = clamp(result, 0.0, 6.0);  // ReLU6
            let out_idx = idx_nhwc(batch_idx, h_out, w_out, oc,
                                   out_height, out_width, out_channels);
            output[out_idx] = result;
        }
    }
}

// ============================================================================
// NHWC Conv2d with Fused Swish/SiLU
// ============================================================================

@compute @workgroup_size(256)
fn conv2d_nhwc_swish(
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

    for (var kh = 0u; kh < kernel_h; kh++) {
        for (var kw = 0u; kw < kernel_w; kw++) {
            let h_in = i32(h_out * stride_h + kh * dilation_h) - i32(pad_h);
            let w_in = i32(w_out * stride_w + kw * dilation_w) - i32(pad_w);

            if (h_in >= 0 && h_in < i32(in_height) &&
                w_in >= 0 && w_in < i32(in_width)) {

                for (var ic = 0u; ic < in_channels; ic++) {
                    let in_idx = idx_nhwc(batch_idx, u32(h_in), u32(w_in), ic,
                                          in_height, in_width, in_channels);
                    let in_val = input[in_idx];

                    for (var oc_local = 0u; oc_local < 16u; oc_local++) {
                        let oc = oc_start + oc_local;
                        if (oc < out_channels) {
                            let w_idx = ((kh * kernel_w + kw) * in_channels + ic) * out_channels + oc;
                            acc[oc_local] += in_val * weight[w_idx];
                        }
                    }
                }
            }
        }
    }

    // Write with Swish/SiLU: x * sigmoid(x)
    for (var oc_local = 0u; oc_local < 16u; oc_local++) {
        let oc = oc_start + oc_local;
        if (oc < out_channels) {
            let x = acc[oc_local] + bias[oc];
            let result = x / (1.0 + exp(-x));  // Swish/SiLU
            let out_idx = idx_nhwc(batch_idx, h_out, w_out, oc,
                                   out_height, out_width, out_channels);
            output[out_idx] = result;
        }
    }
}

// ============================================================================
// NHWC Layout Conversion (NCHW -> NHWC)
// ============================================================================

struct LayoutConvertParams {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
}

@group(0) @binding(5) var<uniform> convert_params: LayoutConvertParams;

@compute @workgroup_size(256)
fn nchw_to_nhwc(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let total = convert_params.batch_size * convert_params.height *
                convert_params.width * convert_params.channels;

    if (idx >= total) { return; }

    let N = convert_params.batch_size;
    let C = convert_params.channels;
    let H = convert_params.height;
    let W = convert_params.width;

    // Output index is in NHWC order
    let n = idx / (H * W * C);
    let rem = idx % (H * W * C);
    let h = rem / (W * C);
    let rem2 = rem % (W * C);
    let w = rem2 / C;
    let c = rem2 % C;

    // Input index in NCHW order
    let nchw_idx = ((n * C + c) * H + h) * W + w;

    output[idx] = input[nchw_idx];
}

// ============================================================================
// NHWC Layout Conversion (NHWC -> NCHW)
// ============================================================================

@compute @workgroup_size(256)
fn nhwc_to_nchw(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let total = convert_params.batch_size * convert_params.height *
                convert_params.width * convert_params.channels;

    if (idx >= total) { return; }

    let N = convert_params.batch_size;
    let C = convert_params.channels;
    let H = convert_params.height;
    let W = convert_params.width;

    // Output index is in NCHW order
    let n = idx / (C * H * W);
    let rem = idx % (C * H * W);
    let c = rem / (H * W);
    let rem2 = rem % (H * W);
    let h = rem2 / W;
    let w = rem2 % W;

    // Input index in NHWC order
    let nhwc_idx = ((n * H + h) * W + w) * C + c;

    output[idx] = input[nhwc_idx];
}
