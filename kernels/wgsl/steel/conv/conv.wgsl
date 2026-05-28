// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Convolution WGSL (umbrella for cuda/kernels/steel/conv/conv.cu and
// metal/src/shaders/steel/conv/conv.metal). Provides direct conv2d, depthwise,
// 1x1 pointwise, and im2col/col2im transforms.

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
    has_bias: u32,
}

struct DepthwiseParams {
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
    has_bias: u32,
    _pad: u32,
}

struct Conv1x1Params {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    height: u32,
    width: u32,
    has_bias: u32,
    _pad0: u32,
    _pad1: u32,
}

struct Im2ColParams {
    in_channels: u32,
    in_height: u32,
    in_width: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read>       input:  array<f32>;
@group(0) @binding(2) var<storage, read>       weight: array<f32>;
@group(0) @binding(3) var<storage, read>       bias:   array<f32>;
@group(0) @binding(4) var<uniform>             conv_p: Conv2dParams;
@group(0) @binding(5) var<uniform>             dw_p:   DepthwiseParams;
@group(0) @binding(6) var<uniform>             p1x1:   Conv1x1Params;
@group(0) @binding(7) var<uniform>             i2c_p:  Im2ColParams;
@group(0) @binding(8) var<storage, read_write> col:    array<f32>;
@group(0) @binding(9) var<storage, read_write> input_grad: array<f32>;
@group(0) @binding(10) var<storage, read>      col_grad:   array<f32>;

// ============================================================================
// Direct Conv2D (NCHW, im2col-free) — 16x16 tile threading
// Grid: (ceil(W/16), ceil(H/16), batch * out_channels)
// ============================================================================
@compute @workgroup_size(16, 16)
fn steel_conv2d_direct(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let p = conv_p;
    let out_height = (p.in_height + 2u * p.pad_h - p.kernel_h) / p.stride_h + 1u;
    let out_width  = (p.in_width  + 2u * p.pad_w - p.kernel_w) / p.stride_w + 1u;

    let n     = wid.z / p.out_channels;
    let c_out = wid.z % p.out_channels;
    let h_out = wid.y * 16u + lid.y;
    let w_out = wid.x * 16u + lid.x;
    if (n >= p.batch_size || h_out >= out_height || w_out >= out_width) { return; }

    var sum: f32 = 0.0;
    for (var c_in: u32 = 0u; c_in < p.in_channels; c_in = c_in + 1u) {
        for (var kh: u32 = 0u; kh < p.kernel_h; kh = kh + 1u) {
            for (var kw: u32 = 0u; kw < p.kernel_w; kw = kw + 1u) {
                let h_in_s = i32(h_out * p.stride_h) - i32(p.pad_h) + i32(kh);
                let w_in_s = i32(w_out * p.stride_w) - i32(p.pad_w) + i32(kw);
                if (h_in_s >= 0 && h_in_s < i32(p.in_height)
                 && w_in_s >= 0 && w_in_s < i32(p.in_width)) {
                    let in_val = input[n * p.in_channels * p.in_height * p.in_width
                                     + c_in * p.in_height * p.in_width
                                     + u32(h_in_s) * p.in_width + u32(w_in_s)];
                    let wt_val = weight[c_out * p.in_channels * p.kernel_h * p.kernel_w
                                      + c_in * p.kernel_h * p.kernel_w
                                      + kh * p.kernel_w + kw];
                    sum = sum + in_val * wt_val;
                }
            }
        }
    }
    if (p.has_bias != 0u) { sum = sum + bias[c_out]; }
    output[n * p.out_channels * out_height * out_width
         + c_out * out_height * out_width
         + h_out * out_width + w_out] = sum;
}

// ============================================================================
// Depthwise Conv2D (NCHW)
// ============================================================================
@compute @workgroup_size(16, 16)
fn steel_conv2d_depthwise(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let p = dw_p;
    let out_height = (p.in_height + 2u * p.pad_h - p.kernel_h) / p.stride_h + 1u;
    let out_width  = (p.in_width  + 2u * p.pad_w - p.kernel_w) / p.stride_w + 1u;

    let n     = wid.z / p.channels;
    let c     = wid.z % p.channels;
    let h_out = wid.y * 16u + lid.y;
    let w_out = wid.x * 16u + lid.x;
    if (n >= p.batch_size || h_out >= out_height || w_out >= out_width) { return; }

    var sum: f32 = 0.0;
    for (var kh: u32 = 0u; kh < p.kernel_h; kh = kh + 1u) {
        for (var kw: u32 = 0u; kw < p.kernel_w; kw = kw + 1u) {
            let h_in_s = i32(h_out * p.stride_h) - i32(p.pad_h) + i32(kh);
            let w_in_s = i32(w_out * p.stride_w) - i32(p.pad_w) + i32(kw);
            if (h_in_s >= 0 && h_in_s < i32(p.in_height)
             && w_in_s >= 0 && w_in_s < i32(p.in_width)) {
                let in_val = input[n * p.channels * p.in_height * p.in_width
                                 + c * p.in_height * p.in_width
                                 + u32(h_in_s) * p.in_width + u32(w_in_s)];
                let wt_val = weight[c * p.kernel_h * p.kernel_w + kh * p.kernel_w + kw];
                sum = sum + in_val * wt_val;
            }
        }
    }
    if (p.has_bias != 0u) { sum = sum + bias[c]; }
    output[n * p.channels * out_height * out_width
         + c * out_height * out_width
         + h_out * out_width + w_out] = sum;
}

// ============================================================================
// 1x1 Conv (Pointwise)
// Grid: (ceil(H*W / 256), out_channels, batch); 256 threads
// ============================================================================
@compute @workgroup_size(256)
fn steel_conv2d_1x1(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let p = p1x1;
    let spatial_idx = wid.x * 256u + lid.x;
    let c_out = wid.y;
    let n     = wid.z;
    if (n >= p.batch_size || spatial_idx >= p.height * p.width) { return; }

    let in_base = n * p.in_channels * p.height * p.width;
    var sum: f32 = 0.0;
    for (var c_in: u32 = 0u; c_in < p.in_channels; c_in = c_in + 1u) {
        sum = sum + input[in_base + c_in * p.height * p.width + spatial_idx]
                  * weight[c_out * p.in_channels + c_in];
    }
    if (p.has_bias != 0u) { sum = sum + bias[c_out]; }
    output[n * p.out_channels * p.height * p.width + c_out * p.height * p.width + spatial_idx] = sum;
}

// ============================================================================
// Im2Col
// ============================================================================
@compute @workgroup_size(256)
fn steel_im2col(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = i2c_p;
    let out_height = (p.in_height + 2u * p.pad_h - p.kernel_h) / p.stride_h + 1u;
    let out_width  = (p.in_width  + 2u * p.pad_w - p.kernel_w) / p.stride_w + 1u;
    let num_col_elements = p.in_channels * p.kernel_h * p.kernel_w * out_height * out_width;
    let g = gid.x;
    if (g >= num_col_elements) { return; }

    let spatial_idx = g % (out_height * out_width);
    let kernel_idx  = g / (out_height * out_width);
    let h_out = spatial_idx / out_width;
    let w_out = spatial_idx % out_width;
    let c_in  = kernel_idx / (p.kernel_h * p.kernel_w);
    let k_idx = kernel_idx % (p.kernel_h * p.kernel_w);
    let kh    = k_idx / p.kernel_w;
    let kw    = k_idx % p.kernel_w;

    let h_in_s = i32(h_out * p.stride_h) - i32(p.pad_h) + i32(kh);
    let w_in_s = i32(w_out * p.stride_w) - i32(p.pad_w) + i32(kw);
    var val: f32 = 0.0;
    if (h_in_s >= 0 && h_in_s < i32(p.in_height)
     && w_in_s >= 0 && w_in_s < i32(p.in_width)) {
        val = input[c_in * p.in_height * p.in_width + u32(h_in_s) * p.in_width + u32(w_in_s)];
    }
    col[kernel_idx * out_height * out_width + spatial_idx] = val;
}

// ============================================================================
// Col2Im (gradient)
// ============================================================================
@compute @workgroup_size(256)
fn steel_col2im(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = i2c_p;
    let out_height = (p.in_height + 2u * p.pad_h - p.kernel_h) / p.stride_h + 1u;
    let out_width  = (p.in_width  + 2u * p.pad_w - p.kernel_w) / p.stride_w + 1u;
    let num_in_elements = p.in_channels * p.in_height * p.in_width;
    let g = gid.x;
    if (g >= num_in_elements) { return; }

    let c_in = g / (p.in_height * p.in_width);
    let spatial_idx = g % (p.in_height * p.in_width);
    let h_in = spatial_idx / p.in_width;
    let w_in = spatial_idx % p.in_width;

    var grad_sum: f32 = 0.0;
    for (var kh: u32 = 0u; kh < p.kernel_h; kh = kh + 1u) {
        for (var kw: u32 = 0u; kw < p.kernel_w; kw = kw + 1u) {
            let h_out_s = i32(h_in) + i32(p.pad_h) - i32(kh);
            let w_out_s = i32(w_in) + i32(p.pad_w) - i32(kw);
            if (h_out_s % i32(p.stride_h) == 0 && w_out_s % i32(p.stride_w) == 0) {
                let h_out_t = h_out_s / i32(p.stride_h);
                let w_out_t = w_out_s / i32(p.stride_w);
                if (h_out_t >= 0 && h_out_t < i32(out_height)
                 && w_out_t >= 0 && w_out_t < i32(out_width)) {
                    let kernel_idx = c_in * p.kernel_h * p.kernel_w + kh * p.kernel_w + kw;
                    let col_idx    = kernel_idx * out_height * out_width
                                    + u32(h_out_t) * out_width + u32(w_out_t);
                    grad_sum = grad_sum + col_grad[col_idx];
                }
            }
        }
    }
    input_grad[g] = grad_sum;
}
