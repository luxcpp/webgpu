// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// RMS Normalization (Root Mean Square Layer Normalization)
// Implements: y = x * gamma / sqrt(mean(x^2) + eps)
//
// RMSNorm is simpler and faster than LayerNorm:
// - No mean subtraction (centers around 0 implicitly)
// - No beta (bias) parameter
// - Used in LLaMA, Gemma, and other modern LLMs
//
// Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
//
// Part of the Lux Network GPU acceleration library

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// ============================================================================
// Parameter Structures
// ============================================================================

struct RMSNormParams {
    batch_size: u32,        // Number of samples in batch
    hidden_size: u32,       // Size of hidden dimension to normalize
    eps: f32,               // Epsilon for numerical stability (typically 1e-6)
    _pad: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;     // Scale parameter (weight)
@group(0) @binding(3) var<uniform> params: RMSNormParams;

// Optional: store RMS values for backward pass
@group(0) @binding(4) var<storage, read_write> rms_out: array<f32>;

// Workgroup shared memory
var<workgroup> shared_sum_sq: array<f32, WORKGROUP_SIZE>;

// ============================================================================
// Main RMS Normalization Kernel
// Each workgroup processes one sample (row)
// ============================================================================

@compute @workgroup_size(256)
fn rms_norm(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_idx = wid.x;
    let hidden_size = params.hidden_size;
    let eps = params.eps;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let base_idx = batch_idx * hidden_size;

    // Step 1: Compute sum of squares
    var local_sum_sq: f32 = 0.0;

    var i = lid.x;
    while (i < hidden_size) {
        let val = input[base_idx + i];
        local_sum_sq += val * val;
        i += WORKGROUP_SIZE;
    }

    // Store to shared memory
    shared_sum_sq[lid.x] = local_sum_sq;
    workgroupBarrier();

    // Parallel reduction for sum of squares
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_sum_sq[lid.x] += shared_sum_sq[lid.x + stride];
        }
        workgroupBarrier();
    }

    // Compute RMS = 1 / sqrt(mean(x^2) + eps)
    let mean_sq = shared_sum_sq[0] / f32(hidden_size);
    let rms = 1.0 / sqrt(mean_sq + eps);

    // Store RMS value if output buffer is bound
    if (lid.x == 0u) {
        rms_out[batch_idx] = rms;
    }

    workgroupBarrier();

    // Step 2: Normalize and apply scale
    i = lid.x;
    while (i < hidden_size) {
        let idx = base_idx + i;
        output[idx] = input[idx] * rms * gamma[i];
        i += WORKGROUP_SIZE;
    }
}

// ============================================================================
// RMS Norm without Gamma (Pre-normalization)
// ============================================================================

@compute @workgroup_size(256)
fn rms_norm_no_gamma(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_idx = wid.x;
    let hidden_size = params.hidden_size;
    let eps = params.eps;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let base_idx = batch_idx * hidden_size;

    // Compute sum of squares
    var local_sum_sq: f32 = 0.0;

    var i = lid.x;
    while (i < hidden_size) {
        let val = input[base_idx + i];
        local_sum_sq += val * val;
        i += WORKGROUP_SIZE;
    }

    shared_sum_sq[lid.x] = local_sum_sq;
    workgroupBarrier();

    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_sum_sq[lid.x] += shared_sum_sq[lid.x + stride];
        }
        workgroupBarrier();
    }

    let rms = 1.0 / sqrt(shared_sum_sq[0] / f32(hidden_size) + eps);
    workgroupBarrier();

    // Normalize without gamma
    i = lid.x;
    while (i < hidden_size) {
        let idx = base_idx + i;
        output[idx] = input[idx] * rms;
        i += WORKGROUP_SIZE;
    }
}

// ============================================================================
// Fused RMS Norm + Residual Add
// output = RMSNorm(input + residual)
// ============================================================================

@group(0) @binding(5) var<storage, read> residual: array<f32>;

@compute @workgroup_size(256)
fn rms_norm_residual(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_idx = wid.x;
    let hidden_size = params.hidden_size;
    let eps = params.eps;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let base_idx = batch_idx * hidden_size;

    // Compute sum of squares on (input + residual)
    var local_sum_sq: f32 = 0.0;

    var i = lid.x;
    while (i < hidden_size) {
        let idx = base_idx + i;
        let val = input[idx] + residual[idx];
        local_sum_sq += val * val;
        i += WORKGROUP_SIZE;
    }

    shared_sum_sq[lid.x] = local_sum_sq;
    workgroupBarrier();

    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_sum_sq[lid.x] += shared_sum_sq[lid.x + stride];
        }
        workgroupBarrier();
    }

    let rms = 1.0 / sqrt(shared_sum_sq[0] / f32(hidden_size) + eps);

    if (lid.x == 0u) {
        rms_out[batch_idx] = rms;
    }

    workgroupBarrier();

    // Normalize and apply scale
    i = lid.x;
    while (i < hidden_size) {
        let idx = base_idx + i;
        let x = input[idx] + residual[idx];
        output[idx] = x * rms * gamma[i];
        i += WORKGROUP_SIZE;
    }
}

// ============================================================================
// Fused RMS Norm + Residual (in-place residual update)
// residual = input + residual
// output = RMSNorm(residual)
// ============================================================================

@group(0) @binding(6) var<storage, read_write> residual_inplace: array<f32>;

@compute @workgroup_size(256)
fn rms_norm_residual_inplace(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_idx = wid.x;
    let hidden_size = params.hidden_size;
    let eps = params.eps;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let base_idx = batch_idx * hidden_size;

    // First pass: update residual and compute sum of squares
    var local_sum_sq: f32 = 0.0;

    var i = lid.x;
    while (i < hidden_size) {
        let idx = base_idx + i;
        let val = input[idx] + residual_inplace[idx];
        residual_inplace[idx] = val;  // Update residual in-place
        local_sum_sq += val * val;
        i += WORKGROUP_SIZE;
    }

    shared_sum_sq[lid.x] = local_sum_sq;
    workgroupBarrier();

    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_sum_sq[lid.x] += shared_sum_sq[lid.x + stride];
        }
        workgroupBarrier();
    }

    let rms = 1.0 / sqrt(shared_sum_sq[0] / f32(hidden_size) + eps);

    if (lid.x == 0u) {
        rms_out[batch_idx] = rms;
    }

    workgroupBarrier();

    // Second pass: normalize from updated residual
    i = lid.x;
    while (i < hidden_size) {
        let idx = base_idx + i;
        output[idx] = residual_inplace[idx] * rms * gamma[i];
        i += WORKGROUP_SIZE;
    }
}

// ============================================================================
// Vectorized RMS Norm (vec4 loads for better memory bandwidth)
// ============================================================================

@compute @workgroup_size(256)
fn rms_norm_vec4(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_idx = wid.x;
    let hidden_size = params.hidden_size;
    let eps = params.eps;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let base_idx = batch_idx * hidden_size;
    let hidden_vec4 = hidden_size / 4u;
    let hidden_remainder = hidden_size % 4u;

    // Vectorized sum of squares
    var local_sum_sq: f32 = 0.0;

    var i = lid.x;
    while (i < hidden_vec4) {
        let idx = base_idx + i * 4u;
        let v0 = input[idx];
        let v1 = input[idx + 1u];
        let v2 = input[idx + 2u];
        let v3 = input[idx + 3u];
        local_sum_sq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
        i += WORKGROUP_SIZE;
    }

    // Handle remainder
    if (lid.x == 0u) {
        for (var r = hidden_vec4 * 4u; r < hidden_size; r++) {
            let val = input[base_idx + r];
            local_sum_sq += val * val;
        }
    }

    shared_sum_sq[lid.x] = local_sum_sq;
    workgroupBarrier();

    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_sum_sq[lid.x] += shared_sum_sq[lid.x + stride];
        }
        workgroupBarrier();
    }

    let rms = 1.0 / sqrt(shared_sum_sq[0] / f32(hidden_size) + eps);

    if (lid.x == 0u) {
        rms_out[batch_idx] = rms;
    }

    workgroupBarrier();

    // Vectorized normalization
    i = lid.x;
    while (i < hidden_vec4) {
        let idx = base_idx + i * 4u;
        let g_idx = i * 4u;
        output[idx] = input[idx] * rms * gamma[g_idx];
        output[idx + 1u] = input[idx + 1u] * rms * gamma[g_idx + 1u];
        output[idx + 2u] = input[idx + 2u] * rms * gamma[g_idx + 2u];
        output[idx + 3u] = input[idx + 3u] * rms * gamma[g_idx + 3u];
        i += WORKGROUP_SIZE;
    }

    // Handle remainder
    if (lid.x == 0u) {
        for (var r = hidden_vec4 * 4u; r < hidden_size; r++) {
            output[base_idx + r] = input[base_idx + r] * rms * gamma[r];
        }
    }
}

// ============================================================================
// RMS Norm Backward (for training)
// ============================================================================

struct RMSNormBackwardParams {
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
    _pad: u32,
}

@group(1) @binding(0) var<storage, read> grad_output: array<f32>;
@group(1) @binding(1) var<storage, read_write> grad_input: array<f32>;
@group(1) @binding(2) var<storage, read_write> grad_gamma: array<f32>;
@group(1) @binding(3) var<storage, read> saved_rms: array<f32>;
@group(1) @binding(4) var<uniform> backward_params: RMSNormBackwardParams;

var<workgroup> shared_dot: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(256)
fn rms_norm_backward(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_idx = wid.x;
    let hidden_size = backward_params.hidden_size;

    if (batch_idx >= backward_params.batch_size) {
        return;
    }

    let base_idx = batch_idx * hidden_size;
    let rms = saved_rms[batch_idx];

    // Compute dot product: sum(dy * x_normalized * gamma)
    var local_dot: f32 = 0.0;

    var i = lid.x;
    while (i < hidden_size) {
        let idx = base_idx + i;
        let dy = grad_output[idx];
        let x_norm = input[idx] * rms;  // x * rms = normalized x
        local_dot += dy * x_norm * gamma[i];
        i += WORKGROUP_SIZE;
    }

    shared_dot[lid.x] = local_dot;
    workgroupBarrier();

    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_dot[lid.x] += shared_dot[lid.x + stride];
        }
        workgroupBarrier();
    }

    let total_dot = shared_dot[0];
    let inv_n = 1.0 / f32(hidden_size);
    workgroupBarrier();

    // Compute gradient for input
    // dx = rms * (gamma * dy - x_norm * mean(x_norm * gamma * dy))
    i = lid.x;
    while (i < hidden_size) {
        let idx = base_idx + i;
        let dy = grad_output[idx];
        let x_norm = input[idx] * rms;

        let dx = rms * (gamma[i] * dy - x_norm * total_dot * inv_n);
        grad_input[idx] = dx;

        i += WORKGROUP_SIZE;
    }
}

// ============================================================================
// Simple RMS Norm - One thread per element (for very small hidden sizes)
// ============================================================================

@compute @workgroup_size(256)
fn rms_norm_simple(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let hidden_size = params.hidden_size;
    let eps = params.eps;
    let batch_size = params.batch_size;

    let batch_idx = idx / hidden_size;
    let hidden_idx = idx % hidden_size;

    if (batch_idx >= batch_size) {
        return;
    }

    let base_idx = batch_idx * hidden_size;

    // Compute sum of squares (each thread does full reduction)
    var sum_sq: f32 = 0.0;
    for (var i = 0u; i < hidden_size; i++) {
        let val = input[base_idx + i];
        sum_sq += val * val;
    }

    let rms = 1.0 / sqrt(sum_sq / f32(hidden_size) + eps);
    output[idx] = input[idx] * rms * gamma[hidden_idx];
}
