// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Softmax Kernels
// Implements: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//
// Numerically stable implementation using the max-subtraction trick
// Supports:
// - Row-wise softmax (default)
// - Temperature scaling
// - Masked softmax (for attention)
// - Online softmax (Flash Attention style)
//
// Part of the Lux Network GPU acceleration library

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;
const NEG_INF: f32 = -3.402823466e+38;

// ============================================================================
// Parameter Structures
// ============================================================================

struct SoftmaxParams {
    batch_size: u32,    // Number of rows
    dim: u32,           // Size of dimension to softmax over
    temperature: f32,   // Temperature for scaling (1.0 = no scaling)
    _pad: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: SoftmaxParams;

// Optional: for masked softmax
@group(0) @binding(3) var<storage, read> mask: array<u32>;  // Bit-packed mask

// Workgroup shared memory
var<workgroup> shared_max: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_sum: array<f32, WORKGROUP_SIZE>;

// ============================================================================
// Safe Exponential
// ============================================================================

fn safe_exp(x: f32) -> f32 {
    return exp(clamp(x, -88.0, 88.0));
}

// ============================================================================
// Standard Softmax - Each workgroup processes one row
// ============================================================================

@compute @workgroup_size(256)
fn softmax(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let row = wid.x;
    let dim = params.dim;
    let temperature = params.temperature;

    if (row >= params.batch_size) {
        return;
    }

    let base_idx = row * dim;

    // Step 1: Find maximum value
    var local_max: f32 = NEG_INF;

    var i = lid.x;
    while (i < dim) {
        let val = input[base_idx + i] / temperature;
        local_max = max(local_max, val);
        i += WORKGROUP_SIZE;
    }

    shared_max[lid.x] = local_max;
    workgroupBarrier();

    // Parallel reduction for max
    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            shared_max[lid.x] = max(shared_max[lid.x], shared_max[lid.x + s]);
        }
        workgroupBarrier();
    }

    let max_val = shared_max[0];
    workgroupBarrier();

    // Step 2: Compute sum of exp(x - max)
    var local_sum: f32 = 0.0;

    i = lid.x;
    while (i < dim) {
        let val = input[base_idx + i] / temperature;
        local_sum += safe_exp(val - max_val);
        i += WORKGROUP_SIZE;
    }

    shared_sum[lid.x] = local_sum;
    workgroupBarrier();

    // Parallel reduction for sum
    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            shared_sum[lid.x] += shared_sum[lid.x + s];
        }
        workgroupBarrier();
    }

    let sum_exp = shared_sum[0];
    let inv_sum = 1.0 / sum_exp;
    workgroupBarrier();

    // Step 3: Normalize
    i = lid.x;
    while (i < dim) {
        let val = input[base_idx + i] / temperature;
        output[base_idx + i] = safe_exp(val - max_val) * inv_sum;
        i += WORKGROUP_SIZE;
    }
}

// ============================================================================
// In-place Softmax
// ============================================================================

@group(0) @binding(4) var<storage, read_write> inout: array<f32>;

@compute @workgroup_size(256)
fn softmax_inplace(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let row = wid.x;
    let dim = params.dim;
    let temperature = params.temperature;

    if (row >= params.batch_size) {
        return;
    }

    let base_idx = row * dim;

    // Find max
    var local_max: f32 = NEG_INF;

    var i = lid.x;
    while (i < dim) {
        local_max = max(local_max, inout[base_idx + i] / temperature);
        i += WORKGROUP_SIZE;
    }

    shared_max[lid.x] = local_max;
    workgroupBarrier();

    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            shared_max[lid.x] = max(shared_max[lid.x], shared_max[lid.x + s]);
        }
        workgroupBarrier();
    }

    let max_val = shared_max[0];
    workgroupBarrier();

    // Compute exp and sum
    var local_sum: f32 = 0.0;

    i = lid.x;
    while (i < dim) {
        let idx = base_idx + i;
        let exp_val = safe_exp(inout[idx] / temperature - max_val);
        inout[idx] = exp_val;  // Store exp temporarily
        local_sum += exp_val;
        i += WORKGROUP_SIZE;
    }

    shared_sum[lid.x] = local_sum;
    workgroupBarrier();

    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            shared_sum[lid.x] += shared_sum[lid.x + s];
        }
        workgroupBarrier();
    }

    let inv_sum = 1.0 / shared_sum[0];
    workgroupBarrier();

    // Normalize
    i = lid.x;
    while (i < dim) {
        let idx = base_idx + i;
        inout[idx] = inout[idx] * inv_sum;
        i += WORKGROUP_SIZE;
    }
}

// ============================================================================
// Masked Softmax (for causal attention)
// ============================================================================

struct MaskedSoftmaxParams {
    batch_size: u32,
    seq_len: u32,       // Query length (rows)
    key_len: u32,       // Key length (cols)
    temperature: f32,
    mask_value: f32,    // Value for masked positions (-inf)
    causal: u32,        // 1 = causal mask, 0 = use provided mask
    _pad0: u32,
    _pad1: u32,
}

@group(1) @binding(0) var<uniform> masked_params: MaskedSoftmaxParams;

@compute @workgroup_size(256)
fn softmax_causal(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_idx = wid.x / masked_params.seq_len;
    let query_idx = wid.x % masked_params.seq_len;
    let key_len = masked_params.key_len;
    let temperature = masked_params.temperature;

    if (batch_idx >= masked_params.batch_size) {
        return;
    }

    let base_idx = wid.x * key_len;

    // Causal mask: only attend to positions <= query_idx
    let max_valid_key = query_idx + 1u;

    // Find max (only over valid positions)
    var local_max: f32 = NEG_INF;

    var i = lid.x;
    while (i < max_valid_key && i < key_len) {
        local_max = max(local_max, input[base_idx + i] / temperature);
        i += WORKGROUP_SIZE;
    }

    shared_max[lid.x] = local_max;
    workgroupBarrier();

    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            shared_max[lid.x] = max(shared_max[lid.x], shared_max[lid.x + s]);
        }
        workgroupBarrier();
    }

    let max_val = shared_max[0];
    workgroupBarrier();

    // Sum exp (only valid positions)
    var local_sum: f32 = 0.0;

    i = lid.x;
    while (i < max_valid_key && i < key_len) {
        local_sum += safe_exp(input[base_idx + i] / temperature - max_val);
        i += WORKGROUP_SIZE;
    }

    shared_sum[lid.x] = local_sum;
    workgroupBarrier();

    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            shared_sum[lid.x] += shared_sum[lid.x + s];
        }
        workgroupBarrier();
    }

    let inv_sum = select(0.0, 1.0 / shared_sum[0], shared_sum[0] > 0.0);
    workgroupBarrier();

    // Write output: softmax for valid, 0 for masked
    i = lid.x;
    while (i < key_len) {
        let idx = base_idx + i;
        if (i < max_valid_key) {
            output[idx] = safe_exp(input[idx] / temperature - max_val) * inv_sum;
        } else {
            output[idx] = 0.0;
        }
        i += WORKGROUP_SIZE;
    }
}

// ============================================================================
// Online Softmax (Flash Attention style - single pass)
// Computes softmax in a streaming fashion
// ============================================================================

struct OnlineSoftmaxState {
    max_val: f32,
    sum_exp: f32,
}

fn online_softmax_init() -> OnlineSoftmaxState {
    return OnlineSoftmaxState(NEG_INF, 0.0);
}

fn online_softmax_update(state: OnlineSoftmaxState, x: f32) -> OnlineSoftmaxState {
    if (x > state.max_val) {
        let scale = safe_exp(state.max_val - x);
        return OnlineSoftmaxState(x, state.sum_exp * scale + 1.0);
    } else {
        return OnlineSoftmaxState(state.max_val, state.sum_exp + safe_exp(x - state.max_val));
    }
}

fn online_softmax_combine(a: OnlineSoftmaxState, b: OnlineSoftmaxState) -> OnlineSoftmaxState {
    if (a.max_val > b.max_val) {
        let scale = safe_exp(b.max_val - a.max_val);
        return OnlineSoftmaxState(a.max_val, a.sum_exp + b.sum_exp * scale);
    } else {
        let scale = safe_exp(a.max_val - b.max_val);
        return OnlineSoftmaxState(b.max_val, a.sum_exp * scale + b.sum_exp);
    }
}

var<workgroup> shared_state_max: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_state_sum: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(256)
fn softmax_online(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let row = wid.x;
    let dim = params.dim;
    let temperature = params.temperature;

    if (row >= params.batch_size) {
        return;
    }

    let base_idx = row * dim;

    // Single pass: compute max and sum simultaneously
    var state = online_softmax_init();

    var i = lid.x;
    while (i < dim) {
        let val = input[base_idx + i] / temperature;
        state = online_softmax_update(state, val);
        i += WORKGROUP_SIZE;
    }

    shared_state_max[lid.x] = state.max_val;
    shared_state_sum[lid.x] = state.sum_exp;
    workgroupBarrier();

    // Combine states
    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            let a = OnlineSoftmaxState(shared_state_max[lid.x], shared_state_sum[lid.x]);
            let b = OnlineSoftmaxState(shared_state_max[lid.x + s], shared_state_sum[lid.x + s]);
            let combined = online_softmax_combine(a, b);
            shared_state_max[lid.x] = combined.max_val;
            shared_state_sum[lid.x] = combined.sum_exp;
        }
        workgroupBarrier();
    }

    let max_val = shared_state_max[0];
    let inv_sum = 1.0 / shared_state_sum[0];
    workgroupBarrier();

    // Second pass: normalize
    i = lid.x;
    while (i < dim) {
        let val = input[base_idx + i] / temperature;
        output[base_idx + i] = safe_exp(val - max_val) * inv_sum;
        i += WORKGROUP_SIZE;
    }
}

// ============================================================================
// Softmax Backward (for training)
// ============================================================================

@group(2) @binding(0) var<storage, read> softmax_output: array<f32>;
@group(2) @binding(1) var<storage, read> grad_output: array<f32>;
@group(2) @binding(2) var<storage, read_write> grad_input: array<f32>;

@compute @workgroup_size(256)
fn softmax_backward(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let row = wid.x;
    let dim = params.dim;

    if (row >= params.batch_size) {
        return;
    }

    let base_idx = row * dim;

    // Compute dot product: sum(y * dy)
    var local_dot: f32 = 0.0;

    var i = lid.x;
    while (i < dim) {
        let idx = base_idx + i;
        local_dot += softmax_output[idx] * grad_output[idx];
        i += WORKGROUP_SIZE;
    }

    shared_sum[lid.x] = local_dot;
    workgroupBarrier();

    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            shared_sum[lid.x] += shared_sum[lid.x + s];
        }
        workgroupBarrier();
    }

    let dot_product = shared_sum[0];
    workgroupBarrier();

    // Compute gradient: dx = y * (dy - dot_product)
    i = lid.x;
    while (i < dim) {
        let idx = base_idx + i;
        let y = softmax_output[idx];
        let dy = grad_output[idx];
        grad_input[idx] = y * (dy - dot_product);
        i += WORKGROUP_SIZE;
    }
}

// ============================================================================
// Vectorized Softmax (vec4 for better memory bandwidth)
// ============================================================================

@compute @workgroup_size(256)
fn softmax_vec4(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let row = wid.x;
    let dim = params.dim;
    let temperature = params.temperature;

    if (row >= params.batch_size) {
        return;
    }

    let base_idx = row * dim;
    let dim_vec4 = dim / 4u;

    // Find max using vec4
    var local_max: f32 = NEG_INF;

    var i = lid.x;
    while (i < dim_vec4) {
        let idx = base_idx + i * 4u;
        let v0 = input[idx] / temperature;
        let v1 = input[idx + 1u] / temperature;
        let v2 = input[idx + 2u] / temperature;
        let v3 = input[idx + 3u] / temperature;
        local_max = max(local_max, max(max(v0, v1), max(v2, v3)));
        i += WORKGROUP_SIZE;
    }

    // Handle remainder
    if (lid.x == 0u) {
        for (var r = dim_vec4 * 4u; r < dim; r++) {
            local_max = max(local_max, input[base_idx + r] / temperature);
        }
    }

    shared_max[lid.x] = local_max;
    workgroupBarrier();

    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            shared_max[lid.x] = max(shared_max[lid.x], shared_max[lid.x + s]);
        }
        workgroupBarrier();
    }

    let max_val = shared_max[0];
    workgroupBarrier();

    // Sum exp using vec4
    var local_sum: f32 = 0.0;

    i = lid.x;
    while (i < dim_vec4) {
        let idx = base_idx + i * 4u;
        local_sum += safe_exp(input[idx] / temperature - max_val);
        local_sum += safe_exp(input[idx + 1u] / temperature - max_val);
        local_sum += safe_exp(input[idx + 2u] / temperature - max_val);
        local_sum += safe_exp(input[idx + 3u] / temperature - max_val);
        i += WORKGROUP_SIZE;
    }

    if (lid.x == 0u) {
        for (var r = dim_vec4 * 4u; r < dim; r++) {
            local_sum += safe_exp(input[base_idx + r] / temperature - max_val);
        }
    }

    shared_sum[lid.x] = local_sum;
    workgroupBarrier();

    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            shared_sum[lid.x] += shared_sum[lid.x + s];
        }
        workgroupBarrier();
    }

    let inv_sum = 1.0 / shared_sum[0];
    workgroupBarrier();

    // Write output
    i = lid.x;
    while (i < dim_vec4) {
        let idx = base_idx + i * 4u;
        output[idx] = safe_exp(input[idx] / temperature - max_val) * inv_sum;
        output[idx + 1u] = safe_exp(input[idx + 1u] / temperature - max_val) * inv_sum;
        output[idx + 2u] = safe_exp(input[idx + 2u] / temperature - max_val) * inv_sum;
        output[idx + 3u] = safe_exp(input[idx + 3u] / temperature - max_val) * inv_sum;
        i += WORKGROUP_SIZE;
    }

    if (lid.x == 0u) {
        for (var r = dim_vec4 * 4u; r < dim; r++) {
            output[base_idx + r] = safe_exp(input[base_idx + r] / temperature - max_val) * inv_sum;
        }
    }
}

// ============================================================================
// Simple Softmax - One thread per row (for small dimensions)
// ============================================================================

@compute @workgroup_size(256)
fn softmax_simple(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let row = gid.x;
    let dim = params.dim;
    let temperature = params.temperature;

    if (row >= params.batch_size) {
        return;
    }

    let base_idx = row * dim;

    // Find max
    var max_val: f32 = NEG_INF;
    for (var i = 0u; i < dim; i++) {
        max_val = max(max_val, input[base_idx + i] / temperature);
    }

    // Sum exp
    var sum_exp: f32 = 0.0;
    for (var i = 0u; i < dim; i++) {
        sum_exp += safe_exp(input[base_idx + i] / temperature - max_val);
    }

    // Normalize
    let inv_sum = 1.0 / sum_exp;
    for (var i = 0u; i < dim; i++) {
        output[base_idx + i] = safe_exp(input[base_idx + i] / temperature - max_val) * inv_sum;
    }
}
