// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Log-Sum-Exp (LSE) Kernels
// Computes: logsumexp(x) = log(sum(exp(x_i)))
//
// Numerically stable implementation:
//   LSE(x) = max(x) + log(sum(exp(x_i - max(x))))
//
// Used for:
// - Softmax normalization (log-domain)
// - Attention score computation
// - Probability distribution normalization
//
// Part of the Lux Network GPU acceleration library

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;
const NEG_INF: f32 = -3.402823466e+38;  // Approximate -inf for f32
const LOG_EPSILON: f32 = -87.33654475;   // log(FLT_MIN) approximately

// ============================================================================
// Parameter Structures
// ============================================================================

struct LogSumExpParams {
    batch_size: u32,    // Number of independent LSE computations
    dim: u32,           // Size of dimension to reduce
    stride: u32,        // Stride between elements
    keepdim: u32,       // Whether to keep dimension in output
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: LogSumExpParams;

// Additional output for max values (useful for softmax)
@group(0) @binding(3) var<storage, read_write> max_vals: array<f32>;

// Workgroup shared memory
var<workgroup> shared_max: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_sum: array<f32, WORKGROUP_SIZE>;

// ============================================================================
// Safe Exponential (Prevents Overflow)
// ============================================================================

fn safe_exp(x: f32) -> f32 {
    // Clamp to prevent overflow/underflow
    let clamped = clamp(x, -88.0, 88.0);
    return exp(clamped);
}

// ============================================================================
// Main LogSumExp Kernel - Row-wise Reduction
// Each workgroup computes LSE for one row
// ============================================================================

@compute @workgroup_size(256)
fn logsumexp(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_idx = wid.x;
    let dim = params.dim;
    let stride = params.stride;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let base_idx = batch_idx * dim * stride;

    // Step 1: Find maximum value for numerical stability
    var local_max: f32 = NEG_INF;

    var i = lid.x;
    while (i < dim) {
        let idx = base_idx + i * stride;
        local_max = max(local_max, input[idx]);
        i += WORKGROUP_SIZE;
    }

    // Reduce to find global max
    shared_max[lid.x] = local_max;
    workgroupBarrier();

    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            shared_max[lid.x] = max(shared_max[lid.x], shared_max[lid.x + s]);
        }
        workgroupBarrier();
    }

    let max_val = shared_max[0];

    // Store max value if output buffer is bound
    if (lid.x == 0u) {
        max_vals[batch_idx] = max_val;
    }

    workgroupBarrier();

    // Step 2: Compute sum of exp(x - max)
    var local_sum: f32 = 0.0;

    i = lid.x;
    while (i < dim) {
        let idx = base_idx + i * stride;
        local_sum += safe_exp(input[idx] - max_val);
        i += WORKGROUP_SIZE;
    }

    // Reduce to find total sum
    shared_sum[lid.x] = local_sum;
    workgroupBarrier();

    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            shared_sum[lid.x] += shared_sum[lid.x + s];
        }
        workgroupBarrier();
    }

    // Step 3: Compute final LSE = max + log(sum)
    if (lid.x == 0u) {
        let sum_exp = shared_sum[0];
        // Handle edge case: if sum is 0 (all -inf), return -inf
        if (sum_exp > 0.0) {
            output[batch_idx] = max_val + log(sum_exp);
        } else {
            output[batch_idx] = NEG_INF;
        }
    }
}

// ============================================================================
// LogSumExp with Subtraction (for log-softmax)
// Computes: x - logsumexp(x) = log_softmax(x)
// ============================================================================

@compute @workgroup_size(256)
fn log_softmax(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_idx = wid.x;
    let dim = params.dim;
    let stride = params.stride;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let base_idx = batch_idx * dim * stride;

    // Step 1: Find maximum
    var local_max: f32 = NEG_INF;

    var i = lid.x;
    while (i < dim) {
        let idx = base_idx + i * stride;
        local_max = max(local_max, input[idx]);
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

    // Step 2: Compute sum of exp(x - max)
    var local_sum: f32 = 0.0;

    i = lid.x;
    while (i < dim) {
        let idx = base_idx + i * stride;
        local_sum += safe_exp(input[idx] - max_val);
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

    let log_sum_exp = log(shared_sum[0]);
    workgroupBarrier();

    // Step 3: Compute log_softmax = x - max - log(sum_exp)
    i = lid.x;
    while (i < dim) {
        let idx = base_idx + i * stride;
        output[idx] = input[idx] - max_val - log_sum_exp;
        i += WORKGROUP_SIZE;
    }
}

// ============================================================================
// Batched 2D LogSumExp (for attention scores)
// Reduces along the last dimension of a 2D tensor
// ============================================================================

struct LogSumExp2DParams {
    M: u32,             // First dimension (batch * heads)
    N: u32,             // Second dimension (sequence length to reduce)
    _pad0: u32,
    _pad1: u32,
}

@group(1) @binding(0) var<uniform> params_2d: LogSumExp2DParams;

@compute @workgroup_size(256)
fn logsumexp_2d(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let row = wid.x;
    let N = params_2d.N;

    if (row >= params_2d.M) {
        return;
    }

    let base_idx = row * N;

    // Find max
    var local_max: f32 = NEG_INF;

    var j = lid.x;
    while (j < N) {
        local_max = max(local_max, input[base_idx + j]);
        j += WORKGROUP_SIZE;
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

    // Compute sum of exp
    var local_sum: f32 = 0.0;

    j = lid.x;
    while (j < N) {
        local_sum += safe_exp(input[base_idx + j] - max_val);
        j += WORKGROUP_SIZE;
    }

    shared_sum[lid.x] = local_sum;
    workgroupBarrier();

    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            shared_sum[lid.x] += shared_sum[lid.x + s];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        output[row] = max_val + log(shared_sum[0]);
    }
}

// ============================================================================
// Online LogSumExp (for streaming computation)
// Computes LSE incrementally without storing all values
// ============================================================================

struct OnlineLSEState {
    max_val: f32,
    sum_exp: f32,
}

fn online_lse_init() -> OnlineLSEState {
    return OnlineLSEState(NEG_INF, 0.0);
}

fn online_lse_update(state: OnlineLSEState, x: f32) -> OnlineLSEState {
    if (x > state.max_val) {
        // New maximum: rescale existing sum
        let scale = safe_exp(state.max_val - x);
        return OnlineLSEState(x, state.sum_exp * scale + 1.0);
    } else {
        // Add to sum with current scale
        return OnlineLSEState(state.max_val, state.sum_exp + safe_exp(x - state.max_val));
    }
}

fn online_lse_combine(a: OnlineLSEState, b: OnlineLSEState) -> OnlineLSEState {
    if (a.max_val > b.max_val) {
        let scale = safe_exp(b.max_val - a.max_val);
        return OnlineLSEState(a.max_val, a.sum_exp + b.sum_exp * scale);
    } else {
        let scale = safe_exp(a.max_val - b.max_val);
        return OnlineLSEState(b.max_val, a.sum_exp * scale + b.sum_exp);
    }
}

fn online_lse_finalize(state: OnlineLSEState) -> f32 {
    if (state.sum_exp > 0.0) {
        return state.max_val + log(state.sum_exp);
    }
    return NEG_INF;
}

var<workgroup> shared_state_max: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_state_sum: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(256)
fn logsumexp_online(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_idx = wid.x;
    let dim = params.dim;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let base_idx = batch_idx * dim;

    // Each thread computes local LSE state
    var state = online_lse_init();

    var i = lid.x;
    while (i < dim) {
        state = online_lse_update(state, input[base_idx + i]);
        i += WORKGROUP_SIZE;
    }

    // Store state for reduction
    shared_state_max[lid.x] = state.max_val;
    shared_state_sum[lid.x] = state.sum_exp;
    workgroupBarrier();

    // Reduce online LSE states
    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            let a = OnlineLSEState(shared_state_max[lid.x], shared_state_sum[lid.x]);
            let b = OnlineLSEState(shared_state_max[lid.x + s], shared_state_sum[lid.x + s]);
            let combined = online_lse_combine(a, b);
            shared_state_max[lid.x] = combined.max_val;
            shared_state_sum[lid.x] = combined.sum_exp;
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        let final_state = OnlineLSEState(shared_state_max[0], shared_state_sum[0]);
        output[batch_idx] = online_lse_finalize(final_state);
        max_vals[batch_idx] = final_state.max_val;
    }
}

// ============================================================================
// Vectorized LogSumExp (vec4 loads)
// ============================================================================

@compute @workgroup_size(256)
fn logsumexp_vec4(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_idx = wid.x;
    let dim = params.dim;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let base_idx = batch_idx * dim;
    let dim_vec4 = dim / 4u;
    let dim_remainder = dim % 4u;

    // Step 1: Find max using vec4
    var local_max: f32 = NEG_INF;

    var i = lid.x;
    while (i < dim_vec4) {
        let idx = base_idx + i * 4u;
        let v0 = input[idx];
        let v1 = input[idx + 1u];
        let v2 = input[idx + 2u];
        let v3 = input[idx + 3u];
        local_max = max(local_max, max(max(v0, v1), max(v2, v3)));
        i += WORKGROUP_SIZE;
    }

    // Handle remainder
    if (lid.x == 0u) {
        for (var r = dim_vec4 * 4u; r < dim; r++) {
            local_max = max(local_max, input[base_idx + r]);
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

    // Step 2: Sum exp using vec4
    var local_sum: f32 = 0.0;

    i = lid.x;
    while (i < dim_vec4) {
        let idx = base_idx + i * 4u;
        local_sum += safe_exp(input[idx] - max_val);
        local_sum += safe_exp(input[idx + 1u] - max_val);
        local_sum += safe_exp(input[idx + 2u] - max_val);
        local_sum += safe_exp(input[idx + 3u] - max_val);
        i += WORKGROUP_SIZE;
    }

    if (lid.x == 0u) {
        for (var r = dim_vec4 * 4u; r < dim; r++) {
            local_sum += safe_exp(input[base_idx + r] - max_val);
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

    if (lid.x == 0u) {
        output[batch_idx] = max_val + log(shared_sum[0]);
        max_vals[batch_idx] = max_val;
    }
}
