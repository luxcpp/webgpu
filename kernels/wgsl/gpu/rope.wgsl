// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Rotary Position Embedding (RoPE) Kernels
//
// Implements: x' = x * cos(m*theta) + rotate_half(x) * sin(m*theta)
//
// RoPE applies a rotation to pairs of dimensions based on position:
// - Uses complex number rotation in 2D subspaces
// - Position-dependent rotation angles
// - Supports various theta base frequencies (10000 for GPT, 500000 for Llama3)
//
// Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
//
// Part of the Lux Network GPU acceleration library

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;
const PI: f32 = 3.14159265358979323846;

// ============================================================================
// Parameter Structures
// ============================================================================

struct RoPEParams {
    batch_size: u32,        // Batch size
    seq_len: u32,           // Sequence length
    num_heads: u32,         // Number of attention heads
    head_dim: u32,          // Dimension per head (must be even)
    theta_base: f32,        // Base frequency (10000.0 for GPT, 500000.0 for Llama3)
    max_seq_len: u32,       // Maximum sequence length for precomputed freqs
    position_offset: u32,   // Offset for position indices (for KV cache)
    _pad: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read_write> input: array<f32>;  // [batch, seq, heads, dim]
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> cos_cache: array<f32>;    // Precomputed cos values
@group(0) @binding(3) var<storage, read> sin_cache: array<f32>;    // Precomputed sin values
@group(0) @binding(4) var<uniform> params: RoPEParams;

// Optional: position ids for non-contiguous positions
@group(0) @binding(5) var<storage, read> position_ids: array<u32>;

// ============================================================================
// Compute Frequency for a Given Dimension
// ============================================================================

fn compute_freq(dim_idx: u32, head_dim: u32, theta_base: f32) -> f32 {
    // theta = 1 / (theta_base^(2*i/d)) where i is the dimension index
    let exponent = -2.0 * f32(dim_idx) / f32(head_dim);
    return pow(theta_base, exponent);
}

// ============================================================================
// Main RoPE Kernel - Apply rotation with precomputed cos/sin
// ============================================================================

@compute @workgroup_size(256)
fn rope_forward(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_size = params.batch_size;
    let seq_len = params.seq_len;
    let num_heads = params.num_heads;
    let head_dim = params.head_dim;
    let position_offset = params.position_offset;

    // Compute indices
    let total_elements = batch_size * seq_len * num_heads * head_dim;
    if (gid.x >= total_elements / 2u) {
        return;  // Process pairs
    }

    // Decode position: each thread handles one pair of dimensions
    let pair_idx = gid.x;
    let dim_pair = pair_idx % (head_dim / 2u);
    let temp = pair_idx / (head_dim / 2u);
    let head_idx = temp % num_heads;
    let temp2 = temp / num_heads;
    let seq_idx = temp2 % seq_len;
    let batch_idx = temp2 / seq_len;

    // Get position (may be from position_ids or computed)
    let position = position_offset + seq_idx;

    // Get cos/sin from cache
    let cache_idx = position * (head_dim / 2u) + dim_pair;
    let cos_val = cos_cache[cache_idx];
    let sin_val = sin_cache[cache_idx];

    // Compute input indices
    let base_idx = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
    let idx0 = base_idx + dim_pair * 2u;
    let idx1 = idx0 + 1u;

    // Load pair of values
    let x0 = input[idx0];
    let x1 = input[idx1];

    // Apply rotation: (x0, x1) * (cos, sin) = (x0*cos - x1*sin, x0*sin + x1*cos)
    output[idx0] = x0 * cos_val - x1 * sin_val;
    output[idx1] = x0 * sin_val + x1 * cos_val;
}

// ============================================================================
// RoPE with On-the-fly Frequency Computation (no cache needed)
// ============================================================================

@compute @workgroup_size(256)
fn rope_forward_no_cache(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_size = params.batch_size;
    let seq_len = params.seq_len;
    let num_heads = params.num_heads;
    let head_dim = params.head_dim;
    let theta_base = params.theta_base;
    let position_offset = params.position_offset;

    let total_pairs = batch_size * seq_len * num_heads * (head_dim / 2u);
    if (gid.x >= total_pairs) {
        return;
    }

    // Decode position
    let pair_idx = gid.x;
    let dim_pair = pair_idx % (head_dim / 2u);
    let temp = pair_idx / (head_dim / 2u);
    let head_idx = temp % num_heads;
    let temp2 = temp / num_heads;
    let seq_idx = temp2 % seq_len;
    let batch_idx = temp2 / seq_len;

    let position = position_offset + seq_idx;

    // Compute frequency on the fly
    let freq = compute_freq(dim_pair, head_dim, theta_base);
    let angle = f32(position) * freq;
    let cos_val = cos(angle);
    let sin_val = sin(angle);

    // Apply rotation
    let base_idx = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
    let idx0 = base_idx + dim_pair * 2u;
    let idx1 = idx0 + 1u;

    let x0 = input[idx0];
    let x1 = input[idx1];

    output[idx0] = x0 * cos_val - x1 * sin_val;
    output[idx1] = x0 * sin_val + x1 * cos_val;
}

// ============================================================================
// RoPE In-place (modifies input directly)
// ============================================================================

@compute @workgroup_size(256)
fn rope_inplace(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_size = params.batch_size;
    let seq_len = params.seq_len;
    let num_heads = params.num_heads;
    let head_dim = params.head_dim;
    let position_offset = params.position_offset;

    let total_pairs = batch_size * seq_len * num_heads * (head_dim / 2u);
    if (gid.x >= total_pairs) {
        return;
    }

    let pair_idx = gid.x;
    let dim_pair = pair_idx % (head_dim / 2u);
    let temp = pair_idx / (head_dim / 2u);
    let head_idx = temp % num_heads;
    let temp2 = temp / num_heads;
    let seq_idx = temp2 % seq_len;
    let batch_idx = temp2 / seq_len;

    let position = position_offset + seq_idx;
    let cache_idx = position * (head_dim / 2u) + dim_pair;
    let cos_val = cos_cache[cache_idx];
    let sin_val = sin_cache[cache_idx];

    let base_idx = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
    let idx0 = base_idx + dim_pair * 2u;
    let idx1 = idx0 + 1u;

    let x0 = input[idx0];
    let x1 = input[idx1];

    input[idx0] = x0 * cos_val - x1 * sin_val;
    input[idx1] = x0 * sin_val + x1 * cos_val;
}

// ============================================================================
// RoPE Backward (for training)
// Gradient: dx = dy * cos - rotate_half(dy) * sin  (note the sign change)
// ============================================================================

@group(1) @binding(0) var<storage, read> grad_output: array<f32>;
@group(1) @binding(1) var<storage, read_write> grad_input: array<f32>;

@compute @workgroup_size(256)
fn rope_backward(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_size = params.batch_size;
    let seq_len = params.seq_len;
    let num_heads = params.num_heads;
    let head_dim = params.head_dim;
    let position_offset = params.position_offset;

    let total_pairs = batch_size * seq_len * num_heads * (head_dim / 2u);
    if (gid.x >= total_pairs) {
        return;
    }

    let pair_idx = gid.x;
    let dim_pair = pair_idx % (head_dim / 2u);
    let temp = pair_idx / (head_dim / 2u);
    let head_idx = temp % num_heads;
    let temp2 = temp / num_heads;
    let seq_idx = temp2 % seq_len;
    let batch_idx = temp2 / seq_len;

    let position = position_offset + seq_idx;
    let cache_idx = position * (head_dim / 2u) + dim_pair;
    let cos_val = cos_cache[cache_idx];
    let sin_val = sin_cache[cache_idx];

    let base_idx = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
    let idx0 = base_idx + dim_pair * 2u;
    let idx1 = idx0 + 1u;

    let dy0 = grad_output[idx0];
    let dy1 = grad_output[idx1];

    // Backward is the inverse rotation: rotate by -theta
    // cos(-theta) = cos(theta), sin(-theta) = -sin(theta)
    grad_input[idx0] = dy0 * cos_val + dy1 * sin_val;
    grad_input[idx1] = -dy0 * sin_val + dy1 * cos_val;
}

// ============================================================================
// Precompute Cos/Sin Cache
// ============================================================================

struct CacheParams {
    max_seq_len: u32,
    head_dim: u32,
    theta_base: f32,
    _pad: u32,
}

@group(2) @binding(0) var<storage, read_write> cos_out: array<f32>;
@group(2) @binding(1) var<storage, read_write> sin_out: array<f32>;
@group(2) @binding(2) var<uniform> cache_params: CacheParams;

@compute @workgroup_size(256)
fn rope_precompute_cache(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let max_seq_len = cache_params.max_seq_len;
    let head_dim = cache_params.head_dim;
    let theta_base = cache_params.theta_base;
    let half_dim = head_dim / 2u;

    let total = max_seq_len * half_dim;
    if (gid.x >= total) {
        return;
    }

    let pos = gid.x / half_dim;
    let dim_pair = gid.x % half_dim;

    let freq = compute_freq(dim_pair, head_dim, theta_base);
    let angle = f32(pos) * freq;

    cos_out[gid.x] = cos(angle);
    sin_out[gid.x] = sin(angle);
}

// ============================================================================
// RoPE with NTK-aware Scaling (for extended context length)
// Reference: Reddit post on NTK-aware interpolation
// ============================================================================

struct NTKRoPEParams {
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    theta_base: f32,
    scale_factor: f32,      // Context extension factor
    original_max_len: u32,  // Original max sequence length
    _pad: u32,
}

@group(3) @binding(0) var<uniform> ntk_params: NTKRoPEParams;

@compute @workgroup_size(256)
fn rope_ntk_scaled(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_size = ntk_params.batch_size;
    let seq_len = ntk_params.seq_len;
    let num_heads = ntk_params.num_heads;
    let head_dim = ntk_params.head_dim;
    let theta_base = ntk_params.theta_base;
    let scale_factor = ntk_params.scale_factor;

    let total_pairs = batch_size * seq_len * num_heads * (head_dim / 2u);
    if (gid.x >= total_pairs) {
        return;
    }

    let pair_idx = gid.x;
    let dim_pair = pair_idx % (head_dim / 2u);
    let temp = pair_idx / (head_dim / 2u);
    let head_idx = temp % num_heads;
    let temp2 = temp / num_heads;
    let seq_idx = temp2 % seq_len;
    let batch_idx = temp2 / seq_len;

    // NTK-aware scaling: scale the base frequency
    let scaled_theta = theta_base * pow(scale_factor, f32(head_dim) / (f32(head_dim) - 2.0));
    let freq = compute_freq(dim_pair, head_dim, scaled_theta);
    let angle = f32(seq_idx) * freq;

    let cos_val = cos(angle);
    let sin_val = sin(angle);

    let base_idx = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
    let idx0 = base_idx + dim_pair * 2u;
    let idx1 = idx0 + 1u;

    let x0 = input[idx0];
    let x1 = input[idx1];

    output[idx0] = x0 * cos_val - x1 * sin_val;
    output[idx1] = x0 * sin_val + x1 * cos_val;
}

// ============================================================================
// RoPE with Custom Position IDs (for sparse attention, etc.)
// ============================================================================

@compute @workgroup_size(256)
fn rope_custom_positions(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_size = params.batch_size;
    let seq_len = params.seq_len;
    let num_heads = params.num_heads;
    let head_dim = params.head_dim;

    let total_pairs = batch_size * seq_len * num_heads * (head_dim / 2u);
    if (gid.x >= total_pairs) {
        return;
    }

    let pair_idx = gid.x;
    let dim_pair = pair_idx % (head_dim / 2u);
    let temp = pair_idx / (head_dim / 2u);
    let head_idx = temp % num_heads;
    let temp2 = temp / num_heads;
    let seq_idx = temp2 % seq_len;
    let batch_idx = temp2 / seq_len;

    // Get position from position_ids array
    let position = position_ids[batch_idx * seq_len + seq_idx];

    let cache_idx = position * (head_dim / 2u) + dim_pair;
    let cos_val = cos_cache[cache_idx];
    let sin_val = sin_cache[cache_idx];

    let base_idx = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
    let idx0 = base_idx + dim_pair * 2u;
    let idx1 = idx0 + 1u;

    let x0 = input[idx0];
    let x1 = input[idx1];

    output[idx0] = x0 * cos_val - x1 * sin_val;
    output[idx1] = x0 * sin_val + x1 * cos_val;
}

// ============================================================================
// Fused RoPE for Q and K (common pattern in attention)
// ============================================================================

@group(4) @binding(0) var<storage, read_write> query: array<f32>;
@group(4) @binding(1) var<storage, read_write> key: array<f32>;

@compute @workgroup_size(256)
fn rope_qk_fused(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_size = params.batch_size;
    let seq_len = params.seq_len;
    let num_heads = params.num_heads;
    let head_dim = params.head_dim;
    let position_offset = params.position_offset;

    let total_pairs = batch_size * seq_len * num_heads * (head_dim / 2u);
    if (gid.x >= total_pairs) {
        return;
    }

    let pair_idx = gid.x;
    let dim_pair = pair_idx % (head_dim / 2u);
    let temp = pair_idx / (head_dim / 2u);
    let head_idx = temp % num_heads;
    let temp2 = temp / num_heads;
    let seq_idx = temp2 % seq_len;
    let batch_idx = temp2 / seq_len;

    let position = position_offset + seq_idx;
    let cache_idx = position * (head_dim / 2u) + dim_pair;
    let cos_val = cos_cache[cache_idx];
    let sin_val = sin_cache[cache_idx];

    let base_idx = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
    let idx0 = base_idx + dim_pair * 2u;
    let idx1 = idx0 + 1u;

    // Apply RoPE to query
    let q0 = query[idx0];
    let q1 = query[idx1];
    query[idx0] = q0 * cos_val - q1 * sin_val;
    query[idx1] = q0 * sin_val + q1 * cos_val;

    // Apply RoPE to key
    let k0 = key[idx0];
    let k1 = key[idx1];
    key[idx0] = k0 * cos_val - k1 * sin_val;
    key[idx1] = k0 * sin_val + k1 * cos_val;
}
