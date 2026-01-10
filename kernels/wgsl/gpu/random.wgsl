// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Random Number Generation Kernels - PCG-based PRNG
// Implements PCG (Permuted Congruential Generator) for high-quality
// parallel random number generation on GPU.
// Supports uniform, normal, and other distributions.
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

// ============================================================================
// Constants
// ============================================================================

// PCG constants (PCG-XSH-RR variant)
const PCG_MULTIPLIER: u32 = 747796405u;
const PCG_INCREMENT: u32 = 2891336453u;

// For Box-Muller transform
const TWO_PI: f32 = 6.28318530717958647693f;

// ============================================================================
// Parameter Structure
// ============================================================================

struct RandomParams {
    size: u32,           // Number of random numbers to generate
    seed: u32,           // Base seed
    stream: u32,         // Stream offset for parallel generation
    _pad: u32,
}

struct NormalParams {
    size: u32,
    seed: u32,
    mean: f32,           // Mean of normal distribution
    stddev: f32,         // Standard deviation
}

struct UniformParams {
    size: u32,
    seed: u32,
    low: f32,            // Lower bound
    high: f32,           // Upper bound
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read_write> state: array<u32>;
@group(0) @binding(2) var<uniform> params: RandomParams;

// ============================================================================
// PCG Core Functions
// ============================================================================

// PCG step function - advance state and return output
fn pcg_step(s: ptr<function, u32>) -> u32 {
    let old_state = *s;
    // Advance state
    *s = old_state * PCG_MULTIPLIER + PCG_INCREMENT;
    // XSH-RR output function
    let word = ((old_state >> ((old_state >> 28u) + 4u)) ^ old_state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Initialize PCG state from seed and stream
fn pcg_init(seed: u32, stream: u32) -> u32 {
    var state = 0u;
    state = state * PCG_MULTIPLIER + (stream | 1u);
    state = state * PCG_MULTIPLIER + (stream | 1u);
    state = state + seed;
    state = state * PCG_MULTIPLIER + (stream | 1u);
    return state;
}

// Convert u32 to uniform f32 in [0, 1)
fn u32_to_f32_01(x: u32) -> f32 {
    // Use high 24 bits for f32 precision
    return f32(x >> 8u) * (1.0f / 16777216.0f);
}

// Convert u32 to uniform f32 in (0, 1]
fn u32_to_f32_open_01(x: u32) -> f32 {
    return (f32(x >> 8u) + 1.0f) * (1.0f / 16777216.0f);
}

// ============================================================================
// Uniform Random [0, 1)
// ============================================================================

@compute @workgroup_size(256)
fn random_uniform_01(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    // Initialize per-thread state
    var s = pcg_init(params.seed, params.stream + idx);

    // Generate random number
    let r = pcg_step(&s);
    output[idx] = u32_to_f32_01(r);

    // Save state for continuation
    state[idx] = s;
}

// ============================================================================
// Uniform Random in Range [low, high)
// ============================================================================

@group(0) @binding(3) var<uniform> uniform_params: UniformParams;

@compute @workgroup_size(256)
fn random_uniform(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniform_params.size) { return; }

    var s = pcg_init(uniform_params.seed, idx);
    let r = pcg_step(&s);
    let u = u32_to_f32_01(r);

    output[idx] = uniform_params.low + u * (uniform_params.high - uniform_params.low);
    state[idx] = s;
}

// ============================================================================
// Normal Distribution (Box-Muller Transform)
// ============================================================================

@group(0) @binding(4) var<uniform> normal_params: NormalParams;

@compute @workgroup_size(256)
fn random_normal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= normal_params.size) { return; }

    var s = pcg_init(normal_params.seed, idx);

    // Box-Muller transform requires two uniform random numbers
    let r1 = pcg_step(&s);
    let r2 = pcg_step(&s);

    let u1 = u32_to_f32_open_01(r1);  // (0, 1] to avoid log(0)
    let u2 = u32_to_f32_01(r2);       // [0, 1)

    // Box-Muller
    let mag = sqrt(-2.0f * log(u1));
    let z0 = mag * cos(TWO_PI * u2);

    output[idx] = normal_params.mean + z0 * normal_params.stddev;
    state[idx] = s;
}

// ============================================================================
// Normal Distribution - Paired output (both Box-Muller outputs)
// More efficient when generating many normals
// ============================================================================

@group(0) @binding(5) var<storage, read_write> output2: array<f32>;

@compute @workgroup_size(256)
fn random_normal_paired(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= normal_params.size / 2u) { return; }

    var s = pcg_init(normal_params.seed, idx);

    let r1 = pcg_step(&s);
    let r2 = pcg_step(&s);

    let u1 = u32_to_f32_open_01(r1);
    let u2 = u32_to_f32_01(r2);

    let mag = sqrt(-2.0f * log(u1));
    let z0 = mag * cos(TWO_PI * u2);
    let z1 = mag * sin(TWO_PI * u2);

    let idx2 = idx * 2u;
    output[idx2] = normal_params.mean + z0 * normal_params.stddev;
    output[idx2 + 1u] = normal_params.mean + z1 * normal_params.stddev;
    state[idx] = s;
}

// ============================================================================
// Exponential Distribution
// X ~ Exp(lambda): f(x) = lambda * exp(-lambda * x)
// ============================================================================

struct ExponentialParams {
    size: u32,
    seed: u32,
    lambda: f32,         // Rate parameter (1/mean)
    _pad: u32,
}

@group(0) @binding(6) var<uniform> exp_params: ExponentialParams;

@compute @workgroup_size(256)
fn random_exponential(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= exp_params.size) { return; }

    var s = pcg_init(exp_params.seed, idx);
    let r = pcg_step(&s);
    let u = u32_to_f32_open_01(r);  // (0, 1] to avoid log(0)

    output[idx] = -log(u) / exp_params.lambda;
    state[idx] = s;
}

// ============================================================================
// Bernoulli Distribution (0 or 1 with probability p)
// ============================================================================

struct BernoulliParams {
    size: u32,
    seed: u32,
    p: f32,              // Probability of 1
    _pad: u32,
}

@group(0) @binding(7) var<uniform> bernoulli_params: BernoulliParams;

@compute @workgroup_size(256)
fn random_bernoulli(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bernoulli_params.size) { return; }

    var s = pcg_init(bernoulli_params.seed, idx);
    let r = pcg_step(&s);
    let u = u32_to_f32_01(r);

    output[idx] = select(0.0f, 1.0f, u < bernoulli_params.p);
    state[idx] = s;
}

// ============================================================================
// Random Integers in Range [0, n)
// ============================================================================

struct IntParams {
    size: u32,
    seed: u32,
    n: u32,              // Upper bound (exclusive)
    _pad: u32,
}

@group(0) @binding(8) var<uniform> int_params: IntParams;
@group(0) @binding(9) var<storage, read_write> output_u32: array<u32>;

@compute @workgroup_size(256)
fn random_int(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= int_params.size) { return; }

    var s = pcg_init(int_params.seed, idx);
    let r = pcg_step(&s);

    // Unbiased bounded random using rejection sampling
    // For simplicity, use modulo (slight bias for non-power-of-2)
    output_u32[idx] = r % int_params.n;
    state[idx] = s;
}

// ============================================================================
// Random Permutation (Fisher-Yates shuffle indices)
// Generates indices for a random permutation
// ============================================================================

@compute @workgroup_size(256)
fn random_permutation_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    // Initialize with identity permutation
    output_u32[idx] = idx;
}

// Note: Full Fisher-Yates requires sequential processing
// This kernel does one step of the shuffle
struct ShuffleParams {
    size: u32,
    seed: u32,
    step: u32,           // Current step in shuffle
    _pad: u32,
}

@group(0) @binding(10) var<uniform> shuffle_params: ShuffleParams;

@compute @workgroup_size(1)
fn random_permutation_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = shuffle_params.step;
    if (i >= shuffle_params.size - 1u) { return; }

    var s = pcg_init(shuffle_params.seed, i);
    let r = pcg_step(&s);

    // Pick random index from [i, size)
    let remaining = shuffle_params.size - i;
    let j = i + (r % remaining);

    // Swap
    let temp = output_u32[i];
    output_u32[i] = output_u32[j];
    output_u32[j] = temp;
}

// ============================================================================
// Truncated Normal Distribution
// Normal distribution clamped to [low, high]
// ============================================================================

struct TruncatedNormalParams {
    size: u32,
    seed: u32,
    mean: f32,
    stddev: f32,
    low: f32,
    high: f32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(11) var<uniform> trunc_normal_params: TruncatedNormalParams;

@compute @workgroup_size(256)
fn random_truncated_normal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= trunc_normal_params.size) { return; }

    var s = pcg_init(trunc_normal_params.seed, idx);

    // Rejection sampling
    var result = 0.0f;
    for (var attempt = 0u; attempt < 100u; attempt = attempt + 1u) {
        let r1 = pcg_step(&s);
        let r2 = pcg_step(&s);

        let u1 = u32_to_f32_open_01(r1);
        let u2 = u32_to_f32_01(r2);

        let mag = sqrt(-2.0f * log(u1));
        let z = mag * cos(TWO_PI * u2);

        result = trunc_normal_params.mean + z * trunc_normal_params.stddev;

        if (result >= trunc_normal_params.low && result <= trunc_normal_params.high) {
            break;
        }
    }

    output[idx] = clamp(result, trunc_normal_params.low, trunc_normal_params.high);
    state[idx] = s;
}

// ============================================================================
// Dropout Mask Generation
// Generates 0/1 mask for dropout with probability p of being 0
// ============================================================================

struct DropoutParams {
    size: u32,
    seed: u32,
    p: f32,              // Probability of dropping (setting to 0)
    scale: f32,          // Scale factor for remaining values (1/(1-p))
}

@group(0) @binding(12) var<uniform> dropout_params: DropoutParams;

@compute @workgroup_size(256)
fn random_dropout_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= dropout_params.size) { return; }

    var s = pcg_init(dropout_params.seed, idx);
    let r = pcg_step(&s);
    let u = u32_to_f32_01(r);

    // Output scale factor if kept, 0 if dropped
    output[idx] = select(dropout_params.scale, 0.0f, u < dropout_params.p);
    state[idx] = s;
}

// ============================================================================
// Categorical/Multinomial Sampling
// Sample from discrete distribution given cumulative probabilities
// ============================================================================

@group(0) @binding(13) var<storage, read> cumprobs: array<f32>;  // Cumulative probabilities

struct CategoricalParams {
    size: u32,           // Number of samples
    seed: u32,
    num_categories: u32, // Number of categories
    _pad: u32,
}

@group(0) @binding(14) var<uniform> cat_params: CategoricalParams;

@compute @workgroup_size(256)
fn random_categorical(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= cat_params.size) { return; }

    var s = pcg_init(cat_params.seed, idx);
    let r = pcg_step(&s);
    let u = u32_to_f32_01(r);

    // Binary search for category
    var low = 0u;
    var high = cat_params.num_categories;

    while (low < high) {
        let mid = (low + high) / 2u;
        if (cumprobs[mid] < u) {
            low = mid + 1u;
        } else {
            high = mid;
        }
    }

    output_u32[idx] = low;
    state[idx] = s;
}

// ============================================================================
// Stateful Random - Continue from saved state
// ============================================================================

@compute @workgroup_size(256)
fn random_continue(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    // Load saved state
    var s = state[idx];

    // Generate next random number
    let r = pcg_step(&s);
    output[idx] = u32_to_f32_01(r);

    // Save updated state
    state[idx] = s;
}

// ============================================================================
// Batch State Initialization
// ============================================================================

@compute @workgroup_size(256)
fn random_init_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    state[idx] = pcg_init(params.seed, params.stream + idx);
}
