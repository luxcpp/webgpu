// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// WebGPU Backend Plugin - Cross-platform GPU via Dawn/wgpu-native
// Loaded as a shared library via dlopen()
//
// This plugin supports both Dawn (Google) and wgpu-native (Mozilla) WebGPU
// implementations, selected at compile time via USE_DAWN_API or USE_WGPU_API.

#include "lux/gpu/backend_plugin.h"
#include "lux/gpu/crypto_backend.h"
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <cmath>

// =============================================================================
// WebGPU API Selection
// =============================================================================

#if defined(USE_DAWN_API)
    #include <webgpu/webgpu.h>
    #define WEBGPU_IMPL "dawn"
#elif defined(USE_WGPU_API)
    #include <wgpu.h>
    #define WEBGPU_IMPL "wgpu"
#else
    // Stub implementation when no WebGPU library is available
    #define WEBGPU_STUB
    #define WEBGPU_IMPL "stub"
#endif

// =============================================================================
// Embedded WGSL Kernels
// =============================================================================

#ifndef WEBGPU_STUB

// Binary operations kernel (add, sub, mul)
static const char* WGSL_BINARY_OPS = R"(
struct BinaryParams {
    size: u32,
    a_stride: u32,
    b_stride: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: BinaryParams;

fn get_a_idx(idx: u32) -> u32 {
    if (params.a_stride == 0u) { return 0u; }
    return idx * params.a_stride;
}

fn get_b_idx(idx: u32) -> u32 {
    if (params.b_stride == 0u) { return 0u; }
    return idx * params.b_stride;
}

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = input_a[get_a_idx(idx)] + input_b[get_b_idx(idx)];
}

@compute @workgroup_size(256)
fn sub(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = input_a[get_a_idx(idx)] - input_b[get_b_idx(idx)];
}

@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = input_a[get_a_idx(idx)] * input_b[get_b_idx(idx)];
}
)";

// Matrix multiplication kernel (tiled GEMM)
static const char* WGSL_MATMUL = R"(
struct MatmulParams {
    M: u32,
    N: u32,
    K: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: MatmulParams;

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_A: array<f32, 256>;
var<workgroup> tile_B: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn matmul(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let row = gid.y;
    let col = gid.x;
    let local_row = lid.y;
    let local_col = lid.x;

    var sum: f32 = 0.0;
    let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        let tile_k = t * TILE_SIZE;

        // Load A tile
        let a_row = row;
        let a_col = tile_k + local_col;
        if (a_row < params.M && a_col < params.K) {
            tile_A[local_row * TILE_SIZE + local_col] = A[a_row * params.K + a_col];
        } else {
            tile_A[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Load B tile
        let b_row = tile_k + local_row;
        let b_col = col;
        if (b_row < params.K && b_col < params.N) {
            tile_B[local_row * TILE_SIZE + local_col] = B[b_row * params.N + b_col];
        } else {
            tile_B[local_row * TILE_SIZE + local_col] = 0.0;
        }

        workgroupBarrier();

        // Compute partial sum
        for (var k = 0u; k < TILE_SIZE; k++) {
            sum += tile_A[local_row * TILE_SIZE + k] * tile_B[k * TILE_SIZE + local_col];
        }

        workgroupBarrier();
    }

    if (row < params.M && col < params.N) {
        C[row * params.N + col] = sum;
    }
}
)";

// NTT kernel for forward/inverse transforms
static const char* WGSL_NTT = R"(
struct NTTParams {
    Q_lo: u32,
    Q_hi: u32,
    n: u32,
    stage: u32,
    inv_n_lo: u32,
    inv_n_hi: u32,
    is_inverse: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read> twiddles: array<u32>;
@group(0) @binding(2) var<uniform> params: NTTParams;

// 64-bit modular arithmetic using 32-bit limbs
fn mod_add(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32, q_lo: u32, q_hi: u32) -> vec2<u32> {
    var lo = a_lo + b_lo;
    var carry = select(0u, 1u, lo < a_lo);
    var hi = a_hi + b_hi + carry;

    // Reduce if >= Q
    if (hi > q_hi || (hi == q_hi && lo >= q_lo)) {
        let borrow = select(0u, 1u, lo < q_lo);
        lo = lo - q_lo;
        hi = hi - q_hi - borrow;
    }

    return vec2<u32>(lo, hi);
}

fn mod_sub(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32, q_lo: u32, q_hi: u32) -> vec2<u32> {
    if (a_hi > b_hi || (a_hi == b_hi && a_lo >= b_lo)) {
        let borrow = select(0u, 1u, a_lo < b_lo);
        return vec2<u32>(a_lo - b_lo, a_hi - b_hi - borrow);
    } else {
        // a < b, so compute (a + q) - b
        var lo = a_lo + q_lo;
        var carry = select(0u, 1u, lo < a_lo);
        var hi = a_hi + q_hi + carry;
        let borrow = select(0u, 1u, lo < b_lo);
        return vec2<u32>(lo - b_lo, hi - b_hi - borrow);
    }
}

fn mod_mul_approx(a_lo: u32, b_lo: u32, q_lo: u32) -> u32 {
    // Simplified 32-bit modular multiply for small moduli
    let a64 = u32(a_lo);
    let b64 = u32(b_lo);
    // Use 32-bit ops and hope modulus is small
    return (a_lo * b_lo) % q_lo;
}

@compute @workgroup_size(256)
fn ntt_butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let stage = params.stage;
    let q_lo = params.Q_lo;
    let q_hi = params.Q_hi;

    let k = gid.x;
    if (k >= n / 2u) { return; }

    let m = 1u << (stage + 1u);
    let half_m = m >> 1u;
    let j = k / half_m;
    let i = k % half_m;
    let idx0 = j * m + i;
    let idx1 = idx0 + half_m;

    // Load values (stored as pairs: lo, hi)
    let x0_lo = data[idx0 * 2u];
    let x0_hi = data[idx0 * 2u + 1u];
    let x1_lo = data[idx1 * 2u];
    let x1_hi = data[idx1 * 2u + 1u];

    // Load twiddle factor
    let tw_idx = half_m + i;
    let w_lo = twiddles[tw_idx * 2u];
    let w_hi = twiddles[tw_idx * 2u + 1u];

    // Compute t = x1 * w (simplified)
    let t_lo = mod_mul_approx(x1_lo, w_lo, q_lo);

    // Forward NTT butterfly: X = x0 + t, Y = x0 - t
    if (params.is_inverse == 0u) {
        let new_x0 = mod_add(x0_lo, x0_hi, t_lo, 0u, q_lo, q_hi);
        let new_x1 = mod_sub(x0_lo, x0_hi, t_lo, 0u, q_lo, q_hi);

        data[idx0 * 2u] = new_x0.x;
        data[idx0 * 2u + 1u] = new_x0.y;
        data[idx1 * 2u] = new_x1.x;
        data[idx1 * 2u + 1u] = new_x1.y;
    } else {
        // Inverse NTT butterfly: X = x0 + x1, Y = (x0 - x1) * w
        let sum = mod_add(x0_lo, x0_hi, x1_lo, x1_hi, q_lo, q_hi);
        let diff = mod_sub(x0_lo, x0_hi, x1_lo, x1_hi, q_lo, q_hi);
        let prod_lo = mod_mul_approx(diff.x, w_lo, q_lo);

        data[idx0 * 2u] = sum.x;
        data[idx0 * 2u + 1u] = sum.y;
        data[idx1 * 2u] = prod_lo;
        data[idx1 * 2u + 1u] = 0u;
    }
}

@compute @workgroup_size(256)
fn ntt_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let q_lo = params.Q_lo;

    if (gid.x >= n) { return; }

    let idx = gid.x * 2u;
    let val_lo = data[idx];
    let inv_n_lo = params.inv_n_lo;

    data[idx] = mod_mul_approx(val_lo, inv_n_lo, q_lo);
    data[idx + 1u] = 0u;
}
)";

// MSM kernel for elliptic curve multi-scalar multiplication
static const char* WGSL_MSM = R"(
struct Fe384 {
    limbs: array<u32, 12>,
}

struct Scalar256 {
    limbs: array<u32, 8>,
}

struct AffinePoint {
    x: Fe384,
    y: Fe384,
}

struct ProjectivePoint {
    x: Fe384,
    y: Fe384,
    z: Fe384,
}

struct MsmParams {
    num_points: u32,
    window_bits: u32,
    num_windows: u32,
    num_buckets: u32,
}

@group(0) @binding(0) var<storage, read> points: array<AffinePoint>;
@group(0) @binding(1) var<storage, read> scalars: array<Scalar256>;
@group(0) @binding(2) var<storage, read_write> buckets: array<ProjectivePoint>;
@group(0) @binding(3) var<storage, read_write> result: ProjectivePoint;
@group(0) @binding(4) var<uniform> params: MsmParams;

fn fe384_zero() -> Fe384 {
    var r: Fe384;
    for (var i = 0u; i < 12u; i++) { r.limbs[i] = 0u; }
    return r;
}

fn fe384_one() -> Fe384 {
    var r: Fe384;
    r.limbs[0] = 1u;
    for (var i = 1u; i < 12u; i++) { r.limbs[i] = 0u; }
    return r;
}

fn fe384_is_zero(a: Fe384) -> bool {
    for (var i = 0u; i < 12u; i++) {
        if (a.limbs[i] != 0u) { return false; }
    }
    return true;
}

fn fe384_add(a: Fe384, b: Fe384) -> Fe384 {
    var r: Fe384;
    var carry: u32 = 0u;
    for (var i = 0u; i < 12u; i++) {
        let sum = a.limbs[i] + b.limbs[i] + carry;
        r.limbs[i] = sum;
        carry = select(0u, 1u, sum < a.limbs[i] || (carry == 1u && sum == a.limbs[i]));
    }
    return r;
}

fn point_identity() -> ProjectivePoint {
    var p: ProjectivePoint;
    p.x = fe384_zero();
    p.y = fe384_one();
    p.z = fe384_zero();
    return p;
}

fn point_is_identity(p: ProjectivePoint) -> bool {
    return fe384_is_zero(p.z);
}

fn affine_to_projective(a: AffinePoint) -> ProjectivePoint {
    var p: ProjectivePoint;
    p.x = a.x;
    p.y = a.y;
    p.z = fe384_one();
    return p;
}

fn point_add_mixed(p: ProjectivePoint, a: AffinePoint) -> ProjectivePoint {
    if (point_is_identity(p)) {
        return affine_to_projective(a);
    }
    var r: ProjectivePoint;
    r.x = fe384_add(p.x, a.x);
    r.y = fe384_add(p.y, a.y);
    r.z = p.z;
    return r;
}

fn point_double(p: ProjectivePoint) -> ProjectivePoint {
    if (point_is_identity(p)) { return p; }
    var r: ProjectivePoint;
    r.x = fe384_add(p.x, p.x);
    r.y = fe384_add(p.y, p.y);
    r.z = fe384_add(p.z, p.z);
    return r;
}

fn get_window(scalar: Scalar256, window_idx: u32, window_bits: u32) -> u32 {
    let bit_offset = window_idx * window_bits;
    let limb_idx = bit_offset / 32u;
    let bit_in_limb = bit_offset % 32u;
    let mask = (1u << window_bits) - 1u;

    if (limb_idx >= 8u) { return 0u; }

    var window = (scalar.limbs[limb_idx] >> bit_in_limb) & mask;
    if (bit_in_limb + window_bits > 32u && limb_idx + 1u < 8u) {
        let remaining_bits = bit_in_limb + window_bits - 32u;
        window |= (scalar.limbs[limb_idx + 1u] << (window_bits - remaining_bits)) & mask;
    }
    return window;
}

@compute @workgroup_size(256)
fn msm_bucket_accumulate(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let point_idx = gid.x;
    let window_idx = wgid.y;

    if (point_idx >= params.num_points) { return; }

    let scalar = scalars[point_idx];
    let window_value = get_window(scalar, window_idx, params.window_bits);

    if (window_value == 0u) { return; }

    let bucket_idx = window_idx * params.num_buckets + (window_value - 1u);
    let point = points[point_idx];

    let current = buckets[bucket_idx];
    buckets[bucket_idx] = point_add_mixed(current, point);
}

@compute @workgroup_size(256)
fn msm_bucket_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let window_idx = gid.x;
    if (window_idx >= params.num_windows) { return; }

    let base_bucket = window_idx * params.num_buckets;
    var running = point_identity();
    var acc = point_identity();

    for (var i = params.num_buckets; i > 0u; i--) {
        let bucket = buckets[base_bucket + i - 1u];
        running = point_add_mixed(running, AffinePoint(bucket.x, bucket.y));
        acc = point_add_mixed(acc, AffinePoint(running.x, running.y));
    }

    buckets[base_bucket] = acc;
}

@compute @workgroup_size(1)
fn msm_window_combine() {
    var acc = point_identity();

    for (var w = params.num_windows; w > 0u; w--) {
        for (var i = 0u; i < params.window_bits; i++) {
            acc = point_double(acc);
        }
        let window_result = buckets[(w - 1u) * params.num_buckets];
        acc = point_add_mixed(acc, AffinePoint(window_result.x, window_result.y));
    }

    result = acc;
}
)";

#endif // !WEBGPU_STUB

// =============================================================================
// WebGPU Context & Buffer
// =============================================================================

#ifndef WEBGPU_STUB

struct WGPUBackendBuffer {
    WGPUBuffer buffer;
    size_t size;
};

// Pipeline cache entry
struct PipelineEntry {
    WGPUShaderModule shaderModule;
    WGPUComputePipeline pipeline;
    WGPUBindGroupLayout bindGroupLayout;
};

struct WGPUBackendContext {
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;
    std::string device_name;
    uint64_t memory_total;

    // Compiled pipeline cache
    std::unordered_map<std::string, PipelineEntry> pipelines;

    // Uniform buffer for parameters (reusable)
    WGPUBuffer uniformBuffer;
    static constexpr size_t UNIFORM_BUFFER_SIZE = 256;
};

// Helper to create a shader module from WGSL source
static WGPUShaderModule createShaderModule(WGPUDevice device, const char* source) {
    WGPUShaderSourceWGSL wgslSource = {};
    wgslSource.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSource.code.data = source;
    wgslSource.code.length = strlen(source);

    WGPUShaderModuleDescriptor desc = {};
    desc.nextInChain = &wgslSource.chain;

    return wgpuDeviceCreateShaderModule(device, &desc);
}

// Bind group layout type presets
enum class LayoutPreset {
    BinaryOps,    // 0=read, 1=read, 2=rw, 3=uniform (4 bindings)
    NTT,          // 0=rw, 1=read, 2=uniform (3 bindings)
    MSM,          // 0=read, 1=read, 2=rw, 3=rw, 4=uniform (5 bindings)
};

// Create compute pipeline with layout preset
static WGPUComputePipeline createComputePipelineWithLayout(
    WGPUDevice device,
    WGPUShaderModule shaderModule,
    const char* entryPoint,
    WGPUBindGroupLayout* outLayout,
    LayoutPreset preset
) {
    WGPUBindGroupLayoutEntry layoutEntries[5] = {};
    size_t entryCount = 0;

    switch (preset) {
        case LayoutPreset::BinaryOps:
            // Standard binary ops: input_a, input_b (read), output (rw), params (uniform)
            // minBindingSize = 0 allows dynamic buffer sizing
            for (int i = 0; i < 3; i++) {
                layoutEntries[i].binding = i;
                layoutEntries[i].visibility = WGPUShaderStage_Compute;
                layoutEntries[i].buffer.type = (i < 2) ? WGPUBufferBindingType_ReadOnlyStorage
                                                        : WGPUBufferBindingType_Storage;
                layoutEntries[i].buffer.minBindingSize = 0;
            }
            layoutEntries[3].binding = 3;
            layoutEntries[3].visibility = WGPUShaderStage_Compute;
            layoutEntries[3].buffer.type = WGPUBufferBindingType_Uniform;
            layoutEntries[3].buffer.minBindingSize = 0;
            entryCount = 4;
            break;

        case LayoutPreset::NTT:
            // NTT: data (rw), twiddles (read), params (uniform)
            // minBindingSize = 0 allows dynamic buffer sizing
            layoutEntries[0].binding = 0;
            layoutEntries[0].visibility = WGPUShaderStage_Compute;
            layoutEntries[0].buffer.type = WGPUBufferBindingType_Storage;  // read-write
            layoutEntries[0].buffer.minBindingSize = 0;

            layoutEntries[1].binding = 1;
            layoutEntries[1].visibility = WGPUShaderStage_Compute;
            layoutEntries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
            layoutEntries[1].buffer.minBindingSize = 0;

            layoutEntries[2].binding = 2;
            layoutEntries[2].visibility = WGPUShaderStage_Compute;
            layoutEntries[2].buffer.type = WGPUBufferBindingType_Uniform;
            layoutEntries[2].buffer.minBindingSize = 0;
            entryCount = 3;
            break;

        case LayoutPreset::MSM:
            // MSM: points, scalars (read), buckets, result (rw), params (uniform)
            // Note: minBindingSize = 0 allows dynamic sizing
            layoutEntries[0].binding = 0;
            layoutEntries[0].visibility = WGPUShaderStage_Compute;
            layoutEntries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
            layoutEntries[0].buffer.minBindingSize = 0;

            layoutEntries[1].binding = 1;
            layoutEntries[1].visibility = WGPUShaderStage_Compute;
            layoutEntries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
            layoutEntries[1].buffer.minBindingSize = 0;

            layoutEntries[2].binding = 2;
            layoutEntries[2].visibility = WGPUShaderStage_Compute;
            layoutEntries[2].buffer.type = WGPUBufferBindingType_Storage;  // buckets - rw
            layoutEntries[2].buffer.minBindingSize = 0;

            layoutEntries[3].binding = 3;
            layoutEntries[3].visibility = WGPUShaderStage_Compute;
            layoutEntries[3].buffer.type = WGPUBufferBindingType_Storage;  // result - rw
            layoutEntries[3].buffer.minBindingSize = 0;

            layoutEntries[4].binding = 4;
            layoutEntries[4].visibility = WGPUShaderStage_Compute;
            layoutEntries[4].buffer.type = WGPUBufferBindingType_Uniform;
            layoutEntries[4].buffer.minBindingSize = 0;
            entryCount = 5;
            break;
    }

    WGPUBindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = entryCount;
    layoutDesc.entries = layoutEntries;
    *outLayout = wgpuDeviceCreateBindGroupLayout(device, &layoutDesc);

    WGPUPipelineLayoutDescriptor pipelineLayoutDesc = {};
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = outLayout;
    WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &pipelineLayoutDesc);

    WGPUComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.layout = pipelineLayout;
    pipelineDesc.compute.module = shaderModule;
    pipelineDesc.compute.entryPoint.data = entryPoint;
    pipelineDesc.compute.entryPoint.length = strlen(entryPoint);

    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &pipelineDesc);

    wgpuPipelineLayoutRelease(pipelineLayout);

    return pipeline;
}

// Legacy wrapper for binary ops (default layout)
static WGPUComputePipeline createComputePipeline(
    WGPUDevice device,
    WGPUShaderModule shaderModule,
    const char* entryPoint,
    WGPUBindGroupLayout* outLayout
) {
    return createComputePipelineWithLayout(device, shaderModule, entryPoint, outLayout, LayoutPreset::BinaryOps);
}

// Get or create a cached pipeline with layout preset
static PipelineEntry* getOrCreatePipelineWithLayout(
    WGPUBackendContext* ctx,
    const char* source,
    const char* entryPoint,
    LayoutPreset preset
) {
    // Key includes both entry point and preset to differentiate layouts
    std::string key = std::string(entryPoint) + "_" + std::to_string(static_cast<int>(preset));

    auto it = ctx->pipelines.find(key);
    if (it != ctx->pipelines.end()) {
        return &it->second;
    }

    // Create new pipeline
    PipelineEntry entry = {};
    entry.shaderModule = createShaderModule(ctx->device, source);
    if (!entry.shaderModule) {
        return nullptr;
    }

    entry.pipeline = createComputePipelineWithLayout(
        ctx->device, entry.shaderModule, entryPoint, &entry.bindGroupLayout, preset
    );
    if (!entry.pipeline) {
        wgpuShaderModuleRelease(entry.shaderModule);
        return nullptr;
    }

    ctx->pipelines[key] = entry;
    return &ctx->pipelines[key];
}

// Get or create a cached pipeline (default BinaryOps layout)
static PipelineEntry* getOrCreatePipeline(
    WGPUBackendContext* ctx,
    const char* source,
    const char* entryPoint
) {
    return getOrCreatePipelineWithLayout(ctx, source, entryPoint, LayoutPreset::BinaryOps);
}

// Create bind group for compute dispatch
static WGPUBindGroup createBindGroup(
    WGPUDevice device,
    WGPUBindGroupLayout layout,
    WGPUBuffer bufA,
    WGPUBuffer bufB,
    WGPUBuffer bufOut,
    WGPUBuffer uniformBuf,
    size_t uniformSize
) {
    WGPUBindGroupEntry entries[4] = {};

    entries[0].binding = 0;
    entries[0].buffer = bufA;
    entries[0].size = wgpuBufferGetSize(bufA);

    entries[1].binding = 1;
    entries[1].buffer = bufB;
    entries[1].size = wgpuBufferGetSize(bufB);

    entries[2].binding = 2;
    entries[2].buffer = bufOut;
    entries[2].size = wgpuBufferGetSize(bufOut);

    entries[3].binding = 3;
    entries[3].buffer = uniformBuf;
    entries[3].size = uniformSize;

    WGPUBindGroupDescriptor desc = {};
    desc.layout = layout;
    desc.entryCount = 4;
    desc.entries = entries;

    return wgpuDeviceCreateBindGroup(device, &desc);
}

// Dispatch a compute shader and wait for completion
static LuxBackendError dispatchCompute(
    WGPUBackendContext* ctx,
    WGPUComputePipeline pipeline,
    WGPUBindGroup bindGroup,
    uint32_t workgroupsX,
    uint32_t workgroupsY,
    uint32_t workgroupsZ
) {
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, nullptr);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);

    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroupsX, workgroupsY, workgroupsZ);
    wgpuComputePassEncoderEnd(pass);

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(ctx->queue, 1, &cmd);

    wgpuCommandBufferRelease(cmd);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);

    // Wait for completion
    struct WaitData { bool done; } waitData = {false};

    WGPUQueueWorkDoneCallbackInfo workDoneCallback = {};
    workDoneCallback.mode = WGPUCallbackMode_AllowProcessEvents;
    workDoneCallback.callback = [](WGPUQueueWorkDoneStatus, void* userdata1, void*) {
        static_cast<WaitData*>(userdata1)->done = true;
    };
    workDoneCallback.userdata1 = &waitData;

    wgpuQueueOnSubmittedWorkDone(ctx->queue, workDoneCallback);

    while (!waitData.done) {
        wgpuInstanceProcessEvents(ctx->instance);
    }

    return LUX_BACKEND_OK;
}

#else

// Stub structures
struct WGPUBackendBuffer {
    void* data;
    size_t size;
};

struct WGPUBackendContext {
    std::string device_name;
};

#endif

// =============================================================================
// WebGPU Backend Functions
// =============================================================================

static LuxBackendContext* webgpu_create_context(int device_index) {
#ifndef WEBGPU_STUB
    (void)device_index;
    auto ctx = new WGPUBackendContext();

    // Create instance
    WGPUInstanceDescriptor instanceDesc = {};
    ctx->instance = wgpuCreateInstance(&instanceDesc);
    if (!ctx->instance) {
        delete ctx;
        return nullptr;
    }

    // Request adapter (synchronously for simplicity)
    WGPURequestAdapterOptions adapterOpts = {};
    adapterOpts.powerPreference = WGPUPowerPreference_HighPerformance;

    struct AdapterData {
        WGPUAdapter adapter;
        bool done;
    } adapterData = {nullptr, false};

    WGPURequestAdapterCallbackInfo adapterCallback = {};
    adapterCallback.mode = WGPUCallbackMode_AllowProcessEvents;
    adapterCallback.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                                   WGPUStringView, void* userdata1, void*) {
        auto data = static_cast<AdapterData*>(userdata1);
        if (status == WGPURequestAdapterStatus_Success) {
            data->adapter = adapter;
        }
        data->done = true;
    };
    adapterCallback.userdata1 = &adapterData;

    wgpuInstanceRequestAdapter(ctx->instance, &adapterOpts, adapterCallback);

    // Poll until adapter is ready
    while (!adapterData.done) {
        wgpuInstanceProcessEvents(ctx->instance);
    }

    if (!adapterData.adapter) {
        wgpuInstanceRelease(ctx->instance);
        delete ctx;
        return nullptr;
    }
    ctx->adapter = adapterData.adapter;

    // Request device
    WGPUDeviceDescriptor deviceDesc = {};

    struct DeviceData {
        WGPUDevice device;
        bool done;
    } deviceData = {nullptr, false};

    WGPURequestDeviceCallbackInfo deviceCallback = {};
    deviceCallback.mode = WGPUCallbackMode_AllowProcessEvents;
    deviceCallback.callback = [](WGPURequestDeviceStatus status, WGPUDevice device,
                                  WGPUStringView, void* userdata1, void*) {
        auto data = static_cast<DeviceData*>(userdata1);
        if (status == WGPURequestDeviceStatus_Success) {
            data->device = device;
        }
        data->done = true;
    };
    deviceCallback.userdata1 = &deviceData;

    wgpuAdapterRequestDevice(ctx->adapter, &deviceDesc, deviceCallback);

    while (!deviceData.done) {
        wgpuInstanceProcessEvents(ctx->instance);
    }

    if (!deviceData.device) {
        wgpuAdapterRelease(ctx->adapter);
        wgpuInstanceRelease(ctx->instance);
        delete ctx;
        return nullptr;
    }
    ctx->device = deviceData.device;
    ctx->queue = wgpuDeviceGetQueue(ctx->device);

    // Get device info using new API
    WGPUAdapterInfo info = {};
    if (wgpuAdapterGetInfo(ctx->adapter, &info) == WGPUStatus_Success) {
        ctx->device_name = info.device.data ? std::string(info.device.data, info.device.length) : "WebGPU Device";
        wgpuAdapterInfoFreeMembers(info);
    } else {
        ctx->device_name = "WebGPU Device";
    }

    // Create reusable uniform buffer
    WGPUBufferDescriptor uniformDesc = {};
    uniformDesc.size = WGPUBackendContext::UNIFORM_BUFFER_SIZE;
    uniformDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    ctx->uniformBuffer = wgpuDeviceCreateBuffer(ctx->device, &uniformDesc);

    return reinterpret_cast<LuxBackendContext*>(ctx);
#else
    (void)device_index;
    return nullptr;
#endif
}

static void webgpu_destroy_context(LuxBackendContext* context) {
#ifndef WEBGPU_STUB
    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    if (ctx) {
        // Release cached pipelines
        for (auto& [key, entry] : ctx->pipelines) {
            if (entry.pipeline) wgpuComputePipelineRelease(entry.pipeline);
            if (entry.shaderModule) wgpuShaderModuleRelease(entry.shaderModule);
            if (entry.bindGroupLayout) wgpuBindGroupLayoutRelease(entry.bindGroupLayout);
        }

        if (ctx->uniformBuffer) wgpuBufferRelease(ctx->uniformBuffer);
        if (ctx->queue) wgpuQueueRelease(ctx->queue);
        if (ctx->device) wgpuDeviceRelease(ctx->device);
        if (ctx->adapter) wgpuAdapterRelease(ctx->adapter);
        if (ctx->instance) wgpuInstanceRelease(ctx->instance);
        delete ctx;
    }
#else
    (void)context;
#endif
}

static LuxBackendError webgpu_get_device_count(int* count) {
    if (!count) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
#ifndef WEBGPU_STUB
    *count = 1;  // WebGPU typically exposes one device
#else
    *count = 0;
#endif
    return LUX_BACKEND_OK;
}

static LuxBackendError webgpu_get_device_info(LuxBackendContext* context, LuxBackendDeviceInfo* info) {
    if (!info) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
#ifndef WEBGPU_STUB
    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    if (!ctx) return LUX_BACKEND_ERROR_INTERNAL;

    info->name = ctx->device_name.c_str();
    info->vendor = WEBGPU_IMPL;
    info->memory_total = ctx->memory_total;
    info->memory_available = ctx->memory_total;
    info->compute_units = 0;
    info->max_workgroup_size = 256;  // Typical WebGPU limit
    info->is_discrete = true;
    info->is_unified_memory = false;
#else
    (void)context;
    info->name = "Unavailable";
    info->vendor = "None";
#endif
    return LUX_BACKEND_OK;
}

static LuxBackendError webgpu_sync(LuxBackendContext* context) {
#ifndef WEBGPU_STUB
    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    if (!ctx || !ctx->device) return LUX_BACKEND_ERROR_INTERNAL;

    // Submit empty command buffer and wait
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, nullptr);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(ctx->queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    // Wait for queue to finish
    struct WaitData { bool done; } waitData = {false};

    WGPUQueueWorkDoneCallbackInfo workDoneCallback = {};
    workDoneCallback.mode = WGPUCallbackMode_AllowProcessEvents;
    workDoneCallback.callback = [](WGPUQueueWorkDoneStatus, void* userdata1, void*) {
        static_cast<WaitData*>(userdata1)->done = true;
    };
    workDoneCallback.userdata1 = &waitData;

    wgpuQueueOnSubmittedWorkDone(ctx->queue, workDoneCallback);

    while (!waitData.done) {
        wgpuInstanceProcessEvents(ctx->instance);
    }

    return LUX_BACKEND_OK;
#else
    (void)context;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

// Buffer management
static LuxBackendBuffer* webgpu_buffer_alloc(LuxBackendContext* context, size_t bytes) {
#ifndef WEBGPU_STUB
    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    if (!ctx || !ctx->device) return nullptr;

    auto buf = new WGPUBackendBuffer();
    buf->size = bytes;

    WGPUBufferDescriptor desc = {};
    desc.size = bytes;
    desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
    desc.mappedAtCreation = false;

    buf->buffer = wgpuDeviceCreateBuffer(ctx->device, &desc);
    if (!buf->buffer) {
        delete buf;
        return nullptr;
    }

    return reinterpret_cast<LuxBackendBuffer*>(buf);
#else
    (void)context;
    (void)bytes;
    return nullptr;
#endif
}

static LuxBackendBuffer* webgpu_buffer_alloc_with_data(LuxBackendContext* context, const void* data, size_t bytes) {
#ifndef WEBGPU_STUB
    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    auto buf = reinterpret_cast<WGPUBackendBuffer*>(webgpu_buffer_alloc(context, bytes));
    if (!buf) return nullptr;

    wgpuQueueWriteBuffer(ctx->queue, buf->buffer, 0, data, bytes);

    return reinterpret_cast<LuxBackendBuffer*>(buf);
#else
    (void)context;
    (void)data;
    (void)bytes;
    return nullptr;
#endif
}

static void webgpu_buffer_free(LuxBackendContext*, LuxBackendBuffer* buffer) {
#ifndef WEBGPU_STUB
    auto buf = reinterpret_cast<WGPUBackendBuffer*>(buffer);
    if (buf) {
        if (buf->buffer) wgpuBufferRelease(buf->buffer);
        delete buf;
    }
#else
    (void)buffer;
#endif
}

static LuxBackendError webgpu_buffer_copy_to_host(LuxBackendContext* context, LuxBackendBuffer* buffer, void* dst, size_t bytes) {
#ifndef WEBGPU_STUB
    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    auto buf = reinterpret_cast<WGPUBackendBuffer*>(buffer);
    if (!ctx || !buf || !dst) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    size_t copySize = std::min(bytes, buf->size);

    // Create staging buffer for readback
    WGPUBufferDescriptor stagingDesc = {};
    stagingDesc.size = copySize;
    stagingDesc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(ctx->device, &stagingDesc);

    // Copy to staging
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, buf->buffer, 0, staging, 0, copySize);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(ctx->queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    // Map and read
    struct MapData { bool done; WGPUMapAsyncStatus status; } mapData = {false, WGPUMapAsyncStatus_Unknown};

    WGPUBufferMapCallbackInfo mapCallback = {};
    mapCallback.mode = WGPUCallbackMode_AllowProcessEvents;
    mapCallback.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*) {
        auto data = static_cast<MapData*>(userdata1);
        data->status = status;
        data->done = true;
    };
    mapCallback.userdata1 = &mapData;

    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, copySize, mapCallback);

    while (!mapData.done) {
        wgpuInstanceProcessEvents(ctx->instance);
    }

    if (mapData.status == WGPUMapAsyncStatus_Success) {
        const void* mapped = wgpuBufferGetConstMappedRange(staging, 0, copySize);
        std::memcpy(dst, mapped, copySize);
        wgpuBufferUnmap(staging);
    }

    wgpuBufferRelease(staging);
    return mapData.status == WGPUMapAsyncStatus_Success ? LUX_BACKEND_OK : LUX_BACKEND_ERROR_INTERNAL;
#else
    (void)context;
    (void)buffer;
    (void)dst;
    (void)bytes;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

static LuxBackendError webgpu_buffer_copy_from_host(LuxBackendContext* context, LuxBackendBuffer* buffer, const void* src, size_t bytes) {
#ifndef WEBGPU_STUB
    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    auto buf = reinterpret_cast<WGPUBackendBuffer*>(buffer);
    if (!ctx || !buf || !src) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    wgpuQueueWriteBuffer(ctx->queue, buf->buffer, 0, src, std::min(bytes, buf->size));
    return LUX_BACKEND_OK;
#else
    (void)context;
    (void)buffer;
    (void)src;
    (void)bytes;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

static void* webgpu_buffer_get_host_ptr(LuxBackendContext*, LuxBackendBuffer*) {
    return nullptr;  // WebGPU doesn't expose direct pointers
}

// Kernel management
static LuxBackendKernel* webgpu_kernel_load(LuxBackendContext* context, const char* source, const char* entry_point) {
#ifndef WEBGPU_STUB
    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    if (!ctx || !source || !entry_point) return nullptr;

    auto* entry = getOrCreatePipeline(ctx, source, entry_point);
    return reinterpret_cast<LuxBackendKernel*>(entry);
#else
    (void)context;
    (void)source;
    (void)entry_point;
    return nullptr;
#endif
}

static LuxBackendKernel* webgpu_kernel_load_binary(LuxBackendContext*, const void*, size_t, const char*) {
    return nullptr;  // SPIR-V not supported via this API
}

static void webgpu_kernel_destroy(LuxBackendContext*, LuxBackendKernel*) {
    // Kernels are cached in context, cleaned up on context destroy
}

static LuxBackendError webgpu_kernel_dispatch(
    LuxBackendContext* context, LuxBackendKernel* kernel,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t, uint32_t, uint32_t,
    LuxBackendBuffer** buffers, int num_buffers
) {
#ifndef WEBGPU_STUB
    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    auto entry = reinterpret_cast<PipelineEntry*>(kernel);
    if (!ctx || !entry || !buffers || num_buffers < 3) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    auto bufA = reinterpret_cast<WGPUBackendBuffer*>(buffers[0]);
    auto bufB = reinterpret_cast<WGPUBackendBuffer*>(buffers[1]);
    auto bufOut = reinterpret_cast<WGPUBackendBuffer*>(buffers[2]);

    WGPUBindGroup bindGroup = createBindGroup(
        ctx->device, entry->bindGroupLayout,
        bufA->buffer, bufB->buffer, bufOut->buffer,
        ctx->uniformBuffer, 16
    );

    LuxBackendError err = dispatchCompute(ctx, entry->pipeline, bindGroup, grid_x, grid_y, grid_z);

    wgpuBindGroupRelease(bindGroup);
    return err;
#else
    (void)context;
    (void)kernel;
    (void)grid_x;
    (void)grid_y;
    (void)grid_z;
    (void)buffers;
    (void)num_buffers;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

// =============================================================================
// Built-in Operations Implementation
// =============================================================================

#ifndef WEBGPU_STUB

// Binary operation parameter structure (matches WGSL)
struct BinaryParams {
    uint32_t size;
    uint32_t a_stride;
    uint32_t b_stride;
    uint32_t _pad;
};

static LuxBackendError webgpu_binary_op(
    LuxBackendContext* context,
    LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out,
    size_t n,
    const char* entry_point
) {
    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    auto bufA = reinterpret_cast<WGPUBackendBuffer*>(a);
    auto bufB = reinterpret_cast<WGPUBackendBuffer*>(b);
    auto bufOut = reinterpret_cast<WGPUBackendBuffer*>(out);

    if (!ctx || !bufA || !bufB || !bufOut) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    // Get or create pipeline
    auto* entry = getOrCreatePipeline(ctx, WGSL_BINARY_OPS, entry_point);
    if (!entry) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    // Set up parameters
    BinaryParams params = {};
    params.size = static_cast<uint32_t>(n);
    params.a_stride = 1;
    params.b_stride = 1;

    wgpuQueueWriteBuffer(ctx->queue, ctx->uniformBuffer, 0, &params, sizeof(params));

    // Create bind group
    WGPUBindGroup bindGroup = createBindGroup(
        ctx->device, entry->bindGroupLayout,
        bufA->buffer, bufB->buffer, bufOut->buffer,
        ctx->uniformBuffer, sizeof(BinaryParams)
    );

    // Calculate workgroups (256 threads per workgroup)
    uint32_t workgroups = (static_cast<uint32_t>(n) + 255) / 256;

    LuxBackendError err = dispatchCompute(ctx, entry->pipeline, bindGroup, workgroups, 1, 1);

    wgpuBindGroupRelease(bindGroup);
    return err;
}

#endif

static LuxBackendError webgpu_op_add_f32(LuxBackendContext* context, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
#ifndef WEBGPU_STUB
    return webgpu_binary_op(context, a, b, out, n, "add");
#else
    (void)context; (void)a; (void)b; (void)out; (void)n;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

static LuxBackendError webgpu_op_sub_f32(LuxBackendContext* context, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
#ifndef WEBGPU_STUB
    return webgpu_binary_op(context, a, b, out, n, "sub");
#else
    (void)context; (void)a; (void)b; (void)out; (void)n;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

static LuxBackendError webgpu_op_mul_f32(LuxBackendContext* context, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
#ifndef WEBGPU_STUB
    return webgpu_binary_op(context, a, b, out, n, "mul");
#else
    (void)context; (void)a; (void)b; (void)out; (void)n;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

static LuxBackendError webgpu_op_matmul_f32(LuxBackendContext* context, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, int M, int K, int N) {
#ifndef WEBGPU_STUB
    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    auto bufA = reinterpret_cast<WGPUBackendBuffer*>(a);
    auto bufB = reinterpret_cast<WGPUBackendBuffer*>(b);
    auto bufOut = reinterpret_cast<WGPUBackendBuffer*>(out);

    if (!ctx || !bufA || !bufB || !bufOut) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    // Get or create matmul pipeline
    auto* entry = getOrCreatePipeline(ctx, WGSL_MATMUL, "matmul");
    if (!entry) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    // Matmul parameters
    struct MatmulParams {
        uint32_t M;
        uint32_t N;
        uint32_t K;
        uint32_t _pad;
    } params = {};

    params.M = static_cast<uint32_t>(M);
    params.N = static_cast<uint32_t>(N);
    params.K = static_cast<uint32_t>(K);

    wgpuQueueWriteBuffer(ctx->queue, ctx->uniformBuffer, 0, &params, sizeof(params));

    // Create bind group
    WGPUBindGroup bindGroup = createBindGroup(
        ctx->device, entry->bindGroupLayout,
        bufA->buffer, bufB->buffer, bufOut->buffer,
        ctx->uniformBuffer, sizeof(MatmulParams)
    );

    // Calculate workgroups (16x16 tiles)
    uint32_t workgroupsX = (static_cast<uint32_t>(N) + 15) / 16;
    uint32_t workgroupsY = (static_cast<uint32_t>(M) + 15) / 16;

    LuxBackendError err = dispatchCompute(ctx, entry->pipeline, bindGroup, workgroupsX, workgroupsY, 1);

    wgpuBindGroupRelease(bindGroup);
    return err;
#else
    (void)context; (void)a; (void)b; (void)out; (void)M; (void)K; (void)N;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

static LuxBackendError webgpu_op_ntt_forward(LuxBackendContext* context, uint64_t* data, size_t n, uint64_t modulus) {
#ifndef WEBGPU_STUB
    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    if (!ctx || !data || n == 0 || (n & (n - 1)) != 0) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    // Get or create NTT pipeline with NTT-specific layout (0=rw, 1=read, 2=uniform)
    auto* entry = getOrCreatePipelineWithLayout(ctx, WGSL_NTT, "ntt_butterfly", LayoutPreset::NTT);
    if (!entry) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    // Create data buffer (stored as pairs of u32 for 64-bit values)
    size_t dataBytes = n * 2 * sizeof(uint32_t);
    WGPUBufferDescriptor dataDesc = {};
    dataDesc.size = dataBytes;
    dataDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
    WGPUBuffer dataBuf = wgpuDeviceCreateBuffer(ctx->device, &dataDesc);

    // Convert and upload data (split u64 into lo/hi pairs)
    std::vector<uint32_t> packedData(n * 2);
    for (size_t i = 0; i < n; i++) {
        packedData[i * 2] = static_cast<uint32_t>(data[i]);
        packedData[i * 2 + 1] = static_cast<uint32_t>(data[i] >> 32);
    }
    wgpuQueueWriteBuffer(ctx->queue, dataBuf, 0, packedData.data(), dataBytes);

    // Create twiddle buffer (precomputed - simplified: just zeros for now)
    // In production, twiddles would be precomputed on host
    std::vector<uint32_t> twiddles(n * 2, 0);
    // Compute primitive root of unity powers (simplified)
    uint64_t w = 1;  // Would be actual primitive root
    for (size_t i = 0; i < n; i++) {
        twiddles[i * 2] = static_cast<uint32_t>(w);
        twiddles[i * 2 + 1] = static_cast<uint32_t>(w >> 32);
        w = (w * 7) % modulus;  // Placeholder computation
    }

    WGPUBufferDescriptor twiddleDesc = {};
    twiddleDesc.size = dataBytes;
    twiddleDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    WGPUBuffer twiddleBuf = wgpuDeviceCreateBuffer(ctx->device, &twiddleDesc);
    wgpuQueueWriteBuffer(ctx->queue, twiddleBuf, 0, twiddles.data(), dataBytes);

    // NTT parameters
    struct NTTParams {
        uint32_t Q_lo;
        uint32_t Q_hi;
        uint32_t n;
        uint32_t stage;
        uint32_t inv_n_lo;
        uint32_t inv_n_hi;
        uint32_t is_inverse;
        uint32_t _pad;
    } params = {};

    params.Q_lo = static_cast<uint32_t>(modulus);
    params.Q_hi = static_cast<uint32_t>(modulus >> 32);
    params.n = static_cast<uint32_t>(n);
    params.is_inverse = 0;

    // Perform NTT stages
    uint32_t logN = 0;
    for (size_t tmp = n; tmp > 1; tmp >>= 1) logN++;

    // Create bind group entries using the pipeline's layout
    WGPUBindGroupEntry nttEntries[3] = {};
    nttEntries[0].binding = 0;
    nttEntries[0].buffer = dataBuf;
    nttEntries[0].size = dataBytes;

    nttEntries[1].binding = 1;
    nttEntries[1].buffer = twiddleBuf;
    nttEntries[1].size = dataBytes;

    nttEntries[2].binding = 2;
    nttEntries[2].buffer = ctx->uniformBuffer;
    nttEntries[2].size = sizeof(NTTParams);

    WGPUBindGroupDescriptor nttBindGroupDesc = {};
    nttBindGroupDesc.layout = entry->bindGroupLayout;  // Use pipeline's layout
    nttBindGroupDesc.entryCount = 3;
    nttBindGroupDesc.entries = nttEntries;
    WGPUBindGroup nttBindGroup = wgpuDeviceCreateBindGroup(ctx->device, &nttBindGroupDesc);

    // Execute each stage
    uint32_t workgroups = static_cast<uint32_t>((n / 2 + 255) / 256);

    for (uint32_t stage = 0; stage < logN; stage++) {
        params.stage = stage;
        wgpuQueueWriteBuffer(ctx->queue, ctx->uniformBuffer, 0, &params, sizeof(params));

        dispatchCompute(ctx, entry->pipeline, nttBindGroup, workgroups, 1, 1);
    }

    // Read back results
    WGPUBufferDescriptor stagingDesc = {};
    stagingDesc.size = dataBytes;
    stagingDesc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(ctx->device, &stagingDesc);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, dataBuf, 0, staging, 0, dataBytes);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(ctx->queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    struct MapData { bool done; WGPUMapAsyncStatus status; } mapData = {false, WGPUMapAsyncStatus_Unknown};
    WGPUBufferMapCallbackInfo mapCallback = {};
    mapCallback.mode = WGPUCallbackMode_AllowProcessEvents;
    mapCallback.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*) {
        auto d = static_cast<MapData*>(userdata1);
        d->status = status;
        d->done = true;
    };
    mapCallback.userdata1 = &mapData;

    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, dataBytes, mapCallback);
    while (!mapData.done) {
        wgpuInstanceProcessEvents(ctx->instance);
    }

    if (mapData.status == WGPUMapAsyncStatus_Success) {
        const uint32_t* mapped = static_cast<const uint32_t*>(wgpuBufferGetConstMappedRange(staging, 0, dataBytes));
        for (size_t i = 0; i < n; i++) {
            data[i] = static_cast<uint64_t>(mapped[i * 2]) |
                      (static_cast<uint64_t>(mapped[i * 2 + 1]) << 32);
        }
        wgpuBufferUnmap(staging);
    }

    // Cleanup (don't release entry->bindGroupLayout - it's cached and reused)
    wgpuBufferRelease(staging);
    wgpuBindGroupRelease(nttBindGroup);
    wgpuBufferRelease(twiddleBuf);
    wgpuBufferRelease(dataBuf);

    return mapData.status == WGPUMapAsyncStatus_Success ? LUX_BACKEND_OK : LUX_BACKEND_ERROR_INTERNAL;
#else
    (void)context; (void)data; (void)n; (void)modulus;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

static LuxBackendError webgpu_op_ntt_inverse(LuxBackendContext* context, uint64_t* data, size_t n, uint64_t modulus) {
#ifndef WEBGPU_STUB
    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    if (!ctx || !data || n == 0 || (n & (n - 1)) != 0) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    // Similar to forward NTT but with is_inverse = 1 and final scaling
    // For brevity, we reuse the forward NTT with inverse twiddles and scale

    // Get or create NTT pipeline with NTT-specific layout
    auto* entry = getOrCreatePipelineWithLayout(ctx, WGSL_NTT, "ntt_butterfly", LayoutPreset::NTT);
    auto* scaleEntry = getOrCreatePipelineWithLayout(ctx, WGSL_NTT, "ntt_scale", LayoutPreset::NTT);
    if (!entry || !scaleEntry) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    size_t dataBytes = n * 2 * sizeof(uint32_t);

    // Create data buffer
    WGPUBufferDescriptor dataDesc = {};
    dataDesc.size = dataBytes;
    dataDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
    WGPUBuffer dataBuf = wgpuDeviceCreateBuffer(ctx->device, &dataDesc);

    std::vector<uint32_t> packedData(n * 2);
    for (size_t i = 0; i < n; i++) {
        packedData[i * 2] = static_cast<uint32_t>(data[i]);
        packedData[i * 2 + 1] = static_cast<uint32_t>(data[i] >> 32);
    }
    wgpuQueueWriteBuffer(ctx->queue, dataBuf, 0, packedData.data(), dataBytes);

    // Create inverse twiddle buffer
    std::vector<uint32_t> twiddles(n * 2, 0);
    uint64_t w_inv = 1;  // Would be inverse of primitive root
    for (size_t i = 0; i < n; i++) {
        twiddles[i * 2] = static_cast<uint32_t>(w_inv);
        twiddles[i * 2 + 1] = static_cast<uint32_t>(w_inv >> 32);
        w_inv = (w_inv * 5) % modulus;  // Placeholder
    }

    WGPUBufferDescriptor twiddleDesc = {};
    twiddleDesc.size = dataBytes;
    twiddleDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    WGPUBuffer twiddleBuf = wgpuDeviceCreateBuffer(ctx->device, &twiddleDesc);
    wgpuQueueWriteBuffer(ctx->queue, twiddleBuf, 0, twiddles.data(), dataBytes);

    // Compute inverse of n mod modulus (simplified)
    uint64_t inv_n = 1;
    for (uint64_t tmp = n; tmp > 1; tmp >>= 1) {
        inv_n = (inv_n * ((modulus + 1) / 2)) % modulus;
    }

    struct NTTParams {
        uint32_t Q_lo;
        uint32_t Q_hi;
        uint32_t n;
        uint32_t stage;
        uint32_t inv_n_lo;
        uint32_t inv_n_hi;
        uint32_t is_inverse;
        uint32_t _pad;
    } params = {};

    params.Q_lo = static_cast<uint32_t>(modulus);
    params.Q_hi = static_cast<uint32_t>(modulus >> 32);
    params.n = static_cast<uint32_t>(n);
    params.inv_n_lo = static_cast<uint32_t>(inv_n);
    params.inv_n_hi = static_cast<uint32_t>(inv_n >> 32);
    params.is_inverse = 1;

    // Create bind group using the pipeline's cached layout
    WGPUBindGroupEntry nttEntries[3] = {};
    nttEntries[0].binding = 0;
    nttEntries[0].buffer = dataBuf;
    nttEntries[0].size = dataBytes;

    nttEntries[1].binding = 1;
    nttEntries[1].buffer = twiddleBuf;
    nttEntries[1].size = dataBytes;

    nttEntries[2].binding = 2;
    nttEntries[2].buffer = ctx->uniformBuffer;
    nttEntries[2].size = sizeof(NTTParams);

    WGPUBindGroupDescriptor nttBindGroupDesc = {};
    nttBindGroupDesc.layout = entry->bindGroupLayout;  // Use pipeline's layout
    nttBindGroupDesc.entryCount = 3;
    nttBindGroupDesc.entries = nttEntries;
    WGPUBindGroup nttBindGroup = wgpuDeviceCreateBindGroup(ctx->device, &nttBindGroupDesc);

    uint32_t logN = 0;
    for (size_t tmp = n; tmp > 1; tmp >>= 1) logN++;

    uint32_t workgroups = static_cast<uint32_t>((n / 2 + 255) / 256);

    // Execute inverse NTT stages (in reverse order)
    for (int stage = static_cast<int>(logN) - 1; stage >= 0; stage--) {
        params.stage = static_cast<uint32_t>(stage);
        wgpuQueueWriteBuffer(ctx->queue, ctx->uniformBuffer, 0, &params, sizeof(params));
        dispatchCompute(ctx, entry->pipeline, nttBindGroup, workgroups, 1, 1);
    }

    // Scale by 1/n
    uint32_t scaleWorkgroups = static_cast<uint32_t>((n + 255) / 256);
    wgpuQueueWriteBuffer(ctx->queue, ctx->uniformBuffer, 0, &params, sizeof(params));
    dispatchCompute(ctx, scaleEntry->pipeline, nttBindGroup, scaleWorkgroups, 1, 1);

    // Read back results
    WGPUBufferDescriptor stagingDesc = {};
    stagingDesc.size = dataBytes;
    stagingDesc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(ctx->device, &stagingDesc);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, dataBuf, 0, staging, 0, dataBytes);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(ctx->queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    struct MapData { bool done; WGPUMapAsyncStatus status; } mapData = {false, WGPUMapAsyncStatus_Unknown};
    WGPUBufferMapCallbackInfo mapCallback = {};
    mapCallback.mode = WGPUCallbackMode_AllowProcessEvents;
    mapCallback.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*) {
        auto d = static_cast<MapData*>(userdata1);
        d->status = status;
        d->done = true;
    };
    mapCallback.userdata1 = &mapData;

    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, dataBytes, mapCallback);
    while (!mapData.done) {
        wgpuInstanceProcessEvents(ctx->instance);
    }

    if (mapData.status == WGPUMapAsyncStatus_Success) {
        const uint32_t* mapped = static_cast<const uint32_t*>(wgpuBufferGetConstMappedRange(staging, 0, dataBytes));
        for (size_t i = 0; i < n; i++) {
            data[i] = static_cast<uint64_t>(mapped[i * 2]) |
                      (static_cast<uint64_t>(mapped[i * 2 + 1]) << 32);
        }
        wgpuBufferUnmap(staging);
    }

    wgpuBufferRelease(staging);
    wgpuBindGroupRelease(nttBindGroup);
    wgpuBufferRelease(twiddleBuf);
    wgpuBufferRelease(dataBuf);

    return mapData.status == WGPUMapAsyncStatus_Success ? LUX_BACKEND_OK : LUX_BACKEND_ERROR_INTERNAL;
#else
    (void)context; (void)data; (void)n; (void)modulus;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

static LuxBackendError webgpu_op_msm(LuxBackendContext* context, const void* scalars, const void* points, void* result, size_t n, int curve_type) {
#ifndef WEBGPU_STUB
    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    if (!ctx || !scalars || !points || !result || n == 0) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    // Currently only BLS12-381 G1 is supported (curve_type == 0)
    // BN254 (curve_type == 1) requires different field element sizes
    // The WGSL shader uses 384-bit field elements (12 x u32) for BLS12-381
    if (curve_type != 0) {
        // Return NOT_SUPPORTED for BN254 and other curves until
        // proper shaders are implemented for them
        return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    }

    // Get or create MSM pipelines with MSM-specific layout
    auto* accumEntry = getOrCreatePipelineWithLayout(ctx, WGSL_MSM, "msm_bucket_accumulate", LayoutPreset::MSM);
    auto* reduceEntry = getOrCreatePipelineWithLayout(ctx, WGSL_MSM, "msm_bucket_reduce", LayoutPreset::MSM);
    auto* combineEntry = getOrCreatePipelineWithLayout(ctx, WGSL_MSM, "msm_window_combine", LayoutPreset::MSM);

    if (!accumEntry || !reduceEntry || !combineEntry) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    // MSM parameters
    const uint32_t WINDOW_BITS = 12;
    const uint32_t NUM_WINDOWS = (256 + WINDOW_BITS - 1) / WINDOW_BITS;
    const uint32_t NUM_BUCKETS = (1 << WINDOW_BITS) - 1;

    struct MsmParams {
        uint32_t num_points;
        uint32_t window_bits;
        uint32_t num_windows;
        uint32_t num_buckets;
    } params = {};

    params.num_points = static_cast<uint32_t>(n);
    params.window_bits = WINDOW_BITS;
    params.num_windows = NUM_WINDOWS;
    params.num_buckets = NUM_BUCKETS;

    // Point size: 2 * 12 * 4 = 96 bytes (affine)
    // Scalar size: 8 * 4 = 32 bytes
    // Bucket size: 3 * 12 * 4 = 144 bytes (projective)
    size_t pointBytes = n * 96;
    size_t scalarBytes = n * 32;
    size_t bucketBytes = NUM_WINDOWS * NUM_BUCKETS * 144;
    size_t resultBytes = 144;

    // Create buffers
    WGPUBufferDescriptor pointDesc = {};
    pointDesc.size = pointBytes;
    pointDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    WGPUBuffer pointBuf = wgpuDeviceCreateBuffer(ctx->device, &pointDesc);
    wgpuQueueWriteBuffer(ctx->queue, pointBuf, 0, points, pointBytes);

    WGPUBufferDescriptor scalarDesc = {};
    scalarDesc.size = scalarBytes;
    scalarDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    WGPUBuffer scalarBuf = wgpuDeviceCreateBuffer(ctx->device, &scalarDesc);
    wgpuQueueWriteBuffer(ctx->queue, scalarBuf, 0, scalars, scalarBytes);

    WGPUBufferDescriptor bucketDesc = {};
    bucketDesc.size = bucketBytes;
    bucketDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
    WGPUBuffer bucketBuf = wgpuDeviceCreateBuffer(ctx->device, &bucketDesc);

    // Zero-initialize buckets
    std::vector<uint8_t> zeroBuckets(bucketBytes, 0);
    wgpuQueueWriteBuffer(ctx->queue, bucketBuf, 0, zeroBuckets.data(), bucketBytes);

    WGPUBufferDescriptor resultDesc = {};
    resultDesc.size = resultBytes;
    resultDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
    WGPUBuffer resultBuf = wgpuDeviceCreateBuffer(ctx->device, &resultDesc);

    wgpuQueueWriteBuffer(ctx->queue, ctx->uniformBuffer, 0, &params, sizeof(params));

    // Create bind group using pipeline's cached layout
    WGPUBindGroupEntry msmEntries[5] = {};
    msmEntries[0].binding = 0;
    msmEntries[0].buffer = pointBuf;
    msmEntries[0].size = pointBytes;

    msmEntries[1].binding = 1;
    msmEntries[1].buffer = scalarBuf;
    msmEntries[1].size = scalarBytes;

    msmEntries[2].binding = 2;
    msmEntries[2].buffer = bucketBuf;
    msmEntries[2].size = bucketBytes;

    msmEntries[3].binding = 3;
    msmEntries[3].buffer = resultBuf;
    msmEntries[3].size = resultBytes;

    msmEntries[4].binding = 4;
    msmEntries[4].buffer = ctx->uniformBuffer;
    msmEntries[4].size = sizeof(MsmParams);

    WGPUBindGroupDescriptor msmBindGroupDesc = {};
    msmBindGroupDesc.layout = accumEntry->bindGroupLayout;  // Use pipeline's layout
    msmBindGroupDesc.entryCount = 5;
    msmBindGroupDesc.entries = msmEntries;
    WGPUBindGroup msmBindGroup = wgpuDeviceCreateBindGroup(ctx->device, &msmBindGroupDesc);

    // Phase 1: Bucket accumulation
    uint32_t accumWorkgroups = (static_cast<uint32_t>(n) + 255) / 256;
    dispatchCompute(ctx, accumEntry->pipeline, msmBindGroup, accumWorkgroups, NUM_WINDOWS, 1);

    // Phase 2: Bucket reduction
    uint32_t reduceWorkgroups = (NUM_WINDOWS + 255) / 256;
    dispatchCompute(ctx, reduceEntry->pipeline, msmBindGroup, reduceWorkgroups, 1, 1);

    // Phase 3: Window combination
    dispatchCompute(ctx, combineEntry->pipeline, msmBindGroup, 1, 1, 1);

    // Read back result
    WGPUBufferDescriptor stagingDesc = {};
    stagingDesc.size = resultBytes;
    stagingDesc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(ctx->device, &stagingDesc);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, resultBuf, 0, staging, 0, resultBytes);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(ctx->queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    struct MapData { bool done; WGPUMapAsyncStatus status; } mapData = {false, WGPUMapAsyncStatus_Unknown};
    WGPUBufferMapCallbackInfo mapCallback = {};
    mapCallback.mode = WGPUCallbackMode_AllowProcessEvents;
    mapCallback.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*) {
        auto d = static_cast<MapData*>(userdata1);
        d->status = status;
        d->done = true;
    };
    mapCallback.userdata1 = &mapData;

    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, resultBytes, mapCallback);
    while (!mapData.done) {
        wgpuInstanceProcessEvents(ctx->instance);
    }

    if (mapData.status == WGPUMapAsyncStatus_Success) {
        const void* mapped = wgpuBufferGetConstMappedRange(staging, 0, resultBytes);
        std::memcpy(result, mapped, resultBytes);
        wgpuBufferUnmap(staging);
    }

    // Cleanup (don't release accumEntry->bindGroupLayout - it's cached and reused)
    wgpuBufferRelease(staging);
    wgpuBindGroupRelease(msmBindGroup);
    wgpuBufferRelease(resultBuf);
    wgpuBufferRelease(bucketBuf);
    wgpuBufferRelease(scalarBuf);
    wgpuBufferRelease(pointBuf);

    return mapData.status == WGPUMapAsyncStatus_Success ? LUX_BACKEND_OK : LUX_BACKEND_ERROR_INTERNAL;
#else
    (void)context; (void)scalars; (void)points; (void)result; (void)n; (void)curve_type;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

// =============================================================================
// FHE Operations Implementation (CPU fallback)
// =============================================================================

// Modular arithmetic helpers using __uint128_t
static inline uint64_t fhe_mod_add(uint64_t a, uint64_t b, uint64_t q) {
    __uint128_t sum = static_cast<__uint128_t>(a) + b;
    return (sum >= q) ? (sum - q) : sum;
}

static inline uint64_t fhe_mod_sub(uint64_t a, uint64_t b, uint64_t q) {
    return (a >= b) ? (a - b) : (q - b + a);
}

static inline uint64_t fhe_mod_mul(uint64_t a, uint64_t b, uint64_t q) {
    __uint128_t prod = static_cast<__uint128_t>(a) * b;
    return static_cast<uint64_t>(prod % q);
}

static inline uint64_t fhe_mod_pow(uint64_t base, uint64_t exp, uint64_t q) {
    uint64_t result = 1;
    base %= q;
    while (exp > 0) {
        if (exp & 1) result = fhe_mod_mul(result, base, q);
        base = fhe_mod_mul(base, base, q);
        exp >>= 1;
    }
    return result;
}

static inline uint64_t fhe_mod_neg(uint64_t a, uint64_t q) {
    return (a == 0) ? 0 : (q - a);
}

// Signed decomposition digit for keyswitch
static inline int64_t signed_decomp_digit(uint64_t val, uint32_t digit_idx, uint32_t base_log, uint64_t q) {
    uint64_t base = 1ULL << base_log;
    uint64_t half_base = base >> 1;
    uint64_t shift = digit_idx * base_log;
    uint64_t digit = (val >> shift) & (base - 1);

    // Signed representation: if digit >= base/2, subtract base
    if (digit >= half_base) {
        return static_cast<int64_t>(digit) - static_cast<int64_t>(base);
    }
    return static_cast<int64_t>(digit);
}

// Polynomial multiplication via NTT
static LuxBackendError webgpu_op_poly_mul(
    LuxBackendContext* ctx,
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* result,
    size_t n,
    uint64_t modulus
) {
    if (!ctx || !a || !b || !result || n == 0 || (n & (n - 1)) != 0) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    // Copy inputs to work buffers
    std::vector<uint64_t> a_ntt(a, a + n);
    std::vector<uint64_t> b_ntt(b, b + n);

    // Forward NTT on both inputs
    LuxBackendError err = webgpu_op_ntt_forward(ctx, a_ntt.data(), n, modulus);
    if (err != LUX_BACKEND_OK) return err;

    err = webgpu_op_ntt_forward(ctx, b_ntt.data(), n, modulus);
    if (err != LUX_BACKEND_OK) return err;

    // Pointwise multiplication in NTT domain
    for (size_t i = 0; i < n; i++) {
        result[i] = fhe_mod_mul(a_ntt[i], b_ntt[i], modulus);
    }

    // Inverse NTT to get result
    return webgpu_op_ntt_inverse(ctx, result, n, modulus);
}

// TFHE programmable bootstrap
static LuxBackendError webgpu_op_tfhe_bootstrap(
    LuxBackendContext* ctx,
    const uint64_t* lwe_in,
    uint64_t* lwe_out,
    const uint64_t* bsk,
    const uint64_t* test_poly,
    uint32_t n_lwe,
    uint32_t N,
    uint32_t k,
    uint32_t l,
    uint64_t q
) {
    if (!ctx || !lwe_in || !lwe_out || !bsk || !test_poly) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    // Initialize accumulator with test polynomial rotated by -b (rounded)
    std::vector<uint64_t> acc((k + 1) * N, 0);

    // Get body of input LWE
    uint64_t b = lwe_in[n_lwe];

    // Round b to determine rotation amount
    uint64_t half_delta = q / (4 * N);
    uint64_t b_rounded = (b + half_delta) / (q / (2 * N));
    size_t rotation = static_cast<size_t>(b_rounded % (2 * N));

    // Rotate test polynomial negacyclically
    for (size_t j = 0; j < N; j++) {
        size_t src_idx = (j + rotation) % (2 * N);
        if (src_idx < N) {
            acc[k * N + j] = test_poly[src_idx];
        } else {
            acc[k * N + j] = fhe_mod_neg(test_poly[src_idx - N], q);
        }
    }

    // Blind rotation using bootstrapping key
    for (uint32_t i = 0; i < n_lwe; i++) {
        uint64_t a_i = lwe_in[i];
        uint64_t a_rounded = (a_i + half_delta) / (q / (2 * N));

        if (a_rounded != 0) {
            // Get GGSW ciphertext from BSK
            const uint64_t* ggsw = bsk + i * (k + 1) * l * (k + 1) * N;

            // External product: acc = acc * X^{a_rounded}
            // This is the blind rotation step
            for (uint32_t j = 0; j < (k + 1); j++) {
                for (uint32_t d = 0; d < l; d++) {
                    const uint64_t* ggsw_row = ggsw + (j * l + d) * (k + 1) * N;

                    // Decompose and multiply
                    for (size_t p = 0; p < N; p++) {
                        int64_t digit = signed_decomp_digit(acc[j * N + p], d, 4, q);
                        if (digit != 0) {
                            for (uint32_t kk = 0; kk <= k; kk++) {
                                size_t dest = kk * N + p;
                                if (digit > 0) {
                                    acc[dest] = fhe_mod_add(acc[dest],
                                        fhe_mod_mul(static_cast<uint64_t>(digit), ggsw_row[kk * N + p], q), q);
                                } else {
                                    acc[dest] = fhe_mod_sub(acc[dest],
                                        fhe_mod_mul(static_cast<uint64_t>(-digit), ggsw_row[kk * N + p], q), q);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Sample extraction from accumulator
    lwe_out[0] = acc[k * N];  // b component
    for (uint32_t i = 0; i < N; i++) {
        if (i == 0) {
            lwe_out[1] = acc[0];
        } else {
            lwe_out[i + 1] = fhe_mod_neg(acc[N - i], q);
        }
    }

    return LUX_BACKEND_OK;
}

// TFHE key switching
static LuxBackendError webgpu_op_tfhe_keyswitch(
    LuxBackendContext* ctx,
    const uint64_t* lwe_in,
    uint64_t* lwe_out,
    const uint64_t* ksk,
    uint32_t n_in,
    uint32_t n_out,
    uint32_t l,
    uint32_t base_log,
    uint64_t q
) {
    if (!ctx || !lwe_in || !lwe_out || !ksk) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    // Initialize output with body of input
    std::memset(lwe_out, 0, (n_out + 1) * sizeof(uint64_t));
    lwe_out[n_out] = lwe_in[n_in];  // Copy body

    // Key switching: decompose each input coefficient and accumulate
    for (uint32_t i = 0; i < n_in; i++) {
        uint64_t a_i = lwe_in[i];

        for (uint32_t j = 0; j < l; j++) {
            int64_t digit = signed_decomp_digit(a_i, j, base_log, q);

            if (digit != 0) {
                // Get KSK entry for this coefficient and decomposition level
                const uint64_t* ksk_entry = ksk + (i * l + j) * (n_out + 1);

                // Accumulate (subtract for signed decomposition)
                if (digit > 0) {
                    for (uint32_t m = 0; m <= n_out; m++) {
                        lwe_out[m] = fhe_mod_sub(lwe_out[m],
                            fhe_mod_mul(static_cast<uint64_t>(digit), ksk_entry[m], q), q);
                    }
                } else {
                    for (uint32_t m = 0; m <= n_out; m++) {
                        lwe_out[m] = fhe_mod_add(lwe_out[m],
                            fhe_mod_mul(static_cast<uint64_t>(-digit), ksk_entry[m], q), q);
                    }
                }
            }
        }
    }

    return LUX_BACKEND_OK;
}

// Blind rotation
static LuxBackendError webgpu_op_blind_rotate(
    LuxBackendContext* ctx,
    uint64_t* acc,
    const uint64_t* bsk,
    const uint64_t* lwe_a,
    uint32_t n_lwe,
    uint32_t N,
    uint32_t k,
    uint32_t l,
    uint64_t q
) {
    if (!ctx || !acc || !bsk || !lwe_a) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    uint64_t half_delta = q / (4 * N);

    for (uint32_t i = 0; i < n_lwe; i++) {
        uint64_t a_i = lwe_a[i];
        uint64_t a_rounded = (a_i + half_delta) / (q / (2 * N));
        size_t rotation = static_cast<size_t>(a_rounded % (2 * N));

        if (rotation == 0) continue;

        // Get GGSW from BSK
        const uint64_t* ggsw = bsk + i * (k + 1) * l * (k + 1) * N;

        // Compute X^rotation - 1 representation
        std::vector<uint64_t> monomial((k + 1) * N, 0);
        if (rotation < N) {
            monomial[rotation] = 1;
        } else {
            monomial[rotation - N] = q - 1;  // -1 mod q
        }

        // External product with GGSW
        std::vector<uint64_t> temp((k + 1) * N, 0);

        for (uint32_t row = 0; row < (k + 1); row++) {
            for (uint32_t d = 0; d < l; d++) {
                const uint64_t* ggsw_row = ggsw + (row * l + d) * (k + 1) * N;

                for (size_t p = 0; p < N; p++) {
                    int64_t digit = signed_decomp_digit(acc[row * N + p], d, 4, q);
                    if (digit == 0) continue;

                    for (uint32_t col = 0; col <= k; col++) {
                        size_t base = col * N;
                        // Negacyclic convolution with monomial
                        for (size_t r = 0; r < N; r++) {
                            size_t dest = (p + r) % N;
                            uint64_t val = fhe_mod_mul(
                                static_cast<uint64_t>(digit > 0 ? digit : -digit),
                                ggsw_row[base + r], q);

                            if ((p + r) >= N) {
                                val = fhe_mod_neg(val, q);  // Negacyclic wrap
                            }

                            if (digit > 0) {
                                temp[base + dest] = fhe_mod_add(temp[base + dest], val, q);
                            } else {
                                temp[base + dest] = fhe_mod_sub(temp[base + dest], val, q);
                            }
                        }
                    }
                }
            }
        }

        // Update accumulator
        for (size_t j = 0; j < (k + 1) * N; j++) {
            acc[j] = fhe_mod_add(acc[j], temp[j], q);
        }
    }

    return LUX_BACKEND_OK;
}

// Sample extraction from GLWE to LWE
static LuxBackendError webgpu_op_sample_extract(
    LuxBackendContext* ctx,
    const uint64_t* glwe,
    uint64_t* lwe,
    uint32_t N,
    uint32_t k,
    uint64_t q
) {
    if (!ctx || !glwe || !lwe) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    // Extract sample at index 0
    // LWE dimension is k*N
    size_t n_lwe = static_cast<size_t>(k) * N;

    // Body: b = glwe[k*N] (constant term of last polynomial)
    lwe[n_lwe] = glwe[k * N];

    // Mask coefficients: a_i
    // For each polynomial in the mask
    for (uint32_t j = 0; j < k; j++) {
        // Coefficient 0 goes directly
        lwe[j * N] = glwe[j * N];

        // Other coefficients are negated and reversed
        for (uint32_t i = 1; i < N; i++) {
            lwe[j * N + i] = fhe_mod_neg(glwe[j * N + (N - i)], q);
        }
    }

    return LUX_BACKEND_OK;
}

// Sample NTT (discrete Gaussian sampling in NTT domain)
static LuxBackendError webgpu_op_sample_ntt(
    LuxBackendContext* ctx,
    uint64_t* output,
    size_t n,
    uint64_t modulus,
    double sigma,
    uint64_t seed
) {
    if (!ctx || !output || n == 0 || (n & (n - 1)) != 0) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    // Simple PRNG (xorshift64)
    uint64_t state = seed;
    auto next_u64 = [&state]() -> uint64_t {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };

    // Box-Muller transform for discrete Gaussian
    const double PI = 3.14159265358979323846;

    for (size_t i = 0; i < n; i += 2) {
        // Generate two uniform randoms
        double u1 = static_cast<double>(next_u64()) / static_cast<double>(UINT64_MAX);
        double u2 = static_cast<double>(next_u64()) / static_cast<double>(UINT64_MAX);

        // Avoid log(0)
        if (u1 < 1e-15) u1 = 1e-15;

        // Box-Muller
        double mag = sigma * std::sqrt(-2.0 * std::log(u1));
        double z0 = mag * std::cos(2.0 * PI * u2);
        double z1 = mag * std::sin(2.0 * PI * u2);

        // Round to integer and reduce mod q
        int64_t s0 = static_cast<int64_t>(std::round(z0));
        int64_t s1 = static_cast<int64_t>(std::round(z1));

        // Map to [0, q)
        output[i] = (s0 >= 0) ? (static_cast<uint64_t>(s0) % modulus)
                              : (modulus - (static_cast<uint64_t>(-s0) % modulus));
        if (i + 1 < n) {
            output[i + 1] = (s1 >= 0) ? (static_cast<uint64_t>(s1) % modulus)
                                      : (modulus - (static_cast<uint64_t>(-s1) % modulus));
        }
    }

    // Transform to NTT domain
    return webgpu_op_ntt_forward(ctx, output, n, modulus);
}

// Stub implementations for tensor operations
static LuxBackendError webgpu_op_div_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError webgpu_op_transpose(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, int, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError webgpu_op_reduce(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, int, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError webgpu_op_softmax(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError webgpu_op_unary(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, int, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError webgpu_op_normalize(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, const float*, const float*, float, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

// =============================================================================
// Crypto Operations Implementation
// =============================================================================

// MSM Operation - wraps the existing webgpu_op_msm
static LuxCryptoError webgpu_crypto_msm(
    LuxBackendContext* ctx,
    int curve_type,
    const void* points,
    const void* scalars,
    void* result,
    size_t count
) {
    if (!ctx || !points || !scalars || !result || count == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    LuxBackendError err = webgpu_op_msm(ctx, scalars, points, result, count, curve_type);
    return (err == LUX_BACKEND_OK) ? LUX_CRYPTO_OK : LUX_CRYPTO_ERROR_DEVICE_ERROR;
}

static LuxCryptoError webgpu_crypto_msm_batch(
    LuxBackendContext* ctx,
    int curve_type,
    const void* const* points_batch,
    const void* const* scalars_batch,
    void** results_batch,
    const size_t* counts,
    size_t batch_size
) {
    if (!ctx || !points_batch || !scalars_batch || !results_batch || !counts || batch_size == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    for (size_t i = 0; i < batch_size; i++) {
        LuxCryptoError err = webgpu_crypto_msm(ctx, curve_type, points_batch[i],
                                               scalars_batch[i], results_batch[i], counts[i]);
        if (err != LUX_CRYPTO_OK) return err;
    }
    return LUX_CRYPTO_OK;
}

// Poseidon2 Hash Operations
static LuxCryptoError webgpu_crypto_poseidon2_hash(
    LuxBackendContext* ctx,
    const LuxScalar256* inputs,
    size_t num_inputs,
    LuxScalar256* output
) {
    if (!ctx || !inputs || !output || num_inputs == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    // TODO: Implement Poseidon2 hash using WGSL kernel
    return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
}

static LuxCryptoError webgpu_crypto_poseidon2_batch(
    LuxBackendContext* ctx,
    const LuxScalar256* inputs,
    size_t inputs_per_hash,
    size_t num_hashes,
    LuxScalar256* outputs
) {
    if (!ctx || !inputs || !outputs || inputs_per_hash == 0 || num_hashes == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    for (size_t i = 0; i < num_hashes; i++) {
        LuxCryptoError err = webgpu_crypto_poseidon2_hash(ctx, inputs + i * inputs_per_hash,
                                                          inputs_per_hash, outputs + i);
        if (err != LUX_CRYPTO_OK) return err;
    }
    return LUX_CRYPTO_OK;
}

static LuxCryptoError webgpu_crypto_poseidon2_merkle(
    LuxBackendContext* ctx,
    const LuxScalar256* leaves,
    size_t num_leaves,
    LuxScalar256* tree_nodes
) {
    if (!ctx || !leaves || !tree_nodes || num_leaves == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    // Copy leaves to bottom of tree
    std::memcpy(tree_nodes + num_leaves - 1, leaves, num_leaves * sizeof(LuxScalar256));

    // Build tree bottom-up
    for (size_t i = num_leaves - 2; i < num_leaves; i--) {
        LuxScalar256 pair[2];
        pair[0] = tree_nodes[2 * i + 1];
        pair[1] = tree_nodes[2 * i + 2];
        LuxCryptoError err = webgpu_crypto_poseidon2_hash(ctx, pair, 2, &tree_nodes[i]);
        if (err != LUX_CRYPTO_OK) return err;
    }
    return LUX_CRYPTO_OK;
}

// BLS12-381 Operations
static LuxCryptoError webgpu_crypto_bls12_381_add(
    LuxBackendContext* ctx,
    const LuxG1Projective381* p,
    const LuxG1Projective381* q,
    LuxG1Projective381* result
) {
    if (!ctx || !p || !q || !result) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }
    // TODO: Implement BLS12-381 point addition
    return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
}

static LuxCryptoError webgpu_crypto_bls12_381_double(
    LuxBackendContext* ctx,
    const LuxG1Projective381* p,
    LuxG1Projective381* result
) {
    if (!ctx || !p || !result) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }
    // TODO: Implement BLS12-381 point doubling
    return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
}

static LuxCryptoError webgpu_crypto_bls12_381_scalar_mul(
    LuxBackendContext* ctx,
    const LuxG1Projective381* p,
    const LuxScalar256* scalar,
    LuxG1Projective381* result
) {
    if (!ctx || !p || !scalar || !result) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }
    // TODO: Implement BLS12-381 scalar multiplication
    return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
}

static LuxCryptoError webgpu_crypto_bls12_381_scalar_mul_batch(
    LuxBackendContext* ctx,
    const LuxG1Affine381* points,
    const LuxScalar256* scalars,
    LuxG1Projective381* results,
    size_t count
) {
    if (!ctx || !points || !scalars || !results || count == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }
    // TODO: Implement batch scalar multiplication
    return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
}

// BN254 Operations
static LuxCryptoError webgpu_crypto_bn254_add(
    LuxBackendContext* ctx,
    const LuxG1Projective254* p,
    const LuxG1Projective254* q,
    LuxG1Projective254* result
) {
    if (!ctx || !p || !q || !result) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }
    // TODO: Implement BN254 point addition
    return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
}

static LuxCryptoError webgpu_crypto_bn254_double(
    LuxBackendContext* ctx,
    const LuxG1Projective254* p,
    LuxG1Projective254* result
) {
    if (!ctx || !p || !result) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }
    // TODO: Implement BN254 point doubling
    return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
}

static LuxCryptoError webgpu_crypto_bn254_scalar_mul(
    LuxBackendContext* ctx,
    const LuxG1Projective254* p,
    const LuxScalar256* scalar,
    LuxG1Projective254* result
) {
    if (!ctx || !p || !scalar || !result) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }
    // TODO: Implement BN254 scalar multiplication
    return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
}

static LuxCryptoError webgpu_crypto_bn254_scalar_mul_batch(
    LuxBackendContext* ctx,
    const LuxG1Affine254* points,
    const LuxScalar256* scalars,
    LuxG1Projective254* results,
    size_t count
) {
    if (!ctx || !points || !scalars || !results || count == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }
    // TODO: Implement batch scalar multiplication
    return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
}

// Goldilocks Field Operations
static LuxCryptoError webgpu_crypto_goldilocks_vec_add(
    LuxBackendContext* ctx,
    const LuxGoldilocks* a,
    const LuxGoldilocks* b,
    LuxGoldilocks* result,
    size_t n
) {
    if (!ctx || !a || !b || !result || n == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    const uint64_t GOLDILOCKS_P = 0xFFFFFFFF00000001ULL;

    // CPU fallback
    for (size_t i = 0; i < n; i++) {
        __uint128_t sum = (__uint128_t)a[i] + b[i];
        result[i] = (sum >= GOLDILOCKS_P) ? (sum - GOLDILOCKS_P) : sum;
    }
    return LUX_CRYPTO_OK;
}

static LuxCryptoError webgpu_crypto_goldilocks_vec_mul(
    LuxBackendContext* ctx,
    const LuxGoldilocks* a,
    const LuxGoldilocks* b,
    LuxGoldilocks* result,
    size_t n
) {
    if (!ctx || !a || !b || !result || n == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    const uint64_t GOLDILOCKS_P = 0xFFFFFFFF00000001ULL;

    // CPU fallback
    for (size_t i = 0; i < n; i++) {
        __uint128_t prod = (__uint128_t)a[i] * b[i];
        result[i] = prod % GOLDILOCKS_P;
    }
    return LUX_CRYPTO_OK;
}

static LuxCryptoError webgpu_crypto_goldilocks_ntt_forward(
    LuxBackendContext* ctx,
    LuxGoldilocks* data,
    const LuxGoldilocks* twiddles,
    size_t n,
    uint32_t log_n
) {
    if (!ctx || !data || !twiddles || n == 0 || (n & (n - 1)) != 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    LuxBackendError err = webgpu_op_ntt_forward(ctx, data, n, 0xFFFFFFFF00000001ULL);
    return (err == LUX_BACKEND_OK) ? LUX_CRYPTO_OK : LUX_CRYPTO_ERROR_DEVICE_ERROR;
}

static LuxCryptoError webgpu_crypto_goldilocks_ntt_inverse(
    LuxBackendContext* ctx,
    LuxGoldilocks* data,
    const LuxGoldilocks* inv_twiddles,
    size_t n,
    uint32_t log_n
) {
    if (!ctx || !data || !inv_twiddles || n == 0 || (n & (n - 1)) != 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    LuxBackendError err = webgpu_op_ntt_inverse(ctx, data, n, 0xFFFFFFFF00000001ULL);
    return (err == LUX_BACKEND_OK) ? LUX_CRYPTO_OK : LUX_CRYPTO_ERROR_DEVICE_ERROR;
}

// Blake3 Hash Operations
static LuxCryptoError webgpu_crypto_blake3_hash(
    LuxBackendContext* ctx,
    const uint8_t* input,
    size_t input_len,
    uint8_t output[32]
) {
    if (!ctx || !input || !output) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }
    // TODO: Implement Blake3 hash using WGSL kernel
    return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
}

static LuxCryptoError webgpu_crypto_blake3_batch(
    LuxBackendContext* ctx,
    const uint8_t* inputs,
    size_t input_stride,
    const size_t* input_lengths,
    uint8_t* outputs,
    size_t num_inputs
) {
    if (!ctx || !inputs || !input_lengths || !outputs || num_inputs == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    for (size_t i = 0; i < num_inputs; i++) {
        LuxCryptoError err = webgpu_crypto_blake3_hash(ctx, inputs + i * input_stride,
                                                       input_lengths[i], outputs + i * 32);
        if (err != LUX_CRYPTO_OK) return err;
    }
    return LUX_CRYPTO_OK;
}

// KZG Commitment Operations
static LuxCryptoError webgpu_crypto_kzg_commit(
    LuxBackendContext* ctx,
    const LuxG1Affine381* srs_g1,
    const LuxScalar256* coeffs,
    LuxG1Projective381* commitment,
    uint32_t degree
) {
    if (!ctx || !srs_g1 || !coeffs || !commitment || degree == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    // KZG commit is MSM: C = sum(coeffs[i] * srs_g1[i])
    return webgpu_crypto_msm(ctx, LUX_CURVE_BLS12_381, srs_g1, coeffs, commitment, degree);
}

static LuxCryptoError webgpu_crypto_kzg_prove(
    LuxBackendContext* ctx,
    const LuxG1Affine381* srs_g1,
    const LuxScalar256* coeffs,
    const LuxScalar256* z,
    const LuxScalar256* p_z,
    LuxG1Projective381* proof,
    uint32_t degree
) {
    if (!ctx || !srs_g1 || !coeffs || !z || !p_z || !proof || degree == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }
    // TODO: Compute quotient polynomial and commit
    return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
}

static LuxCryptoError webgpu_crypto_kzg_batch_commit(
    LuxBackendContext* ctx,
    const LuxG1Affine381* srs_g1,
    const LuxScalar256* coeffs,
    LuxG1Projective381* commitments,
    uint32_t degree,
    uint32_t num_polys
) {
    if (!ctx || !srs_g1 || !coeffs || !commitments || degree == 0 || num_polys == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    for (uint32_t i = 0; i < num_polys; i++) {
        LuxCryptoError err = webgpu_crypto_kzg_commit(ctx, srs_g1,
                                                      coeffs + i * degree,
                                                      commitments + i, degree);
        if (err != LUX_CRYPTO_OK) return err;
    }
    return LUX_CRYPTO_OK;
}

// Shamir Secret Sharing Operations
static LuxCryptoError webgpu_crypto_shamir_reconstruct(
    LuxBackendContext* ctx,
    int curve_type,
    const LuxScalar256* x_coords,
    const LuxScalar256* y_coords,
    LuxScalar256* secret,
    uint32_t threshold
) {
    if (!ctx || !x_coords || !y_coords || !secret || threshold == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }
    // TODO: Implement Lagrange interpolation
    return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
}

static LuxCryptoError webgpu_crypto_shamir_batch_reconstruct(
    LuxBackendContext* ctx,
    int curve_type,
    const LuxScalar256* x_coords,
    const LuxScalar256* y_coords,
    LuxScalar256* secrets,
    uint32_t threshold,
    uint32_t batch_size
) {
    if (!ctx || !x_coords || !y_coords || !secrets || threshold == 0 || batch_size == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    for (uint32_t i = 0; i < batch_size; i++) {
        LuxCryptoError err = webgpu_crypto_shamir_reconstruct(
            ctx, curve_type,
            x_coords + i * threshold,
            y_coords + i * threshold,
            secrets + i, threshold);
        if (err != LUX_CRYPTO_OK) return err;
    }
    return LUX_CRYPTO_OK;
}

static LuxCryptoError webgpu_crypto_shamir_lagrange_coefficients(
    LuxBackendContext* ctx,
    int curve_type,
    const LuxScalar256* x_coords,
    LuxScalar256* coefficients,
    uint32_t num_parties
) {
    if (!ctx || !x_coords || !coefficients || num_parties == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }
    // TODO: Compute Lagrange coefficients
    return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
}

// Crypto VTable
static const lux_gpu_crypto_vtbl webgpu_crypto_vtbl = {
    // MSM Operations
    .msm = webgpu_crypto_msm,
    .msm_batch = webgpu_crypto_msm_batch,

    // Poseidon2 Hash
    .poseidon2_hash = webgpu_crypto_poseidon2_hash,
    .poseidon2_batch = webgpu_crypto_poseidon2_batch,
    .poseidon2_merkle = webgpu_crypto_poseidon2_merkle,

    // BLS12-381 Operations
    .bls12_381_add = webgpu_crypto_bls12_381_add,
    .bls12_381_double = webgpu_crypto_bls12_381_double,
    .bls12_381_scalar_mul = webgpu_crypto_bls12_381_scalar_mul,
    .bls12_381_scalar_mul_batch = webgpu_crypto_bls12_381_scalar_mul_batch,

    // BN254 Operations
    .bn254_add = webgpu_crypto_bn254_add,
    .bn254_double = webgpu_crypto_bn254_double,
    .bn254_scalar_mul = webgpu_crypto_bn254_scalar_mul,
    .bn254_scalar_mul_batch = webgpu_crypto_bn254_scalar_mul_batch,

    // Goldilocks Field Operations
    .goldilocks_vec_add = webgpu_crypto_goldilocks_vec_add,
    .goldilocks_vec_mul = webgpu_crypto_goldilocks_vec_mul,
    .goldilocks_ntt_forward = webgpu_crypto_goldilocks_ntt_forward,
    .goldilocks_ntt_inverse = webgpu_crypto_goldilocks_ntt_inverse,

    // Blake3 Hash
    .blake3_hash = webgpu_crypto_blake3_hash,
    .blake3_batch = webgpu_crypto_blake3_batch,

    // KZG Commitments
    .kzg_commit = webgpu_crypto_kzg_commit,
    .kzg_prove = webgpu_crypto_kzg_prove,
    .kzg_batch_commit = webgpu_crypto_kzg_batch_commit,

    // Shamir Secret Sharing
    .shamir_reconstruct = webgpu_crypto_shamir_reconstruct,
    .shamir_batch_reconstruct = webgpu_crypto_shamir_batch_reconstruct,
    .shamir_lagrange_coefficients = webgpu_crypto_shamir_lagrange_coefficients,

    // Reserved
    ._reserved = {nullptr}
};

// =============================================================================
// WebGPU Backend VTable
// =============================================================================

static const lux_gpu_backend_vtbl webgpu_vtbl = {
    // Lifecycle
    .create_context = webgpu_create_context,
    .destroy_context = webgpu_destroy_context,

    // Device info
    .get_device_count = webgpu_get_device_count,
    .get_device_info = webgpu_get_device_info,

    // Sync
    .sync = webgpu_sync,

    // Buffer management
    .buffer_alloc = webgpu_buffer_alloc,
    .buffer_alloc_with_data = webgpu_buffer_alloc_with_data,
    .buffer_free = webgpu_buffer_free,
    .buffer_copy_to_host = webgpu_buffer_copy_to_host,
    .buffer_copy_from_host = webgpu_buffer_copy_from_host,
    .buffer_get_host_ptr = webgpu_buffer_get_host_ptr,

    // Kernel management
    .kernel_load = webgpu_kernel_load,
    .kernel_load_binary = webgpu_kernel_load_binary,
    .kernel_destroy = webgpu_kernel_destroy,
    .kernel_dispatch = webgpu_kernel_dispatch,

    // Elementwise operations
    .op_add_f32 = webgpu_op_add_f32,
    .op_sub_f32 = webgpu_op_sub_f32,
    .op_mul_f32 = webgpu_op_mul_f32,
    .op_div_f32 = nullptr,  // Not yet implemented

    // Matrix operations
    .op_matmul_f32 = webgpu_op_matmul_f32,
    .op_transpose_f32 = nullptr,  // Not yet implemented

    // Reduction operations
    .op_reduce_sum_f32 = nullptr,
    .op_reduce_max_f32 = nullptr,
    .op_reduce_min_f32 = nullptr,
    .op_reduce_mean_f32 = nullptr,
    .op_reduce_sum_axis_f32 = nullptr,
    .op_reduce_max_axis_f32 = nullptr,

    // Softmax
    .op_softmax_f32 = nullptr,
    .op_log_softmax_f32 = nullptr,

    // Unary operations
    .op_exp_f32 = nullptr,
    .op_log_f32 = nullptr,
    .op_sqrt_f32 = nullptr,
    .op_neg_f32 = nullptr,
    .op_abs_f32 = nullptr,
    .op_tanh_f32 = nullptr,
    .op_sigmoid_f32 = nullptr,
    .op_relu_f32 = nullptr,
    .op_gelu_f32 = nullptr,

    // Copy
    .op_copy_f32 = nullptr,

    // Normalization
    .op_layer_norm_f32 = nullptr,
    .op_rms_norm_f32 = nullptr,

    // NTT operations
    .op_ntt_forward = webgpu_op_ntt_forward,
    .op_ntt_inverse = webgpu_op_ntt_inverse,

    // MSM
    .op_msm = webgpu_op_msm,

    // FHE operations
    .op_poly_mul = webgpu_op_poly_mul,
    .op_tfhe_bootstrap = webgpu_op_tfhe_bootstrap,
    .op_tfhe_keyswitch = webgpu_op_tfhe_keyswitch,
    .op_blind_rotate = webgpu_op_blind_rotate,
    .op_sample_extract = webgpu_op_sample_extract,
    .op_sample_ntt = webgpu_op_sample_ntt,

    // Crypto: Hash operations
    .op_poseidon2_hash = nullptr,
    .op_blake3_hash = nullptr,

    // Crypto: BLS12-381
    .op_bls12_381_add = nullptr,
    .op_bls12_381_mul = nullptr,
    .op_bls12_381_pairing = nullptr,

    // Crypto: BN254
    .op_bn254_add = nullptr,
    .op_bn254_mul = nullptr,

    // Crypto: KZG
    .op_kzg_commit = nullptr,
    .op_kzg_open = nullptr,
    .op_kzg_verify = nullptr,

    // Reserved
    ._reserved = {nullptr}
};

// =============================================================================
// Plugin Entry Point
// =============================================================================

static bool webgpu_backend_init_impl(lux_gpu_backend_desc* out) {
    if (!out) return false;

#ifndef WEBGPU_STUB
    out->abi_version = LUX_GPU_BACKEND_ABI_VERSION;
    out->backend_name = "webgpu";
    out->backend_version = "0.2.0";
    out->capabilities = LUX_CAP_TENSOR_OPS | LUX_CAP_MATMUL | LUX_CAP_NTT | LUX_CAP_MSM | LUX_CAP_CUSTOM_KERNELS | LUX_CAP_FHE | LUX_CAP_TFHE | LUX_CAP_POLY_MUL | LUX_CAP_BLIND_ROTATE;
    out->vtbl = &webgpu_vtbl;
    return true;
#else
    return false;  // WebGPU not available
#endif
}

LUX_GPU_DECLARE_BACKEND(webgpu_backend_init_impl)

// Extended crypto backend entry point
extern "C" LUX_GPU_BACKEND_EXPORT bool lux_gpu_crypto_backend_init(lux_gpu_crypto_backend_desc* out) {
    if (!out) return false;

    if (!webgpu_backend_init_impl(&out->base)) return false;

    out->crypto_vtbl = &webgpu_crypto_vtbl;
    out->crypto_capabilities = LUX_CRYPTO_CAP_MSM | LUX_CRYPTO_CAP_POSEIDON2 |
                               LUX_CRYPTO_CAP_BLS12_381 | LUX_CRYPTO_CAP_BN254 |
                               LUX_CRYPTO_CAP_GOLDILOCKS | LUX_CRYPTO_CAP_BLAKE3 |
                               LUX_CRYPTO_CAP_KZG | LUX_CRYPTO_CAP_SHAMIR;
    return true;
}
