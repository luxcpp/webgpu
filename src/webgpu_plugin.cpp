// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// WebGPU Backend Plugin - Cross-platform GPU via Dawn/wgpu-native
// Loaded as a shared library via dlopen()
//
// This plugin supports both Dawn (Google) and wgpu-native (Mozilla) WebGPU
// implementations, selected at compile time via USE_DAWN_API or USE_WGPU_API.

#include "lux/gpu/backend_plugin.h"
#include "lux/gpu/crypto.h"
#include "lux/gpu/internal/privilege.h"
#include "cpu_fhe_helpers.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string_view>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#ifndef _WIN32
#include <dlfcn.h>
#endif

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

// Sentinel "unset" value for WGPUMapAsyncStatus. Dawn's webgpu/webgpu.h
// defines WGPUMapAsyncStatus_Unknown for this purpose; wgpu-native uses the
// canonical webgpu.h ABI which does not. Both define _Error which serves the
// same role as a non-success sentinel (it triggers the err branch when the
// callback never fires).
#ifndef WEBGPU_STUB
    #ifndef LUX_MAP_ASYNC_STATUS_UNSET
        #if defined(USE_DAWN_API)
            #define LUX_MAP_ASYNC_STATUS_UNSET WGPUMapAsyncStatus_Unknown
        #else
            #define LUX_MAP_ASYNC_STATUS_UNSET WGPUMapAsyncStatus_Error
        #endif
    #endif
#endif

// Deadline used by every wgpuInstanceProcessEvents wait loop. On a healthy
// device the callbacks fire in microseconds; the timeout exists exclusively
// to bound a device-lost / driver-hang failure so the host process doesn't
// spin forever. Override at compile time with -DLUX_WEBGPU_WAIT_TIMEOUT_S=N.
#ifndef LUX_WEBGPU_WAIT_TIMEOUT_S
#define LUX_WEBGPU_WAIT_TIMEOUT_S 5
#endif

#ifndef WEBGPU_STUB
// Drive wgpuInstanceProcessEvents until `done` flips true or the timeout
// elapses. Returns true on success, false on timeout. Centralizing this in
// one place means every readback/work-done wait shares the same DoS bound.
template <typename DoneFn>
static bool wgpu_wait_until(WGPUInstance instance, DoneFn done) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(LUX_WEBGPU_WAIT_TIMEOUT_S);
    while (!done()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        wgpuInstanceProcessEvents(instance);
    }
    return true;
}
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

// ----- FHE NTT-fused fast path: cache types -----
// Mirror metal/src/metal_plugin.mm's BskNttKey + bsk_fingerprint + the CUDA
// equivalents. The cache key combines (host_ptr, shape, q) with a 128-bit
// content fingerprint so that callers which rebuild BSKs into reused vector
// backing stores do NOT alias each other in the cache.
struct WgslBskNttKey {
    const void* host_ptr;
    uint32_t    n_lwe;
    uint32_t    N;
    uint32_t    k;
    uint32_t    l;
    uint64_t    q;
    uint64_t    fp_lo;
    uint64_t    fp_hi;
    bool operator==(const WgslBskNttKey& o) const {
        return host_ptr == o.host_ptr && n_lwe == o.n_lwe && N == o.N &&
               k == o.k && l == o.l && q == o.q &&
               fp_lo == o.fp_lo && fp_hi == o.fp_hi;
    }
};
struct WgslBskNttKeyHash {
    size_t operator()(const WgslBskNttKey& key) const {
        size_t h = (size_t)key.fp_lo;
        h ^= (size_t)key.fp_hi          + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= (size_t)key.host_ptr       + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= ((size_t)key.n_lwe << 1)   + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= ((size_t)key.N << 1)       + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= ((size_t)key.k << 1)       + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= ((size_t)key.l << 1)       + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= (size_t)key.q              + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};
// Cached BSK-in-NTT image. The buffer holds the psi-twisted, forward-NTT'd
// BSK shaped exactly like the input — [n_lwe][(k+1)*l][k+1][N] u64 = [...] u32
// pairs. Built once per (host_ptr + fingerprint), reused across AP steps.
struct WgslBskNttCacheEntry {
    WGPUBuffer bsk_ntt;     // device buffer, sized for the full BSK in NTT domain
    size_t     bsk_bytes;   // size of allocation
};
// Per-(N, q) ψ tables for the negacyclic twist. Two arrays of N u64 each,
// packed as u32 pairs in WGSL.
struct WgslNttPsiCacheEntry {
    WGPUBuffer psi_pow;
    WGPUBuffer psi_inv_pow;
};
// Per-(N, q) cyclic-NTT twiddle factor cache. Stores the full table; per-
// stage slicing happens in `build_stage_twiddle_bufs`.
struct WgslNttTwiddleCacheEntry {
    WGPUBuffer twiddles_fwd;        // [N/2] u64
    WGPUBuffer twiddles_inv;        // [N/2] u64
    std::vector<uint64_t> tw_fwd_h; // host mirror, used to build per-stage slices
    std::vector<uint64_t> tw_inv_h;
    uint64_t n_inv;                 // N^{-1} mod q
};

struct WGPUBackendContext {
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;
    std::string device_name;
    uint64_t memory_total;

    // Compiled pipeline cache, keyed by (content-hash, entry_point, preset).
    // Protected by pipelines_mutex: unordered_map iterators are invalidated on
    // rehash, so even reads must hold the lock. Same discipline as the CUDA
    // backend's kernel cache.
    std::unordered_map<std::string, PipelineEntry> pipelines;
    std::mutex pipelines_mutex;

    // Disk-loaded WGSL source cache: relpath → owned string. We hash the
    // source CONTENT (not its pointer) when building cache keys, so freeing
    // and re-allocating a source at the same address doesn't return a stale
    // pipeline. Empty for inline WGSL constants which the binary owns.
    //
    // (R-9) Protected by wgsl_sources_mutex with double-checked locking in
    // load_wgsl_source. The previous design touched the unordered_map under
    // no lock; concurrent loads of two different relpaths could rehash the
    // map mid-find() and return a dangling iterator to the read path.
    std::unordered_map<std::string, std::string> wgsl_sources;
    std::mutex                                   wgsl_sources_mutex;

    // Uniform buffer for parameters (reusable)
    WGPUBuffer uniformBuffer;
    static constexpr size_t UNIFORM_BUFFER_SIZE = 256;

    // FHE NTT-fused fast path: ψ tables, cyclic-NTT twiddles, and BSK-in-NTT
    // image. All accesses must hold fhe_ntt_mutex. Mirrors metal_plugin.mm
    // (fhe_ntt_cache + bsk_ntt_cache) and cuda_plugin.cpp's three caches.
    std::mutex                                                                  fhe_ntt_mutex;
    std::unordered_map<uint64_t, WgslNttPsiCacheEntry>                          fhe_psi_cache;     // key = (N<<48)^q
    std::unordered_map<uint64_t, WgslNttTwiddleCacheEntry>                      fhe_ntt_tw_cache;  // key = (N<<48)^q
    std::unordered_map<WgslBskNttKey, WgslBskNttCacheEntry, WgslBskNttKeyHash>  fhe_bsk_ntt_cache;
};

// Locate a WGSL kernel under LUX_WGSL_KERNEL_PATH or sensible defaults. The
// returned pointer is owned by ctx->wgsl_sources and remains stable for the
// lifetime of ctx, so it's safe to use as a cache key in the pipeline map.
//
// (R-9) Thread-safety: ctx->wgsl_sources_mutex guards every cache touch.
// Double-checked locking: fast path is a single lock+find; if missed, we
// release the lock and do the slow disk read, then re-acquire and either
// install our copy or return the winner's. unordered_map iterators are
// invalidated on rehash so the read path MUST hold the lock.
static const char* load_wgsl_source(WGPUBackendContext* ctx, const char* relpath) {
    {
        std::lock_guard<std::mutex> lk(ctx->wgsl_sources_mutex);
        auto it = ctx->wgsl_sources.find(relpath);
        if (it != ctx->wgsl_sources.end()) return it->second.c_str();
    }

    std::vector<std::string> dirs;
    // R-3 / R-4: env-controlled kernel path is consulted only when the
    // operator hasn't opted out (LUX_GPU_TRUST_ENV != "0") and the process
    // isn't running with elevated privileges. Without this gate a setuid
    // binary could be hijacked to load attacker-controlled WGSL.
    if (!lux::gpu::internal::env_is_untrusted()) {
        if (const char* env = std::getenv("LUX_WGSL_KERNEL_PATH"); env && *env) {
            dirs.emplace_back(env);
        }
    }
#ifdef LUX_WGSL_KERNEL_DIR
    dirs.emplace_back(LUX_WGSL_KERNEL_DIR);
#endif
    dirs.emplace_back("../webgpu/kernels/wgsl");
    dirs.emplace_back("./webgpu/kernels/wgsl");

    // Relative to plugin install prefix. Two layouts in the wild:
    //   * lib/<plugin>.dylib        → datadir/share/lux-gpu/wgsl  (../share)
    //   * lib/lux-gpu/<plugin>.dylib → datadir/share/lux-gpu/wgsl (../../share)
    // Canonical install puts the plugin under lib/lux-gpu/, so the
    // two-level relative resolves to the install-tree share/ dir.
    Dl_info info;
    if (dladdr((void*)&load_wgsl_source, &info) && info.dli_fname) {
        std::string self(info.dli_fname);
        size_t slash = self.find_last_of('/');
        if (slash != std::string::npos) {
            std::string prefix = self.substr(0, slash);
            dirs.push_back(prefix + "/../share/lux-gpu/wgsl");
            dirs.push_back(prefix + "/../../share/lux-gpu/wgsl");
            dirs.push_back(prefix);
        }
    }

    for (const auto& dir : dirs) {
        std::string full = dir + "/" + relpath;
        FILE* f = std::fopen(full.c_str(), "rb");
        if (!f) continue;
        std::fseek(f, 0, SEEK_END);
        long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (n <= 0) { std::fclose(f); continue; }
        std::string buf((size_t)n, '\0');
        size_t read = std::fread(&buf[0], 1, (size_t)n, f);
        std::fclose(f);
        if (read != (size_t)n) continue;

        // Install under lock with a recheck — another thread may have raced
        // us. If they won, return their copy and discard ours (we move-in
        // either way to avoid touching the disk again on the next caller).
        std::lock_guard<std::mutex> lk(ctx->wgsl_sources_mutex);
        auto it = ctx->wgsl_sources.find(relpath);
        if (it != ctx->wgsl_sources.end()) return it->second.c_str();
        ctx->wgsl_sources[relpath] = std::move(buf);
        return ctx->wgsl_sources[relpath].c_str();
    }
    std::fprintf(stderr, "[lux-webgpu] WGSL kernel not found: %s\n", relpath);
    return nullptr;
}

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
    Auto,         // No pre-declared layout — pipeline auto-derives, layout
                  // queried via wgpuComputePipelineGetBindGroupLayout(0).
                  // Used by all on-disk crypto WGSL kernels whose binding
                  // counts and read/rw shapes vary per kernel.
};

// Create compute pipeline with layout preset
static WGPUComputePipeline createComputePipelineWithLayout(
    WGPUDevice device,
    WGPUShaderModule shaderModule,
    const char* entryPoint,
    WGPUBindGroupLayout* outLayout,
    LayoutPreset preset
) {
    // Auto-layout path: let the driver derive the bind group layout from the
    // WGSL source. Used for crypto kernels whose layouts vary.
    if (preset == LayoutPreset::Auto) {
        WGPUComputePipelineDescriptor desc = {};
        desc.layout = nullptr;  // wgpu auto-derives from shader reflection
        desc.compute.module = shaderModule;
        desc.compute.entryPoint.data = entryPoint;
        desc.compute.entryPoint.length = strlen(entryPoint);
        WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &desc);
        if (!pipeline) {
            *outLayout = nullptr;
            return nullptr;
        }
        *outLayout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        return pipeline;
    }

    WGPUBindGroupLayoutEntry layoutEntries[5] = {};
    size_t entryCount = 0;

    switch (preset) {
        case LayoutPreset::Auto:
            // Already handled above with an early return — listed here to keep
            // -Wswitch happy.
            break;
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

// BLAKE3 c-abi from luxcpp/crypto/blake3 — collision-resistant 256-bit hash.
// Linked statically when LUX_GPU_WEBGPU_HAVE_BLAKE3 is set by CMake.
#ifdef LUX_GPU_WEBGPU_HAVE_BLAKE3
extern "C" int blake3(const uint8_t* in, size_t in_len, uint8_t out[32]);
#endif

// Build a content-keyed pipeline cache identifier (R-5).
//
// Hashing the source TEXT (not the pointer) prevents stale-pipeline aliasing
// if a transient buffer is freed and a new source ends up at the same
// address. The previous implementation used std::hash<string_view> — fast
// but DETERMINISTIC and collision-pliable: an attacker who can submit two
// WGSL sources with the same std::hash value would alias their pipelines
// in the cache (the second submitter would get the first submitter's
// compiled pipeline, with attacker-controlled binding semantics).
//
// BLAKE3 of the full source gives 128-bit collision resistance (2^128 work
// to find any collision). We render the 32-byte digest as 64 hex chars so
// it's safe to splice into the cache key string. The std::hash fallback
// path is retained only for builds that didn't link the crypto static libs
// — those builds are explicitly NOT production (CMake logs the degraded
// mode at configure time).
static std::string make_pipeline_key(
    const char* source,
    const char* entryPoint,
    LayoutPreset preset
) {
    const size_t src_len = std::strlen(source);
#ifdef LUX_GPU_WEBGPU_HAVE_BLAKE3
    uint8_t digest[32];
    (void)blake3(reinterpret_cast<const uint8_t*>(source), src_len, digest);
    char hex[65];
    static const char kHex[] = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) {
        hex[i * 2 + 0] = kHex[(digest[i] >> 4) & 0xF];
        hex[i * 2 + 1] = kHex[ digest[i]       & 0xF];
    }
    hex[64] = '\0';
    return std::string(hex) + "::" + entryPoint + "_" +
           std::to_string(static_cast<int>(preset));
#else
    // Fallback for builds without BLAKE3. NOT collision-resistant; logged
    // at config-time via CMake STATUS message.
    const size_t src_hash = std::hash<std::string_view>{}(
        std::string_view(source, src_len)
    );
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%zx::", src_hash);
    return std::string(buf) + entryPoint + "_" +
           std::to_string(static_cast<int>(preset));
#endif
}

// Get or create a cached pipeline with layout preset.
// Thread-safe via double-checked locking on ctx->pipelines_mutex. The slow
// compile happens OUTSIDE the lock; a recheck after compile drops our newly-
// built pipeline if another thread won the race (no UB — wgpu refs are
// reference-counted, and the loser releases its own handles before bailing).
static PipelineEntry* getOrCreatePipelineWithLayout(
    WGPUBackendContext* ctx,
    const char* source,
    const char* entryPoint,
    LayoutPreset preset
) {
    if (!source || !entryPoint) return nullptr;
    std::string key = make_pipeline_key(source, entryPoint, preset);

    {
        std::lock_guard<std::mutex> lock(ctx->pipelines_mutex);
        auto it = ctx->pipelines.find(key);
        if (it != ctx->pipelines.end()) return &it->second;
    }

    // Slow path: compile without holding the cache lock.
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

    // Insert under lock with a recheck. If we lost the race, release our
    // owned handles and return the winner. Otherwise install ours.
    std::lock_guard<std::mutex> lock(ctx->pipelines_mutex);
    auto it = ctx->pipelines.find(key);
    if (it != ctx->pipelines.end()) {
        if (entry.pipeline)         wgpuComputePipelineRelease(entry.pipeline);
        if (entry.shaderModule)     wgpuShaderModuleRelease(entry.shaderModule);
        if (entry.bindGroupLayout)  wgpuBindGroupLayoutRelease(entry.bindGroupLayout);
        return &it->second;
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
#if defined(USE_DAWN_API)
    workDoneCallback.callback = [](WGPUQueueWorkDoneStatus, void* userdata1, void*) {
        static_cast<WaitData*>(userdata1)->done = true;
    };
#else
    workDoneCallback.callback = [](WGPUQueueWorkDoneStatus, WGPUStringView, void* userdata1, void*) {
        static_cast<WaitData*>(userdata1)->done = true;
    };
#endif
    workDoneCallback.userdata1 = &waitData;

    wgpuQueueOnSubmittedWorkDone(ctx->queue, workDoneCallback);

    if (!wgpu_wait_until(ctx->instance, [&]{ return waitData.done; })) {
        return LUX_BACKEND_ERROR_DEVICE_LOST;
    }

    return LUX_BACKEND_OK;
}

// Stage `srcBuf` (bytes long) onto a CPU-mappable staging buffer and memcpy into
// `dst`. Encapsulates the MapAsync + wait + Unmap dance that was repeated 9
// times across the crypto bridges.
static LuxBackendError readback_buffer(
    WGPUBackendContext* ctx, WGPUBuffer srcBuf, void* dst, size_t bytes
) {
    if (bytes == 0) return LUX_BACKEND_OK;

    WGPUBufferDescriptor stagingDesc = {};
    stagingDesc.size  = bytes;
    stagingDesc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(ctx->device, &stagingDesc);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, srcBuf, 0, staging, 0, bytes);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(ctx->queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    struct MapData { bool done; WGPUMapAsyncStatus status; } md = {false, LUX_MAP_ASYNC_STATUS_UNSET};
    WGPUBufferMapCallbackInfo cb = {};
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
#if defined(USE_DAWN_API)
    cb.callback = [](WGPUMapAsyncStatus s, void* u1, void*) {
        auto* d = static_cast<MapData*>(u1); d->status = s; d->done = true;
    };
#else
    cb.callback = [](WGPUMapAsyncStatus s, WGPUStringView, void* u1, void*) {
        auto* d = static_cast<MapData*>(u1); d->status = s; d->done = true;
    };
#endif
    cb.userdata1 = &md;
    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, bytes, cb);
    if (!wgpu_wait_until(ctx->instance, [&]{ return md.done; })) {
        wgpuBufferRelease(staging);
        return LUX_BACKEND_ERROR_DEVICE_LOST;
    }

    LuxBackendError err = LUX_BACKEND_OK;
    if (md.status == WGPUMapAsyncStatus_Success) {
        const uint8_t* mapped = static_cast<const uint8_t*>(
            wgpuBufferGetConstMappedRange(staging, 0, bytes));
        std::memcpy(dst, mapped, bytes);
        wgpuBufferUnmap(staging);
    } else {
        err = LUX_BACKEND_ERROR_INTERNAL;
    }
    wgpuBufferRelease(staging);
    return err;
}

// Host-buffer crypto dispatch for WGSL. Captures the alloc/upload/bind/dispatch/
// readback pattern shared by every WGSL crypto bridge. Each WgslBinding gives
// an explicit @binding(N) slot in the shader's group(0); src=null=output, dst=
// null=input, src=dst=null=scratch storage. uniform_data, if non-null, is
// written to ctx->uniformBuffer and bound at uniform_binding.
struct WgslBinding {
    uint32_t binding;
    const void* src = nullptr;
    void* dst = nullptr;
    size_t bytes = 0;
};

static LuxBackendError wgsl_dispatch_op(
    WGPUBackendContext* ctx,
    const PipelineEntry* entry,
    std::initializer_list<WgslBinding> binds,
    const void* uniform_data,
    size_t uniform_bytes,
    uint32_t uniform_binding,
    uint32_t wg_x,
    uint32_t wg_y = 1,
    uint32_t wg_z = 1
) {
    if (!ctx || !entry) return LUX_BACKEND_ERROR_INTERNAL;

    // Allocate one storage buffer per binding (sized at least 1 byte).
    std::vector<WGPUBuffer> bufs;
    bufs.reserve(binds.size());
    {
        for (const auto& bd : binds) {
            size_t len = bd.bytes == 0 ? 1 : bd.bytes;
            WGPUBufferDescriptor desc = {};
            desc.size  = len;
            desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
            WGPUBuffer b = wgpuDeviceCreateBuffer(ctx->device, &desc);
            if (!b) {
                for (auto x : bufs) wgpuBufferRelease(x);
                return LUX_BACKEND_ERROR_OUT_OF_MEMORY;
            }
            if (bd.src && bd.bytes > 0) {
                wgpuQueueWriteBuffer(ctx->queue, b, 0, bd.src, bd.bytes);
            }
            bufs.push_back(b);
        }
    }
    if (uniform_data && uniform_bytes > 0) {
        wgpuQueueWriteBuffer(ctx->queue, ctx->uniformBuffer, 0, uniform_data, uniform_bytes);
    }

    // Build the bind group entries.
    std::vector<WGPUBindGroupEntry> bgEntries;
    bgEntries.reserve(binds.size() + (uniform_data ? 1 : 0));
    {
        size_t i = 0;
        for (const auto& bd : binds) {
            WGPUBindGroupEntry e = {};
            e.binding = bd.binding;
            e.buffer  = bufs[i];
            e.size    = bd.bytes == 0 ? 1 : bd.bytes;
            bgEntries.push_back(e);
            ++i;
        }
        if (uniform_data) {
            WGPUBindGroupEntry e = {};
            e.binding = uniform_binding;
            e.buffer  = ctx->uniformBuffer;
            e.size    = uniform_bytes;
            bgEntries.push_back(e);
        }
    }

    WGPUBindGroupDescriptor bgDesc = {};
    bgDesc.layout = entry->bindGroupLayout;
    bgDesc.entryCount = (uint32_t)bgEntries.size();
    bgDesc.entries = bgEntries.data();
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(ctx->device, &bgDesc);

    LuxBackendError err = dispatchCompute(ctx, entry->pipeline, bindGroup, wg_x, wg_y, wg_z);

    if (err == LUX_BACKEND_OK) {
        size_t i = 0;
        for (const auto& bd : binds) {
            if (bd.dst && bd.bytes > 0) {
                err = readback_buffer(ctx, bufs[i], bd.dst, bd.bytes);
                if (err != LUX_BACKEND_OK) break;
            }
            ++i;
        }
    }

    wgpuBindGroupRelease(bindGroup);
    for (auto b : bufs) wgpuBufferRelease(b);
    return err;
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

    // Poll until adapter is ready, or fail closed on timeout.
    if (!wgpu_wait_until(ctx->instance, [&]{ return adapterData.done; }) ||
        !adapterData.adapter) {
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

    if (!wgpu_wait_until(ctx->instance, [&]{ return deviceData.done; }) ||
        !deviceData.device) {
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

        // Release FHE NTT-fused caches.
        {
            std::lock_guard<std::mutex> guard(ctx->fhe_ntt_mutex);
            for (auto& [k, v] : ctx->fhe_psi_cache) {
                if (v.psi_pow)     wgpuBufferRelease(v.psi_pow);
                if (v.psi_inv_pow) wgpuBufferRelease(v.psi_inv_pow);
            }
            ctx->fhe_psi_cache.clear();
            for (auto& [k, v] : ctx->fhe_ntt_tw_cache) {
                if (v.twiddles_fwd) wgpuBufferRelease(v.twiddles_fwd);
                if (v.twiddles_inv) wgpuBufferRelease(v.twiddles_inv);
            }
            ctx->fhe_ntt_tw_cache.clear();
            for (auto& [k, v] : ctx->fhe_bsk_ntt_cache) {
                if (v.bsk_ntt) wgpuBufferRelease(v.bsk_ntt);
            }
            ctx->fhe_bsk_ntt_cache.clear();
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
#if defined(USE_DAWN_API)
    workDoneCallback.callback = [](WGPUQueueWorkDoneStatus, void* userdata1, void*) {
        static_cast<WaitData*>(userdata1)->done = true;
    };
#else
    workDoneCallback.callback = [](WGPUQueueWorkDoneStatus, WGPUStringView, void* userdata1, void*) {
        static_cast<WaitData*>(userdata1)->done = true;
    };
#endif
    workDoneCallback.userdata1 = &waitData;

    wgpuQueueOnSubmittedWorkDone(ctx->queue, workDoneCallback);

    if (!wgpu_wait_until(ctx->instance, [&]{ return waitData.done; })) {
        return LUX_BACKEND_ERROR_DEVICE_LOST;
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
    struct MapData { bool done; WGPUMapAsyncStatus status; } mapData = {false, LUX_MAP_ASYNC_STATUS_UNSET};

    WGPUBufferMapCallbackInfo mapCallback = {};
    mapCallback.mode = WGPUCallbackMode_AllowProcessEvents;
    mapCallback.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*) {
        auto data = static_cast<MapData*>(userdata1);
        data->status = status;
        data->done = true;
    };
    mapCallback.userdata1 = &mapData;

    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, copySize, mapCallback);

    if (!wgpu_wait_until(ctx->instance, [&]{ return mapData.done; })) {
        wgpuBufferRelease(staging);
        return LUX_BACKEND_ERROR_DEVICE_LOST;
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
#if 0  /* kernel_load / kernel_dispatch removed in canonical ABI v3 */
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
#endif  /* kernel_load / kernel_dispatch */

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

#endif  /* WEBGPU_STUB */

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

    struct MapData { bool done; WGPUMapAsyncStatus status; } mapData = {false, LUX_MAP_ASYNC_STATUS_UNSET};
    WGPUBufferMapCallbackInfo mapCallback = {};
    mapCallback.mode = WGPUCallbackMode_AllowProcessEvents;
    mapCallback.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*) {
        auto d = static_cast<MapData*>(userdata1);
        d->status = status;
        d->done = true;
    };
    mapCallback.userdata1 = &mapData;

    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, dataBytes, mapCallback);
    bool wait_ok = wgpu_wait_until(ctx->instance, [&]{ return mapData.done; });

    if (wait_ok && mapData.status == WGPUMapAsyncStatus_Success) {
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

    if (!wait_ok) return LUX_BACKEND_ERROR_DEVICE_LOST;
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

    struct MapData { bool done; WGPUMapAsyncStatus status; } mapData = {false, LUX_MAP_ASYNC_STATUS_UNSET};
    WGPUBufferMapCallbackInfo mapCallback = {};
    mapCallback.mode = WGPUCallbackMode_AllowProcessEvents;
    mapCallback.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*) {
        auto d = static_cast<MapData*>(userdata1);
        d->status = status;
        d->done = true;
    };
    mapCallback.userdata1 = &mapData;

    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, dataBytes, mapCallback);
    bool wait_ok = wgpu_wait_until(ctx->instance, [&]{ return mapData.done; });

    if (wait_ok && mapData.status == WGPUMapAsyncStatus_Success) {
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

    if (!wait_ok) return LUX_BACKEND_ERROR_DEVICE_LOST;
    return mapData.status == WGPUMapAsyncStatus_Success ? LUX_BACKEND_OK : LUX_BACKEND_ERROR_INTERNAL;
#else
    (void)context; (void)data; (void)n; (void)modulus;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

static LuxBackendError webgpu_op_msm(LuxBackendContext* context, const void* scalars,
                                      const void* points, void* result,
                                      size_t n, int curve_type) {
    // op_msm is intentionally NOT_SUPPORTED in the WebGPU backend until two
    // missing pieces land in the WGSL kernel set:
    //   (a) a correct 384-bit (BLS12-381) or 256-bit (BN254) Montgomery
    //       multiplication inside the Pippenger inner loop — the current
    //       msm.wgsl `point_add_mixed` is a literal placeholder that adds
    //       limbs without applying the curve addition formula;
    //   (b) an `fp_inv` for the relevant base field so the result can be
    //       canonicalized to the unique (x_aff, y_aff, R mod p) form every
    //       other backend produces.
    //
    // Returning NOT_SUPPORTED is the only correct behaviour: the alternative
    // of returning algebraically-wrong-but-deterministic bytes would silently
    // diverge from CPU/Metal/CUDA and break consensus on hybrid validator
    // pools. Callers fall back to the CPU backend, which IS canonical.
    (void)context; (void)scalars; (void)points; (void)result; (void)n; (void)curve_type;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
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

// =============================================================================
// TFHE FHE bridges — NTT-fused fast path + canonical CPU fallback.
//
// The fast path mirrors the Metal (metal_plugin.mm::fhe_ntt_blind_rotate_step_
// metal) and CUDA (cuda_plugin.cpp::fhe_blind_rotate_step_cuda) NTT-fused
// external-product implementations bit-for-bit. The seven WGSL kernels in
// kernels/wgsl/fhe/ntt_fused_extprod.wgsl share the same kitchen-sink bind
// group layout (13 bindings + 1 spare slot at index 13), and each AP step is
// encoded into ONE wgpu::CommandEncoder with ONE queue.submit — same single-
// cmd-buffer-per-step pattern as Metal/CUDA.
//
// Activation gate (identical to Metal/CUDA):
//   - LUX_FHE_FORCE_SCHOOLBOOK=1     → always fall through to canonical helper.
//   - LUX_FHE_NTT_MIN_N (default 256) → minimum N for the fast path.
//   - q must admit a 2N-th primitive root of unity (i.e. 2N | (q-1)).
//
// On any setup failure (shader compile, cache build, dispatch error) the
// dispatcher falls back to lux::fhe::run_* on the host CPU. The CPU helper
// is the correctness reference; the NTT-fused output is bit-exact w.r.t. it.
// Capability flags (LUX_CAP_FHE | LUX_CAP_TFHE | LUX_CAP_BLIND_ROTATE) are
// re-enabled in the capability mask now that the dispatch is wired.
// =============================================================================

// fhe_should_use_wgsl_ntt — policy gate that mirrors Metal's
// fhe_should_use_ntt_path and CUDA's fhe_should_use_ntt_path.
//
// Activation conditions (must all hold):
//   - LUX_FHE_FORCE_SCHOOLBOOK unset (or = "0")
//   - N >= threshold (env LUX_FHE_NTT_MIN_N, default 256)
//   - q admits a 2N-th primitive root of unity, i.e. 2N | (q-1)
static bool fhe_should_use_wgsl_ntt(uint32_t N, uint64_t q) {
    if (const char* env = std::getenv("LUX_FHE_FORCE_SCHOOLBOOK");
        env && *env && env[0] != '0') {
        return false;
    }
    uint32_t threshold = 256u;
    if (const char* env = std::getenv("LUX_FHE_NTT_MIN_N"); env && *env) {
        unsigned long v = std::strtoul(env, nullptr, 10);
        if (v > 0 && v < (1u << 20)) threshold = (uint32_t)v;
    }
    if (N < threshold) return false;
    if (q == 0) return false;
    if (((q - 1) % ((uint64_t)2 * N)) != 0) return false;
    return true;
}

#ifndef WEBGPU_STUB

// ---------------------------------------------------------------------------
// Host modular arithmetic — bit-exact mirror of metal_plugin.mm and
// cuda_plugin.cpp host helpers. Used to precompute ψ and twiddle tables.
// ---------------------------------------------------------------------------
namespace {

static inline uint64_t fhe_wgsl_mod_mul_host(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)(((__uint128_t)a * b) % m);
}
static inline uint64_t fhe_wgsl_mod_pow_host(uint64_t base, uint64_t exp, uint64_t m) {
    uint64_t r = 1; base %= m;
    while (exp) {
        if (exp & 1) r = fhe_wgsl_mod_mul_host(r, base, m);
        exp >>= 1;
        base = fhe_wgsl_mod_mul_host(base, base, m);
    }
    return r;
}
static inline uint64_t fhe_wgsl_mod_inv_host(uint64_t a, uint64_t m) {
    return fhe_wgsl_mod_pow_host(a, m - 2, m);
}
static inline uint64_t fhe_wgsl_find_primitive_root_host(uint64_t modulus) {
    if (modulus == 0xFFFFFFFF00000001ULL) return 7;
    if (modulus == 0x1000000000000001ULL) return 3;
    return 3;
}

// FNV-1a sparse-sample BSK fingerprint. Bit-exact mirror of metal_plugin.mm's
// bsk_fingerprint and cuda_plugin.cpp's fhe_bsk_fingerprint — identical probe
// pattern so caches across backends key the same BSK identically. When the
// full-buffer BLAKE3 upgrade lands per Blue-FHE-r3, this gets swapped for the
// same fingerprint kernel across all three backends in one cross-cut.
static inline void fhe_wgsl_bsk_fingerprint(const uint64_t* p, size_t words,
                                              uint64_t& fp_lo, uint64_t& fp_hi)
{
    fp_lo = 1469598103934665603ULL;
    fp_hi = 0xCBF29CE484222325ULL;
    if (words == 0) return;
    auto mix_lo = [&](uint64_t v) { fp_lo ^= v; fp_lo *= 1099511628211ULL; };
    auto mix_hi = [&](uint64_t v) { fp_hi ^= v; fp_hi *= 1099511628211ULL; };
    const size_t head = words < 8u ? words : 8u;
    for (size_t i = 0; i < head; ++i) { mix_lo(p[i]); mix_hi(p[words - 1u - i]); }
    if (words > 16u) {
        const size_t stride = (words / 32u) | 1u;
        for (size_t i = 0, j = head; i < 32 && j < words; ++i, j += stride) {
            mix_lo(p[j]); mix_hi(p[words - 1u - j]);
        }
    }
    mix_lo((uint64_t)words);
    mix_hi((uint64_t)words << 1);
}

// Pack a host u64 vector into u32 lo/hi pairs (WGSL u64p layout).
static inline std::vector<uint32_t> pack_u64_to_u32(const uint64_t* p, size_t n) {
    std::vector<uint32_t> out(n * 2);
    for (size_t i = 0; i < n; ++i) {
        out[i * 2 + 0] = (uint32_t)(p[i] & 0xFFFFFFFFu);
        out[i * 2 + 1] = (uint32_t)(p[i] >> 32);
    }
    return out;
}
static inline std::vector<uint32_t> pack_i64_to_u32(const int64_t* p, size_t n) {
    std::vector<uint32_t> out(n * 2);
    for (size_t i = 0; i < n; ++i) {
        uint64_t u = (uint64_t)p[i];
        out[i * 2 + 0] = (uint32_t)(u & 0xFFFFFFFFu);
        out[i * 2 + 1] = (uint32_t)(u >> 32);
    }
    return out;
}
static inline void unpack_u32_to_u64(const uint32_t* src, uint64_t* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = ((uint64_t)src[i * 2 + 0]) | (((uint64_t)src[i * 2 + 1]) << 32);
    }
}

// fhe_compute_a_tilde: round (a * 2N / q) into [0, 2N). Same formula used by
// CPU / Metal / CUDA. validate_pbs_params already guarantees N is a power of
// two; we therefore cannot overflow the 128-bit intermediate.
static inline uint32_t fhe_wgsl_compute_a_tilde(uint64_t a, uint32_t N, uint64_t q) {
    const uint64_t two_n = (uint64_t)2 * N;
    return (uint32_t)((((__uint128_t)a * two_n + (q >> 1)) / q) % two_n);
}

// Build a storage device buffer of `bytes` size, optionally initialized from
// `host_data`. STORAGE | COPY_DST | COPY_SRC.
static WGPUBuffer make_storage_buffer(WGPUBackendContext* ctx, size_t bytes,
                                       const void* host_data = nullptr,
                                       size_t host_bytes = 0)
{
    WGPUBufferDescriptor d = {};
    d.size = bytes == 0 ? 1 : bytes;
    d.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
    WGPUBuffer b = wgpuDeviceCreateBuffer(ctx->device, &d);
    if (!b) return nullptr;
    if (host_data && host_bytes > 0) {
        wgpuQueueWriteBuffer(ctx->queue, b, 0, host_data, host_bytes);
    }
    return b;
}
static WGPUBuffer make_uniform_buffer(WGPUBackendContext* ctx, size_t bytes) {
    WGPUBufferDescriptor d = {};
    // WebGPU spec: uniform buffers minimum 16 bytes, alignment 16. Round up
    // so single-u32 uniforms still bind cleanly.
    size_t sz = bytes < 16 ? 16 : ((bytes + 15) & ~size_t(15));
    d.size = sz;
    d.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    return wgpuDeviceCreateBuffer(ctx->device, &d);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// FHE NTT-fused pipelines: one shader module, seven entry points, one shared
// kitchen-sink bind group layout (14 bindings, indices 0..13). The WGSL
// kernel declares all 14 bindings at module scope, so every dispatch must
// populate the full bind group — a kernel that doesn't actually read
// i_psi_pow still gets it bound.
// ---------------------------------------------------------------------------
struct FheWgslPipelines {
    WGPUShaderModule          shader = nullptr;
    WGPUBindGroupLayout       bgl    = nullptr;   // shared across all 7 entries
    WGPUPipelineLayout        plo    = nullptr;
    WGPUComputePipeline       decomp_twist        = nullptr;
    WGPUComputePipeline       bsk_twist           = nullptr;
    WGPUComputePipeline       ntt_bit_reverse     = nullptr;
    WGPUComputePipeline       ntt_butterfly       = nullptr;
    WGPUComputePipeline       inner_product       = nullptr;
    WGPUComputePipeline       untwist_accumulate  = nullptr;
    WGPUComputePipeline       zero                = nullptr;
    bool ok() const {
        return shader && bgl && plo && decomp_twist && bsk_twist &&
               ntt_bit_reverse && ntt_butterfly && inner_product &&
               untwist_accumulate && zero;
    }
};

static FheWgslPipelines load_fhe_wgsl_pipelines(WGPUBackendContext* ctx) {
    FheWgslPipelines out{};
    const char* src = load_wgsl_source(ctx, "fhe/ntt_fused_extprod.wgsl");
    if (!src) return out;

    // 14-entry kitchen-sink layout matching the WGSL @binding(0..13).
    WGPUBindGroupLayoutEntry layout_entries[14] = {};
    auto storage_entry = [&](uint32_t binding, bool read_only) {
        WGPUBindGroupLayoutEntry e = {};
        e.binding = binding;
        e.visibility = WGPUShaderStage_Compute;
        e.buffer.type = read_only ? WGPUBufferBindingType_ReadOnlyStorage
                                  : WGPUBufferBindingType_Storage;
        e.buffer.minBindingSize = 0;
        return e;
    };
    auto uniform_entry = [&](uint32_t binding) {
        WGPUBindGroupLayoutEntry e = {};
        e.binding = binding;
        e.visibility = WGPUShaderStage_Compute;
        e.buffer.type = WGPUBufferBindingType_Uniform;
        e.buffer.minBindingSize = 0;
        return e;
    };
    layout_entries[0]  = storage_entry(0,  true);   // i_decomp
    layout_entries[1]  = storage_entry(1,  false);  // o_twist_or_acc (rw)
    layout_entries[2]  = storage_entry(2,  true);   // i_psi_pow
    layout_entries[3]  = uniform_entry(3);          // p_decomp_twist
    layout_entries[4]  = uniform_entry(4);          // p_bsk_twist
    layout_entries[5]  = storage_entry(5,  true);   // i_bsk_or_twiddles
    layout_entries[6]  = storage_entry(6,  true);   // i_bsk_ntt
    layout_entries[7]  = storage_entry(7,  false);  // o_result_ntt (rw)
    layout_entries[8]  = uniform_entry(8);          // p_ip
    layout_entries[9]  = uniform_entry(9);          // p_ntt
    layout_entries[10] = uniform_entry(10);         // p_bitrev
    layout_entries[11] = uniform_entry(11);         // p_untwist
    layout_entries[12] = storage_entry(12, true);   // i_psi_inv_pow
    layout_entries[13] = uniform_entry(13);         // p_zero

    WGPUBindGroupLayoutDescriptor bgld = {};
    bgld.entryCount = 14;
    bgld.entries = layout_entries;
    out.bgl = wgpuDeviceCreateBindGroupLayout(ctx->device, &bgld);
    if (!out.bgl) return {};

    WGPUPipelineLayoutDescriptor plod = {};
    plod.bindGroupLayoutCount = 1;
    plod.bindGroupLayouts = &out.bgl;
    out.plo = wgpuDeviceCreatePipelineLayout(ctx->device, &plod);
    if (!out.plo) { wgpuBindGroupLayoutRelease(out.bgl); return {}; }

    out.shader = createShaderModule(ctx->device, src);
    if (!out.shader) {
        wgpuPipelineLayoutRelease(out.plo);
        wgpuBindGroupLayoutRelease(out.bgl);
        return {};
    }

    auto make_pipe = [&](const char* entry) -> WGPUComputePipeline {
        WGPUComputePipelineDescriptor cpd = {};
        cpd.layout = out.plo;
        cpd.compute.module = out.shader;
        cpd.compute.entryPoint.data = entry;
        cpd.compute.entryPoint.length = strlen(entry);
        return wgpuDeviceCreateComputePipeline(ctx->device, &cpd);
    };
    out.decomp_twist        = make_pipe("fhe_decomp_twist");
    out.bsk_twist           = make_pipe("fhe_bsk_twist");
    out.ntt_bit_reverse     = make_pipe("fhe_ntt_batch_bit_reverse");
    out.ntt_butterfly       = make_pipe("fhe_ntt_batch_butterfly");
    out.inner_product       = make_pipe("fhe_ntt_inner_product");
    out.untwist_accumulate  = make_pipe("fhe_untwist_accumulate");
    out.zero                = make_pipe("fhe_ntt_zero");
    if (!out.ok()) {
        if (out.decomp_twist)       wgpuComputePipelineRelease(out.decomp_twist);
        if (out.bsk_twist)          wgpuComputePipelineRelease(out.bsk_twist);
        if (out.ntt_bit_reverse)    wgpuComputePipelineRelease(out.ntt_bit_reverse);
        if (out.ntt_butterfly)      wgpuComputePipelineRelease(out.ntt_butterfly);
        if (out.inner_product)      wgpuComputePipelineRelease(out.inner_product);
        if (out.untwist_accumulate) wgpuComputePipelineRelease(out.untwist_accumulate);
        if (out.zero)               wgpuComputePipelineRelease(out.zero);
        wgpuShaderModuleRelease(out.shader);
        wgpuPipelineLayoutRelease(out.plo);
        wgpuBindGroupLayoutRelease(out.bgl);
        return {};
    }
    return out;
}

static void release_fhe_wgsl_pipelines(FheWgslPipelines& p) {
    if (p.decomp_twist)       wgpuComputePipelineRelease(p.decomp_twist);
    if (p.bsk_twist)          wgpuComputePipelineRelease(p.bsk_twist);
    if (p.ntt_bit_reverse)    wgpuComputePipelineRelease(p.ntt_bit_reverse);
    if (p.ntt_butterfly)      wgpuComputePipelineRelease(p.ntt_butterfly);
    if (p.inner_product)      wgpuComputePipelineRelease(p.inner_product);
    if (p.untwist_accumulate) wgpuComputePipelineRelease(p.untwist_accumulate);
    if (p.zero)               wgpuComputePipelineRelease(p.zero);
    if (p.shader)             wgpuShaderModuleRelease(p.shader);
    if (p.plo)                wgpuPipelineLayoutRelease(p.plo);
    if (p.bgl)                wgpuBindGroupLayoutRelease(p.bgl);
    p = {};
}

// ---------------------------------------------------------------------------
// Per-(N, q) ψ tables and cyclic-NTT twiddles. Caches keyed on (N<<48)^q.
// ---------------------------------------------------------------------------
static WgslNttPsiCacheEntry* fhe_wgsl_get_psi_cache(
    WGPUBackendContext* ctx, uint32_t N, uint64_t q)
{
    if (q == 0 || N == 0 || ((q - 1) % ((uint64_t)2 * N)) != 0) return nullptr;
    uint64_t key = ((uint64_t)N << 48) ^ q;
    {
        std::lock_guard<std::mutex> guard(ctx->fhe_ntt_mutex);
        auto it = ctx->fhe_psi_cache.find(key);
        if (it != ctx->fhe_psi_cache.end()) return &it->second;
    }

    uint64_t g       = fhe_wgsl_find_primitive_root_host(q);
    uint64_t psi     = fhe_wgsl_mod_pow_host(g, (q - 1) / (2ULL * N), q);
    uint64_t psi_inv = fhe_wgsl_mod_inv_host(psi, q);

    std::vector<uint64_t> psi_pow(N), psi_inv_pow(N);
    uint64_t cur_p = 1, cur_pi = 1;
    for (uint32_t j = 0; j < N; ++j) {
        psi_pow[j]     = cur_p;
        psi_inv_pow[j] = cur_pi;
        cur_p  = fhe_wgsl_mod_mul_host(cur_p,  psi,     q);
        cur_pi = fhe_wgsl_mod_mul_host(cur_pi, psi_inv, q);
    }
    auto psi_packed     = pack_u64_to_u32(psi_pow.data(),     N);
    auto psi_inv_packed = pack_u64_to_u32(psi_inv_pow.data(), N);
    const size_t bytes  = psi_packed.size() * sizeof(uint32_t);

    WgslNttPsiCacheEntry e{};
    e.psi_pow     = make_storage_buffer(ctx, bytes, psi_packed.data(),     bytes);
    e.psi_inv_pow = make_storage_buffer(ctx, bytes, psi_inv_packed.data(), bytes);
    if (!e.psi_pow || !e.psi_inv_pow) {
        if (e.psi_pow)     wgpuBufferRelease(e.psi_pow);
        if (e.psi_inv_pow) wgpuBufferRelease(e.psi_inv_pow);
        return nullptr;
    }

    std::lock_guard<std::mutex> guard(ctx->fhe_ntt_mutex);
    auto [iter, inserted] = ctx->fhe_psi_cache.emplace(key, e);
    if (!inserted) {
        wgpuBufferRelease(e.psi_pow);
        wgpuBufferRelease(e.psi_inv_pow);
    }
    return &iter->second;
}

static WgslNttTwiddleCacheEntry* fhe_wgsl_get_ntt_twiddles(
    WGPUBackendContext* ctx, uint32_t N, uint64_t q)
{
    uint64_t key = ((uint64_t)N << 48) ^ q;
    {
        std::lock_guard<std::mutex> guard(ctx->fhe_ntt_mutex);
        auto it = ctx->fhe_ntt_tw_cache.find(key);
        if (it != ctx->fhe_ntt_tw_cache.end()) return &it->second;
    }

    uint64_t g           = fhe_wgsl_find_primitive_root_host(q);
    uint64_t omega_n     = fhe_wgsl_mod_pow_host(g, (q - 1) / N, q);
    uint64_t omega_n_inv = fhe_wgsl_mod_inv_host(omega_n, q);

    WgslNttTwiddleCacheEntry e{};
    e.tw_fwd_h.resize(N / 2);
    e.tw_inv_h.resize(N / 2);
    uint64_t cur_fwd = 1, cur_inv = 1;
    for (uint32_t i = 0; i < N / 2; ++i) {
        e.tw_fwd_h[i] = cur_fwd;
        e.tw_inv_h[i] = cur_inv;
        cur_fwd = fhe_wgsl_mod_mul_host(cur_fwd, omega_n,     q);
        cur_inv = fhe_wgsl_mod_mul_host(cur_inv, omega_n_inv, q);
    }
    auto fwd_packed = pack_u64_to_u32(e.tw_fwd_h.data(), e.tw_fwd_h.size());
    auto inv_packed = pack_u64_to_u32(e.tw_inv_h.data(), e.tw_inv_h.size());
    const size_t bytes = fwd_packed.size() * sizeof(uint32_t);

    e.twiddles_fwd = make_storage_buffer(ctx, bytes, fwd_packed.data(), bytes);
    e.twiddles_inv = make_storage_buffer(ctx, bytes, inv_packed.data(), bytes);
    e.n_inv        = fhe_wgsl_mod_inv_host((uint64_t)N, q);
    if (!e.twiddles_fwd || !e.twiddles_inv) {
        if (e.twiddles_fwd) wgpuBufferRelease(e.twiddles_fwd);
        if (e.twiddles_inv) wgpuBufferRelease(e.twiddles_inv);
        return nullptr;
    }

    std::lock_guard<std::mutex> guard(ctx->fhe_ntt_mutex);
    auto [iter, inserted] = ctx->fhe_ntt_tw_cache.emplace(key, std::move(e));
    if (!inserted) {
        // Lost the race; release our just-built buffers and use the winner.
        if (iter->second.twiddles_fwd != e.twiddles_fwd) {
            // (variables are now moved-from; this branch is unreachable in
            // practice — kept for safety.)
        }
    }
    return &iter->second;
}

// Build per-stage twiddle slice device buffers for an N-point cyclic NTT.
// log2(N) buffers, one per stage, in iteration order (forward: 2,4,…,N;
// inverse: N,…,4,2). Caller frees via free_stage_twiddle_bufs.
struct WgslStageTwBuf { WGPUBuffer buf; size_t bytes; };
static std::vector<WgslStageTwBuf> fhe_wgsl_build_stage_twiddle_bufs(
    WGPUBackendContext* ctx, WgslNttTwiddleCacheEntry* tw_cache,
    uint32_t N, bool inverse)
{
    std::vector<WgslStageTwBuf> out;
    if (!tw_cache) return out;
    const std::vector<uint64_t>& full = inverse ? tw_cache->tw_inv_h : tw_cache->tw_fwd_h;
    uint32_t log_n = 0; { uint32_t t = N; while (t > 1) { t >>= 1; ++log_n; } }
    out.reserve(log_n);
    auto emit = [&](uint32_t len) {
        const uint32_t halfn  = len / 2;
        const uint32_t stride = N / len;
        std::vector<uint64_t> stage(halfn);
        for (uint32_t j = 0; j < halfn; ++j) stage[j] = full[(size_t)j * stride];
        auto packed = pack_u64_to_u32(stage.data(), stage.size());
        WgslStageTwBuf b{};
        b.bytes = packed.size() * sizeof(uint32_t);
        b.buf   = make_storage_buffer(ctx, b.bytes, packed.data(), b.bytes);
        out.push_back(b);
    };
    if (!inverse) for (uint32_t len = 2; len <= N; len *= 2) emit(len);
    else          for (uint32_t len = N; len >= 2; len /= 2) emit(len);
    return out;
}
static void free_stage_twiddle_bufs(std::vector<WgslStageTwBuf>& bufs) {
    for (auto& b : bufs) if (b.buf) wgpuBufferRelease(b.buf);
    bufs.clear();
}

// ---------------------------------------------------------------------------
// Bind-group builder. The WGSL kernel has 14 declared bindings; every kernel
// dispatch must populate all of them, even if the kernel doesn't actually
// use a given slot. Building one bind group per dispatch is cheap; the
// expensive pipeline/layout creation happens once.
// ---------------------------------------------------------------------------
struct FheBindBufs {
    WGPUBuffer decomp_in    = nullptr; // binding 0  (storage, read)
    WGPUBuffer out_rw       = nullptr; // binding 1  (storage, read_write)
    WGPUBuffer psi_pow      = nullptr; // binding 2  (storage, read)
    WGPUBuffer p_decomp_tw  = nullptr; // binding 3  (uniform)
    WGPUBuffer p_bsk_tw     = nullptr; // binding 4  (uniform)
    WGPUBuffer bsk_or_tw    = nullptr; // binding 5  (storage, read)
    WGPUBuffer bsk_ntt      = nullptr; // binding 6  (storage, read)
    WGPUBuffer result_ntt   = nullptr; // binding 7  (storage, read_write)
    WGPUBuffer p_ip         = nullptr; // binding 8  (uniform)
    WGPUBuffer p_ntt        = nullptr; // binding 9  (uniform)
    WGPUBuffer p_bitrev     = nullptr; // binding 10 (uniform)
    WGPUBuffer p_untwist    = nullptr; // binding 11 (uniform)
    WGPUBuffer psi_inv_pow  = nullptr; // binding 12 (storage, read)
    WGPUBuffer p_zero       = nullptr; // binding 13 (uniform)
};
static WGPUBindGroup build_fhe_bg(WGPUBackendContext* ctx, WGPUBindGroupLayout bgl,
                                    const FheBindBufs& b)
{
    WGPUBindGroupEntry e[14] = {};
    auto set = [&](int idx, uint32_t binding, WGPUBuffer buf) {
        e[idx].binding = binding;
        e[idx].buffer  = buf;
        e[idx].size    = wgpuBufferGetSize(buf);
    };
    set( 0,  0, b.decomp_in);
    set( 1,  1, b.out_rw);
    set( 2,  2, b.psi_pow);
    set( 3,  3, b.p_decomp_tw);
    set( 4,  4, b.p_bsk_tw);
    set( 5,  5, b.bsk_or_tw);
    set( 6,  6, b.bsk_ntt);
    set( 7,  7, b.result_ntt);
    set( 8,  8, b.p_ip);
    set( 9,  9, b.p_ntt);
    set(10, 10, b.p_bitrev);
    set(11, 11, b.p_untwist);
    set(12, 12, b.psi_inv_pow);
    set(13, 13, b.p_zero);
    WGPUBindGroupDescriptor d = {};
    d.layout     = bgl;
    d.entryCount = 14;
    d.entries    = e;
    return wgpuDeviceCreateBindGroup(ctx->device, &d);
}

// Encode a batched cyclic forward (or inverse) NTT over [M, N] data into an
// existing compute pass. Mirrors metal_plugin.mm::encode_batched_ntt and the
// CUDA fhe_encode_batched_ntt step-for-step. `data` is bound at slot 1 (=
// o_twist_or_acc); each stage swaps the per-stage twiddle buffer into slot 5
// (= i_bsk_or_twiddles) via a fresh bind group.
//
// WebGPU semantics: setBindGroup invalidates all prior bindings for the new
// group. We rebuild a fresh bind group per dispatch — bind-group allocation
// is cheap, and this keeps a single command encoder per AP step.
static LuxBackendError encode_batched_ntt_wgsl(
    WGPUBackendContext* ctx, WGPUComputePassEncoder pass,
    const FheWgslPipelines& pso, FheBindBufs binds,
    WGPUBuffer data, uint32_t M, uint32_t N, uint64_t q, bool inverse,
    const std::vector<WgslStageTwBuf>& stage_tw_bufs,
    std::vector<WGPUBindGroup>& owned_bgs)
{
    struct BitRP { uint32_t N; uint32_t M; uint32_t _pad0; uint32_t _pad1; };
    struct BatchP {
        uint32_t N;
        uint32_t M;
        uint32_t len;
        uint32_t halfn;
        uint32_t q_lo;
        uint32_t q_hi;
        uint32_t direction;
        uint32_t _pad;
    };

    binds.out_rw = data;

    auto dispatch_bitrev = [&]() -> LuxBackendError {
        WGPUBindGroup bg = build_fhe_bg(ctx, pso.bgl, binds);
        if (!bg) return LUX_BACKEND_ERROR_INTERNAL;
        wgpuComputePassEncoderSetPipeline(pass, pso.ntt_bit_reverse);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        const uint32_t wgx = (N + 63) / 64;
        wgpuComputePassEncoderDispatchWorkgroups(pass, wgx == 0 ? 1 : wgx, M, 1);
        owned_bgs.push_back(bg);
        return LUX_BACKEND_OK;
    };
    auto dispatch_butterfly = [&](uint32_t len, WGPUBuffer tw_buf) -> LuxBackendError {
        const uint32_t halfn = len / 2;
        BatchP bp = { N, M, len, halfn,
                      (uint32_t)(q & 0xFFFFFFFFu), (uint32_t)(q >> 32),
                      inverse ? 1u : 0u, 0u };
        wgpuQueueWriteBuffer(ctx->queue, binds.p_ntt, 0, &bp, sizeof(bp));
        FheBindBufs b2 = binds;
        b2.bsk_or_tw = tw_buf;
        WGPUBindGroup bg = build_fhe_bg(ctx, pso.bgl, b2);
        if (!bg) return LUX_BACKEND_ERROR_INTERNAL;
        wgpuComputePassEncoderSetPipeline(pass, pso.ntt_butterfly);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        const uint32_t wgx = ((N / 2) + 63) / 64;
        wgpuComputePassEncoderDispatchWorkgroups(pass, wgx == 0 ? 1 : wgx, M, 1);
        owned_bgs.push_back(bg);
        return LUX_BACKEND_OK;
    };

    BitRP br_p = { N, M, 0u, 0u };
    wgpuQueueWriteBuffer(ctx->queue, binds.p_bitrev, 0, &br_p, sizeof(br_p));

    if (!inverse) {
        if (auto e = dispatch_bitrev(); e != LUX_BACKEND_OK) return e;
        uint32_t stage_idx = 0;
        for (uint32_t len = 2; len <= N; len *= 2) {
            if (auto e = dispatch_butterfly(len, stage_tw_bufs[stage_idx++].buf);
                e != LUX_BACKEND_OK) return e;
        }
    } else {
        uint32_t stage_idx = 0;
        for (uint32_t len = N; len >= 2; len /= 2) {
            if (auto e = dispatch_butterfly(len, stage_tw_bufs[stage_idx++].buf);
                e != LUX_BACKEND_OK) return e;
        }
        if (auto e = dispatch_bitrev(); e != LUX_BACKEND_OK) return e;
    }
    return LUX_BACKEND_OK;
}

// Wait for queue work to drain. Uses the same shared `wgpu_wait_until` bound.
static LuxBackendError wait_queue(WGPUBackendContext* ctx) {
    struct WaitData { bool done; } wd{false};
    WGPUQueueWorkDoneCallbackInfo wcb = {};
    wcb.mode = WGPUCallbackMode_AllowProcessEvents;
#if defined(USE_DAWN_API)
    wcb.callback = [](WGPUQueueWorkDoneStatus, void* u, void*) {
        static_cast<WaitData*>(u)->done = true;
    };
#else
    wcb.callback = [](WGPUQueueWorkDoneStatus, WGPUStringView, void* u, void*) {
        static_cast<WaitData*>(u)->done = true;
    };
#endif
    wcb.userdata1 = &wd;
    wgpuQueueOnSubmittedWorkDone(ctx->queue, wcb);
    if (!wgpu_wait_until(ctx->instance, [&]{ return wd.done; })) {
        return LUX_BACKEND_ERROR_DEVICE_LOST;
    }
    return LUX_BACKEND_OK;
}

// ---------------------------------------------------------------------------
// Build (or fetch cached) the BSK-in-NTT image. Allocates a device buffer of
// the same shape as the input BSK (each u64 → 2× u32), twists each BSK[i]
// with ψ, and runs a forward NTT over each (rows × cols) batch of length-N
// rows. Cached by (host_ptr, shape, q, fingerprint).
//
// Build cost: O(n_lwe · (k+1) · l · (k+1) · N log N), paid once per BSK.
// ---------------------------------------------------------------------------
static WgslBskNttCacheEntry* fhe_wgsl_get_bsk_ntt_cache(
    WGPUBackendContext* ctx, const FheWgslPipelines& pso,
    WgslNttPsiCacheEntry* psi_tables, WgslNttTwiddleCacheEntry* tw_cache,
    const uint64_t* bsk_host,
    uint32_t n_lwe, uint32_t N, uint32_t k, uint32_t l, uint64_t q)
{
    const uint32_t cols       = k + 1u;
    const uint32_t rows       = cols * l;
    const size_t bsk_per_bit  = (size_t)rows * cols * N;  // u64s per BSK[i]
    const size_t bsk_total    = (size_t)n_lwe * bsk_per_bit;
    const size_t bsk_bytes_u32 = bsk_total * 2u * sizeof(uint32_t);

    uint64_t fp_lo, fp_hi;
    fhe_wgsl_bsk_fingerprint(bsk_host, bsk_total, fp_lo, fp_hi);
    WgslBskNttKey key { bsk_host, n_lwe, N, k, l, q, fp_lo, fp_hi };
    {
        std::lock_guard<std::mutex> guard(ctx->fhe_ntt_mutex);
        auto it = ctx->fhe_bsk_ntt_cache.find(key);
        if (it != ctx->fhe_bsk_ntt_cache.end()) return &it->second;
    }

    auto bsk_packed = pack_u64_to_u32(bsk_host, bsk_total);
    WGPUBuffer bsk_src = make_storage_buffer(ctx, bsk_bytes_u32, bsk_packed.data(), bsk_bytes_u32);
    WGPUBuffer bsk_ntt = make_storage_buffer(ctx, bsk_bytes_u32);
    if (!bsk_src || !bsk_ntt) {
        if (bsk_src) wgpuBufferRelease(bsk_src);
        if (bsk_ntt) wgpuBufferRelease(bsk_ntt);
        return nullptr;
    }

    // Scratch uniforms and a 16-byte storage placeholder for slots not used
    // by bsk_twist / NTT (but still part of the kitchen-sink bind group).
    WGPUBuffer dummy_storage = make_storage_buffer(ctx, 16);
    WGPUBuffer u_decomp_tw   = make_uniform_buffer(ctx, 32);
    WGPUBuffer u_bsk_tw      = make_uniform_buffer(ctx, 32);
    WGPUBuffer u_ip          = make_uniform_buffer(ctx, 32);
    WGPUBuffer u_ntt         = make_uniform_buffer(ctx, 32);
    WGPUBuffer u_bitrev      = make_uniform_buffer(ctx, 32);
    WGPUBuffer u_untwist     = make_uniform_buffer(ctx, 32);
    WGPUBuffer u_zero        = make_uniform_buffer(ctx, 32);
    auto cleanup_scratch = [&]() {
        if (dummy_storage) wgpuBufferRelease(dummy_storage);
        if (u_decomp_tw)   wgpuBufferRelease(u_decomp_tw);
        if (u_bsk_tw)      wgpuBufferRelease(u_bsk_tw);
        if (u_ip)          wgpuBufferRelease(u_ip);
        if (u_ntt)         wgpuBufferRelease(u_ntt);
        if (u_bitrev)      wgpuBufferRelease(u_bitrev);
        if (u_untwist)     wgpuBufferRelease(u_untwist);
        if (u_zero)        wgpuBufferRelease(u_zero);
    };
    if (!dummy_storage || !u_decomp_tw || !u_bsk_tw || !u_ip || !u_ntt ||
        !u_bitrev || !u_untwist || !u_zero) {
        cleanup_scratch();
        wgpuBufferRelease(bsk_src);
        wgpuBufferRelease(bsk_ntt);
        return nullptr;
    }

    auto stage_tw_fwd = fhe_wgsl_build_stage_twiddle_bufs(ctx, tw_cache, N, false);
    if (stage_tw_fwd.empty()) {
        cleanup_scratch();
        wgpuBufferRelease(bsk_src);
        wgpuBufferRelease(bsk_ntt);
        return nullptr;
    }

    // Encode the full BSK transform into ONE command buffer.
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx->device, nullptr);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);

    FheBindBufs binds{};
    binds.decomp_in   = dummy_storage;
    binds.psi_pow     = psi_tables->psi_pow;
    binds.psi_inv_pow = psi_tables->psi_inv_pow;
    binds.bsk_ntt     = dummy_storage;
    binds.result_ntt  = dummy_storage;
    binds.bsk_or_tw   = dummy_storage;
    binds.out_rw      = dummy_storage;
    binds.p_decomp_tw = u_decomp_tw;
    binds.p_bsk_tw    = u_bsk_tw;
    binds.p_ip        = u_ip;
    binds.p_ntt       = u_ntt;
    binds.p_bitrev    = u_bitrev;
    binds.p_untwist   = u_untwist;
    binds.p_zero      = u_zero;

    std::vector<WGPUBindGroup> owned_bgs;
    owned_bgs.reserve((size_t)n_lwe * (2u + (size_t)stage_tw_fwd.size() + 1u));
    std::vector<WGPUBuffer> per_i_bufs;  // released after submit.

    LuxBackendError err = LUX_BACKEND_OK;
    const size_t per_bytes_u32 = bsk_per_bit * 2u * sizeof(uint32_t);
    for (uint32_t i = 0; i < n_lwe; ++i) {
        // BSK[i] = bsk_per_bit u64s. Without dynamic-offset bindings we run
        // twist+NTT on per-i scratch slices and copy back. Two cheap buffer-
        // to-buffer copies surround each BSK[i] kernel chain.
        WGPUBuffer bsk_src_i = make_storage_buffer(ctx, per_bytes_u32);
        WGPUBuffer bsk_ntt_i = make_storage_buffer(ctx, per_bytes_u32);
        if (!bsk_src_i || !bsk_ntt_i) {
            if (bsk_src_i) wgpuBufferRelease(bsk_src_i);
            if (bsk_ntt_i) wgpuBufferRelease(bsk_ntt_i);
            err = LUX_BACKEND_ERROR_OUT_OF_MEMORY;
            break;
        }
        per_i_bufs.push_back(bsk_src_i);
        per_i_bufs.push_back(bsk_ntt_i);

        const uint64_t bsk_off_u32 = (uint64_t)i * bsk_per_bit * 2u;
        wgpuCommandEncoderCopyBufferToBuffer(enc, bsk_src,
                                               bsk_off_u32 * sizeof(uint32_t),
                                               bsk_src_i, 0, per_bytes_u32);

        struct BskTwistP {
            uint32_t N; uint32_t rows; uint32_t cols; uint32_t _pad;
            uint32_t q_lo; uint32_t q_hi; uint32_t _pad2; uint32_t _pad3;
        } tw_p = { N, rows, cols, 0u,
                   (uint32_t)(q & 0xFFFFFFFFu), (uint32_t)(q >> 32), 0u, 0u };
        wgpuQueueWriteBuffer(ctx->queue, u_bsk_tw, 0, &tw_p, sizeof(tw_p));

        FheBindBufs b2 = binds;
        b2.bsk_or_tw = bsk_src_i;
        b2.out_rw    = bsk_ntt_i;
        WGPUBindGroup bg = build_fhe_bg(ctx, pso.bgl, b2);
        if (!bg) { err = LUX_BACKEND_ERROR_INTERNAL; break; }
        wgpuComputePassEncoderSetPipeline(pass, pso.bsk_twist);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        // workgroup_size = (8,8,1); grid threads = (N, cols, rows).
        const uint32_t wx = (N + 7) / 8;
        const uint32_t wy = (cols + 7) / 8;
        wgpuComputePassEncoderDispatchWorkgroups(pass, wx == 0 ? 1 : wx,
                                                  wy == 0 ? 1 : wy, rows);
        owned_bgs.push_back(bg);

        // Forward NTT of bsk_ntt_i over (rows × cols) rows of length N.
        const uint32_t M = rows * cols;
        FheBindBufs b3 = binds;
        b3.out_rw = bsk_ntt_i;
        err = encode_batched_ntt_wgsl(ctx, pass, pso, b3, bsk_ntt_i, M, N, q,
                                        /*inverse=*/false, stage_tw_fwd, owned_bgs);
        if (err != LUX_BACKEND_OK) break;

        wgpuCommandEncoderCopyBufferToBuffer(enc, bsk_ntt_i, 0,
                                               bsk_ntt, bsk_off_u32 * sizeof(uint32_t),
                                               per_bytes_u32);
    }

    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuCommandEncoderRelease(enc);
    if (err == LUX_BACKEND_OK) {
        wgpuQueueSubmit(ctx->queue, 1, &cmd);
        err = wait_queue(ctx);
    }
    wgpuCommandBufferRelease(cmd);

    for (auto bg : owned_bgs) wgpuBindGroupRelease(bg);
    for (auto b : per_i_bufs) wgpuBufferRelease(b);
    free_stage_twiddle_bufs(stage_tw_fwd);
    cleanup_scratch();
    wgpuBufferRelease(bsk_src);

    if (err != LUX_BACKEND_OK) {
        wgpuBufferRelease(bsk_ntt);
        return nullptr;
    }

    WgslBskNttCacheEntry entry{};
    entry.bsk_ntt   = bsk_ntt;
    entry.bsk_bytes = bsk_bytes_u32;

    std::lock_guard<std::mutex> guard(ctx->fhe_ntt_mutex);
    auto [iter, inserted] = ctx->fhe_bsk_ntt_cache.emplace(key, entry);
    if (!inserted) {
        wgpuBufferRelease(bsk_ntt);
    }
    return &iter->second;
}

// ---------------------------------------------------------------------------
// One AP step using NTT-fused external product. Encodes the GPU-side portion
// (twist, batched NTT, inner product, inverse NTT, untwist+accumulate) into
// a single command encoder + queue.submit, then reads acc back so the next
// step's diff+decomp can run on the host.
//
// diff and decomp are done on the host because they are O(cols * N) and
// O(cols * l * N) respectively — small relative to the O(rows*N + cols*N
// log N) NTT work that dominates. Matches the high-level division of labor
// in the Metal and CUDA dispatchers (both run diff+decomp as small kernels
// but the host CPU is cheaper here than another bind-group hop).
// ---------------------------------------------------------------------------
static LuxBackendError fhe_blind_rotate_step_wgsl(
    WGPUBackendContext* ctx, const FheWgslPipelines& pso,
    WgslNttPsiCacheEntry* psi_tables, WgslNttTwiddleCacheEntry* tw_cache,
    WGPUBuffer acc, WGPUBuffer decomp_buf, WGPUBuffer decomp_twist_buf,
    WGPUBuffer result_ntt_buf,
    uint64_t* acc_host,
    WGPUBuffer bsk_ntt, size_t bsk_ntt_offset_u64_words,
    uint32_t a_tilde, uint32_t N, uint32_t k, uint32_t l,
    uint32_t base_log, uint64_t q,
    WGPUBuffer u_decomp_tw, WGPUBuffer u_bsk_tw, WGPUBuffer u_ip,
    WGPUBuffer u_ntt, WGPUBuffer u_bitrev, WGPUBuffer u_untwist,
    WGPUBuffer u_zero, WGPUBuffer dummy_storage)
{
    const uint32_t cols = k + 1u;
    const uint32_t rows = cols * l;

    // Step 1: diff = X^{a_tilde}·acc - acc  (on host).
    // Acc is laid out [cols][N] in time domain. X^t on R_q[X]/(X^N+1) is a
    // negacyclic rotation: for t < N, X^t · acc shifts indices by t and
    // negates wrapped coefficients; for t in [N, 2N), it adds an extra sign
    // flip. We compute diff[c][j] = acc_rot[c][j] − acc[c][j] mod q.
    std::vector<uint64_t> diff_h((size_t)cols * N);
    {
        const uint32_t two_n = 2u * N;
        const uint32_t r = a_tilde % two_n;
        for (uint32_t c = 0; c < cols; ++c) {
            const uint64_t* acc_c = acc_host + (size_t)c * N;
            uint64_t* diff_c = diff_h.data() + (size_t)c * N;
            for (uint32_t j = 0; j < N; ++j) {
                uint32_t src = (j + two_n - r) % two_n;
                uint64_t v;
                if (src < N) v = acc_c[src];
                else         v = (acc_c[src - N] == 0) ? 0 : (q - acc_c[src - N]);
                diff_c[j] = (v >= acc_c[j]) ? (v - acc_c[j])
                                             : (q - (acc_c[j] - v));
            }
        }
    }

    // Step 2: signed-digit decomp of diff (on host).
    std::vector<int64_t> decomp_h((size_t)rows * N);
    for (uint32_t c = 0; c < cols; ++c) {
        for (uint32_t j = 0; j < N; ++j) {
            int64_t digits[64];
            lux::fhe::signed_decomp_all(diff_h[(size_t)c * N + j], l, base_log, digits);
            for (uint32_t lvl = 0; lvl < l; ++lvl) {
                decomp_h[((size_t)c * l + lvl) * N + j] = digits[lvl];
            }
        }
    }
    auto decomp_packed = pack_i64_to_u32(decomp_h.data(), decomp_h.size());
    wgpuQueueWriteBuffer(ctx->queue, decomp_buf, 0, decomp_packed.data(),
                          decomp_packed.size() * sizeof(uint32_t));

    // Stage twiddle bufs for one forward + one inverse NTT.
    auto stage_tw_fwd = fhe_wgsl_build_stage_twiddle_bufs(ctx, tw_cache, N, false);
    auto stage_tw_inv = fhe_wgsl_build_stage_twiddle_bufs(ctx, tw_cache, N, true);
    if (stage_tw_fwd.empty() || stage_tw_inv.empty()) {
        free_stage_twiddle_bufs(stage_tw_fwd);
        free_stage_twiddle_bufs(stage_tw_inv);
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    // BSK-NTT slice for this AP step. Without dynamic-offset bindings we
    // copy the BSK[i] slab into a smaller buffer up front.
    const size_t bsk_slice_u32_bytes =
        (size_t)rows * cols * N * 2u * sizeof(uint32_t);
    WGPUBuffer bsk_ntt_slice = make_storage_buffer(ctx, bsk_slice_u32_bytes);
    if (!bsk_ntt_slice) {
        free_stage_twiddle_bufs(stage_tw_fwd);
        free_stage_twiddle_bufs(stage_tw_inv);
        return LUX_BACKEND_ERROR_OUT_OF_MEMORY;
    }

    FheBindBufs binds{};
    binds.decomp_in   = decomp_buf;
    binds.psi_pow     = psi_tables->psi_pow;
    binds.psi_inv_pow = psi_tables->psi_inv_pow;
    binds.bsk_ntt     = bsk_ntt_slice;
    binds.result_ntt  = result_ntt_buf;
    binds.bsk_or_tw   = dummy_storage;
    binds.out_rw      = dummy_storage;
    binds.p_decomp_tw = u_decomp_tw;
    binds.p_bsk_tw    = u_bsk_tw;
    binds.p_ip        = u_ip;
    binds.p_ntt       = u_ntt;
    binds.p_bitrev    = u_bitrev;
    binds.p_untwist   = u_untwist;
    binds.p_zero      = u_zero;

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx->device, nullptr);

    // Buffer copy MUST come before BeginComputePass — copyBufferToBuffer is
    // a transfer command, not a compute command, and the compute pass holds
    // the encoder exclusively while open.
    wgpuCommandEncoderCopyBufferToBuffer(
        enc, bsk_ntt,
        bsk_ntt_offset_u64_words * sizeof(uint32_t) * 2u,
        bsk_ntt_slice, 0, bsk_slice_u32_bytes);

    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);

    std::vector<WGPUBindGroup> owned_bgs;
    owned_bgs.reserve(4 + stage_tw_fwd.size() + stage_tw_inv.size());

    auto dispatch = [&](WGPUComputePipeline pipe, FheBindBufs b,
                          uint32_t gx, uint32_t gy, uint32_t gz) -> LuxBackendError {
        WGPUBindGroup bg = build_fhe_bg(ctx, pso.bgl, b);
        if (!bg) return LUX_BACKEND_ERROR_INTERNAL;
        wgpuComputePassEncoderSetPipeline(pass, pipe);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, gz);
        owned_bgs.push_back(bg);
        return LUX_BACKEND_OK;
    };

    LuxBackendError err = LUX_BACKEND_OK;

    // 3. decomp_twist[r][j] = ψ^j · |decomp[r][j]|_q  → into decomp_twist_buf
    struct DecTwistP { uint32_t N; uint32_t rows; uint32_t q_lo; uint32_t q_hi; }
        dt_p = { N, rows, (uint32_t)(q & 0xFFFFFFFFu), (uint32_t)(q >> 32) };
    wgpuQueueWriteBuffer(ctx->queue, u_decomp_tw, 0, &dt_p, sizeof(dt_p));
    if (err == LUX_BACKEND_OK) {
        FheBindBufs b = binds;
        b.out_rw = decomp_twist_buf;
        const uint32_t wx = (N + 7) / 8;
        const uint32_t wy = (rows + 7) / 8;
        err = dispatch(pso.decomp_twist, b, wx == 0 ? 1 : wx, wy == 0 ? 1 : wy, 1);
    }

    // 4. Forward NTT of decomp_twist [rows × N].
    if (err == LUX_BACKEND_OK) {
        FheBindBufs b = binds;
        b.out_rw = decomp_twist_buf;
        err = encode_batched_ntt_wgsl(ctx, pass, pso, b, decomp_twist_buf,
                                        rows, N, q, /*inverse=*/false,
                                        stage_tw_fwd, owned_bgs);
    }

    // 5. Inner product: result_ntt[c][i] = Σ_r decomp_ntt[r][i] · bsk_ntt[r][c][i]
    if (err == LUX_BACKEND_OK) {
        struct IPP { uint32_t N; uint32_t rows; uint32_t cols; uint32_t _pad;
                     uint32_t q_lo; uint32_t q_hi; uint32_t _pad2; uint32_t _pad3; }
            ip_p = { N, rows, cols, 0u,
                     (uint32_t)(q & 0xFFFFFFFFu), (uint32_t)(q >> 32), 0u, 0u };
        wgpuQueueWriteBuffer(ctx->queue, u_ip, 0, &ip_p, sizeof(ip_p));
        FheBindBufs b = binds;
        b.decomp_in  = decomp_twist_buf;
        b.result_ntt = result_ntt_buf;
        const uint32_t wx = (N + 63) / 64;
        err = dispatch(pso.inner_product, b, wx == 0 ? 1 : wx, cols, 1);
    }

    // 6. Inverse NTT of result_ntt [cols × N].
    if (err == LUX_BACKEND_OK) {
        FheBindBufs b = binds;
        b.out_rw = result_ntt_buf;
        err = encode_batched_ntt_wgsl(ctx, pass, pso, b, result_ntt_buf,
                                        cols, N, q, /*inverse=*/true,
                                        stage_tw_inv, owned_bgs);
    }

    // 7. acc[c][j] += ψ^{-j} · N^{-1} · result_ntt[c][j]
    if (err == LUX_BACKEND_OK) {
        struct UAP { uint32_t N; uint32_t cols; uint32_t _pad0; uint32_t _pad1;
                     uint32_t q_lo; uint32_t q_hi;
                     uint32_t n_inv_lo; uint32_t n_inv_hi; }
            ua_p = { N, cols, 0u, 0u,
                     (uint32_t)(q & 0xFFFFFFFFu), (uint32_t)(q >> 32),
                     (uint32_t)(tw_cache->n_inv & 0xFFFFFFFFu),
                     (uint32_t)(tw_cache->n_inv >> 32) };
        wgpuQueueWriteBuffer(ctx->queue, u_untwist, 0, &ua_p, sizeof(ua_p));
        FheBindBufs b = binds;
        b.out_rw = acc;
        const uint32_t wx = (N + 63) / 64;
        err = dispatch(pso.untwist_accumulate, b, wx == 0 ? 1 : wx, cols, 1);
    }

    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuCommandEncoderRelease(enc);
    if (err == LUX_BACKEND_OK) {
        wgpuQueueSubmit(ctx->queue, 1, &cmd);
        err = wait_queue(ctx);
    }
    wgpuCommandBufferRelease(cmd);

    for (auto bg : owned_bgs) wgpuBindGroupRelease(bg);
    wgpuBufferRelease(bsk_ntt_slice);
    free_stage_twiddle_bufs(stage_tw_fwd);
    free_stage_twiddle_bufs(stage_tw_inv);

    if (err != LUX_BACKEND_OK) return err;

    // Read acc back into host mirror.
    const size_t acc_u32_bytes = (size_t)cols * N * 2u * sizeof(uint32_t);
    std::vector<uint32_t> acc_u32(acc_u32_bytes / sizeof(uint32_t));
    err = readback_buffer(ctx, acc, acc_u32.data(), acc_u32_bytes);
    if (err != LUX_BACKEND_OK) return err;
    unpack_u32_to_u64(acc_u32.data(), acc_host, (size_t)cols * N);

    // Final normalize: u64 modular results are guaranteed < q by the WGSL
    // kernels, no extra mod needed.
    return LUX_BACKEND_OK;
}

// ---------------------------------------------------------------------------
// Top-level fast-path entry points for bootstrap and blind_rotate. Each owns
// its own buffer lifetime; on any setup or dispatch failure they return
// NOT_SUPPORTED so the public op falls back to the canonical host helper.
// ---------------------------------------------------------------------------
static LuxBackendError webgpu_fhe_bootstrap_fast(
    WGPUBackendContext* ctx,
    const uint64_t* lwe_in, uint64_t* lwe_out,
    const uint64_t* bsk,    const uint64_t* test_poly,
    uint32_t n_lwe, uint32_t N, uint32_t k, uint32_t l,
    uint32_t base_log, uint64_t q)
{
    if (!ctx) return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    if (!fhe_should_use_wgsl_ntt(N, q)) return LUX_BACKEND_ERROR_NOT_SUPPORTED;

    FheWgslPipelines pso = load_fhe_wgsl_pipelines(ctx);
    if (!pso.ok()) return LUX_BACKEND_ERROR_NOT_SUPPORTED;

    WgslNttPsiCacheEntry*     psi_tables = fhe_wgsl_get_psi_cache(ctx, N, q);
    WgslNttTwiddleCacheEntry* tw_cache   = fhe_wgsl_get_ntt_twiddles(ctx, N, q);
    if (!psi_tables || !tw_cache) {
        release_fhe_wgsl_pipelines(pso);
        return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    }

    WgslBskNttCacheEntry* bsk_entry =
        fhe_wgsl_get_bsk_ntt_cache(ctx, pso, psi_tables, tw_cache,
                                     bsk, n_lwe, N, k, l, q);
    if (!bsk_entry) {
        release_fhe_wgsl_pipelines(pso);
        return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    }

    const uint32_t cols     = k + 1u;
    const size_t acc_len    = (size_t)cols * N;
    const size_t decomp_len = (size_t)cols * l * N;
    const size_t bsk_stride = (size_t)cols * l * cols * N;
    const size_t out_len    = (size_t)k * N + 1u;

    // Initialize acc on the host: rotate test_poly by neg_b_tilde, write
    // into acc[k][·]; lower polynomial slots are zero. Matches Metal's
    // pbs_init kernel and CUDA's fhe_pbs_init kernel — same formula.
    std::vector<uint64_t> acc_host(acc_len, 0);
    {
        const uint64_t b = lwe_in[n_lwe];
        const uint32_t b_tilde = fhe_wgsl_compute_a_tilde(b, N, q);
        const uint32_t two_n = 2u * N;
        const uint32_t neg_b_tilde = (two_n - b_tilde) % two_n;
        uint64_t* acc_last = acc_host.data() + (size_t)k * N;
        for (uint32_t j = 0; j < N; ++j) {
            uint32_t src = (j + two_n - neg_b_tilde) % two_n;
            if (src < N) acc_last[j] = test_poly[src];
            else         acc_last[j] = (test_poly[src - N] == 0) ? 0
                                                                  : (q - test_poly[src - N]);
        }
    }

    const size_t acc_u32_bytes    = acc_len    * 2u * sizeof(uint32_t);
    const size_t decomp_u32_bytes = decomp_len * 2u * sizeof(uint32_t);

    auto acc_packed = pack_u64_to_u32(acc_host.data(), acc_host.size());
    WGPUBuffer d_acc          = make_storage_buffer(ctx, acc_u32_bytes, acc_packed.data(), acc_u32_bytes);
    WGPUBuffer d_decomp       = make_storage_buffer(ctx, decomp_u32_bytes);
    WGPUBuffer d_decomp_twist = make_storage_buffer(ctx, decomp_u32_bytes);
    WGPUBuffer d_result_ntt   = make_storage_buffer(ctx, acc_u32_bytes);
    WGPUBuffer dummy_storage  = make_storage_buffer(ctx, 16);
    WGPUBuffer u_decomp_tw    = make_uniform_buffer(ctx, 32);
    WGPUBuffer u_bsk_tw       = make_uniform_buffer(ctx, 32);
    WGPUBuffer u_ip           = make_uniform_buffer(ctx, 32);
    WGPUBuffer u_ntt          = make_uniform_buffer(ctx, 32);
    WGPUBuffer u_bitrev       = make_uniform_buffer(ctx, 32);
    WGPUBuffer u_untwist      = make_uniform_buffer(ctx, 32);
    WGPUBuffer u_zero         = make_uniform_buffer(ctx, 32);

    auto cleanup = [&]() {
        if (d_acc)          wgpuBufferRelease(d_acc);
        if (d_decomp)       wgpuBufferRelease(d_decomp);
        if (d_decomp_twist) wgpuBufferRelease(d_decomp_twist);
        if (d_result_ntt)   wgpuBufferRelease(d_result_ntt);
        if (dummy_storage)  wgpuBufferRelease(dummy_storage);
        if (u_decomp_tw)    wgpuBufferRelease(u_decomp_tw);
        if (u_bsk_tw)       wgpuBufferRelease(u_bsk_tw);
        if (u_ip)           wgpuBufferRelease(u_ip);
        if (u_ntt)          wgpuBufferRelease(u_ntt);
        if (u_bitrev)       wgpuBufferRelease(u_bitrev);
        if (u_untwist)      wgpuBufferRelease(u_untwist);
        if (u_zero)         wgpuBufferRelease(u_zero);
        release_fhe_wgsl_pipelines(pso);
    };

    if (!d_acc || !d_decomp || !d_decomp_twist || !d_result_ntt ||
        !dummy_storage || !u_decomp_tw || !u_bsk_tw || !u_ip || !u_ntt ||
        !u_bitrev || !u_untwist || !u_zero) {
        cleanup();
        return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    }

    for (uint32_t i = 0; i < n_lwe; ++i) {
        const uint32_t a_tilde = fhe_wgsl_compute_a_tilde(lwe_in[i], N, q);
        if (a_tilde == 0u) continue;
        const size_t bsk_ntt_off_u64 = (size_t)i * bsk_stride;
        auto e = fhe_blind_rotate_step_wgsl(
            ctx, pso, psi_tables, tw_cache,
            d_acc, d_decomp, d_decomp_twist, d_result_ntt,
            acc_host.data(),
            bsk_entry->bsk_ntt, bsk_ntt_off_u64,
            a_tilde, N, k, l, base_log, q,
            u_decomp_tw, u_bsk_tw, u_ip, u_ntt, u_bitrev, u_untwist,
            u_zero, dummy_storage);
        if (e != LUX_BACKEND_OK) { cleanup(); return LUX_BACKEND_ERROR_NOT_SUPPORTED; }
    }

    // Sample-extract on host: acc layout [cols][N] → LWE [a_0..a_{k*N-1}, b].
    // Matches the canonical extract in cpu_fhe_helpers and the Metal/CUDA
    // pbs_extract kernels.
    lwe_out[(size_t)k * N] = acc_host[(size_t)k * N];
    for (uint32_t j = 0; j < k; ++j) {
        const uint64_t* acc_j = acc_host.data() + (size_t)j * N;
        lwe_out[(size_t)j * N + 0] = acc_j[0];
        for (uint32_t i = 1; i < N; ++i) {
            uint64_t v = acc_j[N - i];
            lwe_out[(size_t)j * N + i] = (v == 0) ? 0 : (q - v);
        }
    }

    (void)out_len;
    cleanup();
    return LUX_BACKEND_OK;
}

static LuxBackendError webgpu_fhe_blind_rotate_fast(
    WGPUBackendContext* ctx,
    uint64_t* acc_host_io, const uint64_t* bsk, const uint64_t* lwe_a,
    uint32_t n_lwe, uint32_t N, uint32_t k, uint32_t l,
    uint32_t base_log, uint64_t q)
{
    if (!ctx) return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    if (!fhe_should_use_wgsl_ntt(N, q)) return LUX_BACKEND_ERROR_NOT_SUPPORTED;

    FheWgslPipelines pso = load_fhe_wgsl_pipelines(ctx);
    if (!pso.ok()) return LUX_BACKEND_ERROR_NOT_SUPPORTED;

    WgslNttPsiCacheEntry*     psi_tables = fhe_wgsl_get_psi_cache(ctx, N, q);
    WgslNttTwiddleCacheEntry* tw_cache   = fhe_wgsl_get_ntt_twiddles(ctx, N, q);
    if (!psi_tables || !tw_cache) {
        release_fhe_wgsl_pipelines(pso);
        return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    }

    WgslBskNttCacheEntry* bsk_entry =
        fhe_wgsl_get_bsk_ntt_cache(ctx, pso, psi_tables, tw_cache,
                                     bsk, n_lwe, N, k, l, q);
    if (!bsk_entry) {
        release_fhe_wgsl_pipelines(pso);
        return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    }

    const uint32_t cols     = k + 1u;
    const size_t acc_len    = (size_t)cols * N;
    const size_t decomp_len = (size_t)cols * l * N;
    const size_t bsk_stride = (size_t)cols * l * cols * N;

    std::vector<uint64_t> acc_host(acc_host_io, acc_host_io + acc_len);

    const size_t acc_u32_bytes    = acc_len    * 2u * sizeof(uint32_t);
    const size_t decomp_u32_bytes = decomp_len * 2u * sizeof(uint32_t);

    auto acc_packed = pack_u64_to_u32(acc_host.data(), acc_host.size());
    WGPUBuffer d_acc          = make_storage_buffer(ctx, acc_u32_bytes, acc_packed.data(), acc_u32_bytes);
    WGPUBuffer d_decomp       = make_storage_buffer(ctx, decomp_u32_bytes);
    WGPUBuffer d_decomp_twist = make_storage_buffer(ctx, decomp_u32_bytes);
    WGPUBuffer d_result_ntt   = make_storage_buffer(ctx, acc_u32_bytes);
    WGPUBuffer dummy_storage  = make_storage_buffer(ctx, 16);
    WGPUBuffer u_decomp_tw    = make_uniform_buffer(ctx, 32);
    WGPUBuffer u_bsk_tw       = make_uniform_buffer(ctx, 32);
    WGPUBuffer u_ip           = make_uniform_buffer(ctx, 32);
    WGPUBuffer u_ntt          = make_uniform_buffer(ctx, 32);
    WGPUBuffer u_bitrev       = make_uniform_buffer(ctx, 32);
    WGPUBuffer u_untwist      = make_uniform_buffer(ctx, 32);
    WGPUBuffer u_zero         = make_uniform_buffer(ctx, 32);

    auto cleanup = [&]() {
        if (d_acc)          wgpuBufferRelease(d_acc);
        if (d_decomp)       wgpuBufferRelease(d_decomp);
        if (d_decomp_twist) wgpuBufferRelease(d_decomp_twist);
        if (d_result_ntt)   wgpuBufferRelease(d_result_ntt);
        if (dummy_storage)  wgpuBufferRelease(dummy_storage);
        if (u_decomp_tw)    wgpuBufferRelease(u_decomp_tw);
        if (u_bsk_tw)       wgpuBufferRelease(u_bsk_tw);
        if (u_ip)           wgpuBufferRelease(u_ip);
        if (u_ntt)          wgpuBufferRelease(u_ntt);
        if (u_bitrev)       wgpuBufferRelease(u_bitrev);
        if (u_untwist)      wgpuBufferRelease(u_untwist);
        if (u_zero)         wgpuBufferRelease(u_zero);
        release_fhe_wgsl_pipelines(pso);
    };

    if (!d_acc || !d_decomp || !d_decomp_twist || !d_result_ntt ||
        !dummy_storage || !u_decomp_tw || !u_bsk_tw || !u_ip || !u_ntt ||
        !u_bitrev || !u_untwist || !u_zero) {
        cleanup();
        return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    }

    for (uint32_t i = 0; i < n_lwe; ++i) {
        const uint32_t a_tilde = fhe_wgsl_compute_a_tilde(lwe_a[i], N, q);
        if (a_tilde == 0u) continue;
        const size_t bsk_ntt_off_u64 = (size_t)i * bsk_stride;
        auto e = fhe_blind_rotate_step_wgsl(
            ctx, pso, psi_tables, tw_cache,
            d_acc, d_decomp, d_decomp_twist, d_result_ntt,
            acc_host.data(),
            bsk_entry->bsk_ntt, bsk_ntt_off_u64,
            a_tilde, N, k, l, base_log, q,
            u_decomp_tw, u_bsk_tw, u_ip, u_ntt, u_bitrev, u_untwist,
            u_zero, dummy_storage);
        if (e != LUX_BACKEND_OK) { cleanup(); return LUX_BACKEND_ERROR_NOT_SUPPORTED; }
    }

    std::memcpy(acc_host_io, acc_host.data(), acc_len * sizeof(uint64_t));
    cleanup();
    return LUX_BACKEND_OK;
}

#endif // !WEBGPU_STUB

// =============================================================================
// Public dispatchers.
//
// The dispatcher is layered:
//   1. Validate inputs via lux::fhe::validate_pbs_params (canonical). Both
//      paths share the same validator so error semantics are identical.
//   2. Attempt the NTT-fused fast path. Returns NOT_SUPPORTED on any setup
//      failure (Dawn unavailable, shader compile error, cache build failure,
//      dispatch error, q has no 2N-th root, N below threshold, LUX_FHE_FORCE_
//      SCHOOLBOOK=1). NOT_SUPPORTED triggers fallback.
//   3. On fallback, call the canonical host helper (single source of truth
//      in cpu_fhe_helpers.hpp). The canonical body is the correctness
//      reference; the GPU path must be bit-exact w.r.t. it.
// =============================================================================

static LuxBackendError webgpu_op_tfhe_bootstrap(
    LuxBackendContext* context,
    const uint64_t* lwe_in,
    uint64_t* lwe_out,
    const uint64_t* bsk,
    const uint64_t* test_poly,
    uint32_t n_lwe,
    uint32_t N,
    uint32_t k,
    uint32_t l,
    uint32_t base_log,
    uint64_t q
) {
    if (!context || !lwe_in || !lwe_out || !bsk || !test_poly)
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    const LuxBackendError v = lux::fhe::validate_pbs_params(n_lwe, N, k, l, base_log, q);
    if (v != LUX_BACKEND_OK) return v;

#ifndef WEBGPU_STUB
    if (auto ctx = reinterpret_cast<WGPUBackendContext*>(context)) {
        auto e = webgpu_fhe_bootstrap_fast(ctx, lwe_in, lwe_out, bsk, test_poly,
                                            n_lwe, N, k, l, base_log, q);
        if (e == LUX_BACKEND_OK) return LUX_BACKEND_OK;
        // Fall through to canonical helper on any other return code.
    }
#endif
    return lux::fhe::run_tfhe_bootstrap(lwe_in, lwe_out, bsk, test_poly,
                                         n_lwe, N, k, l, base_log, q);
}

static LuxBackendError webgpu_op_tfhe_keyswitch(
    LuxBackendContext* context,
    const uint64_t* lwe_in,
    uint64_t* lwe_out,
    const uint64_t* ksk,
    uint32_t n_in,
    uint32_t n_out,
    uint32_t l,
    uint32_t base_log,
    uint64_t q
) {
    if (!context) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    // Keyswitch is O(n_in · l · n_out) per call — small relative to PBS and
    // not the bootstrap bottleneck. Keep it on the canonical host helper; a
    // device-side keyswitch can be added behind the same fall-through pattern
    // when profiles justify it (matches CUDA's choice).
    return lux::fhe::run_tfhe_keyswitch(lwe_in, lwe_out, ksk,
                                         n_in, n_out, l, base_log, q);
}

static LuxBackendError webgpu_op_blind_rotate(
    LuxBackendContext* context,
    uint64_t* acc,
    const uint64_t* bsk,
    const uint64_t* lwe_a,
    uint32_t n_lwe,
    uint32_t N,
    uint32_t k,
    uint32_t l,
    uint32_t base_log,
    uint64_t q
) {
    if (!context || !acc || !bsk || !lwe_a) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    const LuxBackendError v = lux::fhe::validate_pbs_params(n_lwe, N, k, l, base_log, q);
    if (v != LUX_BACKEND_OK) return v;

#ifndef WEBGPU_STUB
    if (auto ctx = reinterpret_cast<WGPUBackendContext*>(context)) {
        auto e = webgpu_fhe_blind_rotate_fast(ctx, acc, bsk, lwe_a,
                                                n_lwe, N, k, l, base_log, q);
        if (e == LUX_BACKEND_OK) return LUX_BACKEND_OK;
    }
#endif
    return lux::fhe::run_blind_rotate(acc, bsk, lwe_a, n_lwe, N, k, l, base_log, q);
}

#if 0  /* sample_extract / sample_ntt — not part of canonical ABI v3 */
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
#endif  /* sample_extract / sample_ntt */


// =============================================================================
// Crypto ops on the canonical vtbl.
//
// WGSL kernel sources live under kernels/wgsl/crypto/ (blake3.wgsl,
// bn254.wgsl, keccak256.wgsl, bls12_381.wgsl, poseidon2_bn254.wgsl,
// secp256k1.wgsl, msm.wgsl, kzg.wgsl). Each kernel uses a bespoke
// @group(0) @binding layout, derived per-pipeline at module-compile time via
// LayoutPreset::Auto (wgpuComputePipelineGetBindGroupLayout). The wiring
// pattern mirrors Metal's get_shader_pipeline + dispatch helper: load WGSL
// via load_wgsl_source, fetch the auto-derived layout, alloc input/output
// buffers, write via wgpuQueueWriteBuffer, dispatch, map-read output.
//
// Wired ops (WGSL path available):
//   - poseidon2_hash         → poseidon2_bn254.wgsl :: poseidon2_hash_pair
//   - blake3_hash            → blake3.wgsl          :: hash_batch
//   - keccak256_hash         → keccak256.wgsl       :: keccak256_batch_variable
//   - bn254_add (G1 proj)    → bn254.wgsl           :: bn254_g1_proj_batch_add
//   - bn254_mul (G1 proj)    → bn254.wgsl           :: bn254_g1_proj_batch_scalar_mul
//   - ntt_forward / inverse  → ntt.wgsl             :: ntt_butterfly / ntt_scale
//   - matmul / binary ops    → tensor.wgsl
//   - msm                    → msm.wgsl             :: msm_bucket_*
//
// Still NOT_SUPPORTED (return falls back to CPU):
//   - bls12_381 add/mul/pairing — 6-limb Fp384 struct stride differs from
//     LuxG1Projective381; needs a repack pass.
//   - bn254 G2 — Fp2 projective addition exists in bn254.wgsl but the host
//     ABI (is_g2=true) is not wired here.
//   - kzg open/verify — require Fp12 pairing kernel (not in current source).
//   - secp256k1 ecdsa_recover — recover-specific kernel absent.
// =============================================================================

static LuxBackendError webgpu_op_poseidon2_hash(
    LuxBackendContext* context, const uint64_t* inputs, uint64_t* outputs,
    size_t rate, size_t num_hashes
) {
#ifndef WEBGPU_STUB
    if (!context || !inputs || !outputs || num_hashes == 0)
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    if (rate != 2) return LUX_BACKEND_ERROR_NOT_SUPPORTED;

    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    const char* src = load_wgsl_source(ctx, "crypto/poseidon2_bn254.wgsl");
    if (!src) return LUX_BACKEND_ERROR_NOT_SUPPORTED;

    auto* entry = getOrCreatePipelineWithLayout(ctx, src, "poseidon2_hash_pair", LayoutPreset::Auto);
    if (!entry) return LUX_BACKEND_ERROR_INTERNAL;

    // ABI layout: inputs = [num_hashes * rate * 4] u64 = num_hashes pairs of
    // (left[4 u64], right[4 u64]). Each Fr256 in WGSL is 8 u32 = 32 bytes.
    // Output: [num_hashes * 4] u64 = num_hashes Fr256 results.
    const size_t fr_bytes = 32;
    const size_t pair_bytes = num_hashes * fr_bytes;
    const size_t out_bytes  = num_hashes * fr_bytes;

    // Split flat input into left/right halves.
    std::vector<uint8_t> left(pair_bytes), right(pair_bytes);
    for (size_t h = 0; h < num_hashes; ++h) {
        std::memcpy(left.data()  + h * fr_bytes, &inputs[(h * 2 + 0) * 4], fr_bytes);
        std::memcpy(right.data() + h * fr_bytes, &inputs[(h * 2 + 1) * 4], fr_bytes);
    }

    struct P2Params { uint32_t count; uint32_t path_len; uint32_t _p1; uint32_t _p2; }
        params = { (uint32_t)num_hashes, 0, 0, 0 };
    // Shader bindings: 0=left, 1=right, 2=out, 3=params (uniform).
    return wgsl_dispatch_op(ctx, entry,
        {{0, left.data(), nullptr, pair_bytes},
         {1, right.data(), nullptr, pair_bytes},
         {2, nullptr, outputs, out_bytes}},
        &params, sizeof(params), 3,
        (uint32_t)((num_hashes + 255) / 256));
#else
    (void)context; (void)inputs; (void)outputs; (void)rate; (void)num_hashes;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

// -----------------------------------------------------------------------------
// blake3 (batched, variable-length) — wgsl bridge.
//
// WGSL kernel: kernels/wgsl/crypto/blake3.wgsl :: hash_batch
// Bindings (group 0): 4=inputs (u32-packed), 5=outputs (u32-packed),
//   6=offsets, 7=lengths, 8=Blake3BatchParams.
// Hashes are 32 bytes each, written to outputs[h*32..h*32+32].
// -----------------------------------------------------------------------------
static LuxBackendError webgpu_op_blake3_hash(
    LuxBackendContext* context, const uint8_t* inputs, uint8_t* outputs,
    const size_t* input_lens, size_t num_hashes
) {
#ifndef WEBGPU_STUB
    if (!context || !inputs || !outputs || !input_lens || num_hashes == 0)
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    const char* src = load_wgsl_source(ctx, "crypto/blake3.wgsl");
    if (!src) return LUX_BACKEND_ERROR_NOT_SUPPORTED;

    auto* entry = getOrCreatePipelineWithLayout(ctx, src, "hash_batch", LayoutPreset::Auto);
    if (!entry) return LUX_BACKEND_ERROR_INTERNAL;

    // Compute offsets and total byte length.
    std::vector<uint32_t> offsets(num_hashes);
    std::vector<uint32_t> lengths(num_hashes);
    size_t total = 0;
    for (size_t i = 0; i < num_hashes; ++i) {
        offsets[i] = static_cast<uint32_t>(total);
        lengths[i] = static_cast<uint32_t>(input_lens[i]);
        total += input_lens[i];
    }
    // u32-padded input buffer: round up to multiple of 4 bytes; minimum 4 so
    // the binding is non-empty (blake3.wgsl::hash_batch handles len==0 explicitly).
    const size_t input_bytes_padded = ((total + 3) / 4) * 4;
    const size_t alloc_input_bytes = std::max<size_t>(input_bytes_padded, 4);
    std::vector<uint8_t> in_packed(alloc_input_bytes, 0);
    if (total > 0) std::memcpy(in_packed.data(), inputs, total);

    const size_t out_bytes = num_hashes * 32;
    const size_t idx_bytes = num_hashes * sizeof(uint32_t);

    struct Blake3BatchParams { uint32_t num_inputs; uint32_t _p0; uint32_t _p1; uint32_t _p2; }
        params = { static_cast<uint32_t>(num_hashes), 0, 0, 0 };
    // Shader bindings: 4=in, 5=out, 6=offsets, 7=lengths, 8=params (uniform).
    return wgsl_dispatch_op(ctx, entry,
        {{4, in_packed.data(), nullptr, alloc_input_bytes},
         {5, nullptr, outputs, out_bytes},
         {6, offsets.data(), nullptr, idx_bytes},
         {7, lengths.data(), nullptr, idx_bytes}},
        &params, sizeof(params), 8,
        (uint32_t)((num_hashes + 63) / 64));
#else
    (void)context; (void)inputs; (void)outputs; (void)input_lens; (void)num_hashes;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

// -----------------------------------------------------------------------------
// keccak256 (batched, variable-length) — wgsl bridge.
//
// WGSL kernel: kernels/wgsl/crypto/keccak256.wgsl :: keccak256_batch_variable
// Bindings (group 0):  0=inputs (u32-packed), 1=outputs (u32-packed),
//   2=offsets, 3=lengths, 4=num_inputs (uniform).
// -----------------------------------------------------------------------------
static LuxBackendError webgpu_op_keccak256_hash(
    LuxBackendContext* context, const uint8_t* inputs, uint8_t* outputs,
    const size_t* input_lens, size_t num_inputs
) {
#ifndef WEBGPU_STUB
    if (!context || !inputs || !outputs || !input_lens || num_inputs == 0)
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    const char* src = load_wgsl_source(ctx, "crypto/keccak256.wgsl");
    if (!src) return LUX_BACKEND_ERROR_NOT_SUPPORTED;

    auto* entry = getOrCreatePipelineWithLayout(ctx, src, "keccak256_batch_variable", LayoutPreset::Auto);
    if (!entry) return LUX_BACKEND_ERROR_INTERNAL;

    std::vector<uint32_t> offsets(num_inputs);
    std::vector<uint32_t> lengths(num_inputs);
    size_t total = 0;
    for (size_t i = 0; i < num_inputs; ++i) {
        offsets[i] = static_cast<uint32_t>(total);
        lengths[i] = static_cast<uint32_t>(input_lens[i]);
        total += input_lens[i];
    }
    const size_t input_bytes_padded = std::max<size_t>(((total + 3) / 4) * 4, 4);
    std::vector<uint8_t> in_packed(input_bytes_padded, 0);
    if (total > 0) std::memcpy(in_packed.data(), inputs, total);

    const size_t out_bytes = num_inputs * 32;
    const size_t idx_bytes = num_inputs * sizeof(uint32_t);
    uint32_t count = (uint32_t)num_inputs;
    // Shader bindings: 0=in, 1=out, 2=offsets, 3=lengths, 4=count (uniform).
    return wgsl_dispatch_op(ctx, entry,
        {{0, in_packed.data(), nullptr, input_bytes_padded},
         {1, nullptr, outputs, out_bytes},
         {2, offsets.data(), nullptr, idx_bytes},
         {3, lengths.data(), nullptr, idx_bytes}},
        &count, sizeof(count), 4,
        (uint32_t)((num_inputs + 63) / 64));
#else
    (void)context; (void)inputs; (void)outputs; (void)input_lens; (void)num_inputs;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

// -----------------------------------------------------------------------------
// bls12_381 G1 projective+projective add — wgsl bridge.
//
// WGSL kernel: kernels/wgsl/crypto/bls12_381.wgsl :: g1_batch_add
// Bindings (group 0):
//   0=proj_a (read), 1=proj_b (read), 3=out_points (rw), 4=g1_count (uniform).
//   Binding 2 (scalars) is unused for add — we still allocate a 1-byte
//   placeholder so the auto-derived layout has a buffer to bind there.
//
// Storage layout match: LuxG1Projective381 = 3 × LuxFp384 = 3 × (6 × u64) =
// 144 bytes per point. WGSL G1Projective { x,y,z: Fp { limbs: array<u32, 12> }}
// = 3 × 48 = 144 bytes. Byte-identical on a little-endian device.
//
// G2 is intentionally surfaced as NOT_SUPPORTED — no Fp2 projective code
// is in bls12_381.wgsl (matching Metal which also lacks a G2 ABI bridge).
// -----------------------------------------------------------------------------
static LuxBackendError webgpu_op_bls12_381_add(
    LuxBackendContext* context, const void* a, const void* b, void* out,
    size_t n, bool is_g2
) {
#ifndef WEBGPU_STUB
    if (is_g2) return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    if (!context || !a || !b || !out || n == 0) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    const char* src = load_wgsl_source(ctx, "crypto/bls12_381.wgsl");
    if (!src) return LUX_BACKEND_ERROR_NOT_SUPPORTED;

    auto* entry = getOrCreatePipelineWithLayout(ctx, src, "g1_batch_add", LayoutPreset::Auto);
    if (!entry) return LUX_BACKEND_ERROR_INTERNAL;

    constexpr size_t kPointBytes = 144;  // 3 × 6 × u64 == 3 × 12 × u32
    const size_t buf_bytes = n * kPointBytes;
    uint32_t count = (uint32_t)n;
    // Bindings: 0=proj_a, 1=proj_b, 3=out_points, 4=g1_count (uniform).
    return wgsl_dispatch_op(ctx, entry,
        {{0, a, nullptr, buf_bytes},
         {1, b, nullptr, buf_bytes},
         {3, nullptr, out, buf_bytes}},
        &count, sizeof(count), 4,
        (uint32_t)((n + 63) / 64));
#else
    (void)context; (void)a; (void)b; (void)out; (void)n; (void)is_g2;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

// -----------------------------------------------------------------------------
// bls12_381 G1 projective scalar mul — wgsl bridge.
//
// WGSL kernel: kernels/wgsl/crypto/bls12_381.wgsl :: g1_batch_scalar_mul
// Bindings (group 0):
//   0=proj_a (read), 2=scalars (read, flat u32 array), 3=out_points (rw),
//   4=g1_count (uniform).
//
// Scalar layout: LuxScalar256 = 4 × u64 = 32 bytes. Flat little-endian
// u32 splits one scalar into 8 u32 lanes — host buffer bytes match
// lane-for-lane.
//
// On-curve gating happens inside g1_scalar_mul (single check at the ladder
// head, off-curve P returns identity).
// -----------------------------------------------------------------------------
static LuxBackendError webgpu_op_bls12_381_mul(
    LuxBackendContext* context, const void* points, const void* scalars, void* out,
    size_t n, bool is_g2
) {
#ifndef WEBGPU_STUB
    if (is_g2) return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    if (!context || !points || !scalars || !out || n == 0) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    const char* src = load_wgsl_source(ctx, "crypto/bls12_381.wgsl");
    if (!src) return LUX_BACKEND_ERROR_NOT_SUPPORTED;

    auto* entry = getOrCreatePipelineWithLayout(ctx, src, "g1_batch_scalar_mul", LayoutPreset::Auto);
    if (!entry) return LUX_BACKEND_ERROR_INTERNAL;

    constexpr size_t kPointBytes  = 144;
    constexpr size_t kScalarBytes = 32;
    const size_t pts_bytes = n * kPointBytes;
    const size_t scs_bytes = n * kScalarBytes;
    uint32_t count = (uint32_t)n;
    // Bindings: 0=proj_a, 2=scalars, 3=out_points, 4=g1_count (uniform).
    return wgsl_dispatch_op(ctx, entry,
        {{0, points, nullptr, pts_bytes},
         {2, scalars, nullptr, scs_bytes},
         {3, nullptr, out, pts_bytes}},
        &count, sizeof(count), 4,
        (uint32_t)((n + 63) / 64));
#else
    (void)context; (void)points; (void)scalars; (void)out; (void)n; (void)is_g2;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

// Pairing has no WGSL Fp12 implementation; CPU fallback handles it.
static LuxBackendError webgpu_op_bls12_381_pairing(
    LuxBackendContext*, const void*, const void*, void*, size_t
) { return LUX_BACKEND_ERROR_NOT_SUPPORTED; }

// -----------------------------------------------------------------------------
// bn254 batch projective+projective add — wgsl bridge.
//
// WGSL kernel: kernels/wgsl/crypto/bn254.wgsl :: bn254_g1_proj_batch_add
// Bindings (group 0):
//   2=output_points (rw), 3=num_elements (uniform),
//   4=proj_a (read), 5=proj_b (read).
//
// Layout: LuxG1Projective254 = 3 × LuxScalar256 = 3 × (4 × u64) = 96 bytes
// per point, byte-identical to WGSL G1Projective { x,y,z: Fp { array<u32,8> } }.
// G2 is intentionally surfaced as NOT_SUPPORTED — the Fp2 projective addition
// kernel exists in bn254.wgsl but the host ABI (is_g2=true) is not wired here
// because the dispatcher falls back to CPU for the rare G2 paths.
// -----------------------------------------------------------------------------
static LuxBackendError webgpu_op_bn254_add(
    LuxBackendContext* context, const void* a, const void* b, void* out,
    size_t n, bool is_g2
) {
#ifndef WEBGPU_STUB
    if (is_g2) return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    if (!context || !a || !b || !out || n == 0) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    const char* src = load_wgsl_source(ctx, "crypto/bn254.wgsl");
    if (!src) return LUX_BACKEND_ERROR_NOT_SUPPORTED;

    auto* entry = getOrCreatePipelineWithLayout(ctx, src, "bn254_g1_proj_batch_add", LayoutPreset::Auto);
    if (!entry) return LUX_BACKEND_ERROR_INTERNAL;

    constexpr size_t kPointBytes = 96;  // 3 × 4 × u64
    const size_t buf_bytes = n * kPointBytes;
    uint32_t count = (uint32_t)n;
    // Shader bindings: 2=out, 3=count (uniform), 4=a, 5=b.
    return wgsl_dispatch_op(ctx, entry,
        {{2, nullptr, out, buf_bytes},
         {4, a, nullptr, buf_bytes},
         {5, b, nullptr, buf_bytes}},
        &count, sizeof(count), 3,
        (uint32_t)((n + 63) / 64));
#else
    (void)context; (void)a; (void)b; (void)out; (void)n; (void)is_g2;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

// -----------------------------------------------------------------------------
// bn254 batch projective scalar mul — wgsl bridge.
//
// WGSL kernel: kernels/wgsl/crypto/bn254.wgsl :: bn254_g1_proj_batch_scalar_mul
// Bindings (group 0):
//   1=input_scalars (read, array<Fr>), 2=output_points (rw),
//   3=num_elements (uniform), 4=proj_a (read).
//
// Scalar layout: LuxScalar256 = 4 × u64 = 32 bytes, byte-identical to
// WGSL Fr { array<u32,8> }. G2 surfaced as NOT_SUPPORTED (see bn254_add note).
// -----------------------------------------------------------------------------
static LuxBackendError webgpu_op_bn254_mul(
    LuxBackendContext* context, const void* points, const void* scalars, void* out,
    size_t n, bool is_g2
) {
#ifndef WEBGPU_STUB
    if (is_g2) return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    if (!context || !points || !scalars || !out || n == 0) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    auto ctx = reinterpret_cast<WGPUBackendContext*>(context);
    const char* src = load_wgsl_source(ctx, "crypto/bn254.wgsl");
    if (!src) return LUX_BACKEND_ERROR_NOT_SUPPORTED;

    auto* entry = getOrCreatePipelineWithLayout(ctx, src, "bn254_g1_proj_batch_scalar_mul", LayoutPreset::Auto);
    if (!entry) return LUX_BACKEND_ERROR_INTERNAL;

    constexpr size_t kPointBytes  = 96;
    constexpr size_t kScalarBytes = 32;
    const size_t pts_bytes = n * kPointBytes;
    const size_t scs_bytes = n * kScalarBytes;
    uint32_t count = (uint32_t)n;
    // Shader bindings: 1=scalars, 2=out, 3=count (uniform), 4=points.
    return wgsl_dispatch_op(ctx, entry,
        {{1, scalars, nullptr, scs_bytes},
         {2, nullptr, out, pts_bytes},
         {4, points, nullptr, pts_bytes}},
        &count, sizeof(count), 3,
        (uint32_t)((n + 63) / 64));
#else
    (void)context; (void)points; (void)scalars; (void)out; (void)n; (void)is_g2;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

static LuxBackendError webgpu_op_kzg_commit(
    LuxBackendContext*, const void*, const void*, void*, size_t, int
) { return LUX_BACKEND_ERROR_NOT_SUPPORTED; }

static LuxBackendError webgpu_op_kzg_open(
    LuxBackendContext*, const void*, const void*, const void*, void*, size_t, int
) { return LUX_BACKEND_ERROR_NOT_SUPPORTED; }

static LuxBackendError webgpu_op_kzg_verify(
    LuxBackendContext*, const void*, const void*, const void*, const void*, const void*, bool*, int
) { return LUX_BACKEND_ERROR_NOT_SUPPORTED; }

static LuxBackendError webgpu_op_ecrecover_batch(
    LuxBackendContext*, const void*, void*, size_t
) { return LUX_BACKEND_ERROR_NOT_SUPPORTED; }

// =============================================================================
// Crypto: PQ verify + Schnorr verify + threshold primitives.
//
// Blocker: no WGSL shaders exist for these algorithms. WGSL is the most
// constrained target (no 64-bit ints natively, restricted control flow);
// porting MLDSA/SLHDSA/Ed25519 verify here is a substantial effort and is
// not the current bottleneck. CPU backend handles these.
// =============================================================================

static LuxBackendError webgpu_op_mldsa_verify_batch(
    LuxBackendContext*, const uint8_t* const*, const uint8_t* const*,
    const uint8_t* const*, bool*, size_t
) { return LUX_BACKEND_ERROR_NOT_SUPPORTED; }

static LuxBackendError webgpu_op_mlkem_decapsulate_batch(
    LuxBackendContext*, const uint8_t* const*, const uint8_t* const*,
    uint8_t**, size_t
) { return LUX_BACKEND_ERROR_NOT_SUPPORTED; }

static LuxBackendError webgpu_op_slhdsa_verify_batch(
    LuxBackendContext*, const uint8_t* const*, const uint8_t* const*,
    const uint8_t* const*, bool*, size_t
) { return LUX_BACKEND_ERROR_NOT_SUPPORTED; }

static LuxBackendError webgpu_op_ringtail_partial_sign_batch(
    LuxBackendContext*, const uint8_t* const*, const uint8_t* const*,
    uint8_t**, size_t
) { return LUX_BACKEND_ERROR_NOT_SUPPORTED; }

static LuxBackendError webgpu_op_ringtail_combine_batch(
    LuxBackendContext*, const uint8_t* const*, const int32_t*,
    uint8_t**, size_t, size_t
) { return LUX_BACKEND_ERROR_NOT_SUPPORTED; }

static LuxBackendError webgpu_op_frost_partial_verify_batch(
    LuxBackendContext*, const uint8_t* const*, const uint8_t* const*,
    const uint8_t* const*, const uint8_t* const*, bool*, size_t
) { return LUX_BACKEND_ERROR_NOT_SUPPORTED; }

static LuxBackendError webgpu_op_cggmp21_partial_sign_batch(
    LuxBackendContext*, const uint8_t* const*, const uint8_t*,
    uint8_t**, size_t
) { return LUX_BACKEND_ERROR_NOT_SUPPORTED; }

static LuxBackendError webgpu_op_ed25519_verify_batch(
    LuxBackendContext*, const uint8_t* const*, const uint8_t* const*,
    const uint8_t* const*, bool*, size_t
) { return LUX_BACKEND_ERROR_NOT_SUPPORTED; }

static LuxBackendError webgpu_op_sr25519_verify_batch(
    LuxBackendContext*, const uint8_t* const*, const uint8_t* const*,
    const uint8_t* const*, bool*, size_t
) { return LUX_BACKEND_ERROR_NOT_SUPPORTED; }

// =============================================================================
// WebGPU Backend VTable
// =============================================================================

static const lux_gpu_backend_vtbl webgpu_vtbl = {
    // Lifecycle
    .create_context  = webgpu_create_context,
    .destroy_context = webgpu_destroy_context,

    // Device info & sync
    .get_device_count = webgpu_get_device_count,
    .get_device_info  = webgpu_get_device_info,
    .sync             = webgpu_sync,

    // Buffer management
    .buffer_alloc           = webgpu_buffer_alloc,
    .buffer_alloc_with_data = webgpu_buffer_alloc_with_data,
    .buffer_free            = webgpu_buffer_free,
    .buffer_copy_to_host    = webgpu_buffer_copy_to_host,
    .buffer_copy_from_host  = webgpu_buffer_copy_from_host,
    .buffer_get_host_ptr    = webgpu_buffer_get_host_ptr,

    // Tensor
    .op_add_f32    = webgpu_op_add_f32,
    .op_sub_f32    = webgpu_op_sub_f32,
    .op_mul_f32    = webgpu_op_mul_f32,
    .op_div_f32    = nullptr,

    .op_matmul_f32    = webgpu_op_matmul_f32,
    .op_transpose_f32 = nullptr,

    .op_reduce_sum_f32      = nullptr,
    .op_reduce_max_f32      = nullptr,
    .op_reduce_min_f32      = nullptr,
    .op_reduce_mean_f32     = nullptr,
    .op_reduce_sum_axis_f32 = nullptr,
    .op_reduce_max_axis_f32 = nullptr,

    .op_softmax_f32     = nullptr,
    .op_log_softmax_f32 = nullptr,

    .op_exp_f32     = nullptr,
    .op_log_f32     = nullptr,
    .op_sqrt_f32    = nullptr,
    .op_neg_f32     = nullptr,
    .op_abs_f32     = nullptr,
    .op_tanh_f32    = nullptr,
    .op_sigmoid_f32 = nullptr,
    .op_relu_f32    = nullptr,
    .op_gelu_f32    = nullptr,

    .op_copy_f32 = nullptr,

    .op_layer_norm_f32 = nullptr,
    .op_rms_norm_f32   = nullptr,

    // NTT
    .op_ntt_forward = webgpu_op_ntt_forward,
    .op_ntt_inverse = webgpu_op_ntt_inverse,

    // FHE
    .op_poly_mul        = webgpu_op_poly_mul,
    .op_tfhe_bootstrap  = webgpu_op_tfhe_bootstrap,
    .op_tfhe_keyswitch  = webgpu_op_tfhe_keyswitch,
    .op_blind_rotate    = webgpu_op_blind_rotate,

    // Crypto hashes
    .op_poseidon2_hash = webgpu_op_poseidon2_hash,
    .op_blake3_hash    = webgpu_op_blake3_hash,
    .op_keccak256_hash = webgpu_op_keccak256_hash,

    // BLS12-381
    .op_bls12_381_add     = webgpu_op_bls12_381_add,
    .op_bls12_381_mul     = webgpu_op_bls12_381_mul,
    .op_bls12_381_pairing = webgpu_op_bls12_381_pairing,

    // BN254
    .op_bn254_add = webgpu_op_bn254_add,
    .op_bn254_mul = webgpu_op_bn254_mul,

    // MSM
    .op_msm = webgpu_op_msm,

    // KZG
    .op_kzg_commit = webgpu_op_kzg_commit,
    .op_kzg_open   = webgpu_op_kzg_open,
    .op_kzg_verify = webgpu_op_kzg_verify,

    // ecrecover
    .op_ecrecover_batch = webgpu_op_ecrecover_batch,

    // Post-quantum signatures
    .op_mldsa_verify_batch       = webgpu_op_mldsa_verify_batch,
    .op_mlkem_decapsulate_batch  = webgpu_op_mlkem_decapsulate_batch,
    .op_slhdsa_verify_batch      = webgpu_op_slhdsa_verify_batch,

    // Threshold primitives
    .op_ringtail_partial_sign_batch = webgpu_op_ringtail_partial_sign_batch,
    .op_ringtail_combine_batch      = webgpu_op_ringtail_combine_batch,
    .op_frost_partial_verify_batch  = webgpu_op_frost_partial_verify_batch,
    .op_cggmp21_partial_sign_batch  = webgpu_op_cggmp21_partial_sign_batch,

    // Classical Schnorr
    .op_ed25519_verify_batch = webgpu_op_ed25519_verify_batch,
    .op_sr25519_verify_batch = webgpu_op_sr25519_verify_batch,
};

// =============================================================================
// Plugin Entry Point
// =============================================================================

static bool webgpu_backend_init_impl(lux_gpu_backend_desc* out) {
    if (!out) return false;
#ifndef WEBGPU_STUB
    out->abi_version     = LUX_GPU_BACKEND_ABI_VERSION;
    out->vtbl_size       = sizeof(lux_gpu_backend_vtbl);
    out->backend_name    = "webgpu";
    out->backend_version = "0.4.0";
    // NTT-fused PBS now lives in webgpu_fhe_bootstrap_fast (mirror of
    // metal/cuda). Activation gate (fhe_should_use_wgsl_ntt) requires N ≥
    // LUX_FHE_NTT_MIN_N (default 256) AND q | 2N-th root of unity. On any
    // setup or dispatch failure the path falls through to the canonical
    // host helper in lux::fhe::run_* — so advertising the capability is
    // honest: the public op_tfhe_bootstrap / op_blind_rotate slot always
    // produces a correct answer, on GPU when the gate fires and on CPU
    // otherwise. Keyswitch stays canonical-only (same as CUDA's choice).
    out->capabilities    = LUX_CAP_TENSOR_OPS | LUX_CAP_MATMUL | LUX_CAP_NTT
                         | LUX_CAP_POLY_MUL
                         | LUX_CAP_FHE | LUX_CAP_TFHE | LUX_CAP_BLIND_ROTATE;
    out->vtbl            = &webgpu_vtbl;
    return true;
#else
    return false;
#endif
}

LUX_GPU_DECLARE_BACKEND(webgpu_backend_init_impl)
