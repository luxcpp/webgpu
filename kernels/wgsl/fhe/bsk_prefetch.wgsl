// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// TFHE Bootstrapping Key Prefetch Kernel for WebGPU
// Asynchronous prefetching of BSK entries to minimize latency during blind rotation
// Compatible with Metal/Vulkan/D3D12 via Dawn/wgpu

// ============================================================================
// 64-bit Integer Emulation
// ============================================================================

struct u64 {
    lo: u32,
    hi: u32,
}

fn u64_zero() -> u64 { return u64(0u, 0u); }

fn u64_from_u32(v: u32) -> u64 { return u64(v, 0u); }

// ============================================================================
// BSK Prefetch Parameters
// ============================================================================

struct BSKPrefetchParams {
    // BSK dimensions
    N: u32,              // Polynomial degree (1024, 2048)
    n: u32,              // LWE dimension (~630)
    k: u32,              // GLWE dimension (1)
    l: u32,              // Decomposition levels (3-4)
    base_log: u32,       // Base log (8)

    // Prefetch control
    prefetch_start: u32, // Starting BSK index to prefetch
    prefetch_count: u32, // Number of BSK entries to prefetch
    cache_line_size: u32,// Cache line size in u32 elements

    // Memory layout
    bsk_stride: u32,     // Stride between BSK entries
    total_entries: u32,  // Total BSK entries
    _pad: u32,
}

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> bsk_global: array<u32>;      // Global BSK storage
@group(0) @binding(1) var<storage, read_write> bsk_cache: array<u32>; // L1/L2 cache simulation
@group(0) @binding(2) var<storage, read_write> prefetch_status: array<atomic<u32>>; // Prefetch completion flags
@group(0) @binding(3) var<uniform> params: BSKPrefetchParams;

// Workgroup shared memory for staging
var<workgroup> staging_buffer: array<u32, 4096>;  // 16KB staging area
var<workgroup> prefetch_ready: atomic<u32>;

// ============================================================================
// Cache-Aware Prefetch
// ============================================================================

// Prefetch a single BSK entry (RGSW ciphertext) into cache
// Each BSK entry has dimensions: [(k+1)][l][(k+1)][N][2] for u64 coefficients
@compute @workgroup_size(256)
fn bsk_prefetch_entry(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let l = params.l;
    let tpg = 256u;

    let entry_idx = params.prefetch_start + wgid.x;

    if (entry_idx >= params.total_entries) {
        return;
    }

    // BSK entry size: (k+1) * l * (k+1) * N * 2 u32 elements
    let entry_size = (k + 1u) * l * (k + 1u) * N * 2u;
    let global_offset = entry_idx * entry_size;
    let cache_offset = wgid.x * entry_size;

    // Cooperative prefetch: each thread handles a portion
    let elements_per_thread = (entry_size + tpg - 1u) / tpg;

    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid + i * tpg;
        if (idx < entry_size) {
            // Read from global memory (triggers cache prefetch)
            let value = bsk_global[global_offset + idx];

            // Write to cache buffer
            bsk_cache[cache_offset + idx] = value;
        }
    }

    // Memory fence to ensure writes are visible
    storageBarrier();

    // Mark entry as prefetched
    if (tid == 0u) {
        atomicStore(&prefetch_status[entry_idx], 1u);
    }
}

// ============================================================================
// Strided Prefetch for NTT-Domain BSK
// ============================================================================

// Prefetch with stride pattern for better memory coalescing
@compute @workgroup_size(256)
fn bsk_prefetch_strided(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let l = params.l;
    let tpg = 256u;
    let cache_line = params.cache_line_size;

    let entry_idx = params.prefetch_start + wgid.x;
    let poly_idx = wgid.y;

    if (entry_idx >= params.total_entries) {
        return;
    }

    // Calculate offsets for this polynomial within the BSK entry
    let entry_size = (k + 1u) * l * (k + 1u) * N * 2u;
    let poly_size = N * 2u;
    let global_base = entry_idx * entry_size + poly_idx * poly_size;
    let cache_base = (wgid.x * (k + 1u) * l * (k + 1u) + poly_idx) * poly_size;

    // Prefetch in cache-line aligned chunks
    let chunks = (poly_size + cache_line - 1u) / cache_line;
    let chunks_per_thread = (chunks + tpg - 1u) / tpg;

    for (var c = 0u; c < chunks_per_thread; c++) {
        let chunk_idx = tid + c * tpg;
        if (chunk_idx < chunks) {
            let chunk_start = chunk_idx * cache_line;
            let chunk_end = min(chunk_start + cache_line, poly_size);

            for (var i = chunk_start; i < chunk_end; i++) {
                let value = bsk_global[global_base + i];
                bsk_cache[cache_base + i] = value;
            }
        }
    }
}

// ============================================================================
// Async Prefetch with Double Buffering
// ============================================================================

struct DoubleBufferParams {
    N: u32,
    k: u32,
    l: u32,
    current_buffer: u32,  // 0 or 1
    next_entry_idx: u32,  // Entry to prefetch into next buffer
    current_entry_idx: u32, // Entry being used
    buffer_size: u32,
    _pad: u32,
}

@group(1) @binding(0) var<storage, read_write> buffer_a: array<u32>;
@group(1) @binding(1) var<storage, read_write> buffer_b: array<u32>;
@group(1) @binding(2) var<storage, read_write> buffer_ready: array<atomic<u32>>;
@group(1) @binding(3) var<uniform> db_params: DoubleBufferParams;

// Prefetch next entry while current entry is being processed
@compute @workgroup_size(256)
fn bsk_prefetch_double_buffer(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = db_params.N;
    let k = db_params.k;
    let l = db_params.l;
    let tpg = 256u;

    let entry_idx = db_params.next_entry_idx;
    let next_buffer = 1u - db_params.current_buffer;

    // Calculate entry size
    let entry_size = (k + 1u) * l * (k + 1u) * N * 2u;
    let global_offset = entry_idx * entry_size;

    // Select target buffer
    let elements_per_thread = (entry_size + tpg - 1u) / tpg;

    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid + i * tpg;
        if (idx < entry_size) {
            let value = bsk_global[global_offset + idx];

            if (next_buffer == 0u) {
                buffer_a[idx] = value;
            } else {
                buffer_b[idx] = value;
            }
        }
    }

    storageBarrier();

    // Signal buffer is ready
    if (tid == 0u) {
        atomicStore(&buffer_ready[next_buffer], 1u);
    }
}

// ============================================================================
// Hierarchical Prefetch for Multi-Level Cache
// ============================================================================

struct HierarchicalParams {
    N: u32,
    k: u32,
    l: u32,
    l1_size: u32,    // L1 cache capacity (elements)
    l2_size: u32,    // L2 cache capacity (elements)
    prefetch_depth: u32, // How many entries ahead to prefetch
    current_idx: u32,
    _pad: u32,
}

@group(2) @binding(0) var<storage, read_write> l1_cache: array<u32>;
@group(2) @binding(1) var<storage, read_write> l2_cache: array<u32>;
@group(2) @binding(2) var<uniform> hier_params: HierarchicalParams;

// L2 prefetch: fetch multiple entries in background
@compute @workgroup_size(256)
fn bsk_prefetch_l2(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = hier_params.N;
    let k = hier_params.k;
    let l = hier_params.l;
    let tpg = 256u;

    let depth_idx = wgid.x;
    let entry_idx = hier_params.current_idx + depth_idx + 1u;

    if (depth_idx >= hier_params.prefetch_depth) {
        return;
    }

    let entry_size = (k + 1u) * l * (k + 1u) * N * 2u;
    let global_offset = entry_idx * entry_size;
    let l2_offset = (depth_idx % (hier_params.l2_size / entry_size)) * entry_size;

    let elements_per_thread = (entry_size + tpg - 1u) / tpg;

    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid + i * tpg;
        if (idx < entry_size) {
            l2_cache[l2_offset + idx] = bsk_global[global_offset + idx];
        }
    }
}

// L1 prefetch: move from L2 to L1 for immediate use
@compute @workgroup_size(256)
fn bsk_prefetch_l1(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = hier_params.N;
    let k = hier_params.k;
    let l = hier_params.l;
    let tpg = 256u;

    let entry_idx = hier_params.current_idx + 1u;

    let entry_size = (k + 1u) * l * (k + 1u) * N * 2u;
    let l2_offset = (0u % (hier_params.l2_size / entry_size)) * entry_size;
    let l1_offset = 0u;  // Always at start of L1

    // First, load to shared memory for coalesced access
    let elements_per_thread = (min(entry_size, hier_params.l1_size) + tpg - 1u) / tpg;

    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid + i * tpg;
        if (idx < entry_size && idx < hier_params.l1_size) {
            staging_buffer[idx % 4096u] = l2_cache[l2_offset + idx];
        }
    }

    workgroupBarrier();

    // Write to L1 cache
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid + i * tpg;
        if (idx < entry_size && idx < hier_params.l1_size) {
            l1_cache[l1_offset + idx] = staging_buffer[idx % 4096u];
        }
    }
}

// ============================================================================
// Speculative Prefetch Based on LWE Coefficients
// ============================================================================

@group(3) @binding(0) var<storage, read> lwe_coefficients: array<u32>;
@group(3) @binding(1) var<storage, read_write> speculative_cache: array<u32>;

// Analyze LWE coefficients to predict which BSK entries will be needed
@compute @workgroup_size(256)
fn bsk_analyze_lwe(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let n = params.n;
    let tpg = 256u;

    // Each thread analyzes a portion of LWE coefficients
    let coeffs_per_thread = (n + tpg - 1u) / tpg;

    for (var i = 0u; i < coeffs_per_thread; i++) {
        let coeff_idx = tid + i * tpg;
        if (coeff_idx >= n) {
            break;
        }

        // Get LWE coefficient (stored as u64)
        let a_lo = lwe_coefficients[coeff_idx * 2u];
        let a_hi = lwe_coefficients[coeff_idx * 2u + 1u];

        // Calculate rotation amount: round(a * 2N / 2^64)
        let rotation = (a_hi >> (32u - params.base_log)) & ((1u << params.base_log) - 1u);

        // Store predicted access pattern
        speculative_cache[coeff_idx] = rotation;
    }
}

// Prefetch based on analyzed patterns
@compute @workgroup_size(256)
fn bsk_speculative_prefetch(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let l = params.l;
    let tpg = 256u;

    let entry_idx = wgid.x;

    if (entry_idx >= params.prefetch_count) {
        return;
    }

    // Check if this entry is predicted to be needed
    let predicted = speculative_cache[entry_idx];

    // Skip if zero (no rotation needed)
    if (predicted == 0u) {
        return;
    }

    // Prefetch the entry
    let entry_size = (k + 1u) * l * (k + 1u) * N * 2u;
    let global_offset = entry_idx * entry_size;
    let cache_offset = (entry_idx % 16u) * entry_size;  // Circular buffer

    let elements_per_thread = (entry_size + tpg - 1u) / tpg;

    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid + i * tpg;
        if (idx < entry_size) {
            bsk_cache[cache_offset + idx] = bsk_global[global_offset + idx];
        }
    }

    storageBarrier();

    if (tid == 0u) {
        atomicStore(&prefetch_status[entry_idx], 1u);
    }
}

// ============================================================================
// Compact Prefetch for Low-Memory Devices
// ============================================================================

// Only prefetch the most frequently accessed parts of BSK
@compute @workgroup_size(256)
fn bsk_prefetch_compact(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = params.N;
    let k = params.k;
    let l = params.l;
    let tpg = 256u;

    let entry_idx = params.prefetch_start + wgid.x;
    let level_idx = wgid.y;  // Only prefetch specific levels

    if (entry_idx >= params.total_entries || level_idx >= l) {
        return;
    }

    // Only prefetch the body polynomial (index k) of each level
    // This is the most frequently accessed part
    let full_entry_size = (k + 1u) * l * (k + 1u) * N * 2u;
    let level_size = (k + 1u) * N * 2u;
    let poly_size = N * 2u;

    // Offset to body polynomial in this level
    let global_offset = entry_idx * full_entry_size + level_idx * level_size + k * poly_size;
    let cache_offset = (wgid.x * l + level_idx) * poly_size;

    let elements_per_thread = (poly_size + tpg - 1u) / tpg;

    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid + i * tpg;
        if (idx < poly_size) {
            bsk_cache[cache_offset + idx] = bsk_global[global_offset + idx];
        }
    }
}

// ============================================================================
// Wait for Prefetch Completion
// ============================================================================

@compute @workgroup_size(1)
fn bsk_wait_prefetch(
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let entry_idx = wgid.x;

    // Spin until prefetch is complete
    // In practice, host should use fence/barrier instead
    var ready = atomicLoad(&prefetch_status[entry_idx]);
    while (ready == 0u) {
        ready = atomicLoad(&prefetch_status[entry_idx]);
    }
}

// Reset prefetch status for next round
@compute @workgroup_size(256)
fn bsk_reset_status(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx < params.total_entries) {
        atomicStore(&prefetch_status[idx], 0u);
    }
}
