// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// ntt_unified_memory.wgsl - NTT with optimized unified memory access
//
// Implements NTT optimized for unified memory architectures (Apple Silicon, etc.)
// Features:
// - Coalesced memory access patterns
// - Minimal memory traffic through register blocking
// - Hierarchical twiddle access
// - Adaptive workgroup sizing based on transform size
// - Support for batched transforms with optimal memory layout

// ============================================================================
// 64-bit integer emulation
// ============================================================================

struct u64 {
    lo: u32,
    hi: u32,
}

fn u64_zero() -> u64 {
    return u64(0u, 0u);
}

fn u64_from_u32(x: u32) -> u64 {
    return u64(x, 0u);
}

fn u64_add(a: u64, b: u64) -> u64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    let hi = a.hi + b.hi + carry;
    return u64(lo, hi);
}

fn u64_sub(a: u64, b: u64) -> u64 {
    let borrow = select(0u, 1u, a.lo < b.lo);
    let lo = a.lo - b.lo;
    let hi = a.hi - b.hi - borrow;
    return u64(lo, hi);
}

fn u64_mul(a: u64, b: u64) -> u64 {
    let a_lo = a.lo & 0xFFFFu;
    let a_hi = a.lo >> 16u;
    let b_lo = b.lo & 0xFFFFu;
    let b_hi = b.lo >> 16u;

    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;

    let mid = p1 + p2;
    let lo = p0 + ((mid & 0xFFFFu) << 16u);
    let carry = select(0u, 1u, lo < p0);
    let hi = p3 + (mid >> 16u) + carry + a.lo * b.hi + a.hi * b.lo;

    return u64(lo, hi);
}

fn u64_ge(a: u64, b: u64) -> bool {
    if (a.hi != b.hi) {
        return a.hi > b.hi;
    }
    return a.lo >= b.lo;
}

fn u64_mulhi(a: u64, b: u64) -> u64 {
    let a0 = a.lo & 0xFFFFu;
    let a1 = a.lo >> 16u;
    let a2 = a.hi & 0xFFFFu;
    let a3 = a.hi >> 16u;
    let b0 = b.lo & 0xFFFFu;
    let b1 = b.lo >> 16u;
    let b2 = b.hi & 0xFFFFu;
    let b3 = b.hi >> 16u;

    var acc = u64_zero();
    acc = u64_add(acc, u64_from_u32(a2 * b2));
    acc = u64_add(acc, u64_from_u32(a3 * b1));
    acc = u64_add(acc, u64_from_u32(a1 * b3));
    let hi_lo = acc.lo + a3 * b2 + a2 * b3;
    let hi_hi = acc.hi + a3 * b3;

    return u64(hi_lo, hi_hi);
}

// ============================================================================
// Modular arithmetic
// ============================================================================

fn mod_add(a: u64, b: u64, Q: u64) -> u64 {
    var result = u64_add(a, b);
    if (u64_ge(result, Q)) {
        result = u64_sub(result, Q);
    }
    return result;
}

fn mod_sub(a: u64, b: u64, Q: u64) -> u64 {
    if (u64_ge(a, b)) {
        return u64_sub(a, b);
    }
    return u64_sub(u64_add(a, Q), b);
}

fn barrett_mul(a: u64, b: u64, Q: u64, mu: u64) -> u64 {
    let product = u64_mul(a, b);
    let q_approx = u64_mulhi(product, mu);
    let qQ = u64_mul(q_approx, Q);
    var result = u64_sub(product, qQ);
    if (u64_ge(result, Q)) {
        result = u64_sub(result, Q);
    }
    if (u64_ge(result, Q)) {
        result = u64_sub(result, Q);
    }
    return result;
}

// ============================================================================
// Shared memory - sized for unified memory optimal access
// ============================================================================

const WG_SIZE: u32 = 256u;
const ELEMENTS_PER_THREAD: u32 = 4u;  // Register blocking factor
const SHARED_SIZE: u32 = 1024u;       // WG_SIZE * ELEMENTS_PER_THREAD

// Split shared memory for better cache behavior
var<workgroup> shared_even: array<u64, 512>;
var<workgroup> shared_odd: array<u64, 512>;
var<workgroup> shared_twiddles: array<u64, 256>;

// ============================================================================
// Parameters
// ============================================================================

struct UnifiedParams {
    N: u32,              // Transform size
    log_N: u32,
    num_polys: u32,      // Batch size
    prime_idx: u32,
    is_inverse: u32,
    memory_layout: u32,  // 0=standard, 1=interleaved, 2=blocked
    block_size: u32,     // For blocked layout
    prefetch_dist: u32,  // Prefetch distance (cache lines ahead)
}

@group(0) @binding(0) var<uniform> params: UnifiedParams;
@group(0) @binding(1) var<storage, read_write> data: array<u64>;
@group(0) @binding(2) var<storage, read> primes: array<u64>;
@group(0) @binding(3) var<storage, read> barrett_mu: array<u64>;
@group(0) @binding(4) var<storage, read> twiddles: array<u64>;
@group(0) @binding(5) var<storage, read> twiddles_inv: array<u64>;
@group(0) @binding(6) var<storage, read> N_inv: array<u64>;

// ============================================================================
// Memory access helpers for unified memory
// ============================================================================

// Convert logical index to physical index based on memory layout
fn get_physical_idx(poly_idx: u32, logical_idx: u32, N: u32) -> u32 {
    if (params.memory_layout == 0u) {
        // Standard layout: poly0[0..N], poly1[0..N], ...
        return poly_idx * N + logical_idx;
    } else if (params.memory_layout == 1u) {
        // Interleaved layout: poly0[0], poly1[0], ..., poly0[1], poly1[1], ...
        return logical_idx * params.num_polys + poly_idx;
    } else {
        // Blocked layout: cache-line aligned blocks
        let block_idx = logical_idx / params.block_size;
        let within_block = logical_idx % params.block_size;
        return (block_idx * params.num_polys + poly_idx) * params.block_size + within_block;
    }
}

// Coalesced load for a group of threads
fn coalesced_load(base_idx: u32, local_id: u32, stride: u32) -> u64 {
    let idx = base_idx + local_id * stride;
    return data[idx];
}

// ============================================================================
// Butterflies
// ============================================================================

fn ct_butterfly(a: ptr<function, u64>, b: ptr<function, u64>, w: u64, Q: u64, mu: u64) {
    let t = barrett_mul(*b, w, Q, mu);
    let u = *a;
    *a = mod_add(u, t, Q);
    *b = mod_sub(u, t, Q);
}

fn gs_butterfly(a: ptr<function, u64>, b: ptr<function, u64>, w: u64, Q: u64, mu: u64) {
    let u = *a;
    let v = *b;
    *a = mod_add(u, v, Q);
    let diff = mod_sub(u, v, Q);
    *b = barrett_mul(diff, w, Q, mu);
}

// ============================================================================
// Register-blocked NTT for small transforms
// ============================================================================

@compute @workgroup_size(256)
fn ntt_register_blocked(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_id = lid.x;
    let poly_idx = wid.x;

    let N = params.N;
    let log_N = params.log_N;
    let prime_idx = params.prime_idx;

    if (N > SHARED_SIZE) {
        return;  // Use four-step NTT for large transforms
    }

    let Q = primes[prime_idx];
    let mu = barrett_mu[prime_idx];

    // Each thread loads ELEMENTS_PER_THREAD consecutive elements
    var regs: array<u64, 4>;

    let base_idx = local_id * ELEMENTS_PER_THREAD;
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let idx = base_idx + i;
        if (idx < N) {
            let phys_idx = get_physical_idx(poly_idx, idx, N);
            regs[i] = data[phys_idx];
        }
    }

    // Store to shared memory in split layout for bank conflict avoidance
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let idx = base_idx + i;
        if (idx < N) {
            if ((idx & 1u) == 0u) {
                shared_even[idx >> 1u] = regs[i];
            } else {
                shared_odd[idx >> 1u] = regs[i];
            }
        }
    }

    // Load twiddles into shared memory
    let half_N = N >> 1u;
    if (local_id < half_N && local_id < 256u) {
        let twiddle_table = select(twiddles, twiddles_inv, params.is_inverse != 0u);
        shared_twiddles[local_id] = twiddle_table[local_id];
    }

    workgroupBarrier();

    // Perform NTT stages
    // First stages: use shared memory
    var m = N;
    var stage = 0u;

    while (m > 1u && m > ELEMENTS_PER_THREAD * 2u) {
        let half_m = m >> 1u;
        let num_groups = N / m;

        for (var k = local_id; k < N >> 1u; k += WG_SIZE) {
            let group = k / half_m;
            let j = k % half_m;
            let idx0 = group * m + j;
            let idx1 = idx0 + half_m;

            // Load from split shared memory
            var a: u64;
            var b: u64;
            if ((idx0 & 1u) == 0u) {
                a = shared_even[idx0 >> 1u];
            } else {
                a = shared_odd[idx0 >> 1u];
            }
            if ((idx1 & 1u) == 0u) {
                b = shared_even[idx1 >> 1u];
            } else {
                b = shared_odd[idx1 >> 1u];
            }

            let twiddle_idx = j * num_groups;
            let w = shared_twiddles[twiddle_idx % 256u];

            ct_butterfly(&a, &b, w, Q, mu);

            // Store back
            if ((idx0 & 1u) == 0u) {
                shared_even[idx0 >> 1u] = a;
            } else {
                shared_odd[idx0 >> 1u] = a;
            }
            if ((idx1 & 1u) == 0u) {
                shared_even[idx1 >> 1u] = b;
            } else {
                shared_odd[idx1 >> 1u] = b;
            }
        }

        workgroupBarrier();
        m = half_m;
        stage += 1u;
    }

    // Final stages: use registers (no barrier needed)
    // Load back into registers
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let idx = base_idx + i;
        if (idx < N) {
            if ((idx & 1u) == 0u) {
                regs[i] = shared_even[idx >> 1u];
            } else {
                regs[i] = shared_odd[idx >> 1u];
            }
        }
    }

    // In-register butterflies for final stages
    while (m > 1u) {
        let half_m = m >> 1u;
        let num_groups = ELEMENTS_PER_THREAD / m;

        for (var g = 0u; g < num_groups; g++) {
            for (var j = 0u; j < half_m; j++) {
                let idx0 = g * m + j;
                let idx1 = idx0 + half_m;

                if (idx0 < ELEMENTS_PER_THREAD && idx1 < ELEMENTS_PER_THREAD) {
                    let global_idx0 = base_idx + idx0;
                    let twiddle_idx = j * (N / m);
                    let w = shared_twiddles[twiddle_idx % 256u];

                    var a = regs[idx0];
                    var b = regs[idx1];
                    ct_butterfly(&a, &b, w, Q, mu);
                    regs[idx0] = a;
                    regs[idx1] = b;
                }
            }
        }

        m = half_m;
    }

    // Write back with coalesced access pattern
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let idx = base_idx + i;
        if (idx < N) {
            // Bit-reverse for output
            var rev = 0u;
            var temp = idx;
            for (var b = 0u; b < log_N; b++) {
                rev = (rev << 1u) | (temp & 1u);
                temp = temp >> 1u;
            }
            let phys_idx = get_physical_idx(poly_idx, rev, N);
            data[phys_idx] = regs[i];
        }
    }
}

// ============================================================================
// Streaming NTT for very large transforms
// ============================================================================

struct StreamParams {
    stream_size: u32,    // Size of each streaming chunk
    current_chunk: u32,
    total_chunks: u32,
    _pad: u32,
}

@group(1) @binding(0) var<uniform> stream_params: StreamParams;
@group(1) @binding(1) var<storage, read_write> stream_buffer: array<u64>;

@compute @workgroup_size(256)
fn ntt_streaming_load(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let poly_idx = gid.y;
    let chunk = stream_params.current_chunk;

    let chunk_size = stream_params.stream_size;
    let N = params.N;

    if (idx >= chunk_size) {
        return;
    }

    let global_idx = chunk * chunk_size + idx;
    if (global_idx >= N) {
        return;
    }

    let phys_idx = get_physical_idx(poly_idx, global_idx, N);
    let buffer_idx = poly_idx * chunk_size + idx;

    stream_buffer[buffer_idx] = data[phys_idx];
}

@compute @workgroup_size(256)
fn ntt_streaming_compute(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_id = lid.x;
    let poly_idx = wid.y;

    let chunk_size = stream_params.stream_size;
    let prime_idx = params.prime_idx;

    let Q = primes[prime_idx];
    let mu = barrett_mu[prime_idx];

    // Load chunk into shared memory
    let buffer_base = poly_idx * chunk_size;

    for (var i = local_id; i < chunk_size && i < SHARED_SIZE; i += WG_SIZE) {
        if ((i & 1u) == 0u) {
            shared_even[i >> 1u] = stream_buffer[buffer_base + i];
        } else {
            shared_odd[i >> 1u] = stream_buffer[buffer_base + i];
        }
    }

    workgroupBarrier();

    // Perform partial NTT on chunk
    let log_chunk = 32u - countLeadingZeros(chunk_size - 1u);  // ceil(log2(chunk_size))
    var m = chunk_size;

    while (m > 1u) {
        let half_m = m >> 1u;
        let num_groups = chunk_size / m;

        for (var k = local_id; k < chunk_size >> 1u; k += WG_SIZE) {
            let group = k / half_m;
            let j = k % half_m;
            let idx0 = group * m + j;
            let idx1 = idx0 + half_m;

            var a: u64;
            var b: u64;
            if ((idx0 & 1u) == 0u) {
                a = shared_even[idx0 >> 1u];
            } else {
                a = shared_odd[idx0 >> 1u];
            }
            if ((idx1 & 1u) == 0u) {
                b = shared_even[idx1 >> 1u];
            } else {
                b = shared_odd[idx1 >> 1u];
            }

            let twiddle_idx = j * num_groups;
            let twiddle_table = select(twiddles, twiddles_inv, params.is_inverse != 0u);
            let w = twiddle_table[twiddle_idx];

            ct_butterfly(&a, &b, w, Q, mu);

            if ((idx0 & 1u) == 0u) {
                shared_even[idx0 >> 1u] = a;
            } else {
                shared_odd[idx0 >> 1u] = a;
            }
            if ((idx1 & 1u) == 0u) {
                shared_even[idx1 >> 1u] = b;
            } else {
                shared_odd[idx1 >> 1u] = b;
            }
        }

        workgroupBarrier();
        m = half_m;
    }

    // Store back to stream buffer
    for (var i = local_id; i < chunk_size && i < SHARED_SIZE; i += WG_SIZE) {
        if ((i & 1u) == 0u) {
            stream_buffer[buffer_base + i] = shared_even[i >> 1u];
        } else {
            stream_buffer[buffer_base + i] = shared_odd[i >> 1u];
        }
    }
}

@compute @workgroup_size(256)
fn ntt_streaming_store(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let poly_idx = gid.y;
    let chunk = stream_params.current_chunk;

    let chunk_size = stream_params.stream_size;
    let N = params.N;

    if (idx >= chunk_size) {
        return;
    }

    let global_idx = chunk * chunk_size + idx;
    if (global_idx >= N) {
        return;
    }

    let phys_idx = get_physical_idx(poly_idx, global_idx, N);
    let buffer_idx = poly_idx * chunk_size + idx;

    data[phys_idx] = stream_buffer[buffer_idx];
}

// ============================================================================
// Batched NTT with optimal memory coalescing
// ============================================================================

@compute @workgroup_size(256)
fn ntt_batched_coalesced(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_id = lid.x;
    let batch_idx = wid.x;

    let N = params.N;
    let log_N = params.log_N;
    let prime_idx = params.prime_idx;
    let num_polys = params.num_polys;

    if (batch_idx >= num_polys || N > SHARED_SIZE) {
        return;
    }

    let Q = primes[prime_idx];
    let mu = barrett_mu[prime_idx];

    // Coalesced load: threads load consecutive elements across batch
    // This maximizes memory bandwidth utilization
    for (var i = local_id; i < N; i += WG_SIZE) {
        let phys_idx = get_physical_idx(batch_idx, i, N);
        let val = data[phys_idx];

        if ((i & 1u) == 0u) {
            shared_even[i >> 1u] = val;
        } else {
            shared_odd[i >> 1u] = val;
        }
    }

    // Load twiddles
    let half_N = N >> 1u;
    if (local_id < half_N && local_id < 256u) {
        let twiddle_table = select(twiddles, twiddles_inv, params.is_inverse != 0u);
        shared_twiddles[local_id] = twiddle_table[local_id];
    }

    workgroupBarrier();

    // NTT computation
    var m = N;
    while (m > 1u) {
        let half_m = m >> 1u;
        let num_groups = N / m;

        for (var k = local_id; k < N >> 1u; k += WG_SIZE) {
            let group = k / half_m;
            let j = k % half_m;
            let idx0 = group * m + j;
            let idx1 = idx0 + half_m;

            var a: u64;
            var b: u64;

            if ((idx0 & 1u) == 0u) {
                a = shared_even[idx0 >> 1u];
            } else {
                a = shared_odd[idx0 >> 1u];
            }
            if ((idx1 & 1u) == 0u) {
                b = shared_even[idx1 >> 1u];
            } else {
                b = shared_odd[idx1 >> 1u];
            }

            let twiddle_idx = j * num_groups;
            let w = shared_twiddles[twiddle_idx % 256u];

            ct_butterfly(&a, &b, w, Q, mu);

            if ((idx0 & 1u) == 0u) {
                shared_even[idx0 >> 1u] = a;
            } else {
                shared_odd[idx0 >> 1u] = a;
            }
            if ((idx1 & 1u) == 0u) {
                shared_even[idx1 >> 1u] = b;
            } else {
                shared_odd[idx1 >> 1u] = b;
            }
        }

        workgroupBarrier();
        m = half_m;
    }

    // Bit-reversed store
    for (var i = local_id; i < N; i += WG_SIZE) {
        var rev = 0u;
        var temp = i;
        for (var b = 0u; b < log_N; b++) {
            rev = (rev << 1u) | (temp & 1u);
            temp = temp >> 1u;
        }

        var val: u64;
        if ((i & 1u) == 0u) {
            val = shared_even[i >> 1u];
        } else {
            val = shared_odd[i >> 1u];
        }

        let phys_idx = get_physical_idx(batch_idx, rev, N);
        data[phys_idx] = val;
    }
}

// ============================================================================
// Inverse NTT with scaling
// ============================================================================

@compute @workgroup_size(256)
fn intt_unified(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_id = lid.x;
    let poly_idx = wid.x;

    let N = params.N;
    let log_N = params.log_N;
    let prime_idx = params.prime_idx;

    if (N > SHARED_SIZE) {
        return;
    }

    let Q = primes[prime_idx];
    let mu = barrett_mu[prime_idx];
    let n_inv = N_inv[prime_idx];

    // Load
    for (var i = local_id; i < N; i += WG_SIZE) {
        let phys_idx = get_physical_idx(poly_idx, i, N);
        let val = data[phys_idx];

        if ((i & 1u) == 0u) {
            shared_even[i >> 1u] = val;
        } else {
            shared_odd[i >> 1u] = val;
        }
    }

    // Load inverse twiddles
    let half_N = N >> 1u;
    if (local_id < half_N && local_id < 256u) {
        shared_twiddles[local_id] = twiddles_inv[local_id];
    }

    workgroupBarrier();

    // Inverse NTT using Gentleman-Sande (DIT)
    var m = 2u;
    while (m <= N) {
        let half_m = m >> 1u;

        for (var k = local_id; k < N >> 1u; k += WG_SIZE) {
            let group = k / half_m;
            let j = k % half_m;
            let idx0 = group * m + j;
            let idx1 = idx0 + half_m;

            var a: u64;
            var b: u64;

            if ((idx0 & 1u) == 0u) {
                a = shared_even[idx0 >> 1u];
            } else {
                a = shared_odd[idx0 >> 1u];
            }
            if ((idx1 & 1u) == 0u) {
                b = shared_even[idx1 >> 1u];
            } else {
                b = shared_odd[idx1 >> 1u];
            }

            let twiddle_idx = j * (N / m);
            let w = shared_twiddles[twiddle_idx % 256u];

            gs_butterfly(&a, &b, w, Q, mu);

            if ((idx0 & 1u) == 0u) {
                shared_even[idx0 >> 1u] = a;
            } else {
                shared_odd[idx0 >> 1u] = a;
            }
            if ((idx1 & 1u) == 0u) {
                shared_even[idx1 >> 1u] = b;
            } else {
                shared_odd[idx1 >> 1u] = b;
            }
        }

        workgroupBarrier();
        m = m << 1u;
    }

    // Scale by 1/N and store
    for (var i = local_id; i < N; i += WG_SIZE) {
        var val: u64;
        if ((i & 1u) == 0u) {
            val = shared_even[i >> 1u];
        } else {
            val = shared_odd[i >> 1u];
        }

        // Scale
        val = barrett_mul(val, n_inv, Q, mu);

        let phys_idx = get_physical_idx(poly_idx, i, N);
        data[phys_idx] = val;
    }
}

// ============================================================================
// Memory layout conversion
// ============================================================================

@compute @workgroup_size(256)
fn convert_layout(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let N = params.N;
    let num_polys = params.num_polys;

    if (idx >= N * num_polys) {
        return;
    }

    let poly_idx = idx / N;
    let within_poly = idx % N;

    // Calculate source and destination indices for layout conversion
    let src_layout = params.memory_layout;
    let dst_layout = (src_layout + 1u) % 3u;

    var src_idx: u32;
    var dst_idx: u32;

    // Source index
    if (src_layout == 0u) {
        src_idx = poly_idx * N + within_poly;
    } else if (src_layout == 1u) {
        src_idx = within_poly * num_polys + poly_idx;
    } else {
        let block_idx = within_poly / params.block_size;
        let within_block = within_poly % params.block_size;
        src_idx = (block_idx * num_polys + poly_idx) * params.block_size + within_block;
    }

    // Destination index
    if (dst_layout == 0u) {
        dst_idx = poly_idx * N + within_poly;
    } else if (dst_layout == 1u) {
        dst_idx = within_poly * num_polys + poly_idx;
    } else {
        let block_idx = within_poly / params.block_size;
        let within_block = within_poly % params.block_size;
        dst_idx = (block_idx * num_polys + poly_idx) * params.block_size + within_block;
    }

    // Use stream_buffer as temporary storage
    stream_buffer[dst_idx] = data[src_idx];
}

@compute @workgroup_size(256)
fn copy_from_temp(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let total = params.N * params.num_polys;

    if (idx >= total) {
        return;
    }

    data[idx] = stream_buffer[idx];
}
