// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Memory Fence and Barrier Operations
//
// Provides synchronization primitives for WebGPU compute shaders:
// - Workgroup barriers (workgroupBarrier)
// - Storage barriers (storageBarrier)
// - Texture barriers (textureBarrier)
// - Memory coherence operations
//
// Note: WGSL provides three built-in barrier functions:
// - workgroupBarrier(): Synchronizes threads within a workgroup
// - storageBarrier(): Ensures storage operations are visible
// - textureBarrier(): Ensures texture operations are visible
//
// Part of the Lux Network GPU acceleration library

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// ============================================================================
// Parameter Structures
// ============================================================================

struct FenceParams {
    size: u32,          // Number of elements
    stride: u32,        // Stride between elements
    iterations: u32,    // Number of synchronization iterations
    _pad: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> flags: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: FenceParams;

// Workgroup shared memory for synchronization
var<workgroup> shared_data: array<u32, WORKGROUP_SIZE>;
var<workgroup> shared_flag: atomic<u32>;

// ============================================================================
// Basic Workgroup Barrier Example
// Demonstrates workgroupBarrier() for thread synchronization
// ============================================================================

@compute @workgroup_size(256)
fn workgroup_barrier_example(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let size = params.size;
    let base = wid.x * WORKGROUP_SIZE;

    // Phase 1: Load data to shared memory
    if (base + lid.x < size) {
        shared_data[lid.x] = data[base + lid.x];
    } else {
        shared_data[lid.x] = 0u;
    }

    // Barrier: Ensure all threads have loaded their data
    workgroupBarrier();

    // Phase 2: Process data (e.g., swap neighbors)
    let neighbor_idx = lid.x ^ 1u;  // XOR with 1 to get neighbor
    var my_val = shared_data[lid.x];
    var neighbor_val = shared_data[neighbor_idx];

    // Barrier: Ensure all reads complete before writes
    workgroupBarrier();

    // Phase 3: Write swapped values
    shared_data[lid.x] = neighbor_val;

    // Barrier: Ensure all writes complete
    workgroupBarrier();

    // Phase 4: Write back to global memory
    if (base + lid.x < size) {
        data[base + lid.x] = shared_data[lid.x];
    }
}

// ============================================================================
// Storage Barrier Example
// Ensures writes to storage buffers are visible across invocations
// ============================================================================

@compute @workgroup_size(256)
fn storage_barrier_example(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let size = params.size;
    let iterations = params.iterations;

    if (gid.x >= size) {
        return;
    }

    // Multiple iterations with storage barrier between each
    for (var iter = 0u; iter < iterations; iter++) {
        // Read from previous iteration's result
        var val = data[gid.x];

        // Simple transformation
        val = val + 1u;

        // Write result
        data[gid.x] = val;

        // Storage barrier: ensure write is visible before next read
        // Note: storageBarrier() synchronizes within workgroup only
        storageBarrier();
    }
}

// ============================================================================
// Parallel Reduction with Barriers
// Classic tree reduction using workgroup barriers
// ============================================================================

var<workgroup> reduction_shared: array<u32, WORKGROUP_SIZE>;

@compute @workgroup_size(256)
fn parallel_reduction(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let size = params.size;
    let base = wid.x * WORKGROUP_SIZE;

    // Load phase
    if (base + lid.x < size) {
        reduction_shared[lid.x] = data[base + lid.x];
    } else {
        reduction_shared[lid.x] = 0u;
    }

    workgroupBarrier();

    // Reduction phase (log2 steps)
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            reduction_shared[lid.x] = reduction_shared[lid.x] + reduction_shared[lid.x + stride];
        }
        workgroupBarrier();
    }

    // Write result
    if (lid.x == 0u) {
        data[wid.x] = reduction_shared[0];
    }
}

// ============================================================================
// Producer-Consumer with Atomic Flags
// Demonstrates inter-workgroup synchronization pattern
// ============================================================================

@compute @workgroup_size(256)
fn producer_consumer(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let size = params.size;

    // Producer workgroups (even IDs)
    if (wid.x % 2u == 0u) {
        let producer_idx = wid.x / 2u;
        let data_idx = producer_idx * WORKGROUP_SIZE + lid.x;

        if (data_idx < size) {
            // Produce data
            data[data_idx] = data_idx * 2u;

            // Ensure write is visible
            storageBarrier();

            // Signal completion (only thread 0)
            if (lid.x == 0u) {
                atomicStore(&flags[producer_idx], 1u);
            }
        }
    }
    // Consumer workgroups (odd IDs)
    else {
        let consumer_idx = (wid.x - 1u) / 2u;
        let data_idx = consumer_idx * WORKGROUP_SIZE + lid.x;

        // Wait for producer (only thread 0 checks)
        if (lid.x == 0u) {
            // Spin-wait for flag (simplified - real impl would be more sophisticated)
            for (var i = 0u; i < 1000u; i++) {
                if (atomicLoad(&flags[consumer_idx]) != 0u) {
                    break;
                }
            }
        }

        // Barrier to ensure all threads wait for flag check
        workgroupBarrier();

        if (data_idx < size) {
            // Consume and process data
            let val = data[data_idx];
            data[data_idx] = val + 1u;
        }
    }
}

// ============================================================================
// Prefix Sum (Scan) with Barriers
// Blelloch-style exclusive scan
// ============================================================================

@compute @workgroup_size(256)
fn prefix_sum(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let size = params.size;
    let base = wid.x * WORKGROUP_SIZE;

    // Load data
    if (base + lid.x < size) {
        shared_data[lid.x] = data[base + lid.x];
    } else {
        shared_data[lid.x] = 0u;
    }

    workgroupBarrier();

    // Up-sweep (reduce) phase
    var offset = 1u;
    for (var d = WORKGROUP_SIZE >> 1u; d > 0u; d >>= 1u) {
        if (lid.x < d) {
            let ai = offset * (2u * lid.x + 1u) - 1u;
            let bi = offset * (2u * lid.x + 2u) - 1u;
            shared_data[bi] = shared_data[bi] + shared_data[ai];
        }
        offset <<= 1u;
        workgroupBarrier();
    }

    // Clear the last element (for exclusive scan)
    if (lid.x == 0u) {
        shared_data[WORKGROUP_SIZE - 1u] = 0u;
    }
    workgroupBarrier();

    // Down-sweep phase
    for (var d = 1u; d < WORKGROUP_SIZE; d <<= 1u) {
        offset >>= 1u;
        if (lid.x < d) {
            let ai = offset * (2u * lid.x + 1u) - 1u;
            let bi = offset * (2u * lid.x + 2u) - 1u;
            let temp = shared_data[ai];
            shared_data[ai] = shared_data[bi];
            shared_data[bi] = shared_data[bi] + temp;
        }
        workgroupBarrier();
    }

    // Write result
    if (base + lid.x < size) {
        data[base + lid.x] = shared_data[lid.x];
    }
}

// ============================================================================
// Barrier-based Matrix Transpose
// Uses shared memory and barriers for coalesced access
// ============================================================================

struct TransposeParams {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(1) @binding(0) var<uniform> transpose_params: TransposeParams;
@group(1) @binding(1) var<storage, read> input_matrix: array<u32>;
@group(1) @binding(2) var<storage, read_write> output_matrix: array<u32>;

var<workgroup> transpose_tile: array<array<u32, 16>, 17>;  // +1 for bank conflict avoidance

@compute @workgroup_size(16, 16)
fn matrix_transpose(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let rows = transpose_params.rows;
    let cols = transpose_params.cols;

    let tile_x = wid.x * 16u;
    let tile_y = wid.y * 16u;

    let in_x = tile_x + lid.x;
    let in_y = tile_y + lid.y;

    // Load tile to shared memory (coalesced read)
    if (in_x < cols && in_y < rows) {
        transpose_tile[lid.y][lid.x] = input_matrix[in_y * cols + in_x];
    }

    // Barrier: ensure all loads complete
    workgroupBarrier();

    // Compute output position (transposed)
    let out_x = tile_y + lid.x;
    let out_y = tile_x + lid.y;

    // Store transposed tile (coalesced write)
    if (out_x < rows && out_y < cols) {
        output_matrix[out_y * rows + out_x] = transpose_tile[lid.x][lid.y];
    }
}

// ============================================================================
// Multi-Phase Computation with Barriers
// Demonstrates complex synchronization pattern
// ============================================================================

var<workgroup> phase_data: array<f32, WORKGROUP_SIZE>;
var<workgroup> phase_complete: u32;

@compute @workgroup_size(256)
fn multi_phase_compute(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let size = params.size;
    let iterations = params.iterations;
    let base = wid.x * WORKGROUP_SIZE;

    // Initialize shared memory
    if (base + lid.x < size) {
        phase_data[lid.x] = f32(data[base + lid.x]);
    } else {
        phase_data[lid.x] = 0.0;
    }

    workgroupBarrier();

    // Multiple phases
    for (var phase = 0u; phase < iterations; phase++) {
        // Phase computation depends on phase number
        switch (phase % 3u) {
            case 0u: {
                // Normalize phase
                var sum: f32 = 0.0;
                for (var i = 0u; i < WORKGROUP_SIZE; i++) {
                    sum += phase_data[i];
                }
                workgroupBarrier();

                if (sum > 0.0) {
                    phase_data[lid.x] = phase_data[lid.x] / sum;
                }
            }
            case 1u: {
                // Shift phase
                let next_idx = (lid.x + 1u) % WORKGROUP_SIZE;
                let temp = phase_data[next_idx];
                workgroupBarrier();
                phase_data[lid.x] = temp;
            }
            case 2u: {
                // Square phase
                phase_data[lid.x] = phase_data[lid.x] * phase_data[lid.x];
            }
            default: {}
        }

        workgroupBarrier();
    }

    // Write final result
    if (base + lid.x < size) {
        data[base + lid.x] = u32(phase_data[lid.x] * 1000.0);
    }
}

// ============================================================================
// Atomic Counter with Barrier
// Thread-safe counter using atomics and barriers
// ============================================================================

@group(2) @binding(0) var<storage, read_write> counter: atomic<u32>;
@group(2) @binding(1) var<storage, read_write> results: array<u32>;

@compute @workgroup_size(256)
fn atomic_counter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let size = params.size;

    if (gid.x >= size) {
        return;
    }

    // Each thread atomically increments counter and gets unique slot
    let slot = atomicAdd(&counter, 1u);

    // Barrier to ensure all increments complete
    storageBarrier();

    // Write thread ID to its unique slot
    results[slot] = gid.x;
}

// ============================================================================
// Lock-free Stack Push (for work stealing)
// ============================================================================

struct StackHeader {
    top: atomic<u32>,
    capacity: u32,
}

@group(3) @binding(0) var<storage, read_write> stack_header: StackHeader;
@group(3) @binding(1) var<storage, read_write> stack_data: array<u32>;

@compute @workgroup_size(256)
fn stack_push(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let size = params.size;
    let capacity = stack_header.capacity;

    if (gid.x >= size) {
        return;
    }

    // Atomically reserve slot
    let slot = atomicAdd(&stack_header.top, 1u);

    // Check bounds
    if (slot < capacity) {
        stack_data[slot] = gid.x;
    }
    // Note: overflow handling would need more sophisticated approach
}

// ============================================================================
// Memory Fence for Double Buffering
// Ping-pong buffer pattern with barrier synchronization
// ============================================================================

@group(4) @binding(0) var<storage, read_write> buffer_a: array<u32>;
@group(4) @binding(1) var<storage, read_write> buffer_b: array<u32>;
@group(4) @binding(2) var<storage, read_write> ping_pong: atomic<u32>;

@compute @workgroup_size(256)
fn double_buffer_compute(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let size = params.size;
    let iterations = params.iterations;

    if (gid.x >= size) {
        return;
    }

    for (var iter = 0u; iter < iterations; iter++) {
        let is_ping = (iter % 2u) == 0u;

        if (is_ping) {
            // Read from A, write to B
            let val = buffer_a[gid.x];
            buffer_b[gid.x] = val + 1u;
        } else {
            // Read from B, write to A
            let val = buffer_b[gid.x];
            buffer_a[gid.x] = val + 1u;
        }

        // Storage barrier between iterations
        storageBarrier();
    }
}
