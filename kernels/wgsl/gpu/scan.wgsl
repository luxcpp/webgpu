// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Prefix Scan Kernels - Inclusive and exclusive prefix operations
// Implements Blelloch's efficient parallel scan algorithm.
// Supports sum, product, min, max operations.
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;
const LOG2_WORKGROUP_SIZE: u32 = 8u;  // log2(256) = 8

// ============================================================================
// Parameter Structure
// ============================================================================

struct ScanParams {
    size: u32,           // Total number of elements
    inclusive: u32,      // 1 = inclusive, 0 = exclusive
    _pad1: u32,
    _pad2: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<f32>;
@group(0) @binding(3) var<uniform> params: ScanParams;

// ============================================================================
// Workgroup-Local Memory
// ============================================================================

var<workgroup> shared_data: array<f32, 512>;  // 2 * WORKGROUP_SIZE for double buffer

// ============================================================================
// Inclusive Scan (Sum) - Single Workgroup
// Uses Hillis-Steele algorithm for small arrays
// ============================================================================

@compute @workgroup_size(256)
fn inclusive_scan_sum_local(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_idx = lid.x;
    let global_idx = gid.x;

    // Load data into shared memory
    if (global_idx < params.size) {
        shared_data[local_idx] = input[global_idx];
    } else {
        shared_data[local_idx] = 0.0f;
    }
    workgroupBarrier();

    // Hillis-Steele scan (inclusive)
    for (var offset = 1u; offset < WORKGROUP_SIZE; offset = offset * 2u) {
        var val = shared_data[local_idx];
        if (local_idx >= offset) {
            val = val + shared_data[local_idx - offset];
        }
        workgroupBarrier();
        shared_data[local_idx] = val;
        workgroupBarrier();
    }

    // Write result
    if (global_idx < params.size) {
        output[global_idx] = shared_data[local_idx];
    }
}

// ============================================================================
// Exclusive Scan (Sum) - Single Workgroup
// ============================================================================

@compute @workgroup_size(256)
fn exclusive_scan_sum_local(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_idx = lid.x;
    let global_idx = gid.x;

    // Load data
    var val = 0.0f;
    if (global_idx < params.size) {
        val = input[global_idx];
    }
    shared_data[local_idx] = val;
    workgroupBarrier();

    // Blelloch scan - Up-sweep (reduce)
    for (var d = 0u; d < LOG2_WORKGROUP_SIZE; d = d + 1u) {
        let stride = 1u << (d + 1u);
        let idx = (local_idx + 1u) * stride - 1u;
        if (idx < WORKGROUP_SIZE) {
            shared_data[idx] = shared_data[idx] + shared_data[idx - (stride >> 1u)];
        }
        workgroupBarrier();
    }

    // Set last element to zero (exclusive scan)
    if (local_idx == 0u) {
        shared_data[WORKGROUP_SIZE - 1u] = 0.0f;
    }
    workgroupBarrier();

    // Down-sweep
    for (var d = LOG2_WORKGROUP_SIZE; d > 0u; d = d - 1u) {
        let stride = 1u << d;
        let idx = (local_idx + 1u) * stride - 1u;
        if (idx < WORKGROUP_SIZE) {
            let temp = shared_data[idx - (stride >> 1u)];
            shared_data[idx - (stride >> 1u)] = shared_data[idx];
            shared_data[idx] = shared_data[idx] + temp;
        }
        workgroupBarrier();
    }

    // Write result
    if (global_idx < params.size) {
        output[global_idx] = shared_data[local_idx];
    }
}

// ============================================================================
// Block-Level Scan (First Pass) - For large arrays
// Each workgroup scans its block and stores the block sum
// ============================================================================

@compute @workgroup_size(256)
fn scan_block_sum(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_idx = lid.x;
    let global_idx = gid.x;
    let block_idx = wid.x;

    // Load data
    var val = 0.0f;
    if (global_idx < params.size) {
        val = input[global_idx];
    }
    shared_data[local_idx] = val;
    workgroupBarrier();

    // Inclusive scan within block
    for (var offset = 1u; offset < WORKGROUP_SIZE; offset = offset * 2u) {
        var temp = shared_data[local_idx];
        if (local_idx >= offset) {
            temp = temp + shared_data[local_idx - offset];
        }
        workgroupBarrier();
        shared_data[local_idx] = temp;
        workgroupBarrier();
    }

    // Write block result
    if (global_idx < params.size) {
        output[global_idx] = shared_data[local_idx];
    }

    // Store block sum (last element of each block)
    if (local_idx == WORKGROUP_SIZE - 1u) {
        block_sums[block_idx] = shared_data[local_idx];
    }
}

// ============================================================================
// Add Block Sums (Second Pass) - Propagate block sums
// ============================================================================

@compute @workgroup_size(256)
fn add_block_sums(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let global_idx = gid.x;
    let block_idx = wid.x;

    if (global_idx >= params.size) { return; }
    if (block_idx == 0u) { return; }  // First block has no prefix to add

    // Add the scanned block sum from all previous blocks
    output[global_idx] = output[global_idx] + block_sums[block_idx - 1u];
}

// ============================================================================
// Product Scan (Inclusive)
// ============================================================================

@compute @workgroup_size(256)
fn inclusive_scan_prod_local(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_idx = lid.x;
    let global_idx = gid.x;

    // Load data (identity for product is 1.0)
    if (global_idx < params.size) {
        shared_data[local_idx] = input[global_idx];
    } else {
        shared_data[local_idx] = 1.0f;
    }
    workgroupBarrier();

    // Hillis-Steele scan with multiplication
    for (var offset = 1u; offset < WORKGROUP_SIZE; offset = offset * 2u) {
        var val = shared_data[local_idx];
        if (local_idx >= offset) {
            val = val * shared_data[local_idx - offset];
        }
        workgroupBarrier();
        shared_data[local_idx] = val;
        workgroupBarrier();
    }

    if (global_idx < params.size) {
        output[global_idx] = shared_data[local_idx];
    }
}

// ============================================================================
// Min Scan (Inclusive)
// ============================================================================

@compute @workgroup_size(256)
fn inclusive_scan_min_local(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_idx = lid.x;
    let global_idx = gid.x;

    // Load data (identity for min is +infinity)
    if (global_idx < params.size) {
        shared_data[local_idx] = input[global_idx];
    } else {
        shared_data[local_idx] = 3.402823466e+38f;  // FLT_MAX
    }
    workgroupBarrier();

    for (var offset = 1u; offset < WORKGROUP_SIZE; offset = offset * 2u) {
        var val = shared_data[local_idx];
        if (local_idx >= offset) {
            val = min(val, shared_data[local_idx - offset]);
        }
        workgroupBarrier();
        shared_data[local_idx] = val;
        workgroupBarrier();
    }

    if (global_idx < params.size) {
        output[global_idx] = shared_data[local_idx];
    }
}

// ============================================================================
// Max Scan (Inclusive)
// ============================================================================

@compute @workgroup_size(256)
fn inclusive_scan_max_local(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_idx = lid.x;
    let global_idx = gid.x;

    // Load data (identity for max is -infinity)
    if (global_idx < params.size) {
        shared_data[local_idx] = input[global_idx];
    } else {
        shared_data[local_idx] = -3.402823466e+38f;  // -FLT_MAX
    }
    workgroupBarrier();

    for (var offset = 1u; offset < WORKGROUP_SIZE; offset = offset * 2u) {
        var val = shared_data[local_idx];
        if (local_idx >= offset) {
            val = max(val, shared_data[local_idx - offset]);
        }
        workgroupBarrier();
        shared_data[local_idx] = val;
        workgroupBarrier();
    }

    if (global_idx < params.size) {
        output[global_idx] = shared_data[local_idx];
    }
}

// ============================================================================
// Integer Scan Operations
// ============================================================================

@group(0) @binding(4) var<storage, read> input_i32: array<i32>;
@group(0) @binding(5) var<storage, read_write> output_i32: array<i32>;
@group(0) @binding(6) var<storage, read_write> block_sums_i32: array<i32>;

var<workgroup> shared_data_i32: array<i32, 512>;

@compute @workgroup_size(256)
fn inclusive_scan_sum_i32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_idx = lid.x;
    let global_idx = gid.x;

    if (global_idx < params.size) {
        shared_data_i32[local_idx] = input_i32[global_idx];
    } else {
        shared_data_i32[local_idx] = 0;
    }
    workgroupBarrier();

    for (var offset = 1u; offset < WORKGROUP_SIZE; offset = offset * 2u) {
        var val = shared_data_i32[local_idx];
        if (local_idx >= offset) {
            val = val + shared_data_i32[local_idx - offset];
        }
        workgroupBarrier();
        shared_data_i32[local_idx] = val;
        workgroupBarrier();
    }

    if (global_idx < params.size) {
        output_i32[global_idx] = shared_data_i32[local_idx];
    }
}

// ============================================================================
// Unsigned Integer Scan (for counting)
// ============================================================================

@group(0) @binding(7) var<storage, read> input_u32: array<u32>;
@group(0) @binding(8) var<storage, read_write> output_u32: array<u32>;

var<workgroup> shared_data_u32: array<u32, 512>;

@compute @workgroup_size(256)
fn inclusive_scan_sum_u32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_idx = lid.x;
    let global_idx = gid.x;

    if (global_idx < params.size) {
        shared_data_u32[local_idx] = input_u32[global_idx];
    } else {
        shared_data_u32[local_idx] = 0u;
    }
    workgroupBarrier();

    for (var offset = 1u; offset < WORKGROUP_SIZE; offset = offset * 2u) {
        var val = shared_data_u32[local_idx];
        if (local_idx >= offset) {
            val = val + shared_data_u32[local_idx - offset];
        }
        workgroupBarrier();
        shared_data_u32[local_idx] = val;
        workgroupBarrier();
    }

    if (global_idx < params.size) {
        output_u32[global_idx] = shared_data_u32[local_idx];
    }
}

// ============================================================================
// Segmented Scan - Reset at segment boundaries
// ============================================================================

struct SegmentedParams {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(9) var<storage, read> flags: array<u32>;  // 1 = start new segment
@group(0) @binding(10) var<uniform> seg_params: SegmentedParams;

var<workgroup> shared_flags: array<u32, 256>;

@compute @workgroup_size(256)
fn segmented_scan_sum(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_idx = lid.x;
    let global_idx = gid.x;

    // Load data and flags
    var val = 0.0f;
    var flag = 0u;
    if (global_idx < seg_params.size) {
        val = input[global_idx];
        flag = flags[global_idx];
    }

    shared_data[local_idx] = val;
    shared_flags[local_idx] = flag;
    workgroupBarrier();

    // Segmented scan - propagate flag along with value
    for (var offset = 1u; offset < WORKGROUP_SIZE; offset = offset * 2u) {
        var my_val = shared_data[local_idx];
        var my_flag = shared_flags[local_idx];

        if (local_idx >= offset) {
            let other_val = shared_data[local_idx - offset];
            let other_flag = shared_flags[local_idx - offset];

            // Only add if not a segment boundary
            if (my_flag == 0u) {
                my_val = my_val + other_val;
                my_flag = other_flag;
            }
        }

        workgroupBarrier();
        shared_data[local_idx] = my_val;
        shared_flags[local_idx] = my_flag;
        workgroupBarrier();
    }

    if (global_idx < seg_params.size) {
        output[global_idx] = shared_data[local_idx];
    }
}

// ============================================================================
// Reverse Scan (scan from right to left)
// ============================================================================

@compute @workgroup_size(256)
fn reverse_scan_sum_local(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_idx = lid.x;
    let global_idx = gid.x;

    // Load in reverse order
    let rev_local_idx = WORKGROUP_SIZE - 1u - local_idx;

    if (global_idx < params.size) {
        let src_idx = params.size - 1u - global_idx;
        shared_data[rev_local_idx] = input[src_idx];
    } else {
        shared_data[rev_local_idx] = 0.0f;
    }
    workgroupBarrier();

    // Standard forward scan on reversed data
    for (var offset = 1u; offset < WORKGROUP_SIZE; offset = offset * 2u) {
        var val = shared_data[local_idx];
        if (local_idx >= offset) {
            val = val + shared_data[local_idx - offset];
        }
        workgroupBarrier();
        shared_data[local_idx] = val;
        workgroupBarrier();
    }

    // Write back in reverse order
    if (global_idx < params.size) {
        let dst_idx = params.size - 1u - global_idx;
        output[dst_idx] = shared_data[rev_local_idx];
    }
}
