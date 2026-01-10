// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Arg Reduction Kernels - Argmin/Argmax with parallel reduction
// Uses workgroup-local memory for efficient tree reduction.
// Returns indices of minimum/maximum values along specified axis.
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// ============================================================================
// Parameter Structure
// ============================================================================

struct ArgReduceParams {
    size: u32,           // Total number of elements
    axis_size: u32,      // Size along reduction axis
    stride: u32,         // Stride along reduction axis
    num_reductions: u32, // Number of independent reductions
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(3) var<uniform> params: ArgReduceParams;

// ============================================================================
// Workgroup-Local Memory
// ============================================================================

var<workgroup> shared_values: array<f32, 256>;
var<workgroup> shared_indices: array<u32, 256>;

// ============================================================================
// Argmax - Single reduction (1D array)
// ============================================================================

@compute @workgroup_size(256)
fn argmax_1d(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_idx = lid.x;
    let global_idx = gid.x;

    // Initialize with identity (minimum value for max)
    var local_val = -3.402823466e+38f; // -FLT_MAX
    var local_idx_val = 0u;

    if (global_idx < params.size) {
        local_val = input[global_idx];
        local_idx_val = global_idx;
    }

    shared_values[local_idx] = local_val;
    shared_indices[local_idx] = local_idx_val;
    workgroupBarrier();

    // Tree reduction within workgroup
    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s = s / 2u) {
        if (local_idx < s) {
            let other_val = shared_values[local_idx + s];
            let other_idx = shared_indices[local_idx + s];

            if (other_val > shared_values[local_idx]) {
                shared_values[local_idx] = other_val;
                shared_indices[local_idx] = other_idx;
            }
        }
        workgroupBarrier();
    }

    // Write partial result
    if (local_idx == 0u) {
        output_values[wid.x] = shared_values[0];
        output_indices[wid.x] = shared_indices[0];
    }
}

// ============================================================================
// Argmin - Single reduction (1D array)
// ============================================================================

@compute @workgroup_size(256)
fn argmin_1d(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_idx = lid.x;
    let global_idx = gid.x;

    // Initialize with identity (maximum value for min)
    var local_val = 3.402823466e+38f; // FLT_MAX
    var local_idx_val = 0u;

    if (global_idx < params.size) {
        local_val = input[global_idx];
        local_idx_val = global_idx;
    }

    shared_values[local_idx] = local_val;
    shared_indices[local_idx] = local_idx_val;
    workgroupBarrier();

    // Tree reduction within workgroup
    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s = s / 2u) {
        if (local_idx < s) {
            let other_val = shared_values[local_idx + s];
            let other_idx = shared_indices[local_idx + s];

            if (other_val < shared_values[local_idx]) {
                shared_values[local_idx] = other_val;
                shared_indices[local_idx] = other_idx;
            }
        }
        workgroupBarrier();
    }

    // Write partial result
    if (local_idx == 0u) {
        output_values[wid.x] = shared_values[0];
        output_indices[wid.x] = shared_indices[0];
    }
}

// ============================================================================
// Argmax along axis (multi-dimensional)
// Each workgroup handles one reduction along the specified axis
// ============================================================================

@compute @workgroup_size(256)
fn argmax_axis(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_idx = lid.x;
    let reduction_idx = wid.x; // Which reduction we're performing

    if (reduction_idx >= params.num_reductions) { return; }

    // Calculate base offset for this reduction
    // This depends on how the tensor is laid out
    let base_offset = (reduction_idx / params.stride) * (params.axis_size * params.stride)
                    + (reduction_idx % params.stride);

    // Each thread handles multiple elements if axis_size > workgroup_size
    var local_val = -3.402823466e+38f;
    var local_idx_val = 0u;

    var i = local_idx;
    while (i < params.axis_size) {
        let offset = base_offset + i * params.stride;
        let val = input[offset];

        if (val > local_val) {
            local_val = val;
            local_idx_val = i;
        }
        i = i + WORKGROUP_SIZE;
    }

    shared_values[local_idx] = local_val;
    shared_indices[local_idx] = local_idx_val;
    workgroupBarrier();

    // Tree reduction
    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s = s / 2u) {
        if (local_idx < s) {
            if (shared_values[local_idx + s] > shared_values[local_idx]) {
                shared_values[local_idx] = shared_values[local_idx + s];
                shared_indices[local_idx] = shared_indices[local_idx + s];
            }
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        output_indices[reduction_idx] = shared_indices[0];
        output_values[reduction_idx] = shared_values[0];
    }
}

// ============================================================================
// Argmin along axis (multi-dimensional)
// ============================================================================

@compute @workgroup_size(256)
fn argmin_axis(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_idx = lid.x;
    let reduction_idx = wid.x;

    if (reduction_idx >= params.num_reductions) { return; }

    let base_offset = (reduction_idx / params.stride) * (params.axis_size * params.stride)
                    + (reduction_idx % params.stride);

    var local_val = 3.402823466e+38f;
    var local_idx_val = 0u;

    var i = local_idx;
    while (i < params.axis_size) {
        let offset = base_offset + i * params.stride;
        let val = input[offset];

        if (val < local_val) {
            local_val = val;
            local_idx_val = i;
        }
        i = i + WORKGROUP_SIZE;
    }

    shared_values[local_idx] = local_val;
    shared_indices[local_idx] = local_idx_val;
    workgroupBarrier();

    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s = s / 2u) {
        if (local_idx < s) {
            if (shared_values[local_idx + s] < shared_values[local_idx]) {
                shared_values[local_idx] = shared_values[local_idx + s];
                shared_indices[local_idx] = shared_indices[local_idx + s];
            }
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        output_indices[reduction_idx] = shared_indices[0];
        output_values[reduction_idx] = shared_values[0];
    }
}

// ============================================================================
// Final reduction pass - Reduces partial results from multiple workgroups
// ============================================================================

struct FinalReduceParams {
    num_partials: u32,   // Number of partial results to reduce
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(4) var<uniform> final_params: FinalReduceParams;

@compute @workgroup_size(256)
fn argmax_final(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_idx = lid.x;

    var local_val = -3.402823466e+38f;
    var local_idx_val = 0u;

    // Each thread loads one partial result (if available)
    if (local_idx < final_params.num_partials) {
        local_val = output_values[local_idx];
        local_idx_val = output_indices[local_idx];
    }

    shared_values[local_idx] = local_val;
    shared_indices[local_idx] = local_idx_val;
    workgroupBarrier();

    // Tree reduction
    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s = s / 2u) {
        if (local_idx < s && local_idx + s < final_params.num_partials) {
            if (shared_values[local_idx + s] > shared_values[local_idx]) {
                shared_values[local_idx] = shared_values[local_idx + s];
                shared_indices[local_idx] = shared_indices[local_idx + s];
            }
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        output_indices[0] = shared_indices[0];
        output_values[0] = shared_values[0];
    }
}

@compute @workgroup_size(256)
fn argmin_final(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_idx = lid.x;

    var local_val = 3.402823466e+38f;
    var local_idx_val = 0u;

    if (local_idx < final_params.num_partials) {
        local_val = output_values[local_idx];
        local_idx_val = output_indices[local_idx];
    }

    shared_values[local_idx] = local_val;
    shared_indices[local_idx] = local_idx_val;
    workgroupBarrier();

    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s = s / 2u) {
        if (local_idx < s && local_idx + s < final_params.num_partials) {
            if (shared_values[local_idx + s] < shared_values[local_idx]) {
                shared_values[local_idx] = shared_values[local_idx + s];
                shared_indices[local_idx] = shared_indices[local_idx + s];
            }
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        output_indices[0] = shared_indices[0];
        output_values[0] = shared_values[0];
    }
}

// ============================================================================
// NaN-aware versions (propagate NaN or treat as missing)
// ============================================================================

fn is_nan(x: f32) -> bool {
    return x != x;
}

@compute @workgroup_size(256)
fn argmax_nan_1d(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_idx = lid.x;
    let global_idx = gid.x;

    var local_val = -3.402823466e+38f;
    var local_idx_val = 0u;

    if (global_idx < params.size) {
        let val = input[global_idx];
        // Skip NaN values
        if (!is_nan(val)) {
            local_val = val;
            local_idx_val = global_idx;
        }
    }

    shared_values[local_idx] = local_val;
    shared_indices[local_idx] = local_idx_val;
    workgroupBarrier();

    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s = s / 2u) {
        if (local_idx < s) {
            let other_val = shared_values[local_idx + s];

            if (!is_nan(other_val) && other_val > shared_values[local_idx]) {
                shared_values[local_idx] = other_val;
                shared_indices[local_idx] = shared_indices[local_idx + s];
            }
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        output_values[wid.x] = shared_values[0];
        output_indices[wid.x] = shared_indices[0];
    }
}

@compute @workgroup_size(256)
fn argmin_nan_1d(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_idx = lid.x;
    let global_idx = gid.x;

    var local_val = 3.402823466e+38f;
    var local_idx_val = 0u;

    if (global_idx < params.size) {
        let val = input[global_idx];
        if (!is_nan(val)) {
            local_val = val;
            local_idx_val = global_idx;
        }
    }

    shared_values[local_idx] = local_val;
    shared_indices[local_idx] = local_idx_val;
    workgroupBarrier();

    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s = s / 2u) {
        if (local_idx < s) {
            let other_val = shared_values[local_idx + s];

            if (!is_nan(other_val) && other_val < shared_values[local_idx]) {
                shared_values[local_idx] = other_val;
                shared_indices[local_idx] = shared_indices[local_idx + s];
            }
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        output_values[wid.x] = shared_values[0];
        output_indices[wid.x] = shared_indices[0];
    }
}
