// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Copy Kernels - Memory copy with striding, broadcasting, and type conversion
// Supports contiguous copy, strided copy, gather, scatter, and broadcasting.
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

// ============================================================================
// Parameter Structures
// ============================================================================

struct CopyParams {
    size: u32,           // Number of elements to copy
    src_offset: u32,     // Starting offset in source
    dst_offset: u32,     // Starting offset in destination
    _pad: u32,
}

struct StridedCopyParams {
    size: u32,           // Number of elements
    src_stride: u32,     // Source stride
    dst_stride: u32,     // Destination stride
    src_offset: u32,     // Source offset
    dst_offset: u32,     // Destination offset
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

// For multi-dimensional broadcasting
struct BroadcastParams {
    size: u32,           // Total output elements
    ndim: u32,           // Number of dimensions
    _pad1: u32,
    _pad2: u32,
    // Shape and stride arrays passed via separate bindings
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: CopyParams;

// ============================================================================
// Simple Contiguous Copy
// ============================================================================

@compute @workgroup_size(256)
fn copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    dst[params.dst_offset + idx] = src[params.src_offset + idx];
}

// ============================================================================
// Vectorized Copy (4 elements at a time)
// ============================================================================

@compute @workgroup_size(256)
fn copy_vec4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x * 4u;
    if (idx >= params.size) { return; }

    let src_idx = params.src_offset + idx;
    let dst_idx = params.dst_offset + idx;

    // Copy 4 elements
    let remaining = params.size - idx;
    if (remaining >= 4u) {
        dst[dst_idx] = src[src_idx];
        dst[dst_idx + 1u] = src[src_idx + 1u];
        dst[dst_idx + 2u] = src[src_idx + 2u];
        dst[dst_idx + 3u] = src[src_idx + 3u];
    } else {
        // Handle tail
        for (var i = 0u; i < remaining; i = i + 1u) {
            dst[dst_idx + i] = src[src_idx + i];
        }
    }
}

// ============================================================================
// Strided Copy
// ============================================================================

@group(0) @binding(3) var<uniform> strided_params: StridedCopyParams;

@compute @workgroup_size(256)
fn copy_strided(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= strided_params.size) { return; }

    let src_idx = strided_params.src_offset + idx * strided_params.src_stride;
    let dst_idx = strided_params.dst_offset + idx * strided_params.dst_stride;

    dst[dst_idx] = src[src_idx];
}

// ============================================================================
// Gather - Index-based copy from source
// ============================================================================

@group(0) @binding(4) var<storage, read> indices: array<u32>;

struct GatherParams {
    size: u32,           // Number of elements to gather
    src_size: u32,       // Source array size (for bounds checking)
    default_value: u32,  // Default value for out-of-bounds
    _pad: u32,
}

@group(0) @binding(5) var<uniform> gather_params: GatherParams;

@compute @workgroup_size(256)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= gather_params.size) { return; }

    let src_idx = indices[idx];

    if (src_idx < gather_params.src_size) {
        dst[idx] = src[src_idx];
    } else {
        dst[idx] = gather_params.default_value;
    }
}

// ============================================================================
// Scatter - Index-based copy to destination
// ============================================================================

struct ScatterParams {
    size: u32,           // Number of elements to scatter
    dst_size: u32,       // Destination array size
    mode: u32,           // 0 = overwrite, 1 = add, 2 = max, 3 = min
    _pad: u32,
}

@group(0) @binding(6) var<uniform> scatter_params: ScatterParams;

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= scatter_params.size) { return; }

    let dst_idx = indices[idx];
    if (dst_idx >= scatter_params.dst_size) { return; }

    let value = src[idx];

    if (scatter_params.mode == 0u) {
        // Overwrite
        dst[dst_idx] = value;
    } else if (scatter_params.mode == 1u) {
        // Add (atomic would be better for correctness)
        dst[dst_idx] = dst[dst_idx] + value;
    } else if (scatter_params.mode == 2u) {
        // Max
        dst[dst_idx] = max(dst[dst_idx], value);
    } else {
        // Min
        dst[dst_idx] = min(dst[dst_idx], value);
    }
}

// ============================================================================
// Scatter (float version with atomic-safe modes)
// ============================================================================

@group(0) @binding(7) var<storage, read> src_f32: array<f32>;
@group(0) @binding(8) var<storage, read_write> dst_f32: array<f32>;

@compute @workgroup_size(256)
fn scatter_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= scatter_params.size) { return; }

    let dst_idx = indices[idx];
    if (dst_idx >= scatter_params.dst_size) { return; }

    let value = src_f32[idx];

    if (scatter_params.mode == 0u) {
        dst_f32[dst_idx] = value;
    } else if (scatter_params.mode == 1u) {
        dst_f32[dst_idx] = dst_f32[dst_idx] + value;
    } else if (scatter_params.mode == 2u) {
        dst_f32[dst_idx] = max(dst_f32[dst_idx], value);
    } else {
        dst_f32[dst_idx] = min(dst_f32[dst_idx], value);
    }
}

// ============================================================================
// Broadcast Copy - Expand dimensions
// ============================================================================

struct BroadcastCopyParams {
    size: u32,           // Output size
    src_size: u32,       // Source size (for modulo)
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(9) var<uniform> broadcast_params: BroadcastCopyParams;

// Simple 1D broadcast (repeat source)
@compute @workgroup_size(256)
fn broadcast_1d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= broadcast_params.size) { return; }

    let src_idx = idx % broadcast_params.src_size;
    dst[idx] = src[src_idx];
}

// ============================================================================
// Multi-dimensional Broadcast
// ============================================================================

struct MDParams {
    out_size: u32,
    ndim: u32,
    _pad1: u32,
    _pad2: u32,
}

// Shape/stride arrays for up to 8 dimensions
@group(0) @binding(10) var<storage, read> out_shape: array<u32>;    // Output shape
@group(0) @binding(11) var<storage, read> out_strides: array<u32>; // Output strides
@group(0) @binding(12) var<storage, read> src_shape: array<u32>;   // Source shape (1 = broadcast)
@group(0) @binding(13) var<storage, read> src_strides: array<u32>; // Source strides

@group(0) @binding(14) var<uniform> md_params: MDParams;

@compute @workgroup_size(256)
fn broadcast_nd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= md_params.out_size) { return; }

    // Convert linear index to multi-dimensional coordinates
    var remaining = idx;
    var src_linear_idx = 0u;

    for (var d = 0u; d < md_params.ndim; d = d + 1u) {
        let coord = remaining / out_strides[d];
        remaining = remaining % out_strides[d];

        // Apply broadcasting: if src_shape[d] == 1, use coord 0
        var src_coord = coord;
        if (src_shape[d] == 1u) {
            src_coord = 0u;
        } else {
            src_coord = coord % src_shape[d];
        }

        src_linear_idx = src_linear_idx + src_coord * src_strides[d];
    }

    dst[idx] = src[src_linear_idx];
}

// ============================================================================
// Transpose (2D)
// ============================================================================

struct TransposeParams {
    rows: u32,
    cols: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(15) var<uniform> transpose_params: TransposeParams;

@compute @workgroup_size(16, 16)
fn transpose_2d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;

    if (row >= transpose_params.rows || col >= transpose_params.cols) { return; }

    let src_idx = row * transpose_params.cols + col;
    let dst_idx = col * transpose_params.rows + row;

    dst[dst_idx] = src[src_idx];
}

// ============================================================================
// Tiled Transpose (for better cache utilization)
// ============================================================================

const TILE_SIZE: u32 = 16u;
var<workgroup> tile: array<array<u32, 17>, 16>; // +1 to avoid bank conflicts

@compute @workgroup_size(16, 16)
fn transpose_2d_tiled(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let row = wid.y * TILE_SIZE + lid.y;
    let col = wid.x * TILE_SIZE + lid.x;

    // Load tile from source
    if (row < transpose_params.rows && col < transpose_params.cols) {
        tile[lid.y][lid.x] = src[row * transpose_params.cols + col];
    }

    workgroupBarrier();

    // Write transposed tile to destination
    let dst_row = wid.x * TILE_SIZE + lid.y;
    let dst_col = wid.y * TILE_SIZE + lid.x;

    if (dst_row < transpose_params.cols && dst_col < transpose_params.rows) {
        dst[dst_row * transpose_params.rows + dst_col] = tile[lid.x][lid.y];
    }
}

// ============================================================================
// Fill Operations
// ============================================================================

struct FillParams {
    size: u32,
    value: u32,
    offset: u32,
    _pad: u32,
}

@group(0) @binding(16) var<uniform> fill_params: FillParams;

@compute @workgroup_size(256)
fn fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= fill_params.size) { return; }

    dst[fill_params.offset + idx] = fill_params.value;
}

// ============================================================================
// Repeat/Tile - Repeat array contents
// ============================================================================

struct RepeatParams {
    src_size: u32,       // Size of source array
    repeats: u32,        // Number of times to repeat
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(17) var<uniform> repeat_params: RepeatParams;

@compute @workgroup_size(256)
fn repeat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_size = repeat_params.src_size * repeat_params.repeats;
    if (idx >= total_size) { return; }

    let src_idx = idx % repeat_params.src_size;
    dst[idx] = src[src_idx];
}

// Repeat each element individually
@compute @workgroup_size(256)
fn repeat_interleave(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_size = repeat_params.src_size * repeat_params.repeats;
    if (idx >= total_size) { return; }

    let src_idx = idx / repeat_params.repeats;
    dst[idx] = src[src_idx];
}

// ============================================================================
// Pad Operations
// ============================================================================

struct PadParams {
    src_size: u32,
    pad_before: u32,
    pad_after: u32,
    pad_value: u32,
}

@group(0) @binding(18) var<uniform> pad_params: PadParams;

@compute @workgroup_size(256)
fn pad_1d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_size = pad_params.pad_before + pad_params.src_size + pad_params.pad_after;
    if (idx >= total_size) { return; }

    if (idx < pad_params.pad_before) {
        // Before padding
        dst[idx] = pad_params.pad_value;
    } else if (idx < pad_params.pad_before + pad_params.src_size) {
        // Source data
        dst[idx] = src[idx - pad_params.pad_before];
    } else {
        // After padding
        dst[idx] = pad_params.pad_value;
    }
}

// ============================================================================
// Type Conversion Copies
// ============================================================================

@group(0) @binding(19) var<storage, read> src_i32: array<i32>;
@group(0) @binding(20) var<storage, read_write> dst_i32: array<i32>;

// f32 to i32
@compute @workgroup_size(256)
fn copy_f32_to_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    dst_i32[idx] = i32(src_f32[params.src_offset + idx]);
}

// i32 to f32
@compute @workgroup_size(256)
fn copy_i32_to_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    dst_f32[idx] = f32(src_i32[params.src_offset + idx]);
}

// u32 to f32
@compute @workgroup_size(256)
fn copy_u32_to_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    dst_f32[idx] = f32(src[params.src_offset + idx]);
}

// f32 to u32
@compute @workgroup_size(256)
fn copy_f32_to_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    dst[idx] = u32(max(0.0f, src_f32[params.src_offset + idx]));
}
