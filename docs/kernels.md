# WGSL Kernel Development Guide

Guide for writing efficient WGSL compute shaders for the Lux WebGPU backend.

## Table of Contents

- [WGSL Basics](#wgsl-basics)
- [Kernel Structure](#kernel-structure)
- [Buffer Bindings](#buffer-bindings)
- [Workgroup Sizing](#workgroup-sizing)
- [Memory Access Patterns](#memory-access-patterns)
- [Reduction Patterns](#reduction-patterns)
- [64-bit Emulation](#64-bit-emulation)
- [Cryptographic Kernels](#cryptographic-kernels)
- [Performance Tips](#performance-tips)

---

## WGSL Basics

WGSL (WebGPU Shading Language) is the shader language for WebGPU. Key characteristics:

- **Statically typed**: All types resolved at compile time
- **No recursion**: All control flow must be statically analyzable
- **Limited 64-bit**: No native 64-bit integers or doubles
- **Explicit workgroups**: Programmer specifies thread organization

### Scalar Types

```wgsl
bool        // Boolean
i32         // 32-bit signed integer
u32         // 32-bit unsigned integer
f32         // 32-bit float
f16         // 16-bit float (requires extension)
```

### Vector Types

```wgsl
vec2<f32>, vec3<f32>, vec4<f32>  // Float vectors
vec2<i32>, vec3<i32>, vec4<i32>  // Integer vectors
vec2<u32>, vec3<u32>, vec4<u32>  // Unsigned integer vectors
```

### Matrix Types

```wgsl
mat2x2<f32>, mat3x3<f32>, mat4x4<f32>  // Square matrices
mat2x3<f32>, mat3x4<f32>               // Non-square matrices
```

---

## Kernel Structure

### Basic Template

```wgsl
// Parameter structure (uniform buffer)
struct Params {
    size: u32,
    scale: f32,
    _pad0: u32,  // Pad to 16-byte alignment
    _pad1: u32,
}

// Storage buffers
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// Compute shader entry point
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }  // Bounds check

    output[idx] = input[idx] * params.scale;
}
```

### Multiple Entry Points

Single WGSL file can define multiple kernels:

```wgsl
@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }
    output[gid.x] = a[gid.x] + b[gid.x];
}

@compute @workgroup_size(256)
fn sub(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }
    output[gid.x] = a[gid.x] - b[gid.x];
}

@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }
    output[gid.x] = a[gid.x] * b[gid.x];
}
```

---

## Buffer Bindings

### Address Spaces

| Address Space | Usage | Access |
|---------------|-------|--------|
| `storage` | Large arrays | read or read_write |
| `uniform` | Small parameters | read-only |
| `workgroup` | Shared memory | read_write (within workgroup) |
| `private` | Per-thread | read_write |

### Binding Layout

```wgsl
// Storage buffer (large data)
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Uniform buffer (small, frequently accessed)
@group(0) @binding(2) var<uniform> params: Params;

// Workgroup shared memory
var<workgroup> shared_data: array<f32, 256>;
```

### Struct Alignment

WGSL requires 16-byte alignment for uniform buffers:

```wgsl
struct Params {
    size: u32,          // offset 0
    stride: u32,        // offset 4
    scale: f32,         // offset 8
    _pad: u32,          // offset 12 (pad to 16 bytes)
}

// For vec3, alignment is 16 bytes:
struct Transform {
    offset: vec3<f32>,  // offset 0, size 12
    _pad: f32,          // offset 12 (pad to 16)
    scale: vec3<f32>,   // offset 16
    _pad2: f32,         // offset 28 (pad to 32)
}
```

---

## Workgroup Sizing

### Workgroup Size Declaration

```wgsl
@compute @workgroup_size(256)           // 1D: 256 threads
@compute @workgroup_size(16, 16)        // 2D: 16x16 = 256 threads
@compute @workgroup_size(8, 8, 4)       // 3D: 8x8x4 = 256 threads
```

### Thread Indexing

```wgsl
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    let global_idx = global_id.x;  // Unique across all threads
    let local_idx = local_id.x;    // Unique within workgroup (0-255)
    let wg_idx = wg_id.x;          // Workgroup index
}
```

### Dispatch Calculation

```c
// Host-side dispatch calculation
uint32_t num_elements = 1000000;
uint32_t workgroup_size = 256;
uint32_t num_workgroups = (num_elements + workgroup_size - 1) / workgroup_size;

// Dispatch
wgpuComputePassEncoderDispatchWorkgroups(pass, num_workgroups, 1, 1);
```

---

## Memory Access Patterns

### Coalesced Access (Good)

Adjacent threads access adjacent memory:

```wgsl
@compute @workgroup_size(256)
fn coalesced(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    output[idx] = input[idx] * 2.0;  // Good: thread 0 -> [0], thread 1 -> [1], ...
}
```

### Strided Access (Bad)

Threads access non-contiguous memory:

```wgsl
@compute @workgroup_size(256)
fn strided(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x * stride;  // Bad: gaps between accesses
    output[idx] = input[idx] * 2.0;
}
```

### Vectorized Access

Process 4 elements per thread for better throughput:

```wgsl
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(256)
fn vectorized(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size / 4u) { return; }

    let v = input[idx];
    output[idx] = v * 2.0;  // Processes 4 floats at once
}
```

---

## Reduction Patterns

### Two-Phase Reduction

Efficient parallel sum using workgroup shared memory:

```wgsl
const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce_sum(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let global_idx = gid.x;
    let local_idx = lid.x;

    // Load into shared memory
    if (global_idx < params.size) {
        shared_data[local_idx] = input[global_idx];
    } else {
        shared_data[local_idx] = 0.0;
    }

    workgroupBarrier();

    // Parallel reduction in shared memory
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
        if (local_idx < stride) {
            shared_data[local_idx] += shared_data[local_idx + stride];
        }
        workgroupBarrier();
    }

    // Write result (one thread per workgroup)
    if (local_idx == 0u) {
        output[wg_id.x] = shared_data[0];
    }
}
```

### Max Reduction

```wgsl
@compute @workgroup_size(256)
fn reduce_max(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let global_idx = gid.x;
    let local_idx = lid.x;

    if (global_idx < params.size) {
        shared_data[local_idx] = input[global_idx];
    } else {
        shared_data[local_idx] = -3.402823e+38;  // -FLT_MAX
    }

    workgroupBarrier();

    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
        if (local_idx < stride) {
            shared_data[local_idx] = max(shared_data[local_idx], shared_data[local_idx + stride]);
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        output[wg_id.x] = shared_data[0];
    }
}
```

---

## 64-bit Emulation

WGSL lacks native 64-bit integers. Emulate using pairs of u32:

### U64 Structure

```wgsl
struct U64 {
    lo: u32,  // Lower 32 bits
    hi: u32,  // Upper 32 bits
}

fn u64_zero() -> U64 { return U64(0u, 0u); }
fn u64_from_u32(v: u32) -> U64 { return U64(v, 0u); }

fn u64_eq(a: U64, b: U64) -> bool {
    return a.lo == b.lo && a.hi == b.hi;
}

fn u64_lt(a: U64, b: U64) -> bool {
    if (a.hi != b.hi) { return a.hi < b.hi; }
    return a.lo < b.lo;
}
```

### 64-bit Addition

```wgsl
fn u64_add(a: U64, b: U64) -> U64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    let hi = a.hi + b.hi + carry;
    return U64(lo, hi);
}
```

### 64-bit Multiplication

```wgsl
// 32x32 -> 64 multiply
fn mul32_wide(a: u32, b: u32) -> U64 {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;

    let mid = p1 + p2;
    let mid_carry = select(0u, 0x10000u, mid < p1);

    let lo = p0 + ((mid & 0xFFFFu) << 16u);
    let lo_carry = select(0u, 1u, lo < p0);
    let hi = p3 + (mid >> 16u) + mid_carry + lo_carry;

    return U64(lo, hi);
}
```

---

## Cryptographic Kernels

### Modular Arithmetic

Barrett reduction for constant modulus:

```wgsl
const Q: i32 = 8380417;  // Dilithium prime
const BARRETT_V: i32 = 8396807;

fn barrett_reduce(a: i32) -> i32 {
    let t = (BARRETT_V * a) >> 23;
    return a - t * Q;
}

fn full_reduce(a: i32) -> i32 {
    var t = barrett_reduce(a);
    t = t + select(0, Q, t < 0);
    t = t - Q;
    t = t + select(0, Q, t < 0);
    return t;
}
```

### Field Arithmetic (Goldilocks)

```wgsl
// p = 2^64 - 2^32 + 1
const GL_P: U64 = U64(0x00000001u, 0xFFFFFFFFu);

fn gl_add(a: U64, b: U64) -> U64 {
    let sum = u64_add(a, b);
    if (u64_ge(sum, GL_P)) {
        return u64_sub(sum, GL_P);
    }
    return sum;
}

fn gl_mul(a: U64, b: U64) -> U64 {
    let full = u64_mul_full(a, b);  // 128-bit result
    return gl_reduce(full[0], full[1]);
}
```

### Poseidon S-box

```wgsl
// x^7 S-box for Poseidon
fn sbox(x: U64) -> U64 {
    let x2 = gl_mul(x, x);
    let x4 = gl_mul(x2, x2);
    let x3 = gl_mul(x2, x);
    return gl_mul(x4, x3);
}
```

---

## Performance Tips

### 1. Avoid Divergence

Bad:
```wgsl
if (gid.x % 2u == 0u) {
    // Half the threads idle
    output[gid.x] = expensive_operation(input[gid.x]);
}
```

Good:
```wgsl
// All threads do same work
let val = input[gid.x];
output[gid.x] = expensive_operation(val);
```

### 2. Minimize Barriers

Barriers synchronize all threads in workgroup. Use sparingly:

```wgsl
// Only barrier when necessary
workgroupBarrier();  // Wait for all threads
```

### 3. Use Constants

```wgsl
const WORKGROUP_SIZE: u32 = 256u;
const TWO_PI: f32 = 6.28318530718;
```

### 4. Prefer Built-in Functions

```wgsl
// Use built-ins instead of manual implementations
let v = sqrt(x);        // Built-in
let v = exp(x);         // Built-in
let v = max(a, b);      // Built-in
let v = clamp(x, 0.0, 1.0);
```

### 5. Batch Small Operations

Process multiple elements per thread:

```wgsl
@compute @workgroup_size(256)
fn process_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = gid.x * 4u;
    if (base >= params.size) { return; }

    // Process 4 elements per thread
    for (var i = 0u; i < 4u && base + i < params.size; i++) {
        output[base + i] = process(input[base + i]);
    }
}
```

### 6. Memory Layout

Structure arrays for coalesced access:

```wgsl
// AoS (Array of Structures) - poor coalescing
struct Point { x: f32, y: f32, z: f32, w: f32 }
var<storage> points: array<Point>;

// SoA (Structure of Arrays) - good coalescing
var<storage> x: array<f32>;
var<storage> y: array<f32>;
var<storage> z: array<f32>;
```

### 7. Profiling

Use WebGPU timestamp queries for profiling:

```c
// Create timestamp query set
WGPUQuerySetDescriptor desc = {
    .type = WGPUQueryType_Timestamp,
    .count = 2,
};
WGPUQuerySet querySet = wgpuDeviceCreateQuerySet(device, &desc);

// Record timestamps
wgpuComputePassEncoderWriteTimestamp(pass, querySet, 0);
// ... kernel dispatch ...
wgpuComputePassEncoderWriteTimestamp(pass, querySet, 1);
```
