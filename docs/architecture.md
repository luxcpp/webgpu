# Architecture

Overview of the Lux WebGPU backend architecture.

## Table of Contents

- [System Overview](#system-overview)
- [Backend Plugin System](#backend-plugin-system)
- [Kernel Management](#kernel-management)
- [Memory Model](#memory-model)
- [Execution Model](#execution-model)
- [WebGPU Abstraction](#webgpu-abstraction)

---

## System Overview

```
+------------------+
|   Application    |
+------------------+
         |
         v
+------------------+
|   lux/gpu.h      |  Unified C API
+------------------+
         |
         v
+------------------+
| Backend Dispatch |  Runtime backend selection
+------------------+
    |    |    |
    v    v    v
+------+------+--------+
|Metal |CUDA  |WebGPU  |  Platform backends
+------+------+--------+
    |    |    |
    v    v    v
+------+------+--------+
|MPS   |cuBLAS|Dawn/   |  Native GPU APIs
|      |      |wgpu    |
+------+------+--------+
```

### Design Principles

1. **Unified API**: Single C interface across all backends
2. **Runtime Selection**: Backend chosen at runtime based on availability
3. **Plugin Architecture**: Backends as dynamically loaded shared libraries
4. **Kernel Embedding**: WGSL/Metal/CUDA sources embedded at build time
5. **Zero-Copy Where Possible**: Minimize host-device transfers

---

## Backend Plugin System

### Plugin Interface

Each backend implements the `lux_gpu_backend_vtbl` virtual table:

```c
typedef struct lux_gpu_backend_vtbl {
    // Lifecycle
    LuxBackendContext* (*create_context)(int device_index);
    void (*destroy_context)(LuxBackendContext* ctx);

    // Device info
    LuxBackendError (*get_device_count)(int* count);
    LuxBackendError (*get_device_info)(LuxBackendContext* ctx, LuxBackendDeviceInfo* info);

    // Buffer management
    LuxBackendBuffer* (*buffer_alloc)(LuxBackendContext* ctx, size_t bytes);
    void (*buffer_free)(LuxBackendContext* ctx, LuxBackendBuffer* buf);
    LuxBackendError (*buffer_copy_to_host)(LuxBackendContext* ctx, ...);
    LuxBackendError (*buffer_copy_from_host)(LuxBackendContext* ctx, ...);

    // Kernel dispatch
    LuxBackendError (*kernel_dispatch)(LuxBackendContext* ctx, ...);

    // Operations
    LuxBackendError (*op_add_f32)(LuxBackendContext* ctx, ...);
    LuxBackendError (*op_matmul_f32)(LuxBackendContext* ctx, ...);
    // ... etc
} lux_gpu_backend_vtbl;
```

### Plugin Loading

```c
// Plugin entry point (each backend exports this)
bool lux_gpu_backend_init(lux_gpu_backend_desc* out) {
    out->abi_version = LUX_GPU_BACKEND_ABI_VERSION;
    out->backend_name = "webgpu";
    out->backend_version = "0.2.0";
    out->capabilities = LUX_CAP_TENSOR_OPS | LUX_CAP_MSM | LUX_CAP_FHE;
    out->vtbl = &g_webgpu_vtbl;
    return true;
}

// Core library loads backends at runtime
void* handle = dlopen("libluxgpu_backend_webgpu.dylib", RTLD_NOW);
lux_gpu_backend_init_fn init = dlsym(handle, "lux_gpu_backend_init");
lux_gpu_backend_desc desc;
if (init(&desc)) {
    // Backend available
}
```

### Capability Flags

```c
#define LUX_CAP_TENSOR_OPS      (1 << 0)   // add, sub, mul, div
#define LUX_CAP_MATMUL          (1 << 1)   // Matrix multiplication
#define LUX_CAP_NTT             (1 << 2)   // Number theoretic transform
#define LUX_CAP_MSM             (1 << 3)   // Multi-scalar multiplication
#define LUX_CAP_CUSTOM_KERNELS  (1 << 4)   // Custom kernel loading
#define LUX_CAP_UNIFIED_MEMORY  (1 << 5)   // Unified memory support
#define LUX_CAP_FHE             (1 << 6)   // FHE operations
#define LUX_CAP_TFHE            (1 << 7)   // TFHE bootstrap/keyswitch
#define LUX_CAP_BLS12_381       (1 << 12)  // BLS12-381 curve
#define LUX_CAP_BN254           (1 << 13)  // BN254 curve
#define LUX_CAP_KZG             (1 << 14)  // KZG commitments
#define LUX_CAP_POSEIDON2       (1 << 15)  // Poseidon2 hash
```

---

## Kernel Management

### Kernel Registry

Kernels organized by domain and embedded at build time:

```
kernels/wgsl/
  gpu/           # ML/tensor operations
  crypto/        # Cryptographic primitives
  zk/            # Zero-knowledge primitives
  fhe/           # Fully homomorphic encryption
```

CMake generates embedded header:

```c
// Auto-generated wgsl_kernels_embedded.h
namespace lux::gpu::wgsl {
constexpr const char* k_binary = "...WGSL source...";
constexpr const char* k_reduce = "...WGSL source...";
constexpr const char* k_poseidon_goldilocks = "...WGSL source...";
}
```

### Kernel Cache

Compiled pipelines cached by variant key:

```c
typedef struct {
    const char* name;     // Kernel name
    uint32_t dtype;       // Data type variant
    uint32_t size_hint;   // Size optimization hint
    uint32_t flags;       // Additional flags
} LuxKernelVariant;

// Cache lookup
LuxKernel* kernel = lux_kernel_cache_get(cache, &variant);
if (!kernel) {
    kernel = compile_kernel(device, source, entry_point);
    lux_kernel_cache_put(cache, &variant, kernel);
}
```

### Pipeline Creation (WebGPU)

```
WGSL Source
    |
    v
+------------------+
| Shader Compile   |  wgpuDeviceCreateShaderModule
+------------------+
    |
    v
+------------------+
| Pipeline Create  |  wgpuDeviceCreateComputePipeline
+------------------+
    |
    v
+------------------+
| Bind Group       |  wgpuDeviceCreateBindGroup
| Layout           |
+------------------+
    |
    v
+------------------+
| Cached Pipeline  |  Ready for dispatch
+------------------+
```

---

## Memory Model

### Buffer Types

```c
// Storage buffer (large arrays)
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Uniform buffer (small parameters)
@group(0) @binding(2) var<uniform> params: Params;

// Workgroup shared memory (per-workgroup)
var<workgroup> shared: array<f32, 256>;
```

### Memory Hierarchy

```
+------------------+
| Host Memory      |  CPU RAM
+------------------+
         |
    PCIe/USB4
         |
         v
+------------------+
| Device Memory    |  GPU VRAM
+------------------+
         |
         v
+------------------+
| L2 Cache         |  Shared across CUs
+------------------+
         |
         v
+------------------+
| Workgroup Memory |  Per-workgroup shared
+------------------+
         |
         v
+------------------+
| Registers        |  Per-thread private
+------------------+
```

### Unified Memory (Apple Silicon)

On Apple Silicon, CPU and GPU share memory:

```c
// Check for unified memory
LuxDeviceInfo info;
lux_gpu_device_info(gpu, &info);
if (info.is_unified_memory) {
    // Can use buffer_get_host_ptr for zero-copy
    void* ptr = vtbl->buffer_get_host_ptr(ctx, buf);
}
```

### Buffer Lifecycle

```c
// Allocate
LuxBackendBuffer* buf = vtbl->buffer_alloc(ctx, size_bytes);

// Upload data
vtbl->buffer_copy_from_host(ctx, buf, host_data, size_bytes);

// Use in kernels
LuxBackendBuffer* buffers[] = { input_buf, output_buf, params_buf };
vtbl->kernel_dispatch(ctx, kernel, grid_x, 1, 1, 256, 1, 1, buffers, 3);

// Download result
vtbl->buffer_copy_to_host(ctx, buf, host_data, size_bytes);

// Free
vtbl->buffer_free(ctx, buf);
```

---

## Execution Model

### Command Encoding

WebGPU uses command buffers for batching:

```
+------------------+
| Command Encoder  |  wgpuDeviceCreateCommandEncoder
+------------------+
         |
         v
+------------------+
| Compute Pass     |  wgpuCommandEncoderBeginComputePass
+------------------+
         |
    +----+----+
    |         |
    v         v
+-------+ +-------+
|SetPipe| |SetBind|  Set pipeline and bind group
+-------+ +-------+
    |         |
    +----+----+
         |
         v
+------------------+
| Dispatch         |  wgpuComputePassEncoderDispatchWorkgroups
+------------------+
         |
         v
+------------------+
| End Pass         |  wgpuComputePassEncoderEnd
+------------------+
         |
         v
+------------------+
| Finish           |  wgpuCommandEncoderFinish
+------------------+
         |
         v
+------------------+
| Queue Submit     |  wgpuQueueSubmit
+------------------+
```

### Synchronization

```c
// Async callback (non-blocking)
wgpuQueueOnSubmittedWorkDone(queue, callback, userdata);

// Blocking sync
lux_gpu_sync(gpu);  // Waits for all work to complete
```

### Stream/Event Model

```c
// Create stream for async operations
LuxStream* stream = lux_stream_create(gpu);

// Submit work to stream
// (operations enqueued, not blocking)

// Wait for stream
lux_stream_sync(stream);

// Events for timing
LuxEvent* start = lux_event_create(gpu);
LuxEvent* end = lux_event_create(gpu);

lux_event_record(start, stream);
// ... kernel dispatch ...
lux_event_record(end, stream);

lux_stream_sync(stream);
float elapsed_ms = lux_event_elapsed(start, end);
```

---

## WebGPU Abstraction

### Dawn vs wgpu-native

The backend supports both WebGPU implementations:

```cpp
#if defined(USE_DAWN_API)
    #include <webgpu/webgpu_cpp.h>
    // Dawn C++ API
    wgpu::Device device;
    wgpu::ComputePipeline pipeline;
#elif defined(USE_WGPU_API)
    #include <wgpu.h>
    // wgpu-native C API
    WGPUDevice device;
    WGPUComputePipeline pipeline;
#endif
```

### Adapter Selection

```cpp
// Request adapter with preferences
WGPURequestAdapterOptions options = {
    .powerPreference = WGPUPowerPreference_HighPerformance,
};
wgpuInstanceRequestAdapter(instance, &options, callback, userdata);

// Create device from adapter
WGPUDeviceDescriptor desc = {};
wgpuAdapterRequestDevice(adapter, &desc, callback, userdata);
```

### Error Handling

```cpp
// Device error callback
wgpuDeviceSetUncapturedErrorCallback(device,
    [](WGPUErrorType type, const char* message, void* userdata) {
        fprintf(stderr, "WebGPU error: %s\n", message);
    },
    nullptr
);

// Device lost callback
wgpuDeviceSetDeviceLostCallback(device,
    [](WGPUDeviceLostReason reason, const char* message, void* userdata) {
        fprintf(stderr, "Device lost: %s\n", message);
    },
    nullptr
);
```

### Limits and Features

```cpp
// Query limits
WGPUSupportedLimits limits = {};
wgpuDeviceGetLimits(device, &limits);

// Key limits:
limits.limits.maxComputeWorkgroupSizeX;      // Usually 256
limits.limits.maxComputeWorkgroupsPerDimension;  // Usually 65535
limits.limits.maxStorageBufferBindingSize;   // Usually 128MB+
limits.limits.maxUniformBufferBindingSize;   // Usually 64KB

// Check features
bool has_f16 = wgpuDeviceHasFeature(device, WGPUFeatureName_ShaderF16);
```

---

## File Organization

```
webgpu/
  CMakeLists.txt           # Build configuration
  Doxyfile                 # Documentation config

  include/
    lux/
      gpu.h                # Public C API
      gpu/
        backend_plugin.h   # Backend vtable interface
        kernel_loader.h    # Kernel management
        crypto.h           # Crypto operations API

  src/
    webgpu_plugin.cpp      # Backend implementation
    webgpu_kernel_loader.cpp

  kernels/wgsl/
    gpu/                   # ML/tensor kernels
    crypto/                # Crypto kernels
    zk/                    # ZK proof kernels
    fhe/                   # FHE kernels

  tests/
    test_binary_ops.cpp
    test_crypto_kernels.cpp
    test_reduction_kernels.cpp
    test_wgsl_syntax.cpp

  docs/
    api.md                 # API reference
    kernels.md             # WGSL guide
    architecture.md        # This document
```
