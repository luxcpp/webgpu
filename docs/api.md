# API Reference

Complete reference for the Lux WebGPU backend C API.

## Table of Contents

- [GPU Context](#gpu-context)
- [Backend Query](#backend-query)
- [Tensor Operations](#tensor-operations)
- [Cryptographic Operations](#cryptographic-operations)
- [FHE Operations](#fhe-operations)
- [Kernel Loading](#kernel-loading)
- [Error Handling](#error-handling)

---

## GPU Context

### Types

```c
typedef struct LuxGPU LuxGPU;

typedef enum {
    LUX_BACKEND_AUTO  = 0,  // Auto-detect best backend
    LUX_BACKEND_CPU   = 1,  // CPU with SIMD
    LUX_BACKEND_METAL = 2,  // Apple Metal
    LUX_BACKEND_CUDA  = 3,  // NVIDIA CUDA
    LUX_BACKEND_DAWN  = 4,  // WebGPU via Dawn
} LuxBackend;

typedef struct {
    LuxBackend backend;
    int index;
    const char* name;
    const char* vendor;
    uint64_t memory_total;
    uint64_t memory_available;
    bool is_discrete;
    bool is_unified_memory;
    int compute_units;
    int max_workgroup_size;
} LuxDeviceInfo;
```

### Functions

#### lux_gpu_create

```c
LuxGPU* lux_gpu_create(void);
```

Create GPU context with auto-detected backend.

**Returns**: Pointer to GPU context, or NULL on failure.

#### lux_gpu_create_with_backend

```c
LuxGPU* lux_gpu_create_with_backend(LuxBackend backend);
```

Create GPU context with specific backend.

**Parameters**:
- `backend`: Target backend (LUX_BACKEND_DAWN for WebGPU)

**Returns**: Pointer to GPU context, or NULL if backend unavailable.

#### lux_gpu_create_with_device

```c
LuxGPU* lux_gpu_create_with_device(LuxBackend backend, int device_index);
```

Create GPU context on specific device.

**Parameters**:
- `backend`: Target backend
- `device_index`: Device index (0-based)

**Returns**: Pointer to GPU context, or NULL on failure.

#### lux_gpu_destroy

```c
void lux_gpu_destroy(LuxGPU* gpu);
```

Destroy GPU context and release resources.

#### lux_gpu_backend

```c
LuxBackend lux_gpu_backend(LuxGPU* gpu);
```

Get active backend type.

#### lux_gpu_backend_name

```c
const char* lux_gpu_backend_name(LuxGPU* gpu);
```

Get backend name string (e.g., "webgpu", "metal").

#### lux_gpu_device_info

```c
LuxError lux_gpu_device_info(LuxGPU* gpu, LuxDeviceInfo* info);
```

Query device capabilities.

#### lux_gpu_sync

```c
LuxError lux_gpu_sync(LuxGPU* gpu);
```

Wait for all pending operations to complete.

#### lux_gpu_error

```c
const char* lux_gpu_error(LuxGPU* gpu);
```

Get last error message string.

---

## Backend Query

### Functions

#### lux_backend_count

```c
int lux_backend_count(void);
```

Get number of available backends.

#### lux_backend_available

```c
bool lux_backend_available(LuxBackend backend);
```

Check if specific backend is available.

#### lux_backend_name

```c
const char* lux_backend_name(LuxBackend backend);
```

Get backend name by type.

#### lux_device_count

```c
int lux_device_count(LuxBackend backend);
```

Get number of devices for backend.

#### lux_device_info

```c
LuxError lux_device_info(LuxBackend backend, int index, LuxDeviceInfo* info);
```

Get device info by backend and index.

---

## Tensor Operations

### Types

```c
typedef struct LuxTensor LuxTensor;

typedef enum {
    LUX_FLOAT32  = 0,
    LUX_FLOAT16  = 1,
    LUX_BFLOAT16 = 2,
    LUX_INT32    = 3,
    LUX_INT64    = 4,
    LUX_UINT32   = 5,
    LUX_UINT64   = 6,
    LUX_BOOL     = 7,
} LuxDtype;
```

### Creation

#### lux_tensor_zeros

```c
LuxTensor* lux_tensor_zeros(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype);
```

Create tensor filled with zeros.

**Parameters**:
- `gpu`: GPU context
- `shape`: Array of dimension sizes
- `ndim`: Number of dimensions
- `dtype`: Data type

#### lux_tensor_ones

```c
LuxTensor* lux_tensor_ones(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype);
```

Create tensor filled with ones.

#### lux_tensor_full

```c
LuxTensor* lux_tensor_full(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype, double value);
```

Create tensor filled with specified value.

#### lux_tensor_from_data

```c
LuxTensor* lux_tensor_from_data(LuxGPU* gpu, const void* data, const int64_t* shape, int ndim, LuxDtype dtype);
```

Create tensor from host data.

#### lux_tensor_destroy

```c
void lux_tensor_destroy(LuxTensor* tensor);
```

Destroy tensor and free memory.

### Properties

```c
int lux_tensor_ndim(LuxTensor* tensor);
int64_t lux_tensor_shape(LuxTensor* tensor, int dim);
int64_t lux_tensor_size(LuxTensor* tensor);
LuxDtype lux_tensor_dtype(LuxTensor* tensor);
```

### Data Transfer

#### lux_tensor_to_host

```c
LuxError lux_tensor_to_host(LuxTensor* tensor, void* data, size_t size);
```

Copy tensor data to host memory.

### Arithmetic Operations

```c
LuxTensor* lux_tensor_add(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
LuxTensor* lux_tensor_sub(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
LuxTensor* lux_tensor_mul(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
LuxTensor* lux_tensor_div(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
LuxTensor* lux_tensor_matmul(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
```

All operations support broadcasting.

### Unary Operations

```c
LuxTensor* lux_tensor_neg(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_exp(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_log(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_sqrt(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_abs(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_tanh(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_sigmoid(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_relu(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_gelu(LuxGPU* gpu, LuxTensor* t);
```

### Reductions

```c
// Full tensor reductions (returns scalar)
float lux_tensor_reduce_sum(LuxGPU* gpu, LuxTensor* t);
float lux_tensor_reduce_max(LuxGPU* gpu, LuxTensor* t);
float lux_tensor_reduce_min(LuxGPU* gpu, LuxTensor* t);
float lux_tensor_reduce_mean(LuxGPU* gpu, LuxTensor* t);

// Axis reductions (returns tensor)
LuxTensor* lux_tensor_sum(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes);
LuxTensor* lux_tensor_mean(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes);
LuxTensor* lux_tensor_max(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes);
LuxTensor* lux_tensor_min(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes);
```

### Normalization

```c
LuxTensor* lux_tensor_softmax(LuxGPU* gpu, LuxTensor* t, int axis);
LuxTensor* lux_tensor_log_softmax(LuxGPU* gpu, LuxTensor* t, int axis);
LuxTensor* lux_tensor_layer_norm(LuxGPU* gpu, LuxTensor* t, LuxTensor* gamma, LuxTensor* beta, float eps);
LuxTensor* lux_tensor_rms_norm(LuxGPU* gpu, LuxTensor* t, LuxTensor* weight, float eps);
```

---

## Cryptographic Operations

### Types

```c
typedef enum {
    LUX_CURVE_BLS12_381 = 0,
    LUX_CURVE_BN254     = 1,
    LUX_CURVE_SECP256K1 = 2,
    LUX_CURVE_ED25519   = 3,
} LuxCurve;
```

### Hash Functions

#### lux_poseidon2_hash

```c
LuxError lux_poseidon2_hash(
    LuxGPU* gpu,
    const uint64_t* inputs,   // [num_hashes * rate]
    uint64_t* outputs,        // [num_hashes]
    size_t rate,              // Poseidon rate parameter
    size_t num_hashes
);
```

Batch Poseidon2 hash (ZK-friendly algebraic hash).

#### lux_blake3_hash

```c
LuxError lux_blake3_hash(
    LuxGPU* gpu,
    const uint8_t* inputs,
    uint8_t* outputs,         // [num_hashes * 32]
    const size_t* input_lens,
    size_t num_hashes
);
```

Batch BLAKE3 hash.

### Multi-Scalar Multiplication

#### lux_msm

```c
LuxError lux_msm(
    LuxGPU* gpu,
    const void* scalars,      // Scalar field elements
    const void* points,       // Affine curve points
    void* result,             // Single output point (projective)
    size_t count,
    LuxCurve curve
);
```

Compute sum of scalar[i] * point[i] using Pippenger's algorithm.

### BLS12-381 Operations

```c
LuxError lux_bls12_381_add(LuxGPU* gpu, const void* a, const void* b, void* out, size_t count, bool is_g2);
LuxError lux_bls12_381_mul(LuxGPU* gpu, const void* points, const void* scalars, void* out, size_t count, bool is_g2);
LuxError lux_bls12_381_pairing(LuxGPU* gpu, const void* g1_points, const void* g2_points, void* out, size_t count);
```

### BN254 Operations

```c
LuxError lux_bn254_add(LuxGPU* gpu, const void* a, const void* b, void* out, size_t count, bool is_g2);
LuxError lux_bn254_mul(LuxGPU* gpu, const void* points, const void* scalars, void* out, size_t count, bool is_g2);
```

### KZG Polynomial Commitments

#### lux_kzg_commit

```c
LuxError lux_kzg_commit(
    LuxGPU* gpu,
    const void* coeffs,       // Polynomial coefficients
    const void* srs,          // SRS G1 points
    void* commitment,         // Output commitment
    size_t degree,
    LuxCurve curve
);
```

Commit to polynomial: C = sum(coeffs[i] * srs[i]).

#### lux_kzg_open

```c
LuxError lux_kzg_open(
    LuxGPU* gpu,
    const void* coeffs,
    const void* srs,
    const void* point,        // Evaluation point
    void* proof,
    size_t degree,
    LuxCurve curve
);
```

Generate opening proof at evaluation point.

#### lux_kzg_verify

```c
LuxError lux_kzg_verify(
    LuxGPU* gpu,
    const void* commitment,
    const void* proof,
    const void* point,
    const void* value,        // Claimed evaluation
    const void* srs_g2,
    bool* result,
    LuxCurve curve
);
```

Verify KZG opening proof.

### BLS Signatures

```c
LuxError lux_bls_verify(LuxGPU* gpu, const uint8_t* sig, size_t sig_len,
                        const uint8_t* msg, size_t msg_len,
                        const uint8_t* pubkey, size_t pubkey_len, bool* result);

LuxError lux_bls_verify_batch(LuxGPU* gpu,
                              const uint8_t* const* sigs, const size_t* sig_lens,
                              const uint8_t* const* msgs, const size_t* msg_lens,
                              const uint8_t* const* pubkeys, const size_t* pubkey_lens,
                              int count, bool* results);

LuxError lux_bls_aggregate(LuxGPU* gpu,
                           const uint8_t* const* sigs, const size_t* sig_lens,
                           int count, uint8_t* out, size_t* out_len);
```

---

## FHE Operations

### NTT (Number Theoretic Transform)

```c
LuxError lux_ntt_forward(LuxGPU* gpu, uint64_t* data, size_t n, uint64_t modulus);
LuxError lux_ntt_inverse(LuxGPU* gpu, uint64_t* data, size_t n, uint64_t modulus);
LuxError lux_ntt_batch(LuxGPU* gpu, uint64_t** polys, size_t count, size_t n, uint64_t modulus);
```

In-place NTT transform for polynomial arithmetic.

### Polynomial Arithmetic

#### lux_poly_mul

```c
LuxError lux_poly_mul(
    LuxGPU* gpu,
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* result,
    size_t n,
    uint64_t modulus
);
```

Polynomial multiplication via NTT: result = a * b mod (X^n + 1) mod modulus.

### TFHE Operations

#### lux_tfhe_bootstrap

```c
LuxError lux_tfhe_bootstrap(
    LuxGPU* gpu,
    const uint64_t* lwe_in,    // Input LWE ciphertext [n_lwe + 1]
    uint64_t* lwe_out,         // Output LWE ciphertext [N + 1]
    const uint64_t* bsk,       // Bootstrapping key
    const uint64_t* test_poly, // Test polynomial (LUT)
    uint32_t n_lwe,            // Input LWE dimension
    uint32_t N,                // GLWE polynomial degree
    uint32_t k,                // GLWE dimension
    uint32_t l,                // Decomposition levels
    uint64_t q                 // Modulus
);
```

TFHE programmable bootstrap: evaluates lookup table on encrypted input.

#### lux_tfhe_keyswitch

```c
LuxError lux_tfhe_keyswitch(
    LuxGPU* gpu,
    const uint64_t* lwe_in,    // Input LWE [n_in + 1]
    uint64_t* lwe_out,         // Output LWE [n_out + 1]
    const uint64_t* ksk,       // Key switching key
    uint32_t n_in,
    uint32_t n_out,
    uint32_t l,                // Decomposition levels
    uint32_t base_log,
    uint64_t q
);
```

Change LWE encryption key.

#### lux_blind_rotate

```c
LuxError lux_blind_rotate(
    LuxGPU* gpu,
    uint64_t* acc,             // Accumulator GLWE [(k+1) * N]
    const uint64_t* bsk,       // Bootstrapping key
    const uint64_t* lwe_a,     // LWE 'a' coefficients [n_lwe]
    uint32_t n_lwe,
    uint32_t N,
    uint32_t k,
    uint32_t l,
    uint64_t q
);
```

Blind rotation: rotate polynomial accumulator by encrypted amount.

---

## Kernel Loading

### Types

```c
typedef struct LuxKernel LuxKernel;
typedef struct LuxKernelCache LuxKernelCache;

typedef enum {
    LUX_KERNEL_SOURCE_EMBEDDED = 0,
    LUX_KERNEL_SOURCE_FILE     = 1,
    LUX_KERNEL_SOURCE_BINARY   = 2,
} LuxKernelSourceType;

typedef enum {
    LUX_KERNEL_LANG_METAL = 0,
    LUX_KERNEL_LANG_CUDA  = 1,
    LUX_KERNEL_LANG_PTX   = 2,
    LUX_KERNEL_LANG_WGSL  = 3,
    LUX_KERNEL_LANG_SPIRV = 4,
} LuxKernelLanguage;
```

### WebGPU Kernel Functions

#### lux_wgpu_kernel_compile

```c
LuxKernel* lux_wgpu_kernel_compile(
    void* device,              // WGPUDevice handle
    const char* wgsl_source,
    size_t source_len,         // 0 for null-terminated
    const char* entry_point
);
```

Compile WGSL shader and create compute pipeline.

#### lux_wgpu_kernel_load_file

```c
LuxKernel* lux_wgpu_kernel_load_file(
    void* device,
    const char* path,
    const char* entry_point
);
```

Load and compile WGSL from file.

#### lux_wgpu_kernel_destroy

```c
void lux_wgpu_kernel_destroy(LuxKernel* kernel);
```

Release kernel resources.

#### lux_wgpu_kernel_pipeline

```c
void* lux_wgpu_kernel_pipeline(LuxKernel* kernel);
```

Get WGPUComputePipeline for dispatch.

#### lux_wgpu_kernel_bind_group_layout

```c
void* lux_wgpu_kernel_bind_group_layout(LuxKernel* kernel);
```

Get bind group layout for creating bind groups.

#### lux_wgpu_kernel_workgroup_size

```c
void lux_wgpu_kernel_workgroup_size(LuxKernel* kernel, uint32_t* x, uint32_t* y, uint32_t* z);
```

Get kernel's declared workgroup size.

### Kernel Cache

```c
LuxKernelCache* lux_kernel_cache_create(void);
void lux_kernel_cache_destroy(LuxKernelCache* cache);
LuxKernel* lux_kernel_cache_get(LuxKernelCache* cache, const LuxKernelVariant* variant);
void lux_kernel_cache_put(LuxKernelCache* cache, const LuxKernelVariant* variant, LuxKernel* kernel);
void lux_kernel_cache_clear(LuxKernelCache* cache);
void lux_kernel_cache_stats(LuxKernelCache* cache, size_t* count, size_t* memory_bytes);
```

### Kernel Registry

```c
const LuxKernelRegistry* lux_kernel_registry_get(const char* backend);
const LuxEmbeddedKernel* lux_kernel_registry_find(const LuxKernelRegistry* registry, const char* name);
```

Access embedded kernels by name.

---

## Error Handling

### Error Codes

```c
typedef enum {
    LUX_OK                         = 0,
    LUX_ERROR_INVALID_ARGUMENT     = 1,
    LUX_ERROR_OUT_OF_MEMORY        = 2,
    LUX_ERROR_BACKEND_NOT_AVAILABLE = 3,
    LUX_ERROR_DEVICE_NOT_FOUND     = 4,
    LUX_ERROR_KERNEL_FAILED        = 5,
    LUX_ERROR_NOT_SUPPORTED        = 6,
} LuxError;
```

### Error Checking Pattern

```c
LuxGPU* gpu = lux_gpu_create_with_backend(LUX_BACKEND_DAWN);
if (!gpu) {
    fprintf(stderr, "Failed to create GPU context\n");
    return 1;
}

LuxError err = lux_gpu_sync(gpu);
if (err != LUX_OK) {
    fprintf(stderr, "Sync failed: %s\n", lux_gpu_error(gpu));
    lux_gpu_destroy(gpu);
    return 1;
}
```
