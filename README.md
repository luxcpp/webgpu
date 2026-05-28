# Lux WebGPU Backend — **ARCHIVED, moved to lux-private/gpu-kernels**

> **This repository has moved.** The WebGPU backend plugin source +
> WGSL shaders now live in
> [`lux-private/gpu-kernels`](https://github.com/lux-private/gpu-kernels)
> at `webgpu/` and `kernels/{domain}/[primitive/]wgsl/`. History was
> migrated via `git filter-repo --to-subdirectory-filter webgpu/` and
> merged with `--allow-unrelated-histories`, so every commit here is
> preserved over there at the same SHA's filtered counterpart.
>
> The public C ABI (`<lux/gpu.h>`) and the plugin loader continue to
> live in [`luxcpp/gpu`](https://github.com/luxcpp/gpu). At runtime
> luxcpp/gpu still dlopens `libluxgpu_backend_webgpu.dylib` from
> `/usr/local/lib/lux-gpu/` — the install layout and ABI are unchanged.
>
> For development, build the plugin via the new home:
>
> ```bash
> cd ~/work/lux-private/gpu-kernels
> cmake -B build -DLUX_GPU_KERNELS_BUILD_METAL=OFF
> cmake --build build -j$(nproc)
> # Plugin lands at build/webgpu_backend/libluxgpu_backend_webgpu.{so,dylib}
> ```

---

Cross-platform GPU compute acceleration via WebGPU (Dawn/wgpu-native).

## Overview

The Lux WebGPU backend provides portable GPU acceleration for:

- **ML/Tensor Operations**: Element-wise ops, reductions, normalization, attention
- **Cryptographic Primitives**: BLS12-381, BN254, secp256k1, Poseidon2, MSM
- **Zero-Knowledge Proofs**: NTT, polynomial arithmetic, Merkle trees
- **Fully Homomorphic Encryption**: TFHE bootstrap, keyswitch, blind rotation

All kernels are written in WGSL (WebGPU Shading Language) and compile at runtime via Dawn or wgpu-native.

## Requirements

- CMake 3.20+
- C++17 compiler
- WebGPU implementation:
  - [Dawn](https://dawn.googlesource.com/dawn) (recommended)
  - [wgpu-native](https://github.com/gfx-rs/wgpu-native) (alternative)

## Installation

### From Source

```bash
git clone https://github.com/luxfi/luxcpp.git
cd luxcpp/webgpu

# Configure (auto-detects Dawn or wgpu-native)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j

# Test
ctest --test-dir build --output-on-failure

# Install
cmake --install build --prefix /usr/local
```

### macOS (Homebrew)

```bash
brew install wgpu-native  # or build Dawn from source
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `LUX_WEBGPU_BUILD_TESTS` | ON | Build test suite |
| `LUX_WEBGPU_EMBED_KERNELS` | ON | Embed WGSL sources in binary |
| `LUX_WEBGPU_USE_DAWN` | ON | Prefer Dawn over wgpu-native |

## Quick Start

### C API

```c
#include <lux/gpu.h>

int main() {
    // Create GPU context (auto-selects WebGPU backend)
    LuxGPU* gpu = lux_gpu_create_with_backend(LUX_BACKEND_DAWN);

    // Create tensors
    int64_t shape[] = {1024};
    LuxTensor* a = lux_tensor_ones(gpu, shape, 1, LUX_FLOAT32);
    LuxTensor* b = lux_tensor_full(gpu, shape, 1, LUX_FLOAT32, 2.0);

    // Element-wise addition
    LuxTensor* c = lux_tensor_add(gpu, a, b);

    // Synchronize and read result
    lux_gpu_sync(gpu);
    float result[1024];
    lux_tensor_to_host(c, result, sizeof(result));

    // Cleanup
    lux_tensor_destroy(a);
    lux_tensor_destroy(b);
    lux_tensor_destroy(c);
    lux_gpu_destroy(gpu);
    return 0;
}
```

### Custom WGSL Kernel

```c
#include <lux/gpu/kernel_loader.h>

const char* wgsl_source = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> size: u32;

@compute @workgroup_size(256)
fn scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= size) { return; }
    output[gid.x] = input[gid.x] * 2.0;
}
)";

// Compile kernel
LuxKernel* kernel = lux_wgpu_kernel_compile(device, wgsl_source, 0, "scale");

// Dispatch via backend vtable
// ...

lux_wgpu_kernel_destroy(kernel);
```

## WGSL Kernel Library

Pre-built kernels organized by domain:

```
kernels/wgsl/
  gpu/           # ML/tensor operations
    binary.wgsl      - add, sub, mul, div, pow, min, max
    unary.wgsl       - exp, log, sqrt, tanh, sigmoid, relu, gelu
    reduce.wgsl      - sum, max, min, mean (full and axis)
    softmax.wgsl     - softmax, log_softmax
    layer_norm.wgsl  - layer normalization
    rms_norm.wgsl    - RMS normalization
    gemv.wgsl        - matrix-vector multiply
    rope.wgsl        - rotary position embedding
    ntt.wgsl         - number theoretic transform
    fft.wgsl         - fast Fourier transform

  crypto/        # Cryptographic primitives
    bls12_381.wgsl   - BLS12-381 curve operations
    bn254.wgsl       - BN254 (alt_bn128) curve
    secp256k1.wgsl   - Bitcoin/Ethereum ECDSA curve
    msm.wgsl         - multi-scalar multiplication
    poseidon_goldilocks.wgsl - Poseidon hash (Goldilocks field)
    kzg.wgsl         - KZG polynomial commitments
    frost_*.wgsl     - FROST threshold signatures
    mldsa_verify.wgsl - ML-DSA (Dilithium) verification

  zk/            # Zero-knowledge proof primitives
    poseidon2.wgsl      - Poseidon2 hash
    poseidon_bn254.wgsl - Poseidon for BN254 scalar field
    merkle.wgsl         - Merkle tree construction

  fhe/           # Fully homomorphic encryption
    tfhe_bootstrap.wgsl     - TFHE programmable bootstrap
    tfhe_keyswitch.wgsl     - TFHE key switching
    blind_rotate.wgsl       - blind rotation
    external_product_*.wgsl - GLWE external product
    ntt_kernels.wgsl        - NTT for FHE polynomials
```

## API Reference

### GPU Context

```c
// Create/destroy
LuxGPU* lux_gpu_create(void);
LuxGPU* lux_gpu_create_with_backend(LuxBackend backend);
void lux_gpu_destroy(LuxGPU* gpu);

// Query
LuxBackend lux_gpu_backend(LuxGPU* gpu);
const char* lux_gpu_backend_name(LuxGPU* gpu);
LuxError lux_gpu_device_info(LuxGPU* gpu, LuxDeviceInfo* info);

// Synchronization
LuxError lux_gpu_sync(LuxGPU* gpu);
```

### Tensor Operations

```c
// Creation
LuxTensor* lux_tensor_zeros(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype);
LuxTensor* lux_tensor_ones(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype);
LuxTensor* lux_tensor_from_data(LuxGPU* gpu, const void* data, const int64_t* shape, int ndim, LuxDtype dtype);

// Arithmetic
LuxTensor* lux_tensor_add(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
LuxTensor* lux_tensor_sub(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
LuxTensor* lux_tensor_mul(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
LuxTensor* lux_tensor_div(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
LuxTensor* lux_tensor_matmul(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);

// Unary
LuxTensor* lux_tensor_exp(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_log(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_tanh(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_relu(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_gelu(LuxGPU* gpu, LuxTensor* t);

// Reductions
float lux_tensor_reduce_sum(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_sum(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes);
LuxTensor* lux_tensor_softmax(LuxGPU* gpu, LuxTensor* t, int axis);
```

### Cryptographic Operations

```c
// Multi-scalar multiplication
LuxError lux_msm(LuxGPU* gpu, const void* scalars, const void* points,
                 void* result, size_t count, LuxCurve curve);

// BLS12-381
LuxError lux_bls12_381_add(LuxGPU* gpu, const void* a, const void* b,
                           void* out, size_t count, bool is_g2);
LuxError lux_bls12_381_mul(LuxGPU* gpu, const void* points, const void* scalars,
                           void* out, size_t count, bool is_g2);
LuxError lux_bls12_381_pairing(LuxGPU* gpu, const void* g1, const void* g2,
                               void* out, size_t count);

// KZG commitments
LuxError lux_kzg_commit(LuxGPU* gpu, const void* coeffs, const void* srs,
                        void* commitment, size_t degree, LuxCurve curve);
LuxError lux_kzg_verify(LuxGPU* gpu, const void* commitment, const void* proof,
                        const void* point, const void* value, const void* srs_g2,
                        bool* result, LuxCurve curve);

// Poseidon2 hash
LuxError lux_poseidon2_hash(LuxGPU* gpu, const uint64_t* inputs, uint64_t* outputs,
                            size_t rate, size_t num_hashes);
```

### FHE Operations

```c
// NTT
LuxError lux_ntt_forward(LuxGPU* gpu, uint64_t* data, size_t n, uint64_t modulus);
LuxError lux_ntt_inverse(LuxGPU* gpu, uint64_t* data, size_t n, uint64_t modulus);

// TFHE
LuxError lux_tfhe_bootstrap(LuxGPU* gpu, const uint64_t* lwe_in, uint64_t* lwe_out,
                            const uint64_t* bsk, const uint64_t* test_poly,
                            uint32_t n_lwe, uint32_t N, uint32_t k, uint32_t l, uint64_t q);
LuxError lux_tfhe_keyswitch(LuxGPU* gpu, const uint64_t* lwe_in, uint64_t* lwe_out,
                            const uint64_t* ksk, uint32_t n_in, uint32_t n_out,
                            uint32_t l, uint32_t base_log, uint64_t q);
LuxError lux_blind_rotate(LuxGPU* gpu, uint64_t* acc, const uint64_t* bsk,
                          const uint64_t* lwe_a, uint32_t n_lwe, uint32_t N,
                          uint32_t k, uint32_t l, uint64_t q);
```

## Backend Plugin Architecture

The WebGPU backend implements the `lux_gpu_backend_vtbl` interface:

```c
// Plugin entry point (exported from shared library)
bool lux_gpu_backend_init(lux_gpu_backend_desc* out);

// Capability flags
LUX_CAP_TENSOR_OPS      // Basic tensor operations
LUX_CAP_MATMUL          // Matrix multiplication
LUX_CAP_NTT             // Number theoretic transform
LUX_CAP_MSM             // Multi-scalar multiplication
LUX_CAP_FHE             // FHE operations
LUX_CAP_TFHE            // TFHE bootstrap/keyswitch
LUX_CAP_BLS12_381       // BLS12-381 curve
LUX_CAP_BN254           // BN254 curve
LUX_CAP_KZG             // KZG commitments
LUX_CAP_POSEIDON2       // Poseidon2 hash
```

## Performance

Benchmarks on Apple M2 (Metal via Dawn):

| Operation | Size | Time | Throughput |
|-----------|------|------|------------|
| add_f32 | 1M | 0.12ms | 8.3 GB/s |
| matmul_f32 | 1024x1024 | 1.8ms | 1.2 TFLOPS |
| reduce_sum | 1M | 0.08ms | - |
| softmax | 4096x4096 | 2.1ms | - |
| MSM (BLS12-381) | 2^16 | 45ms | - |
| NTT | 2^20 | 8.2ms | - |
| Poseidon2 batch | 2^16 | 12ms | - |

## Writing WGSL Kernels

### Kernel Structure

```wgsl
// Parameter struct (uniform buffer)
struct Params {
    size: u32,
    // ... other parameters
}

// Storage bindings
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// Compute shader
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }
    output[gid.x] = input[gid.x] * 2.0;
}
```

### Best Practices

1. **Workgroup size**: Use 256 for general compute, 64 for memory-bound ops
2. **Coalesced access**: Ensure adjacent threads access adjacent memory
3. **Avoid divergence**: Minimize branching within workgroups
4. **Use shared memory**: For reduction patterns, use `var<workgroup>`
5. **Vectorize**: Use `vec4<f32>` for 4x throughput where applicable

## Documentation

- [API Reference](docs/api.md)
- [WGSL Kernel Guide](docs/kernels.md)
- [Architecture](docs/architecture.md)
- [Doxygen](https://luxfi.github.io/luxcpp/webgpu/)

## License

BSD-3-Clause-Eco

Copyright (c) 2024-2026 Lux Industries Inc.
