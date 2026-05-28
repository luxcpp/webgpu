# Lux WebGPU Backend Documentation

Cross-platform GPU compute acceleration via WebGPU (Dawn/wgpu-native).

## Quick Links

- [README](../README.md) - Overview and quick start
- [API Reference](api.md) - Complete C API documentation
- [WGSL Kernel Guide](kernels.md) - Writing custom GPU kernels
- [Architecture](architecture.md) - System design and internals

## Features

### ML/Tensor Operations
- Element-wise: add, sub, mul, div, pow
- Unary: exp, log, sqrt, tanh, sigmoid, relu, gelu
- Reductions: sum, max, min, mean
- Normalization: softmax, layer_norm, rms_norm
- Matrix: matmul, transpose

### Cryptographic Primitives
- Elliptic curves: BLS12-381, BN254, secp256k1
- Multi-scalar multiplication (MSM)
- Hash functions: Poseidon2, BLAKE3
- KZG polynomial commitments
- FROST threshold signatures

### Zero-Knowledge Proofs
- Number theoretic transform (NTT)
- Poseidon hash (BN254, Goldilocks)
- Merkle tree construction

### Fully Homomorphic Encryption
- TFHE programmable bootstrap
- Key switching
- Blind rotation
- External product

## Supported Platforms

| Platform | Backend | Status |
|----------|---------|--------|
| macOS | Dawn/Metal | Stable |
| Linux | Dawn/Vulkan | Stable |
| Windows | Dawn/D3D12 | Beta |
| Web | WebGPU | Planned |

## Getting Started

```c
#include <lux/gpu.h>

int main() {
    LuxGPU* gpu = lux_gpu_create();

    int64_t shape[] = {1024};
    LuxTensor* a = lux_tensor_ones(gpu, shape, 1, LUX_FLOAT32);
    LuxTensor* b = lux_tensor_ones(gpu, shape, 1, LUX_FLOAT32);
    LuxTensor* c = lux_tensor_add(gpu, a, b);

    lux_gpu_sync(gpu);

    lux_tensor_destroy(a);
    lux_tensor_destroy(b);
    lux_tensor_destroy(c);
    lux_gpu_destroy(gpu);
}
```

## License

BSD-3-Clause-Eco

Copyright (c) 2024-2026 Lux Industries Inc.
