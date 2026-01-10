// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Cryptographic Backend Extension
// Extends the base backend plugin with crypto-specific operations.

#ifndef LUX_GPU_CRYPTO_BACKEND_H
#define LUX_GPU_CRYPTO_BACKEND_H

#include "lux/gpu/backend_plugin.h"
#include "lux/gpu/crypto.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Crypto Backend VTable Extension
// =============================================================================

typedef struct lux_gpu_crypto_vtbl {
    // MSM Operations
    LuxCryptoError (*msm)(
        LuxBackendContext* ctx,
        int curve_type,
        const void* points,
        const void* scalars,
        void* result,
        size_t count
    );

    LuxCryptoError (*msm_batch)(
        LuxBackendContext* ctx,
        int curve_type,
        const void* const* points_batch,
        const void* const* scalars_batch,
        void** results_batch,
        const size_t* counts,
        size_t batch_size
    );

    // Poseidon2 Hash
    LuxCryptoError (*poseidon2_hash)(
        LuxBackendContext* ctx,
        const LuxScalar256* inputs,
        size_t num_inputs,
        LuxScalar256* output
    );

    LuxCryptoError (*poseidon2_batch)(
        LuxBackendContext* ctx,
        const LuxScalar256* inputs,
        size_t inputs_per_hash,
        size_t num_hashes,
        LuxScalar256* outputs
    );

    LuxCryptoError (*poseidon2_merkle)(
        LuxBackendContext* ctx,
        const LuxScalar256* leaves,
        size_t num_leaves,
        LuxScalar256* tree_nodes
    );

    // BLS12-381 Operations
    LuxCryptoError (*bls12_381_add)(
        LuxBackendContext* ctx,
        const LuxG1Projective381* p,
        const LuxG1Projective381* q,
        LuxG1Projective381* result
    );

    LuxCryptoError (*bls12_381_double)(
        LuxBackendContext* ctx,
        const LuxG1Projective381* p,
        LuxG1Projective381* result
    );

    LuxCryptoError (*bls12_381_scalar_mul)(
        LuxBackendContext* ctx,
        const LuxG1Projective381* p,
        const LuxScalar256* scalar,
        LuxG1Projective381* result
    );

    LuxCryptoError (*bls12_381_scalar_mul_batch)(
        LuxBackendContext* ctx,
        const LuxG1Affine381* points,
        const LuxScalar256* scalars,
        LuxG1Projective381* results,
        size_t count
    );

    // BN254 Operations
    LuxCryptoError (*bn254_add)(
        LuxBackendContext* ctx,
        const LuxG1Projective254* p,
        const LuxG1Projective254* q,
        LuxG1Projective254* result
    );

    LuxCryptoError (*bn254_double)(
        LuxBackendContext* ctx,
        const LuxG1Projective254* p,
        LuxG1Projective254* result
    );

    LuxCryptoError (*bn254_scalar_mul)(
        LuxBackendContext* ctx,
        const LuxG1Projective254* p,
        const LuxScalar256* scalar,
        LuxG1Projective254* result
    );

    LuxCryptoError (*bn254_scalar_mul_batch)(
        LuxBackendContext* ctx,
        const LuxG1Affine254* points,
        const LuxScalar256* scalars,
        LuxG1Projective254* results,
        size_t count
    );

    // Goldilocks Field Operations
    LuxCryptoError (*goldilocks_vec_add)(
        LuxBackendContext* ctx,
        const LuxGoldilocks* a,
        const LuxGoldilocks* b,
        LuxGoldilocks* result,
        size_t n
    );

    LuxCryptoError (*goldilocks_vec_mul)(
        LuxBackendContext* ctx,
        const LuxGoldilocks* a,
        const LuxGoldilocks* b,
        LuxGoldilocks* result,
        size_t n
    );

    LuxCryptoError (*goldilocks_ntt_forward)(
        LuxBackendContext* ctx,
        LuxGoldilocks* data,
        const LuxGoldilocks* twiddles,
        size_t n,
        uint32_t log_n
    );

    LuxCryptoError (*goldilocks_ntt_inverse)(
        LuxBackendContext* ctx,
        LuxGoldilocks* data,
        const LuxGoldilocks* inv_twiddles,
        size_t n,
        uint32_t log_n
    );

    // Blake3 Hash
    LuxCryptoError (*blake3_hash)(
        LuxBackendContext* ctx,
        const uint8_t* input,
        size_t input_len,
        uint8_t output[32]
    );

    LuxCryptoError (*blake3_batch)(
        LuxBackendContext* ctx,
        const uint8_t* inputs,
        size_t input_stride,
        const size_t* input_lengths,
        uint8_t* outputs,
        size_t num_inputs
    );

    // KZG Commitments
    LuxCryptoError (*kzg_commit)(
        LuxBackendContext* ctx,
        const LuxG1Affine381* srs_g1,
        const LuxScalar256* coeffs,
        LuxG1Projective381* commitment,
        uint32_t degree
    );

    LuxCryptoError (*kzg_prove)(
        LuxBackendContext* ctx,
        const LuxG1Affine381* srs_g1,
        const LuxScalar256* coeffs,
        const LuxScalar256* z,
        const LuxScalar256* p_z,
        LuxG1Projective381* proof,
        uint32_t degree
    );

    LuxCryptoError (*kzg_batch_commit)(
        LuxBackendContext* ctx,
        const LuxG1Affine381* srs_g1,
        const LuxScalar256* coeffs,
        LuxG1Projective381* commitments,
        uint32_t degree,
        uint32_t num_polys
    );

    // Shamir Secret Sharing
    LuxCryptoError (*shamir_reconstruct)(
        LuxBackendContext* ctx,
        int curve_type,
        const LuxScalar256* x_coords,
        const LuxScalar256* y_coords,
        LuxScalar256* secret,
        uint32_t threshold
    );

    LuxCryptoError (*shamir_batch_reconstruct)(
        LuxBackendContext* ctx,
        int curve_type,
        const LuxScalar256* x_coords,
        const LuxScalar256* y_coords,
        LuxScalar256* secrets,
        uint32_t threshold,
        uint32_t batch_size
    );

    LuxCryptoError (*shamir_lagrange_coefficients)(
        LuxBackendContext* ctx,
        int curve_type,
        const LuxScalar256* x_coords,
        LuxScalar256* coefficients,
        uint32_t num_parties
    );

    // Reserved for future expansion
    void* _reserved[8];

} lux_gpu_crypto_vtbl;

// =============================================================================
// Extended Backend Descriptor
// =============================================================================

typedef struct {
    lux_gpu_backend_desc base;              // Base backend descriptor
    const lux_gpu_crypto_vtbl* crypto_vtbl; // Crypto operations vtable
    uint32_t crypto_capabilities;           // Crypto-specific caps
} lux_gpu_crypto_backend_desc;

// Crypto capability flags
#define LUX_CRYPTO_CAP_MSM           (1 << 0)  // Multi-scalar multiplication
#define LUX_CRYPTO_CAP_POSEIDON2     (1 << 1)  // Poseidon2 hash
#define LUX_CRYPTO_CAP_BLS12_381     (1 << 2)  // BLS12-381 curve
#define LUX_CRYPTO_CAP_BN254         (1 << 3)  // BN254 curve
#define LUX_CRYPTO_CAP_GOLDILOCKS    (1 << 4)  // Goldilocks field
#define LUX_CRYPTO_CAP_BLAKE3        (1 << 5)  // Blake3 hash
#define LUX_CRYPTO_CAP_KZG           (1 << 6)  // KZG commitments
#define LUX_CRYPTO_CAP_SHAMIR        (1 << 7)  // Shamir secret sharing

#ifdef __cplusplus
}
#endif

#endif // LUX_GPU_CRYPTO_BACKEND_H
