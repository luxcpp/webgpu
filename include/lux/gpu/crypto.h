// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Cryptographic GPU Kernel API
// Provides unified interface for ZK/blockchain cryptographic operations.
//
// Supported Operations:
//   - MSM (Multi-Scalar Multiplication) - Pippenger's algorithm
//   - Poseidon2 Hash - ZK-friendly algebraic hash
//   - BLS12-381 Curve Operations - Ethereum consensus
//   - BN254 Curve Operations - ZK-SNARKs
//   - Goldilocks Field - Plonky2/Polygon Zero
//   - Blake3 Hash - High-performance hash
//   - KZG Commitments - EIP-4844 blobs
//   - Shamir Secret Sharing - Threshold cryptography

#ifndef LUX_GPU_CRYPTO_H
#define LUX_GPU_CRYPTO_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Curve Type Identifiers
// =============================================================================

#ifndef LUX_GPU_CURVE_TYPE_DEFINED
#define LUX_GPU_CURVE_TYPE_DEFINED
typedef enum {
    LUX_CURVE_BN254       = 0,   // alt_bn128 (Ethereum precompile)
    LUX_CURVE_BLS12_381   = 1,   // Ethereum 2.0 consensus
    LUX_CURVE_SECP256K1   = 2,   // Bitcoin/Ethereum ECDSA
    LUX_CURVE_ED25519     = 3,   // EdDSA signatures
} LuxCurveType;
#endif

// =============================================================================
// Field Types
// =============================================================================

// 256-bit scalar (BN254, secp256k1, ed25519)
typedef struct {
    uint64_t limbs[4];
} LuxScalar256;

// 384-bit base field element (BLS12-381)
typedef struct {
    uint64_t limbs[6];
} LuxFp384;

// 64-bit Goldilocks field element
typedef uint64_t LuxGoldilocks;

// =============================================================================
// Point Types
// =============================================================================

// G1 Affine Point (BN254) - 256-bit coordinates
typedef struct {
    LuxScalar256 x;
    LuxScalar256 y;
    uint32_t infinity;
    uint32_t _pad;
} LuxG1Affine254;

// G1 Projective Point (BN254)
typedef struct {
    LuxScalar256 x;
    LuxScalar256 y;
    LuxScalar256 z;
} LuxG1Projective254;

// G1 Affine Point (BLS12-381) - 384-bit coordinates
typedef struct {
    LuxFp384 x;
    LuxFp384 y;
    uint32_t infinity;
    uint32_t _pad;
} LuxG1Affine381;

// G1 Projective Point (BLS12-381)
typedef struct {
    LuxFp384 x;
    LuxFp384 y;
    LuxFp384 z;
} LuxG1Projective381;

// =============================================================================
// Error Codes
// =============================================================================

typedef enum {
    LUX_CRYPTO_OK                    = 0,
    LUX_CRYPTO_ERROR_INVALID_ARG     = 1,
    LUX_CRYPTO_ERROR_OUT_OF_MEMORY   = 2,
    LUX_CRYPTO_ERROR_NOT_SUPPORTED   = 3,
    LUX_CRYPTO_ERROR_INVALID_CURVE   = 4,
    LUX_CRYPTO_ERROR_INVALID_POINT   = 5,
    LUX_CRYPTO_ERROR_DEVICE_ERROR    = 6,
} LuxCryptoError;

// =============================================================================
// Opaque Context Handle
// =============================================================================

typedef struct LuxCryptoContext LuxCryptoContext;

// =============================================================================
// MSM (Multi-Scalar Multiplication)
// =============================================================================

// Compute sum of scalar[i] * point[i] using Pippenger's algorithm
// Result is in projective coordinates.
//
// curve_type: LUX_CURVE_BN254 or LUX_CURVE_BLS12_381
// points:     Affine points (LuxG1Affine254[] or LuxG1Affine381[])
// scalars:    Scalars (LuxScalar256[])
// result:     Projective result (LuxG1Projective254* or LuxG1Projective381*)
// count:      Number of point-scalar pairs
LuxCryptoError lux_crypto_msm(
    LuxCryptoContext* ctx,
    int curve_type,
    const void* points,
    const void* scalars,
    void* result,
    size_t count
);

// Batch MSM: multiple independent MSM computations
LuxCryptoError lux_crypto_msm_batch(
    LuxCryptoContext* ctx,
    int curve_type,
    const void* const* points_batch,
    const void* const* scalars_batch,
    void** results_batch,
    const size_t* counts,
    size_t batch_size
);

// =============================================================================
// Poseidon2 Hash
// =============================================================================

// Poseidon2 sponge hash (ZK-friendly, BN254 scalar field)
// state_width: 3 for 2-to-1 hash, 4+ for higher rate
// inputs:      LuxScalar256[] input elements
// output:      LuxScalar256* output hash
LuxCryptoError lux_crypto_poseidon2_hash(
    LuxCryptoContext* ctx,
    const LuxScalar256* inputs,
    size_t num_inputs,
    LuxScalar256* output
);

// Batch Poseidon2 (multiple independent hashes)
LuxCryptoError lux_crypto_poseidon2_batch(
    LuxCryptoContext* ctx,
    const LuxScalar256* inputs,
    size_t inputs_per_hash,
    size_t num_hashes,
    LuxScalar256* outputs
);

// Poseidon2 Merkle tree construction
LuxCryptoError lux_crypto_poseidon2_merkle(
    LuxCryptoContext* ctx,
    const LuxScalar256* leaves,
    size_t num_leaves,
    LuxScalar256* tree_nodes
);

// =============================================================================
// BLS12-381 Curve Operations
// =============================================================================

// Point addition: R = P + Q
LuxCryptoError lux_crypto_bls12_381_add(
    LuxCryptoContext* ctx,
    const LuxG1Projective381* p,
    const LuxG1Projective381* q,
    LuxG1Projective381* result
);

// Point doubling: R = 2P
LuxCryptoError lux_crypto_bls12_381_double(
    LuxCryptoContext* ctx,
    const LuxG1Projective381* p,
    LuxG1Projective381* result
);

// Scalar multiplication: R = scalar * P
LuxCryptoError lux_crypto_bls12_381_scalar_mul(
    LuxCryptoContext* ctx,
    const LuxG1Projective381* p,
    const LuxScalar256* scalar,
    LuxG1Projective381* result
);

// Batch scalar multiplication
LuxCryptoError lux_crypto_bls12_381_scalar_mul_batch(
    LuxCryptoContext* ctx,
    const LuxG1Affine381* points,
    const LuxScalar256* scalars,
    LuxG1Projective381* results,
    size_t count
);

// Convert projective to affine
LuxCryptoError lux_crypto_bls12_381_to_affine(
    LuxCryptoContext* ctx,
    const LuxG1Projective381* p,
    LuxG1Affine381* result
);

// =============================================================================
// BN254 Curve Operations
// =============================================================================

// Point addition
LuxCryptoError lux_crypto_bn254_add(
    LuxCryptoContext* ctx,
    const LuxG1Projective254* p,
    const LuxG1Projective254* q,
    LuxG1Projective254* result
);

// Point doubling
LuxCryptoError lux_crypto_bn254_double(
    LuxCryptoContext* ctx,
    const LuxG1Projective254* p,
    LuxG1Projective254* result
);

// Scalar multiplication
LuxCryptoError lux_crypto_bn254_scalar_mul(
    LuxCryptoContext* ctx,
    const LuxG1Projective254* p,
    const LuxScalar256* scalar,
    LuxG1Projective254* result
);

// Batch scalar multiplication
LuxCryptoError lux_crypto_bn254_scalar_mul_batch(
    LuxCryptoContext* ctx,
    const LuxG1Affine254* points,
    const LuxScalar256* scalars,
    LuxG1Projective254* results,
    size_t count
);

// =============================================================================
// Goldilocks Field Operations
// =============================================================================

// Vector addition: result[i] = (a[i] + b[i]) mod p
LuxCryptoError lux_crypto_goldilocks_vec_add(
    LuxCryptoContext* ctx,
    const LuxGoldilocks* a,
    const LuxGoldilocks* b,
    LuxGoldilocks* result,
    size_t n
);

// Vector multiplication: result[i] = (a[i] * b[i]) mod p
LuxCryptoError lux_crypto_goldilocks_vec_mul(
    LuxCryptoContext* ctx,
    const LuxGoldilocks* a,
    const LuxGoldilocks* b,
    LuxGoldilocks* result,
    size_t n
);

// NTT (Number Theoretic Transform)
LuxCryptoError lux_crypto_goldilocks_ntt_forward(
    LuxCryptoContext* ctx,
    LuxGoldilocks* data,
    const LuxGoldilocks* twiddles,
    size_t n,
    uint32_t log_n
);

LuxCryptoError lux_crypto_goldilocks_ntt_inverse(
    LuxCryptoContext* ctx,
    LuxGoldilocks* data,
    const LuxGoldilocks* inv_twiddles,
    size_t n,
    uint32_t log_n
);

// Batch polynomial evaluation
LuxCryptoError lux_crypto_goldilocks_poly_eval(
    LuxCryptoContext* ctx,
    const LuxGoldilocks* coeffs,
    uint32_t degree,
    const LuxGoldilocks* points,
    LuxGoldilocks* results,
    size_t num_points
);

// =============================================================================
// Blake3 Hash
// =============================================================================

// Single hash
LuxCryptoError lux_crypto_blake3_hash(
    LuxCryptoContext* ctx,
    const uint8_t* input,
    size_t input_len,
    uint8_t output[32]
);

// Batch hash (multiple independent hashes)
LuxCryptoError lux_crypto_blake3_batch(
    LuxCryptoContext* ctx,
    const uint8_t* inputs,
    size_t input_stride,
    const size_t* input_lengths,
    uint8_t* outputs,
    size_t num_inputs
);

// =============================================================================
// KZG Polynomial Commitments
// =============================================================================

// Commit to polynomial: C = sum(coeffs[i] * g1[i])
LuxCryptoError lux_crypto_kzg_commit(
    LuxCryptoContext* ctx,
    const LuxG1Affine381* srs_g1,
    const LuxScalar256* coeffs,
    LuxG1Projective381* commitment,
    uint32_t degree
);

// Evaluate polynomial at point
LuxCryptoError lux_crypto_kzg_evaluate(
    LuxCryptoContext* ctx,
    const LuxScalar256* coeffs,
    const LuxScalar256* z,
    LuxScalar256* result,
    uint32_t degree
);

// Generate proof: W = (p(x) - p(z)) / (x - z) evaluated at SRS
LuxCryptoError lux_crypto_kzg_prove(
    LuxCryptoContext* ctx,
    const LuxG1Affine381* srs_g1,
    const LuxScalar256* coeffs,
    const LuxScalar256* z,
    const LuxScalar256* p_z,
    LuxG1Projective381* proof,
    uint32_t degree
);

// Batch commit multiple polynomials
LuxCryptoError lux_crypto_kzg_batch_commit(
    LuxCryptoContext* ctx,
    const LuxG1Affine381* srs_g1,
    const LuxScalar256* coeffs,
    LuxG1Projective381* commitments,
    uint32_t degree,
    uint32_t num_polys
);

// =============================================================================
// Shamir Secret Sharing
// =============================================================================

// Reconstruct secret from threshold shares
LuxCryptoError lux_crypto_shamir_reconstruct(
    LuxCryptoContext* ctx,
    int curve_type,
    const LuxScalar256* x_coords,
    const LuxScalar256* y_coords,
    LuxScalar256* secret,
    uint32_t threshold
);

// Batch reconstruct multiple secrets
LuxCryptoError lux_crypto_shamir_batch_reconstruct(
    LuxCryptoContext* ctx,
    int curve_type,
    const LuxScalar256* x_coords,
    const LuxScalar256* y_coords,
    LuxScalar256* secrets,
    uint32_t threshold,
    uint32_t batch_size
);

// Generate shares from polynomial
LuxCryptoError lux_crypto_shamir_generate_shares(
    LuxCryptoContext* ctx,
    int curve_type,
    const LuxScalar256* coefficients,
    const uint32_t* party_ids,
    LuxScalar256* shares,
    uint32_t threshold,
    uint32_t num_parties
);

// Compute Lagrange coefficients for reconstruction
LuxCryptoError lux_crypto_shamir_lagrange_coefficients(
    LuxCryptoContext* ctx,
    int curve_type,
    const LuxScalar256* x_coords,
    LuxScalar256* coefficients,
    uint32_t num_parties
);

// Proactive share refresh
LuxCryptoError lux_crypto_shamir_refresh_shares(
    LuxCryptoContext* ctx,
    int curve_type,
    const LuxScalar256* old_shares,
    const LuxScalar256* refresh_coeffs,
    const uint32_t* party_ids,
    LuxScalar256* new_shares,
    uint32_t threshold,
    uint32_t num_parties
);

#ifdef __cplusplus
}
#endif

#endif // LUX_GPU_CRYPTO_H
