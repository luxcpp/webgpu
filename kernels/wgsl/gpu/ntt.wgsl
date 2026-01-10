// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Number Theoretic Transform (NTT) Kernels for Lattice Cryptography
// Implements:
// - Forward NTT (negacyclic) for Kyber/Dilithium
// - Inverse NTT with scaling
// - Pointwise multiplication in NTT domain
// - Fused NTT operations for performance
//
// Supports prime moduli:
// - q = 8380417 (Dilithium)
// - q = 3329 (Kyber)
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// Dilithium parameters (n=256, q=8380417)
const DILITHIUM_N: u32 = 256u;
const DILITHIUM_Q: i32 = 8380417;
const DILITHIUM_QINV: u32 = 58728449u;  // q^(-1) mod 2^32
const DILITHIUM_ROOT: i32 = 1753;       // 256th root of unity mod q
const DILITHIUM_ROOT_INV: i32 = 731434; // Inverse root

// Kyber parameters (n=256, q=3329)
const KYBER_N: u32 = 256u;
const KYBER_Q: i32 = 3329;
const KYBER_QINV: u32 = 62209u;         // q^(-1) mod 2^16
const KYBER_ROOT: i32 = 17;             // 256th root of unity mod q

// Precomputed roots of unity for Dilithium (first 8 levels)
// zeta[i] = root^(bitrev(i)) mod q
const DILITHIUM_ZETAS_0: i32 = 0;       // Not used (zeta[0])
const DILITHIUM_ZETAS_1: i32 = 25847;
const DILITHIUM_ZETAS_2: i32 = -2608894;
const DILITHIUM_ZETAS_3: i32 = -518909;
const DILITHIUM_ZETAS_4: i32 = 237124;
const DILITHIUM_ZETAS_5: i32 = -777960;
const DILITHIUM_ZETAS_6: i32 = -876248;
const DILITHIUM_ZETAS_7: i32 = 466468;

// Precomputed roots of unity for Kyber
const KYBER_ZETAS: array<i32, 128> = array<i32, 128>(
    1, 1729, 2580, 3289, 2642, 630, 1897, 848,
    1062, 1919, 193, 797, 2786, 3260, 569, 1746,
    296, 2447, 1339, 1476, 3046, 56, 2240, 1333,
    1426, 2094, 535, 2882, 2393, 2879, 1974, 821,
    289, 331, 3253, 1756, 1197, 2304, 2277, 2055,
    650, 1977, 2513, 632, 2865, 33, 1320, 1915,
    2319, 1435, 807, 452, 1438, 2868, 1534, 2402,
    2647, 2617, 1481, 648, 2474, 3110, 1227, 910,
    17, 2761, 583, 2649, 1637, 723, 2288, 1100,
    1409, 2662, 3281, 233, 756, 2156, 3015, 3050,
    1703, 1651, 2789, 1789, 1847, 952, 1461, 2687,
    939, 2308, 2437, 2388, 733, 2337, 268, 641,
    1584, 2298, 2037, 3220, 375, 2549, 2090, 1645,
    1063, 319, 2773, 757, 2099, 561, 2466, 2594,
    2804, 1092, 403, 1026, 1143, 2150, 2775, 886,
    1722, 1212, 1874, 1029, 2110, 2935, 885, 2154
);

// ============================================================================
// Parameter Structures
// ============================================================================

struct NTTParams {
    n: u32,              // Polynomial degree (256 for Kyber/Dilithium)
    batch_size: u32,     // Number of polynomials to transform
    mode: u32,           // 0 = Dilithium, 1 = Kyber
    _pad: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read_write> data: array<i32>;
@group(0) @binding(1) var<storage, read> zetas: array<i32>;  // Precomputed roots
@group(0) @binding(2) var<uniform> params: NTTParams;

// For pointwise operations
@group(0) @binding(3) var<storage, read> input_a: array<i32>;
@group(0) @binding(4) var<storage, read> input_b: array<i32>;
@group(0) @binding(5) var<storage, read_write> output: array<i32>;

// ============================================================================
// Workgroup Shared Memory
// ============================================================================

var<workgroup> shared_data: array<i32, 512>;  // 2 * WORKGROUP_SIZE

// ============================================================================
// Modular Arithmetic
// ============================================================================

// Montgomery reduction for Dilithium
fn montgomery_reduce_dilithium(a: i32) -> i32 {
    let t = a * i32(DILITHIUM_QINV);
    return (a - t * DILITHIUM_Q) >> 16;  // Simplified
}

// Barrett reduction for Dilithium
fn barrett_reduce_dilithium(a: i32) -> i32 {
    let t = i32((i32(8396807) * a) >> 23);  // Approximate shift
    var r = a - t * DILITHIUM_Q;
    r = r + select(0, DILITHIUM_Q, r < 0);
    r = r - select(0, DILITHIUM_Q, r >= DILITHIUM_Q);
    return r;
}

// Full reduction for Dilithium to [0, q)
fn full_reduce_dilithium(a: i32) -> i32 {
    var t = barrett_reduce_dilithium(a);
    t = t + select(0, DILITHIUM_Q, t < 0);
    return t;
}

// Montgomery reduction for Kyber
fn montgomery_reduce_kyber(a: i32) -> i32 {
    let t = (a & 0xFFFF) * i32(KYBER_QINV) & 0xFFFF;
    var result = (a - t * KYBER_Q) >> 16;
    return result;
}

// Barrett reduction for Kyber
fn barrett_reduce_kyber(a: i32) -> i32 {
    let t = i32((20159 * a + (1 << 25)) >> 26);
    var r = a - t * KYBER_Q;
    r = r + select(0, KYBER_Q, r < 0);
    r = r - select(0, KYBER_Q, r >= KYBER_Q);
    return r;
}

// Modular multiplication
fn mod_mul_dilithium(a: i32, b: i32) -> i32 {
    // In practice, use Montgomery multiplication
    // Simplified version:
    let product = a * b;
    return barrett_reduce_dilithium(product);
}

fn mod_mul_kyber(a: i32, b: i32) -> i32 {
    let product = a * b;
    return montgomery_reduce_kyber(product);
}

// ============================================================================
// NTT Butterfly Operations
// ============================================================================

// Cooley-Tukey butterfly: (a, b) -> (a + wb, a - wb)
fn butterfly_ct_dilithium(a: i32, b: i32, zeta: i32) -> array<i32, 2> {
    let t = mod_mul_dilithium(b, zeta);
    return array<i32, 2>(
        barrett_reduce_dilithium(a + t),
        barrett_reduce_dilithium(a - t)
    );
}

fn butterfly_ct_kyber(a: i32, b: i32, zeta: i32) -> array<i32, 2> {
    let t = mod_mul_kyber(b, zeta);
    return array<i32, 2>(
        barrett_reduce_kyber(a + t),
        barrett_reduce_kyber(a - t)
    );
}

// Gentleman-Sande butterfly: (a, b) -> (a + b, (a - b) * w)
fn butterfly_gs_dilithium(a: i32, b: i32, zeta: i32) -> array<i32, 2> {
    let sum = barrett_reduce_dilithium(a + b);
    let diff = barrett_reduce_dilithium(a - b);
    let t = mod_mul_dilithium(diff, zeta);
    return array<i32, 2>(sum, t);
}

fn butterfly_gs_kyber(a: i32, b: i32, zeta: i32) -> array<i32, 2> {
    let sum = barrett_reduce_kyber(a + b);
    let diff = barrett_reduce_kyber(a - b);
    let t = mod_mul_kyber(diff, zeta);
    return array<i32, 2>(sum, t);
}

// ============================================================================
// Forward NTT - Dilithium (negacyclic, Cooley-Tukey)
// ============================================================================

@compute @workgroup_size(256)
fn ntt_forward_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let poly_idx = wid.x;
    let n = DILITHIUM_N;

    if (poly_idx >= params.batch_size) { return; }

    let base_idx = poly_idx * n;

    // Load polynomial into shared memory
    if (lid.x < n) {
        shared_data[lid.x] = data[base_idx + lid.x];
    }
    workgroupBarrier();

    // NTT using Cooley-Tukey (decimation-in-time)
    var len = n / 2u;
    var k = 1u;

    while (len >= 1u) {
        // Each thread handles butterflies at this level
        var start = lid.x;
        while (start < n / 2u) {
            let group = start / len;
            let idx_in_group = start % len;
            let idx0 = group * len * 2u + idx_in_group;
            let idx1 = idx0 + len;

            let zeta = zetas[k + group];

            let a = shared_data[idx0];
            let b = shared_data[idx1];
            let result = butterfly_ct_dilithium(a, b, zeta);

            shared_data[idx0] = result[0];
            shared_data[idx1] = result[1];

            start += WORKGROUP_SIZE;
        }
        workgroupBarrier();

        len = len / 2u;
        k = k * 2u;
    }

    // Write result
    if (lid.x < n) {
        data[base_idx + lid.x] = shared_data[lid.x];
    }
}

// ============================================================================
// Forward NTT - Kyber (negacyclic, Cooley-Tukey)
// ============================================================================

@compute @workgroup_size(256)
fn ntt_forward_kyber(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let poly_idx = wid.x;
    let n = KYBER_N;

    if (poly_idx >= params.batch_size) { return; }

    let base_idx = poly_idx * n;

    // Load polynomial
    if (lid.x < n) {
        shared_data[lid.x] = data[base_idx + lid.x];
    }
    workgroupBarrier();

    // Kyber NTT: 7 levels for n=256
    var len = 128u;
    var k = 1u;

    // Level 0: len=128
    var start = lid.x;
    while (start < 128u) {
        let zeta = KYBER_ZETAS[k];
        let result = butterfly_ct_kyber(
            shared_data[start],
            shared_data[start + 128u],
            zeta
        );
        shared_data[start] = result[0];
        shared_data[start + 128u] = result[1];
        start += WORKGROUP_SIZE;
    }
    workgroupBarrier();

    // Levels 1-6
    len = 64u;
    k = 2u;

    while (len >= 2u) {
        start = lid.x;
        while (start < 128u) {
            let group = start / len;
            let idx_in_group = start % len;
            let idx0 = group * len * 2u + idx_in_group;
            let idx1 = idx0 + len;

            let zeta = KYBER_ZETAS[k + group];
            let result = butterfly_ct_kyber(
                shared_data[idx0],
                shared_data[idx1],
                zeta
            );
            shared_data[idx0] = result[0];
            shared_data[idx1] = result[1];

            start += WORKGROUP_SIZE;
        }
        workgroupBarrier();

        len = len / 2u;
        k = k * 2u;
    }

    // Write result
    if (lid.x < n) {
        data[base_idx + lid.x] = shared_data[lid.x];
    }
}

// ============================================================================
// Inverse NTT - Dilithium (Gentleman-Sande)
// ============================================================================

@compute @workgroup_size(256)
fn ntt_inverse_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let poly_idx = wid.x;
    let n = DILITHIUM_N;

    if (poly_idx >= params.batch_size) { return; }

    let base_idx = poly_idx * n;

    // Load polynomial
    if (lid.x < n) {
        shared_data[lid.x] = data[base_idx + lid.x];
    }
    workgroupBarrier();

    // Inverse NTT using Gentleman-Sande (decimation-in-frequency)
    var len = 1u;
    var k = n / 2u;

    while (len < n) {
        var start = lid.x;
        while (start < n / 2u) {
            let group = start / len;
            let idx_in_group = start % len;
            let idx0 = group * len * 2u + idx_in_group;
            let idx1 = idx0 + len;

            // Use inverse zeta (stored in reverse order in zetas buffer)
            let zeta = zetas[k + group];

            let a = shared_data[idx0];
            let b = shared_data[idx1];
            let result = butterfly_gs_dilithium(a, b, zeta);

            shared_data[idx0] = result[0];
            shared_data[idx1] = result[1];

            start += WORKGROUP_SIZE;
        }
        workgroupBarrier();

        len = len * 2u;
        k = k / 2u;
    }

    // Scale by n^(-1) mod q
    // For Dilithium: n^(-1) mod 8380417 = 8347681
    let n_inv = 8347681;
    if (lid.x < n) {
        shared_data[lid.x] = mod_mul_dilithium(shared_data[lid.x], n_inv);
    }
    workgroupBarrier();

    // Write result
    if (lid.x < n) {
        data[base_idx + lid.x] = full_reduce_dilithium(shared_data[lid.x]);
    }
}

// ============================================================================
// Inverse NTT - Kyber (Gentleman-Sande)
// ============================================================================

@compute @workgroup_size(256)
fn ntt_inverse_kyber(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let poly_idx = wid.x;
    let n = KYBER_N;

    if (poly_idx >= params.batch_size) { return; }

    let base_idx = poly_idx * n;

    // Load polynomial
    if (lid.x < n) {
        shared_data[lid.x] = data[base_idx + lid.x];
    }
    workgroupBarrier();

    // Inverse NTT
    var len = 2u;
    var k = 127u;

    while (len <= 128u) {
        var start = lid.x;
        while (start < 128u) {
            let group = start / len;
            let idx_in_group = start % len;
            let idx0 = group * len * 2u + idx_in_group;
            let idx1 = idx0 + len;

            let zeta = -KYBER_ZETAS[k - group];  // Negative for inverse
            let result = butterfly_gs_kyber(
                shared_data[idx0],
                shared_data[idx1],
                zeta
            );
            shared_data[idx0] = result[0];
            shared_data[idx1] = result[1];

            start += WORKGROUP_SIZE;
        }
        workgroupBarrier();

        len = len * 2u;
        k = k - len / 2u;
    }

    // Final level
    var start = lid.x;
    while (start < 128u) {
        let zeta = -KYBER_ZETAS[0];
        let result = butterfly_gs_kyber(
            shared_data[start],
            shared_data[start + 128u],
            zeta
        );
        shared_data[start] = result[0];
        shared_data[start + 128u] = result[1];
        start += WORKGROUP_SIZE;
    }
    workgroupBarrier();

    // Scale by n^(-1) mod q = 3303 for Kyber
    let n_inv = 3303;
    if (lid.x < n) {
        shared_data[lid.x] = mod_mul_kyber(shared_data[lid.x], n_inv);
    }
    workgroupBarrier();

    // Write result
    if (lid.x < n) {
        data[base_idx + lid.x] = barrett_reduce_kyber(shared_data[lid.x]);
    }
}

// ============================================================================
// Pointwise Multiplication in NTT Domain
// ============================================================================

@compute @workgroup_size(256)
fn ntt_pointwise_mul_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let n = params.n;
    let total = params.batch_size * n;

    if (gid.x >= total) { return; }

    let a = input_a[gid.x];
    let b = input_b[gid.x];
    output[gid.x] = mod_mul_dilithium(a, b);
}

@compute @workgroup_size(256)
fn ntt_pointwise_mul_kyber(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let n = params.n;
    let total = params.batch_size * n;

    if (gid.x >= total) { return; }

    let a = input_a[gid.x];
    let b = input_b[gid.x];
    output[gid.x] = mod_mul_kyber(a, b);
}

// ============================================================================
// Pointwise Multiply-Accumulate: output += a * b
// ============================================================================

@compute @workgroup_size(256)
fn ntt_pointwise_mac_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let n = params.n;
    let total = params.batch_size * n;

    if (gid.x >= total) { return; }

    let a = input_a[gid.x];
    let b = input_b[gid.x];
    let c = output[gid.x];
    output[gid.x] = barrett_reduce_dilithium(c + mod_mul_dilithium(a, b));
}

@compute @workgroup_size(256)
fn ntt_pointwise_mac_kyber(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let n = params.n;
    let total = params.batch_size * n;

    if (gid.x >= total) { return; }

    let a = input_a[gid.x];
    let b = input_b[gid.x];
    let c = output[gid.x];
    output[gid.x] = barrett_reduce_kyber(c + mod_mul_kyber(a, b));
}

// ============================================================================
// Polynomial Addition in Coefficient Domain
// ============================================================================

@compute @workgroup_size(256)
fn poly_add_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let total = params.batch_size * params.n;

    if (gid.x >= total) { return; }

    let a = input_a[gid.x];
    let b = input_b[gid.x];
    output[gid.x] = barrett_reduce_dilithium(a + b);
}

@compute @workgroup_size(256)
fn poly_add_kyber(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let total = params.batch_size * params.n;

    if (gid.x >= total) { return; }

    let a = input_a[gid.x];
    let b = input_b[gid.x];
    output[gid.x] = barrett_reduce_kyber(a + b);
}

// ============================================================================
// Polynomial Subtraction
// ============================================================================

@compute @workgroup_size(256)
fn poly_sub_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let total = params.batch_size * params.n;

    if (gid.x >= total) { return; }

    let a = input_a[gid.x];
    let b = input_b[gid.x];
    output[gid.x] = barrett_reduce_dilithium(a - b);
}

@compute @workgroup_size(256)
fn poly_sub_kyber(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let total = params.batch_size * params.n;

    if (gid.x >= total) { return; }

    let a = input_a[gid.x];
    let b = input_b[gid.x];
    output[gid.x] = barrett_reduce_kyber(a - b);
}

// ============================================================================
// Fused Forward NTT + Pointwise Mul + Inverse NTT (Polynomial Multiplication)
// For small batches where fusing operations reduces memory traffic
// ============================================================================

@compute @workgroup_size(256)
fn poly_mul_ntt_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let poly_idx = wid.x;
    let n = DILITHIUM_N;

    if (poly_idx >= params.batch_size) { return; }

    let base_idx = poly_idx * n;

    // This would implement fused NTT(a) * NTT(b) followed by INTT
    // For now, use separate kernels for clarity

    // Load first polynomial
    if (lid.x < n) {
        shared_data[lid.x] = input_a[base_idx + lid.x];
    }
    workgroupBarrier();

    // Forward NTT on a (inline)
    var len = n / 2u;
    var k = 1u;
    while (len >= 1u) {
        var start = lid.x;
        while (start < n / 2u) {
            let group = start / len;
            let idx_in_group = start % len;
            let idx0 = group * len * 2u + idx_in_group;
            let idx1 = idx0 + len;
            let zeta = zetas[k + group];
            let result = butterfly_ct_dilithium(shared_data[idx0], shared_data[idx1], zeta);
            shared_data[idx0] = result[0];
            shared_data[idx1] = result[1];
            start += WORKGROUP_SIZE;
        }
        workgroupBarrier();
        len = len / 2u;
        k = k * 2u;
    }

    // Pointwise multiply with NTT(b) (assume b is already in NTT domain)
    if (lid.x < n) {
        let b_ntt = input_b[base_idx + lid.x];  // Assume already transformed
        shared_data[lid.x] = mod_mul_dilithium(shared_data[lid.x], b_ntt);
    }
    workgroupBarrier();

    // Inverse NTT
    len = 1u;
    k = n / 2u;
    while (len < n) {
        var start = lid.x;
        while (start < n / 2u) {
            let group = start / len;
            let idx_in_group = start % len;
            let idx0 = group * len * 2u + idx_in_group;
            let idx1 = idx0 + len;
            let zeta = zetas[k + group];  // Inverse zetas
            let result = butterfly_gs_dilithium(shared_data[idx0], shared_data[idx1], zeta);
            shared_data[idx0] = result[0];
            shared_data[idx1] = result[1];
            start += WORKGROUP_SIZE;
        }
        workgroupBarrier();
        len = len * 2u;
        k = k / 2u;
    }

    // Scale and write
    let n_inv = 8347681;
    if (lid.x < n) {
        output[base_idx + lid.x] = full_reduce_dilithium(
            mod_mul_dilithium(shared_data[lid.x], n_inv)
        );
    }
}

// ============================================================================
// Vectorized Operations (process 4 coefficients at once)
// ============================================================================

@compute @workgroup_size(256)
fn ntt_pointwise_mul_dilithium_vec4(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let n = params.n;
    let total = params.batch_size * n / 4u;

    if (gid.x >= total) { return; }

    let base = gid.x * 4u;

    for (var i = 0u; i < 4u; i++) {
        let idx = base + i;
        let a = input_a[idx];
        let b = input_b[idx];
        output[idx] = mod_mul_dilithium(a, b);
    }
}

// ============================================================================
// Batch Reduction (reduce all coefficients to canonical form)
// ============================================================================

@compute @workgroup_size(256)
fn ntt_reduce_dilithium(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let total = params.batch_size * params.n;

    if (gid.x >= total) { return; }

    data[gid.x] = full_reduce_dilithium(data[gid.x]);
}

@compute @workgroup_size(256)
fn ntt_reduce_kyber(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let total = params.batch_size * params.n;

    if (gid.x >= total) { return; }

    data[gid.x] = barrett_reduce_kyber(data[gid.x]);
}
