// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Goldilocks Field Arithmetic in WGSL
// p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
//
// This prime has special structure enabling efficient reduction:
//   2^64 ≡ 2^32 - 1 (mod p)
//
// Properties:
// - Order: p - 1 = 2^32 * (2^32 - 1), enabling NTT up to 2^32
// - Generator: 7 is a primitive root
// - Two-adicity: 32 (excellent for NTT)
//
// Part of the Lux Network cryptography library
// Used in: Plonky2, STARK proofs, FRI commitments

// =============================================================================
// Goldilocks Field Constants
// =============================================================================

// p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
const GL_P_LO: u32 = 0x00000001u;
const GL_P_HI: u32 = 0xFFFFFFFFu;

// 2^32 - 1 = 0xFFFFFFFF (for reduction: 2^64 ≡ this value mod p)
const GL_EPSILON: u32 = 0xFFFFFFFFu;

// Primitive root g = 7
const GL_GENERATOR_LO: u32 = 7u;
const GL_GENERATOR_HI: u32 = 0u;

// Two-adicity: 32
const GL_TWO_ADICITY: u32 = 32u;

// 2^32-th root of unity: w = g^((p-1) / 2^32)
// Precomputed: 0x185629dcda58878c
const GL_ROOT_2_32_LO: u32 = 0xda58878cu;
const GL_ROOT_2_32_HI: u32 = 0x185629dcu;

// =============================================================================
// 64-bit Type: vec2<u32> for (lo, hi)
// =============================================================================
// We use vec2<u32> where x = lo (bits 0-31), y = hi (bits 32-63)
// This is more efficient than a struct in WGSL

alias GL = vec2<u32>;  // Goldilocks field element

fn gl_zero() -> GL {
    return GL(0u, 0u);
}

fn gl_one() -> GL {
    return GL(1u, 0u);
}

fn gl_from_u32(x: u32) -> GL {
    return GL(x, 0u);
}

fn gl_from_u64(lo: u32, hi: u32) -> GL {
    return GL(lo, hi);
}

fn gl_modulus() -> GL {
    return GL(GL_P_LO, GL_P_HI);
}

// =============================================================================
// Comparison Operations
// =============================================================================

fn gl_eq(a: GL, b: GL) -> bool {
    return a.x == b.x && a.y == b.y;
}

fn gl_is_zero(a: GL) -> bool {
    return a.x == 0u && a.y == 0u;
}

// a >= b (unsigned)
fn gl_gte(a: GL, b: GL) -> bool {
    return (a.y > b.y) || (a.y == b.y && a.x >= b.x);
}

// a < b (unsigned)
fn gl_lt(a: GL, b: GL) -> bool {
    return (a.y < b.y) || (a.y == b.y && a.x < b.x);
}

// =============================================================================
// Basic Arithmetic (Non-Modular)
// =============================================================================

// Add two 64-bit numbers, returns (result, carry)
fn gl_add_with_carry(a: GL, b: GL) -> vec3<u32> {
    let lo = a.x + b.x;
    let carry1 = select(0u, 1u, lo < a.x);
    let hi = a.y + b.y + carry1;
    let carry2 = select(0u, 1u, hi < a.y || (carry1 == 1u && hi == a.y));
    return vec3<u32>(lo, hi, carry2);
}

// Subtract two 64-bit numbers, returns (result, borrow)
fn gl_sub_with_borrow(a: GL, b: GL) -> vec3<u32> {
    let borrow1 = select(0u, 1u, a.x < b.x);
    let lo = a.x - b.x;
    let hi_after_borrow = a.y - borrow1;
    let borrow2 = select(0u, 1u, a.y < b.y + borrow1);
    let hi = hi_after_borrow - b.y;
    return vec3<u32>(lo, hi, borrow2);
}

// =============================================================================
// Goldilocks Modular Reduction
// =============================================================================

// Reduce 64-bit value mod p using Goldilocks structure
// If a >= p, return a - p
fn gl_reduce_once(a: GL) -> GL {
    let p = gl_modulus();
    if (gl_gte(a, p)) {
        let sub = gl_sub_with_borrow(a, p);
        return GL(sub.x, sub.y);
    }
    return a;
}

// Reduce 96-bit value (a_lo, a_hi, a_top) mod p
// Uses: 2^64 ≡ 2^32 - 1 (mod p)
fn gl_reduce_96(a_lo: u32, a_hi: u32, a_top: u32) -> GL {
    // a_top * 2^64 ≡ a_top * (2^32 - 1) (mod p)
    //              = a_top * 2^32 - a_top

    // Start with (a_lo, a_hi)
    var lo = a_lo;
    var hi = a_hi;

    // Add a_top * 2^32 (shift left by 32)
    let old_hi = hi;
    hi = hi + a_top;
    let carry1 = select(0u, 1u, hi < old_hi);

    // Subtract a_top
    let borrow = select(0u, 1u, lo < a_top);
    lo = lo - a_top;
    hi = hi - borrow;

    // Handle carry from adding a_top * 2^32
    // carry1 * 2^64 ≡ carry1 * (2^32 - 1)
    if (carry1 != 0u) {
        let old_hi2 = hi;
        hi = hi + carry1;  // Add carry1 * 2^32
        let carry2 = select(0u, 1u, hi < old_hi2);

        let borrow2 = select(0u, 1u, lo < carry1);
        lo = lo - carry1;  // Subtract carry1
        hi = hi - borrow2;

        // Recursive correction if needed (at most once more)
        if (carry2 != 0u) {
            hi = hi + 1u;
            let borrow3 = select(0u, 1u, lo < 1u);
            lo = lo - 1u;
            hi = hi - borrow3;
        }
    }

    return gl_reduce_once(GL(lo, hi));
}

// Reduce 128-bit value mod p
// Uses repeated application of 2^64 ≡ 2^32 - 1
fn gl_reduce_128(w0: u32, w1: u32, w2: u32, w3: u32) -> GL {
    // First reduce (w2, w3) * 2^64
    // (w2, w3) * 2^64 ≡ (w2, w3) * (2^32 - 1)
    //                 = (w2 * 2^32 - w2) + (w3 * 2^64 - w3 * 2^32)
    //                 ≡ (w2 * 2^32 - w2) + (w3 * (2^32-1) - w3 * 2^32)
    //                 = w2 * 2^32 - w2 - w3 * 2^32 + w3 * 2^32 - w3
    //                 = w2 * 2^32 - w2 - w3

    // Simpler approach: reduce in stages
    // Stage 1: Reduce w3 * 2^96
    var r = gl_reduce_96(0u, 0u, w3);
    r = gl_reduce_96(r.x, r.y, 0u);  // Apply 2^32 shift from w3's position

    // Stage 2: Add w2 * 2^64 and reduce
    r = gl_reduce_96(r.x, r.y, w2);

    // Stage 3: Add (w0, w1) as base
    let sum = gl_add_with_carry(r, GL(w0, w1));

    if (sum.z != 0u) {
        // Overflow: reduce carry * 2^64 ≡ carry * (2^32 - 1)
        return gl_reduce_96(sum.x, sum.y, sum.z);
    }

    return gl_reduce_once(GL(sum.x, sum.y));
}

// =============================================================================
// Goldilocks Field Arithmetic
// =============================================================================

// Modular addition: (a + b) mod p
fn gl_add(a: GL, b: GL) -> GL {
    let sum = gl_add_with_carry(a, b);

    if (sum.z != 0u) {
        // Overflow occurred, reduce
        // carry * 2^64 ≡ carry * (2^32 - 1) = 2^32 - 1
        let adjusted = GL(sum.x + GL_EPSILON, sum.y);
        return gl_reduce_once(adjusted);
    }

    return gl_reduce_once(GL(sum.x, sum.y));
}

// Modular subtraction: (a - b) mod p
fn gl_sub(a: GL, b: GL) -> GL {
    let sub = gl_sub_with_borrow(a, b);

    if (sub.z != 0u) {
        // Underflow: add p back
        let p = gl_modulus();
        let adj = gl_add_with_carry(GL(sub.x, sub.y), p);
        return GL(adj.x, adj.y);
    }

    return GL(sub.x, sub.y);
}

// Modular negation: -a mod p = p - a
fn gl_neg(a: GL) -> GL {
    if (gl_is_zero(a)) {
        return a;
    }
    return gl_sub(gl_modulus(), a);
}

// 32x32 -> 64 bit multiplication
fn mul32_wide(a: u32, b: u32) -> GL {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;

    let mid = lh + hl;
    let mid_carry = select(0u, 0x10000u, mid < lh);

    let lo = ll + ((mid & 0xFFFFu) << 16u);
    let lo_carry = select(0u, 1u, lo < ll);
    let hi = hh + (mid >> 16u) + mid_carry + lo_carry;

    return GL(lo, hi);
}

// Modular multiplication: (a * b) mod p
fn gl_mul(a: GL, b: GL) -> GL {
    // Schoolbook multiplication to get 128-bit result
    // (a.y * 2^32 + a.x) * (b.y * 2^32 + b.x)
    // = a.y * b.y * 2^64 + (a.y * b.x + a.x * b.y) * 2^32 + a.x * b.x

    let p0 = mul32_wide(a.x, b.x);  // bits 0-63
    let p1 = mul32_wide(a.x, b.y);  // bits 32-95
    let p2 = mul32_wide(a.y, b.x);  // bits 32-95
    let p3 = mul32_wide(a.y, b.y);  // bits 64-127

    // Accumulate into 128-bit result (w0, w1, w2, w3)
    var w0 = p0.x;
    var w1 = p0.y;
    var w2 = p3.x;
    var w3 = p3.y;

    // Add p1 at 32-bit offset
    let sum1 = w1 + p1.x;
    let carry1 = select(0u, 1u, sum1 < w1);
    w1 = sum1;

    let sum2 = w2 + p1.y + carry1;
    let carry2 = select(0u, 1u, sum2 < w2 || (carry1 == 1u && sum2 == w2));
    w2 = sum2;

    w3 = w3 + carry2;

    // Add p2 at 32-bit offset
    let sum3 = w1 + p2.x;
    let carry3 = select(0u, 1u, sum3 < w1);
    w1 = sum3;

    let sum4 = w2 + p2.y + carry3;
    let carry4 = select(0u, 1u, sum4 < w2 || (carry3 == 1u && sum4 == w2));
    w2 = sum4;

    w3 = w3 + carry4;

    // Reduce 128-bit result mod p
    return gl_reduce_128(w0, w1, w2, w3);
}

// Modular squaring: a^2 mod p (slightly optimized)
fn gl_square(a: GL) -> GL {
    return gl_mul(a, a);
}

// Modular exponentiation: a^exp mod p (square-and-multiply)
fn gl_pow(base: GL, exp_lo: u32, exp_hi: u32) -> GL {
    var result = gl_one();
    var b = base;
    var e_lo = exp_lo;
    var e_hi = exp_hi;

    // Process low 32 bits
    for (var i = 0u; i < 32u; i++) {
        if ((e_lo & 1u) != 0u) {
            result = gl_mul(result, b);
        }
        b = gl_square(b);
        e_lo = e_lo >> 1u;
    }

    // Process high 32 bits
    for (var i = 0u; i < 32u; i++) {
        if ((e_hi & 1u) != 0u) {
            result = gl_mul(result, b);
        }
        b = gl_square(b);
        e_hi = e_hi >> 1u;
    }

    return result;
}

// Modular inverse using Fermat's little theorem: a^(-1) = a^(p-2)
fn gl_inv(a: GL) -> GL {
    // p - 2 = 0xFFFFFFFF00000001 - 2 = 0xFFFFFFFEFFFFFFFF
    return gl_pow(a, 0xFFFFFFFFu, 0xFFFFFFFEu);
}

// =============================================================================
// NTT Twiddle Factor Computation
// =============================================================================

// Get 2^k-th root of unity for NTT of size 2^k
fn gl_get_root_of_unity(k: u32) -> GL {
    // w_k = g^((p-1) / 2^k) = w_32^(2^(32-k))
    if (k > GL_TWO_ADICITY) {
        return gl_zero();  // Invalid
    }

    var root = GL(GL_ROOT_2_32_LO, GL_ROOT_2_32_HI);

    // Square (32 - k) times to get 2^k-th root
    for (var i = k; i < GL_TWO_ADICITY; i++) {
        root = gl_square(root);
    }

    return root;
}

// =============================================================================
// NTT Butterfly Operations
// =============================================================================

// Cooley-Tukey butterfly: forward NTT
// x0' = x0 + w * x1
// x1' = x0 - w * x1
fn gl_ct_butterfly(x0: ptr<function, GL>, x1: ptr<function, GL>, w: GL) {
    let t = gl_mul(*x1, w);
    let new_x1 = gl_sub(*x0, t);
    let new_x0 = gl_add(*x0, t);
    *x0 = new_x0;
    *x1 = new_x1;
}

// Gentleman-Sande butterfly: inverse NTT
// x0' = x0 + x1
// x1' = (x0 - x1) * w
fn gl_gs_butterfly(x0: ptr<function, GL>, x1: ptr<function, GL>, w: GL) {
    let t = gl_sub(*x0, *x1);
    let new_x0 = gl_add(*x0, *x1);
    let new_x1 = gl_mul(t, w);
    *x0 = new_x0;
    *x1 = new_x1;
}

// =============================================================================
// Kernel Bindings
// =============================================================================

struct GoldilocksParams {
    size: u32,       // Number of elements
    batch: u32,      // Batch size
    stage: u32,      // NTT stage (for staged kernels)
    log_n: u32,      // log2(N) for NTT
}

@group(0) @binding(0) var<storage, read_write> result: array<u32>;
@group(0) @binding(1) var<storage, read> input_a: array<u32>;
@group(0) @binding(2) var<storage, read> input_b: array<u32>;
@group(0) @binding(3) var<uniform> params: GoldilocksParams;

// =============================================================================
// Batch Field Operations
// =============================================================================

// Batch modular addition
@compute @workgroup_size(256)
fn gl_add_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }

    let idx = gid.x * 2u;
    let a = GL(input_a[idx], input_a[idx + 1u]);
    let b = GL(input_b[idx], input_b[idx + 1u]);

    let r = gl_add(a, b);

    result[idx] = r.x;
    result[idx + 1u] = r.y;
}

// Batch modular subtraction
@compute @workgroup_size(256)
fn gl_sub_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }

    let idx = gid.x * 2u;
    let a = GL(input_a[idx], input_a[idx + 1u]);
    let b = GL(input_b[idx], input_b[idx + 1u]);

    let r = gl_sub(a, b);

    result[idx] = r.x;
    result[idx + 1u] = r.y;
}

// Batch modular multiplication
@compute @workgroup_size(256)
fn gl_mul_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }

    let idx = gid.x * 2u;
    let a = GL(input_a[idx], input_a[idx + 1u]);
    let b = GL(input_b[idx], input_b[idx + 1u]);

    let r = gl_mul(a, b);

    result[idx] = r.x;
    result[idx + 1u] = r.y;
}

// Batch modular negation (unary)
@compute @workgroup_size(256)
fn gl_neg_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }

    let idx = gid.x * 2u;
    let a = GL(input_a[idx], input_a[idx + 1u]);

    let r = gl_neg(a);

    result[idx] = r.x;
    result[idx + 1u] = r.y;
}

// Batch modular squaring
@compute @workgroup_size(256)
fn gl_square_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }

    let idx = gid.x * 2u;
    let a = GL(input_a[idx], input_a[idx + 1u]);

    let r = gl_square(a);

    result[idx] = r.x;
    result[idx + 1u] = r.y;
}

// Batch modular inverse
@compute @workgroup_size(256)
fn gl_inv_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }

    let idx = gid.x * 2u;
    let a = GL(input_a[idx], input_a[idx + 1u]);

    let r = gl_inv(a);

    result[idx] = r.x;
    result[idx + 1u] = r.y;
}

// =============================================================================
// Vector Operations
// =============================================================================

// Scalar-vector multiplication: result = scalar * input_a
@compute @workgroup_size(256)
fn gl_scale_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }

    let idx = gid.x * 2u;
    let scalar = GL(input_b[0], input_b[1]);  // Scalar from first element
    let a = GL(input_a[idx], input_a[idx + 1u]);

    let r = gl_mul(a, scalar);

    result[idx] = r.x;
    result[idx + 1u] = r.y;
}

// Dot product (partial): accumulate in result buffer
@compute @workgroup_size(256)
fn gl_dot_partial_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }

    let idx = gid.x * 2u;
    let a = GL(input_a[idx], input_a[idx + 1u]);
    let b = GL(input_b[idx], input_b[idx + 1u]);

    // Element-wise product
    let prod = gl_mul(a, b);

    // Store for later reduction
    result[idx] = prod.x;
    result[idx + 1u] = prod.y;
}

// Hadamard (element-wise) product: same as gl_mul_kernel but named for clarity
@compute @workgroup_size(256)
fn gl_hadamard_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }

    let idx = gid.x * 2u;
    let a = GL(input_a[idx], input_a[idx + 1u]);
    let b = GL(input_b[idx], input_b[idx + 1u]);

    let r = gl_mul(a, b);

    result[idx] = r.x;
    result[idx + 1u] = r.y;
}

// =============================================================================
// NTT Kernels
// =============================================================================

@group(0) @binding(4) var<storage, read> twiddles: array<u32>;

var<workgroup> shared_data: array<u32, 1024>;  // 512 field elements max

// Forward NTT stage kernel
@compute @workgroup_size(256)
fn gl_ntt_forward_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = 1u << params.log_n;
    let batch = params.batch;
    let stage = params.stage;

    let total = batch * (N / 2u);
    if (gid.x >= total) { return; }

    let batch_idx = gid.x / (N / 2u);
    let k = gid.x % (N / 2u);

    // Butterfly indices
    let m = 1u << (stage + 1u);
    let half_m = m >> 1u;
    let j = k / half_m;
    let i = k % half_m;
    let idx0 = j * m + i;
    let idx1 = idx0 + half_m;

    let base = batch_idx * N * 2u;  // *2 for GL storage

    // Load values
    var x0 = GL(result[base + idx0 * 2u], result[base + idx0 * 2u + 1u]);
    var x1 = GL(result[base + idx1 * 2u], result[base + idx1 * 2u + 1u]);

    // Load twiddle factor
    let tw_idx = (half_m + i) * 2u;
    let w = GL(twiddles[tw_idx], twiddles[tw_idx + 1u]);

    // Cooley-Tukey butterfly
    gl_ct_butterfly(&x0, &x1, w);

    // Store results
    result[base + idx0 * 2u] = x0.x;
    result[base + idx0 * 2u + 1u] = x0.y;
    result[base + idx1 * 2u] = x1.x;
    result[base + idx1 * 2u + 1u] = x1.y;
}

// Inverse NTT stage kernel
@compute @workgroup_size(256)
fn gl_ntt_inverse_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = 1u << params.log_n;
    let batch = params.batch;
    let stage = params.stage;

    let total = batch * (N / 2u);
    if (gid.x >= total) { return; }

    let batch_idx = gid.x / (N / 2u);
    let k = gid.x % (N / 2u);

    // Butterfly indices (reversed compared to forward)
    let m = 1u << (params.log_n - stage);
    let half_m = m >> 1u;
    let j = k / half_m;
    let i = k % half_m;
    let idx0 = j * m + i;
    let idx1 = idx0 + half_m;

    let base = batch_idx * N * 2u;

    // Load values
    var x0 = GL(result[base + idx0 * 2u], result[base + idx0 * 2u + 1u]);
    var x1 = GL(result[base + idx1 * 2u], result[base + idx1 * 2u + 1u]);

    // Load inverse twiddle factor
    let tw_idx = (half_m + i) * 2u;
    let w = GL(twiddles[tw_idx], twiddles[tw_idx + 1u]);

    // Gentleman-Sande butterfly
    gl_gs_butterfly(&x0, &x1, w);

    // Store results
    result[base + idx0 * 2u] = x0.x;
    result[base + idx0 * 2u + 1u] = x0.y;
    result[base + idx1 * 2u] = x1.x;
    result[base + idx1 * 2u + 1u] = x1.y;
}

// Scale by inverse of N (final step of inverse NTT)
@compute @workgroup_size(256)
fn gl_ntt_scale_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size * params.batch;
    if (gid.x >= size) { return; }

    let idx = gid.x * 2u;

    // Load inverse of N from input_b
    let inv_n = GL(input_b[0], input_b[1]);
    let val = GL(result[idx], result[idx + 1u]);

    let scaled = gl_mul(val, inv_n);

    result[idx] = scaled.x;
    result[idx + 1u] = scaled.y;
}

// NTT pointwise multiplication (for polynomial multiplication)
@compute @workgroup_size(256)
fn gl_ntt_pointwise_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size * params.batch;
    if (gid.x >= size) { return; }

    let idx = gid.x * 2u;

    let a = GL(input_a[idx], input_a[idx + 1u]);
    let b = GL(input_b[idx], input_b[idx + 1u]);

    let prod = gl_mul(a, b);

    result[idx] = prod.x;
    result[idx + 1u] = prod.y;
}

// =============================================================================
// Twiddle Factor Generation
// =============================================================================

struct TwiddleParams {
    n: u32,          // NTT size
    log_n: u32,      // log2(N)
    inverse: u32,    // 0 = forward, 1 = inverse
    _pad: u32,
}

@group(1) @binding(0) var<storage, read_write> twiddle_out: array<u32>;
@group(1) @binding(1) var<uniform> twiddle_params: TwiddleParams;

// Generate twiddle factors for NTT
@compute @workgroup_size(256)
fn gl_gen_twiddles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = twiddle_params.n;
    if (gid.x >= n) { return; }

    // Get primitive n-th root of unity
    var w = gl_get_root_of_unity(twiddle_params.log_n);

    // For inverse NTT, use conjugate (w^-1 = w^(n-1))
    if (twiddle_params.inverse != 0u) {
        w = gl_inv(w);
    }

    // Compute w^gid.x
    let power = gl_pow(w, gid.x, 0u);

    let idx = gid.x * 2u;
    twiddle_out[idx] = power.x;
    twiddle_out[idx + 1u] = power.y;
}

// =============================================================================
// Polynomial Evaluation (for FRI)
// =============================================================================

// Evaluate polynomial at single point using Horner's method
// Polynomial coefficients in input_a, point in input_b[0:1]
@compute @workgroup_size(256)
fn gl_poly_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.batch) { return; }

    let point = GL(input_b[gid.x * 2u], input_b[gid.x * 2u + 1u]);
    let deg = params.size;  // Degree + 1

    var acc = gl_zero();

    // Horner's method: acc = a[n-1] + x*(a[n-2] + x*(...))
    for (var i = deg; i > 0u; i--) {
        let coeff_idx = (i - 1u) * 2u;
        let coeff = GL(input_a[coeff_idx], input_a[coeff_idx + 1u]);
        acc = gl_mul(acc, point);
        acc = gl_add(acc, coeff);
    }

    let out_idx = gid.x * 2u;
    result[out_idx] = acc.x;
    result[out_idx + 1u] = acc.y;
}

// =============================================================================
// Reduction Kernel (for dot product, etc.)
// =============================================================================

// Parallel reduction sum
@compute @workgroup_size(256)
fn gl_reduce_sum(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let size = params.size;

    // Load into shared memory
    var local_val = gl_zero();
    if (gid.x < size) {
        let idx = gid.x * 2u;
        local_val = GL(input_a[idx], input_a[idx + 1u]);
    }

    shared_data[tid * 2u] = local_val.x;
    shared_data[tid * 2u + 1u] = local_val.y;

    workgroupBarrier();

    // Tree reduction
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            let a = GL(shared_data[tid * 2u], shared_data[tid * 2u + 1u]);
            let b = GL(shared_data[(tid + stride) * 2u], shared_data[(tid + stride) * 2u + 1u]);
            let sum = gl_add(a, b);
            shared_data[tid * 2u] = sum.x;
            shared_data[tid * 2u + 1u] = sum.y;
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Thread 0 writes result
    if (tid == 0u) {
        let out_idx = wid.x * 2u;
        result[out_idx] = shared_data[0];
        result[out_idx + 1u] = shared_data[1];
    }
}

// =============================================================================
// FRI-Specific Operations
// =============================================================================

// FRI fold operation: fold(f(x), f(-x), alpha) = f(x) + alpha * (f(x) - f(-x)) / (2x)
@compute @workgroup_size(256)
fn gl_fri_fold(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.size;  // Input size
    let half_n = n / 2u;

    if (gid.x >= half_n) { return; }

    let idx = gid.x * 2u;
    let idx_neg = (gid.x + half_n) * 2u;

    // f(x) and f(-x)
    let fx = GL(input_a[idx], input_a[idx + 1u]);
    let fnx = GL(input_a[idx_neg], input_a[idx_neg + 1u]);

    // alpha (folding random)
    let alpha = GL(input_b[0], input_b[1]);

    // x value (from twiddles or precomputed)
    let x = GL(twiddles[idx], twiddles[idx + 1u]);
    let two_x = gl_add(x, x);
    let inv_2x = gl_inv(two_x);

    // fold = (fx + fnx)/2 + alpha * (fx - fnx) / (2x)
    let sum = gl_add(fx, fnx);
    let diff = gl_sub(fx, fnx);

    // Even part: (fx + fnx) / 2
    let two = GL(2u, 0u);
    let inv_two = gl_inv(two);
    let even = gl_mul(sum, inv_two);

    // Odd part: alpha * (fx - fnx) / (2x)
    let odd = gl_mul(gl_mul(alpha, diff), inv_2x);

    // Combined
    let folded = gl_add(even, odd);

    result[idx] = folded.x;
    result[idx + 1u] = folded.y;
}
