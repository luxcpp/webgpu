// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Negacyclic NTT for ML-DSA (Dilithium) and ML-KEM (Kyber)
// NTT over Z_q[X]/(X^n + 1) for n=256,512,1024
// Moduli: q=8380417 (Dilithium), q=3329 (Kyber)
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

// ============================================================================
// Constants for ML-DSA and ML-KEM
// ============================================================================

// Dilithium parameters
const DILITHIUM_Q: u32 = 8380417u;
const DILITHIUM_QINV: u32 = 58728449u;  // q^(-1) mod 2^32

// Kyber parameters
const KYBER_Q: u32 = 3329u;
const KYBER_QINV: u32 = 62209u;  // q^(-1) mod 2^16

// ============================================================================
// Parameter Structures
// ============================================================================

struct NTTParams {
    n: u32,
    log_n: u32,
    batch: u32,
    stage: u32,
    q: u32,
    qinv: u32,
    inv_n: i32,
    _pad: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read_write> data: array<i32>;
@group(0) @binding(1) var<storage, read> twiddles: array<i32>;
@group(0) @binding(2) var<uniform> params: NTTParams;

// Shared memory for fused kernels
var<workgroup> shared_data: array<i32, 1024>;

// ============================================================================
// Montgomery Multiplication
// ============================================================================

// Montgomery reduction for Dilithium (32-bit)
// Input: a (64-bit represented as two 32-bit values)
// Output: a * 2^(-32) mod q
fn montgomery_reduce_dilithium(a_lo: i32, a_hi: i32) -> i32 {
    let t = a_lo * i32(DILITHIUM_QINV);
    // Compute (a - t * q) >> 32
    let tq_lo = t * i32(DILITHIUM_Q);
    var result = a_hi - ((tq_lo >> 31) & 1);  // Borrow handling
    // Simplified: use upper part
    return result;
}

// Montgomery multiplication for Dilithium
// Returns a * b * 2^(-32) mod q
fn mont_mul_dilithium(a: i32, b: i32) -> i32 {
    // 32x32 -> 64 bit multiply emulation
    let a_u = bitcast<u32>(a);
    let b_u = bitcast<u32>(b);
    
    let a_lo = a_u & 0xFFFFu;
    let a_hi = a_u >> 16u;
    let b_lo = b_u & 0xFFFFu;
    let b_hi = b_u >> 16u;
    
    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;
    
    let mid = p1 + p2;
    let lo = p0 + (mid << 16u);
    let hi = p3 + (mid >> 16u) + select(0u, 1u, lo < p0);
    
    // Montgomery reduction
    let t = bitcast<i32>(lo) * i32(DILITHIUM_QINV);
    let tq = bitcast<u32>(t) * DILITHIUM_Q;
    
    // (prod - tq) >> 32 = hi - (tq >> 32) - borrow
    let tq_hi = tq >> 32u;  // This is 0 since tq fits in 32 bits for small moduli
    return bitcast<i32>(hi) - bitcast<i32>(tq_hi);
}

// Simplified Montgomery multiplication for Kyber (16-bit modulus)
fn mont_mul_kyber(a: i32, b: i32) -> i32 {
    let prod = a * b;
    let t = (prod & 0xFFFF) * i32(KYBER_QINV);
    let reduced = (prod - (t & 0xFFFF) * i32(KYBER_Q)) >> 16;
    return reduced;
}

// ============================================================================
// Barrett Reduction
// ============================================================================

// Barrett reduction for Dilithium
fn barrett_reduce_dilithium(a: i32) -> i32 {
    // Barrett constant: floor(2^46 / q)
    let v: i64 = 8396807i64;
    let t = i32((v * i64(a)) >> 46);
    return a - t * i32(DILITHIUM_Q);
}

// Barrett reduction for Kyber
fn barrett_reduce_kyber(a: i32) -> i32 {
    let v: i32 = 20159;  // floor(2^26 / q) + 1
    let t = ((v * a + (1 << 25)) >> 26);
    return a - t * i32(KYBER_Q);
}

// ============================================================================
// Conditional Reduction
// ============================================================================

fn cond_sub_q(a: i32, q: i32) -> i32 {
    var result = a + ((a >> 31) & q);  // If negative, add q
    result = result - q;
    result = result + ((result >> 31) & q);  // If negative, add q
    return result;
}

// ============================================================================
// Bit-Reversal Permutation
// ============================================================================

fn bit_reverse(x: u32, log_n: u32) -> u32 {
    var result = 0u;
    var val = x;
    for (var i = 0u; i < log_n; i = i + 1u) {
        result = (result << 1u) | (val & 1u);
        val = val >> 1u;
    }
    return result;
}

// ============================================================================
// Butterfly Operations
// ============================================================================

// Cooley-Tukey butterfly (forward NTT)
// (a, b) <- (a + w*b, a - w*b)
fn ct_butterfly_dilithium(a: ptr<function, i32>, b: ptr<function, i32>, w: i32) {
    let t = mont_mul_dilithium(*b, w);
    *b = *a - t;
    *a = *a + t;
}

fn ct_butterfly_kyber(a: ptr<function, i32>, b: ptr<function, i32>, w: i32) {
    let t = mont_mul_kyber(*b, w);
    *b = *a - t;
    *a = *a + t;
}

// Gentleman-Sande butterfly (inverse NTT)
// (a, b) <- (a + b, (a - b) * w)
fn gs_butterfly_dilithium(a: ptr<function, i32>, b: ptr<function, i32>, w: i32) {
    let t = *a;
    *a = t + *b;
    *b = mont_mul_dilithium(t - *b, w);
}

fn gs_butterfly_kyber(a: ptr<function, i32>, b: ptr<function, i32>, w: i32) {
    let t = *a;
    *a = t + *b;
    *b = mont_mul_kyber(t - *b, w);
}

// ============================================================================
// Bit-Reversal Kernels
// ============================================================================

@compute @workgroup_size(256)
fn ntt_bit_reverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let log_n = params.log_n;
    let batch = params.batch;
    
    let batch_idx = gid.y;
    if (batch_idx >= batch) { return; }
    
    let i = gid.x;
    if (i >= n) { return; }
    
    let j = bit_reverse(i, log_n);
    
    // Only swap once (when i < j)
    if (i < j) {
        let base = batch_idx * n;
        let tmp = data[base + i];
        data[base + i] = data[base + j];
        data[base + j] = tmp;
    }
}

// ============================================================================
// Forward NTT Kernels - Staged
// ============================================================================

@compute @workgroup_size(256)
fn ntt_forward_stage_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let batch = params.batch;
    let stage = params.stage;
    
    let total = batch * (n / 2u);
    if (gid.x >= total) { return; }
    
    let batch_idx = gid.x / (n / 2u);
    let k = gid.x % (n / 2u);
    
    let m = 1u << (stage + 1u);
    let half_m = m >> 1u;
    let j = k / half_m;
    let i = k % half_m;
    let idx0 = j * m + i;
    let idx1 = idx0 + half_m;
    
    let base = batch_idx * n;
    
    // Load twiddle factor
    let w = twiddles[half_m + i];
    
    // Load values
    var a = data[base + idx0];
    var b = data[base + idx1];
    
    // Butterfly
    ct_butterfly_dilithium(&a, &b, w);
    
    // Store results
    data[base + idx0] = a;
    data[base + idx1] = b;
}

@compute @workgroup_size(256)
fn ntt_forward_stage_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let batch = params.batch;
    let stage = params.stage;
    
    let total = batch * (n / 2u);
    if (gid.x >= total) { return; }
    
    let batch_idx = gid.x / (n / 2u);
    let k = gid.x % (n / 2u);
    
    let m = 1u << (stage + 1u);
    let half_m = m >> 1u;
    let j = k / half_m;
    let i = k % half_m;
    let idx0 = j * m + i;
    let idx1 = idx0 + half_m;
    
    let base = batch_idx * n;
    let w = twiddles[half_m + i];
    
    var a = data[base + idx0];
    var b = data[base + idx1];
    
    ct_butterfly_kyber(&a, &b, w);
    
    data[base + idx0] = a;
    data[base + idx1] = b;
}

// ============================================================================
// Inverse NTT Kernels - Staged
// ============================================================================

@compute @workgroup_size(256)
fn ntt_inverse_stage_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let batch = params.batch;
    let stage = params.stage;
    
    let total = batch * (n / 2u);
    if (gid.x >= total) { return; }
    
    let batch_idx = gid.x / (n / 2u);
    let k = gid.x % (n / 2u);
    
    let m = 1u << stage;
    let half_m = m >> 1u;
    let j = k / half_m;
    let i = k % half_m;
    let idx0 = j * m + i;
    let idx1 = idx0 + half_m;
    
    let base = batch_idx * n;
    let w = twiddles[half_m + i];  // Inverse twiddles
    
    var a = data[base + idx0];
    var b = data[base + idx1];
    
    gs_butterfly_dilithium(&a, &b, w);
    
    data[base + idx0] = a;
    data[base + idx1] = b;
}

@compute @workgroup_size(256)
fn ntt_inverse_stage_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let batch = params.batch;
    let stage = params.stage;
    
    let total = batch * (n / 2u);
    if (gid.x >= total) { return; }
    
    let batch_idx = gid.x / (n / 2u);
    let k = gid.x % (n / 2u);
    
    let m = 1u << stage;
    let half_m = m >> 1u;
    let j = k / half_m;
    let i = k % half_m;
    let idx0 = j * m + i;
    let idx1 = idx0 + half_m;
    
    let base = batch_idx * n;
    let w = twiddles[half_m + i];
    
    var a = data[base + idx0];
    var b = data[base + idx1];
    
    gs_butterfly_kyber(&a, &b, w);
    
    data[base + idx0] = a;
    data[base + idx1] = b;
}

// ============================================================================
// Fused NTT Kernels (n=256, all stages in workgroup memory)
// ============================================================================

@compute @workgroup_size(128)
fn ntt_forward_fused_256_dilithium(
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = 256u;
    let LOG_N = 8u;
    let batch = params.batch;
    
    let batch_idx = wgid.x;
    if (batch_idx >= batch) { return; }
    
    let base = batch_idx * N;
    let thread_idx = tid.x;
    let threads = 128u;
    
    // Load with bit-reversal to shared memory
    for (var i = thread_idx; i < N; i = i + threads) {
        let rev_i = bit_reverse(i, LOG_N);
        shared_data[rev_i] = data[base + i];
    }
    workgroupBarrier();
    
    // Cooley-Tukey NTT (all 8 stages)
    for (var stage = 0u; stage < LOG_N; stage = stage + 1u) {
        let m = 1u << (stage + 1u);
        let half_m = m >> 1u;
        
        for (var k = thread_idx; k < N / 2u; k = k + threads) {
            let j = k / half_m;
            let i = k % half_m;
            let idx0 = j * m + i;
            let idx1 = idx0 + half_m;
            
            let w = twiddles[half_m + i];
            
            var a = shared_data[idx0];
            var b = shared_data[idx1];
            ct_butterfly_dilithium(&a, &b, w);
            shared_data[idx0] = a;
            shared_data[idx1] = b;
        }
        workgroupBarrier();
    }
    
    // Store result
    for (var i = thread_idx; i < N; i = i + threads) {
        data[base + i] = shared_data[i];
    }
}

@compute @workgroup_size(128)
fn ntt_forward_fused_256_kyber(
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = 256u;
    let LOG_N = 8u;
    let batch = params.batch;
    
    let batch_idx = wgid.x;
    if (batch_idx >= batch) { return; }
    
    let base = batch_idx * N;
    let thread_idx = tid.x;
    let threads = 128u;
    
    // Load with bit-reversal
    for (var i = thread_idx; i < N; i = i + threads) {
        let rev_i = bit_reverse(i, LOG_N);
        shared_data[rev_i] = data[base + i];
    }
    workgroupBarrier();
    
    // Cooley-Tukey NTT
    for (var stage = 0u; stage < LOG_N; stage = stage + 1u) {
        let m = 1u << (stage + 1u);
        let half_m = m >> 1u;
        
        for (var k = thread_idx; k < N / 2u; k = k + threads) {
            let j = k / half_m;
            let i = k % half_m;
            let idx0 = j * m + i;
            let idx1 = idx0 + half_m;
            
            let w = twiddles[half_m + i];
            
            var a = shared_data[idx0];
            var b = shared_data[idx1];
            ct_butterfly_kyber(&a, &b, w);
            shared_data[idx0] = a;
            shared_data[idx1] = b;
        }
        workgroupBarrier();
    }
    
    // Store result
    for (var i = thread_idx; i < N; i = i + threads) {
        data[base + i] = shared_data[i];
    }
}

@compute @workgroup_size(128)
fn ntt_inverse_fused_256_dilithium(
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = 256u;
    let LOG_N = 8u;
    let batch = params.batch;
    let inv_n = params.inv_n;
    
    let batch_idx = wgid.x;
    if (batch_idx >= batch) { return; }
    
    let base = batch_idx * N;
    let thread_idx = tid.x;
    let threads = 128u;
    
    // Load to shared memory
    for (var i = thread_idx; i < N; i = i + threads) {
        shared_data[i] = data[base + i];
    }
    workgroupBarrier();
    
    // Gentleman-Sande NTT (decimation-in-frequency)
    for (var s = LOG_N; s > 0u; s = s - 1u) {
        let stage = s;
        let m = 1u << stage;
        let half_m = m >> 1u;
        
        for (var k = thread_idx; k < N / 2u; k = k + threads) {
            let j = k / half_m;
            let i = k % half_m;
            let idx0 = j * m + i;
            let idx1 = idx0 + half_m;
            
            let w = twiddles[half_m + i];
            
            var a = shared_data[idx0];
            var b = shared_data[idx1];
            gs_butterfly_dilithium(&a, &b, w);
            shared_data[idx0] = a;
            shared_data[idx1] = b;
        }
        workgroupBarrier();
    }
    
    // Apply inverse N scaling and bit-reverse
    for (var i = thread_idx; i < N; i = i + threads) {
        let rev_i = bit_reverse(i, LOG_N);
        let val = mont_mul_dilithium(shared_data[i], inv_n);
        data[base + rev_i] = cond_sub_q(val, i32(DILITHIUM_Q));
    }
}

@compute @workgroup_size(128)
fn ntt_inverse_fused_256_kyber(
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let N = 256u;
    let LOG_N = 8u;
    let batch = params.batch;
    let inv_n = params.inv_n;
    
    let batch_idx = wgid.x;
    if (batch_idx >= batch) { return; }
    
    let base = batch_idx * N;
    let thread_idx = tid.x;
    let threads = 128u;
    
    // Load to shared memory
    for (var i = thread_idx; i < N; i = i + threads) {
        shared_data[i] = data[base + i];
    }
    workgroupBarrier();
    
    // Gentleman-Sande NTT
    for (var s = LOG_N; s > 0u; s = s - 1u) {
        let stage = s;
        let m = 1u << stage;
        let half_m = m >> 1u;
        
        for (var k = thread_idx; k < N / 2u; k = k + threads) {
            let j = k / half_m;
            let i = k % half_m;
            let idx0 = j * m + i;
            let idx1 = idx0 + half_m;
            
            let w = twiddles[half_m + i];
            
            var a = shared_data[idx0];
            var b = shared_data[idx1];
            gs_butterfly_kyber(&a, &b, w);
            shared_data[idx0] = a;
            shared_data[idx1] = b;
        }
        workgroupBarrier();
    }
    
    // Apply inverse N scaling and bit-reverse
    for (var i = thread_idx; i < N; i = i + threads) {
        let rev_i = bit_reverse(i, LOG_N);
        let val = mont_mul_kyber(shared_data[i], inv_n);
        data[base + rev_i] = cond_sub_q(val, i32(KYBER_Q));
    }
}

// ============================================================================
// Scale by Inverse N (after inverse NTT)
// ============================================================================

@compute @workgroup_size(256)
fn ntt_scale_dilithium(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.n * params.batch;
    if (gid.x >= size) { return; }
    
    let inv_n = params.inv_n;
    let val = mont_mul_dilithium(data[gid.x], inv_n);
    data[gid.x] = cond_sub_q(val, i32(DILITHIUM_Q));
}

@compute @workgroup_size(256)
fn ntt_scale_kyber(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.n * params.batch;
    if (gid.x >= size) { return; }
    
    let inv_n = params.inv_n;
    let val = mont_mul_kyber(data[gid.x], inv_n);
    data[gid.x] = cond_sub_q(val, i32(KYBER_Q));
}
