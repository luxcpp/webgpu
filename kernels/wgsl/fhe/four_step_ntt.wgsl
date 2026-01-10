// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// four_step_ntt.wgsl - Four-step NTT algorithm
//
// Implements the four-step FFT/NTT algorithm for large transforms:
// 1. Row-wise NTT on N1 x N2 matrix
// 2. Multiply by twiddle factors (inter-stage)
// 3. Transpose
// 4. Column-wise NTT (rows after transpose)
//
// Optimized for transforms too large for single workgroup shared memory.

// ============================================================================
// 64-bit integer emulation
// ============================================================================

struct u64 {
    lo: u32,
    hi: u32,
}

fn u64_zero() -> u64 {
    return u64(0u, 0u);
}

fn u64_from_u32(x: u32) -> u64 {
    return u64(x, 0u);
}

fn u64_add(a: u64, b: u64) -> u64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    let hi = a.hi + b.hi + carry;
    return u64(lo, hi);
}

fn u64_sub(a: u64, b: u64) -> u64 {
    let borrow = select(0u, 1u, a.lo < b.lo);
    let lo = a.lo - b.lo;
    let hi = a.hi - b.hi - borrow;
    return u64(lo, hi);
}

fn u64_mul(a: u64, b: u64) -> u64 {
    let a_lo = a.lo & 0xFFFFu;
    let a_hi = a.lo >> 16u;
    let b_lo = b.lo & 0xFFFFu;
    let b_hi = b.lo >> 16u;

    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;

    let mid = p1 + p2;
    let lo = p0 + ((mid & 0xFFFFu) << 16u);
    let carry = select(0u, 1u, lo < p0);
    let hi = p3 + (mid >> 16u) + carry + a.lo * b.hi + a.hi * b.lo;

    return u64(lo, hi);
}

fn u64_ge(a: u64, b: u64) -> bool {
    if (a.hi != b.hi) {
        return a.hi > b.hi;
    }
    return a.lo >= b.lo;
}

fn u64_mulhi(a: u64, b: u64) -> u64 {
    let a0 = a.lo & 0xFFFFu;
    let a1 = a.lo >> 16u;
    let a2 = a.hi & 0xFFFFu;
    let a3 = a.hi >> 16u;
    let b0 = b.lo & 0xFFFFu;
    let b1 = b.lo >> 16u;
    let b2 = b.hi & 0xFFFFu;
    let b3 = b.hi >> 16u;

    var acc = u64_zero();
    acc = u64_add(acc, u64_from_u32(a2 * b2));
    acc = u64_add(acc, u64_from_u32(a3 * b1));
    acc = u64_add(acc, u64_from_u32(a1 * b3));
    let hi_lo = acc.lo + a3 * b2 + a2 * b3;
    let hi_hi = acc.hi + a3 * b3;

    return u64(hi_lo, hi_hi);
}

// ============================================================================
// Modular arithmetic
// ============================================================================

fn mod_add(a: u64, b: u64, Q: u64) -> u64 {
    var result = u64_add(a, b);
    if (u64_ge(result, Q)) {
        result = u64_sub(result, Q);
    }
    return result;
}

fn mod_sub(a: u64, b: u64, Q: u64) -> u64 {
    if (u64_ge(a, b)) {
        return u64_sub(a, b);
    }
    return u64_sub(u64_add(a, Q), b);
}

fn barrett_mul(a: u64, b: u64, Q: u64, mu: u64) -> u64 {
    let product = u64_mul(a, b);
    let q_approx = u64_mulhi(product, mu);
    let qQ = u64_mul(q_approx, Q);
    var result = u64_sub(product, qQ);
    if (u64_ge(result, Q)) {
        result = u64_sub(result, Q);
    }
    if (u64_ge(result, Q)) {
        result = u64_sub(result, Q);
    }
    return result;
}

// ============================================================================
// Shared memory for small NTT
// ============================================================================

const WG_SIZE: u32 = 256u;
const SHARED_SIZE: u32 = 512u;

var<workgroup> shared_data: array<u64, 512>;
var<workgroup> shared_twiddles: array<u64, 256>;

// ============================================================================
// Buffers
// ============================================================================

struct FourStepParams {
    N: u32,              // Total transform size
    N1: u32,             // Row dimension
    N2: u32,             // Column dimension
    log_N1: u32,
    log_N2: u32,
    prime_idx: u32,
    is_inverse: u32,
    num_polys: u32,      // Batch size
}

@group(0) @binding(0) var<uniform> params: FourStepParams;
@group(0) @binding(1) var<storage, read_write> data: array<u64>;
@group(0) @binding(2) var<storage, read> primes: array<u64>;
@group(0) @binding(3) var<storage, read> barrett_mu: array<u64>;

// Twiddle factors
@group(0) @binding(4) var<storage, read> row_twiddles: array<u64>;    // N1 twiddles
@group(0) @binding(5) var<storage, read> col_twiddles: array<u64>;    // N2 twiddles
@group(0) @binding(6) var<storage, read> inter_twiddles: array<u64>;  // N1*N2 twiddles

// Transpose buffer
@group(0) @binding(7) var<storage, read_write> transpose_buf: array<u64>;

// ============================================================================
// NTT Butterflies
// ============================================================================

// Cooley-Tukey butterfly (DIF - decimation in frequency)
fn ct_butterfly(a: ptr<function, u64>, b: ptr<function, u64>, w: u64, Q: u64, mu: u64) {
    let u = *a;
    let v = barrett_mul(*b, w, Q, mu);
    *a = mod_add(u, v, Q);
    *b = mod_sub(u, v, Q);
}

// Gentleman-Sande butterfly (DIT - decimation in time)
fn gs_butterfly(a: ptr<function, u64>, b: ptr<function, u64>, w: u64, Q: u64, mu: u64) {
    let u = *a;
    let v = *b;
    *a = mod_add(u, v, Q);
    let diff = mod_sub(u, v, Q);
    *b = barrett_mul(diff, w, Q, mu);
}

// ============================================================================
// Step 1: Row-wise NTT
// ============================================================================

@compute @workgroup_size(256)
fn four_step_row_ntt(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_id = lid.x;
    let row_idx = wid.x;
    let poly_idx = wid.y;

    let N1 = params.N1;
    let N2 = params.N2;
    let log_N2 = params.log_N2;
    let prime_idx = params.prime_idx;

    if (row_idx >= N1 || N2 > SHARED_SIZE) {
        return;
    }

    let Q = primes[prime_idx];
    let mu = barrett_mu[prime_idx];

    // Load row into shared memory
    let base_idx = poly_idx * params.N + row_idx * N2;
    for (var i = local_id; i < N2; i += WG_SIZE) {
        shared_data[i] = data[base_idx + i];
    }

    // Load twiddles
    let half_N2 = N2 >> 1u;
    for (var i = local_id; i < half_N2; i += WG_SIZE) {
        shared_twiddles[i] = row_twiddles[i];
    }

    workgroupBarrier();

    // Perform NTT using Cooley-Tukey (DIF)
    var m = N2;
    var stage = 0u;

    while (m > 1u) {
        let half_m = m >> 1u;
        let num_groups = N2 / m;

        for (var k = local_id; k < N2 >> 1u; k += WG_SIZE) {
            let group = k / half_m;
            let j = k % half_m;
            let idx0 = group * m + j;
            let idx1 = idx0 + half_m;

            // Twiddle index for this stage
            let twiddle_idx = j * num_groups;
            let w = select(
                shared_twiddles[twiddle_idx % half_N2],
                row_twiddles[twiddle_idx],
                twiddle_idx >= half_N2
            );

            var a = shared_data[idx0];
            var b = shared_data[idx1];
            ct_butterfly(&a, &b, w, Q, mu);
            shared_data[idx0] = a;
            shared_data[idx1] = b;
        }

        workgroupBarrier();
        m = half_m;
        stage += 1u;
    }

    // Bit-reverse permutation and store back
    for (var i = local_id; i < N2; i += WG_SIZE) {
        var rev = 0u;
        var temp = i;
        for (var b = 0u; b < log_N2; b++) {
            rev = (rev << 1u) | (temp & 1u);
            temp = temp >> 1u;
        }
        if (i < rev) {
            let tmp = shared_data[i];
            shared_data[i] = shared_data[rev];
            shared_data[rev] = tmp;
        }
    }

    workgroupBarrier();

    // Store results
    for (var i = local_id; i < N2; i += WG_SIZE) {
        data[base_idx + i] = shared_data[i];
    }
}

// ============================================================================
// Step 2: Multiply by inter-stage twiddle factors
// ============================================================================

@compute @workgroup_size(256)
fn four_step_twiddle_multiply(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let poly_idx = gid.y;

    let N = params.N;
    let N1 = params.N1;
    let N2 = params.N2;

    if (idx >= N) {
        return;
    }

    let i = idx / N2;  // Row
    let j = idx % N2;  // Column

    let prime_idx = params.prime_idx;
    let Q = primes[prime_idx];
    let mu = barrett_mu[prime_idx];

    // Twiddle factor: omega^(i*j)
    let twiddle = inter_twiddles[idx];

    let data_idx = poly_idx * N + idx;
    let val = data[data_idx];

    // For inverse NTT, use conjugate twiddles (i.e., omega^(-i*j))
    // This is handled by using inverse twiddle table
    data[data_idx] = barrett_mul(val, twiddle, Q, mu);
}

// ============================================================================
// Step 3: Transpose N1 x N2 -> N2 x N1
// ============================================================================

const TILE_SIZE: u32 = 16u;

var<workgroup> tile: array<array<u64, 17>, 16>;  // +1 to avoid bank conflicts

@compute @workgroup_size(16, 16)
fn four_step_transpose(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tile_row = wid.x;
    let tile_col = wid.y;
    let poly_idx = wid.z;

    let N1 = params.N1;
    let N2 = params.N2;
    let N = params.N;

    let num_tile_rows = (N1 + TILE_SIZE - 1u) / TILE_SIZE;
    let num_tile_cols = (N2 + TILE_SIZE - 1u) / TILE_SIZE;

    if (tile_row >= num_tile_rows || tile_col >= num_tile_cols) {
        return;
    }

    let local_row = lid.x;
    let local_col = lid.y;

    // Source position (N1 x N2 layout)
    let src_row = tile_row * TILE_SIZE + local_row;
    let src_col = tile_col * TILE_SIZE + local_col;

    // Load tile into shared memory
    if (src_row < N1 && src_col < N2) {
        let src_idx = poly_idx * N + src_row * N2 + src_col;
        tile[local_row][local_col] = data[src_idx];
    } else {
        tile[local_row][local_col] = u64_zero();
    }

    workgroupBarrier();

    // Destination position (N2 x N1 layout, transposed)
    let dst_row = tile_col * TILE_SIZE + local_row;
    let dst_col = tile_row * TILE_SIZE + local_col;

    // Store from transposed position in shared memory
    if (dst_row < N2 && dst_col < N1) {
        let dst_idx = poly_idx * N + dst_row * N1 + dst_col;
        transpose_buf[dst_idx] = tile[local_col][local_row];
    }
}

// ============================================================================
// Step 4: Column-wise NTT (now rows after transpose)
// ============================================================================

@compute @workgroup_size(256)
fn four_step_col_ntt(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_id = lid.x;
    let row_idx = wid.x;  // Now processing rows of transposed matrix
    let poly_idx = wid.y;

    let N1 = params.N1;
    let N2 = params.N2;
    let log_N1 = params.log_N1;
    let prime_idx = params.prime_idx;

    if (row_idx >= N2 || N1 > SHARED_SIZE) {
        return;
    }

    let Q = primes[prime_idx];
    let mu = barrett_mu[prime_idx];

    // Load row (was column) into shared memory from transpose buffer
    let base_idx = poly_idx * params.N + row_idx * N1;
    for (var i = local_id; i < N1; i += WG_SIZE) {
        shared_data[i] = transpose_buf[base_idx + i];
    }

    // Load column twiddles
    let half_N1 = N1 >> 1u;
    for (var i = local_id; i < half_N1; i += WG_SIZE) {
        shared_twiddles[i] = col_twiddles[i];
    }

    workgroupBarrier();

    // Perform NTT using Cooley-Tukey (DIF)
    var m = N1;

    while (m > 1u) {
        let half_m = m >> 1u;
        let num_groups = N1 / m;

        for (var k = local_id; k < N1 >> 1u; k += WG_SIZE) {
            let group = k / half_m;
            let j = k % half_m;
            let idx0 = group * m + j;
            let idx1 = idx0 + half_m;

            let twiddle_idx = j * num_groups;
            let w = select(
                shared_twiddles[twiddle_idx % half_N1],
                col_twiddles[twiddle_idx],
                twiddle_idx >= half_N1
            );

            var a = shared_data[idx0];
            var b = shared_data[idx1];
            ct_butterfly(&a, &b, w, Q, mu);
            shared_data[idx0] = a;
            shared_data[idx1] = b;
        }

        workgroupBarrier();
        m = half_m;
    }

    // Bit-reverse permutation
    for (var i = local_id; i < N1; i += WG_SIZE) {
        var rev = 0u;
        var temp = i;
        for (var b = 0u; b < log_N1; b++) {
            rev = (rev << 1u) | (temp & 1u);
            temp = temp >> 1u;
        }
        if (i < rev) {
            let tmp = shared_data[i];
            shared_data[i] = shared_data[rev];
            shared_data[rev] = tmp;
        }
    }

    workgroupBarrier();

    // Store results back to original data buffer
    for (var i = local_id; i < N1; i += WG_SIZE) {
        data[base_idx + i] = shared_data[i];
    }
}

// ============================================================================
// Final transpose back (optional, for standard ordering)
// ============================================================================

@compute @workgroup_size(16, 16)
fn four_step_final_transpose(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tile_row = wid.x;
    let tile_col = wid.y;
    let poly_idx = wid.z;

    let N1 = params.N1;
    let N2 = params.N2;
    let N = params.N;

    let num_tile_rows = (N2 + TILE_SIZE - 1u) / TILE_SIZE;
    let num_tile_cols = (N1 + TILE_SIZE - 1u) / TILE_SIZE;

    if (tile_row >= num_tile_rows || tile_col >= num_tile_cols) {
        return;
    }

    let local_row = lid.x;
    let local_col = lid.y;

    // Source position (N2 x N1 layout after column NTT)
    let src_row = tile_row * TILE_SIZE + local_row;
    let src_col = tile_col * TILE_SIZE + local_col;

    if (src_row < N2 && src_col < N1) {
        let src_idx = poly_idx * N + src_row * N1 + src_col;
        tile[local_row][local_col] = data[src_idx];
    }

    workgroupBarrier();

    // Destination position (N1 x N2 layout, transposed back)
    let dst_row = tile_col * TILE_SIZE + local_row;
    let dst_col = tile_row * TILE_SIZE + local_col;

    if (dst_row < N1 && dst_col < N2) {
        let dst_idx = poly_idx * N + dst_row * N2 + dst_col;
        transpose_buf[dst_idx] = tile[local_col][local_row];
    }
}

// ============================================================================
// Copy transpose buffer back to main data
// ============================================================================

@compute @workgroup_size(256)
fn four_step_copy_back(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let poly_idx = gid.y;

    if (idx >= params.N) {
        return;
    }

    let data_idx = poly_idx * params.N + idx;
    data[data_idx] = transpose_buf[data_idx];
}

// ============================================================================
// Combined four-step inverse NTT
// ============================================================================

@compute @workgroup_size(256)
fn four_step_intt_scale(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let poly_idx = gid.y;

    if (idx >= params.N) {
        return;
    }

    let prime_idx = params.prime_idx;
    let Q = primes[prime_idx];
    let mu = barrett_mu[prime_idx];

    // N_inv = modular inverse of N
    // For now, assume N_inv is stored at barrett_mu[prime_idx + num_primes]
    let N_inv = barrett_mu[prime_idx + 8u];  // Assuming max 8 primes

    let data_idx = poly_idx * params.N + idx;
    let val = data[data_idx];
    data[data_idx] = barrett_mul(val, N_inv, Q, mu);
}

// ============================================================================
// Batched four-step NTT (multiple polynomials)
// ============================================================================

struct BatchParams {
    batch_start: u32,
    batch_count: u32,
    step: u32,  // 1=row, 2=twiddle, 3=transpose, 4=col
    _pad: u32,
}

@group(1) @binding(0) var<uniform> batch_params: BatchParams;

@compute @workgroup_size(256)
fn four_step_batch(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_id = lid.x;
    let batch_idx = batch_params.batch_start + wid.y;

    if (batch_idx >= batch_params.batch_start + batch_params.batch_count) {
        return;
    }

    let N1 = params.N1;
    let N2 = params.N2;
    let N = params.N;
    let prime_idx = params.prime_idx;
    let Q = primes[prime_idx];
    let mu = barrett_mu[prime_idx];

    let step = batch_params.step;

    if (step == 1u) {
        // Row NTT
        let row_idx = wid.x;
        if (row_idx < N1 && N2 <= SHARED_SIZE) {
            let base_idx = batch_idx * N + row_idx * N2;

            for (var i = local_id; i < N2; i += WG_SIZE) {
                shared_data[i] = data[base_idx + i];
            }
            workgroupBarrier();

            // Simplified NTT for demonstration
            var m = N2;
            while (m > 1u) {
                let half_m = m >> 1u;
                for (var k = local_id; k < N2 >> 1u; k += WG_SIZE) {
                    let group = k / half_m;
                    let j = k % half_m;
                    let idx0 = group * m + j;
                    let idx1 = idx0 + half_m;
                    let twiddle_idx = j * (N2 / m);
                    let w = row_twiddles[twiddle_idx];

                    var a = shared_data[idx0];
                    var b = shared_data[idx1];
                    ct_butterfly(&a, &b, w, Q, mu);
                    shared_data[idx0] = a;
                    shared_data[idx1] = b;
                }
                workgroupBarrier();
                m = half_m;
            }

            for (var i = local_id; i < N2; i += WG_SIZE) {
                data[base_idx + i] = shared_data[i];
            }
        }
    } else if (step == 2u) {
        // Twiddle multiply
        let idx = wid.x * WG_SIZE + local_id;
        if (idx < N) {
            let data_idx = batch_idx * N + idx;
            let twiddle = inter_twiddles[idx];
            data[data_idx] = barrett_mul(data[data_idx], twiddle, Q, mu);
        }
    }
    // Steps 3 and 4 handled by separate kernels due to workgroup structure
}
