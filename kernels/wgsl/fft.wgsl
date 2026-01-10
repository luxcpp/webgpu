// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Portable Fast Fourier Transform (FFT) kernel in WGSL
// Works on WebGPU (Metal/Vulkan/D3D12 via Dawn/wgpu)
//
// Part of the Lux Network GPU acceleration library

// ============================================================================
// Complex Number Operations
// ============================================================================

struct Complex {
    re: f32,
    im: f32,
}

fn complex_add(a: Complex, b: Complex) -> Complex {
    return Complex(a.re + b.re, a.im + b.im);
}

fn complex_sub(a: Complex, b: Complex) -> Complex {
    return Complex(a.re - b.re, a.im - b.im);
}

fn complex_mul(a: Complex, b: Complex) -> Complex {
    return Complex(
        a.re * b.re - a.im * b.im,
        a.re * b.im + a.im * b.re
    );
}

fn complex_conj(a: Complex) -> Complex {
    return Complex(a.re, -a.im);
}

fn complex_scale(a: Complex, s: f32) -> Complex {
    return Complex(a.re * s, a.im * s);
}

// Compute e^(i * theta) = cos(theta) + i*sin(theta)
fn complex_exp(theta: f32) -> Complex {
    return Complex(cos(theta), sin(theta));
}

// ============================================================================
// Kernel Bindings
// ============================================================================

struct FFTParams {
    N: u32,           // FFT size
    batch: u32,       // Number of batches
    stage: u32,       // Current stage (for staged FFT)
    inverse: u32,     // 1 for inverse FFT, 0 for forward
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;  // Interleaved [re, im, re, im, ...]
@group(0) @binding(1) var<storage, read> twiddles: array<f32>;    // Precomputed twiddles
@group(0) @binding(2) var<uniform> params: FFTParams;

var<workgroup> shared_re: array<f32, 2048>;
var<workgroup> shared_im: array<f32, 2048>;

// ============================================================================
// Radix-2 Cooley-Tukey FFT (in-place, decimation-in-time)
// ============================================================================

@compute @workgroup_size(256)
fn fft_radix2_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params.N;
    let batch = params.batch;
    let stage = params.stage;
    let inverse = params.inverse;
    
    let total = batch * (N / 2u);
    if (gid.x >= total) { return; }
    
    let batch_idx = gid.x / (N / 2u);
    let k = gid.x % (N / 2u);
    
    let m = 1u << (stage + 1u);
    let half_m = m >> 1u;
    let j = k / half_m;
    let i = k % half_m;
    let idx0 = j * m + i;
    let idx1 = idx0 + half_m;
    
    let base = batch_idx * N * 2u;  // *2 for complex (re, im pairs)
    
    // Load values
    let x0 = Complex(data[base + idx0 * 2u], data[base + idx0 * 2u + 1u]);
    let x1 = Complex(data[base + idx1 * 2u], data[base + idx1 * 2u + 1u]);
    
    // Compute twiddle factor
    let angle = -2.0 * 3.14159265358979323846 * f32(i) / f32(m);
    var w = complex_exp(select(angle, -angle, inverse == 1u));
    
    // Butterfly
    let t = complex_mul(x1, w);
    let y0 = complex_add(x0, t);
    let y1 = complex_sub(x0, t);
    
    // Store results
    data[base + idx0 * 2u] = y0.re;
    data[base + idx0 * 2u + 1u] = y0.im;
    data[base + idx1 * 2u] = y1.re;
    data[base + idx1 * 2u + 1u] = y1.im;
}

// ============================================================================
// Fused FFT for small sizes (N <= 2048) using shared memory
// ============================================================================

@compute @workgroup_size(256)
fn fft_fused(@builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(workgroup_id) wid: vec3<u32>) {
    let N = params.N;
    let batch_idx = wid.x;
    let inverse = params.inverse;
    let tid = lid.x;
    let threads = 256u;
    
    if (batch_idx >= params.batch) { return; }
    
    let base = batch_idx * N * 2u;
    
    // Load to shared memory
    for (var i = tid; i < N; i += threads) {
        shared_re[i] = data[base + i * 2u];
        shared_im[i] = data[base + i * 2u + 1u];
    }
    workgroupBarrier();
    
    // Compute log2(N)
    var log_n = 0u;
    var temp = N;
    while (temp > 1u) {
        temp = temp >> 1u;
        log_n += 1u;
    }
    
    // Cooley-Tukey stages
    for (var stage = 0u; stage < log_n; stage += 1u) {
        let m = 1u << (stage + 1u);
        let half_m = m >> 1u;
        
        for (var k = tid; k < N / 2u; k += threads) {
            let j = k / half_m;
            let i = k % half_m;
            let idx0 = j * m + i;
            let idx1 = idx0 + half_m;
            
            let angle = -2.0 * 3.14159265358979323846 * f32(i) / f32(m);
            let w = complex_exp(select(angle, -angle, inverse == 1u));
            
            let x0 = Complex(shared_re[idx0], shared_im[idx0]);
            let x1 = Complex(shared_re[idx1], shared_im[idx1]);
            
            let t = complex_mul(x1, w);
            let y0 = complex_add(x0, t);
            let y1 = complex_sub(x0, t);
            
            shared_re[idx0] = y0.re;
            shared_im[idx0] = y0.im;
            shared_re[idx1] = y1.re;
            shared_im[idx1] = y1.im;
        }
        workgroupBarrier();
    }
    
    // Apply scaling for inverse FFT and store
    let scale = select(1.0, 1.0 / f32(N), inverse == 1u);
    for (var i = tid; i < N; i += threads) {
        data[base + i * 2u] = shared_re[i] * scale;
        data[base + i * 2u + 1u] = shared_im[i] * scale;
    }
}

// ============================================================================
// Real-to-Complex FFT (for real-valued input)
// ============================================================================

@compute @workgroup_size(256)
fn rfft_pack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params.N;
    let batch = params.batch;
    
    if (gid.x >= batch * (N / 2u)) { return; }
    
    let batch_idx = gid.x / (N / 2u);
    let k = gid.x % (N / 2u);
    
    let base = batch_idx * N;
    
    // Pack two real values as one complex
    // Used after real FFT to extract positive frequencies
    if (k == 0u) {
        // DC and Nyquist components
        let dc = data[base * 2u];
        let nyq = data[base * 2u + 1u];
        data[base * 2u] = dc;
        data[base * 2u + 1u] = 0.0;
        // Store Nyquist at N/2
        data[base * 2u + N] = nyq;
        data[base * 2u + N + 1u] = 0.0;
    } else {
        // Use conjugate symmetry: X[N-k] = conj(X[k])
        let idx_pos = k;
        let idx_neg = N - k;
        
        let xp = Complex(data[base * 2u + idx_pos * 2u], data[base * 2u + idx_pos * 2u + 1u]);
        let xn = Complex(data[base * 2u + idx_neg * 2u], data[base * 2u + idx_neg * 2u + 1u]);
        
        // Extract even and odd parts
        let xe = complex_scale(complex_add(xp, complex_conj(xn)), 0.5);
        let xo = complex_scale(complex_sub(xp, complex_conj(xn)), 0.5);
        
        data[base * 2u + idx_pos * 2u] = xe.re;
        data[base * 2u + idx_pos * 2u + 1u] = xe.im;
    }
}

// ============================================================================
// Convolution via FFT (pointwise multiply in frequency domain)
// ============================================================================

@compute @workgroup_size(256)
fn fft_pointwise_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.N * params.batch;
    if (gid.x >= size) { return; }
    
    let idx = gid.x * 2u;
    
    // Load complex numbers from data and twiddles (reusing twiddles buffer for second operand)
    let a = Complex(data[idx], data[idx + 1u]);
    let b = Complex(twiddles[idx], twiddles[idx + 1u]);
    
    let result = complex_mul(a, b);
    
    data[idx] = result.re;
    data[idx + 1u] = result.im;
}
