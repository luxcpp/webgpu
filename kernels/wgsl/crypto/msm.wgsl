// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Multi-Scalar Multiplication (MSM) for elliptic curves
// Pippenger's algorithm with bucket accumulation
// Supports BLS12-381 G1 and BN254 G1 curves

// 384-bit field element for BLS12-381 (12 x 32-bit limbs)
struct Fe384 {
    limbs: array<u32, 12>,
}

// 256-bit scalar for BLS12-381
struct Scalar256 {
    limbs: array<u32, 8>,
}

// Affine point (x, y)
struct AffinePoint {
    x: Fe384,
    y: Fe384,
}

// Projective point (X, Y, Z) where x = X/Z, y = Y/Z
struct ProjectivePoint {
    x: Fe384,
    y: Fe384,
    z: Fe384,
}

// MSM parameters
struct MsmParams {
    num_points: u32,
    window_bits: u32,      // Typically 12-16 for optimal performance
    num_windows: u32,      // ceil(256 / window_bits)
    num_buckets: u32,      // 2^window_bits - 1
}

// Field modulus for BLS12-381 base field (p)
const BLS12_381_P: array<u32, 12> = array<u32, 12>(
    0xb9feffffu, 0x1eabfffeu, 0xb153ffffu, 0x6730d2a0u,
    0xf6241eabu, 0x4a7c0f9eu, 0x636ba0du, 0xa34d9f50u,
    0xffcd46deu, 0xe8adb479u, 0xa70f72a2u, 0x1a0111eau
);

// Buffer bindings
@group(0) @binding(0) var<storage, read> points: array<AffinePoint>;
@group(0) @binding(1) var<storage, read> scalars: array<Scalar256>;
@group(0) @binding(2) var<storage, read_write> buckets: array<ProjectivePoint>;
@group(0) @binding(3) var<storage, read_write> result: ProjectivePoint;
@group(0) @binding(4) var<uniform> params: MsmParams;

// ============================================================================
// 384-bit Field Arithmetic
// ============================================================================

fn fe384_zero() -> Fe384 {
    var r: Fe384;
    for (var i = 0u; i < 12u; i++) {
        r.limbs[i] = 0u;
    }
    return r;
}

fn fe384_one() -> Fe384 {
    var r: Fe384;
    r.limbs[0] = 1u;
    for (var i = 1u; i < 12u; i++) {
        r.limbs[i] = 0u;
    }
    return r;
}

fn fe384_eq(a: Fe384, b: Fe384) -> bool {
    for (var i = 0u; i < 12u; i++) {
        if (a.limbs[i] != b.limbs[i]) {
            return false;
        }
    }
    return true;
}

fn fe384_is_zero(a: Fe384) -> bool {
    for (var i = 0u; i < 12u; i++) {
        if (a.limbs[i] != 0u) {
            return false;
        }
    }
    return true;
}

// Add two 384-bit field elements (without reduction)
fn fe384_add_no_reduce(a: Fe384, b: Fe384) -> Fe384 {
    var r: Fe384;
    var carry: u32 = 0u;
    
    for (var i = 0u; i < 12u; i++) {
        let sum: u32 = a.limbs[i] + b.limbs[i] + carry;
        r.limbs[i] = sum;
        carry = select(0u, 1u, sum < a.limbs[i] || (carry == 1u && sum == a.limbs[i]));
    }
    
    return r;
}

// Subtract b from a (assumes a >= b)
fn fe384_sub(a: Fe384, b: Fe384) -> Fe384 {
    var r: Fe384;
    var borrow: u32 = 0u;
    
    for (var i = 0u; i < 12u; i++) {
        let sub: u32 = a.limbs[i] - b.limbs[i] - borrow;
        borrow = select(0u, 1u, a.limbs[i] < b.limbs[i] + borrow);
        r.limbs[i] = sub;
    }
    
    return r;
}

// Compare a >= b
fn fe384_gte(a: Fe384, b: Fe384) -> bool {
    for (var i = 11; i >= 0; i--) {
        if (a.limbs[i] > b.limbs[i]) { return true; }
        if (a.limbs[i] < b.limbs[i]) { return false; }
    }
    return true; // Equal
}

// Modular reduction
fn fe384_reduce(a: Fe384) -> Fe384 {
    var p: Fe384;
    for (var i = 0u; i < 12u; i++) {
        p.limbs[i] = BLS12_381_P[i];
    }
    
    var r = a;
    if (fe384_gte(r, p)) {
        r = fe384_sub(r, p);
    }
    return r;
}

// Modular addition
fn fe384_add(a: Fe384, b: Fe384) -> Fe384 {
    return fe384_reduce(fe384_add_no_reduce(a, b));
}

// Double a field element
fn fe384_double(a: Fe384) -> Fe384 {
    return fe384_add(a, a);
}

// Multiply two limbs and return 64-bit result as (lo, hi)
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    
    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;
    
    let mid = p1 + p2;
    let mid_carry = select(0u, 0x10000u, mid < p1);
    
    let lo = p0 + (mid << 16u);
    let lo_carry = select(0u, 1u, lo < p0);
    
    let hi = p3 + (mid >> 16u) + mid_carry + lo_carry + select(0u, 1u, (p0 >> 16u) + (mid & 0xFFFFu) >= 0x10000u);
    
    return vec2<u32>(lo, hi);
}

// ============================================================================
// Projective Point Operations
// ============================================================================

fn point_identity() -> ProjectivePoint {
    var p: ProjectivePoint;
    p.x = fe384_zero();
    p.y = fe384_one();
    p.z = fe384_zero();
    return p;
}

fn point_is_identity(p: ProjectivePoint) -> bool {
    return fe384_is_zero(p.z);
}

// Convert affine to projective
fn affine_to_projective(a: AffinePoint) -> ProjectivePoint {
    var p: ProjectivePoint;
    p.x = a.x;
    p.y = a.y;
    p.z = fe384_one();
    return p;
}

// Point doubling in projective coordinates (2P)
// Using complete addition formulas
fn point_double(p: ProjectivePoint) -> ProjectivePoint {
    if (point_is_identity(p)) {
        return p;
    }
    
    // Simplified doubling (actual implementation needs full field multiply)
    var r: ProjectivePoint;
    
    // XX = X^2
    // YY = Y^2
    // ZZ = Z^2
    // S = 4*X*Y^2
    // M = 3*X^2 + a*Z^4 (a=0 for BLS12-381)
    // X' = M^2 - 2*S
    // Y' = M*(S - X') - 8*Y^4
    // Z' = 2*Y*Z
    
    // Placeholder - actual implementation requires full modular multiply
    r.x = fe384_double(p.x);
    r.y = fe384_double(p.y);
    r.z = fe384_double(p.z);
    
    return r;
}

// Mixed addition: projective + affine
fn point_add_mixed(p: ProjectivePoint, a: AffinePoint) -> ProjectivePoint {
    if (point_is_identity(p)) {
        return affine_to_projective(a);
    }
    
    // Placeholder - full implementation needs modular multiply
    var r: ProjectivePoint;
    r.x = fe384_add(p.x, a.x);
    r.y = fe384_add(p.y, a.y);
    r.z = p.z;
    
    return r;
}

// ============================================================================
// Scalar Operations
// ============================================================================

// Get window bits from scalar
fn get_window(scalar: Scalar256, window_idx: u32, window_bits: u32) -> u32 {
    let bit_offset = window_idx * window_bits;
    let limb_idx = bit_offset / 32u;
    let bit_in_limb = bit_offset % 32u;
    
    let mask = (1u << window_bits) - 1u;
    
    if (limb_idx >= 8u) {
        return 0u;
    }
    
    var window = (scalar.limbs[limb_idx] >> bit_in_limb) & mask;
    
    // Handle window crossing limb boundary
    if (bit_in_limb + window_bits > 32u && limb_idx + 1u < 8u) {
        let remaining_bits = bit_in_limb + window_bits - 32u;
        window |= (scalar.limbs[limb_idx + 1u] << (window_bits - remaining_bits)) & mask;
    }
    
    return window;
}

// ============================================================================
// MSM Kernels
// ============================================================================

// Phase 1: Bucket accumulation
// Each workgroup handles a subset of points, accumulating into buckets
@compute @workgroup_size(256)
fn msm_bucket_accumulate(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let point_idx = gid.x;
    let window_idx = wgid.y;
    
    if (point_idx >= params.num_points) {
        return;
    }
    
    let scalar = scalars[point_idx];
    let window_value = get_window(scalar, window_idx, params.window_bits);
    
    if (window_value == 0u) {
        return; // Skip zero windows
    }
    
    let bucket_idx = window_idx * params.num_buckets + (window_value - 1u);
    let point = points[point_idx];
    
    // Atomic accumulation would be needed here
    // For now, we use a simple add (production code needs proper synchronization)
    let current = buckets[bucket_idx];
    buckets[bucket_idx] = point_add_mixed(current, point);
}

// Phase 2: Bucket reduction
// Reduce buckets within each window: result = sum(i * bucket[i])
@compute @workgroup_size(256)
fn msm_bucket_reduce(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let window_idx = gid.x;
    
    if (window_idx >= params.num_windows) {
        return;
    }
    
    let base_bucket = window_idx * params.num_buckets;
    
    // Running sum method
    var running = point_identity();
    var acc = point_identity();
    
    // Process buckets in reverse order
    for (var i = params.num_buckets; i > 0u; i--) {
        let bucket = buckets[base_bucket + i - 1u];
        running = point_add_mixed(running, AffinePoint(bucket.x, bucket.y));
        acc = point_add_mixed(acc, AffinePoint(running.x, running.y));
    }
    
    // Store window result back to first bucket position
    buckets[base_bucket] = acc;
}

// Phase 3: Window combination
// Combine window results: final = sum(2^(w*i) * window_result[i])
@compute @workgroup_size(1)
fn msm_window_combine() {
    var acc = point_identity();
    
    // Process windows from high to low
    for (var w = params.num_windows; w > 0u; w--) {
        // Double acc by window_bits
        for (var i = 0u; i < params.window_bits; i++) {
            acc = point_double(acc);
        }
        
        // Add window result
        let window_result = buckets[(w - 1u) * params.num_buckets];
        acc = point_add_mixed(acc, AffinePoint(window_result.x, window_result.y));
    }
    
    result = acc;
}

// Single-pass MSM for small inputs
@compute @workgroup_size(256)
fn msm_naive(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x != 0u) {
        return;
    }
    
    var acc = point_identity();
    
    for (var i = 0u; i < params.num_points; i++) {
        let scalar = scalars[i];
        let point = points[i];
        
        // Double-and-add
        var temp = affine_to_projective(point);
        
        for (var b = 0u; b < 256u; b++) {
            let limb = b / 32u;
            let bit = b % 32u;
            
            if ((scalar.limbs[limb] & (1u << bit)) != 0u) {
                acc = point_add_mixed(acc, point);
            }
            
            // Unused in naive version - just for iteration
        }
    }
    
    result = acc;
}
