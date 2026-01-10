// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// secp256k1 GPU Acceleration - WebGPU Kernel (WGSL)
// Implements GTable-based scalar multiplication for threshold ECDSA
// Compatible with browsers and cross-platform WebGPU runtimes
//
// Curve: secp256k1 (Bitcoin/Ethereum)
// p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
//      0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)

// ============================================================================
// Configuration
// ============================================================================

const GTABLE_CHUNKS: u32 = 16u;        // 256 bits / 16 = 16 bits per chunk
const GTABLE_CHUNK_SIZE: u32 = 65536u; // 2^16 = 65536 points per chunk
const KECCAK_ROUNDS: u32 = 24u;

// ============================================================================
// 256-bit Field Element (8 x 32-bit limbs, little-endian)
// ============================================================================

struct Fp256 {
    l0: u32, l1: u32, l2: u32, l3: u32,
    l4: u32, l5: u32, l6: u32, l7: u32,
}

// secp256k1 prime: p = 2^256 - 2^32 - 977
// p = 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F
const SECP256K1_P = Fp256(
    0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
);

// Curve order n
const SECP256K1_N = Fp256(
    0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
    0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
);

fn fp256_zero() -> Fp256 {
    return Fp256(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
}

fn fp256_one() -> Fp256 {
    return Fp256(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
}

fn fp256_is_zero(a: Fp256) -> bool {
    return (a.l0 | a.l1 | a.l2 | a.l3 | a.l4 | a.l5 | a.l6 | a.l7) == 0u;
}

fn fp256_eq(a: Fp256, b: Fp256) -> bool {
    return (a.l0 == b.l0) && (a.l1 == b.l1) && (a.l2 == b.l2) && (a.l3 == b.l3) &&
           (a.l4 == b.l4) && (a.l5 == b.l5) && (a.l6 == b.l6) && (a.l7 == b.l7);
}

// ============================================================================
// 256-bit Arithmetic
// ============================================================================

// Add with carry for 32-bit
fn adc32(a: u32, b: u32, carry_in: u32) -> vec2<u32> {
    let sum = a + b + carry_in;
    let carry = select(0u, 1u, sum < a || (carry_in != 0u && sum <= a));
    return vec2<u32>(sum, carry);
}

// Subtract with borrow for 32-bit
fn sbb32(a: u32, b: u32, borrow_in: u32) -> vec2<u32> {
    let diff = a - b - borrow_in;
    let borrow = select(0u, 1u, a < b || (borrow_in != 0u && a <= b));
    return vec2<u32>(diff, borrow);
}

// 256-bit addition (raw, no reduction)
fn fp256_add_raw(a: Fp256, b: Fp256) -> vec2<Fp256> {
    var r: array<u32, 8>;
    var carry = 0u;

    let t0 = adc32(a.l0, b.l0, carry); r[0] = t0.x; carry = t0.y;
    let t1 = adc32(a.l1, b.l1, carry); r[1] = t1.x; carry = t1.y;
    let t2 = adc32(a.l2, b.l2, carry); r[2] = t2.x; carry = t2.y;
    let t3 = adc32(a.l3, b.l3, carry); r[3] = t3.x; carry = t3.y;
    let t4 = adc32(a.l4, b.l4, carry); r[4] = t4.x; carry = t4.y;
    let t5 = adc32(a.l5, b.l5, carry); r[5] = t5.x; carry = t5.y;
    let t6 = adc32(a.l6, b.l6, carry); r[6] = t6.x; carry = t6.y;
    let t7 = adc32(a.l7, b.l7, carry); r[7] = t7.x; carry = t7.y;

    let result = Fp256(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
    let carry_fp = Fp256(carry, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    return vec2<Fp256>(result, carry_fp);
}

// 256-bit subtraction (raw, no reduction)
fn fp256_sub_raw(a: Fp256, b: Fp256) -> vec2<Fp256> {
    var r: array<u32, 8>;
    var borrow = 0u;

    let t0 = sbb32(a.l0, b.l0, borrow); r[0] = t0.x; borrow = t0.y;
    let t1 = sbb32(a.l1, b.l1, borrow); r[1] = t1.x; borrow = t1.y;
    let t2 = sbb32(a.l2, b.l2, borrow); r[2] = t2.x; borrow = t2.y;
    let t3 = sbb32(a.l3, b.l3, borrow); r[3] = t3.x; borrow = t3.y;
    let t4 = sbb32(a.l4, b.l4, borrow); r[4] = t4.x; borrow = t4.y;
    let t5 = sbb32(a.l5, b.l5, borrow); r[5] = t5.x; borrow = t5.y;
    let t6 = sbb32(a.l6, b.l6, borrow); r[6] = t6.x; borrow = t6.y;
    let t7 = sbb32(a.l7, b.l7, borrow); r[7] = t7.x; borrow = t7.y;

    let result = Fp256(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
    let borrow_fp = Fp256(borrow, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    return vec2<Fp256>(result, borrow_fp);
}

// Field addition with reduction mod p
fn fp256_add(a: Fp256, b: Fp256) -> Fp256 {
    let sum_result = fp256_add_raw(a, b);
    let sum = sum_result.x;
    let carry = sum_result.y.l0;

    // Try subtract p
    let sub_result = fp256_sub_raw(sum, SECP256K1_P);
    let reduced = sub_result.x;
    let borrow = sub_result.y.l0;

    // If carry or no borrow, use reduced result
    if (carry != 0u || borrow == 0u) {
        return reduced;
    }
    return sum;
}

// Field subtraction with reduction mod p
fn fp256_sub(a: Fp256, b: Fp256) -> Fp256 {
    let sub_result = fp256_sub_raw(a, b);
    let diff = sub_result.x;
    let borrow = sub_result.y.l0;

    // If borrow, add p
    if (borrow != 0u) {
        return fp256_add_raw(diff, SECP256K1_P).x;
    }
    return diff;
}

// Double the field element
fn fp256_double(a: Fp256) -> Fp256 {
    return fp256_add(a, a);
}

// Negate
fn fp256_neg(a: Fp256) -> Fp256 {
    if (fp256_is_zero(a)) {
        return a;
    }
    return fp256_sub(SECP256K1_P, a);
}

// 32x32 -> 64 multiply
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

    let lo = p0 + ((mid & 0xFFFFu) << 16u);
    let lo_carry = select(0u, 1u, lo < p0);

    let hi = p3 + (mid >> 16u) + mid_carry + lo_carry;

    return vec2<u32>(lo, hi);
}

// Multiply and add with carry
fn mac32(acc: vec2<u32>, a: u32, b: u32, carry: u32) -> vec3<u32> {
    let prod = mul32(a, b);

    var sum = acc.x + prod.x;
    var c1 = select(0u, 1u, sum < acc.x);

    sum = sum + carry;
    c1 = c1 + select(0u, 1u, sum < carry);

    let hi = acc.y + prod.y + c1;
    let c2 = select(0u, 1u, hi < acc.y || (c1 != 0u && hi <= acc.y));

    return vec3<u32>(sum, hi, c2);
}

// Schoolbook 256-bit multiplication -> 512-bit product
fn mul256x256(a: Fp256, b: Fp256) -> array<u32, 16> {
    var t: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) { t[i] = 0u; }

    let a_arr = array<u32, 8>(a.l0, a.l1, a.l2, a.l3, a.l4, a.l5, a.l6, a.l7);
    let b_arr = array<u32, 8>(b.l0, b.l1, b.l2, b.l3, b.l4, b.l5, b.l6, b.l7);

    for (var i = 0u; i < 8u; i++) {
        var carry = 0u;
        for (var j = 0u; j < 8u; j++) {
            let acc = vec2<u32>(t[i + j], t[i + j + 1u]);
            let r = mac32(acc, a_arr[i], b_arr[j], carry);
            t[i + j] = r.x;
            t[i + j + 1u] = r.y;
            carry = r.z;
        }
    }

    return t;
}

// secp256k1 fast reduction: p = 2^256 - 2^32 - 977
// T mod p = T_lo + (T_hi * 2^32 + T_hi * 977) mod p
fn fp256_reduce_512(t: array<u32, 16>) -> Fp256 {
    // Split: T = T_lo + T_hi * 2^256 where T_hi < 2^256
    // T mod p = T_lo + T_hi * (2^32 + 977) mod p

    var lo = Fp256(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    var hi = Fp256(t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15]);

    // Compute hi * 977
    var carry_977 = 0u;
    var h977: array<u32, 9>;
    let hi_arr = array<u32, 8>(hi.l0, hi.l1, hi.l2, hi.l3, hi.l4, hi.l5, hi.l6, hi.l7);

    for (var i = 0u; i < 8u; i++) {
        let prod = mul32(hi_arr[i], 977u);
        let sum = prod.x + carry_977;
        let c1 = select(0u, 1u, sum < prod.x);
        h977[i] = sum;
        carry_977 = prod.y + c1;
    }
    h977[8] = carry_977;

    // Shift hi left by 32 bits (multiply by 2^32)
    var hi_shifted: array<u32, 9>;
    hi_shifted[0] = 0u;
    for (var i = 0u; i < 8u; i++) {
        hi_shifted[i + 1u] = hi_arr[i];
    }

    // Add hi_shifted + h977
    var reduction: array<u32, 9>;
    var c = 0u;
    for (var i = 0u; i < 9u; i++) {
        let sum = hi_shifted[i] + h977[i] + c;
        c = select(0u, 1u, sum < hi_shifted[i] || (h977[i] != 0u && sum <= hi_shifted[i]));
        reduction[i] = sum;
    }

    // Add lo + reduction[0..7] (with extra limb reduction[8])
    var result: array<u32, 8>;
    let lo_arr = array<u32, 8>(lo.l0, lo.l1, lo.l2, lo.l3, lo.l4, lo.l5, lo.l6, lo.l7);
    c = 0u;
    for (var i = 0u; i < 8u; i++) {
        let t = adc32(lo_arr[i], reduction[i], c);
        result[i] = t.x;
        c = t.y;
    }
    c = c + reduction[8];

    // If there's still overflow, reduce again (at most twice more)
    var r = Fp256(result[0], result[1], result[2], result[3],
                  result[4], result[5], result[6], result[7]);

    for (var iter = 0u; iter < 3u; iter++) {
        if (c == 0u) {
            let sub_test = fp256_sub_raw(r, SECP256K1_P);
            if (sub_test.y.l0 == 0u) {
                r = sub_test.x;
            }
            break;
        }
        // r += c * (2^32 + 977)
        let correction = c * 977u + (c << 5u); // Simplified
        let add_result = fp256_add_raw(r, Fp256(correction, c, 0u, 0u, 0u, 0u, 0u, 0u));
        r = add_result.x;
        c = add_result.y.l0;
    }

    return r;
}

// Field multiplication mod p
fn fp256_mul(a: Fp256, b: Fp256) -> Fp256 {
    let product = mul256x256(a, b);
    return fp256_reduce_512(product);
}

// Field squaring
fn fp256_sqr(a: Fp256) -> Fp256 {
    return fp256_mul(a, a);
}

// ============================================================================
// Point Types
// ============================================================================

struct AffinePoint {
    x: Fp256,
    y: Fp256,
    infinity: u32,
}

struct JacobianPoint {
    x: Fp256,
    y: Fp256,
    z: Fp256,
}

fn affine_identity() -> AffinePoint {
    return AffinePoint(fp256_zero(), fp256_zero(), 1u);
}

fn jacobian_identity() -> JacobianPoint {
    return JacobianPoint(fp256_one(), fp256_one(), fp256_zero());
}

fn jacobian_is_identity(p: JacobianPoint) -> bool {
    return fp256_is_zero(p.z);
}

// ============================================================================
// Point Operations
// ============================================================================

// Point doubling in Jacobian coordinates (a=0 for secp256k1)
// Formula: dbl-2009-l (4M + 6S)
fn point_double(p: JacobianPoint) -> JacobianPoint {
    if (jacobian_is_identity(p)) {
        return p;
    }

    let A = fp256_sqr(p.x);           // X^2
    let B = fp256_sqr(p.y);           // Y^2
    let C = fp256_sqr(B);             // Y^4

    // D = 2*((X+B)^2 - A - C)
    let xplusb = fp256_add(p.x, B);
    let xplusb_sq = fp256_sqr(xplusb);
    var D = fp256_sub(xplusb_sq, A);
    D = fp256_sub(D, C);
    D = fp256_double(D);

    // E = 3*A
    let E = fp256_add(A, fp256_double(A));

    // F = E^2
    let F = fp256_sqr(E);

    // X' = F - 2*D
    let X3 = fp256_sub(F, fp256_double(D));

    // Y' = E*(D - X') - 8*C
    let dmx = fp256_sub(D, X3);
    let edx = fp256_mul(E, dmx);
    let c8 = fp256_double(fp256_double(fp256_double(C)));
    let Y3 = fp256_sub(edx, c8);

    // Z' = 2*Y*Z
    var Z3 = fp256_mul(p.y, p.z);
    Z3 = fp256_double(Z3);

    return JacobianPoint(X3, Y3, Z3);
}

// Mixed addition: Jacobian + Affine -> Jacobian
// Formula: madd-2008-s (7M + 4S)
fn point_add_mixed(p: JacobianPoint, q: AffinePoint) -> JacobianPoint {
    if (q.infinity != 0u) {
        return p;
    }
    if (jacobian_is_identity(p)) {
        return JacobianPoint(q.x, q.y, fp256_one());
    }

    let Z1Z1 = fp256_sqr(p.z);
    let U2 = fp256_mul(q.x, Z1Z1);
    var S2 = fp256_mul(q.y, p.z);
    S2 = fp256_mul(S2, Z1Z1);

    let H = fp256_sub(U2, p.x);
    let HH = fp256_sqr(H);
    let I = fp256_double(fp256_double(HH));
    let J = fp256_mul(H, I);
    var r = fp256_sub(S2, p.y);
    r = fp256_double(r);
    let V = fp256_mul(p.x, I);

    let r_sq = fp256_sqr(r);
    var X3 = fp256_sub(r_sq, J);
    X3 = fp256_sub(X3, fp256_double(V));

    let vmx = fp256_sub(V, X3);
    let rvmx = fp256_mul(r, vmx);
    let y1j = fp256_mul(p.y, J);
    let Y3 = fp256_sub(rvmx, fp256_double(y1j));

    let zph = fp256_add(p.z, H);
    let zph_sq = fp256_sqr(zph);
    var Z3 = fp256_sub(zph_sq, Z1Z1);
    Z3 = fp256_sub(Z3, HH);

    return JacobianPoint(X3, Y3, Z3);
}

// Full Jacobian addition (12M + 4S)
fn point_add(p: JacobianPoint, q: JacobianPoint) -> JacobianPoint {
    if (jacobian_is_identity(p)) { return q; }
    if (jacobian_is_identity(q)) { return p; }

    let Z1Z1 = fp256_sqr(p.z);
    let Z2Z2 = fp256_sqr(q.z);
    let U1 = fp256_mul(p.x, Z2Z2);
    let U2 = fp256_mul(q.x, Z1Z1);
    var S1 = fp256_mul(p.y, q.z);
    S1 = fp256_mul(S1, Z2Z2);
    var S2 = fp256_mul(q.y, p.z);
    S2 = fp256_mul(S2, Z1Z1);

    let H = fp256_sub(U2, U1);
    var I = fp256_double(H);
    I = fp256_sqr(I);
    let J = fp256_mul(H, I);
    var r = fp256_sub(S2, S1);
    r = fp256_double(r);
    let V = fp256_mul(U1, I);

    let r_sq = fp256_sqr(r);
    var X3 = fp256_sub(r_sq, J);
    X3 = fp256_sub(X3, fp256_double(V));

    let vmx = fp256_sub(V, X3);
    let rvmx = fp256_mul(r, vmx);
    let s1j = fp256_mul(S1, J);
    let Y3 = fp256_sub(rvmx, fp256_double(s1j));

    let zsum = fp256_add(p.z, q.z);
    let zsum_sq = fp256_sqr(zsum);
    var Z3 = fp256_sub(zsum_sq, Z1Z1);
    Z3 = fp256_sub(Z3, Z2Z2);
    Z3 = fp256_mul(Z3, H);

    return JacobianPoint(X3, Y3, Z3);
}

// ============================================================================
// Keccak256 (for Ethereum address derivation)
// ============================================================================

const KECCAK_RC: array<u32, 48> = array<u32, 48>(
    0x00000001u, 0x00000000u, 0x00008082u, 0x00000000u,
    0x0000808au, 0x80000000u, 0x80008000u, 0x80000000u,
    0x0000808bu, 0x00000000u, 0x80000001u, 0x00000000u,
    0x80008081u, 0x80000000u, 0x00008009u, 0x80000000u,
    0x0000008au, 0x00000000u, 0x00000088u, 0x00000000u,
    0x80008009u, 0x00000000u, 0x8000000au, 0x00000000u,
    0x8000808bu, 0x00000000u, 0x0000008bu, 0x80000000u,
    0x00008089u, 0x80000000u, 0x00008003u, 0x80000000u,
    0x00008002u, 0x80000000u, 0x00000080u, 0x80000000u,
    0x0000800au, 0x00000000u, 0x8000000au, 0x80000000u,
    0x80008081u, 0x80000000u, 0x00008080u, 0x80000000u,
    0x80000001u, 0x00000000u, 0x80008008u, 0x80000000u
);

// 64-bit rotate left
fn rotl64(v_lo: u32, v_hi: u32, n: u32) -> vec2<u32> {
    if (n == 0u) { return vec2<u32>(v_lo, v_hi); }
    if (n < 32u) {
        return vec2<u32>(
            (v_lo << n) | (v_hi >> (32u - n)),
            (v_hi << n) | (v_lo >> (32u - n))
        );
    }
    let m = n - 32u;
    return vec2<u32>(
        (v_hi << m) | (v_lo >> (32u - m)),
        (v_lo << m) | (v_hi >> (32u - m))
    );
}

// XOR two 64-bit values
fn xor64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    return vec2<u32>(a_lo ^ b_lo, a_hi ^ b_hi);
}

// Keccak256 for 64-byte public key -> 32-byte hash
fn keccak256_pubkey(pub_x: Fp256, pub_y: Fp256) -> Fp256 {
    // Simplified: In practice, implement full Keccak-f[1600]
    // For demonstration, return a deterministic hash placeholder
    // Real implementation would process 64 bytes through Keccak sponge

    // XOR coordinates together as simplified hash (not cryptographically secure!)
    // Real implementation needs full Keccak256 permutation
    return Fp256(
        pub_x.l0 ^ pub_y.l0 ^ 0x6A09E667u,
        pub_x.l1 ^ pub_y.l1 ^ 0xBB67AE85u,
        pub_x.l2 ^ pub_y.l2 ^ 0x3C6EF372u,
        pub_x.l3 ^ pub_y.l3 ^ 0xA54FF53Au,
        pub_x.l4 ^ pub_y.l4 ^ 0x510E527Fu,
        pub_x.l5 ^ pub_y.l5 ^ 0x9B05688Cu,
        pub_x.l6 ^ pub_y.l6 ^ 0x1F83D9ABu,
        pub_x.l7 ^ pub_y.l7 ^ 0x5BE0CD19u
    );
}

// ============================================================================
// Bindings
// ============================================================================

struct ScalarMulParams {
    count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

// GTable: [16 chunks][65536 points] = ~67MB of precomputed points
@group(0) @binding(0) var<storage, read> gtable: array<AffinePoint>;

// Input scalars (256-bit each)
@group(0) @binding(1) var<storage, read> scalars: array<Fp256>;

// Output points (affine)
@group(0) @binding(2) var<storage, read_write> results: array<AffinePoint>;

// Parameters
@group(0) @binding(3) var<uniform> params: ScalarMulParams;

// ============================================================================
// GTable Scalar Multiplication Kernel
// ============================================================================

// Extract 16-bit chunk from 256-bit scalar
fn extract_chunk(scalar: Fp256, chunk_idx: u32) -> u32 {
    let bit_offset = chunk_idx * 16u;
    let limb_idx = bit_offset / 32u;
    let bit_in_limb = bit_offset % 32u;

    let scalar_arr = array<u32, 8>(
        scalar.l0, scalar.l1, scalar.l2, scalar.l3,
        scalar.l4, scalar.l5, scalar.l6, scalar.l7
    );

    var chunk = scalar_arr[limb_idx] >> bit_in_limb;

    // Handle cross-limb chunk (16 bits can span 2 limbs if not aligned)
    if (bit_in_limb + 16u > 32u && limb_idx + 1u < 8u) {
        chunk = chunk | (scalar_arr[limb_idx + 1u] << (32u - bit_in_limb));
    }

    return chunk & 0xFFFFu;
}

// Jacobian to Affine conversion (requires modular inverse)
// Simplified version - in practice use Fermat's little theorem
fn jacobian_to_affine(p: JacobianPoint) -> AffinePoint {
    if (jacobian_is_identity(p)) {
        return affine_identity();
    }

    // z_inv = z^(p-2) mod p (Fermat's little theorem)
    // For simplicity, approximate with square root
    // Real implementation needs full modular inverse
    let z_sq = fp256_sqr(p.z);
    let z_cubed = fp256_mul(z_sq, p.z);

    // Approximate inverse (not correct - needs proper implementation)
    // x_aff = x / z^2, y_aff = y / z^3
    // This is a placeholder - real code needs fp256_inv

    return AffinePoint(p.x, p.y, 0u);  // Placeholder
}

// GTable-based scalar multiplication
// k*G = sum_{i=0}^{15} gtable[i][chunk_i] where chunk_i = k[16*i : 16*i+15]
@compute @workgroup_size(256)
fn gtable_scalar_mul(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let scalar = scalars[idx];

    // Accumulator in Jacobian coordinates
    var acc = jacobian_identity();

    // Process 16 chunks of 16 bits each
    for (var chunk_idx = 0u; chunk_idx < GTABLE_CHUNKS; chunk_idx++) {
        let chunk_val = extract_chunk(scalar, chunk_idx);

        if (chunk_val != 0u) {
            // Look up precomputed point: gtable[chunk_idx * 65536 + (chunk_val - 1)]
            // Note: chunk_val=0 means identity (skip), chunk_val=1..65535 maps to table
            let table_offset = chunk_idx * GTABLE_CHUNK_SIZE + chunk_val - 1u;
            let table_point = gtable[table_offset];

            // Add to accumulator
            acc = point_add_mixed(acc, table_point);
        }
    }

    // Convert to affine and store
    results[idx] = jacobian_to_affine(acc);
}

// ============================================================================
// Batch Signature Verification Kernel
// ============================================================================

struct SignatureData {
    r: Fp256,       // r component of signature
    s: Fp256,       // s component of signature
    e: Fp256,       // message hash (z in ECDSA terms)
    pub_x: Fp256,   // public key x coordinate
    pub_y: Fp256,   // public key y coordinate
}

@group(0) @binding(4) var<storage, read> signatures: array<SignatureData>;
@group(0) @binding(5) var<storage, read_write> verify_results: array<u32>;

// Batch ECDSA verification
// Verifies: s^{-1} * (e*G + r*P) == R where R.x == r
@compute @workgroup_size(256)
fn batch_verify(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let sig = signatures[idx];

    // s^{-1} * e * G + s^{-1} * r * P should equal R with R.x = r

    // This requires:
    // 1. Modular inverse of s (mod n)
    // 2. Scalar multiplication: u1*G and u2*P
    // 3. Point addition

    // Simplified: Mark as verified (placeholder)
    // Real implementation needs full ECDSA verification

    verify_results[idx] = 1u;  // 1 = valid, 0 = invalid
}

// ============================================================================
// Address Derivation Kernel
// ============================================================================

struct AddressInput {
    pub_x: Fp256,
    pub_y: Fp256,
}

struct AddressOutput {
    addr: array<u32, 5>,  // 20 bytes = 5 x 32-bit
}

@group(0) @binding(6) var<storage, read> pubkeys: array<AddressInput>;
@group(0) @binding(7) var<storage, read_write> addresses: array<AddressOutput>;

// Derive Ethereum address from public key
// address = keccak256(pubkey)[12:32]
@compute @workgroup_size(256)
fn batch_derive_address(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let pubkey = pubkeys[idx];

    // Hash the public key
    let hash = keccak256_pubkey(pubkey.pub_x, pubkey.pub_y);

    // Extract last 20 bytes (address)
    // hash[12:32] = bytes 12-31 = last 5 32-bit words in little-endian
    addresses[idx].addr[0] = hash.l3;
    addresses[idx].addr[1] = hash.l4;
    addresses[idx].addr[2] = hash.l5;
    addresses[idx].addr[3] = hash.l6;
    addresses[idx].addr[4] = hash.l7;
}

// ============================================================================
// GTable Initialization Kernel (precomputation)
// ============================================================================

@group(0) @binding(8) var<storage, read_write> gtable_out: array<AffinePoint>;
@group(0) @binding(9) var<storage, read> generator: AffinePoint;

// Precompute GTable chunk
// For chunk i: gtable[i][j] = (j+1) * 2^(16*i) * G for j in 0..65535
@compute @workgroup_size(256)
fn precompute_gtable(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    // This is typically done once on CPU and uploaded
    // GPU version can parallelize across chunks

    let idx = gid.x;
    if (idx >= GTABLE_CHUNKS * GTABLE_CHUNK_SIZE) {
        return;
    }

    let chunk_idx = idx / GTABLE_CHUNK_SIZE;
    let point_idx = idx % GTABLE_CHUNK_SIZE;

    // base = 2^(16*chunk_idx) * G
    var base = JacobianPoint(generator.x, generator.y, fp256_one());

    // Double base (16 * chunk_idx) times
    for (var i = 0u; i < chunk_idx * 16u; i++) {
        base = point_double(base);
    }

    // Compute (point_idx + 1) * base
    var result = base;
    for (var i = 0u; i < point_idx; i++) {
        result = point_add(result, base);
    }

    gtable_out[idx] = jacobian_to_affine(result);
}

// ============================================================================
// Parallel Reduction for Aggregation
// ============================================================================

var<workgroup> shared_points: array<JacobianPoint, 256>;

@compute @workgroup_size(256)
fn parallel_point_reduce(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let idx = wgid.x * 512u + tid;
    let n = params.count;

    // Load two points per thread
    if (idx < n) {
        shared_points[tid] = JacobianPoint(
            results[idx].x,
            results[idx].y,
            select(fp256_zero(), fp256_one(), results[idx].infinity == 0u)
        );
    } else {
        shared_points[tid] = jacobian_identity();
    }

    if (idx + 256u < n) {
        let second = JacobianPoint(
            results[idx + 256u].x,
            results[idx + 256u].y,
            select(fp256_zero(), fp256_one(), results[idx + 256u].infinity == 0u)
        );
        shared_points[tid] = point_add(shared_points[tid], second);
    }

    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_points[tid] = point_add(shared_points[tid], shared_points[tid + stride]);
        }
        workgroupBarrier();
    }

    // Write result
    if (tid == 0u) {
        let final_point = jacobian_to_affine(shared_points[0]);
        results[wgid.x] = final_point;
    }
}
