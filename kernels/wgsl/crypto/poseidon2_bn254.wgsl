// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Poseidon2 over BN254 Fr (t=2, gnark default) — WGSL implementation.
//
// Matches gnark-crypto v0.20.1 ecc/bn254/fr/poseidon2 NewDefaultPermutation():
//   - Width t = 2, full rounds rF = 6 (3 pre + 3 post), partial rounds rP = 50.
//   - S-box: x^5. External M_E = [[2,1],[1,2]]. Internal M_I = [[2,1],[1,3]].
//   - Compress(L, R) = permutation(L, R)[1] + R   (Davies-Meyer feed-forward).
//
// CPU reference: luxcpp/gpu/src/bn254_field.hpp::poseidon2_compress.
// KAT vectors (gnark Compress): see luxcpp/gpu/test/test_backend_parity.cpp.

// =============================================================================
// BN254 Fr (8 x u32 limbs, little-endian)
// =============================================================================

struct Fr256 {
    l0: u32, l1: u32, l2: u32, l3: u32,
    l4: u32, l5: u32, l6: u32, l7: u32,
}

// BN254 scalar field modulus r (little-endian u32).
const BN254_R = Fr256(
    0xF0000001u, 0x43E1F593u, 0x79B97091u, 0x2833E848u,
    0x8181585Du, 0xB85045B6u, 0xE131A029u, 0x30644E72u
);

// R mod r — Montgomery form of 1.
const BN254_FR_R = Fr256(
    0x4FFFFFFBu, 0xAC96341Cu, 0x9F60CD29u, 0x36FC7695u,
    0x7879462Eu, 0x666EA36Fu, 0x9A07DF2Fu, 0x0E0A77C1u
);

// R^2 mod r — matches gnark fr.rSquare exactly.
const BN254_FR_R2 = Fr256(
    0xae216da7u, 0x1bb8e645u, 0xe35c59e3u, 0x53fe3ab1u,
    0x53bb8085u, 0x8c49833du, 0xf4e44a5u,  0x0216d0b1u
);

// -r^{-1} mod 2^32.
const BN254_R_INV: u32 = 0xEFFFFFFFu;

// =============================================================================
// Basic Arithmetic
// =============================================================================

fn fr_zero() -> Fr256 {
    return Fr256(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
}

fn fr_one() -> Fr256 {
    return BN254_FR_R;
}

fn fr_is_zero(a: Fr256) -> bool {
    return (a.l0 | a.l1 | a.l2 | a.l3 | a.l4 | a.l5 | a.l6 | a.l7) == 0u;
}

fn adc32(a: u32, b: u32, carry_in: u32) -> vec2<u32> {
    let sum = a + b + carry_in;
    let carry = select(0u, 1u, sum < a || (carry_in != 0u && sum <= a));
    return vec2<u32>(sum, carry);
}

fn sbb32(a: u32, b: u32, borrow_in: u32) -> vec2<u32> {
    let diff = a - b - borrow_in;
    let borrow = select(0u, 1u, a < b + borrow_in);
    return vec2<u32>(diff, borrow);
}

fn fr_add_raw(a: Fr256, b: Fr256) -> vec2<Fr256> {
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
    let result   = Fr256(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
    let carry_fp = Fr256(carry, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    return vec2<Fr256>(result, carry_fp);
}

fn fr_sub_raw(a: Fr256, b: Fr256) -> vec2<Fr256> {
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
    let result    = Fr256(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
    let borrow_fp = Fr256(borrow, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    return vec2<Fr256>(result, borrow_fp);
}

fn fr_add(a: Fr256, b: Fr256) -> Fr256 {
    let sum_result = fr_add_raw(a, b);
    let sum = sum_result.x;
    let carry = sum_result.y.l0;
    let sub_result = fr_sub_raw(sum, BN254_R);
    let reduced = sub_result.x;
    let borrow = sub_result.y.l0;
    if (carry != 0u || borrow == 0u) {
        return reduced;
    }
    return sum;
}

fn fr_sub(a: Fr256, b: Fr256) -> Fr256 {
    let sub_result = fr_sub_raw(a, b);
    let diff = sub_result.x;
    let borrow = sub_result.y.l0;
    if (borrow != 0u) {
        return fr_add_raw(diff, BN254_R).x;
    }
    return diff;
}

fn fr_neg(a: Fr256) -> Fr256 {
    if (fr_is_zero(a)) { return a; }
    return fr_sub(BN254_R, a);
}

// 32x32 -> 64-bit multiply returning (lo, hi).
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

// Schoolbook 256x256 -> 512-bit multiply.
fn mul256x256(a: Fr256, b: Fr256) -> array<u32, 16> {
    var t: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) { t[i] = 0u; }
    let a_arr = array<u32, 8>(a.l0, a.l1, a.l2, a.l3, a.l4, a.l5, a.l6, a.l7);
    let b_arr = array<u32, 8>(b.l0, b.l1, b.l2, b.l3, b.l4, b.l5, b.l6, b.l7);

    // Canonical three-way carry-handling pattern (matches mont_mul_n in
    // luxcpp/metal/src/shaders/crypto/secp256k1.metal). Splits the
    // (t + lo + carry) accumulation into two steps so a double-overflow
    // is correctly attributed.
    for (var i = 0u; i < 8u; i++) {
        var carry = 0u;
        for (var j = 0u; j < 8u; j++) {
            let prod = mul32(a_arr[i], b_arr[j]);
            let s1 = t[i + j] + prod.x;
            let c1 = select(0u, 1u, s1 < prod.x);
            let s2 = s1 + carry;
            let c2 = select(0u, 1u, s2 < carry);
            t[i + j] = s2;
            carry = prod.y + c1 + c2;
        }
        t[i + 8u] = carry;
    }
    return t;
}

fn fr_mont_reduce(t_in: array<u32, 16>) -> Fr256 {
    var temp = t_in;
    let r_arr = array<u32, 8>(
        BN254_R.l0, BN254_R.l1, BN254_R.l2, BN254_R.l3,
        BN254_R.l4, BN254_R.l5, BN254_R.l6, BN254_R.l7
    );
    for (var i = 0u; i < 8u; i++) {
        let m = temp[i] * BN254_R_INV;
        var carry = 0u;
        for (var j = 0u; j < 8u; j++) {
            let prod = mul32(m, r_arr[j]);
            let s1 = temp[i + j] + prod.x;
            let c1 = select(0u, 1u, s1 < prod.x);
            let s2 = s1 + carry;
            let c2 = select(0u, 1u, s2 < carry);
            temp[i + j] = s2;
            carry = prod.y + c1 + c2;
        }
        for (var j = i + 8u; j < 16u; j++) {
            let s = temp[j] + carry;
            carry = select(0u, 1u, s < temp[j]);
            temp[j] = s;
            if (carry == 0u) { break; }
        }
    }
    var result = Fr256(temp[8], temp[9], temp[10], temp[11],
                       temp[12], temp[13], temp[14], temp[15]);
    let sub_result = fr_sub_raw(result, BN254_R);
    if (sub_result.y.l0 == 0u) {
        result = sub_result.x;
    }
    return result;
}

fn fr_mul(a: Fr256, b: Fr256) -> Fr256 {
    let product = mul256x256(a, b);
    return fr_mont_reduce(product);
}

fn fr_square(a: Fr256) -> Fr256 {
    return fr_mul(a, a);
}

fn fr_to_mont(a: Fr256) -> Fr256 {
    return fr_mul(a, BN254_FR_R2);
}

fn poseidon2_sbox(x: Fr256) -> Fr256 {
    let x2 = fr_square(x);
    let x4 = fr_square(x2);
    return fr_mul(x4, x);
}

// =============================================================================
// Poseidon2-BN254 t=2 round keys (Montgomery form, derived from gnark seeds).
// =============================================================================

// 3 pre-full rounds × 2 lanes = 6 entries.
const POSEIDON2_RK_PRE_FULL = array<Fr256, 6>(
    Fr256(0x0ef7ffceu, 0x3e178f38u, 0x3206d4cau, 0xb41c86fbu, 0xeaffe80bu, 0x353e2359u, 0x9f58a8f7u, 0x1034b2bdu),
    Fr256(0xab17c599u, 0xc7872ae1u, 0x3e3ade07u, 0x718a70f8u, 0x2d09dad2u, 0x3c4b61c9u, 0xb03b8101u, 0x03c28439u),
    Fr256(0x8349deacu, 0x12ff67b8u, 0xf1024fe1u, 0x23f977ecu, 0x086d542eu, 0xbb190cd9u, 0xb6022d52u, 0x1ce80bf7u),
    Fr256(0x4622045au, 0x62575d33u, 0x190da3f2u, 0xdc1b41a1u, 0xca37ecfdu, 0x666cad8du, 0xbb4e82b3u, 0x224e59c9u),
    Fr256(0x6f694d1au, 0xa4512f78u, 0xf68bc1c0u, 0x87326258u, 0xba08d638u, 0x916a1677u, 0x9f1865a5u, 0x0a2a6bdcu),
    Fr256(0xbc8502c5u, 0x1cfdce12u, 0xd24e3fcfu, 0x2cea8ba6u, 0x3da698f8u, 0x21f145a2u, 0x1df27eeeu, 0x2ddbecf9u)
);

const POSEIDON2_RK_PARTIAL = array<Fr256, 50>(
    Fr256(0x64db7745u, 0xee282afdu, 0x0eca2f95u, 0x4b15b9e9u, 0x6e2dd360u, 0x261026abu, 0x24528c1du, 0x2029209au),
    Fr256(0x848f4f4bu, 0xeb6f7dc7u, 0x22da499cu, 0xad03394du, 0xb57e8157u, 0x23a7a079u, 0x3ab1aaadu, 0x1d3bc667u),
    Fr256(0x39ad77a7u, 0x4d76a6a8u, 0xc8de0dd4u, 0x489f866bu, 0x8610cb7fu, 0x5c9b1575u, 0xd289bfddu, 0x04d5b745u),
    Fr256(0x24969b37u, 0x93ebc206u, 0x6b54dfe8u, 0xc8768a70u, 0x4babf443u, 0x28ee6ad1u, 0x32dfeb3fu, 0x0018645au),
    Fr256(0xbce0860fu, 0xfe0f7b8du, 0x94af0cbcu, 0xfd790251u, 0x8f413e52u, 0x8d0e0775u, 0x5469d684u, 0x1f63f050u),
    Fr256(0x16772e71u, 0x840b6fd2u, 0x37071f67u, 0xbe889389u, 0xac076648u, 0x0afa9077u, 0x426a92c3u, 0x2d767745u),
    Fr256(0x276961f3u, 0xb71e5f25u, 0x2cea1662u, 0xdc1f319fu, 0x37fdc929u, 0x8925cbe2u, 0x60f3f925u, 0x09a03ac0u),
    Fr256(0x1ead1e6eu, 0xe3c86ae4u, 0x1ede85cau, 0xdeda3920u, 0xcbc449a3u, 0xdd67fdb3u, 0x1e6cad9cu, 0x22cd461au),
    Fr256(0x81c9d8eau, 0x3b1fd68cu, 0x5345ebd1u, 0xa99d2cc0u, 0x88cdbf65u, 0x58e8fff4u, 0x8d92dd57u, 0x109a56d3u),
    Fr256(0xe0316a28u, 0xca1a5862u, 0x872c8256u, 0x04c9f474u, 0xde1e175du, 0x37a2e0d3u, 0x77ddb206u, 0x1d135c57u),
    Fr256(0xaeb9fa02u, 0x366f2bb7u, 0xfe208d24u, 0x2ba6a4c8u, 0xb5eb4a74u, 0x53ff3c46u, 0xe17a5351u, 0x03de8a98u),
    Fr256(0x186941e7u, 0xc3999c9bu, 0x9cf864edu, 0x0f9dd175u, 0x84584741u, 0xd0764581u, 0xf59294c4u, 0x2bcd0790u),
    Fr256(0xe44cc0acu, 0x3991379cu, 0x305847ceu, 0x990bb88au, 0x286383edu, 0x4f16d43eu, 0x229ea886u, 0x0e8a8befu),
    Fr256(0x5cd10ca1u, 0x5260fba8u, 0xd18e649du, 0x2ff0852bu, 0xf745646bu, 0xea6bcc17u, 0x418fe76eu, 0x0f4892d2u),
    Fr256(0x883f78feu, 0xf882b58cu, 0xdf6f1781u, 0x20315c18u, 0xe7f7d81bu, 0x64b8f2e0u, 0x255be1f1u, 0x16c29d7cu),
    Fr256(0x34cd9e08u, 0x8b5bf868u, 0x8af0d6a7u, 0xfb1af17eu, 0xaa4c44d2u, 0xecd851f3u, 0x38dad61eu, 0x12697a99u),
    Fr256(0xd58141acu, 0xa038849eu, 0x4837656fu, 0x8f8b791eu, 0x0fc62aeau, 0x9e18e2f4u, 0xcb86ecfdu, 0x05a11fa0u),
    Fr256(0x5e53760du, 0xa91dfdfcu, 0xd0252491u, 0x68cec7bcu, 0xf03f9a50u, 0x6f4d3aa9u, 0x34797babu, 0x125dfad2u),
    Fr256(0x6edf8932u, 0xf13b7445u, 0x406c10f9u, 0x724e0aedu, 0xe920dac2u, 0xf50a8006u, 0xfa8ca312u, 0x095e73abu),
    Fr256(0xb4c531c6u, 0x012a09a0u, 0xa364d0f8u, 0x7093e46bu, 0x131d8aefu, 0xa4f83255u, 0xcbe60759u, 0x054b61b4u),
    Fr256(0x5d864c65u, 0xa0eee0a1u, 0xeee33742u, 0x73622b94u, 0x7028c988u, 0xc34cf43bu, 0x7c60ea50u, 0x0978422eu),
    Fr256(0x74968aeeu, 0x27a04635u, 0xf4e6e11fu, 0xc257ff22u, 0xd81bb00bu, 0xbfbfebb8u, 0xffaa3704u, 0x0820fbe4u),
    Fr256(0x9544d468u, 0x0e324c88u, 0x40dcb30du, 0x12183378u, 0x6532d74fu, 0x52ac5114u, 0x2b949d3fu, 0x15e37117u),
    Fr256(0xc5479c04u, 0xc1813033u, 0x95c14c8du, 0x1c45181du, 0x4bb060c2u, 0x26e34033u, 0xa160ac93u, 0x1a167926u),
    Fr256(0x2fbfbd61u, 0xe164d2f6u, 0xbfa3a1a5u, 0xe9879643u, 0x4c383d94u, 0xd9062e38u, 0x81b632abu, 0x2a77a479u),
    Fr256(0x26c29c49u, 0xcdd823a3u, 0xf7587e3eu, 0x84d5fab7u, 0xb6bd7ed2u, 0x720ea5e7u, 0xf469b87cu, 0x0ff823b4u),
    Fr256(0xd8c1d6dfu, 0x3e5d4b46u, 0x9a7ed258u, 0x31ac1cfbu, 0x02c31b48u, 0x4b8e94a6u, 0x995f4c90u, 0x2344ee31u),
    Fr256(0x2b1354d2u, 0x1e358ad5u, 0x73d79fdau, 0xe45eb9aau, 0xf9592652u, 0x3a8ae980u, 0x39e0eb3cu, 0x01e37a51u),
    Fr256(0x1114e4a5u, 0xb435dfc8u, 0x11e21e5du, 0xda7a6af2u, 0xbbee0616u, 0x0250a718u, 0xe62e2d46u, 0x2b433386u),
    Fr256(0x296a1454u, 0xab7431d0u, 0x5e11ff9fu, 0x0410d5d1u, 0x15a079a0u, 0x1152785au, 0xaf67c714u, 0x1bd516a9u),
    Fr256(0x7e794552u, 0x1c31b01fu, 0xf59ce62fu, 0x7b482d1au, 0x489dffe8u, 0xc90c5a46u, 0x2a7a1274u, 0x2c906709u),
    Fr256(0x35965945u, 0x3b91d722u, 0x0d170cb3u, 0x10a82874u, 0xdb243d3cu, 0xf27ba8e7u, 0x2b3acc9du, 0x2332dc5du),
    Fr256(0x0f687771u, 0x62b9864eu, 0x8ac56ccdu, 0x3a765cfau, 0xaab16545u, 0x5482c7b8u, 0xae5d9e0fu, 0x221c8363u),
    Fr256(0xa41681ddu, 0xa281a6bfu, 0xb3337c07u, 0x93dcdbc4u, 0xc13e6a93u, 0xb7f4c0d0u, 0xb5a957a0u, 0x124d7c53u),
    Fr256(0x6d5597f5u, 0xf984ecc5u, 0x6fa41af5u, 0x54b4e675u, 0xfeea4e86u, 0xd16a1316u, 0x5036cbadu, 0x10860d00u),
    Fr256(0x8725e308u, 0x82af3503u, 0xacf05599u, 0x701073d7u, 0x7ed9325au, 0x2507bc04u, 0xdf722e35u, 0x000866ffu),
    Fr256(0x7f5a3c32u, 0x38a974e8u, 0x0d318e71u, 0xbc0fbea7u, 0x62cdf71fu, 0x3ff9b911u, 0x6e8574ddu, 0x22b30cbfu),
    Fr256(0xfe1f2a6bu, 0xad2098feu, 0x2c0d9d77u, 0x0def2464u, 0xbb853dc2u, 0xde2bc407u, 0x44578b69u, 0x103f85d5u),
    Fr256(0x75012379u, 0xd44659c4u, 0x5bbc5a2au, 0x8d4aadbbu, 0x70d33b94u, 0x9794f155u, 0x6bfe6b81u, 0x0db232b8u),
    Fr256(0x658c465bu, 0xc0224690u, 0xaf61c2a4u, 0xc4fbf5a2u, 0x2ba03253u, 0x6a1f5f07u, 0xa53157f6u, 0x203d0438u),
    Fr256(0x0b034e67u, 0xeacee4efu, 0xafcefafbu, 0xceafb0edu, 0x99a95fc2u, 0x9776b469u, 0xa05dc1a6u, 0x1d3a57a1u),
    Fr256(0x4b7770e5u, 0x896e56b9u, 0x6e4405e7u, 0x4a231a0fu, 0x0ed24352u, 0x9c6e80e1u, 0x51e594c6u, 0x27993bf4u),
    Fr256(0x69e8c88fu, 0x248ff0e1u, 0x7a24365eu, 0xbef8c0fdu, 0xf6869e3bu, 0xd528071cu, 0xcb8e4b9cu, 0x0df561a6u),
    Fr256(0x8983f03du, 0x5be6f307u, 0xa60d4623u, 0x8954b751u, 0xb0486f90u, 0xd44d0d2cu, 0xf6d1b56cu, 0x195539bcu),
    Fr256(0xf313d679u, 0x75044b91u, 0xe431a315u, 0x847cac18u, 0xfb2aa9e0u, 0x29b9326fu, 0xee3af35eu, 0x1873dd9du),
    Fr256(0x28152077u, 0x97a6713du, 0x9cee4969u, 0xae467322u, 0xf0d60f8eu, 0xcd8a1ad4u, 0xdf613d1cu, 0x26f3b531u),
    Fr256(0x91c760acu, 0x932d607au, 0xd5f2d71fu, 0x83c154f0u, 0x396a3e6au, 0x0624312au, 0x3d26cf0au, 0x0f314c81u),
    Fr256(0x48b0b19au, 0x47a84b34u, 0x515c3fa1u, 0x4e7468ecu, 0x0610be66u, 0x19624710u, 0x488017cbu, 0x038b7cb6u),
    Fr256(0x24307cf0u, 0xd95e1062u, 0xc8d6ef8du, 0x3d14d008u, 0xa21f99b8u, 0x330b20c4u, 0x23f72859u, 0x01a08d60u),
    Fr256(0x6f83354fu, 0x3c47bf71u, 0x0e59604fu, 0x94ab8f51u, 0x0e5f74a1u, 0x27af63bcu, 0x3aaaccbdu, 0x1d85c51eu)
);

const POSEIDON2_RK_POST_FULL = array<Fr256, 6>(
    Fr256(0xc325d3edu, 0x63f96ee9u, 0xeadfb2dau, 0x09a64e7bu, 0x4ac45276u, 0x031c2f7cu, 0x09e353e7u, 0x2b403215u),
    Fr256(0x4b2e6220u, 0xd9c7d0f1u, 0xc7a7f7bdu, 0x3d5b785fu, 0xa2872faeu, 0x2ce41049u, 0xfe795dc0u, 0x08435a73u),
    Fr256(0x6119afd8u, 0x8177875cu, 0x27a9f6e2u, 0xa1925d18u, 0x57f2e7f4u, 0xd9936cbbu, 0xd338651du, 0x00f6b236u),
    Fr256(0xadf5c8c2u, 0xd7181597u, 0xdfa2f936u, 0x97c52d91u, 0xc360830au, 0x0ac7aa7fu, 0xd5f9b2c4u, 0x194d95f0u),
    Fr256(0x0fd73235u, 0x40cb5c4cu, 0x8254dd85u, 0x50c1bf92u, 0x540f5cd1u, 0x376868a8u, 0x00de28c6u, 0x2f19c117u),
    Fr256(0x0f8e60e8u, 0x9abe0ad9u, 0xb40248f8u, 0x2892c140u, 0x20f7001cu, 0xf4742b98u, 0x77341015u, 0x07d9ed29u)
);

// =============================================================================
// Linear layers and permutation (t=2).
// =============================================================================

struct State2 { s0: Fr256, s1: Fr256, };

fn mat_external(st: State2) -> State2 {
    let sum = fr_add(st.s0, st.s1);
    return State2(fr_add(st.s0, sum), fr_add(st.s1, sum));
}

fn mat_internal(st: State2) -> State2 {
    let sum = fr_add(st.s0, st.s1);
    let s0n = fr_add(st.s0, sum);
    let two_s1 = fr_add(st.s1, st.s1);
    let s1n = fr_add(two_s1, sum);
    return State2(s0n, s1n);
}

fn permutation_t2(st_in: State2) -> State2 {
    var st = mat_external(st_in);
    for (var i = 0u; i < 3u; i++) {
        st.s0 = fr_add(st.s0, POSEIDON2_RK_PRE_FULL[i * 2u]);
        st.s1 = fr_add(st.s1, POSEIDON2_RK_PRE_FULL[i * 2u + 1u]);
        st.s0 = poseidon2_sbox(st.s0);
        st.s1 = poseidon2_sbox(st.s1);
        st = mat_external(st);
    }
    for (var i = 0u; i < 50u; i++) {
        st.s0 = fr_add(st.s0, POSEIDON2_RK_PARTIAL[i]);
        st.s0 = poseidon2_sbox(st.s0);
        st = mat_internal(st);
    }
    for (var i = 0u; i < 3u; i++) {
        st.s0 = fr_add(st.s0, POSEIDON2_RK_POST_FULL[i * 2u]);
        st.s1 = fr_add(st.s1, POSEIDON2_RK_POST_FULL[i * 2u + 1u]);
        st.s0 = poseidon2_sbox(st.s0);
        st.s1 = poseidon2_sbox(st.s1);
        st = mat_external(st);
    }
    return st;
}

// =============================================================================
// Bindings & kernels
// =============================================================================

struct Poseidon2Params {
    count: u32,
    path_len: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read>       left_inputs: array<Fr256>;
@group(0) @binding(1) var<storage, read>       right_inputs: array<Fr256>;
@group(0) @binding(2) var<storage, read_write> outputs: array<Fr256>;
@group(0) @binding(3) var<uniform>             params: Poseidon2Params;

// 2-to-1 compress (gnark Davies-Meyer).
@compute @workgroup_size(64)
fn poseidon2_hash_pair(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    let l_can = left_inputs[idx];
    let r_can = right_inputs[idx];
    let s0_in = fr_to_mont(l_can);
    let s1_in = fr_to_mont(r_can);
    let r_mont = s1_in;
    let perm = permutation_t2(State2(s0_in, s1_in));
    let sum_mont = fr_add(perm.s1, r_mont);
    // Drop the Montgomery factor: mul by canonical 1.
    let one_can = Fr256(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    outputs[idx] = fr_mul(sum_mont, one_can);
}

// Build one layer of Merkle tree.
@compute @workgroup_size(64)
fn poseidon2_merkle_layer(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count / 2u) { return; }
    let l_can = left_inputs[2u * idx];
    let r_can = left_inputs[2u * idx + 1u];
    let s0_in = fr_to_mont(l_can);
    let s1_in = fr_to_mont(r_can);
    let r_mont = s1_in;
    let perm = permutation_t2(State2(s0_in, s1_in));
    let sum_mont = fr_add(perm.s1, r_mont);
    let one_can = Fr256(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    outputs[idx] = fr_mul(sum_mont, one_can);
}

// Merkle proof verification — chained t=2 Compress.
@group(0) @binding(4) var<storage, read>       leaves: array<Fr256>;
@group(0) @binding(5) var<storage, read>       path: array<Fr256>;
@group(0) @binding(6) var<storage, read>       path_indices: array<u32>;
@group(0) @binding(7) var<storage, read>       expected_roots: array<Fr256>;
@group(0) @binding(8) var<storage, read_write> verify_results: array<u32>;

@compute @workgroup_size(64)
fn poseidon2_verify_merkle_proof(@builtin(global_invocation_id) gid: vec3<u32>) {
    let proof_idx = gid.x;
    if (proof_idx >= params.count) { return; }
    var current = leaves[proof_idx];
    for (var i = 0u; i < params.path_len; i++) {
        let sibling = path[proof_idx * params.path_len + i];
        let sel = path_indices[proof_idx * params.path_len + i];
        var l_can: Fr256;
        var r_can: Fr256;
        if (sel == 0u) {
            l_can = current; r_can = sibling;
        } else {
            l_can = sibling; r_can = current;
        }
        let s0_in = fr_to_mont(l_can);
        let s1_in = fr_to_mont(r_can);
        let r_mont = s1_in;
        let perm = permutation_t2(State2(s0_in, s1_in));
        let sum_mont = fr_add(perm.s1, r_mont);
        let one_can = Fr256(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
        current = fr_mul(sum_mont, one_can);
    }
    let expected = expected_roots[proof_idx];
    let valid = (current.l0 == expected.l0) && (current.l1 == expected.l1) &&
                (current.l2 == expected.l2) && (current.l3 == expected.l3) &&
                (current.l4 == expected.l4) && (current.l5 == expected.l5) &&
                (current.l6 == expected.l6) && (current.l7 == expected.l7);
    verify_results[proof_idx] = select(0u, 1u, valid);
}

// Nullifier = Compress(Compress(nullifier_key, note_commitment), leaf_index).
@group(0) @binding(9)  var<storage, read>       nullifier_keys: array<Fr256>;
@group(0) @binding(10) var<storage, read>       note_commitments: array<Fr256>;
@group(0) @binding(11) var<storage, read>       leaf_indices: array<Fr256>;
@group(0) @binding(12) var<storage, read_write> nullifiers: array<Fr256>;

fn compress_pair(l_can: Fr256, r_can: Fr256) -> Fr256 {
    let s0_in = fr_to_mont(l_can);
    let s1_in = fr_to_mont(r_can);
    let r_mont = s1_in;
    let perm = permutation_t2(State2(s0_in, s1_in));
    let sum_mont = fr_add(perm.s1, r_mont);
    let one_can = Fr256(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    return fr_mul(sum_mont, one_can);
}

@compute @workgroup_size(64)
fn poseidon2_nullifier(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    let first = compress_pair(nullifier_keys[idx], note_commitments[idx]);
    nullifiers[idx] = compress_pair(first, leaf_indices[idx]);
}

// Commitment = Compress(Compress(value, blinding), salt).
@group(0) @binding(13) var<storage, read> values: array<Fr256>;
@group(0) @binding(14) var<storage, read> blindings: array<Fr256>;
@group(0) @binding(15) var<storage, read> salts: array<Fr256>;

@compute @workgroup_size(64)
fn poseidon2_commitment(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    let first = compress_pair(values[idx], blindings[idx]);
    outputs[idx] = compress_pair(first, salts[idx]);
}
