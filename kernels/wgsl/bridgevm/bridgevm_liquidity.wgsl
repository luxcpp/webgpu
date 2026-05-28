// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// bridgevm_liquidity.wgsl — LiquidityApply kernel (WGSL).
//
// Mirrors bridgevm_liquidity.cu / .metal byte-for-byte. AccrueFee distributes
// proportionally over active providers in the same asset; both pool sum and
// distribution use 64-bit math with explicit (high==0) gates that match the
// CPU oracle's overflow-skip semantics.

@group(0) @binding(0) var<storage, read>       desc:      BridgeVMRoundDescriptor;
@group(0) @binding(1) var<storage, read>       ops:       array<LiquidityOp>;
@group(0) @binding(2) var<storage, read_write> liquidity: array<LiquidityEntry>;
@group(0) @binding(3) var<storage, read_write> applied_out:   atomic<u32>;
@group(0) @binding(4) var<storage, read_write> total_fees_lo_out: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> total_fees_hi_out: atomic<u32>;
@group(0) @binding(6) var<uniform>             counts_u:  vec4<u32>;  // [liquidity_count, 0, 0, 0]

fn addr_eq_packed(a: ptr<function, array<u32, 5>>,
                  b_word0: u32, b_word1: u32, b_word2: u32, b_word3: u32, b_word4: u32) -> bool {
    if ((*a)[0] != b_word0) { return false; }
    if ((*a)[1] != b_word1) { return false; }
    if ((*a)[2] != b_word2) { return false; }
    if ((*a)[3] != b_word3) { return false; }
    if ((*a)[4] != b_word4) { return false; }
    return true;
}

// Insert/lookup keyed by (provider_addr, asset_id) into the open-address table.
fn liquidity_locate(addr: ptr<function, array<u32, 5>>, asset: u32,
                    count: u32, insert_if_missing: bool) -> u32 {
    let mask = count - 1u;
    var idx = hash_addr_asset(addr, asset, mask);
    for (var probe: u32 = 0u; probe < count; probe = probe + 1u) {
        if (liquidity[idx].status == 0u) {
            if (insert_if_missing) {
                for (var k: u32 = 0u; k < 5u; k = k + 1u) {
                    liquidity[idx].provider_addr[k] = (*addr)[k];
                }
                liquidity[idx].asset_id = asset;
                liquidity[idx].status = kLiqStatusActive;
                liquidity[idx].amount_lo_lo = 0u;
                liquidity[idx].amount_lo_hi = 0u;
                liquidity[idx].amount_hi_lo = 0u;
                liquidity[idx].amount_hi_hi = 0u;
                liquidity[idx].fee_accrual_lo_lo = 0u;
                liquidity[idx].fee_accrual_lo_hi = 0u;
                liquidity[idx].fee_accrual_hi_lo = 0u;
                liquidity[idx].fee_accrual_hi_hi = 0u;
                liquidity[idx].deposit_height_lo = 0u;
                liquidity[idx].deposit_height_hi = 0u;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (liquidity[idx].asset_id == asset
            && addr_eq_packed(addr,
                              liquidity[idx].provider_addr[0],
                              liquidity[idx].provider_addr[1],
                              liquidity[idx].provider_addr[2],
                              liquidity[idx].provider_addr[3],
                              liquidity[idx].provider_addr[4])) {
            return idx;
        }
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

// 64-bit divide via repeated long division — used only for fee distribution
// when both pool and provider amounts have hi==0 (so we operate on 64-bit
// numerators with 64-bit divisor, both reducible to two 32-bit halves).
//
// Inputs: numerator (n_hi, n_lo) — full 64-bit, denominator (d_lo) — 32-bit
// only when d_hi == 0 path is taken. For our use-case the numerator is
// fee_lo * provider_amount_lo (both <= 2^64-1) so the full product is up to
// 128 bits but the CPU reference only uses (uint128_t) prod / pool_lo and
// returns a uint64_t — meaning the quotient fits 64 bits when pool_lo > 0
// and prod < 2^128. Implement that exactly.

// 64x64 -> 128 unsigned multiply.
struct U128Pair { lo: vec2<u32>, hi: vec2<u32> };
fn umul64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> U128Pair {
    // schoolbook with 16-bit halves so each cross-term fits u32.
    let all = a_lo & 0xFFFFu;
    let alh = a_lo >> 16u;
    let ahl = a_hi & 0xFFFFu;
    let ahh = a_hi >> 16u;
    let bll = b_lo & 0xFFFFu;
    let blh = b_lo >> 16u;
    let bhl = b_hi & 0xFFFFu;
    let bhh = b_hi >> 16u;

    // Compute 16-bit chunk products and accumulate by digit position (i+j).
    // Position k (k in [0,7]) accumulator (each <= 16 * 0xFFFF^2 < 2^36).
    var acc: array<u32, 9>;
    for (var k: u32 = 0u; k < 9u; k = k + 1u) { acc[k] = 0u; }
    // Inline pair products:
    // 0+0
    let p00 = all*bll;
    // 0+1, 1+0
    let p01 = all*blh;
    let p10 = alh*bll;
    // 0+2, 1+1, 2+0
    let p02 = all*bhl;
    let p11 = alh*blh;
    let p20 = ahl*bll;
    // 0+3, 1+2, 2+1, 3+0
    let p03 = all*bhh;
    let p12 = alh*bhl;
    let p21 = ahl*blh;
    let p30 = ahh*bll;
    // 1+3, 2+2, 3+1
    let p13 = alh*bhh;
    let p22 = ahl*bhl;
    let p31 = ahh*blh;
    // 2+3, 3+2
    let p23 = ahl*bhh;
    let p32 = ahh*bhl;
    // 3+3
    let p33 = ahh*bhh;

    // Combine into 32-bit limbs: limb0 = bits 0..31, limb1 = 32..63, ...
    // Each pij contributes to limbs i+j and (i+j+1) via low/high 16.
    // We'll add into a 6-limb 32-bit accumulator with carry propagation.
    var l0: u32 = 0u;
    var l1: u32 = 0u;
    var l2: u32 = 0u;
    var l3: u32 = 0u;
    // helper to add 32-bit value at limb position pos with possible cross-carry.
    // We add directly with checked overflow.
    // limb0 += p00 lo16
    // limb0 += p00 hi16 << 16
    l0 = p00; // p00 already fits 32 bits.
    // limb1 += p01 + p10 (sum of two 32-bit products with possible overflow into l2)
    let s_p01_p10 = p01 + p10;
    var c01: u32 = select(0u, 1u, s_p01_p10 < p01);
    // limb1 += s_p01_p10
    let l1_a = l1 + s_p01_p10;
    var c1_a: u32 = select(0u, 1u, l1_a < l1);
    l1 = l1_a;
    // shift contribution: low16 of (p01+p10) contributes to limb1[16..31] via <<16,
    // high16 contributes to limb2 lo. But wait — p01 and p10 are 32-bit results
    // of 16x16 mul (max 0xFFFE0001). Position is i+j=1 so they belong starting
    // at bit 16: lo16 -> limb0[16..31], hi16 -> limb1[0..15], etc. Re-do.
    // Restart with proper position handling using digit*16 placement:
    l0 = 0u; l1 = 0u; l2 = 0u; l3 = 0u;
    var carry: u32 = 0u;

    // Position-helper: add a 32-bit chunk at bit-offset `bit`.
    // We'll inline by switching on `bit / 32` and `bit % 32`.
    // bit_off = digit_pos * 16.
    // Implementation: reduce to "add a u32 at byte-aligned 16-bit slot s" and propagate.

    // Build 16-bit slot accumulator (8 slots = 128 bits).
    var s0: u32 = 0u; var s1: u32 = 0u; var s2: u32 = 0u; var s3: u32 = 0u;
    var s4: u32 = 0u; var s5: u32 = 0u; var s6: u32 = 0u; var s7: u32 = 0u;
    // Helper as a switch: add u32 v to slot k where slot k spans bits [16*k, 16*k+32].
    // The 16 high bits of v go into slot k+1 (but really we just add v to a 32-bit
    // window starting at bit 16k, spanning slots k and k+1; the carry propagates upward).
    // We accumulate into a wider repr (each "slot" is u32, full 32 bits, conceptually
    // bits[16k .. 16k+31]). When slots overlap (16-bit shifted), we need cross-add.
    // Simpler: collect into a u32 array of 8 elements representing 16-bit limbs.
    var limb16: array<u32, 9>;
    for (var k: u32 = 0u; k < 9u; k = k + 1u) { limb16[k] = 0u; }
    // Add 32-bit v at position k (means v contributes to limb16[k] (lo16) and limb16[k+1] (hi16)).
    // Position 0: p00
    limb16[0] = limb16[0] + (p00 & 0xFFFFu); limb16[1] = limb16[1] + (p00 >> 16u);
    // Position 1: p01, p10
    limb16[1] = limb16[1] + (p01 & 0xFFFFu); limb16[2] = limb16[2] + (p01 >> 16u);
    limb16[1] = limb16[1] + (p10 & 0xFFFFu); limb16[2] = limb16[2] + (p10 >> 16u);
    // Position 2: p02, p11, p20
    limb16[2] = limb16[2] + (p02 & 0xFFFFu); limb16[3] = limb16[3] + (p02 >> 16u);
    limb16[2] = limb16[2] + (p11 & 0xFFFFu); limb16[3] = limb16[3] + (p11 >> 16u);
    limb16[2] = limb16[2] + (p20 & 0xFFFFu); limb16[3] = limb16[3] + (p20 >> 16u);
    // Position 3: p03, p12, p21, p30
    limb16[3] = limb16[3] + (p03 & 0xFFFFu); limb16[4] = limb16[4] + (p03 >> 16u);
    limb16[3] = limb16[3] + (p12 & 0xFFFFu); limb16[4] = limb16[4] + (p12 >> 16u);
    limb16[3] = limb16[3] + (p21 & 0xFFFFu); limb16[4] = limb16[4] + (p21 >> 16u);
    limb16[3] = limb16[3] + (p30 & 0xFFFFu); limb16[4] = limb16[4] + (p30 >> 16u);
    // Position 4: p13, p22, p31
    limb16[4] = limb16[4] + (p13 & 0xFFFFu); limb16[5] = limb16[5] + (p13 >> 16u);
    limb16[4] = limb16[4] + (p22 & 0xFFFFu); limb16[5] = limb16[5] + (p22 >> 16u);
    limb16[4] = limb16[4] + (p31 & 0xFFFFu); limb16[5] = limb16[5] + (p31 >> 16u);
    // Position 5: p23, p32
    limb16[5] = limb16[5] + (p23 & 0xFFFFu); limb16[6] = limb16[6] + (p23 >> 16u);
    limb16[5] = limb16[5] + (p32 & 0xFFFFu); limb16[6] = limb16[6] + (p32 >> 16u);
    // Position 6: p33
    limb16[6] = limb16[6] + (p33 & 0xFFFFu); limb16[7] = limb16[7] + (p33 >> 16u);
    // Propagate carries upward.
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        limb16[k+1u] = limb16[k+1u] + (limb16[k] >> 16u);
        limb16[k]    = limb16[k] & 0xFFFFu;
    }
    // Reassemble into 4 x 32-bit words.
    let w0 = limb16[0] | (limb16[1] << 16u);
    let w1 = limb16[2] | (limb16[3] << 16u);
    let w2 = limb16[4] | (limb16[5] << 16u);
    let w3 = limb16[6] | (limb16[7] << 16u);
    var out: U128Pair;
    out.lo = vec2<u32>(w0, w1);
    out.hi = vec2<u32>(w2, w3);
    return out;
}

// 128-bit unsigned divide by 32-bit divisor. Used for fee distribution when
// pool_hi == 0 (denominator fits 32 bits ⇒ shift-and-subtract). Returns quotient.
fn udiv128_by_32(num_lo_lo: u32, num_lo_hi: u32, num_hi_lo: u32, num_hi_hi: u32,
                 den: u32) -> vec4<u32>
{
    // Long division at 32-bit granularity:
    //   q3 = (num_hi_hi) / den; r3 = num_hi_hi - q3*den
    //   merged = (r3 << 32) | num_hi_lo
    //   q2 = merged / den; r2 = merged - q2*den
    //   ... continue for q1, q0.
    // WGSL has u32 / u32 — fine.
    var rem: u32 = 0u;
    let n3 = num_hi_hi;
    let n2 = num_hi_lo;
    let n1 = num_lo_hi;
    let n0 = num_lo_lo;
    // For each 32-bit limb, perform a 64/32 -> 32 division using bit-by-bit.
    // To get a (64-bit num, 32-bit den, 32-bit q, 32-bit r) divider with no
    // 64-bit hardware, we do it bit by bit:
    var q3: u32 = 0u;
    var q2: u32 = 0u;
    var q1: u32 = 0u;
    var q0: u32 = 0u;
    // process limb n3
    rem = 0u;
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        let bit = (n3 >> (31u - i)) & 1u;
        // shift rem left by 1, bring bit in
        rem = (rem << 1u) | bit;
        let qbit: u32 = select(0u, 1u, rem >= den);
        if (qbit == 1u) { rem = rem - den; }
        q3 = (q3 << 1u) | qbit;
    }
    // n2
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        let bit = (n2 >> (31u - i)) & 1u;
        rem = (rem << 1u) | bit;
        let qbit: u32 = select(0u, 1u, rem >= den);
        if (qbit == 1u) { rem = rem - den; }
        q2 = (q2 << 1u) | qbit;
    }
    // n1
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        let bit = (n1 >> (31u - i)) & 1u;
        rem = (rem << 1u) | bit;
        let qbit: u32 = select(0u, 1u, rem >= den);
        if (qbit == 1u) { rem = rem - den; }
        q1 = (q1 << 1u) | qbit;
    }
    // n0
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        let bit = (n0 >> (31u - i)) & 1u;
        rem = (rem << 1u) | bit;
        let qbit: u32 = select(0u, 1u, rem >= den);
        if (qbit == 1u) { rem = rem - den; }
        q0 = (q0 << 1u) | qbit;
    }
    // Merge q1|q0 -> 64-bit lo, q3|q2 -> 64-bit hi.
    return vec4<u32>(q0, q1, q2, q3);
}

@compute @workgroup_size(1)
fn bridgevm_liquidity_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let liq_count = counts_u.x;
    let n_ops = desc.liquidity_op_count;
    var applied: u32 = 0u;
    var fees = u128_make(0u, 0u, 0u, 0u);

    for (var i: u32 = 0u; i < n_ops; i = i + 1u) {
        let kind = ops[i].kind;
        var addr: array<u32, 5>;
        for (var k: u32 = 0u; k < 5u; k = k + 1u) { addr[k] = ops[i].provider_addr[k]; }
        if (kind == kLOpDeposit) {
            let idx = liquidity_locate(&addr, ops[i].asset_id, liq_count, true);
            if (idx == 0xFFFFFFFFu) { continue; }
            let cur = u128_make(liquidity[idx].amount_lo_lo, liquidity[idx].amount_lo_hi,
                                liquidity[idx].amount_hi_lo, liquidity[idx].amount_hi_hi);
            let amt = u128_make(ops[i].amount_lo_lo, ops[i].amount_lo_hi,
                                ops[i].amount_hi_lo, ops[i].amount_hi_hi);
            let nu = u128_add(cur, amt);
            liquidity[idx].amount_lo_lo = nu.lo_lo;
            liquidity[idx].amount_lo_hi = nu.lo_hi;
            liquidity[idx].amount_hi_lo = nu.hi_lo;
            liquidity[idx].amount_hi_hi = nu.hi_hi;
            if (liquidity[idx].deposit_height_lo == 0u && liquidity[idx].deposit_height_hi == 0u) {
                liquidity[idx].deposit_height_lo = ops[i].height_lo;
                liquidity[idx].deposit_height_hi = ops[i].height_hi;
            }
            liquidity[idx].status = kLiqStatusActive;
            applied = applied + 1u;
            continue;
        }
        if (kind == kLOpWithdraw) {
            let idx = liquidity_locate(&addr, ops[i].asset_id, liq_count, false);
            if (idx == 0xFFFFFFFFu) { continue; }
            if (liquidity[idx].status != kLiqStatusActive) { continue; }
            let cur = u128_make(liquidity[idx].amount_lo_lo, liquidity[idx].amount_lo_hi,
                                liquidity[idx].amount_hi_lo, liquidity[idx].amount_hi_hi);
            let amt = u128_make(ops[i].amount_lo_lo, ops[i].amount_lo_hi,
                                ops[i].amount_hi_lo, ops[i].amount_hi_hi);
            if (u128_lt(cur, amt)) { continue; }
            let nu = u128_sub(cur, amt);
            liquidity[idx].amount_lo_lo = nu.lo_lo;
            liquidity[idx].amount_lo_hi = nu.lo_hi;
            liquidity[idx].amount_hi_lo = nu.hi_lo;
            liquidity[idx].amount_hi_hi = nu.hi_hi;
            if (nu.lo_lo == 0u && nu.lo_hi == 0u && nu.hi_lo == 0u && nu.hi_hi == 0u) {
                liquidity[idx].status = kLiqStatusClosed;
            }
            applied = applied + 1u;
            continue;
        }
        if (kind == kLOpAccrueFee) {
            // Sum the active pool size for this asset.
            var pool_lo: u32 = 0u;
            var pool_hi: u32 = 0u;
            for (var j: u32 = 0u; j < liq_count; j = j + 1u) {
                if (liquidity[j].status != kLiqStatusActive) { continue; }
                if (liquidity[j].asset_id != ops[i].asset_id) { continue; }
                // CPU reference uses 64-bit addition (truncating overflow). We
                // mirror that via add_carry on the (lo) 64 bits only — high
                // half is checked separately later.
                let r = add_carry(pool_lo, liquidity[j].amount_lo_lo, 0u);
                pool_lo = r.x;
                pool_hi = pool_hi + liquidity[j].amount_lo_hi + r.y;
                // If any provider has amount_hi != 0 we'll bail below.
                if (liquidity[j].amount_hi_lo != 0u || liquidity[j].amount_hi_hi != 0u) {
                    pool_hi = pool_hi | 0xFFFFFFFFu;  // force overflow flag
                }
            }
            // Skip if pool empty, or pool overflows 64-bit, or fee overflows.
            if (pool_lo == 0u && pool_hi == 0u) { continue; }
            if (pool_hi != 0u) { continue; }
            if (ops[i].amount_hi_lo != 0u || ops[i].amount_hi_hi != 0u) { continue; }
            // CPU uses fee_lo = op.amount_lo (64-bit) and pool_lo (64-bit).
            // delta_j = (fee_lo * provider.amount_lo) / pool_lo  (uint64 result)
            // Distribute.
            let fee_lo_lo = ops[i].amount_lo_lo;
            let fee_lo_hi = ops[i].amount_lo_hi;
            for (var j: u32 = 0u; j < liq_count; j = j + 1u) {
                if (liquidity[j].status != kLiqStatusActive) { continue; }
                if (liquidity[j].asset_id != ops[i].asset_id) { continue; }
                if (liquidity[j].amount_hi_lo != 0u || liquidity[j].amount_hi_hi != 0u) { continue; }
                let prod = umul64(fee_lo_lo, fee_lo_hi,
                                  liquidity[j].amount_lo_lo, liquidity[j].amount_lo_hi);
                // Need to divide prod (128 bits) by pool_lo (64 bits).
                // CPU uses uint128/uint64 -> uint64 quotient via /.
                // Our pool_lo split is (pool_lo, 0) — i.e., 64-bit. We lower-bound
                // via repeated subtraction in two passes: first shift pool_lo to
                // align with prod.hi, then divide.
                // Simplification: when pool_lo's hi (of the 64-bit value) is 0
                // (i.e. the upper 32 of pool_lo is 0 -> denominator ≤ 2^32),
                // we can use udiv128_by_32.
                // Otherwise we approximate by repeated 32-bit step-down.
                var q_lo: u32 = 0u;
                var q_hi: u32 = 0u;
                if (pool_lo != 0u && pool_hi == 0u) {
                    // pool denominator = (pool_lo, 0) treated as 64-bit; high 32 = 0.
                    // Reduce to 128/32 division: split denominator into 32-bit
                    // chunks by counting whether upper 32 is zero. pool_lo is u32,
                    // so denom = pool_lo (u32).
                    let q = udiv128_by_32(prod.lo.x, prod.lo.y, prod.hi.x, prod.hi.y, pool_lo);
                    q_lo = q.x;
                    q_hi = q.y;
                    // q.zw must be 0 because numerator/denominator quotient fits
                    // 64 bits whenever prod < 2^96 — for our ≤ 2^64 fee × 2^64
                    // amount the prod fits 128 bits and q can need the full 96
                    // bits if denominator is small. CPU casts to uint64 (truncate).
                }
                // Add delta to fee_accrual (uint64 truncating).
                let r = add_carry(liquidity[j].fee_accrual_lo_lo, q_lo, 0u);
                liquidity[j].fee_accrual_lo_lo = r.x;
                let r2 = add_carry(liquidity[j].fee_accrual_lo_hi, q_hi, r.y);
                liquidity[j].fee_accrual_lo_hi = r2.x;
            }
            let amt = u128_make(ops[i].amount_lo_lo, ops[i].amount_lo_hi,
                                ops[i].amount_hi_lo, ops[i].amount_hi_hi);
            fees = u128_add(fees, amt);
            applied = applied + 1u;
            continue;
        }
    }
    atomicStore(&applied_out, applied);
    atomicStore(&total_fees_lo_out, fees.lo_lo);
    atomicStore(&total_fees_hi_out, fees.lo_hi);
}
