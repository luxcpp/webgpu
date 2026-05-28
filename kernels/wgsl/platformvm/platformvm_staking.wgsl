// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// platformvm_staking.wgsl — WGSL peer of platformvm_staking.metal.
//
// Concatenated by the host with platformvm_kernels_common.wgsl before submission.

@group(0) @binding(0) var<storage, read>       desc:        PVMRoundDescriptor;
@group(0) @binding(1) var<storage, read>       ops:         array<StakeOp>;
@group(0) @binding(2) var<storage, read_write> validators:  array<ValidatorSlot>;
@group(0) @binding(3) var<storage, read_write> stake:       array<StakeRecord>;
@group(0) @binding(4) var<storage, read_write> applied_out: atomic<u32>;
@group(0) @binding(5) var<uniform>             counts_u:    vec4<u32>; // [validator_count, stake_count, 0, 0]

// Per-kernel locate against module-scoped `validators`.
fn validator_locate(count: u32, id_lo: u32, id_hi: u32, insert_if_missing: bool) -> u32 {
    let mask = count - 1u;
    var idx = validator_index_hash(id_lo, id_hi, mask);
    for (var probe: u32 = 0u; probe < count; probe = probe + 1u) {
        if (validators[idx].occupied == 0u) {
            if (insert_if_missing) {
                validators[idx].validator_id_lo = id_lo;
                validators[idx].validator_id_hi = id_hi;
                validators[idx].weight_lo = 0u;
                validators[idx].weight_hi = 0u;
                validators[idx].status = 0u;
                validators[idx].jail_until_epoch = 0u;
                validators[idx].occupied = 1u;
                for (var k: u32 = 0u; k < 12u; k = k + 1u) { validators[idx].bls_pubkey[k] = 0u; }
                for (var k: u32 = 0u; k < 8u;  k = k + 1u) { validators[idx].ringtail_pubkey[k] = 0u; }
                for (var k: u32 = 0u; k < 8u;  k = k + 1u) { validators[idx].mldsa_pubkey[k] = 0u; }
                for (var k: u32 = 0u; k < 8u;  k = k + 1u) { validators[idx].mldsa_groth16_root[k] = 0u; }
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (validators[idx].validator_id_lo == id_lo && validators[idx].validator_id_hi == id_hi) {
            return idx;
        }
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

// Per-kernel locate against module-scoped `stake`.
fn stake_record_locate(count: u32, deleg_lo: u32, deleg_hi: u32,
                       val_lo: u32, val_hi: u32, insert_if_missing: bool) -> u32 {
    let mask = count - 1u;
    var idx = stake_record_index_hash(deleg_lo, deleg_hi, val_lo, val_hi, mask);
    for (var probe: u32 = 0u; probe < count; probe = probe + 1u) {
        if (stake[idx].status == 0u) {
            if (insert_if_missing) {
                stake[idx].delegator_id_lo = deleg_lo;
                stake[idx].delegator_id_hi = deleg_hi;
                stake[idx].validator_id_lo = val_lo;
                stake[idx].validator_id_hi = val_hi;
                stake[idx].amount_lo = 0u;
                stake[idx].amount_hi = 0u;
                stake[idx].lock_until_epoch_lo = 0u;
                stake[idx].lock_until_epoch_hi = 0u;
                stake[idx].reward_accumulator_lo = 0u;
                stake[idx].reward_accumulator_hi = 0u;
                stake[idx].commission_bps = 0u;
                stake[idx].status = kStakeStatusActive;
                stake[idx].epoch_bonded = 0u;
                stake[idx].epoch_unbonded = 0u;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (stake[idx].delegator_id_lo == deleg_lo && stake[idx].delegator_id_hi == deleg_hi
            && stake[idx].validator_id_lo == val_lo && stake[idx].validator_id_hi == val_hi) {
            return idx;
        }
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

@compute @workgroup_size(1)
fn platformvm_stake_transition(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    var applied: u32 = 0u;
    let validator_count = counts_u.x;
    let stake_count     = counts_u.y;
    let count = desc.stake_op_count;

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let kind = ops[i].kind;

        if (kind == kSOpBond) {
            let v_idx = validator_locate(validator_count,
                                          ops[i].validator_id_lo, ops[i].validator_id_hi, false);
            if (v_idx == 0xFFFFFFFFu) { continue; }
            if ((validators[v_idx].status & kStatusTombstoned) != 0u) { continue; }

            let s_idx = stake_record_locate(stake_count,
                                             ops[i].delegator_id_lo, ops[i].delegator_id_hi,
                                             ops[i].validator_id_lo, ops[i].validator_id_hi, true);
            if (s_idx == 0xFFFFFFFFu) { continue; }

            let new_amt = u64_sat_add(stake[s_idx].amount_lo, stake[s_idx].amount_hi,
                                       ops[i].amount_lo, ops[i].amount_hi);
            stake[s_idx].amount_lo = new_amt.x;
            stake[s_idx].amount_hi = new_amt.y;
            if (u64_lt(stake[s_idx].lock_until_epoch_lo, stake[s_idx].lock_until_epoch_hi,
                       ops[i].lock_until_epoch_lo, ops[i].lock_until_epoch_hi)) {
                stake[s_idx].lock_until_epoch_lo = ops[i].lock_until_epoch_lo;
                stake[s_idx].lock_until_epoch_hi = ops[i].lock_until_epoch_hi;
            }
            if (stake[s_idx].epoch_bonded == 0u) { stake[s_idx].epoch_bonded = ops[i].epoch; }
            stake[s_idx].status = kStakeStatusActive;

            let new_w = u64_sat_add(validators[v_idx].weight_lo, validators[v_idx].weight_hi,
                                     ops[i].amount_lo, ops[i].amount_hi);
            validators[v_idx].weight_lo = new_w.x;
            validators[v_idx].weight_hi = new_w.y;
            applied = applied + 1u;
            continue;
        }

        if (kind == kSOpUnbond) {
            let s_idx = stake_record_locate(stake_count,
                                             ops[i].delegator_id_lo, ops[i].delegator_id_hi,
                                             ops[i].validator_id_lo, ops[i].validator_id_hi, false);
            if (s_idx == 0xFFFFFFFFu) { continue; }
            if (stake[s_idx].status != kStakeStatusActive) { continue; }
            // op.epoch < lock_until_epoch -> still locked
            // u32 epoch vs u64 lock — CPU compares (uint32_t)epoch < (uint64_t)lock; we
            // mirror by comparing op.epoch (u32, hi=0) against lock as a u64.
            if (u64_lt(ops[i].epoch, 0u,
                       stake[s_idx].lock_until_epoch_lo, stake[s_idx].lock_until_epoch_hi)) {
                continue;
            }
            // amt = min(op.amount, stake.amount)
            var amt_lo = ops[i].amount_lo;
            var amt_hi = ops[i].amount_hi;
            if (u64_lt(stake[s_idx].amount_lo, stake[s_idx].amount_hi, amt_lo, amt_hi)) {
                amt_lo = stake[s_idx].amount_lo;
                amt_hi = stake[s_idx].amount_hi;
            }
            let after = u64_sat_sub(stake[s_idx].amount_lo, stake[s_idx].amount_hi, amt_lo, amt_hi);
            stake[s_idx].amount_lo = after.x;
            stake[s_idx].amount_hi = after.y;
            stake[s_idx].epoch_unbonded = ops[i].epoch;
            if (u64_is_zero(after.x, after.y)) {
                stake[s_idx].status = kStakeStatusRetired;
            } else {
                stake[s_idx].status = kStakeStatusUnbonding;
            }

            let v_idx = validator_locate(validator_count,
                                          ops[i].validator_id_lo, ops[i].validator_id_hi, false);
            if (v_idx != 0xFFFFFFFFu) {
                let new_w = u64_sat_sub(validators[v_idx].weight_lo, validators[v_idx].weight_hi,
                                         amt_lo, amt_hi);
                validators[v_idx].weight_lo = new_w.x;
                validators[v_idx].weight_hi = new_w.y;
            }
            applied = applied + 1u;
            continue;
        }

        if (kind == kSOpDelegate) {
            let v_idx = validator_locate(validator_count,
                                          ops[i].validator_id_lo, ops[i].validator_id_hi, false);
            if (v_idx == 0xFFFFFFFFu) { continue; }
            if ((validators[v_idx].status & kStatusTombstoned) != 0u) { continue; }
            if ((validators[v_idx].status & kStatusJailed) != 0u) { continue; }

            let s_idx = stake_record_locate(stake_count,
                                             ops[i].delegator_id_lo, ops[i].delegator_id_hi,
                                             ops[i].validator_id_lo, ops[i].validator_id_hi, true);
            if (s_idx == 0xFFFFFFFFu) { continue; }

            let new_amt = u64_sat_add(stake[s_idx].amount_lo, stake[s_idx].amount_hi,
                                       ops[i].amount_lo, ops[i].amount_hi);
            stake[s_idx].amount_lo = new_amt.x;
            stake[s_idx].amount_hi = new_amt.y;
            stake[s_idx].status = kStakeStatusActive;
            if (stake[s_idx].epoch_bonded == 0u) { stake[s_idx].epoch_bonded = ops[i].epoch; }

            let new_w = u64_sat_add(validators[v_idx].weight_lo, validators[v_idx].weight_hi,
                                     ops[i].amount_lo, ops[i].amount_hi);
            validators[v_idx].weight_lo = new_w.x;
            validators[v_idx].weight_hi = new_w.y;
            applied = applied + 1u;
            continue;
        }

        if (kind == kSOpRedelegate) {
            // src == dst -> noop
            if (ops[i].source_validator_id_lo == ops[i].validator_id_lo
                && ops[i].source_validator_id_hi == ops[i].validator_id_hi) { continue; }

            let src_idx = stake_record_locate(stake_count,
                                               ops[i].delegator_id_lo, ops[i].delegator_id_hi,
                                               ops[i].source_validator_id_lo, ops[i].source_validator_id_hi, false);
            if (src_idx == 0xFFFFFFFFu) { continue; }
            if (stake[src_idx].status != kStakeStatusActive) { continue; }
            if (u64_lt(ops[i].epoch, 0u,
                       stake[src_idx].lock_until_epoch_lo, stake[src_idx].lock_until_epoch_hi)) {
                continue;
            }

            let v_dst_idx = validator_locate(validator_count,
                                              ops[i].validator_id_lo, ops[i].validator_id_hi, false);
            if (v_dst_idx == 0xFFFFFFFFu) { continue; }
            if ((validators[v_dst_idx].status & kStatusTombstoned) != 0u) { continue; }

            // amt = min(op.amount, src.amount)
            var amt_lo = ops[i].amount_lo;
            var amt_hi = ops[i].amount_hi;
            if (u64_lt(stake[src_idx].amount_lo, stake[src_idx].amount_hi, amt_lo, amt_hi)) {
                amt_lo = stake[src_idx].amount_lo;
                amt_hi = stake[src_idx].amount_hi;
            }
            let after_src = u64_sat_sub(stake[src_idx].amount_lo, stake[src_idx].amount_hi, amt_lo, amt_hi);
            stake[src_idx].amount_lo = after_src.x;
            stake[src_idx].amount_hi = after_src.y;
            if (u64_is_zero(after_src.x, after_src.y)) {
                stake[src_idx].status = kStakeStatusRetired;
            }

            let v_src_idx = validator_locate(validator_count,
                                              ops[i].source_validator_id_lo, ops[i].source_validator_id_hi, false);
            if (v_src_idx != 0xFFFFFFFFu) {
                let new_w = u64_sat_sub(validators[v_src_idx].weight_lo, validators[v_src_idx].weight_hi,
                                         amt_lo, amt_hi);
                validators[v_src_idx].weight_lo = new_w.x;
                validators[v_src_idx].weight_hi = new_w.y;
            }

            let dst_idx = stake_record_locate(stake_count,
                                               ops[i].delegator_id_lo, ops[i].delegator_id_hi,
                                               ops[i].validator_id_lo, ops[i].validator_id_hi, true);
            if (dst_idx == 0xFFFFFFFFu) { continue; }
            let new_dst = u64_sat_add(stake[dst_idx].amount_lo, stake[dst_idx].amount_hi,
                                       amt_lo, amt_hi);
            stake[dst_idx].amount_lo = new_dst.x;
            stake[dst_idx].amount_hi = new_dst.y;
            stake[dst_idx].status = kStakeStatusActive;
            if (stake[dst_idx].epoch_bonded == 0u) { stake[dst_idx].epoch_bonded = ops[i].epoch; }

            let new_w_dst = u64_sat_add(validators[v_dst_idx].weight_lo, validators[v_dst_idx].weight_hi,
                                         amt_lo, amt_hi);
            validators[v_dst_idx].weight_lo = new_w_dst.x;
            validators[v_dst_idx].weight_hi = new_w_dst.y;
            applied = applied + 1u;
            continue;
        }

        if (kind == kSOpReward) {
            let v_idx = validator_locate(validator_count,
                                          ops[i].validator_id_lo, ops[i].validator_id_hi, false);
            if (v_idx == 0xFFFFFFFFu) { continue; }
            if (u64_is_zero(validators[v_idx].weight_lo, validators[v_idx].weight_hi)) { continue; }

            // scaled = sat_mul(op.amount, kRewardScale). CPU does:
            //   if (amt > UINT64_MAX / kRewardScale) saturate else amt * kRewardScale
            // We mirror the saturate test: amt > UINT64_MAX / kRewardScale.
            // UINT64_MAX / 1e18 == 18; so any amt > 18 saturates.
            var scaled_lo: u32;
            var scaled_hi: u32;
            // Compare amt vs 18 (lo=18, hi=0).
            if (u64_lt(18u, 0u, ops[i].amount_lo, ops[i].amount_hi)) {
                scaled_lo = 0xFFFFFFFFu;
                scaled_hi = 0xFFFFFFFFu;
            } else {
                let m = u64_mul(ops[i].amount_lo, ops[i].amount_hi, kRewardScale_lo, kRewardScale_hi);
                scaled_lo = m.x;
                scaled_hi = m.y;
            }
            // per_unit = scaled / weight (u64 / u64).
            let per = u64_div(scaled_lo, scaled_hi,
                              validators[v_idx].weight_lo, validators[v_idx].weight_hi);
            if (u64_is_zero(per.x, per.y)) { continue; }

            // Walk the entire stake arena, distribute pro-rata.
            for (var si: u32 = 0u; si < stake_count; si = si + 1u) {
                if (stake[si].status != kStakeStatusActive) { continue; }
                if (stake[si].validator_id_lo != ops[i].validator_id_lo
                    || stake[si].validator_id_hi != ops[i].validator_id_hi) { continue; }

                // delta = sat_mul(stake.amount, per_unit). CPU saturate test:
                //   if (s.amount > UINT64_MAX / per_unit) saturate else amount * per_unit
                // We mirror with u64_div + check; for simplicity compute u64 multiply and
                // detect overflow by checking high half.
                var delta_lo: u32;
                var delta_hi: u32;
                // Quick overflow guard: if either operand's high half nonzero AND
                // the other has high or top bits — saturate. The CPU uses
                // (s.amount > UINT64_MAX/per_unit) which only saturates when product
                // overflows. We compute the product naively; if it overflows we
                // detect by recomputing the high half explicitly.
                // For correctness we replicate: limit = UINT64_MAX / per_unit; if amount > limit -> 0xFF...FF.
                let limit = u64_div(0xFFFFFFFFu, 0xFFFFFFFFu, per.x, per.y);
                if (u64_lt(limit.x, limit.y, stake[si].amount_lo, stake[si].amount_hi)) {
                    delta_lo = 0xFFFFFFFFu;
                    delta_hi = 0xFFFFFFFFu;
                } else {
                    let p = u64_mul(stake[si].amount_lo, stake[si].amount_hi, per.x, per.y);
                    delta_lo = p.x;
                    delta_hi = p.y;
                }
                let new_acc = u64_sat_add(stake[si].reward_accumulator_lo,
                                           stake[si].reward_accumulator_hi,
                                           delta_lo, delta_hi);
                stake[si].reward_accumulator_lo = new_acc.x;
                stake[si].reward_accumulator_hi = new_acc.y;
            }
            applied = applied + 1u;
            continue;
        }

        if (kind == kSOpCommission) {
            let v_idx = validator_locate(validator_count,
                                          ops[i].validator_id_lo, ops[i].validator_id_hi, false);
            if (v_idx == 0xFFFFFFFFu) { continue; }
            if (ops[i].commission_bps > 10000u) { continue; }
            // CPU looks up stake[(validator,validator)] (commission self-record).
            let s_idx = stake_record_locate(stake_count,
                                             ops[i].validator_id_lo, ops[i].validator_id_hi,
                                             ops[i].validator_id_lo, ops[i].validator_id_hi, false);
            if (s_idx == 0xFFFFFFFFu) { continue; }
            stake[s_idx].commission_bps = ops[i].commission_bps;
            applied = applied + 1u;
            continue;
        }
    }
    atomicStore(&applied_out, applied);
}
