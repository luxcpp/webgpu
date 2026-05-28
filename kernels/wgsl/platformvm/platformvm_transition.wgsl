// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// platformvm_transition.wgsl — WGSL peer of platformvm_transition.metal.
//
// Computes the four roots and the composed epoch_root. Determinism is
// guaranteed via byte-identical leaf encoding to the CPU/Metal/CUDA
// reference paths.

@group(0) @binding(0) var<storage, read>       desc:        PVMRoundDescriptor;
@group(0) @binding(1) var<storage, read_write> validators:  array<ValidatorSlot>;
@group(0) @binding(2) var<storage, read_write> stake:       array<StakeRecord>;
@group(0) @binding(3) var<storage, read_write> slashing:    array<SlashEvidence>;
@group(0) @binding(4) var<storage, read_write> epoch:       EpochState;
@group(0) @binding(5) var<storage, read_write> result:      PVMTransitionResult;
@group(0) @binding(6) var<uniform>             counts_u:    vec4<u32>; // [validator_count, stake_count, slashing_count, 0]

fn fold_keccak(acc: ptr<function, array<u32, 8>>, leaf: ptr<function, array<u32, 8>>) {
    var buf: array<u32, 48>;
    for (var i: u32 = 0u; i < 48u; i = i + 1u) { buf[i] = 0u; }
    absorb_digest32(&buf, 0u,  acc);
    absorb_digest32(&buf, 32u, leaf);
    keccak256_byte(&buf, 64u, acc);
}

@compute @workgroup_size(1)
fn platformvm_epoch_transition(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }

    let validator_count = counts_u.x;
    let stake_count     = counts_u.y;
    let slashing_count  = counts_u.z;

    // -- promotion / auto-unjail --
    // target_epoch = closing ? desc.epoch + 1 : desc.epoch  (u64 add with carry)
    var target_lo: u32 = desc.epoch_lo;
    var target_hi: u32 = desc.epoch_hi;
    if (desc.closing_flag != 0u) {
        let t = u64_add(target_lo, target_hi, 1u, 0u);
        target_lo = t.x;
        target_hi = t.y;
    }
    var pending_drop: u32 = 0u;
    for (var i: u32 = 0u; i < validator_count; i = i + 1u) {
        if (validators[i].occupied == 0u) { continue; }
        if ((validators[i].status & kStatusPendingAdd) != 0u) {
            validators[i].status = validators[i].status & (~kStatusPendingAdd);
        }
        if ((validators[i].status & kStatusPendingDrop) != 0u) {
            validators[i].status = (validators[i].status & (~kStatusPendingDrop)) | kStatusTombstoned;
            pending_drop = pending_drop + 1u;
        }
        // jail expiration: cast target_epoch to u32 (mirrors CPU `(uint32_t)target_epoch`).
        if ((validators[i].status & kStatusJailed) != 0u
            && validators[i].jail_until_epoch != 0u
            && target_lo >= validators[i].jail_until_epoch
            && (validators[i].status & kStatusTombstoned) == 0u) {
            validators[i].status = (validators[i].status & (~kStatusJailed)) | kStatusActive;
            validators[i].jail_until_epoch = 0u;
        }
    }
    epoch.pending_drop_count = pending_drop;

    // -- validator_set_root + counts + total_active_stake --
    var acc: array<u32, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { acc[k] = 0u; }
    var active_count: u32 = 0u;
    var jailed: u32 = 0u;
    var tombstoned: u32 = 0u;
    var total_stake_lo: u32 = 0u;
    var total_stake_hi: u32 = 0u;
    for (var i: u32 = 0u; i < validator_count; i = i + 1u) {
        if (validators[i].occupied == 0u) { continue; }
        if ((validators[i].status & kStatusTombstoned) != 0u) { tombstoned = tombstoned + 1u; }
        if ((validators[i].status & kStatusJailed) != 0u)     { jailed     = jailed     + 1u; }
        if ((validators[i].status & kStatusActive) != 0u) {
            active_count = active_count + 1u;
            let s = u64_sat_add(total_stake_lo, total_stake_hi,
                                 validators[i].weight_lo, validators[i].weight_hi);
            total_stake_lo = s.x;
            total_stake_hi = s.y;
        }
        // leaf = id||weight||status||jail_until||bls48||rt32||mldsa32||groth32||index
        // size = 8+8+4+4+48+32+32+32+4 = 172 bytes
        var buf: array<u32, 48>;
        for (var j: u32 = 0u; j < 48u; j = j + 1u) { buf[j] = 0u; }
        var o: u32 = 0u;
        absorb_u64(&buf, o, validators[i].validator_id_lo, validators[i].validator_id_hi); o = o + 8u;
        absorb_u64(&buf, o, validators[i].weight_lo, validators[i].weight_hi);             o = o + 8u;
        absorb_u32(&buf, o, validators[i].status);                                          o = o + 4u;
        absorb_u32(&buf, o, validators[i].jail_until_epoch);                                o = o + 4u;
        var bls: array<u32, 12>;
        for (var k: u32 = 0u; k < 12u; k = k + 1u) { bls[k] = validators[i].bls_pubkey[k]; }
        absorb_key48(&buf, o, &bls);                                                        o = o + 48u;
        var rt: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { rt[k] = validators[i].ringtail_pubkey[k]; }
        absorb_digest32(&buf, o, &rt);                                                      o = o + 32u;
        var ml: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { ml[k] = validators[i].mldsa_pubkey[k]; }
        absorb_digest32(&buf, o, &ml);                                                      o = o + 32u;
        var gr: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { gr[k] = validators[i].mldsa_groth16_root[k]; }
        absorb_digest32(&buf, o, &gr);                                                      o = o + 32u;
        absorb_u32(&buf, o, i);                                                             o = o + 4u;

        var leaf_hash: array<u32, 8>;
        keccak256_byte(&buf, o, &leaf_hash);
        fold_keccak(&acc, &leaf_hash);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { epoch.validator_set_root[k] = acc[k]; }

    // -- stake_root + total_rewards --
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { acc[k] = 0u; }
    var total_rewards_lo: u32 = 0u;
    var total_rewards_hi: u32 = 0u;
    for (var i: u32 = 0u; i < stake_count; i = i + 1u) {
        if (stake[i].status == 0u) { continue; }
        let s = u64_sat_add(total_rewards_lo, total_rewards_hi,
                             stake[i].reward_accumulator_lo, stake[i].reward_accumulator_hi);
        total_rewards_lo = s.x;
        total_rewards_hi = s.y;

        // leaf = deleg||val||amount||lock||reward||commission||status||eb||eu||index
        // size = 8+8+8+8+8+4+4+4+4+4 = 60 bytes
        var buf: array<u32, 48>;
        for (var j: u32 = 0u; j < 48u; j = j + 1u) { buf[j] = 0u; }
        var o: u32 = 0u;
        absorb_u64(&buf, o, stake[i].delegator_id_lo,        stake[i].delegator_id_hi);        o = o + 8u;
        absorb_u64(&buf, o, stake[i].validator_id_lo,        stake[i].validator_id_hi);        o = o + 8u;
        absorb_u64(&buf, o, stake[i].amount_lo,              stake[i].amount_hi);              o = o + 8u;
        absorb_u64(&buf, o, stake[i].lock_until_epoch_lo,    stake[i].lock_until_epoch_hi);    o = o + 8u;
        absorb_u64(&buf, o, stake[i].reward_accumulator_lo,  stake[i].reward_accumulator_hi);  o = o + 8u;
        absorb_u32(&buf, o, stake[i].commission_bps);    o = o + 4u;
        absorb_u32(&buf, o, stake[i].status);            o = o + 4u;
        absorb_u32(&buf, o, stake[i].epoch_bonded);      o = o + 4u;
        absorb_u32(&buf, o, stake[i].epoch_unbonded);    o = o + 4u;
        absorb_u32(&buf, o, i);                          o = o + 4u;

        var leaf_hash: array<u32, 8>;
        keccak256_byte(&buf, o, &leaf_hash);
        fold_keccak(&acc, &leaf_hash);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { epoch.stake_root[k] = acc[k]; }

    // -- slashing_root --
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { acc[k] = 0u; }
    for (var i: u32 = 0u; i < slashing_count; i = i + 1u) {
        // empty-slot detection: validator_id == 0 AND height == 0 AND digest is zero
        var zero_digest = true;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            if (slashing[i].evidence_digest[k] != 0u) { zero_digest = false; break; }
        }
        if (slashing[i].validator_id_lo == 0u && slashing[i].validator_id_hi == 0u
            && zero_digest
            && slashing[i].height_lo == 0u && slashing[i].height_hi == 0u) { continue; }

        // leaf = val||height||amount||kind||epoch||jail_for||digest32||index
        // size = 8+8+8+4+4+4+32+4 = 72 bytes
        var buf: array<u32, 48>;
        for (var j: u32 = 0u; j < 48u; j = j + 1u) { buf[j] = 0u; }
        var o: u32 = 0u;
        absorb_u64(&buf, o, slashing[i].validator_id_lo, slashing[i].validator_id_hi); o = o + 8u;
        absorb_u64(&buf, o, slashing[i].height_lo,       slashing[i].height_hi);       o = o + 8u;
        absorb_u64(&buf, o, slashing[i].slash_amount_lo, slashing[i].slash_amount_hi); o = o + 8u;
        absorb_u32(&buf, o, slashing[i].kind);            o = o + 4u;
        absorb_u32(&buf, o, slashing[i].epoch);           o = o + 4u;
        absorb_u32(&buf, o, slashing[i].jail_for_epochs); o = o + 4u;
        var d: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { d[k] = slashing[i].evidence_digest[k]; }
        absorb_digest32(&buf, o, &d);                     o = o + 32u;
        absorb_u32(&buf, o, i);                           o = o + 4u;

        var leaf_hash: array<u32, 8>;
        keccak256_byte(&buf, o, &leaf_hash);
        fold_keccak(&acc, &leaf_hash);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { epoch.slashing_root[k] = acc[k]; }

    // -- epoch metadata --
    epoch.active_validator_count = active_count;
    epoch.total_active_stake_lo = total_stake_lo;
    epoch.total_active_stake_hi = total_stake_hi;
    if (desc.closing_flag != 0u) {
        epoch.current_epoch_lo = target_lo;
        epoch.current_epoch_hi = target_hi;
    }

    // -- composed epoch_root = keccak(parent || vroot || sroot || slroot
    //                                   || epoch || total_stake || active_count)
    // size = 32 + 32 + 32 + 32 + 8 + 8 + 4 = 148 bytes
    var buf: array<u32, 48>;
    for (var j: u32 = 0u; j < 48u; j = j + 1u) { buf[j] = 0u; }
    var o: u32 = 0u;
    var d: array<u32, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { d[k] = desc.parent_epoch_root[k]; }
    absorb_digest32(&buf, o, &d);                                     o = o + 32u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { d[k] = epoch.validator_set_root[k]; }
    absorb_digest32(&buf, o, &d);                                     o = o + 32u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { d[k] = epoch.stake_root[k]; }
    absorb_digest32(&buf, o, &d);                                     o = o + 32u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { d[k] = epoch.slashing_root[k]; }
    absorb_digest32(&buf, o, &d);                                     o = o + 32u;
    absorb_u64(&buf, o, epoch.current_epoch_lo, epoch.current_epoch_hi);     o = o + 8u;
    absorb_u64(&buf, o, epoch.total_active_stake_lo, epoch.total_active_stake_hi); o = o + 8u;
    absorb_u32(&buf, o, epoch.active_validator_count);                 o = o + 4u;

    var epoch_root_local: array<u32, 8>;
    keccak256_byte(&buf, o, &epoch_root_local);
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { epoch.epoch_root[k] = epoch_root_local[k]; }

    // -- write result --
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { result.validator_set_root[k] = epoch.validator_set_root[k]; }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { result.stake_root[k]         = epoch.stake_root[k]; }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { result.slashing_root[k]      = epoch.slashing_root[k]; }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { result.epoch_root[k]         = epoch.epoch_root[k]; }
    result.active_validator_count = active_count;
    result.jailed_count           = jailed;
    result.tombstoned_count       = tombstoned;
    result.total_active_stake_lo  = total_stake_lo;
    result.total_active_stake_hi  = total_stake_hi;
    result.total_rewards_lo       = total_rewards_lo;
    result.total_rewards_hi       = total_rewards_hi;
    result.pending_drop_count     = pending_drop;
    result.epoch_lo               = epoch.current_epoch_lo;
    result.epoch_hi               = epoch.current_epoch_hi;
    result.status                 = 1u;
}
