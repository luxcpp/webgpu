// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// platformvm_slashing.wgsl — WGSL peer of platformvm_slashing.metal.
//
// Concatenated by the host with platformvm_kernels_common.wgsl before submission.

@group(0) @binding(0) var<storage, read>       desc:        PVMRoundDescriptor;
@group(0) @binding(1) var<storage, read>       evidence:    array<SlashEvidence>;
@group(0) @binding(2) var<storage, read_write> validators:  array<ValidatorSlot>;
@group(0) @binding(3) var<storage, read_write> slashing:    array<SlashEvidence>;
@group(0) @binding(4) var<storage, read_write> applied_out: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> total_lo:    atomic<u32>;
@group(0) @binding(6) var<storage, read_write> total_hi:    atomic<u32>;
@group(0) @binding(7) var<uniform>             counts_u:    vec4<u32>; // [validator_count, slashing_count, 0, 0]

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

@compute @workgroup_size(1)
fn platformvm_slashing_transition(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    var applied: u32 = 0u;
    var total_lo_acc: u32 = 0u;
    var total_hi_acc: u32 = 0u;
    var cursor: u32 = 0u;
    let validator_count = counts_u.x;
    let slashing_count  = counts_u.y;
    let count = desc.slash_evidence_count;

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let v_idx = validator_locate(validator_count,
                                      evidence[i].validator_id_lo, evidence[i].validator_id_hi, false);
        if (v_idx == 0xFFFFFFFFu) { continue; }
        if ((validators[v_idx].status & kStatusTombstoned) != 0u) { continue; }

        // amount = ev.slash_amount or default-policy fraction of weight.
        var amt_lo = evidence[i].slash_amount_lo;
        var amt_hi = evidence[i].slash_amount_hi;
        if (u64_is_zero(amt_lo, amt_hi)) {
            // weight / divisor (5% / 1% / 2%)
            var divisor: u32 = 1u;
            if (evidence[i].kind == kEvEquivocation) { divisor = 20u; }
            else if (evidence[i].kind == kEvDowntime) { divisor = 100u; }
            else if (evidence[i].kind == kEvInvalidVote) { divisor = 50u; }
            let q = u64_div_u32(validators[v_idx].weight_lo, validators[v_idx].weight_hi, divisor);
            amt_lo = q.x;
            amt_hi = q.y;
        }
        // amt = min(amt, weight)
        if (u64_lt(validators[v_idx].weight_lo, validators[v_idx].weight_hi, amt_lo, amt_hi)) {
            amt_lo = validators[v_idx].weight_lo;
            amt_hi = validators[v_idx].weight_hi;
        }
        let new_w = u64_sat_sub(validators[v_idx].weight_lo, validators[v_idx].weight_hi, amt_lo, amt_hi);
        validators[v_idx].weight_lo = new_w.x;
        validators[v_idx].weight_hi = new_w.y;

        let new_total = u64_sat_add(total_lo_acc, total_hi_acc, amt_lo, amt_hi);
        total_lo_acc = new_total.x;
        total_hi_acc = new_total.y;

        if (evidence[i].kind == kEvEquivocation) {
            validators[v_idx].status = (validators[v_idx].status | kStatusTombstoned) & (~kStatusActive);
        } else {
            validators[v_idx].status = (validators[v_idx].status | kStatusJailed) & (~kStatusActive);
            var jail_for: u32 = evidence[i].jail_for_epochs;
            if (jail_for == 0u) { jail_for = 100u; }
            let until = evidence[i].epoch + jail_for;
            if (until > validators[v_idx].jail_until_epoch) {
                validators[v_idx].jail_until_epoch = until;
            }
        }

        if (cursor < slashing_count) {
            slashing[cursor].validator_id_lo = evidence[i].validator_id_lo;
            slashing[cursor].validator_id_hi = evidence[i].validator_id_hi;
            slashing[cursor].height_lo       = evidence[i].height_lo;
            slashing[cursor].height_hi       = evidence[i].height_hi;
            slashing[cursor].slash_amount_lo = evidence[i].slash_amount_lo;
            slashing[cursor].slash_amount_hi = evidence[i].slash_amount_hi;
            slashing[cursor].kind            = evidence[i].kind;
            slashing[cursor].epoch           = evidence[i].epoch;
            slashing[cursor].jail_for_epochs = evidence[i].jail_for_epochs;
            slashing[cursor]._pad0           = 0u;
            for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                slashing[cursor].evidence_digest[k] = evidence[i].evidence_digest[k];
            }
            slashing[cursor]._pad1_lo = 0u;
            slashing[cursor]._pad1_hi = 0u;
            cursor = cursor + 1u;
        }
        applied = applied + 1u;
    }
    atomicStore(&applied_out, applied);
    atomicStore(&total_lo, total_lo_acc);
    atomicStore(&total_hi, total_hi_acc);
}
