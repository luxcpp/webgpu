// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// platformvm_validator_set.wgsl — WGSL peer of platformvm_validator_set.metal.
//
// Concatenated by the host with platformvm_kernels_common.wgsl before submission.

@group(0) @binding(0) var<storage, read>       desc:        PVMRoundDescriptor;
@group(0) @binding(1) var<storage, read>       ops:         array<ValidatorOp>;
@group(0) @binding(2) var<storage, read_write> validators:  array<ValidatorSlot>;
@group(0) @binding(3) var<storage, read_write> applied_out: atomic<u32>;
@group(0) @binding(4) var<uniform>             counts_u:    vec4<u32>; // [validator_count, 0,0,0]

// Per-kernel locate against module-scoped `validators` (WGSL disallows
// pointers-to-storage as function parameters). Logic mirrors
// platformvm_cpu_reference.cpp::validator_locate.
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
fn platformvm_validator_set_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    var applied: u32 = 0u;
    let validator_count = counts_u.x;
    let count = desc.validator_op_count;

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let kind     = ops[i].kind;
        let id_lo    = ops[i].validator_id_lo;
        let id_hi    = ops[i].validator_id_hi;

        if (kind == kVOpAdd) {
            let idx = validator_locate(validator_count, id_lo, id_hi, true);
            if (idx == 0xFFFFFFFFu) { continue; }
            validators[idx].weight_lo = ops[i].weight_lo;
            validators[idx].weight_hi = ops[i].weight_hi;
            for (var k: u32 = 0u; k < 12u; k = k + 1u) { validators[idx].bls_pubkey[k]         = ops[i].bls_pubkey[k]; }
            for (var k: u32 = 0u; k < 8u;  k = k + 1u) { validators[idx].ringtail_pubkey[k]    = ops[i].ringtail_pubkey[k]; }
            for (var k: u32 = 0u; k < 8u;  k = k + 1u) { validators[idx].mldsa_pubkey[k]       = ops[i].mldsa_pubkey[k]; }
            for (var k: u32 = 0u; k < 8u;  k = k + 1u) { validators[idx].mldsa_groth16_root[k] = ops[i].mldsa_groth16_root[k]; }
            validators[idx].status = kStatusActive | kStatusPendingAdd;
            validators[idx].jail_until_epoch = 0u;
            applied = applied + 1u;
            continue;
        }
        if (kind == kVOpRemove) {
            let idx = validator_locate(validator_count, id_lo, id_hi, false);
            if (idx == 0xFFFFFFFFu) { continue; }
            if ((validators[idx].status & kStatusTombstoned) != 0u) { continue; }
            validators[idx].status = (validators[idx].status | kStatusPendingDrop) & (~kStatusActive);
            applied = applied + 1u;
            continue;
        }
        if (kind == kVOpUpdateWeight) {
            let idx = validator_locate(validator_count, id_lo, id_hi, false);
            if (idx == 0xFFFFFFFFu) { continue; }
            if ((validators[idx].status & kStatusTombstoned) != 0u) { continue; }
            validators[idx].weight_lo = ops[i].weight_lo;
            validators[idx].weight_hi = ops[i].weight_hi;
            applied = applied + 1u;
            continue;
        }
        if (kind == kVOpJail) {
            let idx = validator_locate(validator_count, id_lo, id_hi, false);
            if (idx == 0xFFFFFFFFu) { continue; }
            if ((validators[idx].status & kStatusTombstoned) != 0u) { continue; }
            validators[idx].status = (validators[idx].status | kStatusJailed) & (~kStatusActive);
            if (ops[i].jail_until_epoch > validators[idx].jail_until_epoch) {
                validators[idx].jail_until_epoch = ops[i].jail_until_epoch;
            }
            applied = applied + 1u;
            continue;
        }
        if (kind == kVOpUnjail) {
            let idx = validator_locate(validator_count, id_lo, id_hi, false);
            if (idx == 0xFFFFFFFFu) { continue; }
            if ((validators[idx].status & kStatusTombstoned) != 0u) { continue; }
            if (ops[i].epoch < validators[idx].jail_until_epoch) { continue; }
            validators[idx].status = (validators[idx].status & (~kStatusJailed)) | kStatusActive;
            validators[idx].jail_until_epoch = 0u;
            applied = applied + 1u;
            continue;
        }
        if (kind == kVOpRotateKeys) {
            let idx = validator_locate(validator_count, id_lo, id_hi, false);
            if (idx == 0xFFFFFFFFu) { continue; }
            if ((validators[idx].status & kStatusTombstoned) != 0u) { continue; }
            for (var k: u32 = 0u; k < 12u; k = k + 1u) { validators[idx].bls_pubkey[k]         = ops[i].bls_pubkey[k]; }
            for (var k: u32 = 0u; k < 8u;  k = k + 1u) { validators[idx].ringtail_pubkey[k]    = ops[i].ringtail_pubkey[k]; }
            for (var k: u32 = 0u; k < 8u;  k = k + 1u) { validators[idx].mldsa_pubkey[k]       = ops[i].mldsa_pubkey[k]; }
            for (var k: u32 = 0u; k < 8u;  k = k + 1u) { validators[idx].mldsa_groth16_root[k] = ops[i].mldsa_groth16_root[k]; }
            applied = applied + 1u;
            continue;
        }
    }
    atomicStore(&applied_out, applied);
}
