// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_attestation.wgsl — WGSL peer of aivm_attestation.metal.
//
// Concatenated by the host with aivm_kernels_common.wgsl before submission.

@group(0) @binding(0) var<storage, read>       desc:         AIVMRoundDescriptor;
@group(0) @binding(1) var<storage, read>       ops:          array<AttestationOp>;
@group(0) @binding(2) var<storage, read_write> attestations: array<Attestation>;
@group(0) @binding(3) var<storage, read_write> applied_out:  atomic<u32>;
@group(0) @binding(4) var<uniform>             att_count_u:  vec4<u32>;  // [count, 0, 0, 0]

// open-address insert/lookup against `attestations` by `digest` (function-local).
fn att_locate(digest: ptr<function, array<u32, 8>>, insert_if_missing: bool) -> u32 {
    let count = att_count_u.x;
    let mask  = count - 1u;
    let key   = key_from_digest(digest);
    var idx   = hash_index(key.x, key.y, mask);
    for (var probe: u32 = 0u; probe < count; probe = probe + 1u) {
        let occupied = attestations[idx].occupied;
        if (occupied == 0u) {
            if (insert_if_missing) {
                for (var k: u32 = 0u; k < 8u;  k = k + 1u) { attestations[idx].tee_quote_digest[k] = (*digest)[k]; }
                for (var k: u32 = 0u; k < 8u;  k = k + 1u) { attestations[idx].measurement[k] = 0u; }
                for (var k: u32 = 0u; k < 12u; k = k + 1u) { attestations[idx].attesting_key[k] = 0u; }
                attestations[idx].expiry_ns_lo = 0u;
                attestations[idx].expiry_ns_hi = 0u;
                attestations[idx].kind = 0u;
                attestations[idx].evidence_offset = 0u;
                attestations[idx].evidence_len = 0u;
                attestations[idx].status = 0u;
                attestations[idx].occupied = 1u;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        var equal = true;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            if (attestations[idx].tee_quote_digest[k] != (*digest)[k]) { equal = false; break; }
        }
        if (equal) { return idx; }
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

@compute @workgroup_size(1)
fn aivm_attestation_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    var applied: u32 = 0u;
    let count = desc.attestation_op_count;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        var digest: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { digest[k] = ops[i].tee_quote_digest[k]; }
        var key: array<u32, 12>;
        for (var k: u32 = 0u; k < 12u; k = k + 1u) { key[k] = ops[i].attesting_key[k]; }

        if (key48_zero_12(&key)) { continue; }
        if (digest_zero_8(&digest)) { continue; }

        let idx = att_locate(&digest, true);
        if (idx == 0xFFFFFFFFu) { continue; }

        for (var k: u32 = 0u; k < 8u;  k = k + 1u) { attestations[idx].measurement[k]   = ops[i].measurement[k]; }
        for (var k: u32 = 0u; k < 12u; k = k + 1u) { attestations[idx].attesting_key[k] = ops[i].attesting_key[k]; }
        attestations[idx].expiry_ns_lo    = ops[i].expiry_ns_lo;
        attestations[idx].expiry_ns_hi    = ops[i].expiry_ns_hi;
        attestations[idx].kind            = ops[i].kind;
        attestations[idx].evidence_offset = ops[i].evidence_offset;
        attestations[idx].evidence_len    = ops[i].evidence_len;
        var status = kAttStatusVerified;
        if (!u64_is_zero(ops[i].expiry_ns_lo, ops[i].expiry_ns_hi)
            && u64_le(ops[i].expiry_ns_lo, ops[i].expiry_ns_hi,
                      desc.timestamp_ns_lo, desc.timestamp_ns_hi)) {
            status = status | kAttStatusExpired;
        }
        attestations[idx].status = status;
        applied = applied + 1u;
    }
    atomicStore(&applied_out, applied);
}
