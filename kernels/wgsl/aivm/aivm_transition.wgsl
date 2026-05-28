// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_transition.wgsl — WGSL peer of aivm_transition.metal.
//
// Computes the four roots and composes aivm_state_root. Determinism is
// guaranteed via byte-identical leaf encoding to the CPU/Metal/CUDA
// reference paths.

@group(0) @binding(0) var<storage, read>       desc:         AIVMRoundDescriptor;
@group(0) @binding(1) var<storage, read_write> attestations: array<Attestation>;
@group(0) @binding(2) var<storage, read_write> models:       array<ModelRegistryEntry>;
@group(0) @binding(3) var<storage, read_write> anchors:      array<AuditAnchor>;
@group(0) @binding(4) var<storage, read_write> epoch:        AIVMEpochState;
@group(0) @binding(5) var<storage, read_write> result:       AIVMTransitionResult;
@group(0) @binding(6) var<uniform>             counts_u:     vec4<u32>; // [att, model, anchor, 0]

fn fold_keccak(acc: ptr<function, array<u32, 8>>, leaf: ptr<function, array<u32, 8>>) {
    var buf: array<u32, 96>;
    for (var i: u32 = 0u; i < 96u; i = i + 1u) { buf[i] = 0u; }
    absorb_digest(&buf, 0u,  acc);
    absorb_digest(&buf, 32u, leaf);
    keccak256_byte(&buf, 64u, acc);
}

@compute @workgroup_size(1)
fn aivm_epoch_transition(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }

    let att_count    = counts_u.x;
    let model_count  = counts_u.y;
    let anchor_count = counts_u.z;

    // -- mark expired against round timestamp --
    for (var i: u32 = 0u; i < att_count; i = i + 1u) {
        if (attestations[i].occupied == 0u) { continue; }
        if (!u64_is_zero(attestations[i].expiry_ns_lo, attestations[i].expiry_ns_hi)
            && u64_le(attestations[i].expiry_ns_lo, attestations[i].expiry_ns_hi,
                      desc.timestamp_ns_lo, desc.timestamp_ns_hi)) {
            attestations[i].status = attestations[i].status | kAttStatusExpired;
        }
    }

    // -- attestation_root + counts --
    var acc: array<u32, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { acc[k] = 0u; }
    var n_active: u32 = 0u;
    var expired: u32 = 0u;
    for (var i: u32 = 0u; i < att_count; i = i + 1u) {
        if (attestations[i].occupied == 0u) { continue; }
        let exp_set = !u64_is_zero(attestations[i].expiry_ns_lo, attestations[i].expiry_ns_hi);
        let exp_le  = u64_le(attestations[i].expiry_ns_lo, attestations[i].expiry_ns_hi,
                             desc.timestamp_ns_lo, desc.timestamp_ns_hi);
        let exp = (exp_set && exp_le) || (attestations[i].status & kAttStatusExpired) != 0u;
        let ver = (attestations[i].status & kAttStatusVerified) != 0u;
        if (exp) { expired = expired + 1u; }
        else if (ver) { n_active = n_active + 1u; }

        var buf: array<u32, 96>;
        for (var j: u32 = 0u; j < 96u; j = j + 1u) { buf[j] = 0u; }
        var o: u32 = 0u;
        var d: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { d[k] = attestations[i].tee_quote_digest[k]; }
        absorb_digest(&buf, o, &d); o = o + 32u;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { d[k] = attestations[i].measurement[k]; }
        absorb_digest(&buf, o, &d); o = o + 32u;
        var key: array<u32, 12>;
        for (var k: u32 = 0u; k < 12u; k = k + 1u) { key[k] = attestations[i].attesting_key[k]; }
        absorb_key48(&buf, o, &key); o = o + 48u;
        absorb_u64(&buf, o, attestations[i].expiry_ns_lo, attestations[i].expiry_ns_hi); o = o + 8u;
        absorb_u32(&buf, o, attestations[i].kind);            o = o + 4u;
        absorb_u32(&buf, o, attestations[i].evidence_offset); o = o + 4u;
        absorb_u32(&buf, o, attestations[i].evidence_len);    o = o + 4u;
        absorb_u32(&buf, o, attestations[i].status);          o = o + 4u;
        var exp_flag: u32 = 0u;
        if (exp) { exp_flag = 1u; }
        absorb_u32(&buf, o, exp_flag);                        o = o + 4u;
        absorb_u32(&buf, o, i);                               o = o + 4u;

        var leaf_hash: array<u32, 8>;
        keccak256_byte(&buf, o, &leaf_hash);
        fold_keccak(&acc, &leaf_hash);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { epoch.attestation_root[k] = acc[k]; }

    // -- model_registry_root --
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { acc[k] = 0u; }
    var mcount: u32 = 0u;
    for (var i: u32 = 0u; i < model_count; i = i + 1u) {
        if (models[i].occupied == 0u) { continue; }
        mcount = mcount + 1u;

        var buf: array<u32, 96>;
        for (var j: u32 = 0u; j < 96u; j = j + 1u) { buf[j] = 0u; }
        var o: u32 = 0u;
        var d: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { d[k] = models[i].model_root[k]; }
        absorb_digest(&buf, o, &d); o = o + 32u;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { d[k] = models[i].weight_hash[k]; }
        absorb_digest(&buf, o, &d); o = o + 32u;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { d[k] = models[i].license_root[k]; }
        absorb_digest(&buf, o, &d); o = o + 32u;
        var addr: array<u32, 5>;
        for (var k: u32 = 0u; k < 5u; k = k + 1u) { addr[k] = models[i].owner_addr[k]; }
        absorb_addr20(&buf, o, &addr); o = o + 20u;
        absorb_u64(&buf, o, models[i].version_lo, models[i].version_hi); o = o + 8u;
        absorb_u64(&buf, o, models[i].parameter_count_lo, models[i].parameter_count_hi); o = o + 8u;
        absorb_u32(&buf, o, models[i].modality); o = o + 4u;
        absorb_u32(&buf, o, i);                  o = o + 4u;

        var leaf_hash: array<u32, 8>;
        keccak256_byte(&buf, o, &leaf_hash);
        fold_keccak(&acc, &leaf_hash);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { epoch.model_registry_root[k] = acc[k]; }

    // -- audit_root --
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { acc[k] = 0u; }
    var acount: u32 = 0u;
    for (var i: u32 = 0u; i < anchor_count; i = i + 1u) {
        if (anchors[i].occupied == 0u) { continue; }
        acount = acount + 1u;

        var buf: array<u32, 96>;
        for (var j: u32 = 0u; j < 96u; j = j + 1u) { buf[j] = 0u; }
        var o: u32 = 0u;
        var d: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { d[k] = anchors[i].commit_root[k]; }
        absorb_digest(&buf, o, &d); o = o + 32u;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { d[k] = anchors[i].parent_root[k]; }
        absorb_digest(&buf, o, &d); o = o + 32u;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { d[k] = anchors[i].validator_set_root_at_commit[k]; }
        absorb_digest(&buf, o, &d); o = o + 32u;
        absorb_u64(&buf, o, anchors[i].height_lo, anchors[i].height_hi); o = o + 8u;
        absorb_u64(&buf, o, anchors[i].timestamp_ns_lo, anchors[i].timestamp_ns_hi); o = o + 8u;
        absorb_u32(&buf, o, i); o = o + 4u;

        var leaf_hash: array<u32, 8>;
        keccak256_byte(&buf, o, &leaf_hash);
        fold_keccak(&acc, &leaf_hash);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { epoch.audit_root[k] = acc[k]; }

    // -- epoch metadata + closing --
    epoch.active_model_count        = mcount;
    epoch.expired_attestation_count = expired;
    epoch.total_active_attestations_lo = n_active;
    epoch.total_active_attestations_hi = 0u;
    if (desc.closing_flag != 0u) {
        // current_epoch = epoch + 1
        var nlo = desc.epoch_lo + 1u;
        var nhi = desc.epoch_hi;
        if (nlo < desc.epoch_lo) { nhi = nhi + 1u; }
        epoch.current_epoch_lo = nlo;
        epoch.current_epoch_hi = nhi;
    }

    // -- composed aivm_state_root --
    var buf: array<u32, 96>;
    for (var j: u32 = 0u; j < 96u; j = j + 1u) { buf[j] = 0u; }
    var o: u32 = 0u;
    var d: array<u32, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { d[k] = desc.parent_aivm_root[k]; }
    absorb_digest(&buf, o, &d); o = o + 32u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { d[k] = epoch.attestation_root[k]; }
    absorb_digest(&buf, o, &d); o = o + 32u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { d[k] = epoch.model_registry_root[k]; }
    absorb_digest(&buf, o, &d); o = o + 32u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { d[k] = epoch.audit_root[k]; }
    absorb_digest(&buf, o, &d); o = o + 32u;
    absorb_u64(&buf, o, epoch.current_epoch_lo, epoch.current_epoch_hi); o = o + 8u;
    absorb_u32(&buf, o, n_active); o = o + 4u;
    absorb_u32(&buf, o, mcount); o = o + 4u;
    absorb_u32(&buf, o, acount); o = o + 4u;

    var state_root_local: array<u32, 8>;
    keccak256_byte(&buf, o, &state_root_local);
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { epoch.aivm_state_root[k] = state_root_local[k]; }

    // -- write result --
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { result.attestation_root[k]    = epoch.attestation_root[k]; }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { result.model_registry_root[k] = epoch.model_registry_root[k]; }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { result.audit_root[k]          = epoch.audit_root[k]; }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { result.aivm_state_root[k]     = epoch.aivm_state_root[k]; }
    result.active_attestations  = n_active;
    result.expired_attestations = expired;
    result.model_count          = mcount;
    result.anchor_count         = acount;
    result.total_models_lo      = mcount;
    result.total_models_hi      = 0u;
    result.total_anchors_lo     = acount;
    result.total_anchors_hi     = 0u;
    result.epoch_lo             = epoch.current_epoch_lo;
    result.epoch_hi             = epoch.current_epoch_hi;
    result.status               = 1u;
}
