// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_provenance.wgsl — WGSL peer of aivm_provenance.metal.

@group(0) @binding(0) var<storage, read>       desc:        AIVMRoundDescriptor;
@group(0) @binding(1) var<storage, read>       ops:         array<ModelOp>;
@group(0) @binding(2) var<storage, read_write> models:      array<ModelRegistryEntry>;
@group(0) @binding(3) var<storage, read_write> applied_out: atomic<u32>;
@group(0) @binding(4) var<uniform>             model_count_u: vec4<u32>; // [count, 0, 0, 0]

fn model_locate(model_root: ptr<function, array<u32, 8>>, insert_if_missing: bool) -> u32 {
    let count = model_count_u.x;
    let mask  = count - 1u;
    let key   = key_from_digest(model_root);
    var idx   = hash_index(key.x, key.y, mask);
    for (var probe: u32 = 0u; probe < count; probe = probe + 1u) {
        let occupied = models[idx].occupied;
        if (occupied == 0u) {
            if (insert_if_missing) {
                for (var k: u32 = 0u; k < 8u; k = k + 1u) { models[idx].model_root[k]   = (*model_root)[k]; }
                for (var k: u32 = 0u; k < 8u; k = k + 1u) { models[idx].weight_hash[k]  = 0u; }
                for (var k: u32 = 0u; k < 8u; k = k + 1u) { models[idx].license_root[k] = 0u; }
                for (var k: u32 = 0u; k < 5u; k = k + 1u) { models[idx].owner_addr[k]   = 0u; }
                models[idx].version_lo = 0u;
                models[idx].version_hi = 0u;
                models[idx].parameter_count_lo = 0u;
                models[idx].parameter_count_hi = 0u;
                models[idx].modality = 0u;
                models[idx].occupied = 1u;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        var equal = true;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            if (models[idx].model_root[k] != (*model_root)[k]) { equal = false; break; }
        }
        if (equal) { return idx; }
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

@compute @workgroup_size(1)
fn aivm_provenance_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    var applied: u32 = 0u;
    let count = desc.model_op_count;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        var mroot: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { mroot[k] = ops[i].model_root[k]; }
        var whash: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { whash[k] = ops[i].weight_hash[k]; }
        if (digest_zero_8(&mroot)) { continue; }
        if (digest_zero_8(&whash)) { continue; }

        let kind = ops[i].kind;
        if (kind == kModelOpRegister) {
            let idx = model_locate(&mroot, true);
            if (idx == 0xFFFFFFFFu) { continue; }
            for (var k: u32 = 0u; k < 8u; k = k + 1u) { models[idx].weight_hash[k]  = ops[i].weight_hash[k]; }
            for (var k: u32 = 0u; k < 8u; k = k + 1u) { models[idx].license_root[k] = ops[i].license_root[k]; }
            for (var k: u32 = 0u; k < 5u; k = k + 1u) { models[idx].owner_addr[k]   = ops[i].owner_addr[k]; }
            models[idx].parameter_count_lo = ops[i].parameter_count_lo;
            models[idx].parameter_count_hi = ops[i].parameter_count_hi;
            models[idx].modality           = ops[i].modality;
            models[idx].version_lo         = 1u;
            models[idx].version_hi         = 0u;
            applied = applied + 1u;
        } else if (kind == kModelOpUpdateWeights) {
            let idx = model_locate(&mroot, false);
            if (idx == 0xFFFFFFFFu) { continue; }
            for (var k: u32 = 0u; k < 8u; k = k + 1u) { models[idx].weight_hash[k] = ops[i].weight_hash[k]; }
            // saturating ++version
            let lo = models[idx].version_lo;
            let hi = models[idx].version_hi;
            var nlo = lo + 1u;
            var nhi = hi;
            if (nlo < lo) { nhi = nhi + 1u; }
            if (lo == 0xFFFFFFFFu && hi == 0xFFFFFFFFu) { nlo = 0xFFFFFFFFu; nhi = 0xFFFFFFFFu; }
            models[idx].version_lo = nlo;
            models[idx].version_hi = nhi;
            if (!(ops[i].parameter_count_lo == 0u && ops[i].parameter_count_hi == 0u)) {
                models[idx].parameter_count_lo = ops[i].parameter_count_lo;
                models[idx].parameter_count_hi = ops[i].parameter_count_hi;
            }
            applied = applied + 1u;
        } else if (kind == kModelOpUpdateLicense) {
            let idx = model_locate(&mroot, false);
            if (idx == 0xFFFFFFFFu) { continue; }
            for (var k: u32 = 0u; k < 8u; k = k + 1u) { models[idx].license_root[k] = ops[i].license_root[k]; }
            applied = applied + 1u;
        } else if (kind == kModelOpTransfer) {
            let idx = model_locate(&mroot, false);
            if (idx == 0xFFFFFFFFu) { continue; }
            for (var k: u32 = 0u; k < 5u; k = k + 1u) { models[idx].owner_addr[k] = ops[i].owner_addr[k]; }
            applied = applied + 1u;
        }
    }
    atomicStore(&applied_out, applied);
}
