// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_anchor.wgsl — WGSL peer of aivm_anchor.metal.

@group(0) @binding(0) var<storage, read>       desc:        AIVMRoundDescriptor;
@group(0) @binding(1) var<storage, read>       ops:         array<AnchorOp>;
@group(0) @binding(2) var<storage, read_write> anchors:     array<AuditAnchor>;
@group(0) @binding(3) var<storage, read_write> applied_out: atomic<u32>;
@group(0) @binding(4) var<uniform>             anchor_count_u: vec4<u32>; // [count, 0, 0, 0]

@compute @workgroup_size(1)
fn aivm_anchor_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }

    let anchor_count = anchor_count_u.x;
    var cursor: u32 = 0u;
    while (cursor < anchor_count && anchors[cursor].occupied != 0u) {
        cursor = cursor + 1u;
    }

    var applied: u32 = 0u;
    let count = desc.anchor_op_count;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        var commit_root: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { commit_root[k] = ops[i].commit_root[k]; }
        if (digest_zero_8(&commit_root)) { continue; }
        if (cursor >= anchor_count) { break; }

        if (cursor > 0u) {
            let prev = cursor - 1u;
            var ok = true;
            for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                if (ops[i].parent_root[k] != anchors[prev].commit_root[k]) { ok = false; break; }
            }
            if (!ok) { continue; }
            if (u64_le(ops[i].height_lo, ops[i].height_hi,
                       anchors[prev].height_lo, anchors[prev].height_hi)) { continue; }
        }

        for (var k: u32 = 0u; k < 8u; k = k + 1u) { anchors[cursor].commit_root[k] = ops[i].commit_root[k]; }
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { anchors[cursor].parent_root[k] = ops[i].parent_root[k]; }
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { anchors[cursor].validator_set_root_at_commit[k]
                                                  = ops[i].validator_set_root_at_commit[k]; }
        anchors[cursor].height_lo       = ops[i].height_lo;
        anchors[cursor].height_hi       = ops[i].height_hi;
        anchors[cursor].timestamp_ns_lo = ops[i].timestamp_ns_lo;
        anchors[cursor].timestamp_ns_hi = ops[i].timestamp_ns_hi;
        anchors[cursor].occupied        = 1u;
        cursor = cursor + 1u;
        applied = applied + 1u;
    }
    atomicStore(&applied_out, applied);
}
