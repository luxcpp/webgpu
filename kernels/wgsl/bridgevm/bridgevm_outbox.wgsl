// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// bridgevm_outbox.wgsl — MessageOutbox kernel (WGSL).

@group(0) @binding(0) var<storage, read>       desc:     BridgeVMRoundDescriptor;
@group(0) @binding(1) var<storage, read>       out_reqs: array<OutboundReq>;
@group(0) @binding(2) var<storage, read_write> daily:    array<DailyLimit>;
@group(0) @binding(3) var<storage, read_write> outbox:   array<Message>;
@group(0) @binding(4) var<storage, read_write> epoch:    BridgeVMEpochState;
@group(0) @binding(5) var<storage, read_write> ob_applied_out:  atomic<u32>;
@group(0) @binding(6) var<storage, read_write> ob_total_lo_out: atomic<u32>;
@group(0) @binding(7) var<storage, read_write> ob_total_hi_out: atomic<u32>;
// counts: [_, daily_slots, _, outbox_slots]
@group(0) @binding(8) var<uniform>             counts_u: vec4<u32>;

fn compute_msg_subject_outbox(dst_chain: u32, nonce_lo: u32, nonce_hi: u32,
                              payload_root_p: ptr<function, array<u32, 8>>,
                              out: ptr<function, array<u32, 8>>)
{
    var buf: array<u32, 256>;
    for (var k: u32 = 0u; k < 256u; k = k + 1u) { buf[k] = 0u; }
    var o: u32 = 0u;
    put_u32_le(&buf, o, dst_chain);              o = o + 4u;
    put_u64_le(&buf, o, nonce_lo, nonce_hi);     o = o + 8u;
    put_digest32(&buf, o, payload_root_p);       o = o + 32u;
    keccak256_buf256(&buf, o, out);
}

fn daily_limit_locate_out(asset: u32, count: u32) -> u32 {
    let mask = count - 1u;
    var idx = hash_u64(asset, 0u, mask);
    for (var probe: u32 = 0u; probe < count; probe = probe + 1u) {
        if (daily[idx].status == 0u) { return 0xFFFFFFFFu; }
        if (daily[idx].asset_id == asset) { return idx; }
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

@compute @workgroup_size(1)
fn bridgevm_message_outbox(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let daily_slots  = counts_u.y;
    let outbox_slots = counts_u.w;

    let n_reqs = desc.outbound_req_count;
    var applied: u32 = 0u;
    var total = u128_make(0u, 0u, 0u, 0u);
    var cursor = epoch.outbox_count;

    for (var i: u32 = 0u; i < n_reqs; i = i + 1u) {
        if (out_reqs[i].kind == kMsgKindMint || out_reqs[i].kind == kMsgKindBurn) {
            let dl_idx = daily_limit_locate_out(out_reqs[i].asset_id, daily_slots);
            if (dl_idx != 0xFFFFFFFFu) {
                let cur = u128_make(daily[dl_idx].used_today_lo_lo,
                                    daily[dl_idx].used_today_lo_hi,
                                    daily[dl_idx].used_today_hi_lo,
                                    daily[dl_idx].used_today_hi_hi);
                let amt = u128_make(out_reqs[i].amount_lo_lo, out_reqs[i].amount_lo_hi,
                                    out_reqs[i].amount_hi_lo, out_reqs[i].amount_hi_hi);
                let nu = u128_add(cur, amt);
                let cap = u128_make(daily[dl_idx].daily_cap_lo_lo,
                                    daily[dl_idx].daily_cap_lo_hi,
                                    daily[dl_idx].daily_cap_hi_lo,
                                    daily[dl_idx].daily_cap_hi_hi);
                if (u128_lt(cap, nu)) { continue; }
                daily[dl_idx].used_today_lo_lo = nu.lo_lo;
                daily[dl_idx].used_today_lo_hi = nu.lo_hi;
                daily[dl_idx].used_today_hi_lo = nu.hi_lo;
                daily[dl_idx].used_today_hi_hi = nu.hi_hi;
            }
        }
        if (cursor >= outbox_slots) { break; }

        var pr: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { pr[k] = out_reqs[i].payload_root[k]; }
        var id: array<u32, 8>;
        compute_msg_subject_outbox(out_reqs[i].dst_chain,
                                   out_reqs[i].nonce_lo, out_reqs[i].nonce_hi,
                                   &pr, &id);
        for (var k: u32 = 0u; k < 8u;  k = k + 1u) { outbox[cursor].msg_id[k] = id[k]; }
        for (var k: u32 = 0u; k < 8u;  k = k + 1u) { outbox[cursor].payload_root[k] = pr[k]; }
        for (var k: u32 = 0u; k < 24u; k = k + 1u) { outbox[cursor].agg_signature[k] = 0u; }
        outbox[cursor].signers_bitmap_lo_lo = 0u;
        outbox[cursor].signers_bitmap_lo_hi = 0u;
        outbox[cursor].signers_bitmap_hi_lo = 0u;
        outbox[cursor].signers_bitmap_hi_hi = 0u;
        outbox[cursor].signer_count = 0u;
        outbox[cursor].nonce_lo = out_reqs[i].nonce_lo;
        outbox[cursor].nonce_hi = out_reqs[i].nonce_hi;
        outbox[cursor].src_chain = out_reqs[i].src_chain;
        outbox[cursor].dst_chain = out_reqs[i].dst_chain;
        outbox[cursor].kind = out_reqs[i].kind;
        outbox[cursor].status = kMsgStatusOutboxEmit;
        outbox[cursor].asset_id = out_reqs[i].asset_id;
        outbox[cursor].amount_lo_lo = out_reqs[i].amount_lo_lo;
        outbox[cursor].amount_lo_hi = out_reqs[i].amount_lo_hi;
        outbox[cursor].amount_hi_lo = out_reqs[i].amount_hi_lo;
        outbox[cursor].amount_hi_hi = out_reqs[i].amount_hi_hi;
        outbox[cursor].arrival_height_lo = out_reqs[i].height_lo;
        outbox[cursor].arrival_height_hi = out_reqs[i].height_hi;
        cursor = cursor + 1u;
        let amt = u128_make(out_reqs[i].amount_lo_lo, out_reqs[i].amount_lo_hi,
                            out_reqs[i].amount_hi_lo, out_reqs[i].amount_hi_hi);
        total = u128_add(total, amt);
        applied = applied + 1u;
    }
    epoch.outbox_count = cursor;
    atomicStore(&ob_applied_out, applied);
    atomicStore(&ob_total_lo_out, total.lo_lo);
    atomicStore(&ob_total_hi_out, total.lo_hi);
}
