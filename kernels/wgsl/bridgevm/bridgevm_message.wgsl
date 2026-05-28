// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// bridgevm_message.wgsl — MessageInbox kernel (WGSL).
//
// The outbox kernel lives in a sibling module so each entry point has a
// minimal bind layout — Dawn's auto-derived layout includes only the
// bindings the entry point references; mismatching the host-side bind group
// triggers a validation error.

@group(0) @binding(0) var<storage, read>       desc:    BridgeVMRoundDescriptor;
@group(0) @binding(1) var<storage, read>       in_msgs: array<Message>;
@group(0) @binding(2) var<storage, read_write> signers: array<Signer>;
@group(0) @binding(3) var<storage, read_write> daily:   array<DailyLimit>;
@group(0) @binding(4) var<storage, read_write> inbox:   array<Message>;
@group(0) @binding(5) var<storage, read_write> ib_applied_out:  atomic<u32>;
@group(0) @binding(6) var<storage, read_write> ib_total_lo_out: atomic<u32>;
@group(0) @binding(7) var<storage, read_write> ib_total_hi_out: atomic<u32>;
// counts: [signer_slots, daily_slots, inbox_slots, outbox_slots]
@group(0) @binding(8) var<uniform>             counts_u: vec4<u32>;

fn count_active_signers_msg(count: u32) -> u32 {
    var n: u32 = 0u;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        if (signers[i].occupied == 0u) { continue; }
        let st = signers[i].status;
        if ((st & kSignerStatusActive) != 0u
            && (st & kSignerStatusJailed) == 0u
            && (st & kSignerStatusTombstoned) == 0u) {
            n = n + 1u;
        }
    }
    return n;
}

fn compute_msg_subject_inbox(dst_chain: u32, nonce_lo: u32, nonce_hi: u32,
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

fn msg_id_eq_in(a: ptr<function, array<u32, 8>>, b: ptr<function, array<u32, 8>>) -> bool {
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        if ((*a)[k] != (*b)[k]) { return false; }
    }
    return true;
}

fn msg_id_zero_in(a: ptr<function, array<u32, 8>>) -> bool {
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        if ((*a)[k] != 0u) { return false; }
    }
    return true;
}

fn daily_limit_locate_in(asset: u32, count: u32) -> u32 {
    let mask = count - 1u;
    var idx = hash_u64(asset, 0u, mask);
    for (var probe: u32 = 0u; probe < count; probe = probe + 1u) {
        if (daily[idx].status == 0u) { return 0xFFFFFFFFu; }
        if (daily[idx].asset_id == asset) { return idx; }
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

fn inbox_locate(id: ptr<function, array<u32, 8>>, count: u32, insert_if_missing: bool) -> u32 {
    let mask = count - 1u;
    var idx = hash_msg_id(id, mask);
    for (var probe: u32 = 0u; probe < count; probe = probe + 1u) {
        var slot_id: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { slot_id[k] = inbox[idx].msg_id[k]; }
        if (inbox[idx].status == 0u && msg_id_zero_in(&slot_id)) {
            if (insert_if_missing) { return idx; }
            return 0xFFFFFFFFu;
        }
        if (msg_id_eq_in(&slot_id, id)) { return idx; }
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

@compute @workgroup_size(1)
fn bridgevm_message_inbox(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let signer_slots = counts_u.x;
    let daily_slots  = counts_u.y;
    let inbox_slots  = counts_u.z;

    let active_n = count_active_signers_msg(signer_slots);
    let n_msgs = desc.inbound_msg_count;
    var applied: u32 = 0u;
    var total = u128_make(0u, 0u, 0u, 0u);

    for (var i: u32 = 0u; i < n_msgs; i = i + 1u) {
        var pr: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { pr[k] = in_msgs[i].payload_root[k]; }
        var subject: array<u32, 8>;
        compute_msg_subject_inbox(in_msgs[i].dst_chain,
                                  in_msgs[i].nonce_lo, in_msgs[i].nonce_hi,
                                  &pr, &subject);
        var have_id: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { have_id[k] = in_msgs[i].msg_id[k]; }
        if (!msg_id_eq_in(&subject, &have_id)) { continue; }

        let dup = inbox_locate(&have_id, inbox_slots, false);
        if (dup != 0xFFFFFFFFu) { continue; }

        if (in_msgs[i].signer_count == 0u) { continue; }
        let pc = popcount_u128_pair(in_msgs[i].signers_bitmap_lo_lo,
                                    in_msgs[i].signers_bitmap_lo_hi,
                                    in_msgs[i].signers_bitmap_hi_lo,
                                    in_msgs[i].signers_bitmap_hi_hi);
        if (pc != in_msgs[i].signer_count) { continue; }
        if (!meets_bft_threshold(in_msgs[i].signer_count, active_n)) { continue; }

        let idx = inbox_locate(&have_id, inbox_slots, true);
        if (idx == 0xFFFFFFFFu) { continue; }

        for (var k: u32 = 0u; k < 8u;  k = k + 1u) { inbox[idx].msg_id[k] = in_msgs[i].msg_id[k]; }
        for (var k: u32 = 0u; k < 8u;  k = k + 1u) { inbox[idx].payload_root[k] = in_msgs[i].payload_root[k]; }
        for (var k: u32 = 0u; k < 24u; k = k + 1u) { inbox[idx].agg_signature[k] = in_msgs[i].agg_signature[k]; }
        inbox[idx].signers_bitmap_lo_lo = in_msgs[i].signers_bitmap_lo_lo;
        inbox[idx].signers_bitmap_lo_hi = in_msgs[i].signers_bitmap_lo_hi;
        inbox[idx].signers_bitmap_hi_lo = in_msgs[i].signers_bitmap_hi_lo;
        inbox[idx].signers_bitmap_hi_hi = in_msgs[i].signers_bitmap_hi_hi;
        inbox[idx].nonce_lo = in_msgs[i].nonce_lo;
        inbox[idx].nonce_hi = in_msgs[i].nonce_hi;
        inbox[idx].src_chain = in_msgs[i].src_chain;
        inbox[idx].dst_chain = in_msgs[i].dst_chain;
        inbox[idx].kind = in_msgs[i].kind;
        inbox[idx].status = kMsgStatusAccepted;
        inbox[idx].asset_id = in_msgs[i].asset_id;
        inbox[idx].signer_count = in_msgs[i].signer_count;
        inbox[idx].amount_lo_lo = in_msgs[i].amount_lo_lo;
        inbox[idx].amount_lo_hi = in_msgs[i].amount_lo_hi;
        inbox[idx].amount_hi_lo = in_msgs[i].amount_hi_lo;
        inbox[idx].amount_hi_hi = in_msgs[i].amount_hi_hi;
        inbox[idx].arrival_height_lo = desc.height_lo;
        inbox[idx].arrival_height_hi = desc.height_hi;

        if (in_msgs[i].kind == kMsgKindMint) {
            let dl_idx = daily_limit_locate_in(in_msgs[i].asset_id, daily_slots);
            if (dl_idx != 0xFFFFFFFFu) {
                let cur_used = u128_make(daily[dl_idx].used_today_lo_lo,
                                         daily[dl_idx].used_today_lo_hi,
                                         daily[dl_idx].used_today_hi_lo,
                                         daily[dl_idx].used_today_hi_hi);
                let amt = u128_make(in_msgs[i].amount_lo_lo, in_msgs[i].amount_lo_hi,
                                    in_msgs[i].amount_hi_lo, in_msgs[i].amount_hi_hi);
                let nu = u128_add(cur_used, amt);
                let cap = u128_make(daily[dl_idx].daily_cap_lo_lo,
                                    daily[dl_idx].daily_cap_lo_hi,
                                    daily[dl_idx].daily_cap_hi_lo,
                                    daily[dl_idx].daily_cap_hi_hi);
                if (u128_lt(cap, nu)) {
                    inbox[idx].status = kMsgStatusVerified;
                    continue;
                }
                daily[dl_idx].used_today_lo_lo = nu.lo_lo;
                daily[dl_idx].used_today_lo_hi = nu.lo_hi;
                daily[dl_idx].used_today_hi_lo = nu.hi_lo;
                daily[dl_idx].used_today_hi_hi = nu.hi_hi;
            }
            let amt = u128_make(in_msgs[i].amount_lo_lo, in_msgs[i].amount_lo_hi,
                                in_msgs[i].amount_hi_lo, in_msgs[i].amount_hi_hi);
            total = u128_add(total, amt);
        }
        applied = applied + 1u;
    }
    atomicStore(&ib_applied_out, applied);
    atomicStore(&ib_total_lo_out, total.lo_lo);
    atomicStore(&ib_total_hi_out, total.lo_hi);
}
