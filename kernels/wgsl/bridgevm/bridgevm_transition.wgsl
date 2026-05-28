// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// bridgevm_transition.wgsl — BridgeVMTransition kernel (WGSL).
//
// Promote pending_add -> active, drop pending_drop, exit-cooldown ->
// tombstone, auto-unjail expired jails, reset daily limits, then compose
// the six roots (signer/liquidity/inbox/outbox/daily + bridgevm_state).

@group(0) @binding(0) var<storage, read>       desc:      BridgeVMRoundDescriptor;
@group(0) @binding(1) var<storage, read_write> signers:   array<Signer>;
@group(0) @binding(2) var<storage, read_write> liquidity: array<LiquidityEntry>;
@group(0) @binding(3) var<storage, read_write> daily:     array<DailyLimit>;
@group(0) @binding(4) var<storage, read_write> inbox:     array<Message>;
@group(0) @binding(5) var<storage, read_write> outbox:    array<Message>;
@group(0) @binding(6) var<storage, read_write> epoch:     BridgeVMEpochState;
@group(0) @binding(7) var<storage, read_write> result:    BridgeVMTransitionResult;
// counts: [signers, liquidity, daily, inbox/outbox-shared]
@group(0) @binding(8) var<uniform>             counts_u:  vec4<u32>;
// counts2: [outbox, 0, 0, 0]  (inbox/outbox sizes both passed; outbox here)
@group(0) @binding(9) var<uniform>             counts2_u: vec4<u32>;

fn fold_keccak(acc: ptr<function, array<u32, 8>>, leaf: ptr<function, array<u32, 8>>) {
    var buf: array<u32, 256>;
    for (var i: u32 = 0u; i < 256u; i = i + 1u) { buf[i] = 0u; }
    put_digest32(&buf, 0u,  acc);
    put_digest32(&buf, 32u, leaf);
    keccak256_buf256(&buf, 64u, acc);
}

fn msg_id_zero_idx(slot_msg_id: ptr<function, array<u32, 8>>) -> bool {
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        if ((*slot_msg_id)[k] != 0u) { return false; }
    }
    return true;
}

@compute @workgroup_size(1)
fn bridgevm_transition(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }

    let signer_count = counts_u.x;
    let liq_count    = counts_u.y;
    let daily_count  = counts_u.z;
    let inbox_count  = counts_u.w;
    let outbox_count = counts2_u.x;

    // -- 1. close-epoch transitions over signers --
    var pending_drop: u32 = 0u;
    var target_epoch: u32 = desc.epoch_lo;
    if (desc.closing_flag != 0u) { target_epoch = target_epoch + 1u; }

    for (var i: u32 = 0u; i < signer_count; i = i + 1u) {
        if (signers[i].occupied == 0u) { continue; }
        var st = signers[i].status;
        if ((st & kSignerStatusPendingAdd) != 0u) {
            st = st & (~kSignerStatusPendingAdd);
        }
        if ((st & kSignerStatusPendingDrop) != 0u) {
            st = st & (~kSignerStatusPendingDrop);
            st = st | kSignerStatusTombstoned;
            pending_drop = pending_drop + 1u;
        }
        if ((st & kSignerStatusExiting) != 0u
            && !(signers[i].exit_epoch_lo == 0u && signers[i].exit_epoch_hi == 0u)
            && !u64_lt(target_epoch, 0u, signers[i].exit_epoch_lo, signers[i].exit_epoch_hi)
            && (st & kSignerStatusTombstoned) == 0u) {
            st = st & (~kSignerStatusExiting);
            st = st | kSignerStatusTombstoned;
            pending_drop = pending_drop + 1u;
        }
        if ((st & kSignerStatusJailed) != 0u
            && signers[i].jail_until_epoch != 0u
            && target_epoch >= signers[i].jail_until_epoch
            && (st & kSignerStatusTombstoned) == 0u) {
            st = st & (~kSignerStatusJailed);
            st = st | kSignerStatusActive;
            signers[i].jail_until_epoch = 0u;
        }
        signers[i].status = st;
    }
    epoch.pending_drop_count = pending_drop;

    // -- 2. reset daily limits whose reset_epoch passed --
    for (var i: u32 = 0u; i < daily_count; i = i + 1u) {
        if (daily[i].status == 0u) { continue; }
        if (target_epoch >= daily[i].reset_epoch_lo) {
            daily[i].used_today_lo_lo = 0u;
            daily[i].used_today_lo_hi = 0u;
            daily[i].used_today_hi_lo = 0u;
            daily[i].used_today_hi_hi = 0u;
            daily[i].reset_epoch_lo = target_epoch + 1u;
            daily[i].reset_epoch_hi = 0u;
        }
    }

    // -- 3. signer_set_root + counts + bond --
    var acc: array<u32, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { acc[k] = 0u; }
    var active_n: u32 = 0u;
    var jailed: u32 = 0u;
    var tombstoned: u32 = 0u;
    var bond = u128_make(0u, 0u, 0u, 0u);
    for (var i: u32 = 0u; i < signer_count; i = i + 1u) {
        if (signers[i].occupied == 0u) { continue; }
        let st = signers[i].status;
        if ((st & kSignerStatusTombstoned) != 0u) { tombstoned = tombstoned + 1u; }
        if ((st & kSignerStatusJailed) != 0u)     { jailed = jailed + 1u; }
        if ((st & kSignerStatusActive) != 0u
            && (st & kSignerStatusJailed) == 0u
            && (st & kSignerStatusTombstoned) == 0u) {
            active_n = active_n + 1u;
            let amt = u128_make(signers[i].bond_amount_lo_lo,
                                signers[i].bond_amount_lo_hi,
                                signers[i].bond_amount_hi_lo,
                                signers[i].bond_amount_hi_hi);
            bond = u128_add(bond, amt);
        }
        // Build leaf: 8+20+4+8+8+8+8+8+48+32+32+4+4+4+4 = 196 bytes.
        var buf: array<u32, 256>;
        for (var k: u32 = 0u; k < 256u; k = k + 1u) { buf[k] = 0u; }
        var o: u32 = 0u;
        put_u64_le(&buf, o, signers[i].signer_id_lo, signers[i].signer_id_hi); o = o + 8u;
        var addr: array<u32, 5>;
        for (var k: u32 = 0u; k < 5u; k = k + 1u) { addr[k] = signers[i].lux_address[k]; }
        put_addr20(&buf, o, &addr);                                            o = o + 20u;
        put_u32_le(&buf, o, 0u);                                               o = o + 4u;
        put_u64_le(&buf, o, signers[i].bond_amount_lo_lo, signers[i].bond_amount_lo_hi); o = o + 8u;
        put_u64_le(&buf, o, signers[i].bond_amount_hi_lo, signers[i].bond_amount_hi_hi); o = o + 8u;
        put_u64_le(&buf, o, signers[i].opt_in_height_lo, signers[i].opt_in_height_hi);   o = o + 8u;
        put_u64_le(&buf, o, signers[i].exit_epoch_lo,    signers[i].exit_epoch_hi);      o = o + 8u;
        put_u64_le(&buf, o, signers[i].sign_count_lo,    signers[i].sign_count_hi);      o = o + 8u;
        var bls: array<u32, 12>;
        for (var k: u32 = 0u; k < 12u; k = k + 1u) { bls[k] = signers[i].bls_pubkey[k]; }
        put_bls48(&buf, o, &bls);                                              o = o + 48u;
        var rt: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { rt[k] = signers[i].ringtail_pubkey[k]; }
        put_digest32(&buf, o, &rt);                                            o = o + 32u;
        var md: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { md[k] = signers[i].mldsa_pubkey[k]; }
        put_digest32(&buf, o, &md);                                            o = o + 32u;
        put_u32_le(&buf, o, signers[i].status);                                o = o + 4u;
        put_u32_le(&buf, o, signers[i].jail_until_epoch);                      o = o + 4u;
        put_u32_le(&buf, o, signers[i].slash_count);                           o = o + 4u;
        put_u32_le(&buf, o, i);                                                o = o + 4u;
        var leaf_hash: array<u32, 8>;
        keccak256_buf256(&buf, o, &leaf_hash);
        fold_keccak(&acc, &leaf_hash);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { epoch.signer_set_root[k] = acc[k]; }

    // -- 4. liquidity_root --
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { acc[k] = 0u; }
    for (var i: u32 = 0u; i < liq_count; i = i + 1u) {
        if (liquidity[i].status == 0u) { continue; }
        var buf: array<u32, 256>;
        for (var k: u32 = 0u; k < 256u; k = k + 1u) { buf[k] = 0u; }
        var o: u32 = 0u;
        var addr: array<u32, 5>;
        for (var k: u32 = 0u; k < 5u; k = k + 1u) { addr[k] = liquidity[i].provider_addr[k]; }
        put_addr20(&buf, o, &addr);                                                  o = o + 20u;
        put_u32_le(&buf, o, 0u);                                                     o = o + 4u;
        put_u32_le(&buf, o, liquidity[i].asset_id);                                  o = o + 4u;
        put_u32_le(&buf, o, liquidity[i].status);                                    o = o + 4u;
        put_u64_le(&buf, o, liquidity[i].amount_lo_lo, liquidity[i].amount_lo_hi);   o = o + 8u;
        put_u64_le(&buf, o, liquidity[i].amount_hi_lo, liquidity[i].amount_hi_hi);   o = o + 8u;
        put_u64_le(&buf, o, liquidity[i].fee_accrual_lo_lo, liquidity[i].fee_accrual_lo_hi); o = o + 8u;
        put_u64_le(&buf, o, liquidity[i].fee_accrual_hi_lo, liquidity[i].fee_accrual_hi_hi); o = o + 8u;
        put_u32_le(&buf, o, i);                                                      o = o + 4u;
        var leaf_hash: array<u32, 8>;
        keccak256_buf256(&buf, o, &leaf_hash);
        fold_keccak(&acc, &leaf_hash);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { epoch.liquidity_root[k] = acc[k]; }

    // -- 5. inbox_root --
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { acc[k] = 0u; }
    for (var i: u32 = 0u; i < inbox_count; i = i + 1u) {
        if (inbox[i].status == 0u) {
            var sid: array<u32, 8>;
            for (var k: u32 = 0u; k < 8u; k = k + 1u) { sid[k] = inbox[i].msg_id[k]; }
            if (msg_id_zero_idx(&sid)) { continue; }
        }
        var buf: array<u32, 256>;
        for (var k: u32 = 0u; k < 256u; k = k + 1u) { buf[k] = 0u; }
        var o: u32 = 0u;
        var msgid: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { msgid[k] = inbox[i].msg_id[k]; }
        put_digest32(&buf, o, &msgid);                                          o = o + 32u;
        var pr: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { pr[k] = inbox[i].payload_root[k]; }
        put_digest32(&buf, o, &pr);                                             o = o + 32u;
        put_u64_le(&buf, o, inbox[i].nonce_lo, inbox[i].nonce_hi);              o = o + 8u;
        put_u32_le(&buf, o, inbox[i].src_chain);                                o = o + 4u;
        put_u32_le(&buf, o, inbox[i].dst_chain);                                o = o + 4u;
        put_u32_le(&buf, o, inbox[i].kind);                                     o = o + 4u;
        put_u32_le(&buf, o, inbox[i].status);                                   o = o + 4u;
        put_u64_le(&buf, o, inbox[i].amount_lo_lo, inbox[i].amount_lo_hi);      o = o + 8u;
        put_u64_le(&buf, o, inbox[i].amount_hi_lo, inbox[i].amount_hi_hi);      o = o + 8u;
        put_u32_le(&buf, o, i);                                                 o = o + 4u;
        var leaf_hash: array<u32, 8>;
        keccak256_buf256(&buf, o, &leaf_hash);
        fold_keccak(&acc, &leaf_hash);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { epoch.inbox_root[k] = acc[k]; }

    // -- 6. outbox_root --
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { acc[k] = 0u; }
    for (var i: u32 = 0u; i < outbox_count; i = i + 1u) {
        if (outbox[i].status == 0u) {
            var sid: array<u32, 8>;
            for (var k: u32 = 0u; k < 8u; k = k + 1u) { sid[k] = outbox[i].msg_id[k]; }
            if (msg_id_zero_idx(&sid)) { continue; }
        }
        var buf: array<u32, 256>;
        for (var k: u32 = 0u; k < 256u; k = k + 1u) { buf[k] = 0u; }
        var o: u32 = 0u;
        var msgid: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { msgid[k] = outbox[i].msg_id[k]; }
        put_digest32(&buf, o, &msgid);                                          o = o + 32u;
        var pr: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { pr[k] = outbox[i].payload_root[k]; }
        put_digest32(&buf, o, &pr);                                             o = o + 32u;
        put_u64_le(&buf, o, outbox[i].nonce_lo, outbox[i].nonce_hi);            o = o + 8u;
        put_u32_le(&buf, o, outbox[i].src_chain);                               o = o + 4u;
        put_u32_le(&buf, o, outbox[i].dst_chain);                               o = o + 4u;
        put_u32_le(&buf, o, outbox[i].kind);                                    o = o + 4u;
        put_u32_le(&buf, o, outbox[i].status);                                  o = o + 4u;
        put_u64_le(&buf, o, outbox[i].amount_lo_lo, outbox[i].amount_lo_hi);    o = o + 8u;
        put_u64_le(&buf, o, outbox[i].amount_hi_lo, outbox[i].amount_hi_hi);    o = o + 8u;
        put_u32_le(&buf, o, i);                                                 o = o + 4u;
        var leaf_hash: array<u32, 8>;
        keccak256_buf256(&buf, o, &leaf_hash);
        fold_keccak(&acc, &leaf_hash);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { epoch.outbox_root[k] = acc[k]; }

    // -- 7. daily_limit_root --
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { acc[k] = 0u; }
    for (var i: u32 = 0u; i < daily_count; i = i + 1u) {
        if (daily[i].status == 0u) { continue; }
        var buf: array<u32, 256>;
        for (var k: u32 = 0u; k < 256u; k = k + 1u) { buf[k] = 0u; }
        var o: u32 = 0u;
        put_u32_le(&buf, o, daily[i].asset_id);                                  o = o + 4u;
        put_u32_le(&buf, o, daily[i].status);                                    o = o + 4u;
        put_u64_le(&buf, o, daily[i].daily_cap_lo_lo, daily[i].daily_cap_lo_hi); o = o + 8u;
        put_u64_le(&buf, o, daily[i].daily_cap_hi_lo, daily[i].daily_cap_hi_hi); o = o + 8u;
        put_u64_le(&buf, o, daily[i].used_today_lo_lo, daily[i].used_today_lo_hi); o = o + 8u;
        put_u64_le(&buf, o, daily[i].used_today_hi_lo, daily[i].used_today_hi_hi); o = o + 8u;
        put_u32_le(&buf, o, i);                                                  o = o + 4u;
        var leaf_hash: array<u32, 8>;
        keccak256_buf256(&buf, o, &leaf_hash);
        fold_keccak(&acc, &leaf_hash);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { epoch.daily_limit_root[k] = acc[k]; }

    // -- 8. epoch metadata --
    epoch.active_signer_count = active_n;
    epoch.total_active_bond_lo_lo = bond.lo_lo;
    epoch.total_active_bond_lo_hi = bond.lo_hi;
    epoch.total_active_bond_hi_lo = bond.hi_lo;
    epoch.total_active_bond_hi_hi = bond.hi_hi;
    if (desc.closing_flag != 0u) {
        epoch.current_epoch_lo = target_epoch;
        epoch.current_epoch_hi = 0u;
    }

    // -- 9. composed bridgevm_state_root --
    var buf: array<u32, 256>;
    for (var k: u32 = 0u; k < 256u; k = k + 1u) { buf[k] = 0u; }
    var o: u32 = 0u;
    var psr: array<u32, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { psr[k] = desc.parent_state_root[k]; }
    put_digest32(&buf, o, &psr);                                  o = o + 32u;
    var sr: array<u32, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { sr[k] = epoch.signer_set_root[k]; }
    put_digest32(&buf, o, &sr);                                   o = o + 32u;
    var lr: array<u32, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { lr[k] = epoch.liquidity_root[k]; }
    put_digest32(&buf, o, &lr);                                   o = o + 32u;
    var ir: array<u32, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { ir[k] = epoch.inbox_root[k]; }
    put_digest32(&buf, o, &ir);                                   o = o + 32u;
    var or_: array<u32, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { or_[k] = epoch.outbox_root[k]; }
    put_digest32(&buf, o, &or_);                                  o = o + 32u;
    var dr: array<u32, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { dr[k] = epoch.daily_limit_root[k]; }
    put_digest32(&buf, o, &dr);                                   o = o + 32u;
    put_u64_le(&buf, o, epoch.current_epoch_lo, epoch.current_epoch_hi); o = o + 8u;
    put_u64_le(&buf, o, epoch.total_active_bond_lo_lo, epoch.total_active_bond_lo_hi); o = o + 8u;
    put_u64_le(&buf, o, epoch.total_active_bond_hi_lo, epoch.total_active_bond_hi_hi); o = o + 8u;
    put_u32_le(&buf, o, epoch.active_signer_count);               o = o + 4u;
    var sroot: array<u32, 8>;
    keccak256_buf256(&buf, o, &sroot);
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { epoch.bridgevm_state_root[k] = sroot[k]; }

    // -- 10. result --
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { result.signer_set_root[k]     = epoch.signer_set_root[k]; }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { result.liquidity_root[k]      = epoch.liquidity_root[k]; }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { result.inbox_root[k]          = epoch.inbox_root[k]; }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { result.outbox_root[k]         = epoch.outbox_root[k]; }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { result.daily_limit_root[k]    = epoch.daily_limit_root[k]; }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { result.bridgevm_state_root[k] = epoch.bridgevm_state_root[k]; }
    result.active_signer_count    = active_n;
    result.jailed_count           = jailed;
    result.tombstoned_count       = tombstoned;
    result.total_active_bond_lo_lo = bond.lo_lo;
    result.total_active_bond_lo_hi = bond.lo_hi;
    result.total_active_bond_hi_lo = bond.hi_lo;
    result.total_active_bond_hi_hi = bond.hi_hi;
    result.epoch_lo               = epoch.current_epoch_lo;
    result.epoch_hi               = epoch.current_epoch_hi;
    result.status                 = 1u;
}
