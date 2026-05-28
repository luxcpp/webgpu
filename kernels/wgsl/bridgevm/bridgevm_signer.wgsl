// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// bridgevm_signer.wgsl — SignerSetApply kernel (WGSL).
//
// Mirrors bridgevm_signer.cu / .metal byte-for-byte. Single-thread canonical
// traversal preserves insertion-order determinism with the CPU oracle.

@group(0) @binding(0) var<storage, read>       desc:    BridgeVMRoundDescriptor;
@group(0) @binding(1) var<storage, read>       ops:     array<SignerOp>;
@group(0) @binding(2) var<storage, read_write> signers: array<Signer>;
@group(0) @binding(3) var<storage, read_write> applied_out: atomic<u32>;
@group(0) @binding(4) var<uniform>             counts_u: vec4<u32>;  // [signer_count, 0, 0, 0]

// Open-address locate by signer_id. mask = signer_count - 1u (power of two).
fn signer_locate(id_lo: u32, id_hi: u32, count: u32, insert_if_missing: bool) -> u32 {
    let mask = count - 1u;
    var idx = hash_u64(id_lo, id_hi, mask);
    for (var probe: u32 = 0u; probe < count; probe = probe + 1u) {
        let occ = signers[idx].occupied;
        if (occ == 0u) {
            if (insert_if_missing) {
                signers[idx].signer_id_lo = id_lo;
                signers[idx].signer_id_hi = id_hi;
                signers[idx].occupied = 1u;
                signers[idx].status = 0u;
                signers[idx].bond_amount_lo_lo = 0u;
                signers[idx].bond_amount_lo_hi = 0u;
                signers[idx].bond_amount_hi_lo = 0u;
                signers[idx].bond_amount_hi_hi = 0u;
                signers[idx].opt_in_height_lo = 0u;
                signers[idx].opt_in_height_hi = 0u;
                signers[idx].exit_epoch_lo = 0u;
                signers[idx].exit_epoch_hi = 0u;
                signers[idx].sign_count_lo = 0u;
                signers[idx].sign_count_hi = 0u;
                signers[idx].jail_until_epoch = 0u;
                signers[idx].slash_count = 0u;
                for (var k: u32 = 0u; k < 5u;  k = k + 1u) { signers[idx].lux_address[k] = 0u; }
                for (var k: u32 = 0u; k < 12u; k = k + 1u) { signers[idx].bls_pubkey[k] = 0u; }
                for (var k: u32 = 0u; k < 8u;  k = k + 1u) { signers[idx].ringtail_pubkey[k] = 0u; }
                for (var k: u32 = 0u; k < 8u;  k = k + 1u) { signers[idx].mldsa_pubkey[k] = 0u; }
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (signers[idx].signer_id_lo == id_lo && signers[idx].signer_id_hi == id_hi) {
            return idx;
        }
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

fn count_active(count: u32) -> u32 {
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

@compute @workgroup_size(1)
fn bridgevm_signer_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    var applied: u32 = 0u;
    let signer_count = counts_u.x;
    let n_ops = desc.signer_op_count;

    for (var i: u32 = 0u; i < n_ops; i = i + 1u) {
        let kind = ops[i].kind;
        if (kind == kSOpOptIn) {
            // Bond must meet the LP-333 minimum.
            let bond = u128_make(ops[i].bond_amount_lo_lo, ops[i].bond_amount_lo_hi,
                                 ops[i].bond_amount_hi_lo, ops[i].bond_amount_hi_hi);
            let mn   = u128_make(kMinSignerBondLoLo, kMinSignerBondLoHi,
                                 kMinSignerBondHiLo, kMinSignerBondHiHi);
            if (u128_lt(bond, mn)) { continue; }
            // Cap on simultaneously-active signers.
            let active_n = count_active(signer_count);
            if (active_n >= kMaxSigners) { continue; }
            let idx = signer_locate(ops[i].signer_id_lo, ops[i].signer_id_hi,
                                    signer_count, true);
            if (idx == 0xFFFFFFFFu) { continue; }
            for (var k: u32 = 0u; k < 5u;  k = k + 1u) { signers[idx].lux_address[k] = ops[i].lux_address[k]; }
            signers[idx].bond_amount_lo_lo = ops[i].bond_amount_lo_lo;
            signers[idx].bond_amount_lo_hi = ops[i].bond_amount_lo_hi;
            signers[idx].bond_amount_hi_lo = ops[i].bond_amount_hi_lo;
            signers[idx].bond_amount_hi_hi = ops[i].bond_amount_hi_hi;
            signers[idx].opt_in_height_lo = ops[i].opt_in_height_lo;
            signers[idx].opt_in_height_hi = ops[i].opt_in_height_hi;
            for (var k: u32 = 0u; k < 12u; k = k + 1u) { signers[idx].bls_pubkey[k] = ops[i].bls_pubkey[k]; }
            for (var k: u32 = 0u; k < 8u;  k = k + 1u) { signers[idx].ringtail_pubkey[k] = ops[i].ringtail_pubkey[k]; }
            for (var k: u32 = 0u; k < 8u;  k = k + 1u) { signers[idx].mldsa_pubkey[k] = ops[i].mldsa_pubkey[k]; }
            signers[idx].status = kSignerStatusActive | kSignerStatusPendingAdd;
            signers[idx].jail_until_epoch = 0u;
            applied = applied + 1u;
            continue;
        }
        if (kind == kSOpOptOut) {
            let idx = signer_locate(ops[i].signer_id_lo, ops[i].signer_id_hi,
                                    signer_count, false);
            if (idx == 0xFFFFFFFFu) { continue; }
            if ((signers[idx].status & kSignerStatusTombstoned) != 0u) { continue; }
            var st = signers[idx].status;
            st = st | kSignerStatusExiting;
            st = st & (~kSignerStatusActive);
            signers[idx].status = st;
            // exit_epoch = epoch + 14
            let new_epoch = ops[i].epoch + 14u;
            signers[idx].exit_epoch_lo = new_epoch;
            signers[idx].exit_epoch_hi = 0u;
            applied = applied + 1u;
            continue;
        }
        if (kind == kSOpSlash) {
            let idx = signer_locate(ops[i].signer_id_lo, ops[i].signer_id_hi,
                                    signer_count, false);
            if (idx == 0xFFFFFFFFu) { continue; }
            if ((signers[idx].status & kSignerStatusTombstoned) != 0u) { continue; }
            let cur = u128_make(signers[idx].bond_amount_lo_lo,
                                signers[idx].bond_amount_lo_hi,
                                signers[idx].bond_amount_hi_lo,
                                signers[idx].bond_amount_hi_hi);
            let sl = u128_make(ops[i].slash_amount_lo, 0u,
                               ops[i].slash_amount_hi, 0u);
            let nb = u128_sub(cur, sl);
            signers[idx].bond_amount_lo_lo = nb.lo_lo;
            signers[idx].bond_amount_lo_hi = nb.lo_hi;
            signers[idx].bond_amount_hi_lo = nb.hi_lo;
            signers[idx].bond_amount_hi_hi = nb.hi_hi;
            signers[idx].slash_count = signers[idx].slash_count + 1u;
            // is_eq = any byte of evidence_digest is non-zero
            var is_eq = false;
            for (var k: u32 = 0u; k < 32u; k = k + 1u) {
                let word = ops[i].evidence_digest[k >> 2u];
                let sh = (k & 3u) * 8u;
                if (((word >> sh) & 0xFFu) != 0u) { is_eq = true; break; }
            }
            var st = signers[idx].status;
            if (is_eq) {
                st = st | kSignerStatusTombstoned;
                st = st & (~kSignerStatusActive);
            } else {
                st = st | kSignerStatusJailed;
                st = st & (~kSignerStatusActive);
                var jail_for = ops[i].jail_until_epoch;
                if (jail_for == 0u) { jail_for = 100u; }
                let until = ops[i].epoch + jail_for;
                if (until > signers[idx].jail_until_epoch) {
                    signers[idx].jail_until_epoch = until;
                }
            }
            // If bond fell below the minimum, force exiting.
            let mn = u128_make(kMinSignerBondLoLo, kMinSignerBondLoHi,
                               kMinSignerBondHiLo, kMinSignerBondHiHi);
            if (u128_lt(nb, mn)) {
                st = st | kSignerStatusExiting;
                st = st & (~kSignerStatusActive);
            }
            signers[idx].status = st;
            applied = applied + 1u;
            continue;
        }
        if (kind == kSOpUnjail) {
            let idx = signer_locate(ops[i].signer_id_lo, ops[i].signer_id_hi,
                                    signer_count, false);
            if (idx == 0xFFFFFFFFu) { continue; }
            if ((signers[idx].status & kSignerStatusTombstoned) != 0u) { continue; }
            if (ops[i].epoch < signers[idx].jail_until_epoch) { continue; }
            var st = signers[idx].status;
            st = st & (~kSignerStatusJailed);
            st = st | kSignerStatusActive;
            signers[idx].status = st;
            signers[idx].jail_until_epoch = 0u;
            applied = applied + 1u;
            continue;
        }
        if (kind == kSOpRotateKeys) {
            let idx = signer_locate(ops[i].signer_id_lo, ops[i].signer_id_hi,
                                    signer_count, false);
            if (idx == 0xFFFFFFFFu) { continue; }
            if ((signers[idx].status & kSignerStatusTombstoned) != 0u) { continue; }
            for (var k: u32 = 0u; k < 12u; k = k + 1u) { signers[idx].bls_pubkey[k] = ops[i].bls_pubkey[k]; }
            for (var k: u32 = 0u; k < 8u;  k = k + 1u) { signers[idx].ringtail_pubkey[k] = ops[i].ringtail_pubkey[k]; }
            for (var k: u32 = 0u; k < 8u;  k = k + 1u) { signers[idx].mldsa_pubkey[k] = ops[i].mldsa_pubkey[k]; }
            applied = applied + 1u;
            continue;
        }
    }
    atomicStore(&applied_out, applied);
}
