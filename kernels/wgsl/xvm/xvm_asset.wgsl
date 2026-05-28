// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_asset.wgsl — XAssetTransition kernel.
//
// Single-thread, canonical-order traversal of asset ops. Mirrors
// xvm_asset.metal byte-for-byte.

// Concatenated after xvm_kernels_common.wgsl + xvm_membership.wgsl + xvm_utxo.wgsl.

struct AssetParams {
    asset_count: u32,
    asset_op_count: u32,
    marker_count: u32,
    pad: u32,
    applied: atomic<u32>,
    exports_n: atomic<u32>,
    imports_n: atomic<u32>,
    minted_lo: atomic<u32>,
    minted_hi: atomic<u32>,
    burned_lo: atomic<u32>,
    burned_hi: atomic<u32>,
};

@group(0) @binding(0) var<storage, read>        as_desc: XVMRoundDescriptor;
@group(0) @binding(1) var<storage, read_write>  as_txs: array<XvmTx>;
@group(0) @binding(2) var<storage, read>        as_ops: array<AssetOp>;
@group(0) @binding(3) var<storage, read_write>  as_assets: array<Asset>;
@group(0) @binding(4) var<storage, read_write>  as_markers: array<AtomicExportMarker>;
@group(0) @binding(5) var<storage, read_write>  as_params: AssetParams;

// Locate (or insert) an asset_id in the open-addressing table. asset_count
// must be a power of two.
fn as_asset_locate(asset_id: ptr<function, array<u32, 8>>, count: u32, insert_if_missing: bool) -> u32 {
    let mask = count - 1u;
    var idx = asset_index_hash(asset_id, mask);
    for (var probe: u32 = 0u; probe < count; probe = probe + 1u) {
        if (as_assets[idx].occupied == 0u) {
            if (insert_if_missing) {
                for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                    as_assets[idx].asset_id[k] = (*asset_id)[k];
                    as_assets[idx].mint_authority[k] = 0u;
                }
                as_assets[idx].total_supply_lo_lo = 0u;
                as_assets[idx].total_supply_lo_hi = 0u;
                as_assets[idx].total_supply_hi_lo = 0u;
                as_assets[idx].total_supply_hi_hi = 0u;
                as_assets[idx].freeze_flag = kAssetActive;
                as_assets[idx].denomination = 0u;
                as_assets[idx].name_offset = 0u;
                as_assets[idx].name_length = 0u;
                as_assets[idx].occupied = 1u;
                as_assets[idx].pad0 = 0u;
                as_assets[idx].pad1_lo = 0u;
                as_assets[idx].pad1_hi = 0u;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        var entry: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            entry[k] = as_assets[idx].asset_id[k];
        }
        if (digest_eq8(&entry, asset_id)) { return idx; }
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

fn as_export_marker_locate(marker_id: ptr<function, array<u32, 8>>, count: u32, insert_if_missing: bool) -> u32 {
    let mask = count - 1u;
    var idx = asset_index_hash(marker_id, mask);
    for (var probe: u32 = 0u; probe < count; probe = probe + 1u) {
        if (as_markers[idx].occupied == 0u) {
            if (insert_if_missing) {
                for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                    as_markers[idx].marker_id[k] = (*marker_id)[k];
                    as_markers[idx].asset_id[k] = 0u;
                    as_markers[idx].recipient_root[k] = 0u;
                }
                as_markers[idx].amount_lo_lo = 0u;
                as_markers[idx].amount_lo_hi = 0u;
                as_markers[idx].amount_hi_lo = 0u;
                as_markers[idx].amount_hi_hi = 0u;
                as_markers[idx].source_chain = 0u;
                as_markers[idx].target_chain = 0u;
                as_markers[idx].status = kExportPending;
                as_markers[idx].occupied = 1u;
                as_markers[idx].pad0_lo = 0u;
                as_markers[idx].pad0_hi = 0u;
                as_markers[idx].pad1_lo = 0u;
                as_markers[idx].pad1_hi = 0u;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        var entry: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            entry[k] = as_markers[idx].marker_id[k];
        }
        if (digest_eq8(&entry, marker_id)) { return idx; }
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

// Compose marker_id = keccak(tx_id || target_chain || amount_lo || amount_hi).
// Matches xvm_kernels_common.h.metal::compose_marker_id byte-for-byte.
fn as_compose_marker_id(tx_id: ptr<function, array<u32, 8>>,
                        target_chain: u32,
                        amount_lo_lo: u32, amount_lo_hi: u32,
                        amount_hi_lo: u32, amount_hi_hi: u32,
                        out: ptr<function, array<u32, 8>>)
{
    // 32 + 4 + 8 + 8 = 52 bytes = 13 u32 words.
    var buf: array<u32, 64>;
    for (var i: u32 = 0u; i < 64u; i = i + 1u) { buf[i] = 0u; }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { buf[i] = (*tx_id)[i]; }
    buf[8] = target_chain;
    buf[9] = amount_lo_lo;
    buf[10] = amount_lo_hi;
    buf[11] = amount_hi_lo;
    buf[12] = amount_hi_hi;
    keccak256(&buf, 52u, out);
}

@compute @workgroup_size(1, 1, 1)
fn xvm_asset_transition(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }

    var applied: u32 = 0u;
    var exports_n: u32 = 0u;
    var imports_n: u32 = 0u;
    var minted_lo: u32 = 0u; var minted_hi: u32 = 0u;
    var burned_lo: u32 = 0u; var burned_hi: u32 = 0u;

    let tx_count = as_desc.tx_count;
    for (var ti: u32 = 0u; ti < tx_count; ti = ti + 1u) {
        if (as_txs[ti].status == kTxStatusRejected) { continue; }
        if (as_txs[ti].asset_changes_count == 0u) { continue; }
        if (as_txs[ti].asset_changes_offset >= as_params.asset_op_count) { continue; }

        var tx_done: bool = false;
        let cnt = as_txs[ti].asset_changes_count;
        let base = as_txs[ti].asset_changes_offset;
        for (var k: u32 = 0u; k < cnt; k = k + 1u) {
            if (tx_done) { break; }
            let off = base + k;
            if (off >= as_params.asset_op_count) { break; }

            var asset_id: array<u32, 8>;
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                asset_id[i] = as_ops[off].asset_id[i];
            }

            var a_idx = as_asset_locate(&asset_id, as_params.asset_count, false);
            if (a_idx == 0xFFFFFFFFu) {
                if (as_ops[off].kind == kAssetOpMint) {
                    a_idx = as_asset_locate(&asset_id, as_params.asset_count, true);
                    if (a_idx == 0xFFFFFFFFu) {
                        as_txs[ti].status = kTxStatusRejected;
                        as_txs[ti].reject_reason = kRejectArenaFull;
                        tx_done = true; continue;
                    }
                    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                        as_assets[a_idx].mint_authority[i] = as_ops[off].authority_witness[i];
                    }
                } else {
                    as_txs[ti].status = kTxStatusRejected;
                    as_txs[ti].reject_reason = kRejectAssetMissing;
                    tx_done = true; continue;
                }
            }

            let kind = as_ops[off].kind;
            let amt_lo_lo = as_ops[off].amount_lo_lo;
            let amt_lo_hi = as_ops[off].amount_lo_hi;
            let amt_hi_lo = as_ops[off].amount_hi_lo;
            let amt_hi_hi = as_ops[off].amount_hi_hi;

            if (kind == kAssetOpMint) {
                var witness: array<u32, 8>;
                var auth: array<u32, 8>;
                for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                    witness[i] = as_ops[off].authority_witness[i];
                    auth[i] = as_assets[a_idx].mint_authority[i];
                }
                if (!digest_eq8(&auth, &witness)) {
                    as_txs[ti].status = kTxStatusRejected;
                    as_txs[ti].reject_reason = kRejectMintAuthority;
                    tx_done = true; continue;
                }
                let new_supply = u64_add(as_assets[a_idx].total_supply_lo_lo,
                                         as_assets[a_idx].total_supply_lo_hi,
                                         amt_lo_lo, amt_lo_hi);
                as_assets[a_idx].total_supply_lo_lo = new_supply.x;
                as_assets[a_idx].total_supply_lo_hi = new_supply.y;
                let mint_sum = u64_add(minted_lo, minted_hi, amt_lo_lo, amt_lo_hi);
                minted_lo = mint_sum.x; minted_hi = mint_sum.y;
                applied = applied + 1u;
            } else if (kind == kAssetOpBurn) {
                let r = u64_sub_checked(as_assets[a_idx].total_supply_lo_lo,
                                        as_assets[a_idx].total_supply_lo_hi,
                                        amt_lo_lo, amt_lo_hi);
                if (r.z == 0u) {
                    as_txs[ti].status = kTxStatusRejected;
                    as_txs[ti].reject_reason = kRejectAmountOverflow;
                    tx_done = true; continue;
                }
                as_assets[a_idx].total_supply_lo_lo = r.x;
                as_assets[a_idx].total_supply_lo_hi = r.y;
                let burn_sum = u64_add(burned_lo, burned_hi, amt_lo_lo, amt_lo_hi);
                burned_lo = burn_sum.x; burned_hi = burn_sum.y;
                applied = applied + 1u;
            } else if (kind == kAssetOpTransfer) {
                applied = applied + 1u;
            } else if (kind == kAssetOpExport) {
                let r = u64_sub_checked(as_assets[a_idx].total_supply_lo_lo,
                                        as_assets[a_idx].total_supply_lo_hi,
                                        amt_lo_lo, amt_lo_hi);
                if (r.z == 0u) {
                    as_txs[ti].status = kTxStatusRejected;
                    as_txs[ti].reject_reason = kRejectAmountOverflow;
                    tx_done = true; continue;
                }
                as_assets[a_idx].total_supply_lo_lo = r.x;
                as_assets[a_idx].total_supply_lo_hi = r.y;
                var tx_id_local: array<u32, 8>;
                for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                    tx_id_local[i] = as_txs[ti].tx_id[i];
                }
                var marker_id: array<u32, 8>;
                as_compose_marker_id(&tx_id_local, as_ops[off].target_chain,
                                     amt_lo_lo, amt_lo_hi, amt_hi_lo, amt_hi_hi,
                                     &marker_id);
                let m_idx = as_export_marker_locate(&marker_id, as_params.marker_count, true);
                if (m_idx == 0xFFFFFFFFu) {
                    as_txs[ti].status = kTxStatusRejected;
                    as_txs[ti].reject_reason = kRejectArenaFull;
                    tx_done = true; continue;
                }
                for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                    as_markers[m_idx].asset_id[i] = as_ops[off].asset_id[i];
                    as_markers[m_idx].recipient_root[i] = as_ops[off].authority_witness[i];
                }
                as_markers[m_idx].amount_lo_lo = amt_lo_lo;
                as_markers[m_idx].amount_lo_hi = amt_lo_hi;
                as_markers[m_idx].amount_hi_lo = amt_hi_lo;
                as_markers[m_idx].amount_hi_hi = amt_hi_hi;
                as_markers[m_idx].source_chain = 0u;
                as_markers[m_idx].target_chain = as_ops[off].target_chain;
                exports_n = exports_n + 1u;
                applied = applied + 1u;
            } else if (kind == kAssetOpImport) {
                var proof: array<u32, 8>;
                for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                    proof[i] = as_txs[ti].proof_digest[i];
                }
                let m_idx = as_export_marker_locate(&proof, as_params.marker_count, false);
                if (m_idx == 0xFFFFFFFFu) {
                    as_txs[ti].status = kTxStatusRejected;
                    as_txs[ti].reject_reason = kRejectImportNoMarker;
                    tx_done = true; continue;
                }
                if (as_markers[m_idx].status == kExportConsumed) {
                    as_txs[ti].status = kTxStatusRejected;
                    as_txs[ti].reject_reason = kRejectImportNoMarker;
                    tx_done = true; continue;
                }
                as_markers[m_idx].status = kExportConsumed;
                let new_supply = u64_add(as_assets[a_idx].total_supply_lo_lo,
                                         as_assets[a_idx].total_supply_lo_hi,
                                         amt_lo_lo, amt_lo_hi);
                as_assets[a_idx].total_supply_lo_lo = new_supply.x;
                as_assets[a_idx].total_supply_lo_hi = new_supply.y;
                let mint_sum = u64_add(minted_lo, minted_hi, amt_lo_lo, amt_lo_hi);
                minted_lo = mint_sum.x; minted_hi = mint_sum.y;
                imports_n = imports_n + 1u;
                applied = applied + 1u;
            }
        }
    }

    atomicStore(&as_params.applied, applied);
    atomicStore(&as_params.exports_n, exports_n);
    atomicStore(&as_params.imports_n, imports_n);
    // Match Metal's split-low/high u32 atomic store pattern: minted/burned
    // stay in (lo, hi) 32-bit accumulators that the host folds into u64.
    atomicStore(&as_params.minted_lo, minted_lo);
    atomicStore(&as_params.minted_hi, minted_hi);
    atomicStore(&as_params.burned_lo, burned_lo);
    atomicStore(&as_params.burned_hi, burned_hi);
}
