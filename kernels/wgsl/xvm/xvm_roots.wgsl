// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_roots.wgsl — XRootUpdate kernel.
//
// Recomputes utxo_root, asset_root, tx_root and execution_root in canonical
// order. Tallies tx_accepted / tx_rejected from the per-tx status field.
// Single-thread, single-threadgroup. Mirrors xvm_roots.metal byte-for-byte.

// Concatenated after xvm_kernels_common.wgsl + previous kernels.

struct RootsParams {
    tx_count: u32,
    utxo_count: u32,
    asset_count: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read>        rt_desc: XVMRoundDescriptor;
@group(0) @binding(1) var<storage, read>        rt_txs: array<XvmTx>;
@group(0) @binding(2) var<storage, read>        rt_utxos: array<UTXO>;
@group(0) @binding(3) var<storage, read>        rt_assets: array<Asset>;
@group(0) @binding(4) var<storage, read_write>  rt_result: XVMTransitionResult;
@group(0) @binding(5) var<uniform>              rt_params: RootsParams;

// Fold leaf hash into accumulator: acc' = keccak(acc || leaf_hash).
fn rt_fold(acc: ptr<function, array<u32, 8>>, leaf_hash: ptr<function, array<u32, 8>>) {
    var buf: array<u32, 64>;
    for (var i: u32 = 0u; i < 64u; i = i + 1u) { buf[i] = 0u; }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        buf[i]     = (*acc)[i];
        buf[8u+i]  = (*leaf_hash)[i];
    }
    keccak256(&buf, 64u, acc);
}

@compute @workgroup_size(1, 1, 1)
fn xvm_root_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }

    var accepted: u32 = 0u;
    var rejected: u32 = 0u;
    let tx_count = rt_params.tx_count;
    for (var i: u32 = 0u; i < tx_count; i = i + 1u) {
        let st = rt_txs[i].status;
        if (st == kTxStatusAccepted) { accepted = accepted + 1u; }
        else if (st == kTxStatusRejected) { rejected = rejected + 1u; }
    }

    // -- utxo_root --
    // Leaf bytes: utxo_id(32) || asset_id(32) || amount_lo(8) || amount_hi(8)
    //          || owner_root(32) || locktime(8) || threshold(4) || status(4)
    //          || index(4) = 132 bytes = 33 u32 words.
    var acc: array<u32, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { acc[k] = 0u; }
    let utxo_count = rt_params.utxo_count;
    for (var i: u32 = 0u; i < utxo_count; i = i + 1u) {
        if ((rt_utxos[i].status & kUtxoOccupied) == 0u) { continue; }
        var leaf: array<u32, 64>;
        for (var k: u32 = 0u; k < 64u; k = k + 1u) { leaf[k] = 0u; }
        var off: u32 = 0u;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { leaf[off + k] = rt_utxos[i].utxo_id[k]; }   off = off + 8u;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { leaf[off + k] = rt_utxos[i].asset_id[k]; }  off = off + 8u;
        leaf[off] = rt_utxos[i].amount_lo_lo; off = off + 1u;
        leaf[off] = rt_utxos[i].amount_lo_hi; off = off + 1u;
        leaf[off] = rt_utxos[i].amount_hi_lo; off = off + 1u;
        leaf[off] = rt_utxos[i].amount_hi_hi; off = off + 1u;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { leaf[off + k] = rt_utxos[i].owner_root[k]; } off = off + 8u;
        leaf[off] = rt_utxos[i].locktime_lo; off = off + 1u;
        leaf[off] = rt_utxos[i].locktime_hi; off = off + 1u;
        leaf[off] = rt_utxos[i].threshold; off = off + 1u;
        leaf[off] = rt_utxos[i].status;    off = off + 1u;
        leaf[off] = i;                     off = off + 1u;
        var lh: array<u32, 8>;
        keccak256(&leaf, off * 4u, &lh);
        rt_fold(&acc, &lh);
    }
    var utxo_root: array<u32, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { utxo_root[k] = acc[k]; }

    // -- asset_root --
    // Leaf bytes: asset_id(32) || total_supply_lo(8) || total_supply_hi(8)
    //          || mint_authority(32) || freeze_flag(4) || denomination(4)
    //          || index(4) = 92 bytes = 23 u32 words.
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { acc[k] = 0u; }
    let asset_count = rt_params.asset_count;
    for (var i: u32 = 0u; i < asset_count; i = i + 1u) {
        if (rt_assets[i].occupied == 0u) { continue; }
        var leaf: array<u32, 64>;
        for (var k: u32 = 0u; k < 64u; k = k + 1u) { leaf[k] = 0u; }
        var off: u32 = 0u;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { leaf[off + k] = rt_assets[i].asset_id[k]; } off = off + 8u;
        leaf[off] = rt_assets[i].total_supply_lo_lo; off = off + 1u;
        leaf[off] = rt_assets[i].total_supply_lo_hi; off = off + 1u;
        leaf[off] = rt_assets[i].total_supply_hi_lo; off = off + 1u;
        leaf[off] = rt_assets[i].total_supply_hi_hi; off = off + 1u;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { leaf[off + k] = rt_assets[i].mint_authority[k]; } off = off + 8u;
        leaf[off] = rt_assets[i].freeze_flag;  off = off + 1u;
        leaf[off] = rt_assets[i].denomination; off = off + 1u;
        leaf[off] = i;                          off = off + 1u;
        var lh: array<u32, 8>;
        keccak256(&leaf, off * 4u, &lh);
        rt_fold(&acc, &lh);
    }
    var asset_root: array<u32, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { asset_root[k] = acc[k]; }

    // -- tx_root --
    // Leaf bytes: tx_id(32) || kind(4) || status(4) || reject_reason(4)
    //          || proof_digest(32) || index(4) = 80 bytes = 20 u32 words.
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { acc[k] = 0u; }
    for (var i: u32 = 0u; i < tx_count; i = i + 1u) {
        var leaf: array<u32, 64>;
        for (var k: u32 = 0u; k < 64u; k = k + 1u) { leaf[k] = 0u; }
        var off: u32 = 0u;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { leaf[off + k] = rt_txs[i].tx_id[k]; } off = off + 8u;
        leaf[off] = rt_txs[i].kind;          off = off + 1u;
        leaf[off] = rt_txs[i].status;        off = off + 1u;
        leaf[off] = rt_txs[i].reject_reason; off = off + 1u;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { leaf[off + k] = rt_txs[i].proof_digest[k]; } off = off + 8u;
        leaf[off] = i; off = off + 1u;
        var lh: array<u32, 8>;
        keccak256(&leaf, off * 4u, &lh);
        rt_fold(&acc, &lh);
    }
    var tx_root: array<u32, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { tx_root[k] = acc[k]; }

    // -- execution_root --
    // Bytes: parent(32) || utxo_root(32) || asset_root(32) || tx_root(32) || height(8)
    //      = 136 bytes = 34 u32 words.
    var composed: array<u32, 64>;
    for (var k: u32 = 0u; k < 64u; k = k + 1u) { composed[k] = 0u; }
    var off: u32 = 0u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { composed[off + k] = rt_desc.parent_execution_root[k]; } off = off + 8u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { composed[off + k] = utxo_root[k]; }                    off = off + 8u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { composed[off + k] = asset_root[k]; }                   off = off + 8u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { composed[off + k] = tx_root[k]; }                      off = off + 8u;
    composed[off] = rt_desc.height_lo; off = off + 1u;
    composed[off] = rt_desc.height_hi; off = off + 1u;
    var exec_root: array<u32, 8>;
    keccak256(&composed, off * 4u, &exec_root);

    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        rt_result.utxo_root[k]      = utxo_root[k];
        rt_result.asset_root[k]     = asset_root[k];
        rt_result.tx_root[k]        = tx_root[k];
        rt_result.execution_root[k] = exec_root[k];
    }
    rt_result.status = 1u;
    rt_result.tx_accepted = accepted;
    rt_result.tx_rejected = rejected;
    rt_result.height_lo = rt_desc.height_lo;
    rt_result.height_hi = rt_desc.height_hi;
}
