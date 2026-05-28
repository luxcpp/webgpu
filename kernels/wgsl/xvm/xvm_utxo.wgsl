// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_utxo.wgsl — fused UTXOInputCheck + UTXOTransitionApply kernel.
//
// Single-thread, canonical-order traversal of XvmTx records. Mirrors
// xvm_utxo.metal byte-for-byte.

// Concatenated after xvm_kernels_common.wgsl + xvm_membership.wgsl.

struct UtxoParams {
    utxo_count: u32,
    bloom_bit_count: u32,
    cuckoo_bucket_count: u32,
    input_batch_count: u32,
    output_batch_count: u32,
    outputs_count: u32,
    inputs_consumed: atomic<u32>,
    outputs_created: atomic<u32>,
};

@group(0) @binding(0) var<storage, read>        ut_desc: XVMRoundDescriptor;
@group(0) @binding(1) var<storage, read_write>  ut_txs: array<XvmTx>;
@group(0) @binding(2) var<storage, read>        ut_input_batches: array<InputBatch>;
@group(0) @binding(3) var<storage, read>        ut_output_batches: array<OutputBatch>;
@group(0) @binding(4) var<storage, read>        ut_inputs: array<u32>;        // u32-packed bytes; 8 u32 per utxo_id
@group(0) @binding(5) var<storage, read>        ut_outputs: array<UTXO>;
@group(0) @binding(6) var<storage, read_write>  ut_utxos: array<UTXO>;
@group(0) @binding(7) var<storage, read_write>  ut_bloom: array<u32>;
@group(0) @binding(8) var<storage, read_write>  ut_cuckoo: array<CuckooEntry>;
@group(0) @binding(9) var<storage, read_write>  ut_params: UtxoParams;

// Read a 32-byte utxo_id from inputs[] at byte offset `byte_off`. inputs is
// stored as a u32 array — byte_off must be a multiple of 4 (always true:
// each utxo_id is 32 bytes). Returns 8 u32 lanes.
fn ut_read_input_uid(byte_off: u32, out: ptr<function, array<u32, 8>>) {
    let word_off = byte_off / 4u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        (*out)[i] = ut_inputs[word_off + i];
    }
}

fn ut_inputs_have_duplicates(input_offset_bytes: u32, input_count: u32) -> bool {
    if (input_count < 2u) { return false; }
    for (var i: u32 = 0u; i + 1u < input_count; i = i + 1u) {
        var a: array<u32, 8>;
        ut_read_input_uid(input_offset_bytes + i * 32u, &a);
        for (var j: u32 = i + 1u; j < input_count; j = j + 1u) {
            var b: array<u32, 8>;
            ut_read_input_uid(input_offset_bytes + j * 32u, &b);
            if (digest_eq8(&a, &b)) { return true; }
        }
    }
    return false;
}

// Cuckoo query — returns slot_index (u32::MAX on miss).
fn ut_cuckoo_query(uid: ptr<function, array<u32, 8>>, bucket_count: u32) -> u32 {
    var hashes_lo: array<u32, 4>;
    mb_hash_quad(uid, &hashes_lo);
    for (var k: u32 = 0u; k < 2u; k = k + 1u) {
        let b = mb_cuckoo_bucket(hashes_lo[k], bucket_count);
        for (var s: u32 = 0u; s < kCuckooSlotsPerBucket; s = s + 1u) {
            let idx = b * kCuckooSlotsPerBucket + s;
            if (ut_cuckoo[idx].occupied != 0u) {
                var entry_uid: array<u32, 8>;
                for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                    entry_uid[i] = ut_cuckoo[idx].utxo_id[i];
                }
                if (digest_eq8(&entry_uid, uid)) {
                    return ut_cuckoo[idx].slot_index;
                }
            }
        }
    }
    return 0xFFFFFFFFu;
}

fn ut_cuckoo_remove(uid: ptr<function, array<u32, 8>>, bucket_count: u32) {
    var hashes_lo: array<u32, 4>;
    mb_hash_quad(uid, &hashes_lo);
    for (var k: u32 = 0u; k < 2u; k = k + 1u) {
        let b = mb_cuckoo_bucket(hashes_lo[k], bucket_count);
        for (var s: u32 = 0u; s < kCuckooSlotsPerBucket; s = s + 1u) {
            let idx = b * kCuckooSlotsPerBucket + s;
            if (ut_cuckoo[idx].occupied != 0u) {
                var entry_uid: array<u32, 8>;
                for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                    entry_uid[i] = ut_cuckoo[idx].utxo_id[i];
                }
                if (digest_eq8(&entry_uid, uid)) {
                    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                        ut_cuckoo[idx].utxo_id[i] = 0u;
                    }
                    ut_cuckoo[idx].slot_index = 0u;
                    ut_cuckoo[idx].occupied = 0u;
                    ut_cuckoo[idx].pad0_lo = 0u;
                    ut_cuckoo[idx].pad0_hi = 0u;
                    return;
                }
            }
        }
    }
}

// Linear-scan insert into UTXO arena. Returns slot index (u32::MAX on full).
fn ut_arena_insert(src_idx: u32, count: u32) -> u32 {
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        if ((ut_utxos[i].status & kUtxoOccupied) == 0u) {
            for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                ut_utxos[i].utxo_id[k]    = ut_outputs[src_idx].utxo_id[k];
                ut_utxos[i].asset_id[k]   = ut_outputs[src_idx].asset_id[k];
                ut_utxos[i].owner_root[k] = ut_outputs[src_idx].owner_root[k];
            }
            ut_utxos[i].amount_lo_lo = ut_outputs[src_idx].amount_lo_lo;
            ut_utxos[i].amount_lo_hi = ut_outputs[src_idx].amount_lo_hi;
            ut_utxos[i].amount_hi_lo = ut_outputs[src_idx].amount_hi_lo;
            ut_utxos[i].amount_hi_hi = ut_outputs[src_idx].amount_hi_hi;
            ut_utxos[i].locktime_lo  = ut_outputs[src_idx].locktime_lo;
            ut_utxos[i].locktime_hi  = ut_outputs[src_idx].locktime_hi;
            ut_utxos[i].threshold        = ut_outputs[src_idx].threshold;
            ut_utxos[i].addresses_offset = ut_outputs[src_idx].addresses_offset;
            ut_utxos[i].addresses_count  = ut_outputs[src_idx].addresses_count;
            ut_utxos[i].status           = kUtxoOccupied;
            ut_utxos[i].pad0_lo          = 0u;
            ut_utxos[i].pad0_hi          = 0u;
            return i;
        }
    }
    return 0xFFFFFFFFu;
}

// Bloom set — same as mb_bloom_set but on the ut_bloom binding.
fn ut_bloom_set(uid: ptr<function, array<u32, 8>>, bit_count: u32) {
    if (bit_count == 0u) { return; }
    var hashes_lo: array<u32, 4>;
    mb_hash_quad(uid, &hashes_lo);
    let mask = bit_count - 1u;
    for (var i: u32 = 0u; i < kBloomHashes; i = i + 1u) {
        let bit = hashes_lo[i] & mask;
        let word_off = bit >> 5u;
        let bit_off = bit & 31u;
        ut_bloom[word_off] = ut_bloom[word_off] | (1u << bit_off);
    }
}

fn ut_bloom_test(uid: ptr<function, array<u32, 8>>, bit_count: u32) -> bool {
    if (bit_count == 0u) { return true; }
    var hashes_lo: array<u32, 4>;
    mb_hash_quad(uid, &hashes_lo);
    let mask = bit_count - 1u;
    for (var i: u32 = 0u; i < kBloomHashes; i = i + 1u) {
        let bit = hashes_lo[i] & mask;
        let word_off = bit >> 5u;
        let bit_off = bit & 31u;
        if ((ut_bloom[word_off] & (1u << bit_off)) == 0u) { return false; }
    }
    return true;
}

// Cuckoo insert mirroring mb_cuckoo_insert but on ut_cuckoo binding.
fn ut_cuckoo_insert(uid: ptr<function, array<u32, 8>>, slot_index: u32, bucket_count: u32) -> bool {
    var hashes_lo: array<u32, 4>;
    mb_hash_quad(uid, &hashes_lo);
    for (var k: u32 = 0u; k < 2u; k = k + 1u) {
        let b = mb_cuckoo_bucket(hashes_lo[k], bucket_count);
        for (var s: u32 = 0u; s < kCuckooSlotsPerBucket; s = s + 1u) {
            let idx = b * kCuckooSlotsPerBucket + s;
            if (ut_cuckoo[idx].occupied != 0u) {
                var entry_uid: array<u32, 8>;
                for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                    entry_uid[i] = ut_cuckoo[idx].utxo_id[i];
                }
                if (digest_eq8(&entry_uid, uid)) {
                    ut_cuckoo[idx].slot_index = slot_index;
                    return true;
                }
            }
        }
    }
    for (var k: u32 = 0u; k < 2u; k = k + 1u) {
        let b = mb_cuckoo_bucket(hashes_lo[k], bucket_count);
        for (var s: u32 = 0u; s < kCuckooSlotsPerBucket; s = s + 1u) {
            let idx = b * kCuckooSlotsPerBucket + s;
            if (ut_cuckoo[idx].occupied == 0u) {
                for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                    ut_cuckoo[idx].utxo_id[i] = (*uid)[i];
                }
                ut_cuckoo[idx].slot_index = slot_index;
                ut_cuckoo[idx].occupied = 1u;
                ut_cuckoo[idx].pad0_lo = 0u;
                ut_cuckoo[idx].pad0_hi = 0u;
                return true;
            }
        }
    }
    return false;
}

@compute @workgroup_size(1, 1, 1)
fn xvm_utxo_transition(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }

    let height_lo = ut_desc.height_lo;
    let height_hi = ut_desc.height_hi;
    var inputs_consumed: u32 = 0u;
    var outputs_created: u32 = 0u;

    let tx_count = ut_desc.tx_count;
    for (var ti: u32 = 0u; ti < tx_count; ti = ti + 1u) {
        var reject: bool = false;
        var reject_reason: u32 = 0u;

        // Locate input batch (if any).
        let ib_off = ut_txs[ti].input_batch_offset;
        var has_ib: bool = ib_off < ut_params.input_batch_count;

        // Duplicate scan within batch.
        if (has_ib) {
            let in_off = ut_input_batches[ib_off].input_offset;
            let in_cnt = ut_input_batches[ib_off].input_count;
            if (ut_inputs_have_duplicates(in_off, in_cnt)) {
                reject = true;
                reject_reason = kRejectDuplicateInput;
            }
        }

        // Walk inputs. Track up to 256 consumed slots in a local array.
        var consumed_slots: array<u32, 256>;
        var consumed_n: u32 = 0u;
        if (!reject && has_ib) {
            let in_off = ut_input_batches[ib_off].input_offset;
            let in_cnt = ut_input_batches[ib_off].input_count;
            let wcount = ut_input_batches[ib_off].witness_count;
            for (var i: u32 = 0u; i < in_cnt; i = i + 1u) {
                var uid: array<u32, 8>;
                ut_read_input_uid(in_off + i * 32u, &uid);
                if (!ut_bloom_test(&uid, ut_params.bloom_bit_count)) {
                    reject = true; reject_reason = kRejectMissingInput; break;
                }
                let slot = ut_cuckoo_query(&uid, ut_params.cuckoo_bucket_count);
                if (slot == 0xFFFFFFFFu) {
                    reject = true; reject_reason = kRejectMissingInput; break;
                }
                if (slot >= ut_params.utxo_count) {
                    reject = true; reject_reason = kRejectMissingInput; break;
                }
                let st = ut_utxos[slot].status;
                if ((st & kUtxoOccupied) == 0u) {
                    reject = true; reject_reason = kRejectMissingInput; break;
                }
                if ((st & kUtxoSpent) != 0u) {
                    reject = true; reject_reason = kRejectAlreadySpent; break;
                }
                let lt_lo = ut_utxos[slot].locktime_lo;
                let lt_hi = ut_utxos[slot].locktime_hi;
                // locktime > height ?
                let locktime_gt = (lt_hi > height_hi) || (lt_hi == height_hi && lt_lo > height_lo);
                if (locktime_gt) {
                    reject = true; reject_reason = kRejectLocktime; break;
                }
                let thr = ut_utxos[slot].threshold;
                if (thr > 0u && wcount == 0u) {
                    reject = true; reject_reason = kRejectAuth; break;
                }
                if (consumed_n < 256u) {
                    consumed_slots[consumed_n] = slot;
                    consumed_n = consumed_n + 1u;
                }
            }
        }

        if (reject) {
            ut_txs[ti].status = kTxStatusRejected;
            ut_txs[ti].reject_reason = reject_reason;
            continue;
        }

        // Apply: mark inputs spent + remove from cuckoo.
        for (var i: u32 = 0u; i < consumed_n; i = i + 1u) {
            let slot = consumed_slots[i];
            var uid: array<u32, 8>;
            for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                uid[k] = ut_utxos[slot].utxo_id[k];
            }
            ut_utxos[slot].status = ut_utxos[slot].status | kUtxoSpent;
            ut_cuckoo_remove(&uid, ut_params.cuckoo_bucket_count);
            inputs_consumed = inputs_consumed + 1u;
        }

        // Apply: insert outputs.
        var arena_full: bool = false;
        let ob_off = ut_txs[ti].output_batch_offset;
        if (ob_off < ut_params.output_batch_count) {
            let out_off = ut_output_batches[ob_off].output_offset;
            let out_cnt = ut_output_batches[ob_off].output_count;
            for (var j: u32 = 0u; j < out_cnt; j = j + 1u) {
                let off = out_off + j;
                if (off >= ut_params.outputs_count) { break; }
                let new_slot = ut_arena_insert(off, ut_params.utxo_count);
                if (new_slot == 0xFFFFFFFFu) { arena_full = true; break; }
                var uid: array<u32, 8>;
                for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                    uid[k] = ut_outputs[off].utxo_id[k];
                }
                ut_bloom_set(&uid, ut_params.bloom_bit_count);
                if (!ut_cuckoo_insert(&uid, new_slot, ut_params.cuckoo_bucket_count)) {
                    arena_full = true; break;
                }
                outputs_created = outputs_created + 1u;
            }
        }
        if (arena_full) {
            ut_txs[ti].status = kTxStatusRejected;
            ut_txs[ti].reject_reason = kRejectArenaFull;
            continue;
        }
        ut_txs[ti].status = kTxStatusAccepted;
        ut_txs[ti].reject_reason = 0u;
    }

    atomicStore(&ut_params.inputs_consumed, inputs_consumed);
    atomicStore(&ut_params.outputs_created, outputs_created);
}
