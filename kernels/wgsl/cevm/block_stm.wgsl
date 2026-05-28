// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// WGSL port of cuda/kernels/cevm/block_stm.cu and the matching
// metal/src/shaders/cevm/block_stm.metal. GPU-native Block-STM optimistic
// concurrency control loop.
//
// One invocation per worker; same dispatch shape as CUDA / Metal. Each
// worker:
//   1. atomically grabs a tx from execution_idx or validation_idx
//   2. executes the tx (simplified balance-transfer logic — full EVM is
//      dispatched via evm_kernel.wgsl on a separate pass)
//   3. validates the tx's read-set against MvMemory
//   4. on conflict: increments incarnation, re-queues for execution
//
// Layout — all uchar[20]/uchar[32] arrays packed into u32 words. WGSL has
// no per-byte addressing on storage buffers so the host packs accordingly.
//
//   MvEntry  = 24 u32 words (was 96 bytes in Metal): tx_index, incarnation,
//              address[5 words], _pad, slot[8 words], value[8 words],
//              is_estimate, _pad2
//   TxState  =  6 u32 words: incarnation, validated, executed, status,
//              gas_used_lo, gas_used_hi (read_count and write_count split
//              between two extra words in CUDA/Metal but unused once we
//              flatten — host packs the canonical 8-word stride)
//   ReadSetEntry  = 16 u32 words: address[5], _pad, slot[8], read_tx_index,
//                   read_incarnation
//   WriteSetEntry = 21 u32 words: address[5], _pad, slot[8], value[8] (some
//                   compilers pad to 24)
//   Transaction (input): from[5], to[5], gas_limit (lo,hi), value (lo,hi),
//                        nonce (lo, hi), gas_price (lo, hi), calldata_off,
//                        calldata_sz = 18 words. Host pads to 20 for stride.
//
// All address/slot comparisons walk the u32 words.
//
// CRITICAL: The CUDA / Metal sources do per-byte atomic loads on
// `tx_index`. In WGSL we treat the MvEntry table as `array<atomic<u32>>`
// for the tx_index and is_estimate fields, and plain `array<u32>` writes
// for the body (address/slot/value) since the tx_index CAS is the publish
// barrier.

// =============================================================================
// Constants — must match cuda/block_stm.cu and metal/block_stm.metal.
// =============================================================================

const MAX_READS_PER_TX:    u32 = 64u;
const MAX_WRITES_PER_TX:   u32 = 64u;
const MV_TABLE_SIZE:       u32 = 65536u;
const MV_TABLE_MASK:       u32 = 65535u;
const MAX_INCARNATIONS:    u32 = 16u;
const MAX_SCHEDULER_LOOPS: u32 = 65536u;
const VERSION_BASE_STATE:  u32 = 0xFFFFFFFFu;
const MV_EMPTY:            u32 = 0xFFFFFFFFu;

// Strides in u32 words.
const TX_STRIDE_U32: u32 = 20u;       // host-padded Transaction
const MV_ENTRY_STRIDE: u32 = 24u;     // MvEntry
const TX_STATE_STRIDE: u32 = 8u;      // TxState (host-padded)
const READ_SET_STRIDE: u32 = 16u;     // ReadSetEntry
const WRITE_SET_STRIDE: u32 = 24u;    // WriteSetEntry (host-padded to 8-word boundary)
const RESULT_STRIDE: u32 = 4u;        // BlockStmResult

// Field offsets within MvEntry.
const ME_TX_INDEX:     u32 = 0u;
const ME_INCARNATION:  u32 = 1u;
const ME_ADDRESS_W0:   u32 = 2u;   // 5 words
const ME_PAD:          u32 = 7u;
const ME_SLOT_W0:      u32 = 8u;   // 8 words
const ME_VALUE_W0:     u32 = 16u;  // 8 words
const ME_IS_ESTIMATE:  u32 = 0u;   // alias into second atomic table — see below

// TxState fields.
const TS_INCARNATION: u32 = 0u;
const TS_VALIDATED:   u32 = 1u;
const TS_EXECUTED:    u32 = 2u;
const TS_STATUS:      u32 = 3u;
const TS_GAS_USED_LO: u32 = 4u;
const TS_GAS_USED_HI: u32 = 5u;
const TS_READ_COUNT:  u32 = 6u;
const TS_WRITE_COUNT: u32 = 7u;

// Transaction fields.
const TX_FROM_W0:        u32 = 0u;   // 5 words (uchar[20])
const TX_TO_W0:          u32 = 5u;   // 5 words
const TX_GAS_LIMIT_LO:   u32 = 10u;
const TX_GAS_LIMIT_HI:   u32 = 11u;
const TX_VALUE_LO:       u32 = 12u;
const TX_VALUE_HI:       u32 = 13u;
const TX_NONCE_LO:       u32 = 14u;
const TX_NONCE_HI:       u32 = 15u;
const TX_GAS_PRICE_LO:   u32 = 16u;
const TX_GAS_PRICE_HI:   u32 = 17u;
const TX_CALLDATA_OFF:   u32 = 18u;
const TX_CALLDATA_SIZE:  u32 = 19u;

// ReadSetEntry fields.
const RSE_ADDRESS_W0:    u32 = 0u;   // 5 words
const RSE_SLOT_W0:       u32 = 6u;   // 8 words (after _pad at index 5)
const RSE_READ_TX:       u32 = 14u;
const RSE_READ_INC:      u32 = 15u;

// WriteSetEntry fields.
const WSE_ADDRESS_W0:    u32 = 0u;
const WSE_SLOT_W0:       u32 = 6u;
const WSE_VALUE_W0:      u32 = 14u;

// Scheduler state indices into sched_state.
const SS_EXEC_IDX:   u32 = 0u;
const SS_VAL_IDX:    u32 = 1u;
const SS_DONE_COUNT: u32 = 2u;
const SS_ABORT:      u32 = 3u;

// Result fields.
const RES_GAS_USED_LO: u32 = 0u;
const RES_GAS_USED_HI: u32 = 1u;
const RES_STATUS:      u32 = 2u;
const RES_INCARNATION: u32 = 3u;

// =============================================================================
// Bindings. Buffer 1 (mv_memory) is split into a body buffer and a per-slot
// atomic header containing (tx_index, is_estimate) — that's the only data
// the kernel actually races on. Body fields are written-once-per-claim
// under the tx_index CAS.
// =============================================================================

@group(0) @binding(0) var<storage, read>       txs:         array<u32>;
@group(0) @binding(1) var<storage, read_write> mv_body:     array<u32>;
@group(0) @binding(2) var<storage, read_write> mv_atomic:   array<atomic<u32>>; // 2 entries per slot: tx_index, is_estimate
@group(0) @binding(3) var<storage, read_write> sched_state: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> tx_states:   array<u32>;
@group(0) @binding(5) var<storage, read_write> read_sets:   array<u32>;
@group(0) @binding(6) var<storage, read_write> write_sets:  array<u32>;
@group(0) @binding(7) var<storage, read_write> results:     array<u32>;
@group(0) @binding(8) var<uniform>             params:      Params;

struct Params {
    num_txs:        u32,
    max_iterations: u32,
    _pad0:          u32,
    _pad1:          u32,
}

// Helpers to index the split mv_atomic header. Each MvEntry has 2 atomic
// fields packed at &mv_atomic[entry_idx * 2 + 0] = tx_index, +1 = is_estimate.
fn me_tx_idx_ptr(entry_idx: u32) -> u32 { return entry_idx * 2u + 0u; }
fn me_est_ptr(entry_idx: u32)    -> u32 { return entry_idx * 2u + 1u; }

// =============================================================================
// FNV-1a hash over (tx_index || address[20] || slot[32]) = 56 bytes.
// Byte-wise FNV like the CUDA / Metal kernel. address_base / slot_base are
// indices into the corresponding u32 arrays.
// =============================================================================

fn mv_hash(tx_index: u32, address_words: ptr<function, array<u32, 5>>, slot_words: ptr<function, array<u32, 8>>) -> u32 {
    var h: u32 = 2166136261u;
    // tx_index bytes
    for (var b: u32 = 0u; b < 4u; b = b + 1u) {
        h = h ^ ((tx_index >> (b * 8u)) & 0xFFu);
        h = h * 16777619u;
    }
    // address 20 bytes
    for (var w: u32 = 0u; w < 5u; w = w + 1u) {
        let word = (*address_words)[w];
        let limit: u32 = select(4u, 20u - (w * 4u), (w * 4u) + 4u > 20u);
        for (var b: u32 = 0u; b < limit; b = b + 1u) {
            h = h ^ ((word >> (b * 8u)) & 0xFFu);
            h = h * 16777619u;
        }
    }
    // slot 32 bytes
    for (var w: u32 = 0u; w < 8u; w = w + 1u) {
        let word = (*slot_words)[w];
        for (var b: u32 = 0u; b < 4u; b = b + 1u) {
            h = h ^ ((word >> (b * 8u)) & 0xFFu);
            h = h * 16777619u;
        }
    }
    return h & MV_TABLE_MASK;
}

// Address comparison: 5 u32 words.
fn addr_eq(entry_idx: u32, addr_words: ptr<function, array<u32, 5>>) -> bool {
    let base = entry_idx * MV_ENTRY_STRIDE + ME_ADDRESS_W0;
    for (var w: u32 = 0u; w < 5u; w = w + 1u) {
        if (mv_body[base + w] != (*addr_words)[w]) { return false; }
    }
    return true;
}

fn slot_eq(entry_idx: u32, slot_words: ptr<function, array<u32, 8>>) -> bool {
    let base = entry_idx * MV_ENTRY_STRIDE + ME_SLOT_W0;
    for (var w: u32 = 0u; w < 8u; w = w + 1u) {
        if (mv_body[base + w] != (*slot_words)[w]) { return false; }
    }
    return true;
}

// =============================================================================
// MvMemory write. Open-addressing CAS-on-tx_index to claim or update.
// =============================================================================

fn mv_write(
    tx_index: u32,
    incarnation: u32,
    addr_words: ptr<function, array<u32, 5>>,
    slot_words: ptr<function, array<u32, 8>>,
    value_words: ptr<function, array<u32, 8>>,
) {
    let h = mv_hash(tx_index, addr_words, slot_words);

    for (var probe: u32 = 0u; probe < 256u; probe = probe + 1u) {
        let idx = (h + probe) & MV_TABLE_MASK;
        let entry_base = idx * MV_ENTRY_STRIDE;

        let current = atomicLoad(&mv_atomic[me_tx_idx_ptr(idx)]);
        if (current == tx_index && addr_eq(idx, addr_words) && slot_eq(idx, slot_words)) {
            // Update in place.
            mv_body[entry_base + ME_INCARNATION] = incarnation;
            for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                mv_body[entry_base + ME_VALUE_W0 + w] = (*value_words)[w];
            }
            atomicStore(&mv_atomic[me_est_ptr(idx)], 0u);
            return;
        }

        if (current == MV_EMPTY) {
            // Try to claim via CAS.
            let cas = atomicCompareExchangeWeak(&mv_atomic[me_tx_idx_ptr(idx)], MV_EMPTY, tx_index);
            if (cas.exchanged) {
                mv_body[entry_base + ME_INCARNATION] = incarnation;
                for (var w: u32 = 0u; w < 5u; w = w + 1u) {
                    mv_body[entry_base + ME_ADDRESS_W0 + w] = (*addr_words)[w];
                }
                mv_body[entry_base + ME_PAD] = 0u;
                for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                    mv_body[entry_base + ME_SLOT_W0 + w] = (*slot_words)[w];
                }
                for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                    mv_body[entry_base + ME_VALUE_W0 + w] = (*value_words)[w];
                }
                atomicStore(&mv_atomic[me_est_ptr(idx)], 0u);
                return;
            }
            // CAS failed — another thread took this slot. Continue probing.
        }
        // Otherwise: occupied by a different (tx, addr, slot); keep probing.
    }
    // Table full — should not happen with proper sizing.
}

// =============================================================================
// MvMemory read. Scan candidates [0, reader_tx_index) for the latest valid
// non-estimate version at (address, slot). Same semantics as CUDA / Metal.
// =============================================================================

struct MvReadResult {
    found:        bool,
    tx_index:     u32,
    incarnation:  u32,
    value_words:  array<u32, 8>,
}

fn mv_read(
    reader_tx_index: u32,
    addr_words: ptr<function, array<u32, 5>>,
    slot_words: ptr<function, array<u32, 8>>,
) -> MvReadResult {
    var out: MvReadResult;
    out.found = false;
    out.tx_index = VERSION_BASE_STATE;
    out.incarnation = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { out.value_words[i] = 0u; }

    var best_tx: u32 = VERSION_BASE_STATE;

    for (var candidate: u32 = 0u; candidate < reader_tx_index; candidate = candidate + 1u) {
        let h = mv_hash(candidate, addr_words, slot_words);
        for (var probe: u32 = 0u; probe < 64u; probe = probe + 1u) {
            let idx = (h + probe) & MV_TABLE_MASK;
            let etx = atomicLoad(&mv_atomic[me_tx_idx_ptr(idx)]);
            if (etx == MV_EMPTY) { break; }
            if (etx == candidate && addr_eq(idx, addr_words) && slot_eq(idx, slot_words)) {
                let est = atomicLoad(&mv_atomic[me_est_ptr(idx)]);
                if (est == 0u) {
                    // best_tx is VERSION_BASE_STATE (max-u32) on first hit,
                    // so candidate > best_tx is false when best_tx == 0xFFFFFFFF.
                    // Match CUDA semantics: track the largest committed
                    // candidate. We compare with explicit "first hit"
                    // tracking via out.found.
                    if (!out.found || candidate > best_tx) {
                        best_tx = candidate;
                        out.tx_index = candidate;
                        let entry_base = idx * MV_ENTRY_STRIDE;
                        out.incarnation = mv_body[entry_base + ME_INCARNATION];
                        for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                            out.value_words[w] = mv_body[entry_base + ME_VALUE_W0 + w];
                        }
                        out.found = true;
                    }
                }
                break;  // Found entry for this candidate, move to next.
            }
        }
    }
    return out;
}

fn mv_validate_read(
    reader_tx_index: u32,
    addr_words: ptr<function, array<u32, 5>>,
    slot_words: ptr<function, array<u32, 8>>,
    expected_tx: u32,
    expected_inc: u32,
) -> bool {
    let r = mv_read(reader_tx_index, addr_words, slot_words);
    if (!r.found) {
        return expected_tx == VERSION_BASE_STATE;
    }
    return r.tx_index == expected_tx && r.incarnation == expected_inc;
}

fn mv_mark_estimate(tx_index: u32) {
    for (var i: u32 = 0u; i < MV_TABLE_SIZE; i = i + 1u) {
        let etx = atomicLoad(&mv_atomic[me_tx_idx_ptr(i)]);
        if (etx == tx_index) {
            atomicStore(&mv_atomic[me_est_ptr(i)], 1u);
        }
    }
}

// =============================================================================
// Helpers — load address/slot/value from txs/sets into private arrays.
// =============================================================================

fn load_addr20(base_words_idx: u32, out: ptr<function, array<u32, 5>>) {
    for (var w: u32 = 0u; w < 5u; w = w + 1u) {
        (*out)[w] = txs[base_words_idx + w];
    }
}

// =============================================================================
// Main kernel — one worker per invocation.
// =============================================================================

@compute @workgroup_size(64)
fn block_stm_execute_kernel(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let num_txs = params.num_txs;
    if (num_txs == 0u) { return; }

    var loops: u32 = 0u;
    let max_loops: u32 = min(MAX_SCHEDULER_LOOPS, params.max_iterations);

    loop {
        if (loops >= max_loops) { break; }
        loops = loops + 1u;

        // Fast-path done check (read done_count).
        let done = atomicLoad(&sched_state[SS_DONE_COUNT]);
        if (done >= num_txs) {
            var all_valid: bool = true;
            for (var i: u32 = 0u; i < num_txs; i = i + 1u) {
                if (tx_states[i * TX_STATE_STRIDE + TS_VALIDATED] != 1u) {
                    all_valid = false;
                    break;
                }
            }
            if (all_valid) { break; }
        }

        if (atomicLoad(&sched_state[SS_ABORT]) != 0u) { break; }

        // === EXECUTION task ===
        let exec_idx = atomicAdd(&sched_state[SS_EXEC_IDX], 1u);
        if (exec_idx < num_txs) {
            let ts_base = exec_idx * TX_STATE_STRIDE;
            let cur_incarnation = tx_states[ts_base + TS_INCARNATION];

            if (cur_incarnation >= MAX_INCARNATIONS) {
                // Too many re-executions — mark error and skip.
                let r_base = exec_idx * RESULT_STRIDE;
                results[r_base + RES_STATUS] = 3u;
                results[r_base + RES_GAS_USED_LO] = 0u;
                results[r_base + RES_GAS_USED_HI] = 0u;
                tx_states[ts_base + TS_VALIDATED] = 1u;
                let _new_done = atomicAdd(&sched_state[SS_DONE_COUNT], 1u);
                continue;
            }

            // Execute simplified balance-transfer logic (matches Metal).
            let tx_base = exec_idx * TX_STRIDE_U32;

            // from_addr[5 u32], slot[8 u32] (slot 0 = balance), value as 32-byte BE.
            var from_addr: array<u32, 5>;
            load_addr20(tx_base + TX_FROM_W0, &from_addr);
            var to_addr: array<u32, 5>;
            load_addr20(tx_base + TX_TO_W0, &to_addr);

            var sender_slot: array<u32, 8>;
            for (var w: u32 = 0u; w < 8u; w = w + 1u) { sender_slot[w] = 0u; }

            // Encode tx.value into 32-byte BE. value is u64 (lo, hi).
            let value_lo = txs[tx_base + TX_VALUE_LO];
            let value_hi = txs[tx_base + TX_VALUE_HI];
            var value_bytes: array<u32, 8>;
            for (var w: u32 = 0u; w < 8u; w = w + 1u) { value_bytes[w] = 0u; }
            // bytes[24..31] = BE of (hi, lo). Word 6 = hi BE, word 7 = lo BE.
            // BE word 6 has hi.byte0 in high byte; the simplest correct
            // shape: byte_addr[24+i] = byte i of (hi || lo). For WGSL u32
            // packing little-endian, byte at word_idx 6 byte 0 → low byte
            // of word 6. So we put byte31 (LSB of value_lo) at word 7 low,
            // byte30 at word 7 byte 1, etc., reversing.
            // Bytes 24..27 hold value_hi big-endian, 28..31 hold value_lo BE.
            value_bytes[6] = ((value_hi & 0xFFu) << 24u)
                           | (((value_hi >> 8u) & 0xFFu) << 16u)
                           | (((value_hi >> 16u) & 0xFFu) << 8u)
                           | ((value_hi >> 24u) & 0xFFu);
            value_bytes[7] = ((value_lo & 0xFFu) << 24u)
                           | (((value_lo >> 8u) & 0xFFu) << 16u)
                           | (((value_lo >> 16u) & 0xFFu) << 8u)
                           | ((value_lo >> 24u) & 0xFFu);

            mv_write(exec_idx, cur_incarnation, &from_addr, &sender_slot, &value_bytes);

            // Record in write set.
            let wi_base = exec_idx * MAX_WRITES_PER_TX * WRITE_SET_STRIDE;
            for (var w: u32 = 0u; w < 5u; w = w + 1u) { write_sets[wi_base + WSE_ADDRESS_W0 + w] = from_addr[w]; }
            for (var w: u32 = 0u; w < 8u; w = w + 1u) { write_sets[wi_base + WSE_SLOT_W0 + w] = sender_slot[w]; }
            for (var w: u32 = 0u; w < 8u; w = w + 1u) { write_sets[wi_base + WSE_VALUE_W0 + w] = value_bytes[w]; }

            // Read sender's previous version.
            let sr = mv_read(exec_idx, &from_addr, &sender_slot);

            // Record in read set.
            let ri_base = exec_idx * MAX_READS_PER_TX * READ_SET_STRIDE;
            for (var w: u32 = 0u; w < 5u; w = w + 1u) { read_sets[ri_base + RSE_ADDRESS_W0 + w] = from_addr[w]; }
            for (var w: u32 = 0u; w < 8u; w = w + 1u) { read_sets[ri_base + RSE_SLOT_W0 + w] = sender_slot[w]; }
            read_sets[ri_base + RSE_READ_TX]  = select(VERSION_BASE_STATE, sr.tx_index,    sr.found);
            read_sets[ri_base + RSE_READ_INC] = select(0u,                  sr.incarnation, sr.found);

            // Check if to_addr is non-zero.
            var has_to: bool = false;
            for (var w: u32 = 0u; w < 5u; w = w + 1u) {
                if (to_addr[w] != 0u) { has_to = true; break; }
            }

            var wcount: u32 = 1u;
            var rcount: u32 = 1u;

            if (has_to) {
                mv_write(exec_idx, cur_incarnation, &to_addr, &sender_slot, &value_bytes);

                let wi2 = wi_base + WRITE_SET_STRIDE;
                for (var w: u32 = 0u; w < 5u; w = w + 1u) { write_sets[wi2 + WSE_ADDRESS_W0 + w] = to_addr[w]; }
                for (var w: u32 = 0u; w < 8u; w = w + 1u) { write_sets[wi2 + WSE_SLOT_W0 + w] = sender_slot[w]; }
                for (var w: u32 = 0u; w < 8u; w = w + 1u) { write_sets[wi2 + WSE_VALUE_W0 + w] = value_bytes[w]; }
                wcount = 2u;

                let rr = mv_read(exec_idx, &to_addr, &sender_slot);
                let ri2 = ri_base + READ_SET_STRIDE;
                for (var w: u32 = 0u; w < 5u; w = w + 1u) { read_sets[ri2 + RSE_ADDRESS_W0 + w] = to_addr[w]; }
                for (var w: u32 = 0u; w < 8u; w = w + 1u) { read_sets[ri2 + RSE_SLOT_W0 + w] = sender_slot[w]; }
                read_sets[ri2 + RSE_READ_TX]  = select(VERSION_BASE_STATE, rr.tx_index,    rr.found);
                read_sets[ri2 + RSE_READ_INC] = select(0u,                  rr.incarnation, rr.found);
                rcount = 2u;
            }

            // Intrinsic gas: 21000 + 16 per calldata byte (simplified).
            let calldata_size = txs[tx_base + TX_CALLDATA_SIZE];
            let intrinsic_gas: u32 = 21000u;
            let calldata_gas: u32  = calldata_size * 16u;
            let total_gas: u32     = intrinsic_gas + calldata_gas;

            let gas_limit_lo = txs[tx_base + TX_GAS_LIMIT_LO];
            // gas_limit_hi rarely set for realistic txs; if so, total_gas
            // certainly fits → ok.
            let gas_limit_hi = txs[tx_base + TX_GAS_LIMIT_HI];
            let oog: bool = (gas_limit_hi == 0u) && (total_gas > gas_limit_lo);

            tx_states[ts_base + TS_GAS_USED_LO] = total_gas;
            tx_states[ts_base + TS_GAS_USED_HI] = 0u;
            tx_states[ts_base + TS_STATUS]      = select(0u, 2u, oog);
            tx_states[ts_base + TS_READ_COUNT]  = rcount;
            tx_states[ts_base + TS_WRITE_COUNT] = wcount;
            tx_states[ts_base + TS_EXECUTED]    = 1u;
            tx_states[ts_base + TS_VALIDATED]   = 0u;

            let r_base = exec_idx * RESULT_STRIDE;
            results[r_base + RES_GAS_USED_LO] = total_gas;
            results[r_base + RES_GAS_USED_HI] = 0u;
            results[r_base + RES_STATUS]      = tx_states[ts_base + TS_STATUS];
            results[r_base + RES_INCARNATION] = cur_incarnation;
            continue;
        }

        // === VALIDATION task ===
        let val_idx = atomicAdd(&sched_state[SS_VAL_IDX], 1u);
        if (val_idx < num_txs) {
            let ts_base = val_idx * TX_STATE_STRIDE;
            if (tx_states[ts_base + TS_EXECUTED] == 0u) {
                // Not yet executed — put it back via CAS and try again.
                let _ignored = atomicCompareExchangeWeak(
                    &sched_state[SS_VAL_IDX], val_idx + 1u, val_idx);
                continue;
            }

            let rcount = tx_states[ts_base + TS_READ_COUNT];
            let ri_base0 = val_idx * MAX_READS_PER_TX * READ_SET_STRIDE;
            var valid: bool = true;
            for (var r: u32 = 0u; r < rcount; r = r + 1u) {
                if (r >= MAX_READS_PER_TX) { break; }
                let r_base = ri_base0 + r * READ_SET_STRIDE;
                var addr: array<u32, 5>;
                for (var w: u32 = 0u; w < 5u; w = w + 1u) { addr[w] = read_sets[r_base + RSE_ADDRESS_W0 + w]; }
                var slot: array<u32, 8>;
                for (var w: u32 = 0u; w < 8u; w = w + 1u) { slot[w] = read_sets[r_base + RSE_SLOT_W0 + w]; }
                let expected_tx  = read_sets[r_base + RSE_READ_TX];
                let expected_inc = read_sets[r_base + RSE_READ_INC];
                if (!mv_validate_read(val_idx, &addr, &slot, expected_tx, expected_inc)) {
                    valid = false;
                    break;
                }
            }

            if (valid) {
                tx_states[ts_base + TS_VALIDATED] = 1u;
                let _d = atomicAdd(&sched_state[SS_DONE_COUNT], 1u);
            } else {
                // Conflict — mark estimates, bump incarnation, reset sched.
                mv_mark_estimate(val_idx);

                tx_states[ts_base + TS_INCARNATION] = tx_states[ts_base + TS_INCARNATION] + 1u;
                tx_states[ts_base + TS_EXECUTED]    = 0u;
                tx_states[ts_base + TS_VALIDATED]   = 0u;

                // Reset execution_idx to ≤ val_idx so the next claimer
                // re-executes this tx.
                loop {
                    let cur = atomicLoad(&sched_state[SS_EXEC_IDX]);
                    if (cur <= val_idx) { break; }
                    let cas = atomicCompareExchangeWeak(&sched_state[SS_EXEC_IDX], cur, val_idx);
                    if (cas.exchanged) { break; }
                }

                // Invalidate later validated txs.
                for (var i: u32 = val_idx + 1u; i < num_txs; i = i + 1u) {
                    if (tx_states[i * TX_STATE_STRIDE + TS_VALIDATED] == 1u) {
                        tx_states[i * TX_STATE_STRIDE + TS_VALIDATED] = 0u;
                        let _s = atomicSub(&sched_state[SS_DONE_COUNT], 1u);
                    }
                }

                // Reset validation_idx.
                loop {
                    let cur = atomicLoad(&sched_state[SS_VAL_IDX]);
                    if (cur <= val_idx) { break; }
                    let cas = atomicCompareExchangeWeak(&sched_state[SS_VAL_IDX], cur, val_idx);
                    if (cas.exchanged) { break; }
                }
            }
            continue;
        }

        // No work available. Re-check done.
        let done2 = atomicLoad(&sched_state[SS_DONE_COUNT]);
        if (done2 >= num_txs) {
            var all_valid: bool = true;
            for (var i: u32 = 0u; i < num_txs; i = i + 1u) {
                if (tx_states[i * TX_STATE_STRIDE + TS_VALIDATED] != 1u) {
                    all_valid = false;
                    break;
                }
            }
            if (all_valid) { break; }
        }
    }
}
