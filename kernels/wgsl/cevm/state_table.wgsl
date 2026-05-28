// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// WGSL port of cevm/lib/evm/gpu/cuda/state_table.cu — lookup + insert paths.
//
// GPU-resident open-addressing hash table for Ethereum state. Two parallel
// flavours:
//   * Account table: keyed by 20-byte address, value is DeviceAccountData
//     (nonce u64 + balance uint256 + code_hash[32] + storage_root[32]).
//   * Storage table: keyed by 20-byte addr + 32-byte slot, value is 32 bytes.
//
// The state-root reduction pipeline (compact, sort, hash, reduce) lives in
// the CUDA/Metal sources but is excluded from this initial WGSL port: it
// needs RLP encoding and full keccak invocation inside the kernel, which
// pulls the keccak module dependency. State-root computation is currently
// driven from the host via a separate keccak_batch dispatch.
//
// Layout invariants (must match CUDA + Metal sources byte-for-byte):
//   DeviceAccountEntry  = 20-byte key + key_valid u32 + _pad u32 + AccountData
//                       = 20 + 4 + 4 + 8 + 32 + 32 + 32 = 132 bytes
//   DeviceStorageEntry  = 20-byte addr + 32-byte slot + key_valid u32 +
//                         _pad u32 + 32-byte value = 92 bytes
//
// As with tx_validate.wgsl, multi-byte fields are packed into u32 words and
// 64-bit values are stored as vec2<u32>(lo, hi).

// =============================================================================
// Stride constants (in u32 words). MUST match the binary layout written by
// the host side. CUDA uses these implicitly via cudaMalloc + memcpy.
// =============================================================================

// 20-byte key (5 words) + key_valid (1) + _pad (1) + AccountData (
//   nonce 2 words + balance 8 words + code_hash 8 words + storage_root 8 words
// ) = 33 words.
const ACCOUNT_ENTRY_STRIDE_U32: u32 = 33u;

// 20-byte addr (5 words) + 32-byte slot (8 words) + key_valid (1) + _pad (1) +
// value 32 bytes (8 words) = 23 words.
const STORAGE_ENTRY_STRIDE_U32: u32 = 23u;

// AccountData stride: nonce(2) + balance(8) + code_hash(8) + storage_root(8) = 26 words.
const ACCOUNT_DATA_STRIDE_U32: u32 = 26u;

// Field offsets inside an AccountEntry.
const AE_KEY_W0:      u32 = 0u;  // 5 words
const AE_KEY_VALID:   u32 = 5u;
const AE_PAD:         u32 = 6u;
const AE_DATA_OFFSET: u32 = 7u;  // 26 words of AccountData starts here

// Field offsets inside a StorageEntry.
const SE_ADDR_W0:    u32 = 0u;   // 5 words
const SE_SLOT_W0:    u32 = 5u;   // 8 words
const SE_KEY_VALID:  u32 = 13u;
const SE_PAD:        u32 = 14u;
const SE_VALUE_W0:   u32 = 15u;  // 8 words

// =============================================================================
// Uniforms — batch parameters.
// =============================================================================

struct BatchParams {
    count:    u32,
    capacity: u32,  // must be power of 2
}

// =============================================================================
// Bindings — account lookup kernel
// =============================================================================

@group(0) @binding(0) var<storage, read>       acct_keys:     array<u32>;  // count * 5 words
@group(0) @binding(1) var<storage, read>       acct_table:    array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> acct_results:  array<u32>;  // count * ACCOUNT_DATA_STRIDE_U32
@group(0) @binding(3) var<storage, read_write> acct_found:    array<u32>;
@group(0) @binding(4) var<uniform>             acct_params:   BatchParams;

// =============================================================================
// FNV-1a over 20-byte address, returned modulo `capacity`.
// =============================================================================

fn hash_address_words(base: u32, capacity: u32) -> u32 {
    var h: u32 = 0x811c9dc5u;
    for (var w: u32 = 0u; w < 5u; w = w + 1u) {
        let word = acct_keys[base + w];
        let limit: u32 = select(4u, 20u - (w * 4u), (w * 4u) + 4u > 20u);
        for (var b: u32 = 0u; b < limit; b = b + 1u) {
            let byte = (word >> (b * 8u)) & 0xFFu;
            h = h ^ byte;
            h = h * 0x01000193u;
        }
    }
    return h & (capacity - 1u);
}

// =============================================================================
// Address equality between a key in `acct_keys` and an in-table slot. The
// 5-word key in the slot is stored under the same packed layout. We compare
// word-by-word; the 5th word's upper 4 bytes are always zero (20-byte key
// fits in 5 u32 with 0 pad), so a word compare is safe.
// =============================================================================

fn key_eq_acct(key_base: u32, slot_base: u32) -> bool {
    // slot_base indexes acct_table as atomic<u32>. We must load with atomic
    // loads because the buffer is bound as atomic<u32>. Use atomicLoad.
    return atomicLoad(&acct_table[slot_base + 0u]) == acct_keys[key_base + 0u] &&
           atomicLoad(&acct_table[slot_base + 1u]) == acct_keys[key_base + 1u] &&
           atomicLoad(&acct_table[slot_base + 2u]) == acct_keys[key_base + 2u] &&
           atomicLoad(&acct_table[slot_base + 3u]) == acct_keys[key_base + 3u] &&
           atomicLoad(&acct_table[slot_base + 4u]) == acct_keys[key_base + 4u];
}

// =============================================================================
// account_lookup_batch — one thread per requested key.
// =============================================================================

@compute @workgroup_size(64)
fn account_lookup_batch(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let tid = gid.x;
    if (tid >= acct_params.count) { return; }

    let capacity = acct_params.capacity;
    let mask     = capacity - 1u;

    let key_base = tid * 5u;
    let h        = hash_address_words(key_base, capacity);

    for (var probe: u32 = 0u; probe < capacity; probe = probe + 1u) {
        let slot_idx  = (h + probe) & mask;
        let slot_base = slot_idx * ACCOUNT_ENTRY_STRIDE_U32;
        let occupied  = atomicLoad(&acct_table[slot_base + AE_KEY_VALID]);
        if (occupied == 0u) {
            acct_found[tid] = 0u;
            return;
        }
        if (key_eq_acct(key_base, slot_base)) {
            // Copy AccountData into results.
            let dst = tid * ACCOUNT_DATA_STRIDE_U32;
            for (var w: u32 = 0u; w < ACCOUNT_DATA_STRIDE_U32; w = w + 1u) {
                acct_results[dst + w] = atomicLoad(&acct_table[slot_base + AE_DATA_OFFSET + w]);
            }
            acct_found[tid] = 1u;
            return;
        }
    }
    acct_found[tid] = 0u;
}

// =============================================================================
// account_insert_batch — one thread per key/data pair. Inserts use atomicCAS
// on key_valid (0 → 1) to claim an empty slot. If the slot was already
// occupied with our key, update in place. Otherwise probe further.
// =============================================================================

@group(1) @binding(0) var<storage, read>       ins_keys:    array<u32>;  // count * 5 words
@group(1) @binding(1) var<storage, read>       ins_data:    array<u32>;  // count * ACCOUNT_DATA_STRIDE_U32
@group(1) @binding(2) var<storage, read_write> ins_table:   array<atomic<u32>>;
@group(1) @binding(3) var<uniform>             ins_params:  BatchParams;

fn hash_address_ins(base: u32, capacity: u32) -> u32 {
    var h: u32 = 0x811c9dc5u;
    for (var w: u32 = 0u; w < 5u; w = w + 1u) {
        let word = ins_keys[base + w];
        let limit: u32 = select(4u, 20u - (w * 4u), (w * 4u) + 4u > 20u);
        for (var b: u32 = 0u; b < limit; b = b + 1u) {
            let byte = (word >> (b * 8u)) & 0xFFu;
            h = h ^ byte;
            h = h * 0x01000193u;
        }
    }
    return h & (capacity - 1u);
}

fn key_eq_ins(key_base: u32, slot_base: u32) -> bool {
    return atomicLoad(&ins_table[slot_base + 0u]) == ins_keys[key_base + 0u] &&
           atomicLoad(&ins_table[slot_base + 1u]) == ins_keys[key_base + 1u] &&
           atomicLoad(&ins_table[slot_base + 2u]) == ins_keys[key_base + 2u] &&
           atomicLoad(&ins_table[slot_base + 3u]) == ins_keys[key_base + 3u] &&
           atomicLoad(&ins_table[slot_base + 4u]) == ins_keys[key_base + 4u];
}

@compute @workgroup_size(64)
fn account_insert_batch(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let tid = gid.x;
    if (tid >= ins_params.count) { return; }

    let capacity = ins_params.capacity;
    let mask     = capacity - 1u;

    let key_base = tid * 5u;
    let data_base = tid * ACCOUNT_DATA_STRIDE_U32;
    let h         = hash_address_ins(key_base, capacity);

    for (var probe: u32 = 0u; probe < capacity; probe = probe + 1u) {
        let slot_idx  = (h + probe) & mask;
        let slot_base = slot_idx * ACCOUNT_ENTRY_STRIDE_U32;

        // Try to claim empty slot via CAS on key_valid: 0 → 1.
        let cas = atomicCompareExchangeWeak(
            &ins_table[slot_base + AE_KEY_VALID], 0u, 1u);
        if (cas.exchanged) {
            // Claimed. Write key + data.
            for (var w: u32 = 0u; w < 5u; w = w + 1u) {
                atomicStore(&ins_table[slot_base + w], ins_keys[key_base + w]);
            }
            // Pad.
            atomicStore(&ins_table[slot_base + AE_PAD], 0u);
            // Data.
            for (var w: u32 = 0u; w < ACCOUNT_DATA_STRIDE_U32; w = w + 1u) {
                atomicStore(&ins_table[slot_base + AE_DATA_OFFSET + w],
                            ins_data[data_base + w]);
            }
            return;
        }

        // Slot occupied. If it's our key, update in place.
        if (key_eq_ins(key_base, slot_base)) {
            for (var w: u32 = 0u; w < ACCOUNT_DATA_STRIDE_U32; w = w + 1u) {
                atomicStore(&ins_table[slot_base + AE_DATA_OFFSET + w],
                            ins_data[data_base + w]);
            }
            return;
        }
        // Otherwise: probe further.
    }
}

// =============================================================================
// Storage lookup/insert kernels — same pattern as account, with 32-byte slot
// disambiguator added to the key. Bound on a separate @group to keep the
// account kernel pipeline layout independent.
// =============================================================================

struct StorageKey {
    addr:  array<u32, 5>,
    slot:  array<u32, 8>,
}

@group(2) @binding(0) var<storage, read>       st_keys:    array<u32>;  // count * 13 words (5 + 8)
@group(2) @binding(1) var<storage, read>       st_values:  array<u32>;  // count * 8 words
@group(2) @binding(2) var<storage, read_write> st_table:   array<atomic<u32>>;
@group(2) @binding(3) var<storage, read_write> st_results: array<u32>;
@group(2) @binding(4) var<storage, read_write> st_found:   array<u32>;
@group(2) @binding(5) var<uniform>             st_params:  BatchParams;

const STORAGE_KEY_WORDS: u32 = 13u;  // 5 addr + 8 slot
const STORAGE_VAL_WORDS: u32 = 8u;

fn hash_storage_key(key_base: u32, capacity: u32) -> u32 {
    var h: u32 = 0x811c9dc5u;
    // addr (20 bytes, 5 words)
    for (var w: u32 = 0u; w < 5u; w = w + 1u) {
        let word = st_keys[key_base + w];
        let limit: u32 = select(4u, 20u - (w * 4u), (w * 4u) + 4u > 20u);
        for (var b: u32 = 0u; b < limit; b = b + 1u) {
            let byte = (word >> (b * 8u)) & 0xFFu;
            h = h ^ byte;
            h = h * 0x01000193u;
        }
    }
    // slot (32 bytes, 8 words)
    for (var w: u32 = 0u; w < 8u; w = w + 1u) {
        let word = st_keys[key_base + 5u + w];
        for (var b: u32 = 0u; b < 4u; b = b + 1u) {
            let byte = (word >> (b * 8u)) & 0xFFu;
            h = h ^ byte;
            h = h * 0x01000193u;
        }
    }
    return h & (capacity - 1u);
}

fn storage_key_eq(key_base: u32, slot_base: u32) -> bool {
    // addr
    for (var w: u32 = 0u; w < 5u; w = w + 1u) {
        if (atomicLoad(&st_table[slot_base + SE_ADDR_W0 + w]) != st_keys[key_base + w]) {
            return false;
        }
    }
    // slot
    for (var w: u32 = 0u; w < 8u; w = w + 1u) {
        if (atomicLoad(&st_table[slot_base + SE_SLOT_W0 + w]) != st_keys[key_base + 5u + w]) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn storage_lookup_batch(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let tid = gid.x;
    if (tid >= st_params.count) { return; }

    let capacity = st_params.capacity;
    let mask     = capacity - 1u;

    let key_base = tid * STORAGE_KEY_WORDS;
    let h        = hash_storage_key(key_base, capacity);

    for (var probe: u32 = 0u; probe < capacity; probe = probe + 1u) {
        let slot_idx  = (h + probe) & mask;
        let slot_base = slot_idx * STORAGE_ENTRY_STRIDE_U32;
        let occupied  = atomicLoad(&st_table[slot_base + SE_KEY_VALID]);
        if (occupied == 0u) {
            st_found[tid] = 0u;
            return;
        }
        if (storage_key_eq(key_base, slot_base)) {
            let dst = tid * STORAGE_VAL_WORDS;
            for (var w: u32 = 0u; w < STORAGE_VAL_WORDS; w = w + 1u) {
                st_results[dst + w] = atomicLoad(&st_table[slot_base + SE_VALUE_W0 + w]);
            }
            st_found[tid] = 1u;
            return;
        }
    }
    st_found[tid] = 0u;
}

@compute @workgroup_size(64)
fn storage_insert_batch(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let tid = gid.x;
    if (tid >= st_params.count) { return; }

    let capacity = st_params.capacity;
    let mask     = capacity - 1u;

    let key_base = tid * STORAGE_KEY_WORDS;
    let val_base = tid * STORAGE_VAL_WORDS;
    let h        = hash_storage_key(key_base, capacity);

    for (var probe: u32 = 0u; probe < capacity; probe = probe + 1u) {
        let slot_idx  = (h + probe) & mask;
        let slot_base = slot_idx * STORAGE_ENTRY_STRIDE_U32;

        let cas = atomicCompareExchangeWeak(
            &st_table[slot_base + SE_KEY_VALID], 0u, 1u);
        if (cas.exchanged) {
            // addr
            for (var w: u32 = 0u; w < 5u; w = w + 1u) {
                atomicStore(&st_table[slot_base + SE_ADDR_W0 + w], st_keys[key_base + w]);
            }
            // slot
            for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                atomicStore(&st_table[slot_base + SE_SLOT_W0 + w], st_keys[key_base + 5u + w]);
            }
            atomicStore(&st_table[slot_base + SE_PAD], 0u);
            // value
            for (var w: u32 = 0u; w < STORAGE_VAL_WORDS; w = w + 1u) {
                atomicStore(&st_table[slot_base + SE_VALUE_W0 + w], st_values[val_base + w]);
            }
            return;
        }

        if (storage_key_eq(key_base, slot_base)) {
            for (var w: u32 = 0u; w < STORAGE_VAL_WORDS; w = w + 1u) {
                atomicStore(&st_table[slot_base + SE_VALUE_W0 + w], st_values[val_base + w]);
            }
            return;
        }
    }
}
