// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_membership.wgsl — Bloom + cuckoo membership maintenance.
//
// Single-thread kernel: clears the cuckoo arena, then re-seeds Bloom +
// cuckoo from the live UTXO arena. Bloom is monotonic — never cleared.
// Mirrors xvm_membership.metal byte-for-byte.

// Concatenated after xvm_kernels_common.wgsl.

struct MembershipParams {
    utxo_count: u32,
    bloom_bit_count: u32,
    cuckoo_bucket_count: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read_write> mb_utxos: array<UTXO>;
@group(0) @binding(1) var<storage, read_write> mb_bloom: array<u32>;
@group(0) @binding(2) var<storage, read_write> mb_cuckoo: array<CuckooEntry>;
@group(0) @binding(3) var<uniform>             mb_params: MembershipParams;

fn mb_hash_quad(uid: ptr<function, array<u32, 8>>, out_lo: ptr<function, array<u32, 4>>) {
    var digest: array<u32, 8>;
    keccak256_32(uid, &digest);
    // The CPU reference reads each of the 4 u64 hashes as little-endian
    // 8-byte groups, but `bit % bit_count` only ever uses the low 32 of
    // each u64 since bit_count <= 2^31. So we propagate just the lo halves.
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        (*out_lo)[i] = digest[2u*i];
    }
}

fn mb_bloom_set(uid: ptr<function, array<u32, 8>>, bit_count: u32) {
    if (bit_count == 0u) { return; }
    var hashes_lo: array<u32, 4>;
    mb_hash_quad(uid, &hashes_lo);
    let mask = bit_count - 1u;  // requires bit_count to be a power of two
    for (var i: u32 = 0u; i < kBloomHashes; i = i + 1u) {
        let bit = hashes_lo[i] & mask;
        let word_off = bit >> 5u;
        let bit_off = bit & 31u;
        mb_bloom[word_off] = mb_bloom[word_off] | (1u << bit_off);
    }
}

fn mb_bloom_test(uid: ptr<function, array<u32, 8>>, bit_count: u32) -> bool {
    if (bit_count == 0u) { return true; }
    var hashes_lo: array<u32, 4>;
    mb_hash_quad(uid, &hashes_lo);
    let mask = bit_count - 1u;
    for (var i: u32 = 0u; i < kBloomHashes; i = i + 1u) {
        let bit = hashes_lo[i] & mask;
        let word_off = bit >> 5u;
        let bit_off = bit & 31u;
        if ((mb_bloom[word_off] & (1u << bit_off)) == 0u) { return false; }
    }
    return true;
}

fn mb_cuckoo_bucket(h_lo: u32, bucket_count: u32) -> u32 {
    return h_lo & (bucket_count - 1u);
}

// Insert (or update) into cuckoo arena. Returns true on success.
fn mb_cuckoo_insert(uid: ptr<function, array<u32, 8>>, slot_index: u32, bucket_count: u32) -> bool {
    var hashes_lo: array<u32, 4>;
    mb_hash_quad(uid, &hashes_lo);
    // Update if utxo_id already present in either bucket.
    for (var k: u32 = 0u; k < 2u; k = k + 1u) {
        let b = mb_cuckoo_bucket(hashes_lo[k], bucket_count);
        for (var s: u32 = 0u; s < kCuckooSlotsPerBucket; s = s + 1u) {
            let idx = b * kCuckooSlotsPerBucket + s;
            if (mb_cuckoo[idx].occupied != 0u) {
                var entry_uid: array<u32, 8>;
                for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                    entry_uid[i] = mb_cuckoo[idx].utxo_id[i];
                }
                if (digest_eq8(&entry_uid, uid)) {
                    mb_cuckoo[idx].slot_index = slot_index;
                    return true;
                }
            }
        }
    }
    // Insert into first free slot in either of the two buckets.
    for (var k: u32 = 0u; k < 2u; k = k + 1u) {
        let b = mb_cuckoo_bucket(hashes_lo[k], bucket_count);
        for (var s: u32 = 0u; s < kCuckooSlotsPerBucket; s = s + 1u) {
            let idx = b * kCuckooSlotsPerBucket + s;
            if (mb_cuckoo[idx].occupied == 0u) {
                for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                    mb_cuckoo[idx].utxo_id[i] = (*uid)[i];
                }
                mb_cuckoo[idx].slot_index = slot_index;
                mb_cuckoo[idx].occupied = 1u;
                mb_cuckoo[idx].pad0_lo = 0u;
                mb_cuckoo[idx].pad0_hi = 0u;
                return true;
            }
        }
    }
    return false;
}

@compute @workgroup_size(1, 1, 1)
fn xvm_membership_rebuild(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }

    let total_slots = mb_params.cuckoo_bucket_count * kCuckooSlotsPerBucket;
    for (var i: u32 = 0u; i < total_slots; i = i + 1u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            mb_cuckoo[i].utxo_id[k] = 0u;
        }
        mb_cuckoo[i].slot_index = 0u;
        mb_cuckoo[i].occupied = 0u;
        mb_cuckoo[i].pad0_lo = 0u;
        mb_cuckoo[i].pad0_hi = 0u;
    }
    for (var i: u32 = 0u; i < mb_params.utxo_count; i = i + 1u) {
        let st = mb_utxos[i].status;
        if ((st & kUtxoOccupied) == 0u) { continue; }
        if ((st & kUtxoSpent) != 0u) { continue; }
        var uid: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            uid[k] = mb_utxos[i].utxo_id[k];
        }
        mb_bloom_set(&uid, mb_params.bloom_bit_count);
        _ = mb_cuckoo_insert(&uid, i, mb_params.cuckoo_bucket_count);
    }
}
