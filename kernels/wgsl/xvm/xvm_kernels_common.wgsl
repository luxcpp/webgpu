// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_kernels_common.wgsl — shared device code for the WGSL XVM kernels.
//
// Layout MUST match xvm_gpu_layout.hpp byte-for-byte. WGSL has no native
// u64 type. We map each 64-bit field to two u32 lanes (lo, hi) with
// little-endian byte order — since both Metal and CPU layouts are
// little-endian, this gives us a bytewise-identical struct layout when
// the host uploads the buffers via plain memcpy of the C++ structs.
//
// 32-byte digests map to array<u32, 8>. Keccak operates on a 25-lane u64
// state realised as array<u32, 50>. Hash leaves are u32-packed byte streams
// where each u32 holds four LE bytes — the absorb path mirrors the CPU
// reference exactly because byte i of the input lands in bit (i%8)*8 of
// lane (i/8) under both encodings.

// =============================================================================
// Layout structs — bytewise identical to xvm_gpu_layout.hpp on LE hosts.
// Each `array<u32, 8>` is the WGSL representation of a uint8_t[32] field.
// Each pair `(lo: u32, hi: u32)` is a uint64_t.
// =============================================================================

struct UTXO {
    utxo_id: array<u32, 8>,                       // 0  : 32 bytes
    asset_id: array<u32, 8>,                      // 32 : 32 bytes
    amount_lo_lo: u32, amount_lo_hi: u32,         // 64 : u64 amount_lo
    amount_hi_lo: u32, amount_hi_hi: u32,         // 72 : u64 amount_hi
    owner_root: array<u32, 8>,                    // 80 : 32 bytes
    locktime_lo: u32, locktime_hi: u32,           // 112: u64
    threshold: u32,                               // 120
    status: u32,                                  // 124
    addresses_offset: u32,                        // 128
    addresses_count: u32,                         // 132
    pad0_lo: u32, pad0_hi: u32,                   // 136 -> 144
};

struct InputBatch {
    tx_id: array<u32, 8>,                         // 0  : 32 bytes
    input_offset: u32,                            // 32
    input_count: u32,                             // 36
    witness_offset: u32,                          // 40
    witness_count: u32,                           // 44
    pad0_lo: u32, pad0_hi: u32,                   // 48 -> 56 (struct is 64 bytes)
    pad1_lo: u32, pad1_hi: u32,                   // 56 -> 64
};

struct OutputBatch {
    tx_id: array<u32, 8>,                         // 0  : 32 bytes
    output_offset: u32,                           // 32
    output_count: u32,                            // 36
    pad0_lo: u32, pad0_hi: u32,                   // 40
    pad1_lo: u32, pad1_hi: u32,                   // 48
    pad2_lo: u32, pad2_hi: u32,                   // 56 -> 64
};

struct Asset {
    asset_id: array<u32, 8>,                                // 0
    total_supply_lo_lo: u32, total_supply_lo_hi: u32,       // 32
    total_supply_hi_lo: u32, total_supply_hi_hi: u32,       // 40
    mint_authority: array<u32, 8>,                          // 48
    freeze_flag: u32,                                       // 80
    denomination: u32,                                      // 84
    name_offset: u32,                                       // 88
    name_length: u32,                                       // 92
    occupied: u32,                                          // 96
    pad0: u32,                                              // 100
    pad1_lo: u32, pad1_hi: u32,                             // 104 -> 112
};

struct CuckooEntry {
    utxo_id: array<u32, 8>,
    slot_index: u32,
    occupied: u32,
    pad0_lo: u32, pad0_hi: u32,
};

struct AtomicExportMarker {
    marker_id: array<u32, 8>,                               // 0
    asset_id: array<u32, 8>,                                // 32
    amount_lo_lo: u32, amount_lo_hi: u32,                   // 64
    amount_hi_lo: u32, amount_hi_hi: u32,                   // 72
    source_chain: u32,                                      // 80
    target_chain: u32,                                      // 84
    status: u32,                                            // 88
    occupied: u32,                                          // 92
    recipient_root: array<u32, 8>,                          // 96
    pad0_lo: u32, pad0_hi: u32,                             // 128
    pad1_lo: u32, pad1_hi: u32,                             // 136 -> 144
};

struct XvmTx {
    tx_id: array<u32, 8>,                                   // 0
    kind: u32,                                              // 32
    input_batch_offset: u32,                                // 36
    output_batch_offset: u32,                               // 40
    asset_changes_offset: u32,                              // 44
    asset_changes_count: u32,                               // 48
    target_chain: u32,                                      // 52
    status: u32,                                            // 56
    reject_reason: u32,                                     // 60
    proof_digest: array<u32, 8>,                            // 64
    pad0_lo: u32, pad0_hi: u32,                             // 96
    pad1_lo: u32, pad1_hi: u32,                             // 104 -> 112
};

struct AssetOp {
    asset_id: array<u32, 8>,                                // 0
    amount_lo_lo: u32, amount_lo_hi: u32,                   // 32
    amount_hi_lo: u32, amount_hi_hi: u32,                   // 40
    authority_witness: array<u32, 8>,                       // 48
    kind: u32,                                              // 80
    target_chain: u32,                                      // 84
    pad0_lo: u32, pad0_hi: u32,                             // 88
    pad1_lo: u32, pad1_hi: u32,                             // 96
    pad2_lo: u32, pad2_hi: u32,                             // 104 -> 112
};

struct XVMRoundDescriptor {
    chain_id_lo: u32, chain_id_hi: u32,
    round_lo: u32, round_hi: u32,
    timestamp_ns_lo: u32, timestamp_ns_hi: u32,
    height_lo: u32, height_hi: u32,
    mode: u32,
    tx_count: u32,
    input_count: u32,
    output_count: u32,
    asset_op_count: u32,
    input_batch_count: u32,
    output_batch_count: u32,
    closing_flag: u32,
    parent_execution_root: array<u32, 8>,
    pad0_lo: u32, pad0_hi: u32,
    pad1_lo: u32, pad1_hi: u32,
};

struct XVMTransitionResult {
    status: u32,
    tx_accepted: u32,
    tx_rejected: u32,
    inputs_consumed: u32,
    outputs_created: u32,
    asset_ops_applied: u32,
    export_markers: u32,
    import_verified: u32,
    total_burned_lo_lo: u32, total_burned_lo_hi: u32,
    total_burned_hi_lo: u32, total_burned_hi_hi: u32,
    total_minted_lo_lo: u32, total_minted_lo_hi: u32,
    total_minted_hi_lo: u32, total_minted_hi_hi: u32,
    height_lo: u32, height_hi: u32,
    pad0_lo: u32, pad0_hi: u32,
    utxo_root: array<u32, 8>,
    asset_root: array<u32, 8>,
    tx_root: array<u32, 8>,
    execution_root: array<u32, 8>,
};

// =============================================================================
// Constants — must match xvm_gpu_layout.hpp values.
// =============================================================================

const kUtxoOccupied: u32 = 0x1u;
const kUtxoSpent: u32 = 0x2u;

const kAssetActive: u32 = 0x1u;
const kAssetFrozen: u32 = 0x2u;

const kExportPending: u32 = 0u;
const kExportConsumed: u32 = 1u;

const kTxStatusPending: u32 = 0u;
const kTxStatusAccepted: u32 = 1u;
const kTxStatusRejected: u32 = 2u;

const kRejectMissingInput: u32 = 1u;
const kRejectDuplicateInput: u32 = 2u;
const kRejectAlreadySpent: u32 = 3u;
const kRejectLocktime: u32 = 4u;
const kRejectAuth: u32 = 5u;
const kRejectMintAuthority: u32 = 6u;
const kRejectAssetMissing: u32 = 7u;
const kRejectImportNoMarker: u32 = 8u;
const kRejectArenaFull: u32 = 9u;
const kRejectAmountOverflow: u32 = 10u;

const kAssetOpMint: u32 = 0u;
const kAssetOpBurn: u32 = 1u;
const kAssetOpTransfer: u32 = 2u;
const kAssetOpExport: u32 = 3u;
const kAssetOpImport: u32 = 4u;

const kBloomHashes: u32 = 4u;
const kCuckooSlotsPerBucket: u32 = 4u;

// =============================================================================
// keccak256 — Keccak-f[1600] / 0x01 / 0x80 padding, byte-identical to CPU.
// State is array<u32, 50>: lane i is (s[2*i], s[2*i+1]) representing u64.
// =============================================================================

const kKeccakRC_lo: array<u32, 24> = array<u32, 24>(
    0x00000001u, 0x00008082u, 0x0000808Au, 0x80008000u,
    0x0000808Bu, 0x80000001u, 0x80008081u, 0x00008009u,
    0x0000008Au, 0x00000088u, 0x80008009u, 0x8000000Au,
    0x8000808Bu, 0x0000008Bu, 0x00008089u, 0x00008003u,
    0x00008002u, 0x00000080u, 0x0000800Au, 0x8000000Au,
    0x80008081u, 0x00008080u, 0x80000001u, 0x80008008u,
);
const kKeccakRC_hi: array<u32, 24> = array<u32, 24>(
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x80000000u, 0x80000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u,
);

const kKeccakRot: array<u32, 25> = array<u32, 25>(
     0u,  1u, 62u, 28u, 27u,
    36u, 44u,  6u, 55u, 20u,
     3u, 10u, 43u, 25u, 39u,
    41u, 45u, 15u, 21u,  8u,
    18u,  2u, 61u, 56u, 14u,
);

// rotl over (lo, hi) by n in [0..63]. Branches because WGSL u32 shifts by 32
// are UB; kKeccakRot[0] == 0 forces the n==0 branch every keccak round.
fn rotl64_split(lo_in: u32, hi_in: u32, n_in: u32) -> vec2<u32> {
    let n = n_in & 63u;
    var lo: u32; var hi: u32;
    if (n == 0u) {
        lo = lo_in; hi = hi_in;
    } else if (n < 32u) {
        let m = 32u - n;
        lo = (lo_in << n) | (hi_in >> m);
        hi = (hi_in << n) | (lo_in >> m);
    } else if (n == 32u) {
        lo = hi_in; hi = lo_in;
    } else {
        let k = n - 32u;
        let m = 32u - k;
        lo = (hi_in << k) | (lo_in >> m);
        hi = (lo_in << k) | (hi_in >> m);
    }
    return vec2<u32>(lo, hi);
}

fn keccak_f1600(s: ptr<function, array<u32, 50>>) {
    for (var round: u32 = 0u; round < 24u; round = round + 1u) {
        var c_lo: array<u32, 5>;
        var c_hi: array<u32, 5>;
        for (var x: u32 = 0u; x < 5u; x = x + 1u) {
            c_lo[x] = (*s)[2u*x] ^ (*s)[2u*(x+5u)] ^ (*s)[2u*(x+10u)] ^ (*s)[2u*(x+15u)] ^ (*s)[2u*(x+20u)];
            c_hi[x] = (*s)[2u*x+1u] ^ (*s)[2u*(x+5u)+1u] ^ (*s)[2u*(x+10u)+1u] ^ (*s)[2u*(x+15u)+1u] ^ (*s)[2u*(x+20u)+1u];
        }
        var d_lo: array<u32, 5>;
        var d_hi: array<u32, 5>;
        for (var x: u32 = 0u; x < 5u; x = x + 1u) {
            let r = rotl64_split(c_lo[(x+1u) % 5u], c_hi[(x+1u) % 5u], 1u);
            d_lo[x] = c_lo[(x+4u) % 5u] ^ r.x;
            d_hi[x] = c_hi[(x+4u) % 5u] ^ r.y;
        }
        for (var y: u32 = 0u; y < 25u; y = y + 5u) {
            for (var x: u32 = 0u; x < 5u; x = x + 1u) {
                (*s)[2u*(y+x)]    = (*s)[2u*(y+x)]    ^ d_lo[x];
                (*s)[2u*(y+x)+1u] = (*s)[2u*(y+x)+1u] ^ d_hi[x];
            }
        }
        var b_lo: array<u32, 25>;
        var b_hi: array<u32, 25>;
        for (var y: u32 = 0u; y < 5u; y = y + 1u) {
            for (var x: u32 = 0u; x < 5u; x = x + 1u) {
                let i = x + 5u * y;
                let j = y + 5u * ((2u * x + 3u * y) % 5u);
                let r = rotl64_split((*s)[2u*i], (*s)[2u*i+1u], kKeccakRot[i]);
                b_lo[j] = r.x;
                b_hi[j] = r.y;
            }
        }
        for (var y: u32 = 0u; y < 25u; y = y + 5u) {
            let t0_lo = b_lo[y+0u]; let t0_hi = b_hi[y+0u];
            let t1_lo = b_lo[y+1u]; let t1_hi = b_hi[y+1u];
            let t2_lo = b_lo[y+2u]; let t2_hi = b_hi[y+2u];
            let t3_lo = b_lo[y+3u]; let t3_hi = b_hi[y+3u];
            let t4_lo = b_lo[y+4u]; let t4_hi = b_hi[y+4u];
            (*s)[2u*(y+0u)]    = t0_lo ^ ((~t1_lo) & t2_lo);
            (*s)[2u*(y+0u)+1u] = t0_hi ^ ((~t1_hi) & t2_hi);
            (*s)[2u*(y+1u)]    = t1_lo ^ ((~t2_lo) & t3_lo);
            (*s)[2u*(y+1u)+1u] = t1_hi ^ ((~t2_hi) & t3_hi);
            (*s)[2u*(y+2u)]    = t2_lo ^ ((~t3_lo) & t4_lo);
            (*s)[2u*(y+2u)+1u] = t2_hi ^ ((~t3_hi) & t4_hi);
            (*s)[2u*(y+3u)]    = t3_lo ^ ((~t4_lo) & t0_lo);
            (*s)[2u*(y+3u)+1u] = t3_hi ^ ((~t4_hi) & t0_hi);
            (*s)[2u*(y+4u)]    = t4_lo ^ ((~t0_lo) & t1_lo);
            (*s)[2u*(y+4u)+1u] = t4_hi ^ ((~t0_hi) & t1_hi);
        }
        (*s)[0] = (*s)[0] ^ kKeccakRC_lo[round];
        (*s)[1] = (*s)[1] ^ kKeccakRC_hi[round];
    }
}

// keccak256 over a u32-packed byte stream (each input u32 holds 4 LE bytes).
// `data` is array<u32, kKeccakInputWords>, `len` is byte length (must be a
// multiple of 4 — all our hash leaves satisfy this). Result is 8 LE u32.
//
// rate=136 bytes = 17 u64 lanes = 34 u32 words.
const kKeccakInputWords: u32 = 64u;  // 256 bytes max input — covers all our leaves.

fn keccak256(data: ptr<function, array<u32, 64>>, len_bytes: u32, out: ptr<function, array<u32, 8>>) {
    var s: array<u32, 50>;
    for (var i: u32 = 0u; i < 50u; i = i + 1u) { s[i] = 0u; }

    let rate_words: u32 = 34u;  // 17 lanes * 2 u32
    let rate_bytes: u32 = 136u;
    let len_words: u32 = (len_bytes + 3u) / 4u;

    // Full-rate blocks. For our leaf sizes (<= 200B) this loop runs at most once.
    var off_words: u32 = 0u;
    var off_bytes: u32 = 0u;
    loop {
        if (len_bytes < off_bytes + rate_bytes) { break; }
        for (var i: u32 = 0u; i < rate_words; i = i + 1u) {
            // Lane index w corresponds to s[w] (low 32 of lane w/2 if w even, high if w odd).
            // Actually each rate_word is one of the 34 u32 words of the 17-lane block.
            // Word i of rate goes into state index i — which is exactly s[i] for i<34.
            s[i] = s[i] ^ (*data)[off_words + i];
        }
        keccak_f1600(&s);
        off_words = off_words + rate_words;
        off_bytes = off_bytes + rate_bytes;
    }
    // Pad final block.
    var block: array<u32, 34>;
    for (var i: u32 = 0u; i < rate_words; i = i + 1u) { block[i] = 0u; }
    let rem_bytes = len_bytes - off_bytes;
    let rem_words = rem_bytes / 4u;          // input is 4-byte-multiple
    for (var i: u32 = 0u; i < rem_words; i = i + 1u) {
        block[i] = (*data)[off_words + i];
    }
    // 0x01 byte at position rem_bytes in the rate block. With 4-byte alignment
    // it lands at the lowest byte of word rem_words.
    block[rem_words] = block[rem_words] ^ 0x00000001u;
    // 0x80 byte at byte position rate_bytes-1 = byte 135 = word 33 byte 3.
    block[rate_words - 1u] = block[rate_words - 1u] ^ 0x80000000u;
    for (var i: u32 = 0u; i < rate_words; i = i + 1u) {
        s[i] = s[i] ^ block[i];
    }
    keccak_f1600(&s);
    // Squeeze 32 bytes = 8 u32.
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        (*out)[i] = s[i];
    }
}

// keccak256 over a 32-byte digest (8 u32). Specialisation used by membership +
// asset locator paths.
fn keccak256_32(in_words: ptr<function, array<u32, 8>>, out: ptr<function, array<u32, 8>>) {
    var buf: array<u32, 64>;
    for (var i: u32 = 0u; i < 64u; i = i + 1u) { buf[i] = 0u; }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { buf[i] = (*in_words)[i]; }
    keccak256(&buf, 32u, out);
}

// =============================================================================
// 32-byte digest equality.
// =============================================================================

fn digest_eq8(a: ptr<function, array<u32, 8>>, b: ptr<function, array<u32, 8>>) -> bool {
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        if ((*a)[i] != (*b)[i]) { return false; }
    }
    return true;
}

// =============================================================================
// 64-bit add/sub on (lo32, hi32) split.
// =============================================================================

fn u64_add(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let new_lo = a_lo + b_lo;
    let carry = select(0u, 1u, new_lo < a_lo);
    let new_hi = a_hi + b_hi + carry;
    return vec2<u32>(new_lo, new_hi);
}

// Returns (out_lo, out_hi, ok) where ok==1u if a >= b else 0u.
fn u64_sub_checked(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec3<u32> {
    let ok = select(0u, 1u, (a_hi > b_hi) || (a_hi == b_hi && a_lo >= b_lo));
    let borrow = select(0u, 1u, a_lo < b_lo);
    let out_lo = a_lo - b_lo;
    let out_hi = a_hi - b_hi - borrow;
    return vec3<u32>(out_lo, out_hi, ok);
}

// 64-bit multiply (h_lo, h_hi) * (p_lo, p_hi) mod 2^64.
// Returns (result_lo, result_hi).
fn mul64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    // Result_lo = low 32 of (a_lo * b_lo).
    // Result_hi = high 32 of (a_lo * b_lo) + (a_lo * b_hi + a_hi * b_lo) mod 2^32.
    let a0 = a_lo & 0xFFFFu;
    let a1 = a_lo >> 16u;
    let b0 = b_lo & 0xFFFFu;
    let b1 = b_lo >> 16u;
    let p00 = a0 * b0;        // 32-bit product (16x16 -> <=32)
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    let prod_lo = (p00 & 0xFFFFu) | (mid << 16u);
    let prod_hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    let cross = a_lo * b_hi + a_hi * b_lo;
    return vec2<u32>(prod_lo, prod_hi + cross);
}

// FNV-1a over a 32-byte asset_id (matches xvm_kernels_common.h.metal).
// Returns (hash_lo & mask).
fn asset_index_hash(words: ptr<function, array<u32, 8>>, mask: u32) -> u32 {
    var h_lo: u32 = 0x84222325u;        // FNV offset (low 32)
    var h_hi: u32 = 0xcbf29ce4u;        // FNV offset (high 32)
    let prime_lo: u32 = 0x000001b3u;    // FNV prime (low 32)
    let prime_hi: u32 = 0x00000100u;    // FNV prime (high 32) — 0x100000001b3
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        let w = (*words)[i / 4u];
        let byte_v = (w >> ((i % 4u) * 8u)) & 0xFFu;
        h_lo = h_lo ^ byte_v;
        let r = mul64(h_lo, h_hi, prime_lo, prime_hi);
        h_lo = r.x;
        h_hi = r.y;
    }
    return h_lo & mask;
}
