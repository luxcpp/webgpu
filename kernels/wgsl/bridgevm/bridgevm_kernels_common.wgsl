// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// bridgevm_kernels_common.wgsl — shared WGSL declarations for BridgeVM.
//
// Layout structs MUST match bridgevm_gpu_layout.hpp byte-for-byte. WGSL has
// no native u64; we split every u64 into (lo, hi) u32 lanes. 32-byte digests
// are stored as 8 u32s (each holding 4 bytes, little-endian). 20-byte EVM
// addresses use 5 u32s (last byte unused in u32[4]); 48-byte BLS pubkey is
// 12 u32s; 96-byte BLS aggregate signature is 24 u32s.
//
// WebGPU storage-buffer stride for an array<T> is sizeof(T) in this packed
// 4-bytes-per-u32 encoding, which matches alignas(16) C structs exactly.

// =============================================================================
// Structs — byte-for-byte sizes match the host
// =============================================================================

// 208 bytes
struct Signer {
    signer_id_lo:        u32,            // 0
    signer_id_hi:        u32,            // 4
    lux_address:         array<u32, 5>,  // 8 (20 bytes)
    pad_addr:            u32,            // 28
    bond_amount_lo_lo:   u32,            // 32  (low 32 of bond_amount_lo)
    bond_amount_lo_hi:   u32,            // 36  (high 32 of bond_amount_lo)
    bond_amount_hi_lo:   u32,            // 40
    bond_amount_hi_hi:   u32,            // 44
    opt_in_height_lo:    u32,            // 48
    opt_in_height_hi:    u32,            // 52
    exit_epoch_lo:       u32,            // 56
    exit_epoch_hi:       u32,            // 60
    sign_count_lo:       u32,            // 64
    sign_count_hi:       u32,            // 68
    bls_pubkey:          array<u32, 12>, // 72 (48 bytes)
    ringtail_pubkey:     array<u32, 8>,  // 120 (32 bytes)
    mldsa_pubkey:        array<u32, 8>,  // 152 (32 bytes)
    status:              u32,            // 184
    jail_until_epoch:    u32,            // 188
    slash_count:         u32,            // 192
    occupied:            u32,            // 196
    pad_tail_lo:         u32,            // 200
    pad_tail_hi:         u32,            // 204 -> 208
};

// 80 bytes
struct LiquidityEntry {
    provider_addr:       array<u32, 5>,  // 0..20
    pad_addr:            u32,            // 20  (note: layout puts pad_addr[4] tail at byte 20)
    asset_id:            u32,            // 24
    status:              u32,            // 28
    amount_lo_lo:        u32,            // 32
    amount_lo_hi:        u32,            // 36
    amount_hi_lo:        u32,            // 40
    amount_hi_hi:        u32,            // 44
    fee_accrual_lo_lo:   u32,            // 48
    fee_accrual_lo_hi:   u32,            // 52
    fee_accrual_hi_lo:   u32,            // 56
    fee_accrual_hi_hi:   u32,            // 60
    deposit_height_lo:   u32,            // 64
    deposit_height_hi:   u32,            // 68
    pad0_lo:             u32,            // 72
    pad0_hi:             u32,            // 76 -> 80
};

// 64 bytes
struct DailyLimit {
    asset_id:            u32,            // 0
    status:              u32,            // 4
    daily_cap_lo_lo:     u32,            // 8
    daily_cap_lo_hi:     u32,            // 12
    daily_cap_hi_lo:     u32,            // 16
    daily_cap_hi_hi:     u32,            // 20
    used_today_lo_lo:    u32,            // 24
    used_today_lo_hi:    u32,            // 28
    used_today_hi_lo:    u32,            // 32
    used_today_hi_hi:    u32,            // 36
    reset_epoch_lo:      u32,            // 40
    reset_epoch_hi:      u32,            // 44
    pad0_lo:             u32,            // 48
    pad0_hi:             u32,            // 52
    pad1_lo:             u32,            // 56
    pad1_hi:             u32,            // 60 -> 64
};

// 240 bytes
struct Message {
    msg_id:              array<u32, 8>,   // 0..32
    payload_root:        array<u32, 8>,   // 32..64
    agg_signature:       array<u32, 24>,  // 64..160
    signers_bitmap_lo_lo: u32,            // 160
    signers_bitmap_lo_hi: u32,            // 164
    signers_bitmap_hi_lo: u32,            // 168
    signers_bitmap_hi_hi: u32,            // 172
    nonce_lo:            u32,             // 176
    nonce_hi:            u32,             // 180
    src_chain:           u32,             // 184
    dst_chain:           u32,             // 188
    kind:                u32,             // 192
    status:              u32,             // 196
    asset_id:            u32,             // 200
    signer_count:        u32,             // 204
    amount_lo_lo:        u32,             // 208
    amount_lo_hi:        u32,             // 212
    amount_hi_lo:        u32,             // 216
    amount_hi_hi:        u32,             // 220
    arrival_height_lo:   u32,             // 224
    arrival_height_hi:   u32,             // 228
    pad_tail_lo:         u32,             // 232
    pad_tail_hi:         u32,             // 236 -> 240
};

// 240 bytes
struct BridgeVMEpochState {
    current_epoch_lo:        u32,        // 0
    current_epoch_hi:        u32,        // 4
    next_epoch_height_lo:    u32,        // 8
    next_epoch_height_hi:    u32,        // 12
    total_active_bond_lo_lo: u32,        // 16
    total_active_bond_lo_hi: u32,        // 20
    total_active_bond_hi_lo: u32,        // 24
    total_active_bond_hi_hi: u32,        // 28
    active_signer_count:     u32,        // 32
    pending_drop_count:      u32,        // 36
    inbox_count:             u32,        // 40
    outbox_count:            u32,        // 44
    signer_set_root:         array<u32, 8>,  // 48..80
    liquidity_root:          array<u32, 8>,  // 80..112
    inbox_root:              array<u32, 8>,  // 112..144
    outbox_root:             array<u32, 8>,  // 144..176
    daily_limit_root:        array<u32, 8>,  // 176..208
    bridgevm_state_root:     array<u32, 8>,  // 208..240
};

// 112 bytes
struct BridgeVMRoundDescriptor {
    chain_id_lo:         u32,            // 0
    chain_id_hi:         u32,            // 4
    round_lo:            u32,            // 8
    round_hi:            u32,            // 12
    timestamp_ns_lo:     u32,            // 16
    timestamp_ns_hi:     u32,            // 20
    epoch_lo:            u32,            // 24
    epoch_hi:            u32,            // 28
    height_lo:           u32,            // 32
    height_hi:           u32,            // 36
    mode:                u32,            // 40
    inbound_msg_count:   u32,            // 44
    signer_op_count:     u32,            // 48
    liquidity_op_count:  u32,            // 52
    outbound_req_count:  u32,            // 56
    closing_flag:        u32,            // 60
    pad0_lo:             u32,            // 64
    pad0_hi:             u32,            // 68
    pad1_lo:             u32,            // 72
    pad1_hi:             u32,            // 76
    parent_state_root:   array<u32, 8>,  // 80..112
};

// 224 bytes
struct SignerOp {
    signer_id_lo:        u32,            // 0
    signer_id_hi:        u32,            // 4
    lux_address:         array<u32, 5>,  // 8..28
    pad_addr:            u32,            // 28
    bond_amount_lo_lo:   u32,            // 32
    bond_amount_lo_hi:   u32,            // 36
    bond_amount_hi_lo:   u32,            // 40
    bond_amount_hi_hi:   u32,            // 44
    opt_in_height_lo:    u32,            // 48
    opt_in_height_hi:    u32,            // 52
    bls_pubkey:          array<u32, 12>, // 56..104
    ringtail_pubkey:     array<u32, 8>,  // 104..136
    mldsa_pubkey:        array<u32, 8>,  // 136..168
    kind:                u32,            // 168
    jail_until_epoch:    u32,            // 172
    epoch:               u32,            // 176
    slash_amount_lo:     u32,            // 180
    slash_amount_hi:     u32,            // 184
    pad0:                u32,            // 188
    evidence_digest:     array<u32, 8>,  // 192..224
};

// 64 bytes
struct LiquidityOp {
    provider_addr:       array<u32, 5>,  // 0..20
    pad_addr:            u32,            // 20
    asset_id:            u32,            // 24
    kind:                u32,            // 28
    amount_lo_lo:        u32,            // 32
    amount_lo_hi:        u32,            // 36
    amount_hi_lo:        u32,            // 40
    amount_hi_hi:        u32,            // 44
    height_lo:           u32,            // 48
    height_hi:           u32,            // 52
    pad0_lo:             u32,            // 56
    pad0_hi:             u32,            // 60 -> 64
};

// 112 bytes
struct OutboundReq {
    payload_root:        array<u32, 8>,  // 0..32
    recipient:           array<u32, 5>,  // 32..52
    pad_addr:            u32,            // 52
    src_chain:           u32,            // 56
    dst_chain:           u32,            // 60
    kind:                u32,            // 64
    asset_id:            u32,            // 68
    nonce_lo:            u32,            // 72
    nonce_hi:            u32,            // 76
    amount_lo_lo:        u32,            // 80
    amount_lo_hi:        u32,            // 84
    amount_hi_lo:        u32,            // 88
    amount_hi_hi:        u32,            // 92
    height_lo:           u32,            // 96
    height_hi:           u32,            // 100
    pad_tail_lo:         u32,            // 104
    pad_tail_hi:         u32,            // 108 -> 112
};

// 304 bytes
struct BridgeVMTransitionResult {
    status:                  u32,        // 0
    inbound_apply_count:     u32,        // 4
    signer_apply_count:      u32,        // 8
    liquidity_apply_count:   u32,        // 12
    outbound_apply_count:    u32,        // 16
    active_signer_count:     u32,        // 20
    jailed_count:            u32,        // 24
    tombstoned_count:        u32,        // 28
    total_active_bond_lo_lo: u32,        // 32
    total_active_bond_lo_hi: u32,        // 36
    total_active_bond_hi_lo: u32,        // 40
    total_active_bond_hi_hi: u32,        // 44
    total_inbound_amount_lo_lo: u32,     // 48
    total_inbound_amount_lo_hi: u32,     // 52
    total_inbound_amount_hi_lo: u32,     // 56
    total_inbound_amount_hi_hi: u32,     // 60
    total_outbound_amount_lo_lo: u32,    // 64
    total_outbound_amount_lo_hi: u32,    // 68
    total_outbound_amount_hi_lo: u32,    // 72
    total_outbound_amount_hi_hi: u32,    // 76
    total_fees_accrued_lo_lo: u32,       // 80
    total_fees_accrued_lo_hi: u32,       // 84
    total_fees_accrued_hi_lo: u32,       // 88
    total_fees_accrued_hi_hi: u32,       // 92
    epoch_lo:                u32,        // 96
    epoch_hi:                u32,        // 100
    pad0_lo:                 u32,        // 104
    pad0_hi:                 u32,        // 108
    signer_set_root:         array<u32, 8>,  // 112..144
    liquidity_root:          array<u32, 8>,  // 144..176
    inbox_root:              array<u32, 8>,  // 176..208
    outbox_root:             array<u32, 8>,  // 208..240
    daily_limit_root:        array<u32, 8>,  // 240..272
    bridgevm_state_root:     array<u32, 8>,  // 272..304
};

// =============================================================================
// Constants
// =============================================================================

const kSignerStatusActive       : u32 = 0x1u;
const kSignerStatusJailed       : u32 = 0x2u;
const kSignerStatusTombstoned   : u32 = 0x4u;
const kSignerStatusPendingAdd   : u32 = 0x8u;
const kSignerStatusPendingDrop  : u32 = 0x10u;
const kSignerStatusExiting      : u32 = 0x20u;

const kLiqStatusActive          : u32 = 1u;
const kLiqStatusClosed          : u32 = 3u;

const kMsgStatusFree            : u32 = 0u;
const kMsgStatusVerified        : u32 = 1u;
const kMsgStatusAccepted        : u32 = 2u;
const kMsgStatusOutboxEmit      : u32 = 4u;

const kMsgKindMint              : u32 = 0u;
const kMsgKindBurn              : u32 = 1u;
const kMsgKindGeneric           : u32 = 2u;

const kSOpOptIn                 : u32 = 0u;
const kSOpOptOut                : u32 = 1u;
const kSOpSlash                 : u32 = 2u;
const kSOpUnjail                : u32 = 3u;
const kSOpRotateKeys            : u32 = 4u;

const kLOpDeposit               : u32 = 0u;
const kLOpWithdraw              : u32 = 1u;
const kLOpAccrueFee             : u32 = 2u;

const kMaxSigners               : u32 = 128u;
// 100M LUX in nLUX: 1e17 = 0x6BC75E2D63100000.
const kMinSignerBondLoLo        : u32 = 0x63100000u;
const kMinSignerBondLoHi        : u32 = 0x6BC75E2Du;
const kMinSignerBondHiLo        : u32 = 0u;
const kMinSignerBondHiHi        : u32 = 0u;

// =============================================================================
// 64-bit emulation
// =============================================================================

fn u64_eq(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    return a_lo == b_lo && a_hi == b_hi;
}
fn u64_lt(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    if (a_hi < b_hi) { return true; }
    if (a_hi > b_hi) { return false; }
    return a_lo < b_lo;
}
fn u64_le(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    if (a_hi < b_hi) { return true; }
    if (a_hi > b_hi) { return false; }
    return a_lo <= b_lo;
}
fn u64_is_zero(a_lo: u32, a_hi: u32) -> bool { return a_lo == 0u && a_hi == 0u; }

// 128-bit add (saturating to UINT128_MAX). Inputs are pairs of u64=(lo,hi),
// each u64 in turn split into (low32, high32) at call sites.
struct U128 { lo_lo: u32, lo_hi: u32, hi_lo: u32, hi_hi: u32 };

fn u128_make(lo_lo: u32, lo_hi: u32, hi_lo: u32, hi_hi: u32) -> U128 {
    var v: U128;
    v.lo_lo = lo_lo; v.lo_hi = lo_hi;
    v.hi_lo = hi_lo; v.hi_hi = hi_hi;
    return v;
}

// add a single u32 with carry, returning (sum, new_carry).
fn add_carry(a: u32, b: u32, cin: u32) -> vec2<u32> {
    let s1 = a + b;
    let c1: u32 = select(0u, 1u, s1 < a);
    let s2 = s1 + cin;
    let c2: u32 = select(0u, 1u, s2 < s1);
    return vec2<u32>(s2, c1 + c2);
}

fn u128_add(a: U128, b: U128) -> U128 {
    let r0 = add_carry(a.lo_lo, b.lo_lo, 0u);
    let r1 = add_carry(a.lo_hi, b.lo_hi, r0.y);
    let r2 = add_carry(a.hi_lo, b.hi_lo, r1.y);
    let r3 = add_carry(a.hi_hi, b.hi_hi, r2.y);
    var out: U128;
    if (r3.y != 0u) {
        // overflow -> saturate
        out.lo_lo = 0xFFFFFFFFu; out.lo_hi = 0xFFFFFFFFu;
        out.hi_lo = 0xFFFFFFFFu; out.hi_hi = 0xFFFFFFFFu;
        return out;
    }
    out.lo_lo = r0.x; out.lo_hi = r1.x;
    out.hi_lo = r2.x; out.hi_hi = r3.x;
    return out;
}

fn sub_borrow(a: u32, b: u32, bin: u32) -> vec2<u32> {
    let d1 = a - b;
    let bo1: u32 = select(0u, 1u, a < b);
    let d2 = d1 - bin;
    let bo2: u32 = select(0u, 1u, d1 < bin);
    return vec2<u32>(d2, bo1 + bo2);
}

fn u128_sub(a: U128, b: U128) -> U128 {
    // saturate to zero on underflow
    if (u128_lt(a, b)) {
        return u128_make(0u, 0u, 0u, 0u);
    }
    let r0 = sub_borrow(a.lo_lo, b.lo_lo, 0u);
    let r1 = sub_borrow(a.lo_hi, b.lo_hi, r0.y);
    let r2 = sub_borrow(a.hi_lo, b.hi_lo, r1.y);
    let r3 = sub_borrow(a.hi_hi, b.hi_hi, r2.y);
    return u128_make(r0.x, r1.x, r2.x, r3.x);
}

fn u128_lt(a: U128, b: U128) -> bool {
    if (a.hi_hi != b.hi_hi) { return a.hi_hi < b.hi_hi; }
    if (a.hi_lo != b.hi_lo) { return a.hi_lo < b.hi_lo; }
    if (a.lo_hi != b.lo_hi) { return a.lo_hi < b.lo_hi; }
    return a.lo_lo < b.lo_lo;
}

fn popcount32(x: u32) -> u32 { return countOneBits(x); }

fn popcount_u128_pair(lo_lo: u32, lo_hi: u32, hi_lo: u32, hi_hi: u32) -> u32 {
    return popcount32(lo_lo) + popcount32(lo_hi) + popcount32(hi_lo) + popcount32(hi_hi);
}

fn meets_bft_threshold(signer_count: u32, active_count: u32) -> bool {
    if (active_count == 0u) { return false; }
    let needed = (2u * active_count + 2u) / 3u;
    return signer_count >= needed;
}

// =============================================================================
// FNV-1a hashes for open-address tables
// =============================================================================
//
// Matches CPU/CUDA: h = 0xcbf29ce484222325 ^ key * 0x100000001b3 (mod 2^64),
// take low 32 & mask. With multiplier 0x100000001b3 = 0x1_000001b3, the
// product low 32 bits = xor_lo * 0x000001b3 (the high u32 of multiplier is
// 1, contributing only to bits ≥ 32). So low32(prod) = xor_lo * 0x1b3.
// This file caches the low/high halves of the FNV state through repeated
// XOR+multiply rounds, then masks to (size-1).

fn fnv1a_init_lo() -> u32 { return 0x84222325u; }  // low 32 of seed
fn fnv1a_init_hi() -> u32 { return 0xcbf29ce4u; }  // high 32

fn fnv_round(h_lo: u32, h_hi: u32, k_lo: u32, k_hi: u32) -> vec2<u32> {
    let xl = h_lo ^ k_lo;
    let xh = h_hi ^ k_hi;
    // Multiply 64-bit (xh:xl) by 0x100000001b3:
    //   prod_lo32 = xl * 0x1b3
    //   prod_hi32 = (xl * 0x1) +              // high u32 of (xl * mult)
    //               (xh * 0x1b3) +            // low u32 of (xh * 0x1b3)
    //               carry-up from prod_lo32 calc
    // For our use we only need the low 32 bits of the product (then mask).
    // But to chain the FNV state across multiple keys, we also need the high
    // 32 bits to remain consistent with CPU. So compute both.
    let mlo = 0x000001b3u;
    let mhi = 0x00000001u;
    // 64x64 -> 64 (low half) — schoolbook with three 32x32 cross-terms.
    // a = xl * mlo (full 64 = (lo, hi))
    let a_lo = xl * mlo;
    // hi half of xl*mlo: requires extended math. WGSL has firstLeadingBit
    // but no native u32x32 -> u64. Use shift-add reconstruction via 16-bit
    // halves to get the high word.
    let xll: u32 = xl & 0xFFFFu;
    let xlh: u32 = xl >> 16u;
    let mll: u32 = mlo & 0xFFFFu;
    let mlh: u32 = mlo >> 16u;
    // Cross products (each <= 0xFFFE0001, fits u32).
    let p00 = xll * mll;
    let p01 = xll * mlh;
    let p10 = xlh * mll;
    let p11 = xlh * mlh;
    let mid = p01 + p10;             // may overflow
    let mid_carry: u32 = select(0u, 1u, mid < p01);
    let lo_combine = p00 + (mid << 16u);
    let lo_carry: u32 = select(0u, 1u, lo_combine < p00);
    let a_hi = p11 + (mid >> 16u) + (mid_carry << 16u) + lo_carry;
    // b = xh * mlo (only need its low 32)
    let b_lo = xh * mlo;
    // c = xl * mhi (only need its low 32 since mhi=1: c_lo = xl)
    let c_lo = xl * mhi;
    // out_lo = a_lo
    // out_hi = a_hi + b_lo + c_lo
    let out_lo = a_lo;
    let s1 = a_hi + b_lo;
    let out_hi = s1 + c_lo;
    return vec2<u32>(out_lo, out_hi);
}

// Hash a u64 key, return low 32 bits masked.
fn hash_u64(klo: u32, khi: u32, mask: u32) -> u32 {
    let r = fnv_round(fnv1a_init_lo(), fnv1a_init_hi(), klo, khi);
    return r.x & mask;
}

// Hash a 20-byte addr (5 u32) plus an asset_id, return low 32 masked.
// CPU folds the 20 bytes one-byte-per-round, then folds asset_id as a single
// u64-XOR-multiply round (NOT per-byte). Match that exactly.
fn hash_addr_asset(addr: ptr<function, array<u32, 5>>, asset: u32, mask: u32) -> u32 {
    var h_lo = fnv1a_init_lo();
    var h_hi = fnv1a_init_hi();
    for (var k: u32 = 0u; k < 20u; k = k + 1u) {
        let word = (*addr)[k >> 2u];
        let sh = (k & 3u) * 8u;
        let byte = (word >> sh) & 0xFFu;
        let r = fnv_round(h_lo, h_hi, byte, 0u);
        h_lo = r.x; h_hi = r.y;
    }
    let r = fnv_round(h_lo, h_hi, asset, 0u);
    return r.x & mask;
}

// Hash a 32-byte msg_id (8 u32), return low 32 masked.
fn hash_msg_id(id: ptr<function, array<u32, 8>>, mask: u32) -> u32 {
    var h_lo = fnv1a_init_lo();
    var h_hi = fnv1a_init_hi();
    for (var k: u32 = 0u; k < 32u; k = k + 1u) {
        let word = (*id)[k >> 2u];
        let sh = (k & 3u) * 8u;
        let byte = (word >> sh) & 0xFFu;
        let r = fnv_round(h_lo, h_hi, byte, 0u);
        h_lo = r.x; h_hi = r.y;
    }
    return h_lo & mask;
}

// =============================================================================
// keccak-f[1600] over a 50-u32 state (25 lanes of u64 = (lo, hi)).
// =============================================================================

const KECCAK_RC_LO = array<u32, 24>(
    0x00000001u, 0x00008082u, 0x0000808Au, 0x80008000u,
    0x0000808Bu, 0x80000001u, 0x80008081u, 0x00008009u,
    0x0000008Au, 0x00000088u, 0x80008009u, 0x8000000Au,
    0x8000808Bu, 0x0000008Bu, 0x00008089u, 0x00008003u,
    0x00008002u, 0x00000080u, 0x0000800Au, 0x8000000Au,
    0x80008081u, 0x00008080u, 0x80000001u, 0x80008008u,
);
const KECCAK_RC_HI = array<u32, 24>(
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x80000000u, 0x80000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u,
);
const KECCAK_ROT = array<u32, 25>(
     0u,  1u, 62u, 28u, 27u,
    36u, 44u,  6u, 55u, 20u,
     3u, 10u, 43u, 25u, 39u,
    41u, 45u, 15u, 21u,  8u,
    18u,  2u, 61u, 56u, 14u,
);

fn rotl64(lo: u32, hi: u32, n: u32) -> vec2<u32> {
    let m = n & 63u;
    if (m == 0u) { return vec2<u32>(lo, hi); }
    if (m < 32u) {
        let new_lo = (lo << m) | (hi >> (32u - m));
        let new_hi = (hi << m) | (lo >> (32u - m));
        return vec2<u32>(new_lo, new_hi);
    }
    let mm = m - 32u;
    if (mm == 0u) { return vec2<u32>(hi, lo); }
    let new_lo = (hi << mm) | (lo >> (32u - mm));
    let new_hi = (lo << mm) | (hi >> (32u - mm));
    return vec2<u32>(new_lo, new_hi);
}

fn keccak_f1600(state: ptr<function, array<u32, 50>>) {
    for (var round: u32 = 0u; round < 24u; round = round + 1u) {
        var c_lo: array<u32, 5>;
        var c_hi: array<u32, 5>;
        for (var x: u32 = 0u; x < 5u; x = x + 1u) {
            var lo = (*state)[2u*x];
            var hi = (*state)[2u*x + 1u];
            for (var y: u32 = 1u; y < 5u; y = y + 1u) {
                lo = lo ^ (*state)[2u*(x + 5u*y)];
                hi = hi ^ (*state)[2u*(x + 5u*y) + 1u];
            }
            c_lo[x] = lo;
            c_hi[x] = hi;
        }
        var d_lo: array<u32, 5>;
        var d_hi: array<u32, 5>;
        for (var x: u32 = 0u; x < 5u; x = x + 1u) {
            let r = rotl64(c_lo[(x + 1u) % 5u], c_hi[(x + 1u) % 5u], 1u);
            d_lo[x] = c_lo[(x + 4u) % 5u] ^ r.x;
            d_hi[x] = c_hi[(x + 4u) % 5u] ^ r.y;
        }
        for (var y: u32 = 0u; y < 25u; y = y + 5u) {
            for (var x: u32 = 0u; x < 5u; x = x + 1u) {
                (*state)[2u*(y + x)]      = (*state)[2u*(y + x)]      ^ d_lo[x];
                (*state)[2u*(y + x) + 1u] = (*state)[2u*(y + x) + 1u] ^ d_hi[x];
            }
        }
        var b_lo: array<u32, 25>;
        var b_hi: array<u32, 25>;
        for (var y: u32 = 0u; y < 5u; y = y + 1u) {
            for (var x: u32 = 0u; x < 5u; x = x + 1u) {
                let i = x + 5u * y;
                let j = y + 5u * ((2u * x + 3u * y) % 5u);
                let r = rotl64((*state)[2u*i], (*state)[2u*i + 1u], KECCAK_ROT[i]);
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
            (*state)[2u*(y+0u)]      = t0_lo ^ ((~t1_lo) & t2_lo);
            (*state)[2u*(y+0u) + 1u] = t0_hi ^ ((~t1_hi) & t2_hi);
            (*state)[2u*(y+1u)]      = t1_lo ^ ((~t2_lo) & t3_lo);
            (*state)[2u*(y+1u) + 1u] = t1_hi ^ ((~t2_hi) & t3_hi);
            (*state)[2u*(y+2u)]      = t2_lo ^ ((~t3_lo) & t4_lo);
            (*state)[2u*(y+2u) + 1u] = t2_hi ^ ((~t3_hi) & t4_hi);
            (*state)[2u*(y+3u)]      = t3_lo ^ ((~t4_lo) & t0_lo);
            (*state)[2u*(y+3u) + 1u] = t3_hi ^ ((~t4_hi) & t0_hi);
            (*state)[2u*(y+4u)]      = t4_lo ^ ((~t0_lo) & t1_lo);
            (*state)[2u*(y+4u) + 1u] = t4_hi ^ ((~t0_hi) & t1_hi);
        }
        (*state)[0] = (*state)[0] ^ KECCAK_RC_LO[round];
        (*state)[1] = (*state)[1] ^ KECCAK_RC_HI[round];
    }
}

// =============================================================================
// keccak256 over a byte buffer (rate=136). Buffer is array<u32, N> with one
// byte packed per low-8-bits pattern (we store byte-per-u32 for simplicity).
// Largest BridgeVM leaf = signer leaf at 8+20+4+8+8+8+8+8+48+32+32+4+4+4+4 = 196 bytes.
// Composed root = 32*6+8+8+8+4 = 220 bytes. We size buffers at 256.
// =============================================================================

fn keccak_absorb_byte(state: ptr<function, array<u32, 50>>, i: u32, byte_val: u32) {
    let lane: u32 = i / 8u;
    let in_lane: u32 = i & 7u;
    let sh: u32 = in_lane * 8u;
    if (in_lane < 4u) {
        (*state)[2u*lane] = (*state)[2u*lane] ^ ((byte_val & 0xFFu) << sh);
    } else {
        (*state)[2u*lane + 1u] = (*state)[2u*lane + 1u] ^ ((byte_val & 0xFFu) << (sh - 32u));
    }
}

fn keccak_read_byte(state: ptr<function, array<u32, 50>>, i: u32) -> u32 {
    let lane: u32 = i / 8u;
    let in_lane: u32 = i & 7u;
    let sh: u32 = in_lane * 8u;
    if (in_lane < 4u) {
        return ((*state)[2u*lane] >> sh) & 0xFFu;
    }
    return ((*state)[2u*lane + 1u] >> (sh - 32u)) & 0xFFu;
}

const kKeccakRate: u32 = 136u;

// Buffer size 256 byte-per-u32. Used for all leaves and the composed root.
fn keccak256_buf256(data: ptr<function, array<u32, 256>>, len: u32,
                    out: ptr<function, array<u32, 8>>)
{
    var s: array<u32, 50>;
    for (var k: u32 = 0u; k < 50u; k = k + 1u) { s[k] = 0u; }
    var off: u32 = 0u;
    while (len - off >= kKeccakRate) {
        for (var i: u32 = 0u; i < kKeccakRate; i = i + 1u) {
            keccak_absorb_byte(&s, i, (*data)[off + i]);
        }
        keccak_f1600(&s);
        off = off + kKeccakRate;
    }
    let rem: u32 = len - off;
    for (var i: u32 = 0u; i < rem; i = i + 1u) {
        keccak_absorb_byte(&s, i, (*data)[off + i]);
    }
    keccak_absorb_byte(&s, rem, 0x01u);
    keccak_absorb_byte(&s, kKeccakRate - 1u, 0x80u);
    keccak_f1600(&s);
    // Squeeze 32 bytes into 8 u32 (4-byte-per-u32 little endian).
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let b0 = keccak_read_byte(&s, k*4u + 0u);
        let b1 = keccak_read_byte(&s, k*4u + 1u);
        let b2 = keccak_read_byte(&s, k*4u + 2u);
        let b3 = keccak_read_byte(&s, k*4u + 3u);
        (*out)[k] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
    }
}

// Helpers to write byte-per-u32 into a 256-byte buffer.
fn put_u32_le(buf: ptr<function, array<u32, 256>>, off: u32, v: u32) {
    (*buf)[off + 0u] = v & 0xFFu;
    (*buf)[off + 1u] = (v >> 8u) & 0xFFu;
    (*buf)[off + 2u] = (v >> 16u) & 0xFFu;
    (*buf)[off + 3u] = (v >> 24u) & 0xFFu;
}
fn put_u64_le(buf: ptr<function, array<u32, 256>>, off: u32, lo: u32, hi: u32) {
    put_u32_le(buf, off,      lo);
    put_u32_le(buf, off + 4u, hi);
}
// Copy a packed array<u32, K> (4 bytes per u32) as bytes into the buffer.
fn put_packed_u32_n(buf: ptr<function, array<u32, 256>>, off: u32,
                    src_word: u32, n_bytes: u32) {
    // unused — see specialised helpers below
}

// Spread 8 packed u32 (32 bytes) into byte form.
fn put_digest32(buf: ptr<function, array<u32, 256>>, off: u32, src: ptr<function, array<u32, 8>>) {
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let v = (*src)[k];
        put_u32_le(buf, off + k*4u, v);
    }
}

// 20-byte address from a packed array<u32, 5>.
fn put_addr20(buf: ptr<function, array<u32, 256>>, off: u32, src: ptr<function, array<u32, 5>>) {
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        put_u32_le(buf, off + k*4u, (*src)[k]);
    }
    // 5th word holds the last 4 bytes but only the low 4 bytes; here addr is
    // exactly 20 bytes — last word's low 32 has the trailing 4 bytes.
    put_u32_le(buf, off + 16u, (*src)[4]);
}

// 48-byte BLS pubkey from a packed array<u32, 12>.
fn put_bls48(buf: ptr<function, array<u32, 256>>, off: u32, src: ptr<function, array<u32, 12>>) {
    for (var k: u32 = 0u; k < 12u; k = k + 1u) {
        put_u32_le(buf, off + k*4u, (*src)[k]);
    }
}
