// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// platformvm_kernels_common.wgsl — shared WGSL device code for the four PVM kernels.
//
// WGSL constraints relative to Metal/CUDA:
//   * no #include — this file is concatenated into each kernel module by
//     the host engine before submission to wgpuDeviceCreateShaderModule.
//   * no native u64 — split into (lo:u32, hi:u32) lanes; the keccak state
//     is held as 50 u32 (lane i is (state[2*i], state[2*i+1])).
//   * structs use u32 fields for raw bytes; 32-byte digests = 8 u32, 48-byte
//     pubkey = 12 u32, all little-endian byte-packed. This matches a
//     memcpy() of platformvm_gpu_layout.hpp on every little-endian host.
//
// Determinism contract: byte-for-byte identical leaf encoding to
// platformvm_cpu_reference.cpp / .metal / .cu, fed to byte-for-byte
// identical keccak256 (Keccak-f[1600] / 0x01 / 0x80 padding, rate 136).

// =============================================================================
// Layout structs — must match platformvm_gpu_layout.hpp byte-for-byte.
// =============================================================================
//
// Each struct's WGSL field layout is intentionally a 1:1 little-endian byte
// reflection of the C++ struct. WGSL `std140`/`std430` rules with arrays of
// u32 align identically to a packed C struct of u32/u8 on little-endian.

struct ValidatorSlot {
    validator_id_lo:    u32,            // offset 0
    validator_id_hi:    u32,            // 4
    weight_lo:          u32,            // 8
    weight_hi:          u32,            // 12
    bls_pubkey:         array<u32, 12>, // 16 .. 64  (48 bytes)
    ringtail_pubkey:    array<u32, 8>,  // 64 .. 96  (32 bytes)
    mldsa_pubkey:       array<u32, 8>,  // 96 .. 128 (32 bytes)
    mldsa_groth16_root: array<u32, 8>,  // 128 .. 160 (32 bytes)
    status:             u32,            // 160
    jail_until_epoch:   u32,            // 164
    occupied:           u32,            // 168
    _pad0:              u32,            // 172 -> 176 (struct stride)
};

struct StakeRecord {
    delegator_id_lo:        u32, // 0
    delegator_id_hi:        u32, // 4
    validator_id_lo:        u32, // 8
    validator_id_hi:        u32, // 12
    amount_lo:              u32, // 16
    amount_hi:              u32, // 20
    lock_until_epoch_lo:    u32, // 24
    lock_until_epoch_hi:    u32, // 28
    reward_accumulator_lo:  u32, // 32
    reward_accumulator_hi:  u32, // 36
    commission_bps:         u32, // 40
    status:                 u32, // 44
    epoch_bonded:           u32, // 48
    epoch_unbonded:         u32, // 52
    _pad0_lo:               u32, // 56
    _pad0_hi:               u32, // 60 -> 64
};

struct SlashEvidence {
    validator_id_lo:    u32,             // 0
    validator_id_hi:    u32,             // 4
    height_lo:          u32,             // 8
    height_hi:          u32,             // 12
    slash_amount_lo:    u32,             // 16
    slash_amount_hi:    u32,             // 20
    kind:               u32,             // 24
    epoch:              u32,             // 28
    jail_for_epochs:    u32,             // 32
    _pad0:              u32,             // 36
    evidence_digest:    array<u32, 8>,   // 40 .. 72 (32 bytes)
    _pad1_lo:           u32,             // 72
    _pad1_hi:           u32,             // 76 -> 80
};

struct EpochState {
    current_epoch_lo:        u32,           // 0
    current_epoch_hi:        u32,           // 4
    next_epoch_height_lo:    u32,           // 8
    next_epoch_height_hi:    u32,           // 12
    total_active_stake_lo:   u32,           // 16
    total_active_stake_hi:   u32,           // 20
    active_validator_count:  u32,           // 24
    pending_drop_count:      u32,           // 28
    validator_set_root:      array<u32, 8>, // 32 .. 64
    stake_root:              array<u32, 8>, // 64 .. 96
    slashing_root:           array<u32, 8>, // 96 .. 128
    epoch_root:              array<u32, 8>, // 128 .. 160
};

struct PVMRoundDescriptor {
    chain_id_lo:           u32,           // 0
    chain_id_hi:           u32,           // 4
    round_lo:              u32,           // 8
    round_hi:              u32,           // 12
    timestamp_ns_lo:       u32,           // 16
    timestamp_ns_hi:       u32,           // 20
    epoch_lo:              u32,           // 24
    epoch_hi:              u32,           // 28
    mode:                  u32,           // 32
    validator_op_count:    u32,           // 36
    stake_op_count:        u32,           // 40
    slash_evidence_count:  u32,           // 44
    closing_flag:          u32,           // 48
    _pad0:                 u32,           // 52
    _pad1_lo:              u32,           // 56
    _pad1_hi:              u32,           // 60
    parent_epoch_root:     array<u32, 8>, // 64 .. 96
};

struct ValidatorOp {
    validator_id_lo:    u32,            // 0
    validator_id_hi:    u32,            // 4
    weight_lo:          u32,            // 8
    weight_hi:          u32,            // 12
    bls_pubkey:         array<u32, 12>, // 16 .. 64
    ringtail_pubkey:    array<u32, 8>,  // 64 .. 96
    mldsa_pubkey:       array<u32, 8>,  // 96 .. 128
    mldsa_groth16_root: array<u32, 8>,  // 128 .. 160
    kind:               u32,            // 160
    jail_until_epoch:   u32,            // 164
    epoch:              u32,            // 168
    _pad0:              u32,            // 172 -> 176
};

struct StakeOp {
    delegator_id_lo:        u32, // 0
    delegator_id_hi:        u32, // 4
    validator_id_lo:        u32, // 8
    validator_id_hi:        u32, // 12
    amount_lo:              u32, // 16
    amount_hi:              u32, // 20
    lock_until_epoch_lo:    u32, // 24
    lock_until_epoch_hi:    u32, // 28
    source_validator_id_lo: u32, // 32
    source_validator_id_hi: u32, // 36
    kind:                   u32, // 40
    commission_bps:         u32, // 44
    epoch:                  u32, // 48
    _pad0:                  u32, // 52
    _pad1_lo:               u32, // 56
    _pad1_hi:               u32, // 60 -> 64
};

struct PVMTransitionResult {
    status:                  u32, // 0
    validator_apply_count:   u32, // 4
    stake_apply_count:       u32, // 8
    slash_apply_count:       u32, // 12
    active_validator_count:  u32, // 16
    pending_drop_count:      u32, // 20
    jailed_count:            u32, // 24
    tombstoned_count:        u32, // 28
    total_active_stake_lo:   u32, // 32
    total_active_stake_hi:   u32, // 36
    total_slashed_lo:        u32, // 40
    total_slashed_hi:        u32, // 44
    total_rewards_lo:        u32, // 48
    total_rewards_hi:        u32, // 52
    epoch_lo:                u32, // 56
    epoch_hi:                u32, // 60
    validator_set_root:      array<u32, 8>, // 64 .. 96
    stake_root:              array<u32, 8>, // 96 .. 128
    slashing_root:           array<u32, 8>, // 128 .. 160
    epoch_root:              array<u32, 8>, // 160 .. 192
};

// =============================================================================
// Status / kind constants — match platformvm_gpu_layout.hpp.
// =============================================================================

const kStatusActive       : u32 = 0x1u;
const kStatusJailed       : u32 = 0x2u;
const kStatusTombstoned   : u32 = 0x4u;
const kStatusPendingAdd   : u32 = 0x8u;
const kStatusPendingDrop  : u32 = 0x10u;

const kStakeStatusActive    : u32 = 1u;
const kStakeStatusUnbonding : u32 = 2u;
const kStakeStatusRetired   : u32 = 3u;

const kVOpAdd          : u32 = 0u;
const kVOpRemove       : u32 = 1u;
const kVOpUpdateWeight : u32 = 2u;
const kVOpJail         : u32 = 3u;
const kVOpUnjail       : u32 = 4u;
const kVOpRotateKeys   : u32 = 5u;

const kSOpBond         : u32 = 0u;
const kSOpUnbond       : u32 = 1u;
const kSOpDelegate     : u32 = 2u;
const kSOpRedelegate   : u32 = 3u;
const kSOpReward       : u32 = 4u;
const kSOpCommission   : u32 = 5u;

const kEvEquivocation  : u32 = 0u;
const kEvDowntime      : u32 = 1u;
const kEvInvalidVote   : u32 = 2u;

// kRewardScale = 1e18 = 0x0DE0B6B3A7640000.
const kRewardScale_lo : u32 = 0xA7640000u;
const kRewardScale_hi : u32 = 0x0DE0B6B3u;

// =============================================================================
// 64-bit-as-(lo,hi) helpers — WGSL has no native u64.
// =============================================================================

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
fn u64_eq(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    return a_lo == b_lo && a_hi == b_hi;
}
fn u64_is_zero(lo: u32, hi: u32) -> bool { return lo == 0u && hi == 0u; }

fn u64_add(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let r_lo = a_lo + b_lo;
    var r_hi = a_hi + b_hi;
    if (r_lo < a_lo) { r_hi = r_hi + 1u; }
    return vec2<u32>(r_lo, r_hi);
}
fn u64_sub(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let r_lo = a_lo - b_lo;
    var r_hi = a_hi - b_hi;
    if (a_lo < b_lo) { r_hi = r_hi - 1u; }
    return vec2<u32>(r_lo, r_hi);
}
fn u64_sat_add(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let r = u64_add(a_lo, a_hi, b_lo, b_hi);
    // overflow if r < a (a_hi after add greater than original a_hi only if no overflow)
    if (u64_lt(r.x, r.y, a_lo, a_hi)) {
        return vec2<u32>(0xFFFFFFFFu, 0xFFFFFFFFu);
    }
    return r;
}
fn u64_sat_sub(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    if (u64_lt(a_lo, a_hi, b_lo, b_hi)) {
        return vec2<u32>(0u, 0u);
    }
    return u64_sub(a_lo, a_hi, b_lo, b_hi);
}

// 64x32 multiply (returns low 64 bits of a * b). Used for reward scaling.
// 32x32 -> 64 unsigned multiply, returned as (lo, hi).
// Split both inputs into 16-bit halves to avoid u32 multiply overflow.
fn u32_mul_full(a: u32, b: u32) -> vec2<u32> {
    let a_l = a & 0xFFFFu;
    let a_h = a >> 16u;
    let b_l = b & 0xFFFFu;
    let b_h = b >> 16u;
    let ll = a_l * b_l;             // 32 bits
    let lh = a_l * b_h;             // 32 bits
    let hl = a_h * b_l;             // 32 bits
    let hh = a_h * b_h;             // 32 bits
    let mid = (ll >> 16u) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
    let r_lo = (ll & 0xFFFFu) | (mid << 16u);
    let r_hi = hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
    return vec2<u32>(r_lo, r_hi);
}

// (a_hi*2^32 + a_lo) * b mod 2^64.
fn u64_mul_u32(a_lo: u32, a_hi: u32, b: u32) -> vec2<u32> {
    let p_lo  = u32_mul_full(a_lo, b);                    // a_lo * b, full 64
    let p_hi  = a_hi * b;                                  // low 32 of a_hi*b
    let r_lo  = p_lo.x;
    let r_hi  = p_lo.y + p_hi;
    return vec2<u32>(r_lo, r_hi);
}

// 64-bit by 64-bit unsigned multiply, returns low 64 bits.
fn u64_mul(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    // (a_hi*2^32 + a_lo) * (b_hi*2^32 + b_lo) mod 2^64
    // = a_lo*b_lo + (a_lo*b_hi + a_hi*b_lo)*2^32 (mod 2^64)
    let lo_lo = u32_mul_full(a_lo, b_lo);   // a_lo * b_lo, full 64
    let lo_hi = a_lo * b_hi;                 // low 32 of a_lo * b_hi
    let hi_lo = a_hi * b_lo;                 // low 32 of a_hi * b_lo
    let r_lo = lo_lo.x;
    let r_hi = lo_lo.y + lo_hi + hi_lo;
    return vec2<u32>(r_lo, r_hi);
}

// 64/32 unsigned divide. Used for reward per-unit scaling.
fn u64_div_u32(a_lo: u32, a_hi: u32, b: u32) -> vec2<u32> {
    if (b == 0u) { return vec2<u32>(0u, 0u); }
    // Long division on the 64-bit dividend using two halves.
    let q_hi = a_hi / b;
    let r_hi = a_hi - q_hi * b;
    // remainder so far = r_hi (32 bits) | a_lo (32 bits) — a 64-bit value <= b<<32.
    // Process the lower 32 bits 16 at a time to avoid 64-bit operations.
    var rem: u32 = r_hi;
    var q_lo: u32 = 0u;
    let a_lo_h = a_lo >> 16u;
    let a_lo_l = a_lo & 0xFFFFu;
    rem = (rem << 16u) | a_lo_h;
    let q1 = rem / b;
    rem = rem - q1 * b;
    rem = (rem << 16u) | a_lo_l;
    let q2 = rem / b;
    q_lo = (q1 << 16u) | q2;
    return vec2<u32>(q_lo, q_hi);
}

// 64/64 unsigned divide. Used when divisor is full u64. Implementation uses
// repeated subtraction by powers of two — fine because validator weights are
// modest and this only runs once per Reward op.
fn u64_div(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    if (u64_is_zero(b_lo, b_hi)) { return vec2<u32>(0u, 0u); }
    // Fast path: if divisor fits in 32 bits.
    if (b_hi == 0u) { return u64_div_u32(a_lo, a_hi, b_lo); }
    // Slow path via shift-subtract. Numerators here fit in 64 bits; loop ≤ 64.
    var q_lo: u32 = 0u;
    var q_hi: u32 = 0u;
    var r_lo: u32 = 0u;
    var r_hi: u32 = 0u;
    var n_lo: u32 = a_lo;
    var n_hi: u32 = a_hi;
    for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        // shift (r,n) left by 1 as a 128-bit unit
        let r_new_hi = (r_hi << 1u) | (r_lo >> 31u);
        let r_new_lo = (r_lo << 1u) | (n_hi >> 31u);
        let n_new_hi = (n_hi << 1u) | (n_lo >> 31u);
        let n_new_lo = (n_lo << 1u);
        r_hi = r_new_hi;
        r_lo = r_new_lo;
        n_hi = n_new_hi;
        n_lo = n_new_lo;
        // shift quotient
        q_hi = (q_hi << 1u) | (q_lo >> 31u);
        q_lo = q_lo << 1u;
        if (!u64_lt(r_lo, r_hi, b_lo, b_hi)) {
            let s = u64_sub(r_lo, r_hi, b_lo, b_hi);
            r_lo = s.x;
            r_hi = s.y;
            q_lo = q_lo | 1u;
        }
    }
    return vec2<u32>(q_lo, q_hi);
}

// =============================================================================
// keccak-f[1600] over a flat (lo,hi)-pair state (50 u32 = 25 lanes).
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

// rotate-left a 64-bit value (split as lo, hi) by n bits in [0, 64).
// Mirrors the masked rotl64 in the CPU/Metal/CUDA common headers.
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

// State is 50 u32: lane i is (state[2*i], state[2*i+1]) = (lo, hi).
fn keccak_f1600(state: ptr<function, array<u32, 50>>) {
    for (var round: u32 = 0u; round < 24u; round = round + 1u) {
        // theta
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
        // rho + pi
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
        // chi
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
        // iota
        (*state)[0] = (*state)[0] ^ KECCAK_RC_LO[round];
        (*state)[1] = (*state)[1] ^ KECCAK_RC_HI[round];
    }
}

// -- byte-level keccak256 with 0x01/0x80 padding (rate=136) --
//
// `data` is provided as a byte-addressable function-local array packed as
// u32s little-endian. We accept up to 192 bytes (48 u32) which covers the
// largest PVM leaf (validator leaf at 172 bytes).

fn keccak256_byte(data: ptr<function, array<u32, 48>>, len: u32, out: ptr<function, array<u32, 8>>) {
    var state: array<u32, 50>;
    for (var i: u32 = 0u; i < 50u; i = i + 1u) { state[i] = 0u; }
    let rate: u32 = 136u;
    var off: u32 = 0u;
    while (len - off >= rate) {
        for (var i: u32 = 0u; i < rate; i = i + 1u) {
            let byte_idx = off + i;
            let word_idx = byte_idx >> 2u;
            let shift    = (byte_idx & 3u) * 8u;
            let byte     = ((*data)[word_idx] >> shift) & 0xFFu;
            let lane     = i / 8u;
            let lane_sh  = (i & 7u) * 8u;
            if (lane_sh < 32u) {
                state[2u*lane] = state[2u*lane] ^ (byte << lane_sh);
            } else {
                state[2u*lane + 1u] = state[2u*lane + 1u] ^ (byte << (lane_sh - 32u));
            }
        }
        keccak_f1600(&state);
        off = off + rate;
    }
    // last (partial) block
    var block: array<u32, 34>; // 136 bytes / 4
    for (var i: u32 = 0u; i < 34u; i = i + 1u) { block[i] = 0u; }
    let rem = len - off;
    for (var i: u32 = 0u; i < rem; i = i + 1u) {
        let byte_idx = off + i;
        let word_idx = byte_idx >> 2u;
        let shift    = (byte_idx & 3u) * 8u;
        let byte     = ((*data)[word_idx] >> shift) & 0xFFu;
        let dst_word = i >> 2u;
        let dst_sh   = (i & 3u) * 8u;
        block[dst_word] = block[dst_word] | (byte << dst_sh);
    }
    // pad
    let pad_word_a = rem >> 2u;
    let pad_sh_a   = (rem & 3u) * 8u;
    block[pad_word_a] = block[pad_word_a] ^ (0x01u << pad_sh_a);
    let last_idx = rate - 1u;
    let pad_word_b = last_idx >> 2u;
    let pad_sh_b   = (last_idx & 3u) * 8u;
    block[pad_word_b] = block[pad_word_b] ^ (0x80u << pad_sh_b);
    // absorb
    for (var i: u32 = 0u; i < rate; i = i + 1u) {
        let word_idx = i >> 2u;
        let shift    = (i & 3u) * 8u;
        let byte     = (block[word_idx] >> shift) & 0xFFu;
        let lane     = i / 8u;
        let lane_sh  = (i & 7u) * 8u;
        if (lane_sh < 32u) {
            state[2u*lane] = state[2u*lane] ^ (byte << lane_sh);
        } else {
            state[2u*lane + 1u] = state[2u*lane + 1u] ^ (byte << (lane_sh - 32u));
        }
    }
    keccak_f1600(&state);
    // squeeze 32 bytes -> 8 u32 little-endian-byte-packed
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lane    = i / 2u;
        let half_hi = (i & 1u) == 1u;
        if (half_hi) {
            (*out)[i] = state[2u*lane + 1u];
        } else {
            (*out)[i] = state[2u*lane];
        }
    }
}

// Little-endian byte writes into the function-local data buffer.
fn absorb_u32(data: ptr<function, array<u32, 48>>, byte_off: u32, v: u32) {
    let word_idx = byte_off >> 2u;
    let shift    = (byte_off & 3u) * 8u;
    if (shift == 0u) {
        (*data)[word_idx] = v;
        return;
    }
    let lo_mask: u32 = (1u << shift) - 1u;
    (*data)[word_idx]      = ((*data)[word_idx]      & lo_mask) | (v << shift);
    (*data)[word_idx + 1u] = ((*data)[word_idx + 1u] & ~lo_mask) | (v >> (32u - shift));
}

fn absorb_u64(data: ptr<function, array<u32, 48>>, byte_off: u32, lo: u32, hi: u32) {
    absorb_u32(data, byte_off,      lo);
    absorb_u32(data, byte_off + 4u, hi);
}

fn absorb_digest32(data: ptr<function, array<u32, 48>>, byte_off: u32,
                   src: ptr<function, array<u32, 8>>) {
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        absorb_u32(data, byte_off + k*4u, (*src)[k]);
    }
}

fn absorb_key48(data: ptr<function, array<u32, 48>>, byte_off: u32,
                src: ptr<function, array<u32, 12>>) {
    for (var k: u32 = 0u; k < 12u; k = k + 1u) {
        absorb_u32(data, byte_off + k*4u, (*src)[k]);
    }
}

// =============================================================================
// Index-hash helpers — match platformvm_cpu_reference.cpp exactly.
// =============================================================================
//
// validator_index = uint32_t( ((0xcbf29ce484222325 ^ id) * 0x100000001b3) ) & mask
//
// 0x100000001b3 = (1 << 32) | 0x000001b3, so:
//   prod_low_64 = (xor * 0x000001b3) + (xor << 32)
// The (xor << 32) contribution is in bits 32+ — clipped from low 32.
// So low 32 = (xor_lo * 0x000001b3) (mod 2^32).
// FNV constant 0xcbf29ce484222325 has lo half 0x84222325.

fn validator_index_hash(id_lo: u32, id_hi: u32, mask: u32) -> u32 {
    let xor_lo = 0x84222325u ^ id_lo;
    let prod_lo = xor_lo * 0x000001b3u;
    return prod_lo & mask;
}

// stake_record_index_hash: composite = delegator ^ (validator + 0x9E3779B97F4A7C15
//                                                     + (delegator << 6) + (delegator >> 2))
// Then `(uint32_t)composite & mask`. The low 32 bits of:
//   (deleg << 6)   = deleg_lo << 6  (mod 2^32)
//   (deleg >> 2)   = (deleg_lo >> 2) | (deleg_hi << 30)
//   golden low     = 0x7F4A7C15

fn stake_record_index_hash(deleg_lo: u32, deleg_hi: u32,
                           val_lo: u32, val_hi: u32, mask: u32) -> u32 {
    let golden_lo: u32 = 0x7F4A7C15u;
    let mix_a_lo = deleg_lo << 6u;
    let mix_b_lo = (deleg_lo >> 2u) | (deleg_hi << 30u);
    let inner_lo = val_lo + golden_lo + mix_a_lo + mix_b_lo;
    let composite_lo = deleg_lo ^ inner_lo;
    return composite_lo & mask;
}

// =============================================================================
// validator_locate / stake_record_locate are defined per-kernel because WGSL
// disallows passing pointers-to-storage as function parameters. Each kernel
// declares its own storage bindings at module scope and defines locate
// helpers that reference those globals directly. The logic is identical
// across kernels (they share the index-hash above); duplication is the
// idiomatic WGSL pattern.
// =============================================================================
