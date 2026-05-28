// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_kernels_common.wgsl — shared WGSL device code for the four AIVM kernels.
//
// WGSL constraints relative to Metal/CUDA:
//   * no #include — this file is concatenated into each kernel module by
//     the host engine before submission to wgpuDeviceCreateShaderModule.
//   * structs use u32/atomic<u32>/array<u32>; we manually pack/unpack 32-byte
//     digests as 8 x u32 to match aivm_gpu_layout.hpp byte-for-byte.
//   * no 64-bit native types — split u64 into (lo:u32, hi:u32) lanes for
//     hash absorbtion. Layout sizes (144/160/128/96/192) still match.
//
// Layout encoding contract:
//   * a 32-byte digest D is stored as 8 little-endian u32s D[0..7], where
//     D[0] = D_bytes[0..3], D[1] = D_bytes[4..7], etc.
//   * a 48-byte key is 12 u32s.
//   * a 20-byte addr is 5 u32s.
//   * a u64 is two consecutive u32s (lo, hi).
//
// This matches a memcpy of the host struct because all fields are aligned
// to 4 bytes and the natural little-endian byte order on every backend.

struct AIVMRoundDescriptor {
    chain_id_lo:           u32,  // offset 0
    chain_id_hi:           u32,  // offset 4
    round_lo:              u32,  // 8
    round_hi:              u32,  // 12
    timestamp_ns_lo:       u32,  // 16
    timestamp_ns_hi:       u32,  // 20
    epoch_lo:              u32,  // 24
    epoch_hi:              u32,  // 28
    mode:                  u32,  // 32
    attestation_op_count:  u32,  // 36
    model_op_count:        u32,  // 40
    anchor_op_count:       u32,  // 44
    closing_flag:          u32,  // 48
    _pad0:                 u32,  // 52
    _pad1_lo:              u32,  // 56
    _pad1_hi:              u32,  // 60
    parent_aivm_root:      array<u32, 8>,  // 64..96
};

// -- ops --

struct AttestationOp {
    tee_quote_digest: array<u32, 8>,   // 0..32
    measurement:      array<u32, 8>,   // 32..64
    attesting_key:    array<u32, 12>,  // 64..112
    expiry_ns_lo:     u32,             // 112
    expiry_ns_hi:     u32,             // 116
    kind:             u32,             // 120
    evidence_offset:  u32,             // 124
    evidence_len:     u32,             // 128
    epoch:            u32,             // 132
    _pad0:            u32,             // 136
    _pad1:            u32,             // 140 -> 144
};

struct ModelOp {
    model_root:       array<u32, 8>,   // 0..32
    weight_hash:      array<u32, 8>,   // 32..64
    license_root:     array<u32, 8>,   // 64..96
    owner_addr:       array<u32, 5>,   // 96..116
    _pad0:            u32,             // 116
    parameter_count_lo: u32,           // 120
    parameter_count_hi: u32,           // 124
    modality:         u32,             // 128
    kind:             u32,             // 132
    epoch:            u32,             // 136
    _pad1:            u32,             // 140
    _pad2_lo:         u32,             // 144
    _pad2_hi:         u32,             // 148
    _pad3_lo:         u32,             // 152
    _pad3_hi:         u32,             // 156 -> 160
};

struct AnchorOp {
    commit_root:                  array<u32, 8>,  // 0..32
    parent_root:                  array<u32, 8>,  // 32..64
    validator_set_root_at_commit: array<u32, 8>,  // 64..96
    height_lo:                    u32,            // 96
    height_hi:                    u32,            // 100
    timestamp_ns_lo:              u32,            // 104
    timestamp_ns_hi:              u32,            // 108
    epoch:                        u32,            // 112
    _pad0:                        u32,            // 116
    _pad1_lo:                     u32,            // 120
    _pad1_hi:                     u32,            // 124 -> 128
};

// -- arenas --

struct Attestation {
    tee_quote_digest: array<u32, 8>,
    measurement:      array<u32, 8>,
    attesting_key:    array<u32, 12>,
    expiry_ns_lo:     u32,
    expiry_ns_hi:     u32,
    kind:             u32,
    evidence_offset:  u32,
    evidence_len:     u32,
    status:           u32,
    occupied:         u32,
    _pad0:            u32,
};

struct ModelRegistryEntry {
    model_root:       array<u32, 8>,
    weight_hash:      array<u32, 8>,
    license_root:     array<u32, 8>,
    owner_addr:       array<u32, 5>,
    _pad0:            u32,
    version_lo:       u32,
    version_hi:       u32,
    parameter_count_lo: u32,
    parameter_count_hi: u32,
    modality:         u32,
    occupied:         u32,
    _pad1_lo:         u32,
    _pad1_hi:         u32,
    _pad2_lo:         u32,
    _pad2_hi:         u32,
};

struct AuditAnchor {
    commit_root:                  array<u32, 8>,
    parent_root:                  array<u32, 8>,
    validator_set_root_at_commit: array<u32, 8>,
    height_lo:                    u32,
    height_hi:                    u32,
    timestamp_ns_lo:              u32,
    timestamp_ns_hi:              u32,
    occupied:                     u32,
    _pad0:                        u32,
    _pad1_lo:                     u32,
    _pad1_hi:                     u32,
};

struct AIVMEpochState {
    current_epoch_lo:                u32,
    current_epoch_hi:                u32,
    next_epoch_height_lo:            u32,
    next_epoch_height_hi:            u32,
    total_active_attestations_lo:    u32,
    total_active_attestations_hi:    u32,
    active_model_count:              u32,
    expired_attestation_count:       u32,
    attestation_root:                array<u32, 8>,
    model_registry_root:             array<u32, 8>,
    audit_root:                      array<u32, 8>,
    aivm_state_root:                 array<u32, 8>,
};

struct AIVMTransitionResult {
    status:                  u32,
    attestation_apply_count: u32,
    model_apply_count:       u32,
    anchor_apply_count:      u32,
    active_attestations:     u32,
    expired_attestations:    u32,
    model_count:             u32,
    anchor_count:            u32,
    epoch_lo:                u32,
    epoch_hi:                u32,
    total_models_lo:         u32,
    total_models_hi:         u32,
    total_anchors_lo:        u32,
    total_anchors_hi:        u32,
    _pad0_lo:                u32,
    _pad0_hi:                u32,
    attestation_root:        array<u32, 8>,
    model_registry_root:     array<u32, 8>,
    audit_root:              array<u32, 8>,
    aivm_state_root:         array<u32, 8>,
};

// =============================================================================
// Constants
// =============================================================================

const kAttStatusVerified : u32 = 2u;
const kAttStatusExpired  : u32 = 4u;

const kModelOpRegister      : u32 = 0u;
const kModelOpUpdateWeights : u32 = 1u;
const kModelOpUpdateLicense : u32 = 2u;
const kModelOpTransfer      : u32 = 3u;

// =============================================================================
// 64-bit-as-(lo,hi) helpers — WGSL has no native u64.
// =============================================================================

fn u64_le(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    if (a_hi < b_hi) { return true; }
    if (a_hi > b_hi) { return false; }
    return a_lo <= b_lo;
}
fn u64_lt(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    if (a_hi < b_hi) { return true; }
    if (a_hi > b_hi) { return false; }
    return a_lo < b_lo;
}
fn u64_eq(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    return a_lo == b_lo && a_hi == b_hi;
}
fn u64_is_zero(a_lo: u32, a_hi: u32) -> bool { return a_lo == 0u && a_hi == 0u; }

// =============================================================================
// keccak-f[1600] over a flat ulong-pair state (50 u32 = 25 lanes of 64 bits)
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
// `data` is provided as a byte-addressable function-local array; we accept
// a maximum of 384 bytes which is sufficient for any AIVM leaf (largest is
// the attestation leaf at 32+32+48+32 = 144 bytes; the composed root is
// 32*4 + 8 + 4*3 = 152 bytes).

fn keccak256_byte(data: ptr<function, array<u32, 96>>, len: u32, out: ptr<function, array<u32, 8>>) {
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
            // state[2*lane | (lane_sh<32 ? 0 : 1)] xor= byte << (lane_sh%32)
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
    var block: array<u32, 34>;  // 136 bytes / 4
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
    // pad: byte at index `rem` ^= 0x01; byte at index 135 ^= 0x80
    let pad_word_a = rem >> 2u;
    let pad_sh_a   = (rem & 3u) * 8u;
    block[pad_word_a] = block[pad_word_a] ^ (0x01u << pad_sh_a);
    let last_idx = rate - 1u;  // 135
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

fn absorb_u32(data: ptr<function, array<u32, 96>>, byte_off: u32, v: u32) {
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

fn absorb_u64(data: ptr<function, array<u32, 96>>, byte_off: u32, lo: u32, hi: u32) {
    absorb_u32(data, byte_off,      lo);
    absorb_u32(data, byte_off + 4u, hi);
}

// 32-byte digest absorbtion (8 u32s).
fn absorb_digest(data: ptr<function, array<u32, 96>>, byte_off: u32, src: ptr<function, array<u32, 8>>) {
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        absorb_u32(data, byte_off + k*4u, (*src)[k]);
    }
}

fn absorb_key48(data: ptr<function, array<u32, 96>>, byte_off: u32, src: ptr<function, array<u32, 12>>) {
    for (var k: u32 = 0u; k < 12u; k = k + 1u) {
        absorb_u32(data, byte_off + k*4u, (*src)[k]);
    }
}

fn absorb_addr20(data: ptr<function, array<u32, 96>>, byte_off: u32, src: ptr<function, array<u32, 5>>) {
    for (var k: u32 = 0u; k < 5u; k = k + 1u) {
        absorb_u32(data, byte_off + k*4u, (*src)[k]);
    }
}

// =============================================================================
// Open-addressing helpers (parallel to CPU/Metal/CUDA)
// =============================================================================

fn key_from_digest(d: ptr<function, array<u32, 8>>) -> vec2<u32> {
    // first 8 bytes -> u64 little-endian = (D[0], D[1])
    return vec2<u32>((*d)[0], (*d)[1]);
}

fn hash_index(klo: u32, khi: u32, mask: u32) -> u32 {
    // FNV-1a-style: h = (h ^ k) * c — only the low 32 bits matter for the mask.
    // To match CPU/Metal/CUDA: h_init = 0xcbf29ce484222325; mul=0x100000001b3.
    // Both are 64-bit constants. We compute the low 32 bits of:
    //   h = (h ^ k) * 0x100000001b3
    // The xor produces 64-bit result lo = h_lo ^ klo. Multiplying by
    // 0x100000001b3 = (0x1_00000001_b3) — its low word is 0x000001b3, high
    // word is 0x00000001. Low-32 of the product is:
    //   (xor_lo * 0x000001b3) + (xor_hi * 0x00000000)*2^32 ... clipped low.
    // i.e. low 32 = xor_lo * 0x000001b3 (mod 2^32) + (xor_hi * 1) << 0
    // Wait — the multiplier's high word is 1, so contribution is xor_lo * 1 << 32
    // which only affects the high half. Net: low32 = xor_lo * 0x000001b3.
    // The CPU side computes the same via 64-bit truncation, so this matches.
    let xor_lo = 0x84222325u ^ klo;
    let prod_lo = xor_lo * 0x000001b3u;
    return prod_lo & mask;
}

// Note: WGSL forbids passing storage pointers to functions, so storage-vs-
// function digest comparisons are inlined at the call site in each kernel.

fn digest_zero_8(d: ptr<function, array<u32, 8>>) -> bool {
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        if ((*d)[k] != 0u) { return false; }
    }
    return true;
}
fn key48_zero_12(d: ptr<function, array<u32, 12>>) -> bool {
    for (var k: u32 = 0u; k < 12u; k = k + 1u) {
        if ((*d)[k] != 0u) { return false; }
    }
    return true;
}
