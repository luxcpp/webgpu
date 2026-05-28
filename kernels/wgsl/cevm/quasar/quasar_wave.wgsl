// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// WGSL port of cuda/kernels/cevm/quasar/quasar_wave.cu and the matching
// metal/src/shaders/cevm/quasar/quasar_wave.metal. QuasarGPU wave-tick
// scheduler — persistent-CTA model, one workgroup per ServiceId, 32 threads
// each, only invocation 0 does the drain.
//
// Backend mapping:
//   __threadfence()                         → storageBarrier()
//   atomicAdd(p, 0u)  // CUDA atomic load   → atomicLoad(&p)
//   atomicAdd(p, n)                         → atomicAdd(&p, n)
//   atomicCAS(p, exp, new)                  → atomicCompareExchangeWeak(&p, exp, new)
//   atomicExch(p, n)                        → atomicExchange(&p, n)
//   threadIdx.x                             → local_invocation_id.x
//   blockIdx.x                              → workgroup_id.x
//   blockDim.x                              → fixed by @workgroup_size
//
// 64-bit emulation: WGSL has no native u64. Every uint64_t field is encoded
// as two consecutive u32 words `(lo, hi)`. All 64-bit reads/writes go
// through helpers below. Critically, the CUDA kernel never does a 64-bit
// atomic — every atomic is on a u32 counter (`head`, `tail`, `pushed`,
// `consumed`, `version`, `status`). The 64-bit fields are written under
// the protection of the ring CAS sequence, so plain stores suffice on the
// WGSL side too.
//
// Layout: each ring item lives in `items_arena: array<u32>` starting at the
// header's `items_ofs` word (u32-words, NOT bytes — host packs structs into
// u32 words for WGSL, byte arenas for CUDA/Metal). Struct widths in u32
// words are listed at the top of the file and host code packs accordingly.
//
// Scheduler shape: dispatch(kNumServices, 1, 1) workgroups of 32 threads.
// Only invocation 0 drains; lanes 1..31 are reserved for SIMT EVM ops in
// v0.42+.

// =============================================================================
// Layout constants — must match cuda/kernels/cevm/quasar/quasar_wave.cu and
// quasar_gpu_layout.hpp.
// =============================================================================

const kNumServices:       u32 = 17u;   // v0.44
const kMaxRWSetPerTx:     u32 = 8u;
const kMaxDagParents:     u32 = 4u;
const kMaxDagChildren:    u32 = 16u;
const kFiberStackDepth:   u32 = 64u;
const kFiberStackLimbs:   u32 = 4u;
const kFiberMemoryBytes:  u32 = 1024u;
const kFiberInstrBudget:  u32 = 100000u;

const kFiberReady:        u32 = 0u;
const kFiberRunning:      u32 = 1u;
const kFiberWaitingState: u32 = 2u;
const kFiberCommittable:  u32 = 3u;
const kFiberReverted:     u32 = 4u;

const kNeedsState: u32 = 0x80000000u;
const kNeedsExec:  u32 = 0x40000000u;
const kFlagMask:   u32 = 0xC0000000u;

const kMvccInvalidIdx: u32 = 0xFFFFFFFFu;

// Item strides in u32 words. Host packs structs into u32-words; these
// strides are the contract.
//
// RingHeader (12 u32): head, tail, capacity, mask, items_ofs_lo,
// items_ofs_hi, item_size, _pad0, pushed, consumed, _pad1, _pad2
const RING_HDR_WORDS: u32 = 12u;
const RH_HEAD:        u32 = 0u;
const RH_TAIL:        u32 = 1u;
const RH_CAPACITY:    u32 = 2u;
const RH_MASK:        u32 = 3u;
const RH_ITEMS_OFS_LO: u32 = 4u;
const RH_ITEMS_OFS_HI: u32 = 5u;
const RH_ITEM_SIZE:   u32 = 6u;
const RH_PUSHED:      u32 = 8u;
const RH_CONSUMED:    u32 = 9u;

// IngressTx (8 u32): blob_offset, blob_size, gas_limit_lo, gas_limit_hi,
// nonce, _pad0, origin_lo, origin_hi.
const INGRESS_TX_WORDS: u32 = 8u;

// DecodedTx (8 u32): tx_index, blob_offset, blob_size,
// gas_limit_lo, gas_limit_hi, nonce, origin_lo, origin_hi (status packed
// into upper bits of origin_hi via flag mask). One canonical layout:
// tx_index, blob_offset, blob_size, gas_limit_lo, gas_limit_hi, nonce,
// origin_lo, origin_hi_with_status.
const DECODED_TX_WORDS: u32 = 8u;

// VerifiedTx (8 u32): tx_index, admission, gas_limit_lo, gas_limit_hi,
// origin_lo, origin_hi, blob_offset, blob_size.
const VERIFIED_TX_WORDS: u32 = 8u;

// CommitItem (12 + 8 = 20 u32):
//   tx_index, status, gas_used_lo, gas_used_hi,
//   cumulative_gas_lo, cumulative_gas_hi, gas_refund_lo, gas_refund_hi,
//   receipt_hash[32 bytes = 8 u32]
const COMMIT_ITEM_WORDS: u32 = 20u;

// QuasarRoundDescriptor — first 24 u32 carry the scalars we actually read
// here (chain_id, round, base_fee, wave_tick_budget, mode, closing_flag,
// epoch). Full struct is wider but we only touch these fields.
const DESC_CHAIN_ID_LO:        u32 = 0u;
const DESC_CHAIN_ID_HI:        u32 = 1u;
const DESC_ROUND_LO:           u32 = 2u;
const DESC_ROUND_HI:           u32 = 3u;
const DESC_TS_NS_LO:           u32 = 4u;
const DESC_TS_NS_HI:           u32 = 5u;
const DESC_DEADLINE_LO:        u32 = 6u;
const DESC_DEADLINE_HI:        u32 = 7u;
const DESC_GAS_LIMIT_LO:       u32 = 8u;
const DESC_GAS_LIMIT_HI:       u32 = 9u;
const DESC_BASE_FEE_LO:        u32 = 10u;
const DESC_BASE_FEE_HI:        u32 = 11u;
const DESC_WAVE_TICK_BUDGET:   u32 = 12u;
const DESC_WAVE_TICK_INDEX:    u32 = 13u;
const DESC_CLOSING_FLAG:       u32 = 14u;
const DESC_MODE:               u32 = 15u;
// 32-byte parent_block_hash    16..23
// 32-byte parent_state_root    24..31
// 32-byte parent_execution_root 32..39
const DESC_EPOCH_LO:           u32 = 40u;
const DESC_EPOCH_HI:           u32 = 41u;

// QuasarRoundResult scalar offsets — atomics live here. Result is u32-aligned.
const RES_STATUS:               u32 = 0u;
const RES_TX_COUNT:             u32 = 1u;
const RES_GAS_USED_LO:          u32 = 2u;
const RES_GAS_USED_HI:          u32 = 3u;
const RES_WAVE_TICK_COUNT:      u32 = 4u;
const RES_CONFLICT_COUNT:       u32 = 5u;
const RES_REPAIR_COUNT:         u32 = 6u;
const RES_FIBERS_SUSPENDED:     u32 = 7u;
const RES_FIBERS_RESUMED:       u32 = 8u;
const RES_QUORUM_STATUS_BLS:    u32 = 9u;
const RES_QUORUM_STATUS_MLDSAG: u32 = 10u;
const RES_QUORUM_STATUS_RT:     u32 = 11u;
const RES_QUORUM_STAKE_BLS:     u32 = 12u;
const RES_QUORUM_STAKE_MLDSAG:  u32 = 13u;
const RES_QUORUM_STAKE_RT:      u32 = 14u;
const RES_MODE:                 u32 = 15u;
// 32-byte block_hash       16..23
// 32-byte state_root       24..31
// 32-byte receipts_root    32..39
// 32-byte execution_root   40..47
// 32-byte mode_root        48..55
const RES_RECEIPTS_ROOT_W0: u32 = 32u;
const RES_EXECUTION_ROOT_W0: u32 = 40u;
const RES_STATE_ROOT_W0:     u32 = 24u;
const RES_MODE_ROOT_W0:      u32 = 48u;
const RES_BLOCK_HASH_W0:     u32 = 16u;

// =============================================================================
// Bindings.
//
// Two contracts hold the world together:
//   * scheduler_state[] is atomic — all the u32 counters that workers race
//     on. Headers (ring counters) live here at known offsets. So do the
//     QuasarRoundResult atomics.
//   * scheduler_data[] holds the payload bytes — ring items, MVCC slots,
//     DAG nodes, fiber slots — without atomic semantics. Updates here are
//     ordered via the ring head/tail atomics in scheduler_state.
//
// Both are dynamic-length storage buffers. The host sizes them once per
// round.
// =============================================================================

@group(0) @binding(0) var<storage, read_write> sched_state: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> sched_data:  array<u32>;
@group(0) @binding(2) var<storage, read_write> desc:        array<u32>;
@group(0) @binding(3) var<storage, read_write> result:      array<atomic<u32>>;

// scratch_data carries the non-atomic side of result (the 5×32-byte hash
// roots that the receipt-chain keccak loop reads/writes). It aliases the
// same buffer as `result` on the host side; we keep two bindings so the
// WGSL type system can express "atomic u32" for the counters and "plain
// u32" for the hash bytes. Index space is the same as `result`.
@group(0) @binding(4) var<storage, read_write> result_hash: array<u32>;

// =============================================================================
// Ring header helpers. The CUDA kernel takes `RingHeader*`; in WGSL we pass
// a base index into `sched_state` (where the atomic counters live) and the
// items_ofs is read from sched_state directly (it's u32-stride packed by
// the host).
// =============================================================================

fn rh_base(service_id: u32) -> u32 {
    return service_id * RING_HDR_WORDS;
}

fn rh_capacity(hb: u32) -> u32 {
    return atomicLoad(&sched_state[hb + RH_CAPACITY]);
}

fn rh_mask(hb: u32) -> u32 {
    return atomicLoad(&sched_state[hb + RH_MASK]);
}

fn rh_items_ofs(hb: u32) -> u32 {
    // items_ofs is a u32-word offset into sched_data. Host packs it as the
    // low half of items_ofs (high half stays zero for sched_data sized
    // under 4GiB).
    return atomicLoad(&sched_state[hb + RH_ITEMS_OFS_LO]);
}

fn rh_item_size(hb: u32) -> u32 {
    return atomicLoad(&sched_state[hb + RH_ITEM_SIZE]);
}

// Ring counts the host polls.
fn rh_pushed(hb: u32) -> u32 {
    return atomicLoad(&sched_state[hb + RH_PUSHED]);
}

fn rh_consumed(hb: u32) -> u32 {
    return atomicLoad(&sched_state[hb + RH_CONSUMED]);
}

// =============================================================================
// ring_try_push / ring_try_pop — generic over item width via `words`.
//
// MSL/CUDA semantics: try to publish `words` u32s from `src_base` into the
// ring's items arena. The CUDA kernel uses a non-locking SPSC-ish queue
// where push/pop both retry under CAS. We retain that contract: head/tail
// are advanced with CAS to detect concurrent pushers/popppers; the body of
// the item is plain copy (no atomic per-word).
// =============================================================================

fn ring_try_push(hb: u32, src_base: u32) -> bool {
    let cap   = rh_capacity(hb);
    if (cap == 0u) { return false; }
    let mask  = rh_mask(hb);
    let words = rh_item_size(hb);
    let arena = rh_items_ofs(hb);

    // Snapshot head/tail. Retry CAS on tail.
    loop {
        let head = atomicLoad(&sched_state[hb + RH_HEAD]);
        let tail = atomicLoad(&sched_state[hb + RH_TAIL]);
        if (tail - head >= cap) { return false; }
        let slot = tail & mask;
        let dst_base = arena + slot * words;
        // Copy item body. WGSL has no per-word atomic write here because
        // the ring tail-CAS below is the publish barrier.
        for (var w: u32 = 0u; w < words; w = w + 1u) {
            sched_data[dst_base + w] = sched_data[src_base + w];
        }
        storageBarrier();
        // Publish: bump tail via CAS so concurrent pushers don't collide.
        let cas = atomicCompareExchangeWeak(&sched_state[hb + RH_TAIL], tail, tail + 1u);
        if (cas.exchanged) {
            atomicAdd(&sched_state[hb + RH_PUSHED], 1u);
            return true;
        }
        // Tail moved under us — retry.
    }
    // Unreachable — the only escapes from the loop above are explicit
    // returns. naga still requires a terminator on this code path.
    return false;
}

fn ring_try_pop(hb: u32, dst_base: u32) -> bool {
    let mask  = rh_mask(hb);
    let words = rh_item_size(hb);
    let arena = rh_items_ofs(hb);

    loop {
        let head = atomicLoad(&sched_state[hb + RH_HEAD]);
        let tail = atomicLoad(&sched_state[hb + RH_TAIL]);
        if (head >= tail) { return false; }
        let cas = atomicCompareExchangeWeak(&sched_state[hb + RH_HEAD], head, head + 1u);
        if (cas.exchanged) {
            storageBarrier();
            let slot = head & mask;
            let src_base = arena + slot * words;
            for (var w: u32 = 0u; w < words; w = w + 1u) {
                sched_data[dst_base + w] = sched_data[src_base + w];
            }
            atomicAdd(&sched_state[hb + RH_CONSUMED], 1u);
            return true;
        }
        // CAS lost — retry.
    }
    // Unreachable; satisfies naga's terminator check.
    return false;
}

// =============================================================================
// Scratch pads in sched_data. The 12 services each get a private scratch
// region used as the local copy of the popped/pushed item, so we don't
// thrash registers and keep the WGSL function signature simple.
//
// Layout: sched_data[scratch_base..scratch_base + 64 u32] per service.
// Host sizes sched_data with `scratch_words * kNumServices` of head room.
// =============================================================================

const SCRATCH_WORDS_PER_SERVICE: u32 = 64u;

fn scratch_base(service_id: u32, scratch_arena: u32) -> u32 {
    return scratch_arena + service_id * SCRATCH_WORDS_PER_SERVICE;
}

// =============================================================================
// keccak-f[1600]. Same constants and recipe as CUDA / Metal. State is 25
// u64 lanes; we store as 50 u32 (lo, hi pairs) in a workgroup-local array.
// =============================================================================

// Round constants (24 entries × 2 u32 = 48 u32). Each pair is (lo, hi).
const RC_W0: u32 = 0u;
// We embed the round-constant table inline as a function returning the
// (lo, hi) pair for a given round. WGSL has no module-scope arrays of u32
// pairs in the same shape as CUDA; this is bit-exact with the CUDA kernel.
fn keccak_rc_lo(r: u32) -> u32 {
    switch (r) {
        case 0u:  { return 0x00000001u; }
        case 1u:  { return 0x00008082u; }
        case 2u:  { return 0x0000808Au; }
        case 3u:  { return 0x80008000u; }
        case 4u:  { return 0x0000808Bu; }
        case 5u:  { return 0x80000001u; }
        case 6u:  { return 0x80008081u; }
        case 7u:  { return 0x00008009u; }
        case 8u:  { return 0x0000008Au; }
        case 9u:  { return 0x00000088u; }
        case 10u: { return 0x80008009u; }
        case 11u: { return 0x8000000Au; }
        case 12u: { return 0x8000808Bu; }
        case 13u: { return 0x0000008Bu; }
        case 14u: { return 0x00008089u; }
        case 15u: { return 0x00008003u; }
        case 16u: { return 0x00008002u; }
        case 17u: { return 0x00000080u; }
        case 18u: { return 0x0000800Au; }
        case 19u: { return 0x8000000Au; }
        case 20u: { return 0x80008081u; }
        case 21u: { return 0x00008080u; }
        case 22u: { return 0x80000001u; }
        case 23u: { return 0x80008008u; }
        default: { return 0u; }
    }
}

fn keccak_rc_hi(r: u32) -> u32 {
    switch (r) {
        case 0u:  { return 0x00000000u; }
        case 1u:  { return 0x00000000u; }
        case 2u:  { return 0x80000000u; }
        case 3u:  { return 0x80000000u; }
        case 4u:  { return 0x00000000u; }
        case 5u:  { return 0x00000000u; }
        case 6u:  { return 0x80000000u; }
        case 7u:  { return 0x80000000u; }
        case 8u:  { return 0x00000000u; }
        case 9u:  { return 0x00000000u; }
        case 10u: { return 0x00000000u; }
        case 11u: { return 0x00000000u; }
        case 12u: { return 0x00000000u; }
        case 13u: { return 0x80000000u; }
        case 14u: { return 0x80000000u; }
        case 15u: { return 0x80000000u; }
        case 16u: { return 0x80000000u; }
        case 17u: { return 0x80000000u; }
        case 18u: { return 0x00000000u; }
        case 19u: { return 0x80000000u; }
        case 20u: { return 0x80000000u; }
        case 21u: { return 0x80000000u; }
        case 22u: { return 0x00000000u; }
        case 23u: { return 0x80000000u; }
        default: { return 0u; }
    }
}

fn keccak_rot(i: u32) -> u32 {
    // ρ rotation offsets per lane (mod 64). Same table as CUDA.
    switch (i) {
        case 0u:  { return 0u;  } case 1u:  { return 1u;  } case 2u:  { return 62u; }
        case 3u:  { return 28u; } case 4u:  { return 27u; } case 5u:  { return 36u; }
        case 6u:  { return 44u; } case 7u:  { return 6u;  } case 8u:  { return 55u; }
        case 9u:  { return 20u; } case 10u: { return 3u;  } case 11u: { return 10u; }
        case 12u: { return 43u; } case 13u: { return 25u; } case 14u: { return 39u; }
        case 15u: { return 41u; } case 16u: { return 45u; } case 17u: { return 15u; }
        case 18u: { return 21u; } case 19u: { return 8u;  } case 20u: { return 18u; }
        case 21u: { return 2u;  } case 22u: { return 61u; } case 23u: { return 56u; }
        case 24u: { return 14u; }
        default: { return 0u; }
    }
}

// 64-bit rotate-left by n in [0, 63]. Inputs (a_lo, a_hi).
fn rotl64(a_lo: u32, a_hi: u32, n: u32) -> vec2<u32> {
    if (n == 0u) { return vec2<u32>(a_lo, a_hi); }
    if (n >= 32u) {
        let s = n - 32u;
        if (s == 0u) { return vec2<u32>(a_hi, a_lo); }
        let new_lo = (a_hi << s) | (a_lo >> (32u - s));
        let new_hi = (a_lo << s) | (a_hi >> (32u - s));
        return vec2<u32>(new_lo, new_hi);
    }
    let new_lo = (a_lo << n) | (a_hi >> (32u - n));
    let new_hi = (a_hi << n) | (a_lo >> (32u - n));
    return vec2<u32>(new_lo, new_hi);
}

// keccak state: 25 lanes × (lo, hi). Held in private memory of the calling
// invocation so multiple invocations don't race.
struct KState {
    lanes: array<vec2<u32>, 25>,
}

fn keccak_f1600(s: ptr<function, KState>) {
    for (var round: u32 = 0u; round < 24u; round = round + 1u) {
        // θ
        var c: array<vec2<u32>, 5>;
        for (var x: u32 = 0u; x < 5u; x = x + 1u) {
            let l0 = (*s).lanes[x];
            let l1 = (*s).lanes[x + 5u];
            let l2 = (*s).lanes[x + 10u];
            let l3 = (*s).lanes[x + 15u];
            let l4 = (*s).lanes[x + 20u];
            c[x] = vec2<u32>(
                l0.x ^ l1.x ^ l2.x ^ l3.x ^ l4.x,
                l0.y ^ l1.y ^ l2.y ^ l3.y ^ l4.y,
            );
        }
        var d: array<vec2<u32>, 5>;
        for (var x: u32 = 0u; x < 5u; x = x + 1u) {
            let cm = c[(x + 4u) % 5u];
            let cp = c[(x + 1u) % 5u];
            let cp_rot = rotl64(cp.x, cp.y, 1u);
            d[x] = vec2<u32>(cm.x ^ cp_rot.x, cm.y ^ cp_rot.y);
        }
        for (var y: u32 = 0u; y < 25u; y = y + 5u) {
            for (var x: u32 = 0u; x < 5u; x = x + 1u) {
                let cur = (*s).lanes[y + x];
                (*s).lanes[y + x] = vec2<u32>(cur.x ^ d[x].x, cur.y ^ d[x].y);
            }
        }
        // ρ + π
        var b: array<vec2<u32>, 25>;
        for (var y: u32 = 0u; y < 5u; y = y + 1u) {
            for (var x: u32 = 0u; x < 5u; x = x + 1u) {
                let i = x + 5u * y;
                let j = y + 5u * ((2u * x + 3u * y) % 5u);
                let li = (*s).lanes[i];
                b[j] = rotl64(li.x, li.y, keccak_rot(i));
            }
        }
        // χ
        for (var y: u32 = 0u; y < 25u; y = y + 5u) {
            let t0 = b[y + 0u];
            let t1 = b[y + 1u];
            let t2 = b[y + 2u];
            let t3 = b[y + 3u];
            let t4 = b[y + 4u];
            (*s).lanes[y + 0u] = vec2<u32>(t0.x ^ ((~t1.x) & t2.x), t0.y ^ ((~t1.y) & t2.y));
            (*s).lanes[y + 1u] = vec2<u32>(t1.x ^ ((~t2.x) & t3.x), t1.y ^ ((~t2.y) & t3.y));
            (*s).lanes[y + 2u] = vec2<u32>(t2.x ^ ((~t3.x) & t4.x), t2.y ^ ((~t3.y) & t4.y));
            (*s).lanes[y + 3u] = vec2<u32>(t3.x ^ ((~t4.x) & t0.x), t3.y ^ ((~t4.y) & t0.y));
            (*s).lanes[y + 4u] = vec2<u32>(t4.x ^ ((~t0.x) & t1.x), t4.y ^ ((~t0.y) & t1.y));
        }
        // ι
        let rc = vec2<u32>(keccak_rc_lo(round), keccak_rc_hi(round));
        let l0 = (*s).lanes[0];
        (*s).lanes[0] = vec2<u32>(l0.x ^ rc.x, l0.y ^ rc.y);
    }
}

// Ethereum keccak256 (rate=136, capacity=512, output=32, padding 0x01...0x80).
// Input lives in a private byte buffer of up to 144 bytes (the longest leaf
// in this kernel — see receipt_hash and finalize_block_hash).
//
// in_bytes[]: little-endian byte stream packed into u32 words (4 bytes per
// word). out_words[]: 8 u32 = 32 bytes, byte i is in word i/4 at bit (i%4)*8.
const KECCAK_RATE_BYTES: u32 = 136u;

fn keccak256_local(in_bytes: ptr<function, array<u32, 36>>, len_bytes: u32, out_words: ptr<function, array<u32, 8>>) {
    var st: KState;
    for (var i: u32 = 0u; i < 25u; i = i + 1u) { st.lanes[i] = vec2<u32>(0u, 0u); }

    // Process full 136-byte blocks. Input is short (≤144 bytes), so at
    // most one full block fits.
    var off: u32 = 0u;
    if (len_bytes >= KECCAK_RATE_BYTES) {
        for (var i: u32 = 0u; i < KECCAK_RATE_BYTES; i = i + 1u) {
            let byte = ((*in_bytes)[(off + i) / 4u] >> (((off + i) % 4u) * 8u)) & 0xFFu;
            let lane  = i / 8u;
            let shift = (i % 8u) * 8u;
            if (shift < 32u) {
                st.lanes[lane].x = st.lanes[lane].x ^ (byte << shift);
            } else {
                st.lanes[lane].y = st.lanes[lane].y ^ (byte << (shift - 32u));
            }
        }
        keccak_f1600(&st);
        off = off + KECCAK_RATE_BYTES;
    }

    // Trailing block: zero-pad to rate then XOR padding bytes.
    var block: array<u32, 36>;  // 144 bytes worth — covers any remainder + padding
    for (var i: u32 = 0u; i < 36u; i = i + 1u) { block[i] = 0u; }
    let rem = len_bytes - off;
    for (var i: u32 = 0u; i < rem; i = i + 1u) {
        let byte = ((*in_bytes)[(off + i) / 4u] >> (((off + i) % 4u) * 8u)) & 0xFFu;
        block[i / 4u] = block[i / 4u] | (byte << ((i % 4u) * 8u));
    }
    // Domain separator 0x01 at position `rem`.
    block[rem / 4u] = block[rem / 4u] | (0x01u << ((rem % 4u) * 8u));
    // Final 0x80 at rate-1.
    let last = KECCAK_RATE_BYTES - 1u;
    block[last / 4u] = block[last / 4u] | (0x80u << ((last % 4u) * 8u));

    for (var i: u32 = 0u; i < KECCAK_RATE_BYTES; i = i + 1u) {
        let byte = (block[i / 4u] >> ((i % 4u) * 8u)) & 0xFFu;
        let lane  = i / 8u;
        let shift = (i % 8u) * 8u;
        if (shift < 32u) {
            st.lanes[lane].x = st.lanes[lane].x ^ (byte << shift);
        } else {
            st.lanes[lane].y = st.lanes[lane].y ^ (byte << (shift - 32u));
        }
    }
    keccak_f1600(&st);

    // Squeeze 32 bytes → 8 u32 (little-endian byte stream).
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        let lane = i / 8u;
        let shift = (i % 8u) * 8u;
        var byte: u32;
        if (shift < 32u) {
            byte = (st.lanes[lane].x >> shift) & 0xFFu;
        } else {
            byte = (st.lanes[lane].y >> (shift - 32u)) & 0xFFu;
        }
        let word_idx = i / 4u;
        let word_shift = (i % 4u) * 8u;
        (*out_words)[word_idx] = (*out_words)[word_idx] | (byte << word_shift);
    }
}

// =============================================================================
// receipt_hash: leaf = (tx_index || origin_lo || origin_hi || gas_limit ||
//                       gas_used || round || chain_id || _pad[4])
// Same recipe as cuda/quasar_wave.cu::receipt_hash. 40 bytes total.
// Result is written to result_hash starting at out_base (8 u32 = 32 bytes).
// =============================================================================

fn receipt_hash(
    tx_index: u32, origin_lo: u32, origin_hi: u32,
    gas_limit_lo: u32, gas_limit_hi: u32,
    gas_used_lo: u32, gas_used_hi: u32,
    round_lo: u32, round_hi: u32,
    chain_id_lo: u32, chain_id_hi: u32,
    out_words: ptr<function, array<u32, 8>>,
) {
    var leaf: array<u32, 36>;
    for (var i: u32 = 0u; i < 36u; i = i + 1u) { leaf[i] = 0u; }
    // tx_index @ 0
    leaf[0] = tx_index;
    // origin_lo @ 4 (word index 1)
    leaf[1] = origin_lo;
    // origin_hi @ 8 (word 2)
    leaf[2] = origin_hi;
    // gas_limit (8 bytes) @ 12 (words 3, 4)
    leaf[3] = gas_limit_lo;
    leaf[4] = gas_limit_hi;
    // gas_used (note: CUDA only writes 4 bytes here — see receipt_hash in
    // quasar_wave.cu lines 497-498. Keep parity: only low 32 bits.)
    leaf[5] = gas_used_lo;
    // round low 32 bits @ 24 (word 6) — CUDA writes only 4 bytes
    leaf[6] = round_lo;
    // chain_id (8 bytes) @ 28 (words 7, 8)
    leaf[7] = chain_id_lo;
    leaf[8] = chain_id_hi;
    // 4 pad bytes @ 36 (already zero)
    keccak256_local(&leaf, 40u, out_words);
}

// =============================================================================
// 64-bit helpers — vec2<u32>(lo, hi). Inlined from common/u64_pair.wgsl.
// =============================================================================

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let s_lo = a.x + b.x;
    let c1: u32 = select(0u, 1u, s_lo < a.x);
    let s_hi = a.y + b.y + c1;
    return vec2<u32>(s_lo, s_hi);
}

// =============================================================================
// Service drains. Each drain handles a single service ID's wave-tick work.
//
// All service drains follow the same shape as the CUDA kernel: pop from
// input ring, transform, push to output ring. If the push fails (ring
// full), re-push the input and break the loop — matches CUDA semantics.
//
// The CUDA kernel passes typed item buffers; in WGSL we read raw u32 words
// out of sched_data. Each service uses its own scratch region for the
// popped input and pushed output (sequential, since we serialise drains
// per service anyway).
// =============================================================================

const IDX_INGRESS:   u32 = 0u;
const IDX_DECODE:    u32 = 1u;
const IDX_CRYPTO:    u32 = 2u;
const IDX_DAGREADY:  u32 = 3u;
const IDX_EXEC:      u32 = 4u;
const IDX_VALIDATE:  u32 = 5u;
const IDX_REPAIR:    u32 = 6u;
const IDX_COMMIT:    u32 = 7u;
const IDX_STATEREQ:  u32 = 8u;
const IDX_STATERESP: u32 = 9u;
const IDX_VOTE:      u32 = 10u;
const IDX_QC:        u32 = 11u;

fn drain_ingress(scratch: u32, tx_index_seq_idx: u32, budget: u32) {
    let in_hb  = rh_base(IDX_INGRESS);
    let out_hb = rh_base(IDX_DECODE);
    let work_in  = scratch;
    let work_out = scratch + INGRESS_TX_WORDS;

    for (var i: u32 = 0u; i < budget; i = i + 1u) {
        if (!ring_try_pop(in_hb, work_in)) { return; }

        // IngressTx layout (8 u32):
        //   [0] blob_offset
        //   [1] blob_size
        //   [2] gas_limit_lo
        //   [3] gas_limit_hi
        //   [4] nonce
        //   [5] _pad0
        //   [6] origin_lo
        //   [7] origin_hi

        // Allocate a new tx_index via atomic on sched_state[tx_index_seq_idx].
        let tx_idx = atomicAdd(&sched_state[tx_index_seq_idx], 1u);

        // DecodedTx layout (8 u32):
        //   [0] tx_index
        //   [1] blob_offset
        //   [2] blob_size
        //   [3] gas_limit_lo
        //   [4] gas_limit_hi
        //   [5] nonce
        //   [6] origin_lo
        //   [7] origin_hi (status flag in upper bits if set)
        sched_data[work_out + 0u] = tx_idx;
        sched_data[work_out + 1u] = sched_data[work_in + 0u];
        sched_data[work_out + 2u] = sched_data[work_in + 1u];
        sched_data[work_out + 3u] = sched_data[work_in + 2u];
        sched_data[work_out + 4u] = sched_data[work_in + 3u];
        sched_data[work_out + 5u] = sched_data[work_in + 4u];
        sched_data[work_out + 6u] = sched_data[work_in + 6u];
        sched_data[work_out + 7u] = sched_data[work_in + 7u];

        if (!ring_try_push(out_hb, work_out)) {
            // Re-push on failure to keep budget contract.
            _ = ring_try_push(in_hb, work_in);
            return;
        }
    }
}

fn drain_decode(scratch: u32, budget: u32) {
    let in_hb       = rh_base(IDX_DECODE);
    let crypto_hb   = rh_base(IDX_CRYPTO);
    let statereq_hb = rh_base(IDX_STATEREQ);

    let work_in       = scratch;
    let work_statereq = scratch + DECODED_TX_WORDS;
    let work_verified = work_statereq + 8u; // StateRequest is also 8 u32

    for (var i: u32 = 0u; i < budget; i = i + 1u) {
        if (!ring_try_pop(in_hb, work_in)) { return; }

        // DecodedTx layout — see above. Branch on the kNeedsState bit in
        // origin_hi.
        let origin_hi = sched_data[work_in + 7u];
        if ((origin_hi & kNeedsState) != 0u) {
            // StateRequest layout (8 u32):
            //   [0] tx_index
            //   [1] key_type
            //   [2] priority
            //   [3] _pad0
            //   [4] key_lo_lo
            //   [5] key_lo_hi
            //   [6] key_hi_lo
            //   [7] key_hi_hi
            let tx_index  = sched_data[work_in + 0u];
            let origin_lo = sched_data[work_in + 6u];
            sched_data[work_statereq + 0u] = tx_index;
            sched_data[work_statereq + 1u] = 0u;
            sched_data[work_statereq + 2u] = 0u;
            sched_data[work_statereq + 3u] = 0u;
            sched_data[work_statereq + 4u] = origin_lo;
            sched_data[work_statereq + 5u] = 0u;
            sched_data[work_statereq + 6u] = origin_hi & ~kFlagMask;
            sched_data[work_statereq + 7u] = 0u;
            if (!ring_try_push(statereq_hb, work_statereq)) {
                _ = ring_try_push(in_hb, work_in);
                return;
            }
            atomicAdd(&result[RES_FIBERS_SUSPENDED], 1u);
            continue;
        }

        // Repackage DecodedTx → VerifiedTx (also 8 u32).
        //   [0] tx_index
        //   [1] admission   (0 if status==0 else 1)
        //   [2] gas_limit_lo
        //   [3] gas_limit_hi
        //   [4] origin_lo
        //   [5] origin_hi
        //   [6] blob_offset
        //   [7] blob_size
        sched_data[work_verified + 0u] = sched_data[work_in + 0u];
        sched_data[work_verified + 1u] = 0u; // admission = 0 = accepted
        sched_data[work_verified + 2u] = sched_data[work_in + 3u];
        sched_data[work_verified + 3u] = sched_data[work_in + 4u];
        sched_data[work_verified + 4u] = sched_data[work_in + 6u];
        sched_data[work_verified + 5u] = sched_data[work_in + 7u];
        sched_data[work_verified + 6u] = sched_data[work_in + 1u];
        sched_data[work_verified + 7u] = sched_data[work_in + 2u];

        if (!ring_try_push(crypto_hb, work_verified)) {
            _ = ring_try_push(in_hb, work_in);
            return;
        }
    }
}

fn drain_crypto(scratch: u32, budget: u32) {
    let in_hb       = rh_base(IDX_CRYPTO);
    let commit_hb   = rh_base(IDX_COMMIT);
    let dagready_hb = rh_base(IDX_DAGREADY);
    let exec_hb     = rh_base(IDX_EXEC);

    let mode = desc[DESC_MODE];

    let work_in     = scratch;
    let work_out    = scratch + VERIFIED_TX_WORDS;

    for (var i: u32 = 0u; i < budget; i = i + 1u) {
        if (!ring_try_pop(in_hb, work_in)) { return; }
        let admission = sched_data[work_in + 1u];
        if (admission != 0u) { continue; }

        let origin_hi = sched_data[work_in + 5u];
        if ((origin_hi & kNeedsExec) != 0u) {
            let next_hb = select(exec_hb, dagready_hb, mode == 1u);
            if (!ring_try_push(next_hb, work_in)) {
                _ = ring_try_push(in_hb, work_in);
                return;
            }
            continue;
        }

        // CommitItem (20 u32):
        //   [0] tx_index
        //   [1] status
        //   [2..3] gas_used (21000)
        //   [4..5] cumulative_gas (0)
        //   [6..7] gas_refund (0)
        //   [8..11] _pad
        //   [12..19] receipt_hash (8 u32)
        sched_data[work_out + 0u] = sched_data[work_in + 0u];        // tx_index
        sched_data[work_out + 1u] = 1u;                              // status = success
        sched_data[work_out + 2u] = 21000u;
        sched_data[work_out + 3u] = 0u;
        sched_data[work_out + 4u] = 0u;
        sched_data[work_out + 5u] = 0u;
        sched_data[work_out + 6u] = 0u;
        sched_data[work_out + 7u] = 0u;
        sched_data[work_out + 8u]  = 0u;
        sched_data[work_out + 9u]  = 0u;
        sched_data[work_out + 10u] = 0u;
        sched_data[work_out + 11u] = 0u;

        var digest: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { digest[k] = 0u; }
        receipt_hash(
            sched_data[work_in + 0u],     // tx_index
            sched_data[work_in + 4u],     // origin_lo
            sched_data[work_in + 5u],     // origin_hi
            sched_data[work_in + 2u],     // gas_limit_lo
            sched_data[work_in + 3u],     // gas_limit_hi
            21000u, 0u,                   // gas_used
            desc[DESC_ROUND_LO], desc[DESC_ROUND_HI],
            desc[DESC_CHAIN_ID_LO], desc[DESC_CHAIN_ID_HI],
            &digest,
        );
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            sched_data[work_out + 12u + k] = digest[k];
        }

        if (!ring_try_push(commit_hb, work_out)) {
            _ = ring_try_push(in_hb, work_in);
            return;
        }
    }
}

fn drain_dagready(scratch: u32, budget: u32) {
    let in_hb  = rh_base(IDX_DAGREADY);
    let out_hb = rh_base(IDX_EXEC);
    let work   = scratch;

    for (var i: u32 = 0u; i < budget; i = i + 1u) {
        if (!ring_try_pop(in_hb, work)) { return; }
        if (!ring_try_push(out_hb, work)) {
            _ = ring_try_push(in_hb, work);
            return;
        }
    }
}

fn drain_commit(scratch: u32, budget: u32) {
    let in_hb = rh_base(IDX_COMMIT);
    let work  = scratch;

    for (var i: u32 = 0u; i < budget; i = i + 1u) {
        if (!ring_try_pop(in_hb, work)) { return; }
        atomicAdd(&result[RES_TX_COUNT], 1u);

        // gas accumulation. CUDA does carry handling via prev+gas_lo
        // wrap detection.
        let gas_lo = sched_data[work + 2u];
        let gas_hi = sched_data[work + 3u];
        let prev = atomicAdd(&result[RES_GAS_USED_LO], gas_lo);
        if (gas_hi != 0u) {
            atomicAdd(&result[RES_GAS_USED_HI], gas_hi);
        }
        if (prev + gas_lo < prev) {
            atomicAdd(&result[RES_GAS_USED_HI], 1u);
        }

        // receipts_root chain: H(running || receipt_hash). 64 bytes input,
        // 32 bytes output.
        var buf: array<u32, 36>;
        for (var k: u32 = 0u; k < 36u; k = k + 1u) { buf[k] = 0u; }
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { buf[k]      = result_hash[RES_RECEIPTS_ROOT_W0 + k]; }
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { buf[8u + k] = sched_data[work + 12u + k]; }
        var nxt: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { nxt[k] = 0u; }
        keccak256_local(&buf, 64u, &nxt);
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            result_hash[RES_RECEIPTS_ROOT_W0 + k] = nxt[k];
        }

        // execution_root chain. Layout: running_root (32) ||
        // tx_index (4) || status (4) || gas_used (8) || first 20 bytes of receipt_hash.
        var erbuf: array<u32, 36>;
        for (var k: u32 = 0u; k < 36u; k = k + 1u) { erbuf[k] = 0u; }
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { erbuf[k] = result_hash[RES_EXECUTION_ROOT_W0 + k]; }
        // tx_index at byte 32 (word 8)
        erbuf[8] = sched_data[work + 0u];
        // status at byte 36 (word 9)
        erbuf[9] = sched_data[work + 1u];
        // gas_used (8 bytes) at byte 40 (words 10, 11)
        erbuf[10] = sched_data[work + 2u];
        erbuf[11] = sched_data[work + 3u];
        // First 20 bytes of receipt_hash at byte 48 (words 12..16, only 5 words = 20 bytes)
        for (var k: u32 = 0u; k < 5u; k = k + 1u) {
            erbuf[12u + k] = sched_data[work + 12u + k];
        }
        var ernxt: array<u32, 8>;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { ernxt[k] = 0u; }
        keccak256_local(&erbuf, 64u, &ernxt);
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            result_hash[RES_EXECUTION_ROOT_W0 + k] = ernxt[k];
        }
    }
}

// =============================================================================
// Main scheduler kernel.
//
// dispatch(kNumServices, 1, 1) workgroups × workgroup_size(32). Only
// invocation 0 of each workgroup drains; lanes 1..31 are reserved.
//
// gid (workgroup_id.x) ∈ [0, kNumServices) selects which service to drain.
// The CUDA kernel hard-codes the gid→drain mapping; we mirror it here.
// =============================================================================

@compute @workgroup_size(32)
fn quasar_wave_kernel(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let tid = lid.x;
    let gid = wid.x;
    if (tid != 0u) { return; }
    if (gid >= kNumServices) { return; }

    if (gid == 0u) {
        atomicAdd(&result[RES_WAVE_TICK_COUNT], 1u);
    }

    let raw_budget = desc[DESC_WAVE_TICK_BUDGET];
    let budget: u32 = max(64u, raw_budget);

    // Scratch arena starts after all ring headers. Host packs:
    //   sched_data[0..rings_data_end)  = ring item arenas
    //   sched_data[rings_data_end..)   = scratch (kNumServices × SCRATCH_WORDS_PER_SERVICE)
    // Host writes scratch_arena into sched_state at a fixed offset past
    // the last ring header. We read that offset.
    let scratch_arena_idx_in_state = kNumServices * RING_HDR_WORDS;
    let scratch_arena_word_idx     = atomicLoad(&sched_state[scratch_arena_idx_in_state]);
    let tx_seq_idx_in_state        = scratch_arena_idx_in_state + 1u;

    let scratch = scratch_base(gid, scratch_arena_word_idx);

    if (gid == IDX_INGRESS) {
        drain_ingress(scratch, tx_seq_idx_in_state, budget);
    } else if (gid == IDX_DECODE) {
        drain_decode(scratch, budget);
    } else if (gid == IDX_CRYPTO) {
        drain_crypto(scratch, budget);
    } else if (gid == IDX_DAGREADY) {
        drain_dagready(scratch, budget);
    } else if (gid == IDX_COMMIT) {
        drain_commit(scratch, budget);
    }
    // Other services (EXEC, VALIDATE, REPAIR, STATEREQ, STATERESP, VOTE,
    // QC) are coupled to MVCC slots / DAG state that lives outside the
    // wave-tick scope on the WGSL path. The CUDA kernel handles them
    // inline; we surface their absence to the host by leaving their drain
    // counters untouched. The host driver retains the CUDA/Metal path as
    // canonical for those services; the WGSL backend covers the four
    // services with no cross-service shared mutable state.
    //
    // Round finalization (closing_flag) — invocation 0 of workgroup 0.
    if (gid == 0u && desc[DESC_CLOSING_FLAG] != 0u) {
        let ingress_hb  = rh_base(IDX_INGRESS);
        let commit_hb   = rh_base(IDX_COMMIT);
        let pushed   = rh_pushed(ingress_hb);
        let consumed = rh_consumed(commit_hb);
        if (pushed == consumed) {
            // block_hash = keccak256(round (8) || mode (4) ||
            //   receipts_root (32) || execution_root (32) ||
            //   state_root (32) || mode_root (32)) = 140 bytes.
            var hdr: array<u32, 36>;
            for (var k: u32 = 0u; k < 36u; k = k + 1u) { hdr[k] = 0u; }
            // round (8 bytes)
            hdr[0] = desc[DESC_ROUND_LO];
            hdr[1] = desc[DESC_ROUND_HI];
            // mode (4 bytes)
            hdr[2] = desc[DESC_MODE];
            // 32-byte hash roots starting at byte 12 (word 3)
            for (var k: u32 = 0u; k < 8u; k = k + 1u) { hdr[3u + k]  = result_hash[RES_RECEIPTS_ROOT_W0 + k]; }
            for (var k: u32 = 0u; k < 8u; k = k + 1u) { hdr[11u + k] = result_hash[RES_EXECUTION_ROOT_W0 + k]; }
            for (var k: u32 = 0u; k < 8u; k = k + 1u) { hdr[19u + k] = result_hash[RES_STATE_ROOT_W0 + k]; }
            for (var k: u32 = 0u; k < 8u; k = k + 1u) { hdr[27u + k] = result_hash[RES_MODE_ROOT_W0 + k]; }
            var bh: array<u32, 8>;
            for (var k: u32 = 0u; k < 8u; k = k + 1u) { bh[k] = 0u; }
            keccak256_local(&hdr, 140u, &bh);
            for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                result_hash[RES_BLOCK_HASH_W0 + k] = bh[k];
                result_hash[RES_MODE_ROOT_W0 + k]  = bh[k];
            }
            atomicExchange(&result[RES_STATUS], 1u);
        }
    }
}
