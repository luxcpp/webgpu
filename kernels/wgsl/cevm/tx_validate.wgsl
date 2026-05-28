// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// WGSL port of cevm/lib/evm/gpu/cuda/tx_validate.cu. One thread per
// transaction. Each thread:
//   1. Reads the packed DeviceTxInput at index tid.
//   2. Looks up the sender account in a GPU-resident open-addressing hash
//      table (FNV-1a of the 20-byte address + linear probe, capacity 16384).
//   3. Checks: sender non-zero, nonce equality, intrinsic gas, balance with
//      EIP-3860 + overflow guards.
//   4. Writes valid_flags[tid] (0/1) and error_codes[tid] (bitfield).
//
// The 64-bit fields (gas_limit, value, nonce, gas_price, balance) are
// packed as vec2<u32>(lo, hi) into the struct so WGSL can read them as
// pairs. Compare/add helpers operate on those pairs.
//
// Layout invariants — must match CUDA + Metal sources byte-for-byte:
//   sizeof(DeviceTxInput)  = 80 bytes
//   sizeof(DeviceAccount)  = 40 bytes
//   ACCOUNT_TABLE_SIZE     = 16384 (power of two)
//
// Test parity: every input must produce the same valid_flags and error
// bitfield as cuda::evm_cuda_tx_validate_launch and the Metal
// counterpart in metal/tx_validate.metal.

// =============================================================================
// Constants
// =============================================================================

const ACCOUNT_TABLE_SIZE: u32 = 16384u;
const ACCOUNT_TABLE_MASK: u32 = 16383u;

const ERR_NONE:         u32 = 0u;
const ERR_NONCE_LOW:    u32 = 1u;
const ERR_NONCE_HIGH:   u32 = 2u;
const ERR_BALANCE_LOW:  u32 = 4u;
const ERR_GAS_LOW:      u32 = 8u;
const ERR_SENDER_ZERO:  u32 = 16u;
const ERR_GAS_OVERFLOW: u32 = 32u;

const MAX_INITCODE_SIZE: u32 = 49152u;
const GAS_BASE_TX:       u32 = 21000u;
const GAS_BASE_CREATE:   u32 = 53000u;
const GAS_PER_CALLDATA:  u32 = 16u;

// =============================================================================
// Packed structs (4-byte aligned to satisfy WGSL storage-buffer alignment).
// 64-bit values are vec2<u32>(lo, hi). 20-byte addresses are packed as
// 5 u32 words (LE). 80-byte struct => 20 u32 words. Storage stride per tx is
// always sizeof / 4 u32s.
//
// One transaction record (20 u32 words):
//   [0..4]    from[20]            // 5 words
//   [5..9]    to[20]              // 5 words
//   [10..11]  gas_limit  (lo, hi)
//   [12..13]  value      (lo, hi)
//   [14..15]  nonce      (lo, hi)
//   [16..17]  gas_price  (lo, hi)
//   [18]      calldata_size
//   [19]      is_create
//
// One account record (10 u32 words):
//   [0..4]    address[20]
//   [5]       occupied
//   [6..7]    nonce      (lo, hi)
//   [8..9]    balance    (lo, hi)
// =============================================================================

const TX_STRIDE_U32:   u32 = 20u;
const ACCT_STRIDE_U32: u32 = 10u;

// =============================================================================
// Bindings
// =============================================================================

@group(0) @binding(0) var<storage, read>       txs:          array<u32>;
@group(0) @binding(1) var<storage, read>       state:        array<u32>;
@group(0) @binding(2) var<storage, read_write> valid_flags:  array<u32>;
@group(0) @binding(3) var<storage, read_write> error_codes:  array<u32>;
@group(0) @binding(4) var<uniform>             num_txs:      u32;

// =============================================================================
// 64-bit helpers — vec2<u32>(lo, hi). All comparisons unsigned.
// =============================================================================

// a < b
fn u64_lt(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    if (a_hi < b_hi) { return true; }
    if (a_hi > b_hi) { return false; }
    return a_lo < b_lo;
}

// a > b
fn u64_gt(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    return u64_lt(b_lo, b_hi, a_lo, a_hi);
}

// a == b
fn u64_eq(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    return a_lo == b_lo && a_hi == b_hi;
}

// a + b, carrying lo→hi. Returns (sum_lo, sum_hi). Caller checks for
// overflow separately via u64_add_overflow.
fn u64_add(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let sum_lo = a_lo + b_lo;
    let carry: u32 = select(0u, 1u, sum_lo < a_lo);
    let sum_hi = a_hi + b_hi + carry;
    return vec2<u32>(sum_lo, sum_hi);
}

// Returns true iff a + b overflows 64 bits.
fn u64_add_overflow(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    let sum_lo = a_lo + b_lo;
    let carry: u32 = select(0u, 1u, sum_lo < a_lo);
    // hi half overflows iff (a_hi + b_hi + carry) wraps.
    let hi_sum  = a_hi + b_hi;
    if (hi_sum < a_hi) { return true; }
    let hi_sum2 = hi_sum + carry;
    return hi_sum2 < hi_sum;
}

// 64-bit unsigned multiplication of two u64 values represented as
// vec2<u32>(lo, hi). Returns the low 64 bits as vec2<u32> and an overflow
// flag. Uses 32×32→64 partial products.
struct Mul64Out {
    lo:       vec2<u32>,
    overflow: bool,
}

fn u32_mul_full(x: u32, y: u32) -> vec2<u32> {
    // 32×32 → 64 via 16-bit halves.
    let xlo = x & 0xFFFFu;
    let xhi = x >> 16u;
    let ylo = y & 0xFFFFu;
    let yhi = y >> 16u;

    let ll = xlo * ylo;
    let lh = xlo * yhi;
    let hl = xhi * ylo;
    let hh = xhi * yhi;

    let mid = lh + hl;
    let mid_carry: u32 = select(0u, 1u << 16u, mid < lh);
    let lo_part = ll + (mid << 16u);
    let lo_carry: u32 = select(0u, 1u, lo_part < ll);
    let hi_part = hh + (mid >> 16u) + mid_carry + lo_carry;
    return vec2<u32>(lo_part, hi_part);
}

fn u64_mul(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> Mul64Out {
    // (a_hi:a_lo) * (b_hi:b_lo) = a_lo*b_lo + ((a_lo*b_hi + a_hi*b_lo) << 32)
    // + (a_hi*b_hi << 64). The last term overflows the 64-bit result.
    let ll = u32_mul_full(a_lo, b_lo);          // low 64 contribution
    let lh = u32_mul_full(a_lo, b_hi);          // contributes to hi half of low 64
    let hl = u32_mul_full(a_hi, b_lo);          // contributes to hi half of low 64
    let hh = u32_mul_full(a_hi, b_hi);          // pure overflow

    // Sum hi halves: ll.hi + lh.lo + hl.lo. Carries propagate into the
    // overflow indicator.
    var hi_sum: u32 = ll.y + lh.x;
    var carry: u32  = select(0u, 1u, hi_sum < ll.y);
    let hi_sum2     = hi_sum + hl.x;
    carry           = carry + select(0u, 1u, hi_sum2 < hi_sum);

    // Anything spilling out of the 64-bit low part (lh.y, hl.y, hh.x/y, carry)
    // is overflow.
    let ov = (lh.y != 0u) || (hl.y != 0u) || (hh.x != 0u) || (hh.y != 0u) || (carry != 0u);

    var out: Mul64Out;
    out.lo       = vec2<u32>(ll.x, hi_sum2);
    out.overflow = ov;
    return out;
}

// =============================================================================
// Address helpers (20 bytes → 5 u32 words).
// =============================================================================

fn addr_is_zero(base: u32) -> bool {
    // 5 words cover 20 bytes — but the last word holds 4 bytes too; check
    // all of them.
    if (txs[base + 0u] != 0u) { return false; }
    if (txs[base + 1u] != 0u) { return false; }
    if (txs[base + 2u] != 0u) { return false; }
    if (txs[base + 3u] != 0u) { return false; }
    if (txs[base + 4u] != 0u) { return false; }
    return true;
}

fn addr_word(tx_base: u32, word: u32) -> u32 {
    return txs[tx_base + word];
}

fn state_addr_word(slot_base: u32, word: u32) -> u32 {
    return state[slot_base + word];
}

fn addr_eq(tx_base: u32, slot_base: u32) -> bool {
    return state[slot_base + 0u] == txs[tx_base + 0u] &&
           state[slot_base + 1u] == txs[tx_base + 1u] &&
           state[slot_base + 2u] == txs[tx_base + 2u] &&
           state[slot_base + 3u] == txs[tx_base + 3u] &&
           state[slot_base + 4u] == txs[tx_base + 4u];
}

// FNV-1a of the 20-byte address at txs[tx_base..tx_base+5].
fn addr_hash(tx_base: u32) -> u32 {
    var h: u32 = 2166136261u;
    for (var w: u32 = 0u; w < 5u; w = w + 1u) {
        let word = txs[tx_base + w];
        // Per-byte FNV-1a.
        let bytes = w * 4u;
        let limit: u32 = select(4u, 20u - bytes, bytes + 4u > 20u);
        for (var b: u32 = 0u; b < limit; b = b + 1u) {
            let byte = (word >> (b * 8u)) & 0xFFu;
            h = h ^ byte;
            h = h * 16777619u;
        }
    }
    return h & ACCOUNT_TABLE_MASK;
}

// Probe the account table for the sender. Returns the slot index, or
// ACCOUNT_TABLE_SIZE on miss. tx_base is the index into `txs` array.
fn find_account(tx_base: u32) -> u32 {
    let h = addr_hash(tx_base);
    for (var probe: u32 = 0u; probe < 256u; probe = probe + 1u) {
        let slot_idx = (h + probe) & ACCOUNT_TABLE_MASK;
        let slot_base = slot_idx * ACCT_STRIDE_U32;
        let occupied = state[slot_base + 5u];
        if (occupied == 0u) {
            return ACCOUNT_TABLE_SIZE;
        }
        if (addr_eq(tx_base, slot_base)) {
            return slot_idx;
        }
    }
    return ACCOUNT_TABLE_SIZE;
}

// =============================================================================
// Validation kernel
// =============================================================================

@compute @workgroup_size(64)
fn validate_transactions(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let tid = gid.x;
    if (tid >= num_txs) { return; }

    let tx_base = tid * TX_STRIDE_U32;

    // -- 1. Sender non-zero --------------------------------------------------
    if (addr_is_zero(tx_base)) {
        valid_flags[tid] = 0u;
        error_codes[tid] = ERR_SENDER_ZERO;
        return;
    }

    var errors: u32 = ERR_NONE;

    // -- 2. Look up sender account -------------------------------------------
    let slot_idx = find_account(tx_base);
    var acct_nonce:   vec2<u32> = vec2<u32>(0u, 0u);
    var acct_balance: vec2<u32> = vec2<u32>(0u, 0u);
    if (slot_idx < ACCOUNT_TABLE_SIZE) {
        let slot_base = slot_idx * ACCT_STRIDE_U32;
        acct_nonce   = vec2<u32>(state[slot_base + 6u], state[slot_base + 7u]);
        acct_balance = vec2<u32>(state[slot_base + 8u], state[slot_base + 9u]);
    }

    // -- 3. Read tx fields ---------------------------------------------------
    let gas_limit_lo = txs[tx_base + 10u];
    let gas_limit_hi = txs[tx_base + 11u];
    let value_lo     = txs[tx_base + 12u];
    let value_hi     = txs[tx_base + 13u];
    let nonce_lo     = txs[tx_base + 14u];
    let nonce_hi     = txs[tx_base + 15u];
    let gas_price_lo = txs[tx_base + 16u];
    let gas_price_hi = txs[tx_base + 17u];
    let calldata_size = txs[tx_base + 18u];
    let is_create     = txs[tx_base + 19u];

    // -- 4. Nonce ------------------------------------------------------------
    if (u64_lt(nonce_lo, nonce_hi, acct_nonce.x, acct_nonce.y)) {
        errors = errors | ERR_NONCE_LOW;
    } else if (u64_gt(nonce_lo, nonce_hi, acct_nonce.x, acct_nonce.y)) {
        errors = errors | ERR_NONCE_HIGH;
    }

    // -- 5. Intrinsic gas (fits in u32 for realistic txs; widen via vec2) ---
    let base_gas: u32 = select(GAS_BASE_TX, GAS_BASE_CREATE, is_create != 0u);
    // calldata_gas = calldata_size * 16. Worst case 49152 * 16 = 786432, still u32.
    let calldata_gas: u32 = calldata_size * GAS_PER_CALLDATA;
    var initcode_gas: u32 = 0u;
    if (is_create != 0u) {
        if (calldata_size > MAX_INITCODE_SIZE) {
            errors = errors | ERR_GAS_LOW;
        }
        // ceil(calldata_size / 32) * 2.
        initcode_gas = ((calldata_size + 31u) / 32u) * 2u;
    }
    // intrinsic_gas fits in u32 for all realistic txs but widen for safety.
    let intrinsic_lo = base_gas + calldata_gas + initcode_gas;
    if (u64_lt(gas_limit_lo, gas_limit_hi, intrinsic_lo, 0u)) {
        errors = errors | ERR_GAS_LOW;
    }

    // -- 6. Balance with overflow guards -------------------------------------
    // gas_cost = gas_limit * gas_price (u64). Overflow indicates BALANCE_LOW
    // because total_cost certainly > acct_balance (acct_balance is u64).
    let gp_nonzero = (gas_price_lo != 0u) || (gas_price_hi != 0u);
    if (gp_nonzero) {
        let mul = u64_mul(gas_limit_lo, gas_limit_hi, gas_price_lo, gas_price_hi);
        if (mul.overflow) {
            errors = errors | ERR_GAS_OVERFLOW;
            errors = errors | ERR_BALANCE_LOW;
        } else {
            // total_cost = gas_cost + value, check second overflow.
            if (u64_add_overflow(mul.lo.x, mul.lo.y, value_lo, value_hi)) {
                errors = errors | ERR_BALANCE_LOW;
            } else {
                let total = u64_add(mul.lo.x, mul.lo.y, value_lo, value_hi);
                if (u64_lt(acct_balance.x, acct_balance.y, total.x, total.y)) {
                    errors = errors | ERR_BALANCE_LOW;
                }
            }
        }
    } else {
        // gas_price == 0 → total_cost = value. Check balance directly.
        if (u64_lt(acct_balance.x, acct_balance.y, value_lo, value_hi)) {
            errors = errors | ERR_BALANCE_LOW;
        }
    }

    valid_flags[tid] = select(0u, 1u, errors == ERR_NONE);
    error_codes[tid] = errors;
}

// =============================================================================
// Nonce-ordering kernel — detect same-sender txs out of order in the block.
// =============================================================================

@compute @workgroup_size(64)
fn validate_nonce_ordering(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let tid = gid.x;
    if (tid >= num_txs) { return; }
    if (valid_flags[tid] == 0u) { return; }

    let my_base = tid * TX_STRIDE_U32;
    let my_nonce_lo = txs[my_base + 14u];
    let my_nonce_hi = txs[my_base + 15u];

    for (var i: u32 = 0u; i < tid; i = i + 1u) {
        if (valid_flags[i] == 0u) { continue; }
        let other_base = i * TX_STRIDE_U32;

        // Same sender?
        let same =
            txs[other_base + 0u] == txs[my_base + 0u] &&
            txs[other_base + 1u] == txs[my_base + 1u] &&
            txs[other_base + 2u] == txs[my_base + 2u] &&
            txs[other_base + 3u] == txs[my_base + 3u] &&
            txs[other_base + 4u] == txs[my_base + 4u];
        if (!same) { continue; }

        let other_nonce_lo = txs[other_base + 14u];
        let other_nonce_hi = txs[other_base + 15u];

        // If an earlier tx with the same sender has nonce >= my nonce,
        // I'm out of order — fail this tx.
        if (!u64_lt(other_nonce_lo, other_nonce_hi, my_nonce_lo, my_nonce_hi)) {
            valid_flags[tid] = 0u;
            return;
        }
    }
}
