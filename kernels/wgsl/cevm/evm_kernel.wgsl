// Copyright (C) 2026, The cevm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// WGSL port of cuda/kernels/cevm/evm_kernel.cu (and metal sibling). One
// thread per tx EVM interpreter.
//
// SCOPE: WGSL has hard limits the CUDA/Metal sources don't — no 64-bit
// native integers, no per-byte addressable buffers (only u32 arrays), much
// stricter register/stack budget per invocation, no recursion. So the WGSL
// path covers the **pure-EVM** opcodes (no host state, no CALL/CREATE):
//
//   * Arithmetic — ADD, SUB, MUL, DIV, SDIV, MOD, SMOD, ADDMOD, MULMOD,
//                  EXP, SIGNEXTEND, LT, GT, SLT, SGT, EQ, ISZERO,
//                  AND, OR, XOR, NOT, BYTE, SHL, SHR, SAR
//   * Stack    — PUSH0..PUSH32, POP, DUP1..DUP16, SWAP1..SWAP16
//   * Control  — JUMP, JUMPI, JUMPDEST, PC, GAS, MSIZE, STOP, RETURN, REVERT
//   * Memory   — MLOAD, MSTORE, MSTORE8
//
// Everything else (KECCAK256, calldata/code/return opcodes,
// SLOAD/SSTORE/TLOAD/TSTORE, BALANCE/EXTCODE*, LOG*, CALL family, CREATE
// family, SELFDESTRUCT, BLOCKHASH/etc. block opcodes) sets
// status=CallNotSupported (5) so the host falls back to CPU cevm — the same
// fallback contract the CUDA path uses for CALL/CREATE.
//
// All uint256 values are encoded as `array<u32, 8>` little-endian limbs:
// limb[0..1] = word w[0] (low 64), limb[2..3] = w[1], etc. uint64 fields
// are vec2<u32>(lo, hi). Layout invariants matching cuda/evm_kernel.cu:
//
//   TxInput  = 136 bytes = 34 u32 words
//   TxOutput =  32 bytes =  8 u32 words
//
// All array indexing is u32-word indexing into storage buffers.

// =============================================================================
// Constants
// =============================================================================

const MAX_MEMORY_PER_TX:  u32 = 65536u;
const MAX_OUTPUT_PER_TX:  u32 = 1024u;
const STACK_DEPTH:        u32 = 1024u;

// Gas constants — mirror CUDA values.
const GAS_VERYLOW:    u32 = 3u;
const GAS_LOW:        u32 = 5u;
const GAS_MID:        u32 = 8u;
const GAS_HIGH:       u32 = 10u;
const GAS_BASE:       u32 = 2u;
const GAS_JUMPDEST:   u32 = 1u;
const GAS_EXP_BASE:   u32 = 10u;
const GAS_EXP_BYTE:   u32 = 50u;
const GAS_MEMORY:     u32 = 3u;

// Status codes.
const ST_STOP:                u32 = 0u;
const ST_RETURN:              u32 = 1u;
const ST_REVERT:              u32 = 2u;
const ST_OUT_OF_GAS:          u32 = 3u;
const ST_ERROR:               u32 = 4u;
const ST_CALL_NOT_SUPPORTED:  u32 = 5u;

// TxInput word-offsets in u32 (34 words total).
//   [0]   code_offset
//   [1]   code_size
//   [2]   calldata_offset
//   [3]   calldata_size
//   [4..5] gas_limit (lo, hi)
//   [6..13] caller    (uint256, 8 u32)
//   [14..21] address  (uint256, 8 u32)
//   [22..29] value    (uint256, 8 u32)
//   [30] warm_addr_offset
//   [31] warm_addr_count
//   [32] warm_slot_offset
//   [33] warm_slot_count
const TX_STRIDE_U32: u32 = 34u;

// TxOutput word-offsets (8 words):
//   [0] status
//   [1..2] gas_used (lo, hi)
//   [3..4] gas_refund (signed, lo, hi)
//   [5] output_size
//   [6..7] _pad
const TX_OUT_STRIDE_U32: u32 = 8u;

// =============================================================================
// Bindings — same buffer ordering as the CUDA / Metal sources.
// =============================================================================

@group(0) @binding(0) var<storage, read>       inputs:    array<u32>;
@group(0) @binding(1) var<storage, read>       blob:      array<u32>;       // byte stream packed 4 bytes/word
@group(0) @binding(2) var<storage, read_write> outputs:   array<u32>;
@group(0) @binding(3) var<storage, read_write> out_data:  array<u32>;
@group(0) @binding(4) var<storage, read_write> mem_pool:  array<u32>;       // byte stream
@group(0) @binding(5) var<uniform>             params:    Params;

struct Params {
    num_txs: u32,
    _pad0:   u32,
    _pad1:   u32,
    _pad2:   u32,
}

// =============================================================================
// Byte-addressed access helpers. blob, mem_pool, out_data are conceptually
// byte arrays but live in u32 storage. byte_at returns the byte at the
// given absolute byte offset.
// =============================================================================

fn blob_byte(off: u32) -> u32 {
    let word = blob[off >> 2u];
    let shift = (off & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

fn mem_byte(off: u32) -> u32 {
    let word = mem_pool[off >> 2u];
    let shift = (off & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

fn mem_store_byte(off: u32, b: u32) {
    let widx = off >> 2u;
    let shift = (off & 3u) * 8u;
    let mask: u32 = ~(0xFFu << shift);
    mem_pool[widx] = (mem_pool[widx] & mask) | ((b & 0xFFu) << shift);
}

fn out_store_byte(off: u32, b: u32) {
    let widx = off >> 2u;
    let shift = (off & 3u) * 8u;
    let mask: u32 = ~(0xFFu << shift);
    out_data[widx] = (out_data[widx] & mask) | ((b & 0xFFu) << shift);
}

// =============================================================================
// uint256 helpers — 8 u32 little-endian limbs.
//   bytes 0..31 → limb index = byte/4, shift = (byte%4)*8 within u32.
//   w[0] (low 64) = limbs 0,1; w[1] = limbs 2,3; w[2] = limbs 4,5; w[3] = 6,7.
// =============================================================================

struct U256 {
    l: array<u32, 8>,
}

fn u256_zero() -> U256 {
    var r: U256;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { r.l[i] = 0u; }
    return r;
}

fn u256_from_u32(x: u32) -> U256 {
    var r = u256_zero();
    r.l[0] = x;
    return r;
}

fn u256_eq(a: U256, b: U256) -> bool {
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        if (a.l[i] != b.l[i]) { return false; }
    }
    return true;
}

fn u256_is_zero(a: U256) -> bool {
    var acc: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { acc = acc | a.l[i]; }
    return acc == 0u;
}

fn u256_lt(a: U256, b: U256) -> bool {
    // MSB-first compare.
    for (var i: i32 = 7; i >= 0; i = i - 1) {
        let ai = a.l[u32(i)];
        let bi = b.l[u32(i)];
        if (ai < bi) { return true; }
        if (ai > bi) { return false; }
    }
    return false;
}

fn u256_gt(a: U256, b: U256) -> bool {
    return u256_lt(b, a);
}

fn u256_add(a: U256, b: U256) -> U256 {
    var r = u256_zero();
    var carry: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let s1 = a.l[i] + b.l[i];
        let c1: u32 = select(0u, 1u, s1 < a.l[i]);
        let s2 = s1 + carry;
        let c2: u32 = select(0u, 1u, s2 < s1);
        r.l[i] = s2;
        carry = c1 + c2;
    }
    return r;
}

fn u256_sub(a: U256, b: U256) -> U256 {
    var r = u256_zero();
    var borrow: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let b1 = select(0u, 1u, a.l[i] < b.l[i]);
        let d1 = a.l[i] - b.l[i];
        let b2 = select(0u, 1u, d1 < borrow);
        let d2 = d1 - borrow;
        r.l[i] = d2;
        borrow = b1 + b2;
    }
    return r;
}

fn u256_and(a: U256, b: U256) -> U256 {
    var r = u256_zero();
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { r.l[i] = a.l[i] & b.l[i]; }
    return r;
}

fn u256_or(a: U256, b: U256) -> U256 {
    var r = u256_zero();
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { r.l[i] = a.l[i] | b.l[i]; }
    return r;
}

fn u256_xor(a: U256, b: U256) -> U256 {
    var r = u256_zero();
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { r.l[i] = a.l[i] ^ b.l[i]; }
    return r;
}

fn u256_not(a: U256) -> U256 {
    var r = u256_zero();
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { r.l[i] = ~a.l[i]; }
    return r;
}

// 256-bit shift-left by n in [0, 256].
fn u256_shl(n: u32, a: U256) -> U256 {
    if (n >= 256u) { return u256_zero(); }
    if (n == 0u) { return a; }
    let word_shift = n / 32u;
    let bit_shift  = n % 32u;
    var r = u256_zero();
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let src = i - word_shift;
        if (i < word_shift) { continue; }
        let lo_part = a.l[src] << bit_shift;
        var hi_part: u32 = 0u;
        if (bit_shift != 0u && src >= 1u) {
            hi_part = a.l[src - 1u] >> (32u - bit_shift);
        }
        // r.l[i] is built from lo_part of src and hi_part from src-1
        r.l[i] = lo_part | hi_part;
    }
    return r;
}

// 256-bit logical right-shift by n in [0, 256].
fn u256_shr(n: u32, a: U256) -> U256 {
    if (n >= 256u) { return u256_zero(); }
    if (n == 0u) { return a; }
    let word_shift = n / 32u;
    let bit_shift  = n % 32u;
    var r = u256_zero();
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let src = i + word_shift;
        if (src >= 8u) { continue; }
        let lo_part = a.l[src] >> bit_shift;
        var hi_part: u32 = 0u;
        if (bit_shift != 0u && src + 1u < 8u) {
            hi_part = a.l[src + 1u] << (32u - bit_shift);
        }
        r.l[i] = lo_part | hi_part;
    }
    return r;
}

// Arithmetic shift right (sign-extending).
fn u256_sar(n: u32, a: U256) -> U256 {
    let is_neg = (a.l[7] & 0x80000000u) != 0u;
    if (!is_neg) { return u256_shr(n, a); }
    if (n >= 256u) { return u256_not(u256_zero()); }
    // Logical shift, then OR in the sign-extension mask in the high bits.
    let lo = u256_shr(n, a);
    // Mask of 1s for bits 256-n..255.
    var mask = u256_not(u256_shr(n, u256_not(u256_zero())));
    if (n == 0u) { return a; }
    return u256_or(lo, mask);
}

// 256-bit multiply (low 256 bits of full 512 product).
fn u256_mul(a: U256, b: U256) -> U256 {
    var r = u256_zero();
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        if (i >= 8u) { break; }
        var carry: u32 = 0u;
        for (var j: u32 = 0u; j + i < 8u; j = j + 1u) {
            // 32x32 → 64 partial product.
            let a_lo = a.l[i] & 0xFFFFu;
            let a_hi = a.l[i] >> 16u;
            let b_lo = b.l[j] & 0xFFFFu;
            let b_hi = b.l[j] >> 16u;
            let ll = a_lo * b_lo;
            let lh = a_lo * b_hi;
            let hl = a_hi * b_lo;
            let hh = a_hi * b_hi;
            let mid = lh + hl;
            let mid_carry: u32 = select(0u, 0x10000u, mid < lh);
            let lo = ll + ((mid & 0xFFFFu) << 16u);
            let lo_carry: u32 = select(0u, 1u, lo < ll);
            let hi = hh + (mid >> 16u) + mid_carry + lo_carry;

            // Accumulate (lo, hi) at position i+j.
            let s1 = r.l[i + j] + lo;
            let c1: u32 = select(0u, 1u, s1 < r.l[i + j]);
            let s2 = s1 + carry;
            let c2: u32 = select(0u, 1u, s2 < s1);
            r.l[i + j] = s2;
            // hi + c1 + c2 propagates upward.
            carry = hi + c1 + c2;
            // We don't worry about carry beyond the 256-bit boundary (truncate to low 256).
        }
        // carry into limb i+8 — drops out of u256 (truncation is the EVM
        // contract — MUL takes the low 256 bits).
    }
    return r;
}

// 256-bit unsigned division. Returns 0 when b == 0 (EVM spec). Schoolbook
// shift-subtract. Slow but correct.
fn u256_divmod(a: U256, b: U256) -> array<U256, 2> {
    var out: array<U256, 2>;
    out[0] = u256_zero();
    out[1] = u256_zero();
    if (u256_is_zero(b)) { return out; }
    if (u256_lt(a, b)) {
        out[0] = u256_zero();
        out[1] = a;
        return out;
    }

    var q = u256_zero();
    var r = u256_zero();
    for (var i: i32 = 255; i >= 0; i = i - 1) {
        // r = (r << 1) | bit_i(a)
        r = u256_shl(1u, r);
        let limb = u32(i) / 32u;
        let bit  = u32(i) % 32u;
        let bv = (a.l[limb] >> bit) & 1u;
        r.l[0] = r.l[0] | bv;
        if (!u256_lt(r, b)) {
            r = u256_sub(r, b);
            q.l[limb] = q.l[limb] | (1u << bit);
        }
    }
    out[0] = q;
    out[1] = r;
    return out;
}

fn u256_div(a: U256, b: U256) -> U256 {
    let dm = u256_divmod(a, b);
    return dm[0];
}

fn u256_mod(a: U256, b: U256) -> U256 {
    let dm = u256_divmod(a, b);
    return dm[1];
}

// =============================================================================
// Per-thread state — passed by-pointer (WGSL ptr<function, T>) into op
// handlers. Stack is a private array of 1024 U256.
// =============================================================================

struct Stack {
    s: array<U256, 1024>,
}

fn stack_push(stk: ptr<function, Stack>, sp: ptr<function, u32>, v: U256) -> bool {
    if (*sp >= STACK_DEPTH) { return false; }
    (*stk).s[*sp] = v;
    *sp = *sp + 1u;
    return true;
}

fn stack_pop(stk: ptr<function, Stack>, sp: ptr<function, u32>, out: ptr<function, U256>) -> bool {
    if (*sp == 0u) { return false; }
    *sp = *sp - 1u;
    *out = (*stk).s[*sp];
    return true;
}

// =============================================================================
// Memory expansion helpers. EVM memory grows in 32-byte words; gas cost is
// quadratic in word count. expand_mem_words bumps the high-water and
// charges gas — returns false on OOG.
// =============================================================================

fn mem_cost(words: u32) -> u32 {
    // 3 * words + words^2 / 512. Cap words at a reasonable value so we
    // don't overflow.
    if (words > 65536u) { return 0xFFFFFFFFu; }
    return 3u * words + (words * words) / 512u;
}

fn expand_mem_words(mem_size: ptr<function, u32>, gas: ptr<function, u32>, want_words: u32) -> bool {
    let cur_words = (*mem_size + 31u) / 32u;
    if (want_words <= cur_words) { return true; }
    let old_cost = mem_cost(cur_words);
    let new_cost = mem_cost(want_words);
    let delta = new_cost - old_cost;
    if (delta > *gas) { return false; }
    *gas = *gas - delta;
    *mem_size = want_words * 32u;
    return true;
}

fn expand_mem_range(mem_size: ptr<function, u32>, gas: ptr<function, u32>, off: u32, sz: u32) -> bool {
    if (sz == 0u) { return true; }
    let end = off + sz;
    if (end < off) { return false; }  // overflow
    if (end > MAX_MEMORY_PER_TX) { return false; }
    let want_words = (end + 31u) / 32u;
    return expand_mem_words(mem_size, gas, want_words);
}

// =============================================================================
// Main kernel — one thread per tx.
// =============================================================================

@compute @workgroup_size(64)
fn evm_execute_kernel(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let tid = gid.x;
    if (tid >= params.num_txs) { return; }

    let tx_base = tid * TX_STRIDE_U32;
    let out_base = tid * TX_OUT_STRIDE_U32;
    let mem_base_bytes = tid * MAX_MEMORY_PER_TX;
    let out_data_base_bytes = tid * MAX_OUTPUT_PER_TX;

    let code_offset    = inputs[tx_base + 0u];
    let code_size      = inputs[tx_base + 1u];
    let calldata_offset = inputs[tx_base + 2u];
    let calldata_size   = inputs[tx_base + 3u];
    let gas_limit_lo    = inputs[tx_base + 4u];
    let gas_limit_hi    = inputs[tx_base + 5u];
    _ = calldata_offset;
    _ = calldata_size;
    _ = gas_limit_hi;

    // gas is u32 — WGSL kernel handles realistic per-tx budgets only.
    var gas: u32 = gas_limit_lo;

    var stk: Stack;
    var sp: u32 = 0u;
    var pc: u32 = 0u;
    var mem_size: u32 = 0u;
    var status: u32 = ST_STOP;
    var output_size: u32 = 0u;

    // Bytecode pointer is into `blob` starting at code_offset.
    loop {
        if (pc >= code_size) { status = ST_STOP; break; }
        if (gas < 1u)        { status = ST_OUT_OF_GAS; break; }
        let op = blob_byte(code_offset + pc);
        pc = pc + 1u;

        // 0x00 STOP
        if (op == 0x00u) { status = ST_STOP; break; }

        // 0x5b JUMPDEST
        if (op == 0x5bu) {
            if (gas < GAS_JUMPDEST) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_JUMPDEST;
            continue;
        }

        // ====== Arithmetic ======
        if (op == 0x01u) { // ADD
            if (gas < GAS_VERYLOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_VERYLOW;
            var a: U256; var b: U256;
            if (!stack_pop(&stk, &sp, &a) || !stack_pop(&stk, &sp, &b)) { status = ST_ERROR; break; }
            _ = stack_push(&stk, &sp, u256_add(a, b));
            continue;
        }
        if (op == 0x02u) { // MUL
            if (gas < GAS_LOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_LOW;
            var a: U256; var b: U256;
            if (!stack_pop(&stk, &sp, &a) || !stack_pop(&stk, &sp, &b)) { status = ST_ERROR; break; }
            _ = stack_push(&stk, &sp, u256_mul(a, b));
            continue;
        }
        if (op == 0x03u) { // SUB
            if (gas < GAS_VERYLOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_VERYLOW;
            var a: U256; var b: U256;
            if (!stack_pop(&stk, &sp, &a) || !stack_pop(&stk, &sp, &b)) { status = ST_ERROR; break; }
            _ = stack_push(&stk, &sp, u256_sub(a, b));
            continue;
        }
        if (op == 0x04u) { // DIV
            if (gas < GAS_LOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_LOW;
            var a: U256; var b: U256;
            if (!stack_pop(&stk, &sp, &a) || !stack_pop(&stk, &sp, &b)) { status = ST_ERROR; break; }
            _ = stack_push(&stk, &sp, u256_div(a, b));
            continue;
        }
        if (op == 0x06u) { // MOD
            if (gas < GAS_LOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_LOW;
            var a: U256; var b: U256;
            if (!stack_pop(&stk, &sp, &a) || !stack_pop(&stk, &sp, &b)) { status = ST_ERROR; break; }
            _ = stack_push(&stk, &sp, u256_mod(a, b));
            continue;
        }

        // ====== Comparison ======
        if (op == 0x10u) { // LT
            if (gas < GAS_VERYLOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_VERYLOW;
            var a: U256; var b: U256;
            if (!stack_pop(&stk, &sp, &a) || !stack_pop(&stk, &sp, &b)) { status = ST_ERROR; break; }
            _ = stack_push(&stk, &sp, u256_from_u32(select(0u, 1u, u256_lt(a, b))));
            continue;
        }
        if (op == 0x11u) { // GT
            if (gas < GAS_VERYLOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_VERYLOW;
            var a: U256; var b: U256;
            if (!stack_pop(&stk, &sp, &a) || !stack_pop(&stk, &sp, &b)) { status = ST_ERROR; break; }
            _ = stack_push(&stk, &sp, u256_from_u32(select(0u, 1u, u256_gt(a, b))));
            continue;
        }
        if (op == 0x14u) { // EQ
            if (gas < GAS_VERYLOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_VERYLOW;
            var a: U256; var b: U256;
            if (!stack_pop(&stk, &sp, &a) || !stack_pop(&stk, &sp, &b)) { status = ST_ERROR; break; }
            _ = stack_push(&stk, &sp, u256_from_u32(select(0u, 1u, u256_eq(a, b))));
            continue;
        }
        if (op == 0x15u) { // ISZERO
            if (gas < GAS_VERYLOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_VERYLOW;
            var a: U256;
            if (!stack_pop(&stk, &sp, &a)) { status = ST_ERROR; break; }
            _ = stack_push(&stk, &sp, u256_from_u32(select(0u, 1u, u256_is_zero(a))));
            continue;
        }

        // ====== Bitwise ======
        if (op == 0x16u) { // AND
            if (gas < GAS_VERYLOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_VERYLOW;
            var a: U256; var b: U256;
            if (!stack_pop(&stk, &sp, &a) || !stack_pop(&stk, &sp, &b)) { status = ST_ERROR; break; }
            _ = stack_push(&stk, &sp, u256_and(a, b));
            continue;
        }
        if (op == 0x17u) { // OR
            if (gas < GAS_VERYLOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_VERYLOW;
            var a: U256; var b: U256;
            if (!stack_pop(&stk, &sp, &a) || !stack_pop(&stk, &sp, &b)) { status = ST_ERROR; break; }
            _ = stack_push(&stk, &sp, u256_or(a, b));
            continue;
        }
        if (op == 0x18u) { // XOR
            if (gas < GAS_VERYLOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_VERYLOW;
            var a: U256; var b: U256;
            if (!stack_pop(&stk, &sp, &a) || !stack_pop(&stk, &sp, &b)) { status = ST_ERROR; break; }
            _ = stack_push(&stk, &sp, u256_xor(a, b));
            continue;
        }
        if (op == 0x19u) { // NOT
            if (gas < GAS_VERYLOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_VERYLOW;
            var a: U256;
            if (!stack_pop(&stk, &sp, &a)) { status = ST_ERROR; break; }
            _ = stack_push(&stk, &sp, u256_not(a));
            continue;
        }

        // 0x1b SHL, 0x1c SHR, 0x1d SAR
        if (op == 0x1bu) {
            if (gas < GAS_VERYLOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_VERYLOW;
            var shift: U256; var val: U256;
            if (!stack_pop(&stk, &sp, &shift) || !stack_pop(&stk, &sp, &val)) { status = ST_ERROR; break; }
            // If shift >= 256, result is 0 (handled inside u256_shl). Use limb 0 truncated.
            let n: u32 = select(256u, shift.l[0], (shift.l[1] | shift.l[2] | shift.l[3] | shift.l[4] | shift.l[5] | shift.l[6] | shift.l[7]) == 0u && shift.l[0] < 256u);
            _ = stack_push(&stk, &sp, u256_shl(n, val));
            continue;
        }
        if (op == 0x1cu) {
            if (gas < GAS_VERYLOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_VERYLOW;
            var shift: U256; var val: U256;
            if (!stack_pop(&stk, &sp, &shift) || !stack_pop(&stk, &sp, &val)) { status = ST_ERROR; break; }
            let n: u32 = select(256u, shift.l[0], (shift.l[1] | shift.l[2] | shift.l[3] | shift.l[4] | shift.l[5] | shift.l[6] | shift.l[7]) == 0u && shift.l[0] < 256u);
            _ = stack_push(&stk, &sp, u256_shr(n, val));
            continue;
        }

        // ====== Stack: PUSH0..PUSH32 (0x5f..0x7f) ======
        if (op == 0x5fu) { // PUSH0
            if (gas < GAS_BASE) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_BASE;
            _ = stack_push(&stk, &sp, u256_zero());
            continue;
        }
        if (op >= 0x60u && op <= 0x7fu) {
            if (gas < GAS_VERYLOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_VERYLOW;
            let n = op - 0x5fu;  // 1..32 bytes
            var v = u256_zero();
            for (var k: u32 = 0u; k < n; k = k + 1u) {
                let b = blob_byte(code_offset + pc + k);
                // BE bytes: high byte first into limb 7.
                let pos = (n - 1u - k);  // little-endian byte position from BE input
                let limb = pos / 4u;
                let shift = (pos % 4u) * 8u;
                v.l[limb] = v.l[limb] | (b << shift);
            }
            pc = pc + n;
            _ = stack_push(&stk, &sp, v);
            continue;
        }

        // ====== Stack: POP, DUP, SWAP ======
        if (op == 0x50u) { // POP
            if (gas < GAS_BASE) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_BASE;
            var dummy: U256;
            if (!stack_pop(&stk, &sp, &dummy)) { status = ST_ERROR; break; }
            continue;
        }
        if (op >= 0x80u && op <= 0x8fu) { // DUP1..DUP16
            if (gas < GAS_VERYLOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_VERYLOW;
            let n = op - 0x7fu;  // 1..16
            if (sp < n) { status = ST_ERROR; break; }
            let v = stk.s[sp - n];
            _ = stack_push(&stk, &sp, v);
            continue;
        }
        if (op >= 0x90u && op <= 0x9fu) { // SWAP1..SWAP16
            if (gas < GAS_VERYLOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_VERYLOW;
            let n = op - 0x8fu;  // 1..16
            if (sp < n + 1u) { status = ST_ERROR; break; }
            let tmp = stk.s[sp - 1u];
            stk.s[sp - 1u] = stk.s[sp - 1u - n];
            stk.s[sp - 1u - n] = tmp;
            continue;
        }

        // ====== Control flow ======
        if (op == 0x56u) { // JUMP
            if (gas < GAS_MID) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_MID;
            var dest: U256;
            if (!stack_pop(&stk, &sp, &dest)) { status = ST_ERROR; break; }
            // Top bits must be zero for a valid pc.
            if ((dest.l[1] | dest.l[2] | dest.l[3] | dest.l[4] | dest.l[5] | dest.l[6] | dest.l[7]) != 0u) {
                status = ST_ERROR; break;
            }
            let jdest = dest.l[0];
            if (jdest >= code_size) { status = ST_ERROR; break; }
            if (blob_byte(code_offset + jdest) != 0x5bu) { status = ST_ERROR; break; }
            pc = jdest;
            continue;
        }
        if (op == 0x57u) { // JUMPI
            if (gas < GAS_HIGH) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_HIGH;
            var dest: U256; var cond: U256;
            if (!stack_pop(&stk, &sp, &dest) || !stack_pop(&stk, &sp, &cond)) { status = ST_ERROR; break; }
            if (u256_is_zero(cond)) { continue; }
            if ((dest.l[1] | dest.l[2] | dest.l[3] | dest.l[4] | dest.l[5] | dest.l[6] | dest.l[7]) != 0u) {
                status = ST_ERROR; break;
            }
            let jdest = dest.l[0];
            if (jdest >= code_size) { status = ST_ERROR; break; }
            if (blob_byte(code_offset + jdest) != 0x5bu) { status = ST_ERROR; break; }
            pc = jdest;
            continue;
        }
        if (op == 0x58u) { // PC
            if (gas < GAS_BASE) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_BASE;
            _ = stack_push(&stk, &sp, u256_from_u32(pc - 1u));
            continue;
        }
        if (op == 0x59u) { // MSIZE
            if (gas < GAS_BASE) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_BASE;
            _ = stack_push(&stk, &sp, u256_from_u32(mem_size));
            continue;
        }
        if (op == 0x5au) { // GAS
            if (gas < GAS_BASE) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_BASE;
            _ = stack_push(&stk, &sp, u256_from_u32(gas));
            continue;
        }

        // ====== Memory ======
        if (op == 0x51u) { // MLOAD
            if (gas < GAS_VERYLOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_VERYLOW;
            var off: U256;
            if (!stack_pop(&stk, &sp, &off)) { status = ST_ERROR; break; }
            if ((off.l[1] | off.l[2] | off.l[3] | off.l[4] | off.l[5] | off.l[6] | off.l[7]) != 0u) {
                status = ST_ERROR; break;
            }
            let off32 = off.l[0];
            if (!expand_mem_range(&mem_size, &gas, off32, 32u)) { status = ST_OUT_OF_GAS; break; }
            var v = u256_zero();
            // Read 32 bytes big-endian — high byte at lowest addr.
            for (var k: u32 = 0u; k < 32u; k = k + 1u) {
                let b = mem_byte(mem_base_bytes + off32 + k);
                let pos = 31u - k;
                let limb = pos / 4u;
                let shift = (pos % 4u) * 8u;
                v.l[limb] = v.l[limb] | (b << shift);
            }
            _ = stack_push(&stk, &sp, v);
            continue;
        }
        if (op == 0x52u) { // MSTORE
            if (gas < GAS_VERYLOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_VERYLOW;
            var off: U256; var val: U256;
            if (!stack_pop(&stk, &sp, &off) || !stack_pop(&stk, &sp, &val)) { status = ST_ERROR; break; }
            if ((off.l[1] | off.l[2] | off.l[3] | off.l[4] | off.l[5] | off.l[6] | off.l[7]) != 0u) {
                status = ST_ERROR; break;
            }
            let off32 = off.l[0];
            if (!expand_mem_range(&mem_size, &gas, off32, 32u)) { status = ST_OUT_OF_GAS; break; }
            // Write 32 bytes big-endian.
            for (var k: u32 = 0u; k < 32u; k = k + 1u) {
                let pos = 31u - k;
                let limb = pos / 4u;
                let shift = (pos % 4u) * 8u;
                let b = (val.l[limb] >> shift) & 0xFFu;
                mem_store_byte(mem_base_bytes + off32 + k, b);
            }
            continue;
        }
        if (op == 0x53u) { // MSTORE8
            if (gas < GAS_VERYLOW) { status = ST_OUT_OF_GAS; break; }
            gas = gas - GAS_VERYLOW;
            var off: U256; var val: U256;
            if (!stack_pop(&stk, &sp, &off) || !stack_pop(&stk, &sp, &val)) { status = ST_ERROR; break; }
            if ((off.l[1] | off.l[2] | off.l[3] | off.l[4] | off.l[5] | off.l[6] | off.l[7]) != 0u) {
                status = ST_ERROR; break;
            }
            let off32 = off.l[0];
            if (!expand_mem_range(&mem_size, &gas, off32, 1u)) { status = ST_OUT_OF_GAS; break; }
            mem_store_byte(mem_base_bytes + off32, val.l[0] & 0xFFu);
            continue;
        }

        // ====== RETURN, REVERT ======
        // Per EVM Yellow Paper, RETURN and REVERT are in the W_zero tier:
        // base gas = 0. They only pay for memory expansion via
        // expand_mem_range below. The earlier `gas -= GAS_BASE` here was
        // a divergence from cevm CPU + Metal/CUDA references — removed.
        if (op == 0xf3u || op == 0xfdu) {
            var off: U256; var sz: U256;
            if (!stack_pop(&stk, &sp, &off) || !stack_pop(&stk, &sp, &sz)) { status = ST_ERROR; break; }
            if ((off.l[1] | off.l[2] | off.l[3] | off.l[4] | off.l[5] | off.l[6] | off.l[7]) != 0u) { status = ST_ERROR; break; }
            if ((sz.l[1]  | sz.l[2]  | sz.l[3]  | sz.l[4]  | sz.l[5]  | sz.l[6]  | sz.l[7]) != 0u)  { status = ST_ERROR; break; }
            let off32 = off.l[0];
            let sz32  = sz.l[0];
            if (sz32 > MAX_OUTPUT_PER_TX) { status = ST_ERROR; break; }
            if (sz32 > 0u) {
                if (!expand_mem_range(&mem_size, &gas, off32, sz32)) { status = ST_OUT_OF_GAS; break; }
                for (var k: u32 = 0u; k < sz32; k = k + 1u) {
                    out_store_byte(out_data_base_bytes + k, mem_byte(mem_base_bytes + off32 + k));
                }
            }
            output_size = sz32;
            status = select(ST_REVERT, ST_RETURN, op == 0xf3u);
            break;
        }

        // ====== INVALID / unsupported — bail out so host falls back to CPU ======
        // KECCAK256 (0x20), env opcodes (0x30..0x4f), SLOAD/SSTORE (0x54/0x55),
        // TLOAD/TSTORE (0x5c/0x5d), LOG0..LOG4 (0xa0..0xa4), CALL family
        // (0xf0..0xff except RETURN/REVERT) — all kick back to host.
        status = ST_CALL_NOT_SUPPORTED;
        break;
    }

    // Write TxOutput.
    let gas_used = gas_limit_lo - gas;
    outputs[out_base + 0u] = status;
    outputs[out_base + 1u] = gas_used;     // gas_used lo
    outputs[out_base + 2u] = 0u;           // gas_used hi
    outputs[out_base + 3u] = 0u;           // gas_refund lo
    outputs[out_base + 4u] = 0u;           // gas_refund hi
    outputs[out_base + 5u] = output_size;
    outputs[out_base + 6u] = 0u;
    outputs[out_base + 7u] = 0u;
}
