// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Keccak-256 (Ethereum variant) — variable-length sponge over arbitrary inputs.
//
// WGSL port of metal/src/shaders/crypto/keccak256.metal. One thread per input.
// Each thread:
//   1. Reads (offset, length) from per-input arrays.
//   2. Absorbs full 136-byte blocks of input into the 25-lane state.
//   3. Final block: copies the tail, appends 0x01 + 0x80 padding (Keccak, NOT
//      SHA-3's 0x06+0x80), absorbs.
//   4. Squeezes the first 32 bytes of state little-endian → output[h*32 .. +32).
//
// Output is bit-identical to cpu_op_keccak256_hash (gpu/src/cpu_backend.cpp),
// keccak256_batch_hash in the Metal kernel, and keccak256_batch_variable in
// the CUDA kernel.
//
// Test vectors (asserted by gpu/test/test_backend_parity.cpp):
//   keccak256("")     = 0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
//   keccak256("abc")  = 0x4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45
//   keccak256(0x00)   = 0xbc36789e7a1e281436464229828f817d6612f7b477d66591ff96a9e064bcc98a
//
// WGSL representation:
//   - State: 25 × u64, represented as array<u32, 50> where state[2*i] = lo,
//     state[2*i + 1] = hi of the i-th 64-bit lane.
//   - Inputs are byte arrays packed into u32 words (4 bytes per u32, LE).
//   - Offsets and lengths are in bytes.

// =============================================================================
// Round constants (split into lo / hi u32 halves)
// =============================================================================

// Lower 32 bits of each round constant.
const KECCAK_RC_LO: array<u32, 24> = array<u32, 24>(
    0x00000001u, 0x00008082u, 0x0000808Au, 0x80008000u,
    0x0000808Bu, 0x80000001u, 0x80008081u, 0x00008009u,
    0x0000008Au, 0x00000088u, 0x80008009u, 0x8000000Au,
    0x8000808Bu, 0x0000008Bu, 0x00008089u, 0x00008003u,
    0x00008002u, 0x00000080u, 0x0000800Au, 0x8000000Au,
    0x80008081u, 0x00008080u, 0x80000001u, 0x80008008u,
);

// Upper 32 bits of each round constant.
const KECCAK_RC_HI: array<u32, 24> = array<u32, 24>(
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x80000000u, 0x80000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u,
);

const KECCAK_PI: array<u32, 24> = array<u32, 24>(
    10u,  7u, 11u, 17u, 18u,  3u,  5u, 16u,  8u, 21u, 24u,  4u,
    15u, 23u, 19u, 13u, 12u,  2u, 20u, 14u, 22u,  9u,  6u,  1u,
);

const KECCAK_RHO: array<u32, 24> = array<u32, 24>(
     1u,  3u,  6u, 10u, 15u, 21u, 28u, 36u, 45u, 55u,  2u, 14u,
    27u, 41u, 56u,  8u, 25u, 43u, 62u, 18u, 39u, 61u, 20u, 44u,
);

// =============================================================================
// 64-bit helpers — emulated with vec2<u32> (lo, hi)
// =============================================================================

// Rotate left by n bits, 0 < n < 64.
// Returns vec2<u32>(lo, hi).
fn rotl64(lo: u32, hi: u32, n: u32) -> vec2<u32> {
    if (n == 0u) {
        return vec2<u32>(lo, hi);
    }
    if (n < 32u) {
        let n32 = 32u - n;
        let new_lo = (lo << n) | (hi >> n32);
        let new_hi = (hi << n) | (lo >> n32);
        return vec2<u32>(new_lo, new_hi);
    }
    // n >= 32 → swap halves and rotate by (n - 32).
    let m = n - 32u;
    if (m == 0u) {
        return vec2<u32>(hi, lo);
    }
    let m32 = 32u - m;
    let new_lo = (hi << m) | (lo >> m32);
    let new_hi = (lo << m) | (hi >> m32);
    return vec2<u32>(new_lo, new_hi);
}

// =============================================================================
// Keccak-f[1600] permutation on a 50-u32 state (25 lanes × 2 u32 each).
// state[2*i] = lo, state[2*i+1] = hi of lane i.
// =============================================================================

fn keccak_f1600(state: ptr<function, array<u32, 50>>) {
    for (var round: u32 = 0u; round < 24u; round = round + 1u) {
        // theta
        var c_lo: array<u32, 5>;
        var c_hi: array<u32, 5>;
        for (var x: u32 = 0u; x < 5u; x = x + 1u) {
            c_lo[x] = (*state)[2u * (x +  0u) + 0u]
                    ^ (*state)[2u * (x +  5u) + 0u]
                    ^ (*state)[2u * (x + 10u) + 0u]
                    ^ (*state)[2u * (x + 15u) + 0u]
                    ^ (*state)[2u * (x + 20u) + 0u];
            c_hi[x] = (*state)[2u * (x +  0u) + 1u]
                    ^ (*state)[2u * (x +  5u) + 1u]
                    ^ (*state)[2u * (x + 10u) + 1u]
                    ^ (*state)[2u * (x + 15u) + 1u]
                    ^ (*state)[2u * (x + 20u) + 1u];
        }
        for (var x: u32 = 0u; x < 5u; x = x + 1u) {
            let xp1 = (x + 1u) % 5u;
            let xm1 = (x + 4u) % 5u;
            let r = rotl64(c_lo[xp1], c_hi[xp1], 1u);
            let d_lo = c_lo[xm1] ^ r.x;
            let d_hi = c_hi[xm1] ^ r.y;
            for (var y: u32 = 0u; y < 5u; y = y + 1u) {
                let idx = 2u * (x + 5u * y);
                (*state)[idx + 0u] = (*state)[idx + 0u] ^ d_lo;
                (*state)[idx + 1u] = (*state)[idx + 1u] ^ d_hi;
            }
        }

        // rho + pi (rotates lane[1] through the PI cycle).
        var t_lo: u32 = (*state)[2u * 1u + 0u];
        var t_hi: u32 = (*state)[2u * 1u + 1u];
        for (var i: u32 = 0u; i < 24u; i = i + 1u) {
            let dest = KECCAK_PI[i];
            let r = KECCAK_RHO[i];
            let tmp_lo = (*state)[2u * dest + 0u];
            let tmp_hi = (*state)[2u * dest + 1u];
            let rot = rotl64(t_lo, t_hi, r);
            (*state)[2u * dest + 0u] = rot.x;
            (*state)[2u * dest + 1u] = rot.y;
            t_lo = tmp_lo;
            t_hi = tmp_hi;
        }

        // chi (operates row by row).
        for (var y: u32 = 0u; y < 5u; y = y + 1u) {
            var row_lo: array<u32, 5>;
            var row_hi: array<u32, 5>;
            for (var x: u32 = 0u; x < 5u; x = x + 1u) {
                let idx = 2u * (x + 5u * y);
                row_lo[x] = (*state)[idx + 0u];
                row_hi[x] = (*state)[idx + 1u];
            }
            for (var x: u32 = 0u; x < 5u; x = x + 1u) {
                let idx = 2u * (x + 5u * y);
                let a = row_lo[(x + 1u) % 5u];
                let b = row_lo[(x + 2u) % 5u];
                let c = row_hi[(x + 1u) % 5u];
                let d = row_hi[(x + 2u) % 5u];
                (*state)[idx + 0u] = row_lo[x] ^ ((~a) & b);
                (*state)[idx + 1u] = row_hi[x] ^ ((~c) & d);
            }
        }

        // iota
        (*state)[0u] = (*state)[0u] ^ KECCAK_RC_LO[round];
        (*state)[1u] = (*state)[1u] ^ KECCAK_RC_HI[round];
    }
}

// =============================================================================
// Byte-level helpers.
// inputs is a flat u8 buffer packed 4 bytes per u32 (little-endian). We index
// into it byte-wise: byte k = (inputs[k >> 2] >> (8 * (k & 3))) & 0xff.
// =============================================================================

fn read_byte(off: u32) -> u32 {
    let word = inputs[off >> 2u];
    let shift = (off & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

// Write byte to output (output is packed same way).
fn write_byte(off: u32, byte: u32) {
    let word_idx = off >> 2u;
    let shift = (off & 3u) * 8u;
    let mask = ~(0xFFu << shift);
    // Read-modify-write — no concurrency issue since each thread writes
    // to its own 32-byte digest slot, and slots are 8-u32 aligned.
    let cur = outputs[word_idx];
    outputs[word_idx] = (cur & mask) | ((byte & 0xFFu) << shift);
}

// =============================================================================
// Bindings
// =============================================================================

@group(0) @binding(0) var<storage, read>       inputs:     array<u32>;
@group(0) @binding(1) var<storage, read_write> outputs:    array<u32>;
@group(0) @binding(2) var<storage, read>       offsets:    array<u32>;
@group(0) @binding(3) var<storage, read>       lengths:    array<u32>;
@group(0) @binding(4) var<uniform>             num_inputs: u32;

// =============================================================================
// Kernel: keccak256_batch_variable
// Same ABI shape as the Metal / CUDA kernels:
//   inputs   : flat u8 buffer (here u32-packed)
//   offsets  : per-thread start offset in bytes into inputs
//   lengths  : per-thread byte length
//   outputs  : 32 bytes per thread, written at outputs[h*8 .. h*8+8] (u32 words)
//   num_inputs: scalar count
// =============================================================================

@compute @workgroup_size(64)
fn keccak256_batch_variable(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let hid = gid.x;
    if (hid >= num_inputs) { return; }

    let off = offsets[hid];
    let len = lengths[hid];

    // State: 25 lanes × 2 u32 = 50 u32, initialised to zero.
    var state: array<u32, 50>;
    for (var i: u32 = 0u; i < 50u; i = i + 1u) {
        state[i] = 0u;
    }

    let rate: u32 = 136u;  // 17 lanes × 8 bytes per lane.
    var absorbed: u32 = 0u;

    // Absorb full 136-byte blocks.
    while (absorbed + rate <= len) {
        for (var w: u32 = 0u; w < 17u; w = w + 1u) {
            var lane_lo: u32 = 0u;
            var lane_hi: u32 = 0u;
            for (var b: u32 = 0u; b < 4u; b = b + 1u) {
                lane_lo = lane_lo | (read_byte(off + absorbed + w * 8u + b) << (b * 8u));
            }
            for (var b: u32 = 0u; b < 4u; b = b + 1u) {
                lane_hi = lane_hi | (read_byte(off + absorbed + w * 8u + 4u + b) << (b * 8u));
            }
            state[2u * w + 0u] = state[2u * w + 0u] ^ lane_lo;
            state[2u * w + 1u] = state[2u * w + 1u] ^ lane_hi;
        }
        keccak_f1600(&state);
        absorbed = absorbed + rate;
    }

    // Final block: build padded buffer (136 bytes), copy tail, apply 0x01 + 0x80.
    var padded: array<u32, 34>;  // 136 bytes packed into 34 u32 words.
    for (var i: u32 = 0u; i < 34u; i = i + 1u) {
        padded[i] = 0u;
    }
    let remaining = len - absorbed;
    for (var i: u32 = 0u; i < remaining; i = i + 1u) {
        let byte = read_byte(off + absorbed + i);
        let word_idx = i >> 2u;
        let shift = (i & 3u) * 8u;
        padded[word_idx] = padded[word_idx] | (byte << shift);
    }
    // Pad: padded[remaining] |= 0x01
    let pad_word = remaining >> 2u;
    let pad_shift = (remaining & 3u) * 8u;
    padded[pad_word] = padded[pad_word] | (0x01u << pad_shift);
    // padded[rate - 1] |= 0x80
    let last_byte = rate - 1u;
    let last_word = last_byte >> 2u;
    let last_shift = (last_byte & 3u) * 8u;
    padded[last_word] = padded[last_word] | (0x80u << last_shift);

    // Absorb padded block.
    for (var w: u32 = 0u; w < 17u; w = w + 1u) {
        let lane_lo = padded[2u * w + 0u];
        let lane_hi = padded[2u * w + 1u];
        state[2u * w + 0u] = state[2u * w + 0u] ^ lane_lo;
        state[2u * w + 1u] = state[2u * w + 1u] ^ lane_hi;
    }
    keccak_f1600(&state);

    // Squeeze 32 bytes = 4 lanes = 8 u32. Write directly to outputs[hid*8..hid*8+8].
    let out_word = hid * 8u;
    for (var w: u32 = 0u; w < 4u; w = w + 1u) {
        outputs[out_word + 2u * w + 0u] = state[2u * w + 0u];
        outputs[out_word + 2u * w + 1u] = state[2u * w + 1u];
    }
}
