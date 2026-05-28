// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// BLAKE3 Cryptographic Hash Function
// High-performance parallel hashing optimized for GPU execution
// Supports keyed hashing, key derivation, and extensible output

// BLAKE3 Constants
const BLAKE3_OUT_LEN: u32 = 32u;
const BLAKE3_KEY_LEN: u32 = 32u;
const BLAKE3_BLOCK_LEN: u32 = 64u;
const BLAKE3_CHUNK_LEN: u32 = 1024u;

// Domain separation flags
const CHUNK_START: u32 = 1u;
const CHUNK_END: u32 = 2u;
const PARENT: u32 = 4u;
const ROOT: u32 = 8u;
const KEYED_HASH: u32 = 16u;
const DERIVE_KEY_CONTEXT: u32 = 32u;
const DERIVE_KEY_MATERIAL: u32 = 64u;

// Initial vector (same as SHA-256)
const IV: array<u32, 8> = array<u32, 8>(
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
);

// Message schedule for BLAKE3 (different from BLAKE2)
const MSG_SCHEDULE: array<array<u32, 16>, 7> = array<array<u32, 16>, 7>(
    array<u32, 16>(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u),
    array<u32, 16>(2u, 6u, 3u, 10u, 7u, 0u, 4u, 13u, 1u, 11u, 12u, 5u, 9u, 14u, 15u, 8u),
    array<u32, 16>(3u, 4u, 10u, 12u, 13u, 2u, 7u, 14u, 6u, 5u, 9u, 0u, 11u, 15u, 8u, 1u),
    array<u32, 16>(10u, 7u, 12u, 9u, 14u, 3u, 13u, 15u, 4u, 0u, 11u, 2u, 5u, 8u, 1u, 6u),
    array<u32, 16>(12u, 13u, 9u, 11u, 15u, 10u, 14u, 8u, 7u, 2u, 5u, 3u, 0u, 1u, 6u, 4u),
    array<u32, 16>(9u, 14u, 11u, 5u, 8u, 12u, 15u, 1u, 13u, 3u, 0u, 10u, 2u, 6u, 4u, 7u),
    array<u32, 16>(11u, 15u, 5u, 0u, 1u, 9u, 8u, 6u, 14u, 10u, 2u, 12u, 3u, 4u, 7u, 13u)
);

// ============================================================================
// Data Structures
// ============================================================================

// Key stored as vec4 pairs to satisfy uniform buffer alignment (16 bytes)
struct Blake3Params {
    input_len: u32,
    flags: u32,                  // Domain separation flags
    num_chunks: u32,
    _pad: u32,
    key_words_0: vec4<u32>,      // key_words[0..3]
    key_words_1: vec4<u32>,      // key_words[4..7]
}

// Helper to extract key_words array from params
fn get_key_words(p: Blake3Params) -> array<u32, 8> {
    return array<u32, 8>(
        p.key_words_0.x, p.key_words_0.y, p.key_words_0.z, p.key_words_0.w,
        p.key_words_1.x, p.key_words_1.y, p.key_words_1.z, p.key_words_1.w
    );
}

struct ChunkState {
    cv: array<u32, 8>,           // Chaining value
    chunk_counter: u32,
    block_len: u32,
    blocks_compressed: u32,
    flags: u32,
}

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read_write> chunk_cvs: array<u32>;  // Chaining values for tree
@group(0) @binding(3) var<uniform> params: Blake3Params;

// Workgroup shared memory for parallel compression
var<workgroup> shared_state: array<u32, 16>;
var<workgroup> shared_msg: array<u32, 16>;

// ============================================================================
// Core Functions
// ============================================================================

// Rotate right
fn rotr(x: u32, n: u32) -> u32 {
    return (x >> n) | (x << (32u - n));
}

// Quarter round (G function)
fn g(
    state: ptr<function, array<u32, 16>>,
    a: u32, b: u32, c: u32, d: u32,
    mx: u32, my: u32
) {
    (*state)[a] = (*state)[a] + (*state)[b] + mx;
    (*state)[d] = rotr((*state)[d] ^ (*state)[a], 16u);
    (*state)[c] = (*state)[c] + (*state)[d];
    (*state)[b] = rotr((*state)[b] ^ (*state)[c], 12u);
    (*state)[a] = (*state)[a] + (*state)[b] + my;
    (*state)[d] = rotr((*state)[d] ^ (*state)[a], 8u);
    (*state)[c] = (*state)[c] + (*state)[d];
    (*state)[b] = rotr((*state)[b] ^ (*state)[c], 7u);
}

// Full round
fn round(state: ptr<function, array<u32, 16>>, msg: ptr<function, array<u32, 16>>, round_idx: u32) {
    let schedule = MSG_SCHEDULE[round_idx % 7u];
    
    // Column step
    g(state, 0u, 4u, 8u, 12u, (*msg)[schedule[0]], (*msg)[schedule[1]]);
    g(state, 1u, 5u, 9u, 13u, (*msg)[schedule[2]], (*msg)[schedule[3]]);
    g(state, 2u, 6u, 10u, 14u, (*msg)[schedule[4]], (*msg)[schedule[5]]);
    g(state, 3u, 7u, 11u, 15u, (*msg)[schedule[6]], (*msg)[schedule[7]]);
    
    // Diagonal step
    g(state, 0u, 5u, 10u, 15u, (*msg)[schedule[8]], (*msg)[schedule[9]]);
    g(state, 1u, 6u, 11u, 12u, (*msg)[schedule[10]], (*msg)[schedule[11]]);
    g(state, 2u, 7u, 8u, 13u, (*msg)[schedule[12]], (*msg)[schedule[13]]);
    g(state, 3u, 4u, 9u, 14u, (*msg)[schedule[14]], (*msg)[schedule[15]]);
}

// Compress a single block
fn compress(
    cv: array<u32, 8>,
    block: array<u32, 16>,
    counter: u32,
    block_len: u32,
    flags: u32
) -> array<u32, 16> {
    var state: array<u32, 16>;
    
    // Initialize state
    state[0] = cv[0];
    state[1] = cv[1];
    state[2] = cv[2];
    state[3] = cv[3];
    state[4] = cv[4];
    state[5] = cv[5];
    state[6] = cv[6];
    state[7] = cv[7];
    state[8] = IV[0];
    state[9] = IV[1];
    state[10] = IV[2];
    state[11] = IV[3];
    state[12] = counter;        // Counter low
    state[13] = 0u;             // Counter high (for simplicity)
    state[14] = block_len;
    state[15] = flags;
    
    var msg = block;
    
    // 7 rounds
    round(&state, &msg, 0u);
    round(&state, &msg, 1u);
    round(&state, &msg, 2u);
    round(&state, &msg, 3u);
    round(&state, &msg, 4u);
    round(&state, &msg, 5u);
    round(&state, &msg, 6u);
    
    // XOR with input CV
    state[0] ^= state[8];
    state[1] ^= state[9];
    state[2] ^= state[10];
    state[3] ^= state[11];
    state[4] ^= state[12];
    state[5] ^= state[13];
    state[6] ^= state[14];
    state[7] ^= state[15];
    state[8] ^= cv[0];
    state[9] ^= cv[1];
    state[10] ^= cv[2];
    state[11] ^= cv[3];
    state[12] ^= cv[4];
    state[13] ^= cv[5];
    state[14] ^= cv[6];
    state[15] ^= cv[7];
    
    return state;
}

// Extract chaining value from compression output
fn compress_to_cv(
    cv: array<u32, 8>,
    block: array<u32, 16>,
    counter: u32,
    block_len: u32,
    flags: u32
) -> array<u32, 8> {
    let state = compress(cv, block, counter, block_len, flags);
    return array<u32, 8>(
        state[0], state[1], state[2], state[3],
        state[4], state[5], state[6], state[7]
    );
}

// ============================================================================
// Chunk Processing
// ============================================================================

// Process a single chunk (1024 bytes = 16 blocks)
fn process_chunk(
    chunk_idx: u32,
    input_offset: u32,
    is_root: bool
) -> array<u32, 8> {
    var cv: array<u32, 8>;
    
    // Initialize CV with key or IV
    if (params.flags & KEYED_HASH) != 0u {
        cv = get_key_words(params);
    } else {
        for (var i = 0u; i < 8u; i++) {
            cv[i] = IV[i];
        }
    }
    
    // Process 16 blocks per chunk
    let num_blocks = 16u;
    
    for (var block_idx = 0u; block_idx < num_blocks; block_idx++) {
        var block: array<u32, 16>;
        let block_offset = input_offset + block_idx * 16u;
        
        // Load block
        for (var i = 0u; i < 16u; i++) {
            block[i] = input[block_offset + i];
        }
        
        // Compute flags for this block
        var block_flags = params.flags;
        if (block_idx == 0u) {
            block_flags |= CHUNK_START;
        }
        if (block_idx == num_blocks - 1u) {
            block_flags |= CHUNK_END;
            if (is_root) {
                block_flags |= ROOT;
            }
        }
        
        cv = compress_to_cv(cv, block, chunk_idx, BLAKE3_BLOCK_LEN, block_flags);
    }
    
    return cv;
}

// ============================================================================
// Compute Kernels
// ============================================================================

// Process chunks in parallel (first level of tree)
@compute @workgroup_size(256)
fn hash_chunks(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let chunk_idx = gid.x;
    
    if (chunk_idx >= params.num_chunks) { return; }
    
    let input_offset = chunk_idx * (BLAKE3_CHUNK_LEN / 4u);  // In u32 units
    let is_root = params.num_chunks == 1u;
    
    let cv = process_chunk(chunk_idx, input_offset, is_root);
    
    // Store chaining value
    let cv_offset = chunk_idx * 8u;
    for (var i = 0u; i < 8u; i++) {
        chunk_cvs[cv_offset + i] = cv[i];
    }
    
    // If single chunk and root, also store as output
    if (is_root) {
        for (var i = 0u; i < 8u; i++) {
            output[i] = cv[i];
        }
    }
}

// Merge parent nodes in parallel (tree construction)
@compute @workgroup_size(256)
fn merge_parents(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let parent_idx = gid.x;
    let num_parents = params.num_chunks / 2u;
    
    if (parent_idx >= num_parents) { return; }
    
    // Load left and right child CVs
    let left_offset = parent_idx * 2u * 8u;
    let right_offset = left_offset + 8u;
    
    var block: array<u32, 16>;
    for (var i = 0u; i < 8u; i++) {
        block[i] = chunk_cvs[left_offset + i];
        block[i + 8u] = chunk_cvs[right_offset + i];
    }
    
    // Parent CV
    var cv: array<u32, 8>;
    if (params.flags & KEYED_HASH) != 0u {
        cv = get_key_words(params);
    } else {
        for (var i = 0u; i < 8u; i++) {
            cv[i] = IV[i];
        }
    }
    
    let is_root = num_parents == 1u;
    var flags = PARENT | params.flags;
    if (is_root) {
        flags |= ROOT;
    }
    
    let parent_cv = compress_to_cv(cv, block, 0u, BLAKE3_BLOCK_LEN, flags);
    
    // Store parent CV
    let out_offset = parent_idx * 8u;
    for (var i = 0u; i < 8u; i++) {
        chunk_cvs[out_offset + i] = parent_cv[i];
    }
    
    if (is_root) {
        for (var i = 0u; i < 8u; i++) {
            output[i] = parent_cv[i];
        }
    }
}

// Hash small input (single chunk or less)
@compute @workgroup_size(1)
fn hash_small(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    var cv: array<u32, 8>;
    
    // Initialize CV
    if (params.flags & KEYED_HASH) != 0u {
        cv = get_key_words(params);
    } else {
        for (var i = 0u; i < 8u; i++) {
            cv[i] = IV[i];
        }
    }
    
    let num_blocks = (params.input_len + BLAKE3_BLOCK_LEN - 1u) / BLAKE3_BLOCK_LEN;
    
    for (var block_idx = 0u; block_idx < num_blocks; block_idx++) {
        var block: array<u32, 16>;
        let block_offset = block_idx * 16u;
        
        // Load block (with padding if necessary)
        for (var i = 0u; i < 16u; i++) {
            let byte_offset = block_offset + i;
            if (byte_offset * 4u < params.input_len) {
                block[i] = input[byte_offset];
            } else {
                block[i] = 0u;
            }
        }
        
        // Compute actual block length
        let remaining = params.input_len - block_idx * BLAKE3_BLOCK_LEN;
        let block_len = min(remaining, BLAKE3_BLOCK_LEN);
        
        // Flags
        var flags = params.flags;
        if (block_idx == 0u) {
            flags |= CHUNK_START;
        }
        if (block_idx == num_blocks - 1u) {
            flags |= CHUNK_END | ROOT;
        }
        
        if (block_idx == num_blocks - 1u) {
            // Last block - get full output
            let state = compress(cv, block, 0u, block_len, flags);
            for (var i = 0u; i < 8u; i++) {
                output[i] = state[i];
            }
        } else {
            cv = compress_to_cv(cv, block, 0u, block_len, flags);
        }
    }
}

// XOF (Extensible Output Function) - generate arbitrary length output
@compute @workgroup_size(256)
fn hash_xof(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let output_block = gid.x;

    // Each block produces 64 bytes (16 u32)
    // Use counter mode from the root compression

    var cv: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) {
        cv[i] = chunk_cvs[i];  // Root CV from first pass
    }

    var block: array<u32, 16>;
    // Block content is empty for XOF extension
    for (var i = 0u; i < 16u; i++) {
        block[i] = 0u;
    }

    let state = compress(cv, block, output_block, 0u, ROOT);

    // Store 64 bytes of output
    let out_offset = output_block * 16u;
    for (var i = 0u; i < 16u; i++) {
        output[out_offset + i] = state[i];
    }
}

// ============================================================================
// Batched variable-length BLAKE3 hash
// ============================================================================
//
// One thread per input. Each thread reads (offsets[h], lengths[h]) from per-
// input arrays and produces a 32-byte digest at outputs[h*8 .. h*8+8] (u32).
//
// This kernel mirrors metal/src/shaders/crypto/blake3.metal::blake3_batch_hash
// and is what the LuxBackend `op_blake3_hash` ABI dispatches into.
//
// Bindings (separate from the small/chunks/parents kernels above):
//   batch_inputs   : flat u8 buffer packed 4 bytes/u32 (LE)
//   batch_outputs  : 32 bytes / 8 u32 words per hash
//   batch_offsets  : per-input start offset (bytes)
//   batch_lengths  : per-input byte length
//   batch_params   : { num_inputs }

struct Blake3BatchParams {
    num_inputs: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(4) var<storage, read>       batch_inputs:  array<u32>;
@group(0) @binding(5) var<storage, read_write> batch_outputs: array<u32>;
@group(0) @binding(6) var<storage, read>       batch_offsets: array<u32>;
@group(0) @binding(7) var<storage, read>       batch_lengths: array<u32>;
@group(0) @binding(8) var<uniform>             batch_params:  Blake3BatchParams;

// Read a single byte from the u32-packed `batch_inputs` buffer at byte offset.
fn batch_read_byte(off: u32) -> u32 {
    let word  = batch_inputs[off >> 2u];
    let shift = (off & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

@compute @workgroup_size(64)
fn hash_batch(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let hid = gid.x;
    if (hid >= batch_params.num_inputs) { return; }

    let off = batch_offsets[hid];
    let len = batch_lengths[hid];

    // Initialize CV = IV.
    var cv: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) {
        cv[i] = IV[i];
    }

    var bytes_processed: u32 = 0u;
    var chunk_counter:   u32 = 0u;

    // Edge case: empty input — single block of zeros, length 0,
    // flags = CHUNK_START | CHUNK_END | ROOT.
    if (len == 0u) {
        var block: array<u32, 16>;
        for (var i = 0u; i < 16u; i++) { block[i] = 0u; }
        let state0 = compress(cv, block, 0u, 0u, CHUNK_START | CHUNK_END | ROOT);
        let out0 = hid * 8u;
        for (var i = 0u; i < 8u; i++) {
            batch_outputs[out0 + i] = state0[i];
        }
        return;
    }

    // Process blocks of up to 64 bytes. The reference Metal implementation
    // emits CHUNK_START on the very first block and CHUNK_END | ROOT on the
    // very last; intermediate blocks carry no chunk flags.
    loop {
        if (bytes_processed >= len) { break; }

        let remaining = len - bytes_processed;
        let block_len = min(64u, remaining);

        // Load up to 64 bytes from batch_inputs into 16 u32 words (LE).
        var block: array<u32, 16>;
        for (var i = 0u; i < 16u; i++) { block[i] = 0u; }
        for (var i = 0u; i < block_len; i++) {
            let byte_val = batch_read_byte(off + bytes_processed + i);
            let word_idx = i / 4u;
            let shift    = (i & 3u) * 8u;
            block[word_idx] = block[word_idx] | (byte_val << shift);
        }

        var flags: u32 = 0u;
        if (bytes_processed == 0u) {
            flags = flags | CHUNK_START;
        }
        if (bytes_processed + block_len >= len) {
            flags = flags | CHUNK_END | ROOT;
        }

        // Compress; on the final block we want the full 16-word state for
        // potential CHUNK_END output. Reference Metal uses cv = compress(...)
        // result truncated to 8 words for both intermediate and final blocks.
        let state = compress(cv, block, chunk_counter, block_len, flags);
        for (var i = 0u; i < 8u; i++) { cv[i] = state[i]; }

        bytes_processed = bytes_processed + block_len;
        if ((bytes_processed % BLAKE3_CHUNK_LEN) == 0u) {
            chunk_counter = chunk_counter + 1u;
        }
    }

    // Write 32-byte digest.
    let out_off = hid * 8u;
    for (var i = 0u; i < 8u; i++) {
        batch_outputs[out_off + i] = cv[i];
    }
}
