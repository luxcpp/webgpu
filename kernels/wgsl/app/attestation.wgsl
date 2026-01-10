// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// GPU Attestation Shader for LP-2000 AI Mining
// Hardware-attested GPU compute proofs with session binding
//
// This shader computes:
// 1. Hardware fingerprint from GPU characteristics
// 2. Compute capability verification (tier detection)
// 3. Attestation hash generation with session binding
// 4. Nonce commitment for TEE-style attestation
//
// Aligns with AIMining.sol WorkProof structure:
// - sessionId: bytes32 (unique session identifier)
// - nonce: uint64 (PoW nonce)
// - gpuId: bytes32 (hardware fingerprint)
// - computeHash: bytes32 (compute work proof)
// - timestamp: uint64 (proof generation time)
// - tier: GPUTier (Consumer/Professional/DataCenter/Sovereign)

// ============================================================================
// Constants
// ============================================================================

// GPU Tier thresholds (TFLOPS * 100 for fixed-point)
const TIER_CONSUMER_MIN: u32 = 1000u;        // 10+ TFLOPS (RTX 30xx/40xx)
const TIER_PROFESSIONAL_MIN: u32 = 4000u;    // 40+ TFLOPS (A4000/A5000/A6000)
const TIER_DATACENTER_MIN: u32 = 15000u;     // 150+ TFLOPS (A100/H100)
const TIER_SOVEREIGN_MIN: u32 = 30000u;      // 300+ TFLOPS (H100 TDX/Blackwell)

// GPU Tier enumeration (matches ChainConfig.sol)
const GPU_TIER_CONSUMER: u32 = 0u;
const GPU_TIER_PROFESSIONAL: u32 = 1u;
const GPU_TIER_DATACENTER: u32 = 2u;
const GPU_TIER_SOVEREIGN: u32 = 3u;

// BLAKE3 constants (reused for attestation hashing)
const BLAKE3_OUT_LEN: u32 = 32u;
const BLAKE3_BLOCK_LEN: u32 = 64u;

// Domain separation flags for attestation
const ATTESTATION_DOMAIN: u32 = 0x41545354u;  // "ATST"
const FINGERPRINT_DOMAIN: u32 = 0x46504E54u;  // "FPNT"
const SESSION_DOMAIN: u32 = 0x53455353u;      // "SESS"

// Initial vector (SHA-256 IV)
const IV: array<u32, 8> = array<u32, 8>(
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
);

// Message schedule for BLAKE3
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

/// GPU hardware characteristics for fingerprinting
struct GPUCharacteristics {
    vendor_id: u32,              // Vendor identifier
    device_id: u32,              // Device identifier
    compute_units: u32,          // Number of compute units/SMs
    max_workgroup_size: u32,     // Maximum workgroup size
    max_memory_mb: u32,          // Maximum memory in MB
    clock_mhz: u32,              // Core clock in MHz
    memory_bandwidth_gbps: u32,  // Memory bandwidth GB/s
    tensor_cores: u32,           // Number of tensor cores (0 if none)
    tee_supported: u32,          // TEE support flag (0/1)
    driver_version: u32,         // Driver version packed
}

/// Attestation parameters
struct AttestationParams {
    session_id: array<u32, 8>,   // 256-bit session ID
    nonce: array<u32, 2>,        // 64-bit nonce (lo, hi)
    timestamp: array<u32, 2>,    // 64-bit timestamp (lo, hi)
    chain_id: u32,               // Target chain ID
    batch_size: u32,             // Number of attestations to generate
    difficulty_target: array<u32, 8>, // 256-bit difficulty target
    _padding: u32,               // Alignment padding
}

/// Work proof output (matches AIMining.sol WorkProof)
struct WorkProof {
    session_id: array<u32, 8>,   // bytes32
    nonce: array<u32, 2>,        // uint64
    gpu_id: array<u32, 8>,       // bytes32 (fingerprint)
    compute_hash: array<u32, 8>, // bytes32
    timestamp: array<u32, 2>,    // uint64
    tier: u32,                   // GPUTier enum
    proof_hash: array<u32, 8>,   // bytes32 (for signature)
}

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<uniform> gpu_chars: GPUCharacteristics;
@group(0) @binding(1) var<uniform> params: AttestationParams;
@group(0) @binding(2) var<storage, read_write> proofs: array<WorkProof>;
@group(0) @binding(3) var<storage, read_write> valid_count: atomic<u32>;

// Workgroup shared memory for parallel hashing
var<workgroup> shared_state: array<u32, 256>;

// ============================================================================
// Core Hashing Functions (BLAKE3-based)
// ============================================================================

fn rotr(x: u32, n: u32) -> u32 {
    return (x >> n) | (x << (32u - n));
}

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

fn blake3_round(state: ptr<function, array<u32, 16>>, msg: ptr<function, array<u32, 16>>, round_idx: u32) {
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

fn blake3_compress(
    cv: array<u32, 8>,
    block: array<u32, 16>,
    counter: u32,
    block_len: u32,
    flags: u32
) -> array<u32, 8> {
    var state: array<u32, 16>;

    // Initialize state from chaining value
    state[0] = cv[0]; state[1] = cv[1]; state[2] = cv[2]; state[3] = cv[3];
    state[4] = cv[4]; state[5] = cv[5]; state[6] = cv[6]; state[7] = cv[7];
    state[8] = IV[0]; state[9] = IV[1]; state[10] = IV[2]; state[11] = IV[3];
    state[12] = counter;
    state[13] = 0u;
    state[14] = block_len;
    state[15] = flags;

    var msg = block;

    // 7 rounds
    for (var r = 0u; r < 7u; r++) {
        blake3_round(&state, &msg, r);
    }

    // XOR with input CV
    return array<u32, 8>(
        state[0] ^ state[8],
        state[1] ^ state[9],
        state[2] ^ state[10],
        state[3] ^ state[11],
        state[4] ^ state[12],
        state[5] ^ state[13],
        state[6] ^ state[14],
        state[7] ^ state[15]
    );
}

// ============================================================================
// Hardware Fingerprint Computation
// ============================================================================

/// Compute unique GPU hardware fingerprint from characteristics
fn compute_fingerprint(chars: GPUCharacteristics) -> array<u32, 8> {
    var block: array<u32, 16>;

    // Pack GPU characteristics into block
    block[0] = FINGERPRINT_DOMAIN;
    block[1] = chars.vendor_id;
    block[2] = chars.device_id;
    block[3] = chars.compute_units;
    block[4] = chars.max_workgroup_size;
    block[5] = chars.max_memory_mb;
    block[6] = chars.clock_mhz;
    block[7] = chars.memory_bandwidth_gbps;
    block[8] = chars.tensor_cores;
    block[9] = chars.tee_supported;
    block[10] = chars.driver_version;

    // Add entropy from workgroup execution characteristics
    // These vary based on actual GPU silicon
    block[11] = 0xDEADBEEFu;  // Placeholder for runtime entropy
    block[12] = 0xCAFEBABEu;
    block[13] = 0x12345678u;
    block[14] = 0x87654321u;
    block[15] = 0xABCDEF00u;

    // Hash to produce fingerprint
    let flags = 0x0Bu; // CHUNK_START | CHUNK_END | ROOT
    return blake3_compress(IV, block, 0u, 64u, flags);
}

// ============================================================================
// Compute Capability Verification
// ============================================================================

/// Calculate approximate TFLOPS * 100 for tier determination
fn estimate_compute_power(chars: GPUCharacteristics) -> u32 {
    // Estimate: CUs * clock * ops_per_cycle / 1e9 * 100
    // Simplified: (CUs * clock_mhz * 128) / 10000
    // 128 = typical FP32 ops per SM per cycle for modern GPUs
    let tflops_x100 = (chars.compute_units * chars.clock_mhz * 128u) / 10000u;

    // Bonus for tensor cores (significant FP16/TF32 acceleration)
    let tensor_bonus = chars.tensor_cores * 10u;

    return tflops_x100 + tensor_bonus;
}

/// Determine GPU tier from hardware characteristics
fn determine_tier(chars: GPUCharacteristics) -> u32 {
    let compute_power = estimate_compute_power(chars);

    // TEE support enables Sovereign tier if compute power is sufficient
    if (chars.tee_supported != 0u && compute_power >= TIER_DATACENTER_MIN) {
        return GPU_TIER_SOVEREIGN;
    }

    if (compute_power >= TIER_DATACENTER_MIN) {
        return GPU_TIER_DATACENTER;
    }

    if (compute_power >= TIER_PROFESSIONAL_MIN) {
        return GPU_TIER_PROFESSIONAL;
    }

    if (compute_power >= TIER_CONSUMER_MIN) {
        return GPU_TIER_CONSUMER;
    }

    // Below minimum threshold - still Consumer but may fail attestation
    return GPU_TIER_CONSUMER;
}

/// Verify compute capability meets minimum requirements
fn verify_compute_capability(chars: GPUCharacteristics) -> bool {
    // Minimum requirements:
    // - At least 16 compute units
    // - At least 256 max workgroup size
    // - At least 4GB memory
    // - At least 500 MHz clock

    if (chars.compute_units < 16u) { return false; }
    if (chars.max_workgroup_size < 256u) { return false; }
    if (chars.max_memory_mb < 4096u) { return false; }
    if (chars.clock_mhz < 500u) { return false; }

    return true;
}

// ============================================================================
// Attestation Hash Generation
// ============================================================================

/// Generate attestation hash binding session, GPU, nonce, and timestamp
fn generate_attestation_hash(
    session_id: array<u32, 8>,
    gpu_id: array<u32, 8>,
    nonce: array<u32, 2>,
    timestamp: array<u32, 2>,
    chain_id: u32
) -> array<u32, 8> {
    // First block: session_id + gpu_id
    var block1: array<u32, 16>;
    for (var i = 0u; i < 8u; i++) {
        block1[i] = session_id[i];
        block1[i + 8u] = gpu_id[i];
    }

    let cv1 = blake3_compress(IV, block1, 0u, 64u, 0x01u); // CHUNK_START

    // Second block: nonce + timestamp + chain_id + domain
    var block2: array<u32, 16>;
    block2[0] = nonce[0];
    block2[1] = nonce[1];
    block2[2] = timestamp[0];
    block2[3] = timestamp[1];
    block2[4] = chain_id;
    block2[5] = ATTESTATION_DOMAIN;
    // Padding
    for (var i = 6u; i < 16u; i++) {
        block2[i] = 0u;
    }

    // Final hash with ROOT flag
    return blake3_compress(cv1, block2, 1u, 24u, 0x0Au); // CHUNK_END | ROOT
}

/// Generate compute work hash (proof of GPU computation)
fn generate_compute_hash(
    gpu_id: array<u32, 8>,
    nonce: array<u32, 2>,
    iteration: u32
) -> array<u32, 8> {
    var block: array<u32, 16>;

    // Mix GPU fingerprint with nonce and iteration
    for (var i = 0u; i < 8u; i++) {
        block[i] = gpu_id[i] ^ (nonce[0] + i);
    }
    block[8] = nonce[0];
    block[9] = nonce[1];
    block[10] = iteration;
    block[11] = ATTESTATION_DOMAIN;

    // Perform multiple hash rounds to prove compute work
    var cv = IV;
    for (var round = 0u; round < 16u; round++) {
        block[12] = round;
        block[13] = cv[0] ^ cv[4];
        block[14] = cv[1] ^ cv[5];
        block[15] = cv[2] ^ cv[6];

        cv = blake3_compress(cv, block, round, 64u, 0x0Bu);
    }

    return cv;
}

// ============================================================================
// Session Binding with Nonces
// ============================================================================

/// Create session binding commitment
fn create_session_binding(
    session_id: array<u32, 8>,
    gpu_id: array<u32, 8>,
    timestamp: array<u32, 2>
) -> array<u32, 8> {
    var block: array<u32, 16>;

    block[0] = SESSION_DOMAIN;
    block[1] = timestamp[0];
    block[2] = timestamp[1];

    // Interleave session_id and gpu_id for binding
    for (var i = 0u; i < 6u; i++) {
        block[3u + i] = session_id[i] ^ gpu_id[i];
    }

    // Remaining session_id
    block[9] = session_id[6];
    block[10] = session_id[7];
    block[11] = gpu_id[6];
    block[12] = gpu_id[7];

    // Padding
    block[13] = 0u;
    block[14] = 0u;
    block[15] = 0u;

    return blake3_compress(IV, block, 0u, 52u, 0x0Bu);
}

/// Verify nonce meets difficulty target
fn check_difficulty(hash: array<u32, 8>, target: array<u32, 8>) -> bool {
    // Compare hash < target (big-endian comparison)
    for (var i = 0u; i < 8u; i++) {
        if (hash[i] < target[i]) { return true; }
        if (hash[i] > target[i]) { return false; }
    }
    return false; // Equal means not below target
}

/// Increment 64-bit nonce
fn increment_nonce(nonce: array<u32, 2>, amount: u32) -> array<u32, 2> {
    let new_lo = nonce[0] + amount;
    let carry = select(0u, 1u, new_lo < nonce[0]);
    return array<u32, 2>(new_lo, nonce[1] + carry);
}

// ============================================================================
// Main Attestation Kernel
// ============================================================================

/// Generate hardware attestation and work proof
/// Each workgroup thread tries different nonces in parallel
@compute @workgroup_size(256)
fn generate_attestation(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let thread_id = gid.x;
    let batch_idx = thread_id;

    if (batch_idx >= params.batch_size) { return; }

    // Verify compute capability
    if (!verify_compute_capability(gpu_chars)) {
        return;
    }

    // Compute GPU fingerprint
    let gpu_id = compute_fingerprint(gpu_chars);

    // Determine GPU tier
    let tier = determine_tier(gpu_chars);

    // Calculate nonce for this thread
    let nonce = increment_nonce(params.nonce, thread_id);

    // Generate compute hash (proof of work)
    let compute_hash = generate_compute_hash(gpu_id, nonce, thread_id);

    // Generate attestation hash
    let attestation_hash = generate_attestation_hash(
        params.session_id,
        gpu_id,
        nonce,
        params.timestamp,
        params.chain_id
    );

    // Check if meets difficulty target
    if (!check_difficulty(attestation_hash, params.difficulty_target)) {
        return; // Doesn't meet difficulty
    }

    // Generate session binding for TEE-style commitment
    let binding = create_session_binding(params.session_id, gpu_id, params.timestamp);

    // Compute final proof hash (for ECDSA signature)
    var proof_block: array<u32, 16>;
    for (var i = 0u; i < 8u; i++) {
        proof_block[i] = attestation_hash[i];
        proof_block[i + 8u] = compute_hash[i];
    }
    let proof_hash = blake3_compress(IV, proof_block, 0u, 64u, 0x0Bu);

    // Atomically claim output slot
    let slot = atomicAdd(&valid_count, 1u);
    if (slot >= params.batch_size) {
        atomicSub(&valid_count, 1u);
        return;
    }

    // Write proof output
    proofs[slot].session_id = params.session_id;
    proofs[slot].nonce = nonce;
    proofs[slot].gpu_id = gpu_id;
    proofs[slot].compute_hash = compute_hash;
    proofs[slot].timestamp = params.timestamp;
    proofs[slot].tier = tier;
    proofs[slot].proof_hash = proof_hash;
}

// ============================================================================
// Fingerprint-Only Kernel
// ============================================================================

/// Compute only the GPU fingerprint (for registration/verification)
@compute @workgroup_size(256)
fn compute_gpu_fingerprint(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x != 0u) { return; } // Only first thread

    let gpu_id = compute_fingerprint(gpu_chars);
    let tier = determine_tier(gpu_chars);

    // Write to first proof slot as fingerprint result
    proofs[0].gpu_id = gpu_id;
    proofs[0].tier = tier;

    atomicStore(&valid_count, 1u);
}

// ============================================================================
// Batch Verification Kernel
// ============================================================================

/// Verify a batch of attestation proofs
@compute @workgroup_size(256)
fn verify_attestation_batch(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;

    if (idx >= params.batch_size) { return; }

    let proof = proofs[idx];

    // Recompute attestation hash
    let expected_hash = generate_attestation_hash(
        proof.session_id,
        proof.gpu_id,
        proof.nonce,
        proof.timestamp,
        params.chain_id
    );

    // Check difficulty
    if (!check_difficulty(expected_hash, params.difficulty_target)) {
        return; // Invalid - doesn't meet difficulty
    }

    // Verify compute hash
    let expected_compute = generate_compute_hash(proof.gpu_id, proof.nonce, idx);

    var compute_valid = true;
    for (var i = 0u; i < 8u; i++) {
        if (expected_compute[i] != proof.compute_hash[i]) {
            compute_valid = false;
            break;
        }
    }

    if (!compute_valid) {
        return; // Invalid compute hash
    }

    // Count valid proofs
    atomicAdd(&valid_count, 1u);
}

// ============================================================================
// Session Heartbeat Kernel
// ============================================================================

/// Generate heartbeat proof for active session
@compute @workgroup_size(256)
fn generate_heartbeat(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x != 0u) { return; }

    let gpu_id = compute_fingerprint(gpu_chars);

    // Create session binding as heartbeat
    let binding = create_session_binding(
        params.session_id,
        gpu_id,
        params.timestamp
    );

    // Heartbeat is just binding hash with current timestamp
    proofs[0].session_id = params.session_id;
    proofs[0].gpu_id = gpu_id;
    proofs[0].compute_hash = binding;
    proofs[0].timestamp = params.timestamp;
    proofs[0].tier = determine_tier(gpu_chars);

    atomicStore(&valid_count, 1u);
}
