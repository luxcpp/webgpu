// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Reduction Kernel Tests
// Tests Barrett reduction, Montgomery reduction for ML-DSA/ML-KEM
// These are critical for post-quantum lattice cryptography

#include "test_framework.h"
#include "webgpu_test_utils.h"
#include <filesystem>
#include <cmath>
#include <random>

namespace fs = std::filesystem;
using namespace lux::test;

// Global test context
static WebGPUTestContext* g_ctx = nullptr;
static WGPUShaderModule g_reduce_shader = nullptr;

// ML-DSA (Dilithium) constants
static const int32_t DILITHIUM_Q = 8380417;
static const int32_t DILITHIUM_Q_HALF = 4190208;

// ML-KEM (Kyber) constants
static const int32_t KYBER_Q = 3329;
static const int32_t KYBER_Q_HALF = 1664;

// Parameters struct matching WGSL
struct ReduceParams {
    uint32_t size;
    int32_t gamma2;
    uint32_t d;
    uint32_t _pad;
};

// =============================================================================
// Reference Implementations
// =============================================================================

// Reference full reduction for Dilithium (canonical mod q)
static int32_t ref_full_reduce_dilithium(int32_t a) {
    // Use int64_t for safe modulo on negative values
    int64_t t = static_cast<int64_t>(a) % static_cast<int64_t>(DILITHIUM_Q);
    if (t < 0) t += DILITHIUM_Q;
    return static_cast<int32_t>(t);
}

// Reference centered reduction for Dilithium
static int32_t ref_centered_reduce_dilithium(int32_t a) {
    int32_t t = ref_full_reduce_dilithium(a);
    if (t > DILITHIUM_Q_HALF) t -= DILITHIUM_Q;
    return t;
}

// Reference full reduction for Kyber
static int16_t ref_full_reduce_kyber(int16_t a) {
    int32_t t = static_cast<int32_t>(a) % static_cast<int32_t>(KYBER_Q);
    if (t < 0) t += KYBER_Q;
    return static_cast<int16_t>(t);
}

// =============================================================================
// Test Setup
// =============================================================================

static bool setup_context() {
    if (g_ctx) return true;

    g_ctx = new WebGPUTestContext();
    if (!g_ctx->init()) {
        delete g_ctx;
        g_ctx = nullptr;
        return false;
    }

    // Load reduce.wgsl
    fs::path kernel_path = fs::path(WGSL_KERNEL_DIR) / "gpu" / "reduce.wgsl";
    std::string source = read_wgsl_file(kernel_path.c_str());
    if (source.empty()) {
        g_ctx->cleanup();
        delete g_ctx;
        g_ctx = nullptr;
        return false;
    }

    g_reduce_shader = g_ctx->create_shader(source.c_str());
    if (!g_reduce_shader) {
        g_ctx->cleanup();
        delete g_ctx;
        g_ctx = nullptr;
        return false;
    }

    return true;
}

// =============================================================================
// Dilithium Reduction Tests (CPU reference)
// =============================================================================

TEST_CASE(dilithium_full_reduce_reference) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(-DILITHIUM_Q * 10, DILITHIUM_Q * 10);

    for (int i = 0; i < 1000; i++) {
        int32_t a = dist(rng);
        int32_t reduced = ref_full_reduce_dilithium(a);

        // Result should be in [0, q)
        TEST_ASSERT(reduced >= 0);
        TEST_ASSERT(reduced < DILITHIUM_Q);

        // Should be congruent to original mod q
        int64_t expected = static_cast<int64_t>(a) % static_cast<int64_t>(DILITHIUM_Q);
        if (expected < 0) expected += DILITHIUM_Q;
        TEST_ASSERT_EQ(static_cast<int32_t>(expected), reduced);
    }

    return true;
}

TEST_CASE(dilithium_centered_reduce_reference) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(-DILITHIUM_Q * 10, DILITHIUM_Q * 10);

    for (int i = 0; i < 1000; i++) {
        int32_t a = dist(rng);
        int32_t reduced = ref_centered_reduce_dilithium(a);

        // Result should be in [-q/2, q/2]
        TEST_ASSERT(reduced >= -DILITHIUM_Q_HALF);
        TEST_ASSERT(reduced <= DILITHIUM_Q_HALF);

        // Should be congruent to original mod q
        int64_t expected = static_cast<int64_t>(a) % static_cast<int64_t>(DILITHIUM_Q);
        if (expected < 0) expected += DILITHIUM_Q;
        int32_t actual = reduced;
        if (actual < 0) actual += DILITHIUM_Q;
        TEST_ASSERT_EQ(static_cast<int32_t>(expected), actual);
    }

    return true;
}

// =============================================================================
// Kyber Reduction Tests (CPU reference)
// =============================================================================

TEST_CASE(kyber_full_reduce_reference) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int16_t> dist(-KYBER_Q * 5, KYBER_Q * 5);

    for (int i = 0; i < 1000; i++) {
        int16_t a = dist(rng);
        int16_t reduced = ref_full_reduce_kyber(a);

        // Result should be in [0, q)
        TEST_ASSERT(reduced >= 0);
        TEST_ASSERT(reduced < KYBER_Q);

        // Should be congruent to original mod q
        int32_t expected = static_cast<int32_t>(a) % static_cast<int32_t>(KYBER_Q);
        if (expected < 0) expected += KYBER_Q;
        TEST_ASSERT_EQ(static_cast<int16_t>(expected), reduced);
    }

    return true;
}

// =============================================================================
// GPU Shader Compilation Tests
// =============================================================================

TEST_CASE(reduce_shader_compiles) {
    if (!setup_context()) {
        printf("(skipped - no WebGPU) ");
        return true;
    }
    TEST_ASSERT(g_reduce_shader != nullptr);
    return true;
}

TEST_CASE(reduce_barrett_dilithium_pipeline) {
    if (!setup_context()) {
        printf("(skipped - no WebGPU) ");
        return true;
    }

#if defined(USE_WGPU_API)
    WGPUComputePipeline pipeline = g_ctx->create_compute_pipeline(
        g_reduce_shader, "reduce_barrett_dilithium");
    TEST_ASSERT(pipeline != nullptr);
    wgpuComputePipelineRelease(pipeline);
#endif
    return true;
}

TEST_CASE(reduce_full_dilithium_pipeline) {
    if (!setup_context()) {
        printf("(skipped - no WebGPU) ");
        return true;
    }

#if defined(USE_WGPU_API)
    WGPUComputePipeline pipeline = g_ctx->create_compute_pipeline(
        g_reduce_shader, "reduce_full_dilithium");
    TEST_ASSERT(pipeline != nullptr);
    wgpuComputePipelineRelease(pipeline);
#endif
    return true;
}

TEST_CASE(reduce_centered_dilithium_pipeline) {
    if (!setup_context()) {
        printf("(skipped - no WebGPU) ");
        return true;
    }

#if defined(USE_WGPU_API)
    WGPUComputePipeline pipeline = g_ctx->create_compute_pipeline(
        g_reduce_shader, "reduce_centered_dilithium");
    TEST_ASSERT(pipeline != nullptr);
    wgpuComputePipelineRelease(pipeline);
#endif
    return true;
}

TEST_CASE(reduce_barrett_kyber_pipeline) {
    if (!setup_context()) {
        printf("(skipped - no WebGPU) ");
        return true;
    }

#if defined(USE_WGPU_API)
    WGPUComputePipeline pipeline = g_ctx->create_compute_pipeline(
        g_reduce_shader, "reduce_barrett_kyber");
    TEST_ASSERT(pipeline != nullptr);
    wgpuComputePipelineRelease(pipeline);
#endif
    return true;
}

TEST_CASE(montgomery_dilithium_pipelines) {
    if (!setup_context()) {
        printf("(skipped - no WebGPU) ");
        return true;
    }

#if defined(USE_WGPU_API)
    WGPUComputePipeline to_mont = g_ctx->create_compute_pipeline(
        g_reduce_shader, "to_montgomery_dilithium");
    TEST_ASSERT(to_mont != nullptr);

    WGPUComputePipeline from_mont = g_ctx->create_compute_pipeline(
        g_reduce_shader, "from_montgomery_dilithium");
    TEST_ASSERT(from_mont != nullptr);

    wgpuComputePipelineRelease(to_mont);
    wgpuComputePipelineRelease(from_mont);
#endif
    return true;
}

TEST_CASE(highbits_lowbits_dilithium_pipelines) {
    if (!setup_context()) {
        printf("(skipped - no WebGPU) ");
        return true;
    }

#if defined(USE_WGPU_API)
    WGPUComputePipeline high = g_ctx->create_compute_pipeline(
        g_reduce_shader, "highbits_dilithium");
    TEST_ASSERT(high != nullptr);

    WGPUComputePipeline low = g_ctx->create_compute_pipeline(
        g_reduce_shader, "lowbits_dilithium");
    TEST_ASSERT(low != nullptr);

    wgpuComputePipelineRelease(high);
    wgpuComputePipelineRelease(low);
#endif
    return true;
}

TEST_CASE(hint_operations_dilithium_pipelines) {
    if (!setup_context()) {
        printf("(skipped - no WebGPU) ");
        return true;
    }

#if defined(USE_WGPU_API)
    WGPUComputePipeline make_hint = g_ctx->create_compute_pipeline(
        g_reduce_shader, "make_hint_dilithium");
    TEST_ASSERT(make_hint != nullptr);

    WGPUComputePipeline use_hint = g_ctx->create_compute_pipeline(
        g_reduce_shader, "use_hint_dilithium");
    TEST_ASSERT(use_hint != nullptr);

    wgpuComputePipelineRelease(make_hint);
    wgpuComputePipelineRelease(use_hint);
#endif
    return true;
}

TEST_CASE(compress_decompress_kyber_pipelines) {
    if (!setup_context()) {
        printf("(skipped - no WebGPU) ");
        return true;
    }

#if defined(USE_WGPU_API)
    WGPUComputePipeline compress = g_ctx->create_compute_pipeline(
        g_reduce_shader, "compress_kyber");
    TEST_ASSERT(compress != nullptr);

    WGPUComputePipeline decompress = g_ctx->create_compute_pipeline(
        g_reduce_shader, "decompress_kyber");
    TEST_ASSERT(decompress != nullptr);

    wgpuComputePipelineRelease(compress);
    wgpuComputePipelineRelease(decompress);
#endif
    return true;
}

TEST_CASE(freeze_pipelines) {
    if (!setup_context()) {
        printf("(skipped - no WebGPU) ");
        return true;
    }

#if defined(USE_WGPU_API)
    WGPUComputePipeline freeze_d = g_ctx->create_compute_pipeline(
        g_reduce_shader, "freeze_dilithium");
    TEST_ASSERT(freeze_d != nullptr);

    WGPUComputePipeline freeze_k = g_ctx->create_compute_pipeline(
        g_reduce_shader, "freeze_kyber");
    TEST_ASSERT(freeze_k != nullptr);

    wgpuComputePipelineRelease(freeze_d);
    wgpuComputePipelineRelease(freeze_k);
#endif
    return true;
}

TEST_CASE(vectorized_reduce_pipelines) {
    if (!setup_context()) {
        printf("(skipped - no WebGPU) ");
        return true;
    }

#if defined(USE_WGPU_API)
    WGPUComputePipeline barrett_vec4 = g_ctx->create_compute_pipeline(
        g_reduce_shader, "reduce_barrett_dilithium_vec4");
    TEST_ASSERT(barrett_vec4 != nullptr);

    WGPUComputePipeline full_vec4 = g_ctx->create_compute_pipeline(
        g_reduce_shader, "reduce_full_dilithium_vec4");
    TEST_ASSERT(full_vec4 != nullptr);

    wgpuComputePipelineRelease(barrett_vec4);
    wgpuComputePipelineRelease(full_vec4);
#endif
    return true;
}

// =============================================================================
// Known Value Tests
// =============================================================================

TEST_CASE(dilithium_known_values) {
    // Test specific known values
    TEST_ASSERT_EQ(ref_full_reduce_dilithium(0), 0);
    TEST_ASSERT_EQ(ref_full_reduce_dilithium(DILITHIUM_Q), 0);
    TEST_ASSERT_EQ(ref_full_reduce_dilithium(DILITHIUM_Q - 1), DILITHIUM_Q - 1);
    TEST_ASSERT_EQ(ref_full_reduce_dilithium(-1), DILITHIUM_Q - 1);
    TEST_ASSERT_EQ(ref_full_reduce_dilithium(-DILITHIUM_Q), 0);
    TEST_ASSERT_EQ(ref_full_reduce_dilithium(2 * DILITHIUM_Q), 0);
    TEST_ASSERT_EQ(ref_full_reduce_dilithium(-2 * DILITHIUM_Q), 0);

    return true;
}

TEST_CASE(kyber_known_values) {
    // Test specific known values
    TEST_ASSERT_EQ(ref_full_reduce_kyber(0), static_cast<int16_t>(0));
    TEST_ASSERT_EQ(ref_full_reduce_kyber(KYBER_Q), static_cast<int16_t>(0));
    TEST_ASSERT_EQ(ref_full_reduce_kyber(KYBER_Q - 1), static_cast<int16_t>(KYBER_Q - 1));
    TEST_ASSERT_EQ(ref_full_reduce_kyber(-1), static_cast<int16_t>(KYBER_Q - 1));
    TEST_ASSERT_EQ(ref_full_reduce_kyber(-KYBER_Q), static_cast<int16_t>(0));

    return true;
}

TEST_MAIN()
