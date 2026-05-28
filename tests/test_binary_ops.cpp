// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Binary Operations Kernel Tests
// Tests add, sub, mul, div, comparisons, etc.

#include "test_framework.h"
#include "webgpu_test_utils.h"
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;
using namespace lux::test;

// Global test context
static WebGPUTestContext* g_ctx = nullptr;
static WGPUShaderModule g_binary_shader = nullptr;

// Parameters struct matching WGSL
struct BinaryParams {
    uint32_t size;
    uint32_t a_stride;
    uint32_t b_stride;
    uint32_t _pad;
};

// =============================================================================
// Test Setup/Teardown
// =============================================================================

static bool setup_context() {
    if (g_ctx) return true;

    g_ctx = new WebGPUTestContext();
    if (!g_ctx->init()) {
        delete g_ctx;
        g_ctx = nullptr;
        return false;
    }

    // Load binary.wgsl
    fs::path kernel_path = fs::path(WGSL_KERNEL_DIR) / "gpu" / "binary.wgsl";
    std::string source = read_wgsl_file(kernel_path.c_str());
    if (source.empty()) {
        g_ctx->cleanup();
        delete g_ctx;
        g_ctx = nullptr;
        return false;
    }

    g_binary_shader = g_ctx->create_shader(source.c_str());
    if (!g_binary_shader) {
        g_ctx->cleanup();
        delete g_ctx;
        g_ctx = nullptr;
        return false;
    }

    return true;
}

// Run a binary operation test
static bool run_binary_op_test(
    const char* entry_point,
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& expected,
    float tolerance = 1e-5f
) {
    if (!setup_context()) {
        printf("(skipped - no WebGPU) ");
        return true;
    }

#if defined(USE_WGPU_API)
    const size_t n = expected.size();
    const size_t data_size = n * sizeof(float);

    // Create pipeline
    WGPUComputePipeline pipeline = g_ctx->create_compute_pipeline(g_binary_shader, entry_point);
    if (!pipeline) {
        fprintf(stderr, "Failed to create pipeline for %s\n", entry_point);
        return false;
    }

    // Create buffers
    WGPUBuffer buf_a = g_ctx->create_buffer(data_size, WGPUBufferUsage_Storage, a.data());
    WGPUBuffer buf_b = g_ctx->create_buffer(data_size, WGPUBufferUsage_Storage, b.data());
    WGPUBuffer buf_out = g_ctx->create_buffer(data_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);

    // Create params buffer
    BinaryParams params = {
        static_cast<uint32_t>(n),
        1, // a_stride
        1, // b_stride
        0
    };
    WGPUBuffer buf_params = g_ctx->create_buffer(sizeof(params), WGPUBufferUsage_Uniform, &params);

    // Create bind group
    WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);

    WGPUBindGroupEntry entries[4] = {};
    entries[0].binding = 0;
    entries[0].buffer = buf_a;
    entries[0].size = data_size;

    entries[1].binding = 1;
    entries[1].buffer = buf_b;
    entries[1].size = data_size;

    entries[2].binding = 2;
    entries[2].buffer = buf_out;
    entries[2].size = data_size;

    entries[3].binding = 3;
    entries[3].buffer = buf_params;
    entries[3].size = sizeof(params);

    WGPUBindGroupDescriptor bgDesc = {};
    bgDesc.layout = layout;
    bgDesc.entryCount = 4;
    bgDesc.entries = entries;

    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(g_ctx->device, &bgDesc);

    // Create command buffer
    WGPUCommandEncoderDescriptor encDesc = {};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(g_ctx->device, &encDesc);

    WGPUComputePassDescriptor passDesc = {};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);

    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);

    uint32_t num_workgroups = (n + 255) / 256;
    wgpuComputePassEncoderDispatchWorkgroups(pass, num_workgroups, 1, 1);
    wgpuComputePassEncoderEnd(pass);

    WGPUCommandBufferDescriptor cmdDesc = {};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);

    g_ctx->submit_and_wait(cmd);

    // Read back results
    std::vector<float> result(n);
    bool read_ok = g_ctx->read_buffer(buf_out, data_size, result.data());

    // Cleanup
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuCommandBufferRelease(cmd);
    wgpuBindGroupRelease(bindGroup);
    wgpuBindGroupLayoutRelease(layout);
    wgpuBufferRelease(buf_a);
    wgpuBufferRelease(buf_b);
    wgpuBufferRelease(buf_out);
    wgpuBufferRelease(buf_params);
    wgpuComputePipelineRelease(pipeline);

    if (!read_ok) {
        fprintf(stderr, "Failed to read output buffer\n");
        return false;
    }

    // Verify results
    for (size_t i = 0; i < n; i++) {
        if (std::abs(result[i] - expected[i]) > tolerance) {
            fprintf(stderr, "  Mismatch at index %zu: expected %g, got %g\n",
                    i, expected[i], result[i]);
            return false;
        }
    }

    return true;
#else
    printf("(skipped - no WebGPU) ");
    return true;
#endif
}

// =============================================================================
// Test Cases
// =============================================================================

TEST_CASE(binary_add_basic) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> expected = {6.0f, 8.0f, 10.0f, 12.0f};

    return run_binary_op_test("add", a, b, expected);
}

TEST_CASE(binary_add_large) {
    const size_t n = 10000;
    std::vector<float> a(n), b(n), expected(n);

    for (size_t i = 0; i < n; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
        expected[i] = a[i] + b[i];
    }

    return run_binary_op_test("add", a, b, expected);
}

TEST_CASE(binary_sub_basic) {
    std::vector<float> a = {10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> expected = {9.0f, 18.0f, 27.0f, 36.0f};

    return run_binary_op_test("sub", a, b, expected);
}

TEST_CASE(binary_mul_basic) {
    std::vector<float> a = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> b = {3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> expected = {6.0f, 12.0f, 20.0f, 30.0f};

    return run_binary_op_test("mul", a, b, expected);
}

TEST_CASE(binary_div_basic) {
    std::vector<float> a = {10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> b = {2.0f, 4.0f, 5.0f, 8.0f};
    std::vector<float> expected = {5.0f, 5.0f, 6.0f, 5.0f};

    return run_binary_op_test("div", a, b, expected);
}

TEST_CASE(binary_minimum_basic) {
    std::vector<float> a = {1.0f, 5.0f, 3.0f, 8.0f};
    std::vector<float> b = {4.0f, 2.0f, 3.0f, 1.0f};
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 1.0f};

    return run_binary_op_test("minimum", a, b, expected);
}

TEST_CASE(binary_maximum_basic) {
    std::vector<float> a = {1.0f, 5.0f, 3.0f, 8.0f};
    std::vector<float> b = {4.0f, 2.0f, 3.0f, 1.0f};
    std::vector<float> expected = {4.0f, 5.0f, 3.0f, 8.0f};

    return run_binary_op_test("maximum", a, b, expected);
}

TEST_CASE(binary_equal_basic) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {1.0f, 5.0f, 3.0f, 8.0f};
    std::vector<float> expected = {1.0f, 0.0f, 1.0f, 0.0f};

    return run_binary_op_test("equal", a, b, expected);
}

TEST_CASE(binary_less_basic) {
    std::vector<float> a = {1.0f, 5.0f, 3.0f, 4.0f};
    std::vector<float> b = {2.0f, 3.0f, 3.0f, 8.0f};
    std::vector<float> expected = {1.0f, 0.0f, 0.0f, 1.0f};

    return run_binary_op_test("less", a, b, expected);
}

TEST_CASE(binary_greater_basic) {
    std::vector<float> a = {1.0f, 5.0f, 3.0f, 4.0f};
    std::vector<float> b = {2.0f, 3.0f, 3.0f, 8.0f};
    std::vector<float> expected = {0.0f, 1.0f, 0.0f, 0.0f};

    return run_binary_op_test("greater", a, b, expected);
}

TEST_CASE(binary_add_negative) {
    std::vector<float> a = {-1.0f, -2.0f, 3.0f, -4.0f};
    std::vector<float> b = {5.0f, -6.0f, -7.0f, 8.0f};
    std::vector<float> expected = {4.0f, -8.0f, -4.0f, 4.0f};

    return run_binary_op_test("add", a, b, expected);
}

TEST_CASE(binary_mul_zero) {
    std::vector<float> a = {0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> b = {5.0f, 0.0f, 0.0f, 4.0f};
    std::vector<float> expected = {0.0f, 0.0f, 0.0f, 12.0f};

    return run_binary_op_test("mul", a, b, expected);
}

TEST_CASE(binary_logical_and) {
    std::vector<float> a = {0.0f, 1.0f, 1.0f, 0.0f};
    std::vector<float> b = {0.0f, 0.0f, 1.0f, 1.0f};
    std::vector<float> expected = {0.0f, 0.0f, 1.0f, 0.0f};

    return run_binary_op_test("logical_and", a, b, expected);
}

TEST_CASE(binary_logical_or) {
    std::vector<float> a = {0.0f, 1.0f, 1.0f, 0.0f};
    std::vector<float> b = {0.0f, 0.0f, 1.0f, 1.0f};
    std::vector<float> expected = {0.0f, 1.0f, 1.0f, 1.0f};

    return run_binary_op_test("logical_or", a, b, expected);
}

TEST_MAIN()
