// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// WebGPU Test Utilities - Common helpers for GPU tests
// Updated for wgpu-native 27.x API

#ifndef LUX_WEBGPU_TEST_UTILS_H
#define LUX_WEBGPU_TEST_UTILS_H

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <functional>

#if defined(USE_DAWN_API)
    #include <webgpu/webgpu.h>
#elif defined(USE_WGPU_API)
    #include <wgpu.h>
#endif

namespace lux::test {

// Read WGSL file
inline std::string read_wgsl_file(const char* path) {
    std::ifstream file(path);
    if (!file) {
        fprintf(stderr, "Failed to open WGSL file: %s\n", path);
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// WebGPU context for tests
class WebGPUTestContext {
public:
    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;

    bool init() {
#if defined(USE_WGPU_API)
        // Create instance
        WGPUInstanceDescriptor instanceDesc = {};
        instance = wgpuCreateInstance(&instanceDesc);
        if (!instance) {
            fprintf(stderr, "Failed to create WebGPU instance\n");
            return false;
        }

        // Request adapter using wgpu-native 27.x callback info API
        WGPURequestAdapterOptions adapterOpts = {};
        adapterOpts.powerPreference = WGPUPowerPreference_HighPerformance;

        struct AdapterData {
            WGPUAdapter adapter = nullptr;
            bool done = false;
        } adapterData;

        WGPURequestAdapterCallbackInfo adapterCallbackInfo = {};
        adapterCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
        adapterCallbackInfo.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                                          WGPUStringView message, void* userdata1, void* userdata2) {
            auto* data = static_cast<AdapterData*>(userdata1);
            if (status == WGPURequestAdapterStatus_Success) {
                data->adapter = adapter;
            }
            data->done = true;
        };
        adapterCallbackInfo.userdata1 = &adapterData;
        adapterCallbackInfo.userdata2 = nullptr;

        wgpuInstanceRequestAdapter(instance, &adapterOpts, adapterCallbackInfo);

        // Poll until done
        while (!adapterData.done) {
            wgpuInstanceProcessEvents(instance);
        }

        adapter = adapterData.adapter;
        if (!adapter) {
            fprintf(stderr, "Failed to get WebGPU adapter\n");
            return false;
        }

        // Request device using wgpu-native 27.x callback info API
        WGPUDeviceDescriptor deviceDesc = {};
        struct DeviceData {
            WGPUDevice device = nullptr;
            bool done = false;
        } deviceData;

        WGPURequestDeviceCallbackInfo deviceCallbackInfo = {};
        deviceCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
        deviceCallbackInfo.callback = [](WGPURequestDeviceStatus status, WGPUDevice device,
                                         WGPUStringView message, void* userdata1, void* userdata2) {
            auto* data = static_cast<DeviceData*>(userdata1);
            if (status == WGPURequestDeviceStatus_Success) {
                data->device = device;
            }
            data->done = true;
        };
        deviceCallbackInfo.userdata1 = &deviceData;
        deviceCallbackInfo.userdata2 = nullptr;

        wgpuAdapterRequestDevice(adapter, &deviceDesc, deviceCallbackInfo);

        while (!deviceData.done) {
            wgpuInstanceProcessEvents(instance);
        }

        device = deviceData.device;
        if (!device) {
            fprintf(stderr, "Failed to get WebGPU device\n");
            return false;
        }

        queue = wgpuDeviceGetQueue(device);
        return queue != nullptr;

#elif defined(USE_DAWN_API)
        // Dawn initialization (similar pattern)
        return false; // TODO: Dawn support
#else
        return false;
#endif
    }

    void cleanup() {
#if defined(USE_WGPU_API)
        if (queue) wgpuQueueRelease(queue);
        if (device) wgpuDeviceRelease(device);
        if (adapter) wgpuAdapterRelease(adapter);
        if (instance) wgpuInstanceRelease(instance);
#endif
        queue = nullptr;
        device = nullptr;
        adapter = nullptr;
        instance = nullptr;
    }

    ~WebGPUTestContext() {
        cleanup();
    }

    // Create buffer with data
    WGPUBuffer create_buffer(size_t size, WGPUBufferUsage usage, const void* data = nullptr) {
#if defined(USE_WGPU_API)
        WGPUBufferDescriptor desc = {};
        desc.size = size;
        desc.usage = usage;
        desc.mappedAtCreation = (data != nullptr);

        WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &desc);
        if (buffer && data) {
            void* mapped = wgpuBufferGetMappedRange(buffer, 0, size);
            if (mapped) {
                memcpy(mapped, data, size);
            }
            wgpuBufferUnmap(buffer);
        }
        return buffer;
#else
        return nullptr;
#endif
    }

    // Read buffer back to host
    bool read_buffer(WGPUBuffer buffer, size_t size, void* dest) {
#if defined(USE_WGPU_API)
        // Create staging buffer for readback
        WGPUBufferDescriptor stagingDesc = {};
        stagingDesc.size = size;
        stagingDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
        WGPUBuffer staging = wgpuDeviceCreateBuffer(device, &stagingDesc);

        // Copy from source to staging
        WGPUCommandEncoderDescriptor encDesc = {};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, buffer, 0, staging, 0, size);

        WGPUCommandBufferDescriptor cmdDesc = {};
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
        wgpuQueueSubmit(queue, 1, &cmd);

        wgpuCommandEncoderRelease(encoder);
        wgpuCommandBufferRelease(cmd);

        // Map and read using wgpu-native 27.x callback info API
        struct MapData {
            bool done = false;
            WGPUMapAsyncStatus status;
        } mapData;

        WGPUBufferMapCallbackInfo mapCallbackInfo = {};
        mapCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
        mapCallbackInfo.callback = [](WGPUMapAsyncStatus status, WGPUStringView message,
                                      void* userdata1, void* userdata2) {
            auto* data = static_cast<MapData*>(userdata1);
            data->status = status;
            data->done = true;
        };
        mapCallbackInfo.userdata1 = &mapData;
        mapCallbackInfo.userdata2 = nullptr;

        wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, size, mapCallbackInfo);

        while (!mapData.done) {
            wgpuInstanceProcessEvents(instance);
        }

        if (mapData.status == WGPUMapAsyncStatus_Success) {
            const void* mapped = wgpuBufferGetConstMappedRange(staging, 0, size);
            if (mapped) {
                memcpy(dest, mapped, size);
            }
            wgpuBufferUnmap(staging);
        }

        wgpuBufferRelease(staging);
        return mapData.status == WGPUMapAsyncStatus_Success;
#else
        return false;
#endif
    }

    // Compile shader from WGSL source
    WGPUShaderModule create_shader(const char* wgsl_source) {
#if defined(USE_WGPU_API)
        WGPUShaderSourceWGSL wgslDesc = {};
        wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
        wgslDesc.code.data = wgsl_source;
        wgslDesc.code.length = strlen(wgsl_source);

        WGPUShaderModuleDescriptor desc = {};
        desc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wgslDesc);

        return wgpuDeviceCreateShaderModule(device, &desc);
#else
        return nullptr;
#endif
    }

    // Create compute pipeline
    WGPUComputePipeline create_compute_pipeline(WGPUShaderModule shader, const char* entry_point) {
#if defined(USE_WGPU_API)
        WGPUComputePipelineDescriptor desc = {};
        desc.compute.module = shader;
        desc.compute.entryPoint.data = entry_point;
        desc.compute.entryPoint.length = strlen(entry_point);

        return wgpuDeviceCreateComputePipeline(device, &desc);
#else
        return nullptr;
#endif
    }

    // Submit and wait for completion
    void submit_and_wait(WGPUCommandBuffer cmd) {
#if defined(USE_WGPU_API)
        wgpuQueueSubmit(queue, 1, &cmd);

        // Wait for completion using wgpu-native 27.x callback info API
        struct DoneData {
            bool done = false;
        } doneData;

        WGPUQueueWorkDoneCallbackInfo workDoneCallbackInfo = {};
        workDoneCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
#if defined(USE_DAWN_API)
        workDoneCallbackInfo.callback = [](WGPUQueueWorkDoneStatus status,
                                           void* userdata1, void* userdata2) {
            auto* data = static_cast<DoneData*>(userdata1);
            (void)status; (void)userdata2;
            data->done = true;
        };
#else
        workDoneCallbackInfo.callback = [](WGPUQueueWorkDoneStatus status,
                                           WGPUStringView,
                                           void* userdata1, void* userdata2) {
            auto* data = static_cast<DoneData*>(userdata1);
            (void)status; (void)userdata2;
            data->done = true;
        };
#endif
        workDoneCallbackInfo.userdata1 = &doneData;
        workDoneCallbackInfo.userdata2 = nullptr;

        wgpuQueueOnSubmittedWorkDone(queue, workDoneCallbackInfo);

        while (!doneData.done) {
            wgpuInstanceProcessEvents(instance);
        }
#endif
    }
};

// RAII buffer wrapper
class TestBuffer {
public:
    WGPUBuffer buffer = nullptr;
    size_t size = 0;

    TestBuffer() = default;
    TestBuffer(WGPUBuffer b, size_t s) : buffer(b), size(s) {}

    ~TestBuffer() {
#if defined(USE_WGPU_API)
        if (buffer) wgpuBufferRelease(buffer);
#endif
    }

    TestBuffer(const TestBuffer&) = delete;
    TestBuffer& operator=(const TestBuffer&) = delete;

    TestBuffer(TestBuffer&& other) noexcept : buffer(other.buffer), size(other.size) {
        other.buffer = nullptr;
        other.size = 0;
    }

    TestBuffer& operator=(TestBuffer&& other) noexcept {
#if defined(USE_WGPU_API)
        if (buffer) wgpuBufferRelease(buffer);
#endif
        buffer = other.buffer;
        size = other.size;
        other.buffer = nullptr;
        other.size = 0;
        return *this;
    }
};

} // namespace lux::test

#endif // LUX_WEBGPU_TEST_UTILS_H
