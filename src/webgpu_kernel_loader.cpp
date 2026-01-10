// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// WebGPU Kernel Loader - Compile WGSL and cache compute pipelines

#include "lux/gpu/kernel_loader.h"
#include <unordered_map>
#include <string>
#include <mutex>
#include <vector>
#include <cstring>
#include <fstream>

#ifdef LUX_GPU_BUILD_WEBGPU

#if defined(USE_DAWN_API)
#include <webgpu/webgpu_cpp.h>
#elif defined(USE_WGPU_API)
#include <wgpu.h>
#endif

namespace lux::gpu::webgpu {

// =============================================================================
// WebGPU Kernel Implementation
// =============================================================================

struct WgpuKernel {
#if defined(USE_DAWN_API)
    wgpu::ShaderModule shader_module;
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bind_group_layout;
#elif defined(USE_WGPU_API)
    WGPUShaderModule shader_module;
    WGPUComputePipeline pipeline;
    WGPUBindGroupLayout bind_group_layout;
#endif
    std::string entry_point;
    uint32_t workgroup_size_x;
    uint32_t workgroup_size_y;
    uint32_t workgroup_size_z;
};

// =============================================================================
// WebGPU Kernel Cache
// =============================================================================

class WgpuKernelCache {
public:
    static WgpuKernelCache& instance() {
        static WgpuKernelCache cache;
        return cache;
    }

#if defined(USE_DAWN_API)
    // Dawn API implementation
    WgpuKernel* compile(
        wgpu::Device device,
        const char* wgsl_source,
        size_t source_len,
        const char* entry_point
    ) {
        // Create shader module
        wgpu::ShaderModuleWGSLDescriptor wgsl_desc{};
        wgsl_desc.code = wgsl_source;

        wgpu::ShaderModuleDescriptor shader_desc{};
        shader_desc.nextInChain = &wgsl_desc;

        wgpu::ShaderModule module = device.CreateShaderModule(&shader_desc);
        if (!module) {
            return nullptr;
        }

        // Create compute pipeline
        wgpu::ComputePipelineDescriptor pipeline_desc{};
        pipeline_desc.compute.module = module;
        pipeline_desc.compute.entryPoint = entry_point;

        wgpu::ComputePipeline pipeline = device.CreateComputePipeline(&pipeline_desc);
        if (!pipeline) {
            return nullptr;
        }

        // Get bind group layout
        wgpu::BindGroupLayout layout = pipeline.GetBindGroupLayout(0);

        auto kernel = new WgpuKernel();
        kernel->shader_module = module;
        kernel->pipeline = pipeline;
        kernel->bind_group_layout = layout;
        kernel->entry_point = entry_point;
        kernel->workgroup_size_x = 256;  // Default, can be extracted from reflection
        kernel->workgroup_size_y = 1;
        kernel->workgroup_size_z = 1;

        return kernel;
    }

#elif defined(USE_WGPU_API)
    // wgpu-native 27.x API implementation
    WgpuKernel* compile(
        WGPUDevice device,
        const char* wgsl_source,
        size_t source_len,
        const char* entry_point
    ) {
        // Create shader module with WGSL source (wgpu-native 27.x API)
        WGPUShaderSourceWGSL wgsl_desc = {};
        wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
        wgsl_desc.code.data = wgsl_source;
        wgsl_desc.code.length = source_len > 0 ? source_len : WGPU_STRLEN;

        WGPUShaderModuleDescriptor shader_desc = {};
        shader_desc.nextInChain = reinterpret_cast<WGPUChainedStruct const*>(&wgsl_desc);

        WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &shader_desc);
        if (!module) {
            return nullptr;
        }

        // Create compute pipeline (wgpu-native 27.x uses WGPUStringView for entry point)
        WGPUComputePipelineDescriptor pipeline_desc = {};
        pipeline_desc.compute.module = module;
        pipeline_desc.compute.entryPoint.data = entry_point;
        pipeline_desc.compute.entryPoint.length = WGPU_STRLEN;

        WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &pipeline_desc);
        if (!pipeline) {
            wgpuShaderModuleRelease(module);
            return nullptr;
        }

        // Get bind group layout
        WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);

        auto kernel = new WgpuKernel();
        kernel->shader_module = module;
        kernel->pipeline = pipeline;
        kernel->bind_group_layout = layout;
        kernel->entry_point = entry_point;
        kernel->workgroup_size_x = 256;
        kernel->workgroup_size_y = 1;
        kernel->workgroup_size_z = 1;

        return kernel;
    }
#endif

    // Load from file
    WgpuKernel* load_file(
        void* device,
        const char* path,
        const char* entry_point
    ) {
        // Read file
        std::ifstream file(path);
        if (!file) {
            fprintf(stderr, "WgpuKernelCache: Failed to open file: %s\n", path);
            return nullptr;
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

#if defined(USE_DAWN_API)
        return compile(static_cast<wgpu::Device>(device), content.c_str(), content.size(), entry_point);
#elif defined(USE_WGPU_API)
        return compile(static_cast<WGPUDevice>(device), content.c_str(), content.size(), entry_point);
#else
        return nullptr;
#endif
    }

    void destroy(WgpuKernel* kernel) {
        if (!kernel) return;

#if defined(USE_WGPU_API)
        if (kernel->bind_group_layout) {
            wgpuBindGroupLayoutRelease(kernel->bind_group_layout);
        }
        if (kernel->pipeline) {
            wgpuComputePipelineRelease(kernel->pipeline);
        }
        if (kernel->shader_module) {
            wgpuShaderModuleRelease(kernel->shader_module);
        }
#endif
        // Dawn uses reference counting, automatic cleanup

        delete kernel;
    }

private:
    WgpuKernelCache() = default;
    std::mutex mutex_;
};

} // namespace lux::gpu::webgpu

// =============================================================================
// C API Implementation for WebGPU
// =============================================================================

extern "C" {

using namespace lux::gpu::webgpu;

LuxKernel* lux_wgpu_kernel_compile(
    void* device,
    const char* wgsl_source,
    size_t source_len,
    const char* entry_point
) {
    if (!device || !wgsl_source || !entry_point) return nullptr;

#if defined(USE_DAWN_API)
    auto kernel = WgpuKernelCache::instance().compile(
        static_cast<wgpu::Device>(device), wgsl_source, source_len, entry_point
    );
#elif defined(USE_WGPU_API)
    auto kernel = WgpuKernelCache::instance().compile(
        static_cast<WGPUDevice>(device), wgsl_source, source_len, entry_point
    );
#else
    WgpuKernel* kernel = nullptr;
#endif
    return reinterpret_cast<LuxKernel*>(kernel);
}

LuxKernel* lux_wgpu_kernel_load_file(
    void* device,
    const char* path,
    const char* entry_point
) {
    if (!device || !path || !entry_point) return nullptr;

    auto kernel = WgpuKernelCache::instance().load_file(device, path, entry_point);
    return reinterpret_cast<LuxKernel*>(kernel);
}

void lux_wgpu_kernel_destroy(LuxKernel* kernel) {
    WgpuKernelCache::instance().destroy(reinterpret_cast<WgpuKernel*>(kernel));
}

const char* lux_wgpu_kernel_entry_point(LuxKernel* kernel) {
    auto k = reinterpret_cast<WgpuKernel*>(kernel);
    return k ? k->entry_point.c_str() : nullptr;
}

// Get WGPUComputePipeline for dispatch
void* lux_wgpu_kernel_pipeline(LuxKernel* kernel) {
    auto k = reinterpret_cast<WgpuKernel*>(kernel);
    if (!k) return nullptr;
#if defined(USE_DAWN_API)
    return k->pipeline.Get();
#elif defined(USE_WGPU_API)
    return k->pipeline;
#else
    return nullptr;
#endif
}

// Get bind group layout
void* lux_wgpu_kernel_bind_group_layout(LuxKernel* kernel) {
    auto k = reinterpret_cast<WgpuKernel*>(kernel);
    if (!k) return nullptr;
#if defined(USE_DAWN_API)
    return k->bind_group_layout.Get();
#elif defined(USE_WGPU_API)
    return k->bind_group_layout;
#else
    return nullptr;
#endif
}

// Get workgroup size
void lux_wgpu_kernel_workgroup_size(LuxKernel* kernel, uint32_t* x, uint32_t* y, uint32_t* z) {
    auto k = reinterpret_cast<WgpuKernel*>(kernel);
    if (!k) {
        if (x) *x = 1;
        if (y) *y = 1;
        if (z) *z = 1;
        return;
    }
    if (x) *x = k->workgroup_size_x;
    if (y) *y = k->workgroup_size_y;
    if (z) *z = k->workgroup_size_z;
}

} // extern "C"

// =============================================================================
// WebGPU Kernel Manager - High-level interface
// =============================================================================

namespace lux::gpu::webgpu {

class WgpuKernelManager {
public:
#if defined(USE_DAWN_API)
    WgpuKernelManager(wgpu::Device device) : device_(device) {
#elif defined(USE_WGPU_API)
    WgpuKernelManager(WGPUDevice device) : device_(device) {
#endif
        cache_ = lux_kernel_cache_create();
    }

    ~WgpuKernelManager() {
        lux_kernel_cache_clear(cache_);
        lux_kernel_cache_destroy(cache_);
    }

    // Get or compile kernel by name
    WgpuKernel* get_kernel(const char* name, uint32_t dtype = 0, uint32_t size_hint = 0) {
        LuxKernelVariant variant = { name, dtype, size_hint, 0 };

        // Check cache
        LuxKernel* cached = lux_kernel_cache_get(cache_, &variant);
        if (cached) {
            return reinterpret_cast<WgpuKernel*>(cached);
        }

        // Find in registry
        const LuxKernelRegistry* registry = lux_kernel_registry_get("webgpu");
        if (!registry) return nullptr;

        const LuxEmbeddedKernel* embedded = lux_kernel_registry_find(registry, name);
        if (!embedded) return nullptr;

        // Compile kernel
        WgpuKernel* kernel = nullptr;
        if (embedded->source && embedded->source_len > 0) {
            kernel = WgpuKernelCache::instance().compile(
                device_, embedded->source, embedded->source_len, embedded->entry_point
            );
        }

        if (kernel) {
            lux_kernel_cache_put(cache_, &variant, reinterpret_cast<LuxKernel*>(kernel));
        }

        return kernel;
    }

    // Preload all embedded kernels
    void preload_all() {
        const LuxKernelRegistry* registry = lux_kernel_registry_get("webgpu");
        if (!registry) return;

        for (size_t i = 0; i < registry->count; ++i) {
            get_kernel(registry->kernels[i].name);
        }
    }

private:
#if defined(USE_DAWN_API)
    wgpu::Device device_;
#elif defined(USE_WGPU_API)
    WGPUDevice device_;
#endif
    LuxKernelCache* cache_;
};

} // namespace lux::gpu::webgpu

#else // !LUX_GPU_BUILD_WEBGPU

// Stub implementations when WebGPU is not available
extern "C" {

LuxKernel* lux_wgpu_kernel_compile(void*, const char*, size_t, const char*) {
    return nullptr;
}

LuxKernel* lux_wgpu_kernel_load_file(void*, const char*, const char*) {
    return nullptr;
}

void lux_wgpu_kernel_destroy(LuxKernel*) {}

const char* lux_wgpu_kernel_entry_point(LuxKernel*) {
    return nullptr;
}

void* lux_wgpu_kernel_pipeline(LuxKernel*) {
    return nullptr;
}

void* lux_wgpu_kernel_bind_group_layout(LuxKernel*) {
    return nullptr;
}

void lux_wgpu_kernel_workgroup_size(LuxKernel*, uint32_t* x, uint32_t* y, uint32_t* z) {
    if (x) *x = 1;
    if (y) *y = 1;
    if (z) *z = 1;
}

} // extern "C"

#endif // LUX_GPU_BUILD_WEBGPU
