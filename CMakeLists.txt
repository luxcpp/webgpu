# Copyright (c) 2024-2026 Lux Industries Inc.
# SPDX-License-Identifier: BSD-3-Clause-Eco
#
# Lux WebGPU Backend - Cross-platform GPU acceleration via Dawn/gpu.cpp
#
# Dependencies:
#   - Dawn (Google's WebGPU implementation) OR wgpu-native
#   - gpu.cpp (lightweight WebGPU compute library)
#
# Builds: libluxgpu_backend_webgpu.(so|dylib|dll)

cmake_minimum_required(VERSION 3.20)

project(lux-webgpu
    VERSION 0.1.0
    DESCRIPTION "WebGPU backend for GPU acceleration"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# Options
# =============================================================================

option(LUX_WEBGPU_BUILD_TESTS "Build WebGPU backend tests" ON)
option(LUX_WEBGPU_EMBED_KERNELS "Embed WGSL kernel sources" ON)
option(LUX_WEBGPU_USE_DAWN "Use Dawn (vs wgpu-native)" ON)

# =============================================================================
# Dependencies
# =============================================================================

# gpu.cpp - Lightweight WebGPU compute library
# https://github.com/AnswerDotAI/gpu.cpp
set(GPUCPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/gpu.cpp)

if(EXISTS ${GPUCPP_DIR})
    message(STATUS "Using gpu.cpp from: ${GPUCPP_DIR}")
    add_library(gpucpp INTERFACE)
    target_include_directories(gpucpp INTERFACE ${GPUCPP_DIR})
else()
    message(STATUS "gpu.cpp not found at ${GPUCPP_DIR}, using system headers")
    add_library(gpucpp INTERFACE)
endif()

# Dawn or wgpu-native
if(LUX_WEBGPU_USE_DAWN)
    find_package(Dawn QUIET)
    if(Dawn_FOUND)
        message(STATUS "Using Dawn WebGPU implementation")
        set(WEBGPU_LIB dawn::webgpu_dawn)
        set(WEBGPU_DEFINES USE_DAWN_API)
    endif()
endif()

if(NOT Dawn_FOUND)
    # Try wgpu-native
    find_package(wgpu QUIET)
    if(wgpu_FOUND)
        message(STATUS "Using wgpu-native WebGPU implementation")
        set(WEBGPU_LIB wgpu::wgpu)
        set(WEBGPU_DEFINES USE_WGPU_API)
    else()
        # Manual detection (Homebrew, system paths)
        find_library(WGPU_LIBRARY NAMES wgpu_native wgpu
            PATHS /opt/homebrew/lib /usr/local/lib /usr/lib
        )
        find_path(WGPU_INCLUDE_DIR NAMES wgpu.h webgpu.h
            PATHS /opt/homebrew/include /usr/local/include /usr/include
        )
        if(WGPU_LIBRARY AND WGPU_INCLUDE_DIR)
            message(STATUS "Found wgpu-native: ${WGPU_LIBRARY}")
            add_library(wgpu_native INTERFACE)
            target_include_directories(wgpu_native INTERFACE ${WGPU_INCLUDE_DIR})
            target_link_libraries(wgpu_native INTERFACE ${WGPU_LIBRARY})
            set(WEBGPU_LIB wgpu_native)
            set(WEBGPU_DEFINES USE_WGPU_API)
        else()
            message(FATAL_ERROR "WebGPU implementation required (Dawn or wgpu-native)")
        endif()
    endif()
endif()

# =============================================================================
# Kernel Sources
# =============================================================================

set(LUX_WGSL_KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels/wgsl)

file(GLOB_RECURSE WGSL_KERNELS ${LUX_WGSL_KERNEL_DIR}/*.wgsl)
list(LENGTH WGSL_KERNELS WGSL_KERNEL_COUNT)
message(STATUS "Found ${WGSL_KERNEL_COUNT} WGSL kernels")

# =============================================================================
# Kernel Embedding
# =============================================================================

if(LUX_WEBGPU_EMBED_KERNELS)
    set(EMBEDDED_KERNELS_HEADER ${CMAKE_CURRENT_BINARY_DIR}/wgsl_kernels_embedded.h)

    # Generate embedded header at configure time
    file(WRITE ${EMBEDDED_KERNELS_HEADER} "// Auto-generated WGSL kernel embeddings\n#pragma once\n\n")
    file(APPEND ${EMBEDDED_KERNELS_HEADER} "#include <string_view>\n#include <unordered_map>\n\n")
    file(APPEND ${EMBEDDED_KERNELS_HEADER} "namespace lux::gpu::wgsl {\n\n")

    foreach(kernel ${WGSL_KERNELS})
        get_filename_component(kernel_name ${kernel} NAME_WE)
        file(READ ${kernel} kernel_content)
        string(REPLACE "\\" "\\\\" kernel_content "${kernel_content}")
        string(REPLACE "\"" "\\\"" kernel_content "${kernel_content}")
        string(REPLACE "\n" "\\n\"\n    \"" kernel_content "${kernel_content}")
        file(APPEND ${EMBEDDED_KERNELS_HEADER} "constexpr const char* k_${kernel_name} = \n    \"${kernel_content}\";\n\n")
    endforeach()

    file(APPEND ${EMBEDDED_KERNELS_HEADER} "} // namespace lux::gpu::wgsl\n")
endif()

# =============================================================================
# Backend Plugin Library
# =============================================================================

add_library(luxgpu_backend_webgpu MODULE
    src/webgpu_plugin.cpp
    src/webgpu_kernel_loader.cpp
)

target_include_directories(luxgpu_backend_webgpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_compile_definitions(luxgpu_backend_webgpu PRIVATE
    LUX_GPU_BUILD_WEBGPU
    ${WEBGPU_DEFINES}
)

target_link_libraries(luxgpu_backend_webgpu PRIVATE
    ${WEBGPU_LIB}
    gpucpp
)

set_target_properties(luxgpu_backend_webgpu PROPERTIES
    OUTPUT_NAME "luxgpu_backend_webgpu"
    PREFIX "lib"
)

# Platform-specific suffix
if(APPLE)
    set_target_properties(luxgpu_backend_webgpu PROPERTIES SUFFIX ".dylib")
elseif(WIN32)
    set_target_properties(luxgpu_backend_webgpu PROPERTIES SUFFIX ".dll")
else()
    set_target_properties(luxgpu_backend_webgpu PROPERTIES SUFFIX ".so")
endif()

# =============================================================================
# Installation
# =============================================================================

include(GNUInstallDirs)

install(TARGETS luxgpu_backend_webgpu
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/lux-gpu
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install WGSL kernels (for runtime compilation if not embedded)
if(NOT LUX_WEBGPU_EMBED_KERNELS)
    install(DIRECTORY ${LUX_WGSL_KERNEL_DIR}/
        DESTINATION ${CMAKE_INSTALL_DATADIR}/lux-gpu/wgsl
        FILES_MATCHING PATTERN "*.wgsl"
    )
endif()

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "=== Lux WebGPU Backend Configuration ===")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  WGSL Kernels: ${WGSL_KERNEL_COUNT}")
message(STATUS "  Embed Kernels: ${LUX_WEBGPU_EMBED_KERNELS}")
message(STATUS "  WebGPU Implementation: ${WEBGPU_DEFINES}")
message(STATUS "")
