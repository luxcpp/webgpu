# Copyright (c) 2024-2026 Lux Industries Inc.
# SPDX-License-Identifier: BSD-3-Clause-Eco
#
# WebGPU Backend Tests

cmake_minimum_required(VERSION 3.20)

# =============================================================================
# Test Dependencies
# =============================================================================

# We use a minimal header-only test framework to avoid external deps
# Tests can run standalone or with CTest

enable_testing()

# =============================================================================
# WGSL Validation Tests (no GPU required)
# =============================================================================

# Test that parses WGSL files for syntax correctness
add_executable(test_wgsl_syntax
    test_wgsl_syntax.cpp
)

target_include_directories(test_wgsl_syntax PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(test_wgsl_syntax PRIVATE
    WGSL_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../kernels/wgsl"
)

add_test(NAME wgsl_syntax COMMAND test_wgsl_syntax)

# =============================================================================
# Kernel Tests (require WebGPU runtime)
# =============================================================================

if(LUX_WEBGPU_BUILD_TESTS AND WEBGPU_LIB)
    # Binary operations tests
    add_executable(test_binary_ops
        test_binary_ops.cpp
    )

    target_include_directories(test_binary_ops PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_BINARY_DIR}/../
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(test_binary_ops PRIVATE
        ${WEBGPU_LIB}
        gpucpp
    )

    target_compile_definitions(test_binary_ops PRIVATE
        ${WEBGPU_DEFINES}
        WGSL_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../kernels/wgsl"
    )

    add_test(NAME binary_ops COMMAND test_binary_ops)

    # Crypto kernel tests
    add_executable(test_crypto_kernels
        test_crypto_kernels.cpp
    )

    target_include_directories(test_crypto_kernels PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_BINARY_DIR}/../
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(test_crypto_kernels PRIVATE
        ${WEBGPU_LIB}
        gpucpp
    )

    target_compile_definitions(test_crypto_kernels PRIVATE
        ${WEBGPU_DEFINES}
        WGSL_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../kernels/wgsl"
    )

    add_test(NAME crypto_kernels COMMAND test_crypto_kernels)

    # Reduction kernel tests (ML-DSA/ML-KEM)
    add_executable(test_reduction_kernels
        test_reduction_kernels.cpp
    )

    target_include_directories(test_reduction_kernels PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_BINARY_DIR}/../
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(test_reduction_kernels PRIVATE
        ${WEBGPU_LIB}
        gpucpp
    )

    target_compile_definitions(test_reduction_kernels PRIVATE
        ${WEBGPU_DEFINES}
        WGSL_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../kernels/wgsl"
    )

    add_test(NAME reduction_kernels COMMAND test_reduction_kernels)
endif()

# =============================================================================
# Summary
# =============================================================================

message(STATUS "WebGPU Tests configured")
message(STATUS "  WGSL Syntax Test: ON")
if(LUX_WEBGPU_BUILD_TESTS AND WEBGPU_LIB)
    message(STATUS "  GPU Runtime Tests: ON")
else()
    message(STATUS "  GPU Runtime Tests: OFF (requires WebGPU)")
endif()
