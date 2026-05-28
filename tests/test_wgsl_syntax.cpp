// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// WGSL Syntax Validation Tests
// Validates WGSL kernel files for basic syntax correctness without GPU

#include "test_framework.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <regex>
#include <set>

namespace fs = std::filesystem;

// WGSL syntax checker - validates basic structure without full parsing
class WgslValidator {
public:
    struct ValidationResult {
        bool valid;
        std::string error;
        int line;
    };

    ValidationResult validate(const std::string& source, const std::string& filename) {
        std::istringstream stream(source);
        std::string line;
        int line_num = 0;
        int brace_depth = 0;
        int paren_depth = 0;
        bool in_comment = false;

        while (std::getline(stream, line)) {
            line_num++;

            // Skip empty lines
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
                continue;
            }

            // Track multi-line comments
            size_t pos = 0;
            while (pos < line.size()) {
                if (in_comment) {
                    size_t end = line.find("*/", pos);
                    if (end != std::string::npos) {
                        in_comment = false;
                        pos = end + 2;
                    } else {
                        break;
                    }
                } else {
                    // Skip single-line comments
                    size_t single_comment = line.find("//", pos);
                    if (single_comment != std::string::npos) {
                        line = line.substr(0, single_comment);
                        break;
                    }

                    // Check for multi-line comment start
                    size_t multi_comment = line.find("/*", pos);
                    if (multi_comment != std::string::npos) {
                        size_t end = line.find("*/", multi_comment + 2);
                        if (end != std::string::npos) {
                            line = line.substr(0, multi_comment) + line.substr(end + 2);
                            pos = multi_comment;
                            continue;
                        } else {
                            in_comment = true;
                            line = line.substr(0, multi_comment);
                            break;
                        }
                    }
                    break;
                }
            }

            // Count braces and parens
            for (char c : line) {
                switch (c) {
                    case '{': brace_depth++; break;
                    case '}': brace_depth--; break;
                    case '(': paren_depth++; break;
                    case ')': paren_depth--; break;
                }

                if (brace_depth < 0) {
                    return {false, "Unmatched closing brace", line_num};
                }
                if (paren_depth < 0) {
                    return {false, "Unmatched closing parenthesis", line_num};
                }
            }
        }

        if (brace_depth != 0) {
            return {false, "Unmatched braces (depth=" + std::to_string(brace_depth) + ")", line_num};
        }
        if (paren_depth != 0) {
            return {false, "Unmatched parentheses (depth=" + std::to_string(paren_depth) + ")", line_num};
        }
        if (in_comment) {
            return {false, "Unterminated multi-line comment", line_num};
        }

        return {true, "", 0};
    }

    // Check for required WGSL constructs
    bool has_compute_shader(const std::string& source) {
        return source.find("@compute") != std::string::npos;
    }

    bool has_workgroup_size(const std::string& source) {
        return source.find("@workgroup_size") != std::string::npos;
    }

    bool has_binding(const std::string& source) {
        return source.find("@binding") != std::string::npos;
    }

    // Extract all entry point names
    std::set<std::string> extract_entry_points(const std::string& source) {
        std::set<std::string> entries;
        std::regex fn_regex(R"(@compute\s+@workgroup_size\([^)]+\)\s*fn\s+(\w+))");
        std::sregex_iterator iter(source.begin(), source.end(), fn_regex);
        std::sregex_iterator end;

        while (iter != end) {
            entries.insert((*iter)[1].str());
            ++iter;
        }
        return entries;
    }
};

// Read file contents
static std::string read_file(const fs::path& path) {
    std::ifstream file(path);
    if (!file) {
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Global validator
static WgslValidator g_validator;

// =============================================================================
// Test Cases
// =============================================================================

TEST_CASE(wgsl_kernel_dir_exists) {
    fs::path kernel_dir(WGSL_KERNEL_DIR);
    TEST_ASSERT(fs::exists(kernel_dir));
    TEST_ASSERT(fs::is_directory(kernel_dir));
    return true;
}

TEST_CASE(wgsl_gpu_kernels_valid) {
    fs::path kernel_dir = fs::path(WGSL_KERNEL_DIR) / "gpu";
    if (!fs::exists(kernel_dir)) {
        TEST_SKIP("gpu kernels directory not found");
    }

    int checked = 0;
    for (const auto& entry : fs::recursive_directory_iterator(kernel_dir)) {
        if (entry.path().extension() == ".wgsl") {
            std::string source = read_file(entry.path());
            TEST_ASSERT(!source.empty());

            auto result = g_validator.validate(source, entry.path().string());
            if (!result.valid) {
                fprintf(stderr, "  Validation failed: %s\n    %s at line %d\n",
                        entry.path().filename().c_str(), result.error.c_str(), result.line);
                return false;
            }

            // GPU compute kernels should have compute shaders
            TEST_ASSERT(g_validator.has_compute_shader(source));
            TEST_ASSERT(g_validator.has_workgroup_size(source));

            checked++;
        }
    }

    printf("(checked %d files) ", checked);
    TEST_ASSERT(checked > 0);
    return true;
}

TEST_CASE(wgsl_crypto_kernels_valid) {
    fs::path kernel_dir = fs::path(WGSL_KERNEL_DIR) / "crypto";
    if (!fs::exists(kernel_dir)) {
        TEST_SKIP("crypto kernels directory not found");
    }

    int checked = 0;
    for (const auto& entry : fs::recursive_directory_iterator(kernel_dir)) {
        if (entry.path().extension() == ".wgsl") {
            std::string source = read_file(entry.path());
            TEST_ASSERT(!source.empty());

            auto result = g_validator.validate(source, entry.path().string());
            if (!result.valid) {
                fprintf(stderr, "  Validation failed: %s\n    %s at line %d\n",
                        entry.path().filename().c_str(), result.error.c_str(), result.line);
                return false;
            }

            checked++;
        }
    }

    printf("(checked %d files) ", checked);
    TEST_ASSERT(checked > 0);
    return true;
}

TEST_CASE(wgsl_zk_kernels_valid) {
    fs::path kernel_dir = fs::path(WGSL_KERNEL_DIR) / "zk";
    if (!fs::exists(kernel_dir)) {
        TEST_SKIP("zk kernels directory not found");
    }

    int checked = 0;
    for (const auto& entry : fs::recursive_directory_iterator(kernel_dir)) {
        if (entry.path().extension() == ".wgsl") {
            std::string source = read_file(entry.path());
            TEST_ASSERT(!source.empty());

            auto result = g_validator.validate(source, entry.path().string());
            if (!result.valid) {
                fprintf(stderr, "  Validation failed: %s\n    %s at line %d\n",
                        entry.path().filename().c_str(), result.error.c_str(), result.line);
                return false;
            }

            checked++;
        }
    }

    printf("(checked %d files) ", checked);
    TEST_ASSERT(checked > 0);
    return true;
}

TEST_CASE(wgsl_fhe_kernels_valid) {
    fs::path kernel_dir = fs::path(WGSL_KERNEL_DIR) / "fhe";
    if (!fs::exists(kernel_dir)) {
        TEST_SKIP("fhe kernels directory not found");
    }

    int checked = 0;
    for (const auto& entry : fs::recursive_directory_iterator(kernel_dir)) {
        if (entry.path().extension() == ".wgsl") {
            std::string source = read_file(entry.path());
            TEST_ASSERT(!source.empty());

            auto result = g_validator.validate(source, entry.path().string());
            if (!result.valid) {
                fprintf(stderr, "  Validation failed: %s\n    %s at line %d\n",
                        entry.path().filename().c_str(), result.error.c_str(), result.line);
                return false;
            }

            checked++;
        }
    }

    printf("(checked %d files) ", checked);
    TEST_ASSERT(checked > 0);
    return true;
}

TEST_CASE(wgsl_steel_kernels_valid) {
    fs::path kernel_dir = fs::path(WGSL_KERNEL_DIR) / "steel";
    if (!fs::exists(kernel_dir)) {
        TEST_SKIP("steel kernels directory not found");
    }

    int checked = 0;
    for (const auto& entry : fs::recursive_directory_iterator(kernel_dir)) {
        if (entry.path().extension() == ".wgsl") {
            std::string source = read_file(entry.path());
            TEST_ASSERT(!source.empty());

            auto result = g_validator.validate(source, entry.path().string());
            if (!result.valid) {
                fprintf(stderr, "  Validation failed: %s\n    %s at line %d\n",
                        entry.path().filename().c_str(), result.error.c_str(), result.line);
                return false;
            }

            checked++;
        }
    }

    printf("(checked %d files) ", checked);
    TEST_ASSERT(checked > 0);
    return true;
}

TEST_CASE(wgsl_lattice_kernels_valid) {
    fs::path kernel_dir = fs::path(WGSL_KERNEL_DIR) / "lattice";
    if (!fs::exists(kernel_dir)) {
        TEST_SKIP("lattice kernels directory not found");
    }

    int checked = 0;
    for (const auto& entry : fs::recursive_directory_iterator(kernel_dir)) {
        if (entry.path().extension() == ".wgsl") {
            std::string source = read_file(entry.path());
            TEST_ASSERT(!source.empty());

            auto result = g_validator.validate(source, entry.path().string());
            if (!result.valid) {
                fprintf(stderr, "  Validation failed: %s\n    %s at line %d\n",
                        entry.path().filename().c_str(), result.error.c_str(), result.line);
                return false;
            }

            checked++;
        }
    }

    printf("(checked %d files) ", checked);
    TEST_ASSERT(checked > 0);
    return true;
}

TEST_CASE(wgsl_binary_ops_entry_points) {
    fs::path kernel_path = fs::path(WGSL_KERNEL_DIR) / "gpu" / "binary.wgsl";
    if (!fs::exists(kernel_path)) {
        TEST_SKIP("binary.wgsl not found");
    }

    std::string source = read_file(kernel_path);
    auto entries = g_validator.extract_entry_points(source);

    // Verify expected entry points exist
    TEST_ASSERT(entries.count("add") > 0);
    TEST_ASSERT(entries.count("sub") > 0);
    TEST_ASSERT(entries.count("mul") > 0);
    TEST_ASSERT(entries.count("div") > 0);

    printf("(found %zu entry points) ", entries.size());
    return true;
}

TEST_CASE(wgsl_reduce_ops_entry_points) {
    fs::path kernel_path = fs::path(WGSL_KERNEL_DIR) / "gpu" / "reduce.wgsl";
    if (!fs::exists(kernel_path)) {
        TEST_SKIP("reduce.wgsl not found");
    }

    std::string source = read_file(kernel_path);
    auto entries = g_validator.extract_entry_points(source);

    // Verify expected entry points exist
    TEST_ASSERT(entries.count("reduce_barrett_dilithium") > 0);
    TEST_ASSERT(entries.count("reduce_full_dilithium") > 0);
    TEST_ASSERT(entries.count("reduce_barrett_kyber") > 0);

    printf("(found %zu entry points) ", entries.size());
    return true;
}

TEST_CASE(wgsl_blake3_entry_points) {
    fs::path kernel_path = fs::path(WGSL_KERNEL_DIR) / "crypto" / "blake3.wgsl";
    if (!fs::exists(kernel_path)) {
        TEST_SKIP("blake3.wgsl not found");
    }

    std::string source = read_file(kernel_path);
    auto entries = g_validator.extract_entry_points(source);

    // Verify expected entry points exist
    TEST_ASSERT(entries.count("hash_chunks") > 0);
    TEST_ASSERT(entries.count("merge_parents") > 0);
    TEST_ASSERT(entries.count("hash_small") > 0);

    printf("(found %zu entry points) ", entries.size());
    return true;
}

TEST_CASE(wgsl_all_kernels_have_bindings) {
    fs::path kernel_dir(WGSL_KERNEL_DIR);
    int checked = 0;
    int with_bindings = 0;

    for (const auto& entry : fs::recursive_directory_iterator(kernel_dir)) {
        if (entry.path().extension() == ".wgsl") {
            std::string source = read_file(entry.path());
            checked++;

            if (g_validator.has_compute_shader(source)) {
                TEST_ASSERT(g_validator.has_binding(source));
                with_bindings++;
            }
        }
    }

    printf("(checked %d files, %d with bindings) ", checked, with_bindings);
    TEST_ASSERT(checked > 0);
    return true;
}

TEST_CASE(wgsl_no_syntax_errors_all) {
    fs::path kernel_dir(WGSL_KERNEL_DIR);
    int checked = 0;
    int errors = 0;

    for (const auto& entry : fs::recursive_directory_iterator(kernel_dir)) {
        if (entry.path().extension() == ".wgsl") {
            std::string source = read_file(entry.path());
            auto result = g_validator.validate(source, entry.path().string());

            if (!result.valid) {
                fprintf(stderr, "  %s: %s at line %d\n",
                        entry.path().filename().c_str(), result.error.c_str(), result.line);
                errors++;
            }
            checked++;
        }
    }

    printf("(checked %d files) ", checked);
    TEST_ASSERT(errors == 0);
    return true;
}

TEST_MAIN()
