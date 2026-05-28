// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Minimal test framework - header only, no dependencies

#ifndef LUX_TEST_FRAMEWORK_H
#define LUX_TEST_FRAMEWORK_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <functional>

namespace lux::test {

// Test result tracking
struct TestStats {
    int passed = 0;
    int failed = 0;
    int skipped = 0;
};

inline TestStats& global_stats() {
    static TestStats stats;
    return stats;
}

// Test case registration
struct TestCase {
    const char* name;
    std::function<bool()> fn;
};

inline std::vector<TestCase>& test_cases() {
    static std::vector<TestCase> cases;
    return cases;
}

inline void register_test(const char* name, std::function<bool()> fn) {
    test_cases().push_back({name, fn});
}

// Run all tests
inline int run_all_tests() {
    auto& stats = global_stats();
    auto& cases = test_cases();

    printf("\n=== Running %zu tests ===\n\n", cases.size());

    for (const auto& tc : cases) {
        printf("  %s ... ", tc.name);
        fflush(stdout);

        try {
            if (tc.fn()) {
                printf("PASS\n");
                stats.passed++;
            } else {
                printf("FAIL\n");
                stats.failed++;
            }
        } catch (const std::exception& e) {
            printf("EXCEPTION: %s\n", e.what());
            stats.failed++;
        } catch (...) {
            printf("UNKNOWN EXCEPTION\n");
            stats.failed++;
        }
    }

    printf("\n=== Results: %d passed, %d failed, %d skipped ===\n\n",
           stats.passed, stats.failed, stats.skipped);

    return stats.failed > 0 ? 1 : 0;
}

// Assertion macros
#define TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "  ASSERTION FAILED: %s\n    at %s:%d\n", \
                    #cond, __FILE__, __LINE__); \
            return false; \
        } \
    } while (0)

#define TEST_ASSERT_EQ(a, b) \
    do { \
        auto _a = (a); \
        auto _b = (b); \
        if (_a != _b) { \
            fprintf(stderr, "  ASSERTION FAILED: %s == %s\n    at %s:%d\n", \
                    #a, #b, __FILE__, __LINE__); \
            return false; \
        } \
    } while (0)

#define TEST_ASSERT_NEAR(a, b, tol) \
    do { \
        auto _a = (a); \
        auto _b = (b); \
        if (std::abs(_a - _b) > (tol)) { \
            fprintf(stderr, "  ASSERTION FAILED: |%s - %s| <= %s\n" \
                    "    values: %g vs %g (diff=%g)\n    at %s:%d\n", \
                    #a, #b, #tol, (double)_a, (double)_b, \
                    (double)std::abs(_a - _b), __FILE__, __LINE__); \
            return false; \
        } \
    } while (0)

#define TEST_SKIP(msg) \
    do { \
        printf("SKIP (%s)\n", msg); \
        lux::test::global_stats().skipped++; \
        return true; \
    } while (0)

// Test registration macro
#define TEST_CASE(name) \
    static bool test_##name(); \
    namespace { \
        struct TestRegistrar_##name { \
            TestRegistrar_##name() { \
                lux::test::register_test(#name, test_##name); \
            } \
        } test_registrar_##name; \
    } \
    static bool test_##name()

// Main function macro
#define TEST_MAIN() \
    int main(int argc, char** argv) { \
        (void)argc; (void)argv; \
        return lux::test::run_all_tests(); \
    }

} // namespace lux::test

#endif // LUX_TEST_FRAMEWORK_H
