// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#define SKIP_IF_WINDOWS()                               \
    do                                                  \
    {                                                   \
        GTEST_SKIP() << "Disable this test in Windows"; \
    } while(0)
#else
#define SKIP_IF_WINDOWS() \
    do                    \
    {                     \
    } while(0)
#endif

#if defined(ADDRESS_SANITIZER) || defined(THREAD_SANITIZER)
#define SKIP_IF_NO_DEVICES()                                              \
    do                                                                    \
    {                                                                     \
        GTEST_SKIP() << "Disable device tests when sanitizer is enabled"; \
    } while(0)

#else
#define SKIP_IF_NO_DEVICES()                                        \
    do                                                              \
    {                                                               \
        int device_count;                                           \
        auto result = hipGetDeviceCount(&device_count);             \
        if(result == hipErrorNoDevice || device_count == 0)         \
        {                                                           \
            GTEST_SKIP() << "No devices available. Skipping test."; \
        }                                                           \
    } while(0)

#endif

#ifdef ADDRESS_SANITIZER
#define SKIP_IF_ASAN()                                            \
    do                                                            \
    {                                                             \
        GTEST_SKIP() << "Disable this test when ASAN is Enabled"; \
    } while(0)
#else
#define SKIP_IF_ASAN() \
    do                 \
    {                  \
    } while(0)
#endif

#ifdef THREAD_SANITIZER
#define SKIP_IF_TSAN()                                            \
    do                                                            \
    {                                                             \
        GTEST_SKIP() << "Disable this test when TSAN is Enabled"; \
    } while(0)
#else
#define SKIP_IF_TSAN() \
    do                 \
    {                  \
    } while(0)
#endif

#define SKIP_IF_DESCRIPTOR_API()                                                   \
    do                                                                             \
    {                                                                              \
        const char* val = std::getenv("HIPDNN_USE_DESCRIPTOR_API");                \
        if(val != nullptr && std::strcmp(val, "1") == 0)                           \
        {                                                                          \
            GTEST_SKIP() << "Skipped: not yet supported with descriptor API path"; \
        }                                                                          \
    } while(0)
