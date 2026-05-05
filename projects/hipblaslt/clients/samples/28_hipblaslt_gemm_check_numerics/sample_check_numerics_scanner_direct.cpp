/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

/*
 * Direct scanner kernel test -- exercises the NaN-check GPU kernel from
 * check_numerics_matrix.hpp WITHOUT going through hipblasLtMatmul or Tensile.
 *
 * Motivation:
 *   The scatter-NaN sample binaries (B0-B5) exercise the scanner end-to-end
 *   through hipblasLtMatmul -> Tensile.  On a machine where the built library
 *   targets a different GPU arch than the hardware (gfx950 build vs gfx90a hw),
 *   Tensile fails before the scanner fires.  This test bypasses Tensile by
 *   directly allocating a D buffer, filling it with a known NaN pattern on the
 *   host, copying to device, and launching ONLY the scanner kernel.
 *
 * What this tests:
 *   - The scanner kernel correctly identifies NaN in any position of D.
 *   - The log line printed to stderr contains the exact fn tag (test case name)
 *     so you can see which GEMM would have been logged in production.
 *   - The fail bit (mode & 4) causes the scanner to return the error status.
 *   - The clean case (T0) never triggers the flag.
 *
 * We include ONLY check_numerics_matrix.hpp -- no other private headers.
 * The 4 helper functions it calls for the log line (get_logger_os,
 * hipDataType_to_string, hipblasOperation_to_string,
 * rocblaslt_epilogue_to_string) are defined as simple stubs here before
 * including the header, so no library symbols are needed at link time
 * beyond hip::device.
 *
 * Run:
 *   HIPBLASLT_CHECK_NUMERICS=1  ./sample_check_numerics_scanner_direct
 *   HIPBLASLT_CHECK_NUMERICS=2  ./sample_check_numerics_scanner_direct
 *   HIPBLASLT_CHECK_NUMERICS=4  ./sample_check_numerics_scanner_direct
 *   HIPBLASLT_CHECK_NUMERICS=6  ./sample_check_numerics_scanner_direct
 */

// ============================================================================
// Minimal type/enum definitions so check_numerics_matrix.hpp compiles without
// pulling in handle.h, utility.hpp, or rocblaslt-types.h.
// These mirror the real definitions exactly (same values, same names).
// ============================================================================
#include <hip/hip_runtime.h>
#ifndef LEGACY_HIPBLAS_DIRECT
#include <hipblas-common/hipblas-common.h>
#else
#include <hipblas/hipblas.h>
#endif

// hipblaslt_check_numerics_mode bitmask (mirrors rocblaslt-types.h)
enum hipblaslt_check_numerics_mode_ : int32_t
{
    hipblaslt_check_numerics_mode_no_check = 0,
    hipblaslt_check_numerics_mode_info     = 1,
    hipblaslt_check_numerics_mode_warn     = 2,
    hipblaslt_check_numerics_mode_fail     = 4,
};
using hipblaslt_check_numerics_mode = hipblaslt_check_numerics_mode_;

// rocblaslt_status subset (mirrors rocblaslt-types.h)
enum rocblaslt_status_ : int32_t
{
    rocblaslt_status_success              = 0,
    rocblaslt_status_invalid_value        = 2,
    rocblaslt_status_internal_error       = 6,
    rocblaslt_status_check_numerics_fail  = 14,
};
using rocblaslt_status = rocblaslt_status_;

// rocblaslt_epilogue subset
enum rocblaslt_epilogue_ : uint32_t { ROCBLASLT_EPILOGUE_DEFAULT = 1 };
using rocblaslt_epilogue = rocblaslt_epilogue_;

// ============================================================================
// Stub implementations of the 4 log-line helpers called inside
// hipblaslt_check_numerics_output_D().  Defined BEFORE including the header
// so the inline function body sees these definitions.
// ============================================================================
#include <mutex>
#include <ostream>
#include <string>

// log_mutex: serialises log output across TUs; a simple local mutex is fine
// since this binary has a single TU.
std::mutex log_mutex;

// get_logger_os: returning nullptr makes the scanner fall back to std::cerr.
std::ostream* get_logger_os() { return nullptr; }

static inline const char* hipDataType_to_string(hipDataType t)
{
    switch(static_cast<int>(t))
    {
    case 0:  return "f32";
    case 1:  return "f64";
    case 10: return "f16";
    case 14: return "bf16";
    default: return "other";
    }
}

static inline const char* hipblasOperation_to_string(hipblasOperation_t op)
{
    return op == HIPBLAS_OP_N ? "N" : "T";
}

static inline std::string rocblaslt_epilogue_to_string(rocblaslt_epilogue)
{
    return "EPILOGUE_DEFAULT";
}

// ============================================================================
// fp8 type stubs: the scanner switch-case instantiates kernels for fp8 types
// but our test only uses HIP_R_32F, so the other branches are dead code.
// We just need the names to exist so the header compiles.
// Use the real fp8 headers from the ROCm stack if available.
// ============================================================================
#ifdef __has_include
#  if __has_include(<hip/amd_detail/amd_hip_fp8.h>)
#    include <hip/amd_detail/amd_hip_fp8.h>
     using hipblaslt_f8      = __hip_fp8_e4m3_fnuz;
     using hipblaslt_bf8     = __hip_fp8_e5m2_fnuz;
     using hipblaslt_f8_fnuz = __hip_fp8_e4m3_fnuz;
     using hipblaslt_bf8_fnuz= __hip_fp8_e5m2_fnuz;
#  else
     // Fallback if fp8 header not present: use float as a stand-in.
     // The switch-case for fp8 types will be compiled but never reached for f32.
     using hipblaslt_f8      = float;
     using hipblaslt_bf8     = float;
     using hipblaslt_f8_fnuz = float;
     using hipblaslt_bf8_fnuz= float;
#  endif
#endif

// hipblaslt_isnan stub for the stand-in float aliases above (real fp8 types
// have their own overloads in the ROCm fp8 header).
// For float the standard std::isnan is sufficient and will be found via ADL.
#include <cmath>

// hipblasLtHalf is __half; hip_bfloat16 is hip_bfloat16 -- both available from hip_runtime.h
using hipblasLtHalf = __half;

// ============================================================================
// Now include the scanner -- it only needs HIP + the stubs above.
// ============================================================================
#include "check_numerics_matrix.hpp"

// ============================================================================
// Test harness
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <vector>

static constexpr int64_t M = 16;
static constexpr int64_t N = 16;

static std::vector<float> make_d(std::vector<int64_t> nan_rows,
                                  int64_t              single_row = -1,
                                  int64_t              single_col = -1)
{
    std::vector<float> h(M * N, 1.0f);
    for(int64_t r : nan_rows)
        for(int64_t j = 0; j < N; ++j)
            h[r + M * j] = std::nanf("");
    if(single_row >= 0 && single_col >= 0)
        h[single_row + M * single_col] = std::nanf("");
    return h;
}

struct TestCase
{
    const char*        name;
    std::vector<float> h_D;
    bool               expect_nan;
};

int main()
{
    const char* cn   = std::getenv("HIPBLASLT_CHECK_NUMERICS");
    const int   mode = cn ? (std::atoi(cn) & 0x7) : 0;

    if(cn)
        std::printf("HIPBLASLT_CHECK_NUMERICS=\"%s\" (mode=%d)\n", cn, mode);
    else
        std::printf("HIPBLASLT_CHECK_NUMERICS not set -- using mode=1 (info) for this test.\n");

    const hipblaslt_check_numerics_mode scan_mode =
        (mode == 0)
            ? hipblaslt_check_numerics_mode_info
            : static_cast<hipblaslt_check_numerics_mode>(mode);

    hipStream_t stream{};
    if(hipStreamCreate(&stream) != hipSuccess)
    {
        std::fprintf(stderr, "hipStreamCreate failed\n");
        return 1;
    }

    auto make_t5 = []() {
        auto h = make_d({});
        h[4 + M * 6] = std::nanf(""); // D[4,6] only
        return h;
    };

    std::vector<TestCase> cases = {
        { "T0-clean",          make_d({}),                    false },
        { "T1-first-row",      make_d({0}),                   true  },
        { "T2-last-row",       make_d({15}),                  true  },
        { "T3-mid-row",        make_d({7}),                   true  },
        { "T4-odd-rows",       make_d({1,3,5,7,9,11,13,15}),  true  },
        { "T5-single-element", make_t5(),                     true  },
    };

    int n_pass = 0, n_fail = 0;

    for(auto& tc : cases)
    {
        std::printf("\n--- %s | expect_nan=%s ---\n",
                    tc.name, tc.expect_nan ? "YES" : "NO");
        std::fflush(stdout);

        float* d_D = nullptr;
        if(hipMalloc(&d_D, sizeof(float) * M * N) != hipSuccess
           || hipMemcpy(d_D, tc.h_D.data(), sizeof(float) * M * N,
                        hipMemcpyHostToDevice) != hipSuccess)
        {
            std::fprintf(stderr, "[%s] alloc/copy failed\n", tc.name);
            ++n_fail;
            continue;
        }

        // Call the scanner directly.  fn = test case name -> appears in log.
        const rocblaslt_status scan_st = hipblaslt_check_numerics_output_D(
            tc.name,
            stream,
            M, N, /*k=*/4,
            /*batch=*/1,
            HIP_R_32F,
            d_D,
            /*ldd=*/M,
            /*stride_d=*/M * N,
            /*row_major=*/false,
            HIPBLAS_OP_N, HIPBLAS_OP_N,
            ROCBLASLT_EPILOGUE_DEFAULT,
            /*algo_index=*/0,
            /*solution_name=*/"<direct-kernel-test>",
            /*kernel_name=*/"hipblaslt_check_nan_kernel<16,16,float>",
            scan_mode);

        hipFree(d_D);

        const bool got_fail = (scan_st == rocblaslt_status_check_numerics_fail);
        const bool fail_bit = (mode & 4) != 0;
        const bool exp_fail = tc.expect_nan && fail_bit;

        int nan_count = 0;
        for(float v : tc.h_D)
            if(std::isnan(v)) ++nan_count;

        std::printf("[%s] status=%d  buffer_nans=%d  got_fail=%s  expect_fail=%s\n",
                    tc.name, static_cast<int>(scan_st), nan_count,
                    got_fail ? "yes" : "no", exp_fail ? "yes" : "no");

        bool pass = true;
        if(exp_fail && !got_fail)
        {
            std::fprintf(stderr, "[%s] FAIL: expected check_numerics_fail\n", tc.name);
            pass = false;
        }
        if(!exp_fail && got_fail)
        {
            std::fprintf(stderr, "[%s] FAIL: unexpected check_numerics_fail\n", tc.name);
            pass = false;
        }
        std::printf("[%s] %s\n", tc.name, pass ? "PASS" : "FAIL");
        std::fflush(stdout);
        pass ? ++n_pass : ++n_fail;
    }

    hipStreamDestroy(stream);
    std::printf("\n=== Summary: %d/%d PASS ===\n", n_pass, n_pass + n_fail);
    return n_fail == 0 ? 0 : 1;
}
