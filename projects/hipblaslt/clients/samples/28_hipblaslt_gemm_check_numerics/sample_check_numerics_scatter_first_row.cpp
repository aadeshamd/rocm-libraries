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
 * Scatter NaN test B1 -- NaN in FIRST ROW of D.
 *
 * 16x16 FP32 NN GEMM, K=4, alpha=1, beta=0.
 * A = ones except A[0,*] (row 0, all K columns) are set to NaN.
 * B = ones.
 *
 * Result: D[0,j] = NaN for all j in [0,N).  Rows 1..15 = 4.0f.
 * The NaN sits at the very beginning of D's column-major buffer (index 0).
 *
 * Expected behaviour under HIPBLASLT_CHECK_NUMERICS:
 *   mode=1 (info)  : logs one line, has_NaN=1
 *   mode=2 (warn)  : logs one line (NaN found)
 *   mode=4 (fail)  : returns HIPBLAS_STATUS_INVALID_VALUE
 *   mode=6         : logs + returns INVALID_VALUE
 *
 * IEEE 754 note: NaN * 1 = NaN and NaN + x = NaN, so poisoning any element
 * of A row r propagates NaN to all N elements of D row r regardless of B.
 *
 * Exit code:
 *   0  -- scanner triggered as expected (or env var not set and D verified)
 *   1  -- unexpected result
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <vector>

#include "helper.h"

static constexpr int64_t M = 16;
static constexpr int64_t N = 16;
static constexpr int64_t K = 4;

int main()
{
    if(const char* cn = std::getenv("HIPBLASLT_CHECK_NUMERICS"))
        std::printf("HIPBLASLT_CHECK_NUMERICS=\"%s\"\n", cn);
    else
        std::printf("HIPBLASLT_CHECK_NUMERICS not set -- scanner disabled.\n");

    std::printf("[scatter-first-row] 16x16 K=4: A row 0 = NaN, B = ones -> D row 0 all NaN.\n");

    hipblasLtHandle_t handle{};
    hipStream_t       stream{};
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // A is column-major MxK. Element [row, col] -> index row + M*col.
    // Poison row 0: A[0, k] = NaN for k in [0,K).
    std::vector<float> h_A(M * K, 1.0f);
    for(int64_t k = 0; k < K; ++k)
        h_A[0 + M * k] = std::nanf(""); // A[0, k]

    std::vector<float> h_B(K * N, 1.0f);

    float *d_A{}, *d_B{}, *d_C{}, *d_D{};
    CHECK_HIP_ERROR(hipMalloc(&d_A, sizeof(float) * M * K));
    CHECK_HIP_ERROR(hipMalloc(&d_B, sizeof(float) * K * N));
    CHECK_HIP_ERROR(hipMalloc(&d_C, sizeof(float) * M * N));
    CHECK_HIP_ERROR(hipMalloc(&d_D, sizeof(float) * M * N));

    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), sizeof(float) * M * K, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, h_B.data(), sizeof(float) * K * N, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(d_C, 0, sizeof(float) * M * N));
    CHECK_HIP_ERROR(hipMemset(d_D, 0, sizeof(float) * M * N));

    hipblasLtMatrixLayout_t matA{}, matB{}, matC{}, matD{};
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M));

    hipblasLtMatmulDesc_t matmul{};
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    hipblasOperation_t opN = HIPBLAS_OP_N;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(int32_t)));

    hipblasLtMatmulPreference_t pref{};
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    uint64_t max_ws = 32u * 1024u * 1024u;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws, sizeof(max_ws)));

    hipblasLtMatmulHeuristicResult_t heur[1]{};
    int                              ret_count = 0;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(
        handle, matmul, matA, matB, matC, matD, pref, 1, heur, &ret_count));
    if(ret_count == 0)
    {
        std::fprintf(stderr, "[scatter-first-row] SKIP: no heuristic returned\n");
        return 0;
    }

    void* d_ws = nullptr;
    if(heur[0].workspaceSize)
        CHECK_HIP_ERROR(hipMalloc(&d_ws, heur[0].workspaceSize));

    const float           alpha = 1.0f, beta = 0.0f;
    const hipblasStatus_t st   = hipblasLtMatmul(handle, matmul, &alpha,
                                               d_A, matA, d_B, matB,
                                               &beta, d_C, matC, d_D, matD,
                                               &heur[0].algo, d_ws,
                                               heur[0].workspaceSize, stream);
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    std::vector<float> h_D(M * N, 0.f);
    CHECK_HIP_ERROR(hipMemcpy(h_D.data(), d_D, sizeof(float) * M * N, hipMemcpyDeviceToHost));

    // Verify: row 0 of D (col-major: indices 0, M, 2M, ...) must all be NaN.
    int row0_nan = 0, other_nan = 0;
    for(int64_t j = 0; j < N; ++j)
    {
        if(std::isnan(h_D[0 + M * j]))
            ++row0_nan;
    }
    for(int64_t i = 1; i < M; ++i)
        for(int64_t j = 0; j < N; ++j)
            if(std::isnan(h_D[i + M * j]))
                ++other_nan;

    std::printf("[scatter-first-row] matmul status=%d  row0 NaN=%d/%lld  other NaN=%d\n",
                static_cast<int>(st), row0_nan, (long long)N, other_nan);

    if(d_ws)
        CHECK_HIP_ERROR(hipFree(d_ws));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIP_ERROR(hipFree(d_D));
    CHECK_HIP_ERROR(hipFree(d_C));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));

    // PASS conditions:
    //   - D row 0 must be all NaN (K reduction of NaN is NaN).
    //   - If HIPBLASLT_CHECK_NUMERICS=4|6, scanner must have returned INVALID_VALUE.
    //   - If =1|2 or unset, status must be SUCCESS (warn/info don't change return code).
    if(row0_nan != static_cast<int>(N))
    {
        std::fprintf(stderr,
                     "[scatter-first-row] FAIL: expected %lld NaN in row 0, got %d\n",
                     (long long)N, row0_nan);
        return 1;
    }
    const char* cn = std::getenv("HIPBLASLT_CHECK_NUMERICS");
    const int   mode = cn ? std::atoi(cn) & 0x7 : 0;
    const bool  expect_fail = (mode & 4) != 0;
    if(expect_fail && st != HIPBLAS_STATUS_INVALID_VALUE)
    {
        std::fprintf(stderr,
                     "[scatter-first-row] FAIL: expected INVALID_VALUE (fail bit set), got %d\n",
                     static_cast<int>(st));
        return 1;
    }
    if(!expect_fail && st != HIPBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "[scatter-first-row] FAIL: unexpected status %d\n",
                     static_cast<int>(st));
        return 1;
    }
    std::printf("[scatter-first-row] PASS\n");
    return 0;
}
