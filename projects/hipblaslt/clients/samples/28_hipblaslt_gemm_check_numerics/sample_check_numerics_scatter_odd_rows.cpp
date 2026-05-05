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
 * Scatter NaN test B4 -- NaN in ODD ROWS of D (interleaved / sparse scatter).
 *
 * 16x16 FP32 NN GEMM, K=4, alpha=1, beta=0.
 * A = ones except rows 1,3,5,7,9,11,13,15 (all odd rows, all K columns) are NaN.
 * B = ones.
 *
 * Result:
 *   D[i, j] = NaN  for odd i  (8 rows x 16 cols = 128 NaNs)
 *   D[i, j] = 4.0  for even i (8 rows x 16 cols = 128 fours)
 *
 * This exercises the scanner with NaN/finite values interleaved across the
 * column-major buffer, rather than grouped at one end. The atomicOr flag
 * must be raised regardless of thread execution order.
 *
 * Exit code: 0 = PASS, 1 = FAIL.
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

    std::printf("[scatter-odd-rows] 16x16 K=4: A odd rows = NaN, B = ones "
                "-> D odd rows all NaN, even rows = 4.0.\n");

    hipblasLtHandle_t handle{};
    hipStream_t       stream{};
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    std::vector<float> h_A(M * K, 1.0f);
    // Poison all odd rows (1, 3, 5, ..., 15).
    for(int64_t row = 1; row < M; row += 2)
        for(int64_t k = 0; k < K; ++k)
            h_A[row + M * k] = std::nanf("");

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
        std::fprintf(stderr, "[scatter-odd-rows] SKIP: no heuristic returned\n");
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

    int odd_nan = 0, even_nan = 0;
    for(int64_t i = 0; i < M; ++i)
        for(int64_t j = 0; j < N; ++j)
        {
            const bool is_nan = std::isnan(h_D[i + M * j]);
            if(i % 2 == 1)
                odd_nan  += is_nan ? 1 : 0;
            else
                even_nan += is_nan ? 1 : 0;
        }

    std::printf("[scatter-odd-rows] matmul status=%d  odd-row NaN=%d/%lld  even-row NaN=%d\n",
                static_cast<int>(st), odd_nan, (long long)(M / 2 * N), even_nan);

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

    const int64_t expected_odd_nan = (M / 2) * N; // 8 * 16 = 128
    if(odd_nan != static_cast<int>(expected_odd_nan))
    {
        std::fprintf(stderr,
                     "[scatter-odd-rows] FAIL: expected %lld NaN in odd rows, got %d\n",
                     (long long)expected_odd_nan, odd_nan);
        return 1;
    }
    if(even_nan != 0)
    {
        std::fprintf(stderr,
                     "[scatter-odd-rows] FAIL: unexpected %d NaN in even rows\n", even_nan);
        return 1;
    }
    const char* cn = std::getenv("HIPBLASLT_CHECK_NUMERICS");
    const int   mode = cn ? std::atoi(cn) & 0x7 : 0;
    const bool  expect_fail = (mode & 4) != 0;
    if(expect_fail && st != HIPBLAS_STATUS_INVALID_VALUE)
    {
        std::fprintf(stderr,
                     "[scatter-odd-rows] FAIL: expected INVALID_VALUE, got %d\n",
                     static_cast<int>(st));
        return 1;
    }
    if(!expect_fail && st != HIPBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "[scatter-odd-rows] FAIL: unexpected status %d\n",
                     static_cast<int>(st));
        return 1;
    }
    std::printf("[scatter-odd-rows] PASS\n");
    return 0;
}
