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
 * Scatter NaN test B2 -- NaN in LAST ROW of D.
 *
 * 16x16 FP32 NN GEMM, K=4, alpha=1, beta=0.
 * A = ones except A[M-1, *] (last row, all K columns) are NaN.
 * B = ones.
 *
 * Result: D[M-1, j] = NaN for all j.  Rows 0..M-2 = 4.0f.
 * The NaN sits at the very END of D's column-major buffer (last M elements
 * of each column, specifically index M-1 + M*j).
 *
 * This case stresses the scanner's coverage of the tail of the buffer.
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

    std::printf("[scatter-last-row] 16x16 K=4: A row %lld = NaN, B = ones -> D row %lld all NaN.\n",
                (long long)(M - 1), (long long)(M - 1));

    hipblasLtHandle_t handle{};
    hipStream_t       stream{};
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    std::vector<float> h_A(M * K, 1.0f);
    for(int64_t k = 0; k < K; ++k)
        h_A[(M - 1) + M * k] = std::nanf(""); // A[M-1, k]

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
        std::fprintf(stderr, "[scatter-last-row] SKIP: no heuristic returned\n");
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

    int last_row_nan = 0, other_nan = 0;
    for(int64_t j = 0; j < N; ++j)
    {
        if(std::isnan(h_D[(M - 1) + M * j]))
            ++last_row_nan;
    }
    for(int64_t i = 0; i < M - 1; ++i)
        for(int64_t j = 0; j < N; ++j)
            if(std::isnan(h_D[i + M * j]))
                ++other_nan;

    std::printf("[scatter-last-row] matmul status=%d  last-row NaN=%d/%lld  other NaN=%d\n",
                static_cast<int>(st), last_row_nan, (long long)N, other_nan);

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

    if(last_row_nan != static_cast<int>(N))
    {
        std::fprintf(stderr,
                     "[scatter-last-row] FAIL: expected %lld NaN in last row, got %d\n",
                     (long long)N, last_row_nan);
        return 1;
    }
    const char* cn = std::getenv("HIPBLASLT_CHECK_NUMERICS");
    const int   mode = cn ? std::atoi(cn) & 0x7 : 0;
    const bool  expect_fail = (mode & 4) != 0;
    if(expect_fail && st != HIPBLAS_STATUS_INVALID_VALUE)
    {
        std::fprintf(stderr,
                     "[scatter-last-row] FAIL: expected INVALID_VALUE, got %d\n",
                     static_cast<int>(st));
        return 1;
    }
    if(!expect_fail && st != HIPBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "[scatter-last-row] FAIL: unexpected status %d\n",
                     static_cast<int>(st));
        return 1;
    }
    std::printf("[scatter-last-row] PASS\n");
    return 0;
}
