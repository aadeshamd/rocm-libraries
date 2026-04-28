/* ************************************************************************
 * nan_scan.hip — Post-GEMM NaN scan kernel for hipBLASLt (PoC)
 *
 * Launched on the SAME stream as the GEMM; no host synchronisation needed.
 * The atomicOr is only hit when a NaN is actually found — the clean path
 * (no NaN) has zero atomic cost.
 * ************************************************************************ */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// GPU kernel
// ---------------------------------------------------------------------------
__global__ void hipblaslt_nan_scan_kernel(const void* __restrict__ D,
                                          int64_t     total_elements,
                                          int32_t     dtype_int, // hipDataType cast to int
                                          uint32_t*   flag_triggered,
                                          uint64_t*   flag_call_id,
                                          int32_t*    flag_algo_idx,
                                          uint64_t    call_id,
                                          int32_t     algo_idx,
                                          int64_t     M,
                                          int64_t     N,
                                          int64_t     K)
{
    // Short-circuit: if a previous call already set the flag, skip all work.
    if(__builtin_expect(*flag_triggered != 0u, 0))
        return;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= total_elements)
        return;

    bool nan_found = false;

    // dtype_int values match hipDataType enum:
    //   HIP_R_32F  = 0
    //   HIP_R_16F  = 1
    //   HIP_R_64F  = 4
    //   HIP_R_16BF = 14
    switch(dtype_int)
    {
    case 0: // HIP_R_32F
        nan_found = __isnanf(((const float*)D)[idx]);
        break;
    case 1: // HIP_R_16F
        nan_found = __hisnan(((const __half*)D)[idx]);
        break;
    case 14: // HIP_R_16BF  — BF16: exp all-ones + non-zero mantissa
    {
        uint16_t bits = ((const uint16_t*)D)[idx];
        nan_found     = ((bits & 0x7F80u) == 0x7F80u) && (bits & 0x007Fu);
        break;
    }
    case 4: // HIP_R_64F
        nan_found = __isnan(((const double*)D)[idx]);
        break;
    default:
        break;
    }

    if(nan_found)
    {
        // Write metadata first (idempotent races are harmless for PoC).
        *flag_call_id  = call_id;
        *flag_algo_idx = algo_idx;
        // Fence so readers see metadata before seeing triggered == 1.
        __threadfence();
        atomicOr(flag_triggered, 1u);
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher (called from nan_check.cpp — plain C++ linkage)
// ---------------------------------------------------------------------------
extern "C" void hipblaslt_launch_nan_scan(const void* D,
                                          int64_t     total_elements,
                                          int32_t     dtype_int,
                                          void*       device_flag, // HipblasltNanFlag*
                                          uint64_t    call_id,
                                          int32_t     algo_idx,
                                          int64_t     M,
                                          int64_t     N,
                                          int64_t     K,
                                          hipStream_t stream)
{
    if(total_elements <= 0)
        return;

    // Device flag layout (matches HipblasltNanFlag):
    //   [0]  uint32_t triggered
    //   [1]  uint32_t _pad
    //   [2-3] uint64_t call_id   (byte offset 8)
    //   [4]  int32_t  algo_idx   (byte offset 16)
    //   [5]  int32_t  _pad2
    //   [6-7] int64_t M          (byte offset 24) — not written by kernel for now
    uint32_t* flag_triggered = (uint32_t*)device_flag;
    uint64_t* flag_call_id   = (uint64_t*)((char*)device_flag + 8);
    int32_t*  flag_algo_idx  = (int32_t*)((char*)device_flag + 16);

    constexpr int BLOCK = 256;
    int64_t       grid  = (total_elements + BLOCK - 1) / BLOCK;
    // Cap grid to avoid launch failures on very large matrices (split if needed).
    if(grid > 0x7fffffff)
        grid = 0x7fffffff;

    hipblaslt_nan_scan_kernel<<<(uint32_t)grid, BLOCK, 0, stream>>>(D,
                                                                     total_elements,
                                                                     dtype_int,
                                                                     flag_triggered,
                                                                     flag_call_id,
                                                                     flag_algo_idx,
                                                                     call_id,
                                                                     algo_idx,
                                                                     M,
                                                                     N,
                                                                     K);
    // Intentionally no hipStreamSynchronize — async, runs after GEMM on same stream.
}
