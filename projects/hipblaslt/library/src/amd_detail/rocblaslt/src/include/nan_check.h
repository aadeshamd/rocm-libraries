/* ************************************************************************
 * NaN detection for hipBLASLt GEMM output — PoC header
 * HIPBLASLT_NAN_CHECK=0  (default) — disabled, zero overhead
 * HIPBLASLT_NAN_CHECK=1  — post-GEMM async scan; log on first NaN found
 * ************************************************************************ */

#pragma once
#ifndef NAN_CHECK_H
#define NAN_CHECK_H

#include <atomic>
#include <cstdint>
#include <hip/hip_runtime_api.h>

// ---------------------------------------------------------------------------
// Device-side flag layout (32 bytes, allocated once per handle)
// ---------------------------------------------------------------------------
struct HipblasltNanFlag
{
    uint32_t triggered; // atomicOr target; 0 = clean, 1 = NaN found
    uint32_t _pad;
    uint64_t call_id;  // which matmul invocation
    int32_t  algo_idx; // solution index
    int32_t  _pad2;
    int64_t  M, N, K; // shape at time of detection
};                     // 32 bytes total

// ---------------------------------------------------------------------------
// Per-handle NaN-check state (lives inside _rocblaslt_handle)
// ---------------------------------------------------------------------------
struct NanCheckState
{
    int      level = 0;       // value of HIPBLASLT_NAN_CHECK (0/1)
    void*    device_flag = nullptr; // HipblasltNanFlag on device
    uint64_t call_counter = 0;      // CPU counter, incremented per matmul

    bool enabled() const { return level > 0; }
};

// ---------------------------------------------------------------------------
// Lifecycle — called from handle constructor / destructor
// ---------------------------------------------------------------------------

// Read env var, allocate device flag buffer if enabled.
// Returns rocblaslt_status_success always (failures are non-fatal for PoC).
void nancheck_init(NanCheckState* state);

// Free device buffer.
void nancheck_destroy(NanCheckState* state);

// ---------------------------------------------------------------------------
// Post-GEMM hook — call immediately after runContractionProblem returns.
// D     : device pointer to GEMM output
// total : M * N * batch_count
// dtype : element type of D (HIP_R_32F / HIP_R_16F / HIP_R_16BF / HIP_R_64F)
// algo_idx : solution index (pass 0 if unavailable)
// M,N,K : problem shape
// stream : HIP stream the GEMM was submitted on
// ---------------------------------------------------------------------------
void nancheck_post_gemm(NanCheckState* state,
                        const void*    D,
                        int64_t        total,
                        hipDataType    dtype,
                        int32_t        algo_idx,
                        int64_t        M,
                        int64_t        N,
                        int64_t        K,
                        hipStream_t    stream);

#endif // NAN_CHECK_H
