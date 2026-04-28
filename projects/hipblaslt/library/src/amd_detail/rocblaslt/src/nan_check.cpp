/* ************************************************************************
 * nan_check.cpp — NaN detection host logic for hipBLASLt (PoC)
 * ************************************************************************ */

#include "nan_check.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <hip/hip_runtime_api.h>

// ---------------------------------------------------------------------------
// Forward declaration of the HIP launcher (defined in nan_scan.hip).
// ---------------------------------------------------------------------------
extern "C" void hipblaslt_launch_nan_scan(const void* D,
                                          int64_t     total_elements,
                                          int32_t     dtype_int,
                                          void*       device_flag,
                                          uint64_t    call_id,
                                          int32_t     algo_idx,
                                          int64_t     M,
                                          int64_t     N,
                                          int64_t     K,
                                          hipStream_t stream);

// ---------------------------------------------------------------------------
// Log helpers
// ---------------------------------------------------------------------------
static std::mutex g_log_mutex;

static FILE* open_nan_log()
{
    const char* path = getenv("HIPBLASLT_NAN_LOG_FILE");
    if(path && path[0] != '\0')
    {
        FILE* f = fopen(path, "a");
        if(f)
            return f;
        fprintf(stderr, "[HIPBLASLT_NAN_CHECK] Warning: cannot open log file '%s', using stderr\n",
                path);
    }
    return stderr;
}

static const char* dtype_name(int32_t dtype_int)
{
    switch(dtype_int)
    {
    case 0: return "F32";
    case 1: return "F16";
    case 4: return "F64";
    case 14: return "BF16";
    default: return "unknown";
    }
}

// Emit log line. Reads device flag synchronously (small, 32-byte copy).
static void emit_nan_log(NanCheckState* state,
                          int32_t        dtype_int,
                          int64_t        M,
                          int64_t        N,
                          int64_t        K,
                          int32_t        algo_idx_arg,
                          uint64_t       call_id_arg,
                          hipStream_t    stream)
{
    // Synchronise only the side-read — we use hipMemcpy (blocking caller
    // thread) so we don't touch the compute stream ordering.
    hipStreamSynchronize(stream); // wait for scan kernel to finish

    HipblasltNanFlag host_flag;
    if(hipMemcpy(&host_flag, state->device_flag, sizeof(host_flag), hipMemcpyDeviceToHost)
       != hipSuccess)
    {
        fprintf(stderr, "[HIPBLASLT_NAN_CHECK] hipMemcpy failed — cannot read flag\n");
        return;
    }

    if(!host_flag.triggered)
        return; // clean run

    std::lock_guard<std::mutex> lk(g_log_mutex);
    FILE*                       log = open_nan_log();

    fprintf(log,
            "[HIPBLASLT_NAN_CHECK] NaN detected in matmul output D\n"
            "  call_id  = %" PRIu64 "\n"
            "  algo_idx = %" PRId32 "\n"
            "  M=%" PRId64 "  N=%" PRId64 "  K=%" PRId64 "\n"
            "  type_d   = %s\n",
            host_flag.call_id,
            host_flag.algo_idx,
            M,
            N,
            K,
            dtype_name(dtype_int));
    fflush(log);
    if(log != stderr)
        fclose(log);
}

// ---------------------------------------------------------------------------
// NanCheckState lifecycle
// ---------------------------------------------------------------------------
void nancheck_init(NanCheckState* state)
{
    const char* env = getenv("HIPBLASLT_NAN_CHECK");
    if(!env)
        return;

    int lvl = atoi(env);
    if(lvl <= 0)
        return;

    state->level = lvl;

    // Allocate device flag buffer and zero-initialise it.
    if(hipMalloc(&state->device_flag, sizeof(HipblasltNanFlag)) != hipSuccess)
    {
        fprintf(stderr, "[HIPBLASLT_NAN_CHECK] hipMalloc failed — NaN check disabled\n");
        state->level       = 0;
        state->device_flag = nullptr;
        return;
    }
    hipMemset(state->device_flag, 0, sizeof(HipblasltNanFlag));

    fprintf(stderr, "[HIPBLASLT_NAN_CHECK] enabled (level=%d)\n", state->level);
}

void nancheck_destroy(NanCheckState* state)
{
    if(state->device_flag)
    {
        hipFree(state->device_flag);
        state->device_flag = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Post-GEMM hook
// ---------------------------------------------------------------------------
void nancheck_post_gemm(NanCheckState* state,
                        const void*    D,
                        int64_t        total,
                        hipDataType    dtype,
                        int32_t        algo_idx,
                        int64_t        M,
                        int64_t        N,
                        int64_t        K,
                        hipStream_t    stream)
{
    if(!state->enabled() || !state->device_flag || !D || total <= 0)
        return;

    uint64_t call_id = ++state->call_counter;

    // Level 1: stop-at-first — flag is never reset; kernel self-short-circuits
    //          once triggered (via the early-return guard in the kernel).
    // Write call_id/algo_idx metadata to host fields too so emit_nan_log can
    // use the values captured by the kernel.
    hipblaslt_launch_nan_scan(D,
                              total,
                              (int32_t)dtype,
                              state->device_flag,
                              call_id,
                              algo_idx,
                              M,
                              N,
                              K,
                              stream);

    // Check result asynchronously only after the kernel runs.
    // For PoC simplicity we do a blocking read here; a production version
    // would defer this to a customer-called hipblasLtNanCheckQuery().
    emit_nan_log(state, (int32_t)dtype, M, N, K, algo_idx, call_id, stream);
}
