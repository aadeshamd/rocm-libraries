// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstdio>

#include "ck_tile/core/arch/arch.hpp"
#include <hip/hip_runtime.h>
#include "ck_tile/host/hip_check_error.hpp"

namespace {

// ---------------------------------------------------------------------------
// Device compiler target query
// ---------------------------------------------------------------------------
// These kernels read compile-time target constants baked into the device binary
// and write them to device memory so host code can retrieve them at runtime.

__attribute__((used)) __global__ void getWaveSizeForSelectedOp(uint32_t* waveSize)
{
    using CompilerTarget = decltype(ck_tile::core::arch::get_compiler_target());

    if(waveSize)
        *waveSize = static_cast<uint32_t>(CompilerTarget::WAVE_SIZE_ID);
}

__global__ void getTargetIdForSelectedOp(uint32_t* targetId)
{
    using CompilerTarget = decltype(ck_tile::core::arch::get_compiler_target());

    if(targetId)
        *targetId = static_cast<uint32_t>(CompilerTarget::TARGET_ID);
}

[[maybe_unused]] static __host__ uint32_t getDeviceWaveSize()
{
    uint32_t* d_wave_size;
    HIP_CHECK_ERROR(hipMalloc(&d_wave_size, sizeof(uint32_t)));
    getWaveSizeForSelectedOp<<<1, 64>>>(d_wave_size);
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    uint32_t wave_size;
    HIP_CHECK_ERROR(hipMemcpy(&wave_size, d_wave_size, sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(hipFree(d_wave_size));
    return wave_size;
}

/// Returns the amdgcn_target_id that the device code was compiled for.
/// This lets host-side test code construct matching compile-time target types.
static __host__ ck_tile::core::arch::amdgcn_target_id getDeviceCompilerTargetId()
{
    uint32_t* d_target_id;
    HIP_CHECK_ERROR(hipMalloc(&d_target_id, sizeof(uint32_t)));
    getTargetIdForSelectedOp<<<1, 64>>>(d_target_id);
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    uint32_t target_id;
    HIP_CHECK_ERROR(hipMemcpy(&target_id, d_target_id, sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(hipFree(d_target_id));
    return static_cast<ck_tile::core::arch::amdgcn_target_id>(target_id);
}

// ---------------------------------------------------------------------------
// Compile-time dispatch: maps a runtime amdgcn_target_id to the corresponding
// amdgcn_target<...> type and invokes a user-provided functor with it.
//
// Usage:
//   dispatchCompilerTarget(getDeviceCompilerTargetId(), [](auto target) {
//       using Target = decltype(target);
//       // Target is now the correct amdgcn_target<...> type
//   });
//
// Returns false if the target_id is not recognized (unsupported).
// ---------------------------------------------------------------------------
template <typename Func>
static __host__ bool dispatchCompilerTarget(ck_tile::core::arch::amdgcn_target_id id, Func&& func)
{
    using namespace ck_tile::core::arch;

    // clang-format off
    switch(id)
    {
    case amdgcn_target_id::GFX908:        func(make_amdgcn_gfx9_target<amdgcn_target_id::GFX908>());        return true;
    case amdgcn_target_id::GFX90A:        func(make_amdgcn_gfx9_target<amdgcn_target_id::GFX90A>());        return true;
    case amdgcn_target_id::GFX942:        func(make_amdgcn_gfx9_target<amdgcn_target_id::GFX942>());        return true;
    case amdgcn_target_id::GFX950:        func(make_amdgcn_gfx9_target<amdgcn_target_id::GFX950>());        return true;
    case amdgcn_target_id::GFX1030:       func(make_amdgcn_gfx10_3_target<amdgcn_target_id::GFX1030>());    return true;
    case amdgcn_target_id::GFX1031:       func(make_amdgcn_gfx10_3_target<amdgcn_target_id::GFX1031>());    return true;
    case amdgcn_target_id::GFX1032:       func(make_amdgcn_gfx10_3_target<amdgcn_target_id::GFX1032>());    return true;
    case amdgcn_target_id::GFX1034:       func(make_amdgcn_gfx10_3_target<amdgcn_target_id::GFX1034>());    return true;
    case amdgcn_target_id::GFX1035:       func(make_amdgcn_gfx10_3_target<amdgcn_target_id::GFX1035>());    return true;
    case amdgcn_target_id::GFX1036:       func(make_amdgcn_gfx10_3_target<amdgcn_target_id::GFX1036>());    return true;
    case amdgcn_target_id::GFX103_GENERIC:func(make_amdgcn_gfx10_3_target<amdgcn_target_id::GFX103_GENERIC>()); return true;
    case amdgcn_target_id::GFX1100:       func(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1100>());      return true;
    case amdgcn_target_id::GFX1101:       func(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1101>());      return true;
    case amdgcn_target_id::GFX1102:       func(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1102>());      return true;
    case amdgcn_target_id::GFX1103:       func(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1103>());      return true;
    case amdgcn_target_id::GFX1150:       func(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1150>());      return true;
    case amdgcn_target_id::GFX1151:       func(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1151>());      return true;
    case amdgcn_target_id::GFX1152:       func(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1152>());      return true;
    case amdgcn_target_id::GFX1153:       func(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1153>());      return true;
    case amdgcn_target_id::GFX11_GENERIC: func(make_amdgcn_gfx11_target<amdgcn_target_id::GFX11_GENERIC>()); return true;
    case amdgcn_target_id::GFX1200:       func(make_amdgcn_gfx12_target<amdgcn_target_id::GFX1200>());      return true;
    case amdgcn_target_id::GFX1201:       func(make_amdgcn_gfx12_target<amdgcn_target_id::GFX1201>());      return true;
    case amdgcn_target_id::GFX12_GENERIC: func(make_amdgcn_gfx12_target<amdgcn_target_id::GFX12_GENERIC>()); return true;
    case amdgcn_target_id::HOST:          return false;
    }
    // clang-format on
}

} // namespace
