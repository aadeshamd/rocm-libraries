# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(CK_BUILD_TILE_ENGINE_GEMM_UNIVERSAL)
    add_subdirectory(gemm_universal EXCLUDE_FROM_ALL)
endif()

if(CK_BUILD_TILE_ENGINE_GEMM_MULTI_D)
    add_subdirectory(gemm_multi_d EXCLUDE_FROM_ALL)
endif()

if(CK_BUILD_TILE_ENGINE_GEMM_PRESHUFFLE)
    add_subdirectory(gemm_preshuffle EXCLUDE_FROM_ALL)
endif()

if(CK_BUILD_TILE_ENGINE_GROUPED_GEMM)
    add_subdirectory(grouped_gemm EXCLUDE_FROM_ALL)
endif()