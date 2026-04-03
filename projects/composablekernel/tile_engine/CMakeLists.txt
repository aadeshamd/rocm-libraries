# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_directories(BEFORE
        ${CMAKE_CURRENT_LIST_DIR}/include
    )

# Master flag to configure all tile engine benchmark ops
option(CK_BUILD_TILE_ENGINE "Configure all tile engine benchmark ops" OFF)

# Per-op flags (each defaults to OFF; setting CK_BUILD_TILE_ENGINE=ON enables all)
option(CK_BUILD_TILE_ENGINE_GEMM_UNIVERSAL "Configure tile engine gemm_universal benchmarks" OFF)
option(CK_BUILD_TILE_ENGINE_GEMM_MULTI_D "Configure tile engine gemm_multi_d benchmarks" OFF)
option(CK_BUILD_TILE_ENGINE_GEMM_PRESHUFFLE "Configure tile engine gemm_preshuffle benchmarks" OFF)
option(CK_BUILD_TILE_ENGINE_GROUPED_GEMM "Configure tile engine grouped_gemm benchmarks" OFF)
option(CK_BUILD_TILE_ENGINE_GEMM_STREAMK "Configure tile engine gemm_streamk benchmarks" OFF)
option(CK_BUILD_TILE_ENGINE_POOLING "Configure tile engine pooling benchmarks" OFF)
option(CK_BUILD_TILE_ENGINE_REDUCE "Configure tile engine reduce benchmarks" OFF)

# Resolve: master flag overrides individual flags
if(CK_BUILD_TILE_ENGINE)
    set(CK_BUILD_TILE_ENGINE_GEMM_UNIVERSAL ON)
    set(CK_BUILD_TILE_ENGINE_GEMM_MULTI_D ON)
    set(CK_BUILD_TILE_ENGINE_GEMM_PRESHUFFLE ON)
    set(CK_BUILD_TILE_ENGINE_GROUPED_GEMM ON)
    set(CK_BUILD_TILE_ENGINE_GEMM_STREAMK ON)
    set(CK_BUILD_TILE_ENGINE_POOLING ON)
    set(CK_BUILD_TILE_ENGINE_REDUCE ON)
endif()

# Any gemm sub-op requested means we enter the gemm subdirectory
if(CK_BUILD_TILE_ENGINE_GEMM_UNIVERSAL OR CK_BUILD_TILE_ENGINE_GEMM_MULTI_D OR CK_BUILD_TILE_ENGINE_GEMM_PRESHUFFLE OR CK_BUILD_TILE_ENGINE_GROUPED_GEMM)
    add_subdirectory(ops/gemm EXCLUDE_FROM_ALL)
endif()

if(CK_BUILD_TILE_ENGINE_GEMM_STREAMK)
    add_subdirectory(ops/gemm_streamk EXCLUDE_FROM_ALL)
endif()

if(CK_BUILD_TILE_ENGINE_POOLING)
    add_subdirectory(ops/pooling EXCLUDE_FROM_ALL)
endif()

if(CK_BUILD_TILE_ENGINE_REDUCE)
    add_subdirectory(ops/reduce EXCLUDE_FROM_ALL)
endif()

