# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_gemm_check_numerics
    "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_gemm_check_numerics.cpp")
target_link_libraries(sample_hipblaslt_gemm_check_numerics PRIVATE sample-helper)

# ---------------------------------------------------------------------------
# Deterministic scatter-NaN tests for HIPBLASLT_CHECK_NUMERICS.
# Each binary is a self-contained test case; run with the env var to exercise
# the scanner.  See the header comment in each .cpp for details.
# ---------------------------------------------------------------------------

# B0: clean control -- no NaN in A or D, scanner must NOT trigger.
add_executable(sample_check_numerics_scatter_clean
    "${CMAKE_CURRENT_SOURCE_DIR}/sample_check_numerics_scatter_clean.cpp")
target_link_libraries(sample_check_numerics_scatter_clean PRIVATE sample-helper)

# B1: NaN in first row of D (A row 0 poisoned).
add_executable(sample_check_numerics_scatter_first_row
    "${CMAKE_CURRENT_SOURCE_DIR}/sample_check_numerics_scatter_first_row.cpp")
target_link_libraries(sample_check_numerics_scatter_first_row PRIVATE sample-helper)

# B2: NaN in last row of D (A row 15 poisoned).
add_executable(sample_check_numerics_scatter_last_row
    "${CMAKE_CURRENT_SOURCE_DIR}/sample_check_numerics_scatter_last_row.cpp")
target_link_libraries(sample_check_numerics_scatter_last_row PRIVATE sample-helper)

# B3: NaN in middle row of D (A row 7 poisoned).
add_executable(sample_check_numerics_scatter_mid_row
    "${CMAKE_CURRENT_SOURCE_DIR}/sample_check_numerics_scatter_mid_row.cpp")
target_link_libraries(sample_check_numerics_scatter_mid_row PRIVATE sample-helper)

# B4: NaN in odd rows of D (A rows 1,3,5,...,15 poisoned -- interleaved scatter).
add_executable(sample_check_numerics_scatter_odd_rows
    "${CMAKE_CURRENT_SOURCE_DIR}/sample_check_numerics_scatter_odd_rows.cpp")
target_link_libraries(sample_check_numerics_scatter_odd_rows PRIVATE sample-helper)

# B5: single interior element NaN (alpha=0, beta=1, C[4,6]=NaN -> D[4,6] only).
add_executable(sample_check_numerics_scatter_single_element
    "${CMAKE_CURRENT_SOURCE_DIR}/sample_check_numerics_scatter_single_element.cpp")
target_link_libraries(sample_check_numerics_scatter_single_element PRIVATE sample-helper)

# ---------------------------------------------------------------------------
# Direct scanner unit test -- calls hipblaslt_check_numerics_output_D() with
# a pre-filled GPU buffer (no GEMM / no Tensile dependency).  Verifies the
# scanner kernel, log lines, and fail-bit status on any GPU architecture.
# ---------------------------------------------------------------------------
add_executable(sample_check_numerics_scanner_direct
    "${CMAKE_CURRENT_SOURCE_DIR}/sample_check_numerics_scanner_direct.cpp"
)
# Only hip::device is needed -- all helper symbols are stubbed inline in the
# test source so we avoid pulling in libhipblaslt.so (which would cause ODR
# conflicts with the stub definitions).
target_link_libraries(sample_check_numerics_scanner_direct PRIVATE hip::device)
# The scanner header lives in the private library include tree.
target_include_directories(sample_check_numerics_scanner_direct PRIVATE
    "${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include"
    "${PROJECT_SOURCE_DIR}/library/src/amd_detail/include"
    "${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/include"
)
target_compile_features(sample_check_numerics_scanner_direct PRIVATE cxx_std_17)
