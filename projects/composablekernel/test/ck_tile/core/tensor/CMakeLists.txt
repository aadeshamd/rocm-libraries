# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

message("-- Adding: test/ck_tile/core/tensor/")

add_gtest_executable(test_tile_distribution_encoding test_tile_distribution_encoding.cpp)
