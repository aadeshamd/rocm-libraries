// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

// Include only the necessary headers, avoiding device-specific code
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

// Forward declare to avoid including the full header with device code
namespace ck_tile {

template <typename RsLengths_,
          typename HsLengthss_,
          typename Ps2RHssMajor_,
          typename Ps2RHssMinor_,
          typename Ys2RHsMajor_,
          typename Ys2RHsMinor_>
struct tile_distribution_encoding;

} // namespace ck_tile

// Include the actual header after forward declarations
#include "ck_tile/core/tensor/tile_distribution_encoding.hpp"

using namespace ck_tile;

// ============================================================================
// tile_distribution_encoding basic construction tests
// ============================================================================

TEST(TileDistributionEncoding, BasicConstruction)
{
    // Simple 2D distribution encoding
    using RsLengths    = sequence<2>;
    using HsLengthss   = tuple<sequence<4>, sequence<8>>;
    using Ps2RHssMajor = tuple<sequence<0, 1>, sequence<1, 2>>;
    using Ps2RHssMinor = tuple<sequence<0, 0>, sequence<0, 0>>;
    using Ys2RHsMajor  = sequence<1, 2>;
    using Ys2RHsMinor  = sequence<0, 0>;

    using Encoding = tile_distribution_encoding<RsLengths,
                                                HsLengthss,
                                                Ps2RHssMajor,
                                                Ps2RHssMinor,
                                                Ys2RHsMajor,
                                                Ys2RHsMinor>;

    EXPECT_EQ(Encoding::NDimX, 2);
    EXPECT_EQ(Encoding::NDimP, 2);
    EXPECT_EQ(Encoding::NDimY, 2);
    EXPECT_EQ(Encoding::NDimR, 1);
}

TEST(TileDistributionEncoding, EmptyRDimension)
{
    // Test with no R dimension
    using RsLengths    = sequence<>;
    using HsLengthss   = tuple<sequence<2, 4>, sequence<16, 8, 8>>;
    using Ps2RHssMajor = tuple<sequence<1>, sequence<2>>;
    using Ps2RHssMinor = tuple<sequence<0>, sequence<0>>;
    using Ys2RHsMajor  = sequence<1, 2>;
    using Ys2RHsMinor  = sequence<0, 0>;

    using Encoding = tile_distribution_encoding<RsLengths,
                                                HsLengthss,
                                                Ps2RHssMajor,
                                                Ps2RHssMinor,
                                                Ys2RHsMajor,
                                                Ys2RHsMinor>;

    EXPECT_EQ(Encoding::NDimX, 2);
    EXPECT_EQ(Encoding::NDimP, 2);
    EXPECT_EQ(Encoding::NDimY, 2);
    EXPECT_EQ(Encoding::NDimR, 0);
}

TEST(TileDistributionEncoding, SingleDimension)
{
    // Test with single dimension
    using RsLengths    = sequence<>;
    using HsLengthss   = tuple<sequence<32>>;
    using Ps2RHssMajor = tuple<sequence<1>>;
    using Ps2RHssMinor = tuple<sequence<0>>;
    using Ys2RHsMajor  = sequence<1>;
    using Ys2RHsMinor  = sequence<0>;

    using Encoding = tile_distribution_encoding<RsLengths,
                                                HsLengthss,
                                                Ps2RHssMajor,
                                                Ps2RHssMinor,
                                                Ys2RHsMajor,
                                                Ys2RHsMinor>;

    EXPECT_EQ(Encoding::NDimX, 1);
    EXPECT_EQ(Encoding::NDimP, 1);
    EXPECT_EQ(Encoding::NDimY, 1);
    EXPECT_EQ(Encoding::NDimR, 0);
}

// ============================================================================
// tile_distribution_encoding::detail tests
// ============================================================================

TEST(TileDistributionEncodingDetail, DimensionCounts)
{
    using RsLengths    = sequence<3>;
    using HsLengthss   = tuple<sequence<1, 4, 32>, sequence<4, 1, 4, 2, 4>>;
    using Ps2RHssMajor = tuple<sequence<0, 1, 2>, sequence<1, 2>>;
    using Ps2RHssMinor = tuple<sequence<0, 0, 0>, sequence<1, 2>>;
    using Ys2RHsMajor  = sequence<1, 2>;
    using Ys2RHsMinor  = sequence<0, 1>;

    using Encoding = tile_distribution_encoding<RsLengths,
                                                HsLengthss,
                                                Ps2RHssMajor,
                                                Ps2RHssMinor,
                                                Ys2RHsMajor,
                                                Ys2RHsMinor>;

    using Detail = typename Encoding::detail;

    EXPECT_EQ(Detail::ndim_rh_major_, 3);      // NDimX + 1
    EXPECT_EQ(Detail::ndim_span_major_, 2);    // NDimX
    EXPECT_EQ(Detail::ndims_rhs_minor_[0], 1); // R dimension has 1 element
    EXPECT_EQ(Detail::ndims_rhs_minor_[1], 3); // First H dimension has 3 elements
    EXPECT_EQ(Detail::ndims_rhs_minor_[2], 5); // Second H dimension has 5 elements
}

TEST(TileDistributionEncodingDetail, RhsLengths)
{
    using RsLengths    = sequence<2>;
    using HsLengthss   = tuple<sequence<4>, sequence<8>>;
    using Ps2RHssMajor = tuple<sequence<0, 1>, sequence<1, 2>>;
    using Ps2RHssMinor = tuple<sequence<0, 0>, sequence<0, 0>>;
    using Ys2RHsMajor  = sequence<1, 2>;
    using Ys2RHsMinor  = sequence<0, 0>;

    using Encoding = tile_distribution_encoding<RsLengths,
                                                HsLengthss,
                                                Ps2RHssMajor,
                                                Ps2RHssMinor,
                                                Ys2RHsMajor,
                                                Ys2RHsMinor>;

    using Detail = typename Encoding::detail;

    // Check RHS lengths
    EXPECT_EQ(Detail::rhs_lengthss_[0][0], 2); // R length
    EXPECT_EQ(Detail::rhs_lengthss_[1][0], 4); // First H length
    EXPECT_EQ(Detail::rhs_lengthss_[2][0], 8); // Second H length
}

TEST(TileDistributionEncodingDetail, YsLengths)
{
    using RsLengths    = sequence<3>;
    using HsLengthss   = tuple<sequence<4, 8>, sequence<16>>;
    using Ps2RHssMajor = tuple<sequence<0>, sequence<1, 2>>;
    using Ps2RHssMinor = tuple<sequence<0>, sequence<0, 0>>;
    using Ys2RHsMajor  = sequence<1, 2>;
    using Ys2RHsMinor  = sequence<1, 0>;

    using Encoding = tile_distribution_encoding<RsLengths,
                                                HsLengthss,
                                                Ps2RHssMajor,
                                                Ps2RHssMinor,
                                                Ys2RHsMajor,
                                                Ys2RHsMinor>;

    using Detail = typename Encoding::detail;

    // Y0 points to H[0][1], Y1 points to H[1][0]
    EXPECT_EQ(Detail::ys_lengths_[0], 8);  // H[0][1] = 8
    EXPECT_EQ(Detail::ys_lengths_[1], 16); // H[1][0] = 16
}

TEST(TileDistributionEncodingDetail, MaxNdimRhMinor)
{
    using RsLengths    = sequence<3>;
    using HsLengthss   = tuple<sequence<1, 4>, sequence<4, 1, 4, 2, 4>>;
    using Ps2RHssMajor = tuple<sequence<0, 1>>;
    using Ps2RHssMinor = tuple<sequence<0, 0>>;
    using Ys2RHsMajor  = sequence<1, 2>;
    using Ys2RHsMinor  = sequence<0, 1>;

    using Encoding = tile_distribution_encoding<RsLengths,
                                                HsLengthss,
                                                Ps2RHssMajor,
                                                Ps2RHssMinor,
                                                Ys2RHsMajor,
                                                Ys2RHsMinor>;

    using Detail = typename Encoding::detail;

    // Maximum of R dimension (1), H0 (2), H1 (5) = 5
    EXPECT_EQ(Detail::max_ndim_rh_minor_, 5);
}

TEST(TileDistributionEncodingDetail, DoesP0wnR)
{
    using RsLengths    = sequence<2, 3>;
    using HsLengthss   = tuple<sequence<4>, sequence<8>>;
    using Ps2RHssMajor = tuple<sequence<0, 1>, sequence<0, 2>>;
    using Ps2RHssMinor = tuple<sequence<0, 0>, sequence<1, 0>>;
    using Ys2RHsMajor  = sequence<1, 2>;
    using Ys2RHsMinor  = sequence<0, 0>;

    using Encoding = tile_distribution_encoding<RsLengths,
                                                HsLengthss,
                                                Ps2RHssMajor,
                                                Ps2RHssMinor,
                                                Ys2RHsMajor,
                                                Ys2RHsMinor>;

    using Detail = typename Encoding::detail;

    // P0 owns R[0], P1 owns R[1]
    EXPECT_TRUE(Detail::does_p_own_r_[0][0]);
    EXPECT_FALSE(Detail::does_p_own_r_[0][1]);
    EXPECT_FALSE(Detail::does_p_own_r_[1][0]);
    EXPECT_TRUE(Detail::does_p_own_r_[1][1]);
}

TEST(TileDistributionEncodingDetail, GetUniformedHDimLengths)
{
    using RsLengths    = sequence<3>;
    using HsLengthss   = tuple<sequence<1, 4, 32>, sequence<4, 1, 4, 2, 4>>;
    using Ps2RHssMajor = tuple<sequence<0, 1>>;
    using Ps2RHssMinor = tuple<sequence<0, 0>>;
    using Ys2RHsMajor  = sequence<1, 2>;
    using Ys2RHsMinor  = sequence<0, 1>;

    using Encoding = tile_distribution_encoding<RsLengths,
                                                HsLengthss,
                                                Ps2RHssMajor,
                                                Ps2RHssMinor,
                                                Ys2RHsMajor,
                                                Ys2RHsMinor>;

    using Detail = typename Encoding::detail;

    constexpr auto h_dim_lengths = Detail::get_uniformed_h_dim_lengths();

    // Should be sequence<3, 5> (lengths of the two H dimensions)
    EXPECT_EQ(h_dim_lengths.size(), 2);
    EXPECT_EQ(h_dim_lengths.at(0), 3);
    EXPECT_EQ(h_dim_lengths.at(1), 5);
}

TEST(TileDistributionEncodingDetail, GetUniformedRhDimLengths)
{
    using RsLengths    = sequence<2, 5>;
    using HsLengthss   = tuple<sequence<2, 4>, sequence<16, 8, 8>>;
    using Ps2RHssMajor = tuple<sequence<0>>;
    using Ps2RHssMinor = tuple<sequence<0>>;
    using Ys2RHsMajor  = sequence<1, 2>;
    using Ys2RHsMinor  = sequence<0, 0>;

    using Encoding = tile_distribution_encoding<RsLengths,
                                                HsLengthss,
                                                Ps2RHssMajor,
                                                Ps2RHssMinor,
                                                Ys2RHsMajor,
                                                Ys2RHsMinor>;

    using Detail = typename Encoding::detail;

    constexpr auto rh_dim_lengths = Detail::get_uniformed_rh_dim_lengths();

    // Should be sequence<2, 2, 3> (R has 2 dims, H0 has 2 dims, H1 has 3 dims)
    EXPECT_EQ(rh_dim_lengths.size(), 3);
    EXPECT_EQ(rh_dim_lengths.at(0), 2); // R dimension count
    EXPECT_EQ(rh_dim_lengths.at(1), 2); // H0 dimension count
    EXPECT_EQ(rh_dim_lengths.at(2), 3); // H1 dimension count
}

TEST(TileDistributionEncodingDetail, GetHDimLengthsPrefixSum)
{
    using RsLengths    = sequence<3>;
    using HsLengthss   = tuple<sequence<1, 4, 32>, sequence<4, 1, 4, 2, 4>>;
    using Ps2RHssMajor = tuple<sequence<0, 1>>;
    using Ps2RHssMinor = tuple<sequence<0, 0>>;
    using Ys2RHsMajor  = sequence<1, 2>;
    using Ys2RHsMinor  = sequence<0, 1>;

    using Encoding = tile_distribution_encoding<RsLengths,
                                                HsLengthss,
                                                Ps2RHssMajor,
                                                Ps2RHssMinor,
                                                Ys2RHsMajor,
                                                Ys2RHsMinor>;

    using Detail = typename Encoding::detail;

    constexpr auto h_dim_prefix_sum = Detail::get_h_dim_lengths_prefix_sum();

    // From sequence<3, 5>, prefix sum should be sequence<0, 3, 8>
    EXPECT_EQ(h_dim_prefix_sum.size(), 3);
    EXPECT_EQ(h_dim_prefix_sum.at(0), 0);
    EXPECT_EQ(h_dim_prefix_sum.at(1), 3);
    EXPECT_EQ(h_dim_prefix_sum.at(2), 8);
}

TEST(TileDistributionEncodingDetail, PsOverRsDerivative)
{
    using RsLengths    = sequence<2, 3>;
    using HsLengthss   = tuple<sequence<4>>;
    using Ps2RHssMajor = tuple<sequence<0, 0, 1>>;
    using Ps2RHssMinor = tuple<sequence<0, 1, 0>>;
    using Ys2RHsMajor  = sequence<1>;
    using Ys2RHsMinor  = sequence<0>;

    using Encoding = tile_distribution_encoding<RsLengths,
                                                HsLengthss,
                                                Ps2RHssMajor,
                                                Ps2RHssMinor,
                                                Ys2RHsMajor,
                                                Ys2RHsMinor>;

    using Detail = typename Encoding::detail;

    // P0 maps to R[0], R[1], H[0][0]
    // In reverse order: H[0][0]=4, R[1]=3, R[0]=2
    // Derivative for R[0] = 4*3 = 12
    // Derivative for R[1] = 4
    EXPECT_EQ(Detail::ps_over_rs_derivative_[0][0], 12);
    EXPECT_EQ(Detail::ps_over_rs_derivative_[0][1], 4);
}

// ============================================================================
// tile_distribution_encoding_shuffle tests
// ============================================================================

TEST(TileDistributionEncodingShuffle, BasicShuffle)
{
    using RsLengths    = sequence<2>;
    using HsLengthss   = tuple<sequence<4>, sequence<8>>;
    using Ps2RHssMajor = tuple<sequence<0, 1>, sequence<1, 2>>;
    using Ps2RHssMinor = tuple<sequence<0, 0>, sequence<0, 0>>;
    using Ys2RHsMajor  = sequence<1, 2>;
    using Ys2RHsMinor  = sequence<0, 0>;

    using Encoding = tile_distribution_encoding<RsLengths,
                                                HsLengthss,
                                                Ps2RHssMajor,
                                                Ps2RHssMinor,
                                                Ys2RHsMajor,
                                                Ys2RHsMinor>;

    // Reverse shuffle: sequence<1, 0>
    using Shuffled = tile_distribution_encoding_shuffle_t<Encoding, sequence<1, 0>>;

    // After shuffle, Y dimensions should be reversed
    EXPECT_EQ(Shuffled::NDimY, 2);
    EXPECT_EQ(Shuffled::ys_to_rhs_major_.at(0), 2); // Was at position 1
    EXPECT_EQ(Shuffled::ys_to_rhs_major_.at(1), 1); // Was at position 0
}

TEST(TileDistributionEncodingShuffle, IdentityShuffle)
{
    using RsLengths    = sequence<2>;
    using HsLengthss   = tuple<sequence<4>, sequence<8>, sequence<16>>;
    using Ps2RHssMajor = tuple<sequence<0, 1>>;
    using Ps2RHssMinor = tuple<sequence<0, 0>>;
    using Ys2RHsMajor  = sequence<1, 2, 3>;
    using Ys2RHsMinor  = sequence<0, 0, 0>;

    using Encoding = tile_distribution_encoding<RsLengths,
                                                HsLengthss,
                                                Ps2RHssMajor,
                                                Ps2RHssMinor,
                                                Ys2RHsMajor,
                                                Ys2RHsMinor>;

    // Identity shuffle: sequence<0, 1, 2>
    using Shuffled = tile_distribution_encoding_shuffle_t<Encoding, sequence<0, 1, 2>>;

    // After identity shuffle, everything should remain the same
    EXPECT_EQ(Shuffled::NDimY, 3);
    EXPECT_EQ(Shuffled::ys_to_rhs_major_.at(0), 1);
    EXPECT_EQ(Shuffled::ys_to_rhs_major_.at(1), 2);
    EXPECT_EQ(Shuffled::ys_to_rhs_major_.at(2), 3);
}

// Note: Tests for make_embed_tile_distribution_encoding and
// make_reduce_tile_distribution_encoding are omitted because these functions
// have constexpr evaluation issues that prevent them from being tested in the
// current test framework setup.
