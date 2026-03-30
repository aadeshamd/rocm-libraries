// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleDequantize.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;

template <typename T1, typename T2>
struct TypePair
{
    using First = T1;
    using Second = T2;
};

// ============================================================================
// Typed tests over input type: float/half/bfloat16 with float scale
// ============================================================================

using TypesBlockScaleDequantize
    = ::testing::Types<TypePair<float, float>, TypePair<half, float>, TypePair<bfloat16, float>>;

template <class T>
class CpuFpReferenceBlockScaleDequantizeTyped : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceBlockScaleDequantizeTyped, TypesBlockScaleDequantize, );

TYPED_TEST(CpuFpReferenceBlockScaleDequantizeTyped, IdentityScale)
{
    using XType = typename TypeParam::First;
    using ScaleType = typename TypeParam::Second;

    // X: 2x4, Scale: 2x2 (block_size=2 along dim 1)
    Tensor<XType> xTensor({2, 4});
    Tensor<ScaleType> scaleTensor({2, 2});
    Tensor<float> yTensor({2, 4});

    xTensor.fillWithValue(static_cast<XType>(3.0f));
    scaleTensor.fillWithValue(static_cast<ScaleType>(1.0f));

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-5f;
    for(int b = 0; b < 2; ++b)
    {
        for(int c = 0; c < 4; ++c)
        {
            EXPECT_NEAR(yTensor.getHostValue(b, c), 3.0f, tolerance);
        }
    }
}

TEST(TestCpuFpReferenceBlockScaleDequantizeTyped, BFloat16InBFloat16Out)
{
    // bfloat16 input, float scale, bfloat16 output — registered signature gap
    Tensor<bfloat16> xTensor({1, 4});
    Tensor<float> scaleTensor({1, 2});
    Tensor<bfloat16> yTensor({1, 4});

    xTensor.setHostValue(bfloat16(2.0f), 0, 0);
    xTensor.setHostValue(bfloat16(4.0f), 0, 1);
    xTensor.setHostValue(bfloat16(1.0f), 0, 2);
    xTensor.setHostValue(bfloat16(3.0f), 0, 3);

    scaleTensor.setHostValue(0.5f, 0, 0); // scales elements [0,1]
    scaleTensor.setHostValue(2.0f, 0, 1); // scales elements [2,3]

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-2f; // bfloat16 has ~2 decimal digits of precision
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 0)), 2.0f * 0.5f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 1)), 4.0f * 0.5f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 2)), 1.0f * 2.0f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 3)), 3.0f * 2.0f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp32, NonTrivialScale)
{
    // X: 1x4, Scale: 1x2 => block_size=2 along dim 1
    Tensor<float> xTensor({1, 4});
    Tensor<float> scaleTensor({1, 2});
    Tensor<float> yTensor({1, 4});

    xTensor.setHostValue(1.0f, 0, 0);
    xTensor.setHostValue(2.0f, 0, 1);
    xTensor.setHostValue(3.0f, 0, 2);
    xTensor.setHostValue(4.0f, 0, 3);

    scaleTensor.setHostValue(2.0f, 0, 0); // scales elements [0,1]
    scaleTensor.setHostValue(0.5f, 0, 1); // scales elements [2,3]

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-5f;
    EXPECT_NEAR(yTensor.getHostValue(0, 0), 1.0f * 2.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1), 2.0f * 2.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 2), 3.0f * 0.5f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3), 4.0f * 0.5f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp32, MultiDimBlocking)
{
    // X: 2x32x32x64, Scale: 2x32x32x2 => block along trailing dim 3, block_size=32
    Tensor<float> xTensor({2, 32, 32, 64});
    Tensor<float> scaleTensor({2, 32, 32, 2});
    Tensor<float> yTensor({2, 32, 32, 64});

    xTensor.fillWithValue(1.0f);
    scaleTensor.fillWithValue(3.0f);

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {32}, false);

    auto tolerance = 1e-5f;
    // Spot check a few values
    EXPECT_NEAR(yTensor.getHostValue(0, 0, 0, 0), 3.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(1, 31, 31, 63), 3.0f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp32, MultiTrailingDimBlockSize)
{
    // X: 1x4x6, Scale: 1x2x3 => blockSize={2,2} blocks trailing two dims simultaneously
    // dim 1: block_size=2 (4/2=2 scale entries), dim 2: block_size=2 (6/2=3 scale entries)
    Tensor<float> xTensor({1, 4, 6});
    Tensor<float> scaleTensor({1, 2, 3});
    Tensor<float> yTensor({1, 4, 6});

    xTensor.fillWithValue(1.0f);

    // Assign distinct scale values per block so we can verify correct index mapping
    // scale[0][s1][s2] covers x[0][s1*2..s1*2+1][s2*2..s2*2+1]
    scaleTensor.setHostValue(1.0f, 0, 0, 0);
    scaleTensor.setHostValue(2.0f, 0, 0, 1);
    scaleTensor.setHostValue(3.0f, 0, 0, 2);
    scaleTensor.setHostValue(4.0f, 0, 1, 0);
    scaleTensor.setHostValue(5.0f, 0, 1, 1);
    scaleTensor.setHostValue(6.0f, 0, 1, 2);

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2, 2}, false);

    auto tolerance = 1e-5f;
    // Block (0,0): x[0][0..1][0..1] => scale[0][0][0] = 1.0
    EXPECT_NEAR(yTensor.getHostValue(0, 0, 0), 1.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1, 1), 1.0f, tolerance);
    // Block (0,1): x[0][0..1][2..3] => scale[0][0][1] = 2.0
    EXPECT_NEAR(yTensor.getHostValue(0, 0, 2), 2.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1, 3), 2.0f, tolerance);
    // Block (0,2): x[0][0..1][4..5] => scale[0][0][2] = 3.0
    EXPECT_NEAR(yTensor.getHostValue(0, 0, 4), 3.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1, 5), 3.0f, tolerance);
    // Block (1,0): x[0][2..3][0..1] => scale[0][1][0] = 4.0
    EXPECT_NEAR(yTensor.getHostValue(0, 2, 0), 4.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3, 1), 4.0f, tolerance);
    // Block (1,2): x[0][2..3][4..5] => scale[0][1][2] = 6.0
    EXPECT_NEAR(yTensor.getHostValue(0, 2, 4), 6.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3, 5), 6.0f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp32, NegativeXValues)
{
    // Dequantize is linear: negative x values should scale correctly
    Tensor<float> xTensor({1, 4});
    Tensor<float> scaleTensor({1, 2});
    Tensor<float> yTensor({1, 4});

    xTensor.setHostValue(-2.0f, 0, 0);
    xTensor.setHostValue(-1.0f, 0, 1);
    xTensor.setHostValue(0.0f, 0, 2);
    xTensor.setHostValue(-4.0f, 0, 3);

    scaleTensor.setHostValue(3.0f, 0, 0); // scales elements [0,1]
    scaleTensor.setHostValue(0.5f, 0, 1); // scales elements [2,3]

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-5f;
    EXPECT_NEAR(yTensor.getHostValue(0, 0), -2.0f * 3.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1), -1.0f * 3.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 2), 0.0f * 0.5f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3), -4.0f * 0.5f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp32, IsNegativeScaleFloat)
{
    // With is_negative_scale=true and float scale, Y = X * 2^(-scale)
    Tensor<float> xTensor({1, 4});
    Tensor<float> scaleTensor({1, 2});
    Tensor<float> yTensor({1, 4});

    xTensor.setHostValue(8.0f, 0, 0);
    xTensor.setHostValue(8.0f, 0, 1);
    xTensor.setHostValue(16.0f, 0, 2);
    xTensor.setHostValue(16.0f, 0, 3);

    scaleTensor.setHostValue(3.0f, 0, 0); // 2^(-3) = 0.125
    scaleTensor.setHostValue(1.0f, 0, 1); // 2^(-1) = 0.5

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, true);

    auto tolerance = 1e-5f;
    EXPECT_NEAR(yTensor.getHostValue(0, 0), 8.0f * 0.125f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1), 8.0f * 0.125f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 2), 16.0f * 0.5f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3), 16.0f * 0.5f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp32, NormalScaleMultiplication)
{
    // Normal (non-negative) scale: Y = X * scale
    Tensor<float> xTensor({1, 2});
    Tensor<float> scaleTensor({1, 1});
    Tensor<float> yTensor({1, 2});

    xTensor.setHostValue(5.0f, 0, 0);
    xTensor.setHostValue(10.0f, 0, 1);
    scaleTensor.setHostValue(0.1f, 0, 0);

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-5f;
    EXPECT_NEAR(yTensor.getHostValue(0, 0), 0.5f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1), 1.0f, tolerance);
}

// ============================================================================
// FP8 E8M0 scale: standalone scale semantics test
// ============================================================================

TEST(TestCpuFpReferenceBlockScaleDequantizeFp8, E8M0ScaleDequantize)
{
    // Test with fp8_e8m0 scale: scale value is 2^(biased_exp - 127)
    // fp8_e8m0 with bits=127 => 2^0 = 1.0
    // fp8_e8m0 with bits=128 => 2^1 = 2.0
    Tensor<float> xTensor({1, 4});
    Tensor<fp8_e8m0> scaleTensor({1, 2});
    Tensor<float> yTensor({1, 4});

    xTensor.setHostValue(3.0f, 0, 0);
    xTensor.setHostValue(3.0f, 0, 1);
    xTensor.setHostValue(5.0f, 0, 2);
    xTensor.setHostValue(5.0f, 0, 3);

    // scale[0] = 1.0 (bits=127), scale[1] = 2.0 (bits=128)
    scaleTensor.setHostValue(fp8_e8m0::from_bits(127), 0, 0);
    scaleTensor.setHostValue(fp8_e8m0::from_bits(128), 0, 1);

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-5f;
    EXPECT_NEAR(yTensor.getHostValue(0, 0), 3.0f * 1.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1), 3.0f * 1.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 2), 5.0f * 2.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3), 5.0f * 2.0f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp8, IsNegativeScaleE8M0)
{
    // isNegativeScale=true with fp8_e8m0: Y = X * 2^(-float(scale))
    // float(fp8_e8m0::from_bits(127)) = 2^0 = 1.0, so 2^(-1.0) = 0.5
    // float(fp8_e8m0::from_bits(128)) = 2^1 = 2.0, so 2^(-2.0) = 0.25
    Tensor<float> xTensor({1, 4});
    Tensor<fp8_e8m0> scaleTensor({1, 2});
    Tensor<float> yTensor({1, 4});

    xTensor.setHostValue(8.0f, 0, 0);
    xTensor.setHostValue(8.0f, 0, 1);
    xTensor.setHostValue(16.0f, 0, 2);
    xTensor.setHostValue(16.0f, 0, 3);

    scaleTensor.setHostValue(fp8_e8m0::from_bits(127), 0, 0); // float value = 1.0 => 2^(-1.0) = 0.5
    scaleTensor.setHostValue(
        fp8_e8m0::from_bits(128), 0, 1); // float value = 2.0 => 2^(-2.0) = 0.25

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, true);

    auto tolerance = 1e-5f;
    EXPECT_NEAR(yTensor.getHostValue(0, 0), 8.0f * 0.5f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1), 8.0f * 0.5f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 2), 16.0f * 0.25f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3), 16.0f * 0.25f, tolerance);
}

// ============================================================================
// FP8 MX dequantize typed tests: fp8_e4m3 and fp8_e5m2 with fp8_e8m0 scale
// ============================================================================

// Typed over XType (fp8_e4m3, fp8_e5m2) with float output
using MxFp8XTypes = ::testing::Types<fp8_e4m3, fp8_e5m2>;

template <class T>
class CpuFpReferenceMxFp8DequantizeFloatOut : public ::testing::Test
{
};
template <class T>
class CpuFpReferenceMxFp8DequantizeHalfOut : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceMxFp8DequantizeFloatOut, MxFp8XTypes, );
TYPED_TEST_SUITE(CpuFpReferenceMxFp8DequantizeHalfOut, MxFp8XTypes, );

TYPED_TEST(CpuFpReferenceMxFp8DequantizeFloatOut, WithE8M0Scale)
{
    // MX dequantize: fp8 input, fp8_e8m0 scale, float output
    // scale[0] = 1.0 (bits=127), scale[1] = 2.0 (bits=128)
    // x: {1, 1, 2, 2}, expected y: {1, 1, 4, 4}
    Tensor<TypeParam> xTensor({1, 4});
    Tensor<fp8_e8m0> scaleTensor({1, 2});
    Tensor<float> yTensor({1, 4});

    xTensor.setHostValue(TypeParam(1.0f), 0, 0);
    xTensor.setHostValue(TypeParam(1.0f), 0, 1);
    xTensor.setHostValue(TypeParam(2.0f), 0, 2);
    xTensor.setHostValue(TypeParam(2.0f), 0, 3);

    scaleTensor.setHostValue(fp8_e8m0::from_bits(127), 0, 0); // 1.0
    scaleTensor.setHostValue(fp8_e8m0::from_bits(128), 0, 1); // 2.0

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-2f;
    EXPECT_NEAR(yTensor.getHostValue(0, 0), 1.0f * 1.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1), 1.0f * 1.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 2), 2.0f * 2.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3), 2.0f * 2.0f, tolerance);
}

TYPED_TEST(CpuFpReferenceMxFp8DequantizeHalfOut, WithE8M0Scale)
{
    // MX dequantize: fp8 input, fp8_e8m0 scale, half output
    // scale[0] = 1.0 (bits=127), scale[1] = 2.0 (bits=128)
    // x: {1, 1, 2, 2}, expected y: {1, 1, 4, 4}
    Tensor<TypeParam> xTensor({1, 4});
    Tensor<fp8_e8m0> scaleTensor({1, 2});
    Tensor<half> yTensor({1, 4});

    xTensor.setHostValue(TypeParam(1.0f), 0, 0);
    xTensor.setHostValue(TypeParam(1.0f), 0, 1);
    xTensor.setHostValue(TypeParam(2.0f), 0, 2);
    xTensor.setHostValue(TypeParam(2.0f), 0, 3);

    scaleTensor.setHostValue(fp8_e8m0::from_bits(127), 0, 0); // 1.0
    scaleTensor.setHostValue(fp8_e8m0::from_bits(128), 0, 1); // 2.0

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-2f;
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 0)), 1.0f * 1.0f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 1)), 1.0f * 1.0f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 2)), 2.0f * 2.0f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 3)), 2.0f * 2.0f, tolerance);
}

// ============================================================================
// FP4 E2M1 MX dequantize tests
// ============================================================================

TEST(TestCpuFpReferenceBlockScaleDequantizeFp4, E2M1WithE8M0Scale_FloatOutput)
{
    // MX dequantize: fp4_e2m1 input, fp8_e8m0 scale, float output
    // fp4_e2m1 representable values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    // scale[0] = 1.0 (bits=127), scale[1] = 4.0 (bits=129)
    Tensor<fp4_e2m1> xTensor({1, 4});
    Tensor<fp8_e8m0> scaleTensor({1, 2});
    Tensor<float> yTensor({1, 4});

    xTensor.setHostValue(fp4_e2m1(1.0f), 0, 0);
    xTensor.setHostValue(fp4_e2m1(1.5f), 0, 1);
    xTensor.setHostValue(fp4_e2m1(2.0f), 0, 2);
    xTensor.setHostValue(fp4_e2m1(3.0f), 0, 3);

    scaleTensor.setHostValue(fp8_e8m0::from_bits(127), 0, 0); // 2^0 = 1.0
    scaleTensor.setHostValue(fp8_e8m0::from_bits(129), 0, 1); // 2^2 = 4.0

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-2f;
    EXPECT_NEAR(yTensor.getHostValue(0, 0), 1.0f * 1.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1), 1.5f * 1.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 2), 2.0f * 4.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3), 3.0f * 4.0f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp4, E2M1WithE8M0Scale_HalfOutput)
{
    Tensor<fp4_e2m1> xTensor({1, 4});
    Tensor<fp8_e8m0> scaleTensor({1, 2});
    Tensor<half> yTensor({1, 4});

    xTensor.setHostValue(fp4_e2m1(0.5f), 0, 0);
    xTensor.setHostValue(fp4_e2m1(1.0f), 0, 1);
    xTensor.setHostValue(fp4_e2m1(4.0f), 0, 2);
    xTensor.setHostValue(fp4_e2m1(6.0f), 0, 3);

    scaleTensor.setHostValue(fp8_e8m0::from_bits(128), 0, 0); // 2^1 = 2.0
    scaleTensor.setHostValue(fp8_e8m0::from_bits(127), 0, 1); // 2^0 = 1.0

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-2f;
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 0)), 0.5f * 2.0f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 1)), 1.0f * 2.0f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 2)), 4.0f * 1.0f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 3)), 6.0f * 1.0f, tolerance);
}

// ============================================================================
// FP6 E2M3 MX dequantize tests
// ============================================================================

// Typed over output type (float, half) for fp6_e2m3 input
using MxFp6E2M3OutputTypes = ::testing::Types<TypePair<fp6_e2m3, float>, TypePair<fp6_e2m3, half>>;

template <class T>
class CpuFpReferenceMxFp6E2M3Dequantize : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceMxFp6E2M3Dequantize, MxFp6E2M3OutputTypes, );

TYPED_TEST(CpuFpReferenceMxFp6E2M3Dequantize, WithE8M0Scale)
{
    using XType = typename TypeParam::First;
    using YType = typename TypeParam::Second;

    // fp6_e2m3 representable values include: 0, 0.125, ..., 1.0, 1.125, ..., 7.5
    // scale[0] = 2^1 = 2.0 (bits=128), scale[1] = 2^0 = 1.0 (bits=127)
    Tensor<XType> xTensor({1, 4});
    Tensor<fp8_e8m0> scaleTensor({1, 2});
    Tensor<YType> yTensor({1, 4});

    xTensor.setHostValue(XType(1.0f), 0, 0);
    xTensor.setHostValue(XType(2.0f), 0, 1);
    xTensor.setHostValue(XType(4.0f), 0, 2);
    xTensor.setHostValue(XType(7.5f), 0, 3);

    scaleTensor.setHostValue(fp8_e8m0::from_bits(128), 0, 0); // 2^1 = 2.0
    scaleTensor.setHostValue(fp8_e8m0::from_bits(127), 0, 1); // 2^0 = 1.0

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-2f;
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 0)), 1.0f * 2.0f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 1)), 2.0f * 2.0f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 2)), 4.0f * 1.0f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 3)), 7.5f * 1.0f, tolerance);
}

// ============================================================================
// FP6 E3M2 MX dequantize tests
// ============================================================================

// Typed over output type (float, half) for fp6_e3m2 input
using MxFp6E3M2OutputTypes = ::testing::Types<TypePair<fp6_e3m2, float>, TypePair<fp6_e3m2, half>>;

template <class T>
class CpuFpReferenceMxFp6E3M2Dequantize : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceMxFp6E3M2Dequantize, MxFp6E3M2OutputTypes, );

TYPED_TEST(CpuFpReferenceMxFp6E3M2Dequantize, WithE8M0Scale)
{
    using XType = typename TypeParam::First;
    using YType = typename TypeParam::Second;

    // fp6_e3m2 representable values include: 0, 0.25, 0.5, 0.75, 1.0, 1.25, ..., 28.0
    // scale[0] = 2^1 = 2.0 (bits=128), scale[1] = 2^0 = 1.0 (bits=127)
    Tensor<XType> xTensor({1, 4});
    Tensor<fp8_e8m0> scaleTensor({1, 2});
    Tensor<YType> yTensor({1, 4});

    xTensor.setHostValue(XType(1.0f), 0, 0);
    xTensor.setHostValue(XType(2.0f), 0, 1);
    xTensor.setHostValue(XType(4.0f), 0, 2);
    xTensor.setHostValue(XType(8.0f), 0, 3);

    scaleTensor.setHostValue(fp8_e8m0::from_bits(128), 0, 0); // 2^1 = 2.0
    scaleTensor.setHostValue(fp8_e8m0::from_bits(127), 0, 1); // 2^0 = 1.0

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-1f;
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 0)), 1.0f * 2.0f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 1)), 2.0f * 2.0f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 2)), 4.0f * 1.0f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 3)), 8.0f * 1.0f, tolerance);
}
