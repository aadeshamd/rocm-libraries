// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleDequantize.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BlockScaleDequantizePlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::flatbuffer_utilities;

TEST(TestBlockScaleDequantizePlan, ExecutePlan)
{
    auto builder = createValidBlockScaleDequantizeGraph();
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graphWrapper.getNode(0);
    const auto& tensorMap = graphWrapper.getTensorMap();

    // Create two tensor bundles with same data for plan vs direct comparison
    const unsigned int seed = getGlobalTestSeed();
    GraphTensorBundle planBundle(tensorMap);
    GraphTensorBundle directBundle(tensorMap);

    // Fill x and scale with same random data
    planBundle.getTensor(1).fillTensorWithRandomValues(0.0f, 1.0f, seed);
    planBundle.getTensor(2).fillTensorWithRandomValues(0.1f, 2.0f, seed);
    directBundle.getTensor(1).fillTensorWithRandomValues(0.0f, 1.0f, seed);
    directBundle.getTensor(2).fillTensorWithRandomValues(0.1f, 2.0f, seed);

    const auto* nodeAttributes = node.attributes_as_BlockScaleDequantizeAttributes();
    ASSERT_NE(nodeAttributes, nullptr);

    std::vector<int32_t> blockSize;
    if(nodeAttributes->block_size() != nullptr)
    {
        const auto* bs = nodeAttributes->block_size();
        blockSize.assign(bs->begin(), bs->end());
    }

    // Execute via plan
    BlockScaleDequantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                      blockSize,
                                      nodeAttributes->is_negative_scale());

    // Direct execution for reference
    auto directXTensor
        = createShallowTensor<float>(params.xTensor, directBundle.getTensor(1).rawHostData());
    auto directScaleTensor
        = createShallowTensor<float>(params.scaleTensor, directBundle.getTensor(2).rawHostData());
    auto directYTensor
        = createShallowTensor<float>(params.yTensor, directBundle.getTensor(3).rawHostData());

    CpuFpReferenceBlockScaleDequantize::dequantize(
        *directXTensor, *directScaleTensor, *directYTensor, blockSize, false);

    // Plan execution
    auto variantPack = planBundle.toHostVariantPack();
    BlockScaleDequantizePlan<float, float, float, float> plan(std::move(params));
    plan.execute(variantPack);

    const float tolerance = 1e-5f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directBundle.getTensor(3), planBundle.getTensor(3)));
}

TEST(TestBlockScaleDequantizePlanBuilder, PlanConstruction)
{
    auto builder = createValidBlockScaleDequantizeGraph();
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const BlockScaleDequantizePlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<BlockScaleDequantizePlan<float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestBlockScaleDequantizePlanBuilder, IsApplicable)
{
    auto builder = createValidBlockScaleDequantizeGraph();
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const BlockScaleDequantizePlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    const BlockScaleDequantizePlanBuilder<DataType::HALF,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        badTypesPlanBuilder;
    EXPECT_FALSE(
        badTypesPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    // MX combinations: FP8 input with E8M0 scale
    auto mxBuilder = createValidBlockScaleDequantizeMxGraph(
        DataType::FP8_E4M3, DataType::FP8_E8M0, DataType::FLOAT);
    const GraphWrapper mxGraphWrapper(mxBuilder.GetBufferPointer(), mxBuilder.GetSize());

    const BlockScaleDequantizePlanBuilder<DataType::FP8_E4M3,
                                          DataType::FP8_E8M0,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        e4m3FloatBuilder;
    EXPECT_TRUE(
        e4m3FloatBuilder.isApplicable(mxGraphWrapper.getNode(0), mxGraphWrapper.getTensorMap()));

    const BlockScaleDequantizePlanBuilder<DataType::FP8_E5M2,
                                          DataType::FP8_E8M0,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        e5m2FloatBuilder;
    EXPECT_FALSE(
        e5m2FloatBuilder.isApplicable(mxGraphWrapper.getNode(0), mxGraphWrapper.getTensorMap()));
}

TEST(TestBlockScaleDequantizePlan, ExecutePlan_E4M3_E8M0_FloatOutput)
{
    using namespace hipdnn_data_sdk::types;

    // Build a 1x4 fp8_e4m3 X, 1x2 fp8_e8m0 scale, 1x4 float Y graph (blockSize=2)
    auto builder = createValidBlockScaleDequantizeMxGraph(
        DataType::FP8_E4M3, DataType::FP8_E8M0, DataType::FLOAT);
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graphWrapper.getNode(0);
    const auto& tensorMap = graphWrapper.getTensorMap();
    const auto* nodeAttributes = node.attributes_as_BlockScaleDequantizeAttributes();
    ASSERT_NE(nodeAttributes, nullptr);

    std::vector<int32_t> blockSize;
    if(nodeAttributes->block_size() != nullptr)
    {
        const auto* bs = nodeAttributes->block_size();
        blockSize.assign(bs->begin(), bs->end());
    }

    GraphTensorBundle planBundle(tensorMap);
    GraphTensorBundle directBundle(tensorMap);

    // Set x values (fp8_e4m3) via raw host data
    auto* planXData = static_cast<fp8_e4m3*>(planBundle.getTensor(1).rawHostData());
    planXData[0] = fp8_e4m3(1.0f);
    planXData[1] = fp8_e4m3(1.0f);
    planXData[2] = fp8_e4m3(2.0f);
    planXData[3] = fp8_e4m3(2.0f);

    auto* directXData = static_cast<fp8_e4m3*>(directBundle.getTensor(1).rawHostData());
    directXData[0] = fp8_e4m3(1.0f);
    directXData[1] = fp8_e4m3(1.0f);
    directXData[2] = fp8_e4m3(2.0f);
    directXData[3] = fp8_e4m3(2.0f);

    // Set scale values (fp8_e8m0): bits=127 => 1.0, bits=128 => 2.0
    auto* planScaleData = static_cast<fp8_e8m0*>(planBundle.getTensor(2).rawHostData());
    planScaleData[0] = fp8_e8m0::from_bits(127);
    planScaleData[1] = fp8_e8m0::from_bits(128);

    auto* directScaleData = static_cast<fp8_e8m0*>(directBundle.getTensor(2).rawHostData());
    directScaleData[0] = fp8_e8m0::from_bits(127);
    directScaleData[1] = fp8_e8m0::from_bits(128);

    BlockScaleDequantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                      blockSize,
                                      nodeAttributes->is_negative_scale());

    // Direct reference execution
    auto directXTensor
        = createShallowTensor<fp8_e4m3>(params.xTensor, directBundle.getTensor(1).rawHostData());
    auto directScaleTensor = createShallowTensor<fp8_e8m0>(params.scaleTensor,
                                                           directBundle.getTensor(2).rawHostData());
    auto directYTensor
        = createShallowTensor<float>(params.yTensor, directBundle.getTensor(3).rawHostData());

    CpuFpReferenceBlockScaleDequantize::dequantize(
        *directXTensor, *directScaleTensor, *directYTensor, blockSize, false);

    // Plan execution
    auto variantPack = planBundle.toHostVariantPack();
    BlockScaleDequantizePlan<fp8_e4m3, fp8_e8m0, float, float> plan(std::move(params));
    plan.execute(variantPack);

    const float tolerance = 1e-2f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directBundle.getTensor(3), planBundle.getTensor(3)));
}

TEST(TestBlockScaleDequantizePlan, ExecutePlan_E5M2_E8M0_FloatOutput)
{
    using namespace hipdnn_data_sdk::types;

    auto builder = createValidBlockScaleDequantizeMxGraph(
        DataType::FP8_E5M2, DataType::FP8_E8M0, DataType::FLOAT);
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graphWrapper.getNode(0);
    const auto& tensorMap = graphWrapper.getTensorMap();
    const auto* nodeAttributes = node.attributes_as_BlockScaleDequantizeAttributes();
    ASSERT_NE(nodeAttributes, nullptr);

    std::vector<int32_t> blockSize;
    if(nodeAttributes->block_size() != nullptr)
    {
        const auto* bs = nodeAttributes->block_size();
        blockSize.assign(bs->begin(), bs->end());
    }

    GraphTensorBundle planBundle(tensorMap);
    GraphTensorBundle directBundle(tensorMap);

    auto* planXData = static_cast<fp8_e5m2*>(planBundle.getTensor(1).rawHostData());
    planXData[0] = fp8_e5m2(1.0f);
    planXData[1] = fp8_e5m2(1.0f);
    planXData[2] = fp8_e5m2(2.0f);
    planXData[3] = fp8_e5m2(2.0f);

    auto* directXData = static_cast<fp8_e5m2*>(directBundle.getTensor(1).rawHostData());
    directXData[0] = fp8_e5m2(1.0f);
    directXData[1] = fp8_e5m2(1.0f);
    directXData[2] = fp8_e5m2(2.0f);
    directXData[3] = fp8_e5m2(2.0f);

    auto* planScaleData = static_cast<fp8_e8m0*>(planBundle.getTensor(2).rawHostData());
    planScaleData[0] = fp8_e8m0::from_bits(127);
    planScaleData[1] = fp8_e8m0::from_bits(128);

    auto* directScaleData = static_cast<fp8_e8m0*>(directBundle.getTensor(2).rawHostData());
    directScaleData[0] = fp8_e8m0::from_bits(127);
    directScaleData[1] = fp8_e8m0::from_bits(128);

    BlockScaleDequantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                      blockSize,
                                      nodeAttributes->is_negative_scale());

    auto directXTensor
        = createShallowTensor<fp8_e5m2>(params.xTensor, directBundle.getTensor(1).rawHostData());
    auto directScaleTensor = createShallowTensor<fp8_e8m0>(params.scaleTensor,
                                                           directBundle.getTensor(2).rawHostData());
    auto directYTensor
        = createShallowTensor<float>(params.yTensor, directBundle.getTensor(3).rawHostData());

    CpuFpReferenceBlockScaleDequantize::dequantize(
        *directXTensor, *directScaleTensor, *directYTensor, blockSize, false);

    auto variantPack = planBundle.toHostVariantPack();
    BlockScaleDequantizePlan<fp8_e5m2, fp8_e8m0, float, float> plan(std::move(params));
    plan.execute(variantPack);

    const float tolerance = 1e-2f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directBundle.getTensor(3), planBundle.getTensor(3)));
}

TEST(TestBlockScaleDequantizePlan, ExecutePlan_E4M3_E8M0_HalfOutput)
{
    using namespace hipdnn_data_sdk::types;

    auto builder = createValidBlockScaleDequantizeMxGraph(
        DataType::FP8_E4M3, DataType::FP8_E8M0, DataType::HALF);
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graphWrapper.getNode(0);
    const auto& tensorMap = graphWrapper.getTensorMap();
    const auto* nodeAttributes = node.attributes_as_BlockScaleDequantizeAttributes();
    ASSERT_NE(nodeAttributes, nullptr);

    std::vector<int32_t> blockSize;
    if(nodeAttributes->block_size() != nullptr)
    {
        const auto* bs = nodeAttributes->block_size();
        blockSize.assign(bs->begin(), bs->end());
    }

    GraphTensorBundle planBundle(tensorMap);
    GraphTensorBundle directBundle(tensorMap);

    auto* planXData = static_cast<fp8_e4m3*>(planBundle.getTensor(1).rawHostData());
    planXData[0] = fp8_e4m3(1.0f);
    planXData[1] = fp8_e4m3(1.0f);
    planXData[2] = fp8_e4m3(2.0f);
    planXData[3] = fp8_e4m3(2.0f);

    auto* directXData = static_cast<fp8_e4m3*>(directBundle.getTensor(1).rawHostData());
    directXData[0] = fp8_e4m3(1.0f);
    directXData[1] = fp8_e4m3(1.0f);
    directXData[2] = fp8_e4m3(2.0f);
    directXData[3] = fp8_e4m3(2.0f);

    auto* planScaleData = static_cast<fp8_e8m0*>(planBundle.getTensor(2).rawHostData());
    planScaleData[0] = fp8_e8m0::from_bits(127);
    planScaleData[1] = fp8_e8m0::from_bits(128);

    auto* directScaleData = static_cast<fp8_e8m0*>(directBundle.getTensor(2).rawHostData());
    directScaleData[0] = fp8_e8m0::from_bits(127);
    directScaleData[1] = fp8_e8m0::from_bits(128);

    BlockScaleDequantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                      blockSize,
                                      nodeAttributes->is_negative_scale());

    auto directXTensor
        = createShallowTensor<fp8_e4m3>(params.xTensor, directBundle.getTensor(1).rawHostData());
    auto directScaleTensor = createShallowTensor<fp8_e8m0>(params.scaleTensor,
                                                           directBundle.getTensor(2).rawHostData());
    auto directYTensor
        = createShallowTensor<half>(params.yTensor, directBundle.getTensor(3).rawHostData());

    CpuFpReferenceBlockScaleDequantize::dequantize(
        *directXTensor, *directScaleTensor, *directYTensor, blockSize, false);

    auto variantPack = planBundle.toHostVariantPack();
    BlockScaleDequantizePlan<fp8_e4m3, fp8_e8m0, half, float> plan(std::move(params));
    plan.execute(variantPack);

    const float tolerance = 1e-2f;
    const CpuFpReferenceValidation<half> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directBundle.getTensor(3), planBundle.getTensor(3)));
}

TEST(TestBlockScaleDequantizePlan, ExecutePlan_E5M2_E8M0_HalfOutput)
{
    using namespace hipdnn_data_sdk::types;

    auto builder = createValidBlockScaleDequantizeMxGraph(
        DataType::FP8_E5M2, DataType::FP8_E8M0, DataType::HALF);
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graphWrapper.getNode(0);
    const auto& tensorMap = graphWrapper.getTensorMap();
    const auto* nodeAttributes = node.attributes_as_BlockScaleDequantizeAttributes();
    ASSERT_NE(nodeAttributes, nullptr);

    std::vector<int32_t> blockSize;
    if(nodeAttributes->block_size() != nullptr)
    {
        const auto* bs = nodeAttributes->block_size();
        blockSize.assign(bs->begin(), bs->end());
    }

    GraphTensorBundle planBundle(tensorMap);
    GraphTensorBundle directBundle(tensorMap);

    auto* planXData = static_cast<fp8_e5m2*>(planBundle.getTensor(1).rawHostData());
    planXData[0] = fp8_e5m2(1.0f);
    planXData[1] = fp8_e5m2(1.0f);
    planXData[2] = fp8_e5m2(2.0f);
    planXData[3] = fp8_e5m2(2.0f);

    auto* directXData = static_cast<fp8_e5m2*>(directBundle.getTensor(1).rawHostData());
    directXData[0] = fp8_e5m2(1.0f);
    directXData[1] = fp8_e5m2(1.0f);
    directXData[2] = fp8_e5m2(2.0f);
    directXData[3] = fp8_e5m2(2.0f);

    auto* planScaleData = static_cast<fp8_e8m0*>(planBundle.getTensor(2).rawHostData());
    planScaleData[0] = fp8_e8m0::from_bits(127);
    planScaleData[1] = fp8_e8m0::from_bits(128);

    auto* directScaleData = static_cast<fp8_e8m0*>(directBundle.getTensor(2).rawHostData());
    directScaleData[0] = fp8_e8m0::from_bits(127);
    directScaleData[1] = fp8_e8m0::from_bits(128);

    BlockScaleDequantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                      blockSize,
                                      nodeAttributes->is_negative_scale());

    auto directXTensor
        = createShallowTensor<fp8_e5m2>(params.xTensor, directBundle.getTensor(1).rawHostData());
    auto directScaleTensor = createShallowTensor<fp8_e8m0>(params.scaleTensor,
                                                           directBundle.getTensor(2).rawHostData());
    auto directYTensor
        = createShallowTensor<half>(params.yTensor, directBundle.getTensor(3).rawHostData());

    CpuFpReferenceBlockScaleDequantize::dequantize(
        *directXTensor, *directScaleTensor, *directYTensor, blockSize, false);

    auto variantPack = planBundle.toHostVariantPack();
    BlockScaleDequantizePlan<fp8_e5m2, fp8_e8m0, half, float> plan(std::move(params));
    plan.execute(variantPack);

    const float tolerance = 1e-2f;
    const CpuFpReferenceValidation<half> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directBundle.getTensor(3), planBundle.getTensor(3)));
}

// ============================================================================
// FP4 E2M1 plan tests
// ============================================================================

TEST(TestBlockScaleDequantizePlan, ExecutePlan_FP4E2M1_E8M0_FloatOutput)
{
    using namespace hipdnn_data_sdk::types;

    auto builder = createValidBlockScaleDequantizeMxGraph(
        DataType::FP4_E2M1, DataType::FP8_E8M0, DataType::FLOAT);
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graphWrapper.getNode(0);
    const auto& tensorMap = graphWrapper.getTensorMap();
    const auto* nodeAttributes = node.attributes_as_BlockScaleDequantizeAttributes();
    ASSERT_NE(nodeAttributes, nullptr);

    std::vector<int32_t> blockSize;
    if(nodeAttributes->block_size() != nullptr)
    {
        const auto* bs = nodeAttributes->block_size();
        blockSize.assign(bs->begin(), bs->end());
    }

    GraphTensorBundle planBundle(tensorMap);
    GraphTensorBundle directBundle(tensorMap);

    auto* planXData = static_cast<fp4_e2m1*>(planBundle.getTensor(1).rawHostData());
    planXData[0] = fp4_e2m1(1.0f);
    planXData[1] = fp4_e2m1(1.5f);
    planXData[2] = fp4_e2m1(2.0f);
    planXData[3] = fp4_e2m1(3.0f);

    auto* directXData = static_cast<fp4_e2m1*>(directBundle.getTensor(1).rawHostData());
    directXData[0] = fp4_e2m1(1.0f);
    directXData[1] = fp4_e2m1(1.5f);
    directXData[2] = fp4_e2m1(2.0f);
    directXData[3] = fp4_e2m1(3.0f);

    auto* planScaleData = static_cast<fp8_e8m0*>(planBundle.getTensor(2).rawHostData());
    planScaleData[0] = fp8_e8m0::from_bits(127);
    planScaleData[1] = fp8_e8m0::from_bits(128);

    auto* directScaleData = static_cast<fp8_e8m0*>(directBundle.getTensor(2).rawHostData());
    directScaleData[0] = fp8_e8m0::from_bits(127);
    directScaleData[1] = fp8_e8m0::from_bits(128);

    BlockScaleDequantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                      blockSize,
                                      nodeAttributes->is_negative_scale());

    auto directXTensor
        = createShallowTensor<fp4_e2m1>(params.xTensor, directBundle.getTensor(1).rawHostData());
    auto directScaleTensor = createShallowTensor<fp8_e8m0>(params.scaleTensor,
                                                           directBundle.getTensor(2).rawHostData());
    auto directYTensor
        = createShallowTensor<float>(params.yTensor, directBundle.getTensor(3).rawHostData());

    CpuFpReferenceBlockScaleDequantize::dequantize(
        *directXTensor, *directScaleTensor, *directYTensor, blockSize, false);

    auto variantPack = planBundle.toHostVariantPack();
    BlockScaleDequantizePlan<fp4_e2m1, fp8_e8m0, float, float> plan(std::move(params));
    plan.execute(variantPack);

    const float tolerance = 1e-2f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directBundle.getTensor(3), planBundle.getTensor(3)));
}

TEST(TestBlockScaleDequantizePlanBuilder, IsApplicable_FP4E2M1)
{
    auto mxBuilder = createValidBlockScaleDequantizeMxGraph(
        DataType::FP4_E2M1, DataType::FP8_E8M0, DataType::FLOAT);
    const GraphWrapper mxGraphWrapper(mxBuilder.GetBufferPointer(), mxBuilder.GetSize());

    const BlockScaleDequantizePlanBuilder<DataType::FP4_E2M1,
                                          DataType::FP8_E8M0,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        fp4FloatBuilder;
    EXPECT_TRUE(
        fp4FloatBuilder.isApplicable(mxGraphWrapper.getNode(0), mxGraphWrapper.getTensorMap()));

    const BlockScaleDequantizePlanBuilder<DataType::FP6_E2M3,
                                          DataType::FP8_E8M0,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        wrongTypeBuilder;
    EXPECT_FALSE(
        wrongTypeBuilder.isApplicable(mxGraphWrapper.getNode(0), mxGraphWrapper.getTensorMap()));
}

// ============================================================================
// FP6 E2M3 plan tests
// ============================================================================

TEST(TestBlockScaleDequantizePlan, ExecutePlan_FP6E2M3_E8M0_FloatOutput)
{
    using namespace hipdnn_data_sdk::types;

    auto builder = createValidBlockScaleDequantizeMxGraph(
        DataType::FP6_E2M3, DataType::FP8_E8M0, DataType::FLOAT);
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graphWrapper.getNode(0);
    const auto& tensorMap = graphWrapper.getTensorMap();
    const auto* nodeAttributes = node.attributes_as_BlockScaleDequantizeAttributes();
    ASSERT_NE(nodeAttributes, nullptr);

    std::vector<int32_t> blockSize;
    if(nodeAttributes->block_size() != nullptr)
    {
        const auto* bs = nodeAttributes->block_size();
        blockSize.assign(bs->begin(), bs->end());
    }

    GraphTensorBundle planBundle(tensorMap);
    GraphTensorBundle directBundle(tensorMap);

    auto* planXData = static_cast<fp6_e2m3*>(planBundle.getTensor(1).rawHostData());
    planXData[0] = fp6_e2m3(1.0f);
    planXData[1] = fp6_e2m3(1.5f);
    planXData[2] = fp6_e2m3(2.0f);
    planXData[3] = fp6_e2m3(3.0f);

    auto* directXData = static_cast<fp6_e2m3*>(directBundle.getTensor(1).rawHostData());
    directXData[0] = fp6_e2m3(1.0f);
    directXData[1] = fp6_e2m3(1.5f);
    directXData[2] = fp6_e2m3(2.0f);
    directXData[3] = fp6_e2m3(3.0f);

    auto* planScaleData = static_cast<fp8_e8m0*>(planBundle.getTensor(2).rawHostData());
    planScaleData[0] = fp8_e8m0::from_bits(127);
    planScaleData[1] = fp8_e8m0::from_bits(128);

    auto* directScaleData = static_cast<fp8_e8m0*>(directBundle.getTensor(2).rawHostData());
    directScaleData[0] = fp8_e8m0::from_bits(127);
    directScaleData[1] = fp8_e8m0::from_bits(128);

    BlockScaleDequantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                      blockSize,
                                      nodeAttributes->is_negative_scale());

    auto directXTensor
        = createShallowTensor<fp6_e2m3>(params.xTensor, directBundle.getTensor(1).rawHostData());
    auto directScaleTensor = createShallowTensor<fp8_e8m0>(params.scaleTensor,
                                                           directBundle.getTensor(2).rawHostData());
    auto directYTensor
        = createShallowTensor<float>(params.yTensor, directBundle.getTensor(3).rawHostData());

    CpuFpReferenceBlockScaleDequantize::dequantize(
        *directXTensor, *directScaleTensor, *directYTensor, blockSize, false);

    auto variantPack = planBundle.toHostVariantPack();
    BlockScaleDequantizePlan<fp6_e2m3, fp8_e8m0, float, float> plan(std::move(params));
    plan.execute(variantPack);

    const float tolerance = 1e-2f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directBundle.getTensor(3), planBundle.getTensor(3)));
}

TEST(TestBlockScaleDequantizePlanBuilder, IsApplicable_FP6E2M3)
{
    auto mxBuilder = createValidBlockScaleDequantizeMxGraph(
        DataType::FP6_E2M3, DataType::FP8_E8M0, DataType::FLOAT);
    const GraphWrapper mxGraphWrapper(mxBuilder.GetBufferPointer(), mxBuilder.GetSize());

    const BlockScaleDequantizePlanBuilder<DataType::FP6_E2M3,
                                          DataType::FP8_E8M0,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        fp6e2m3Builder;
    EXPECT_TRUE(
        fp6e2m3Builder.isApplicable(mxGraphWrapper.getNode(0), mxGraphWrapper.getTensorMap()));
}

// ============================================================================
// FP6 E3M2 plan tests
// ============================================================================

TEST(TestBlockScaleDequantizePlan, ExecutePlan_FP6E3M2_E8M0_FloatOutput)
{
    using namespace hipdnn_data_sdk::types;

    auto builder = createValidBlockScaleDequantizeMxGraph(
        DataType::FP6_E3M2, DataType::FP8_E8M0, DataType::FLOAT);
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graphWrapper.getNode(0);
    const auto& tensorMap = graphWrapper.getTensorMap();
    const auto* nodeAttributes = node.attributes_as_BlockScaleDequantizeAttributes();
    ASSERT_NE(nodeAttributes, nullptr);

    std::vector<int32_t> blockSize;
    if(nodeAttributes->block_size() != nullptr)
    {
        const auto* bs = nodeAttributes->block_size();
        blockSize.assign(bs->begin(), bs->end());
    }

    GraphTensorBundle planBundle(tensorMap);
    GraphTensorBundle directBundle(tensorMap);

    auto* planXData = static_cast<fp6_e3m2*>(planBundle.getTensor(1).rawHostData());
    planXData[0] = fp6_e3m2(1.0f);
    planXData[1] = fp6_e3m2(1.5f);
    planXData[2] = fp6_e3m2(2.0f);
    planXData[3] = fp6_e3m2(4.0f);

    auto* directXData = static_cast<fp6_e3m2*>(directBundle.getTensor(1).rawHostData());
    directXData[0] = fp6_e3m2(1.0f);
    directXData[1] = fp6_e3m2(1.5f);
    directXData[2] = fp6_e3m2(2.0f);
    directXData[3] = fp6_e3m2(4.0f);

    auto* planScaleData = static_cast<fp8_e8m0*>(planBundle.getTensor(2).rawHostData());
    planScaleData[0] = fp8_e8m0::from_bits(127);
    planScaleData[1] = fp8_e8m0::from_bits(128);

    auto* directScaleData = static_cast<fp8_e8m0*>(directBundle.getTensor(2).rawHostData());
    directScaleData[0] = fp8_e8m0::from_bits(127);
    directScaleData[1] = fp8_e8m0::from_bits(128);

    BlockScaleDequantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                      blockSize,
                                      nodeAttributes->is_negative_scale());

    auto directXTensor
        = createShallowTensor<fp6_e3m2>(params.xTensor, directBundle.getTensor(1).rawHostData());
    auto directScaleTensor = createShallowTensor<fp8_e8m0>(params.scaleTensor,
                                                           directBundle.getTensor(2).rawHostData());
    auto directYTensor
        = createShallowTensor<float>(params.yTensor, directBundle.getTensor(3).rawHostData());

    CpuFpReferenceBlockScaleDequantize::dequantize(
        *directXTensor, *directScaleTensor, *directYTensor, blockSize, false);

    auto variantPack = planBundle.toHostVariantPack();
    BlockScaleDequantizePlan<fp6_e3m2, fp8_e8m0, float, float> plan(std::move(params));
    plan.execute(variantPack);

    const float tolerance = 1e-1f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directBundle.getTensor(3), planBundle.getTensor(3)));
}

TEST(TestBlockScaleDequantizePlanBuilder, IsApplicable_FP6E3M2)
{
    auto mxBuilder = createValidBlockScaleDequantizeMxGraph(
        DataType::FP6_E3M2, DataType::FP8_E8M0, DataType::FLOAT);
    const GraphWrapper mxGraphWrapper(mxBuilder.GetBufferPointer(), mxBuilder.GetSize());

    const BlockScaleDequantizePlanBuilder<DataType::FP6_E3M2,
                                          DataType::FP8_E8M0,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        fp6e3m2Builder;
    EXPECT_TRUE(
        fp6e3m2Builder.isApplicable(mxGraphWrapper.getNode(0), mxGraphWrapper.getTensorMap()));
}
