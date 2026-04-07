// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>

#include "ExamplePluginHandle.hpp"
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

namespace example_plugin
{

// TEMPLATE ADAPTATION: Copy as-is and rename the class. This generic engine coordinator delegates
// to PlanBuilders and contains no operation-specific logic. The only change needed is the class
// name.

/// Generic engine coordinator. Copy this class as-is for your own plugin.
///
/// This engine contains no operation-specific logic. It manages a collection
/// of PlanBuilders and delegates all work (applicability checks, knob
/// reporting, workspace sizing, and plan creation) to them. Typically,
/// customized behavior can be added by writing your own PlanBuilder,
/// not by modifying this class.
class ExamplePluginEngine : public hipdnn_plugin_sdk::IEngine<ExamplePluginHandle,
                                                              ExamplePluginSettings,
                                                              ExamplePluginContext>
{
public:
    explicit ExamplePluginEngine(int64_t id);

    int64_t id() const override;

    bool isApplicable(ExamplePluginHandle& handle,
                      const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(ExamplePluginHandle& handle,
                    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    size_t getMaxWorkspaceSize(
        const ExamplePluginHandle& handle,
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) const override;

    void initializeExecutionContext(
        const ExamplePluginHandle& handle,
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        ExamplePluginContext& executionContext) const override;

    void addPlanBuilder(
        std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<ExamplePluginHandle,
                                                        ExamplePluginSettings,
                                                        ExamplePluginContext>> planBuilder);

private:
    int64_t _id;
    std::vector<std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<ExamplePluginHandle,
                                                                ExamplePluginSettings,
                                                                ExamplePluginContext>>>
        _planBuilders;
};

} // namespace example_plugin
