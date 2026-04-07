// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ExamplePluginContainer.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "engines/ExamplePluginEngine.hpp"
#include "engines/plans/ConvFwdPlanBuilder.hpp"
#include "engines/plans/ReluPlanBuilder.hpp"
#include "hip/HipKernelCompiler.hpp"

namespace example_plugin
{

// TEMPLATE ADAPTATION: Register and create engines. To adapt:
// (1) Update the engine names and IDs in the HIPDNN_REGISTER_ENGINE calls (one call for each
//     engine provided by this plugin).
// (2) Update s_engineDefinitions to create your PlanBuilders instead of
//     ReluPlanBuilder/ConvFwdPlanBuilder.
//
// The s_engineDefinitions vector is one approach to reduce coupling between creating
// engines (in the ExamplePluginContainer constructor) and returning the list of engine
// IDs (in (in ExamplePluginContainer::copyEngineIds()). Alternate approaches can be
// used if this approach is not suitable for your plugin.

// The HIPDNN_REGISTER_ENGINE() macro creates _NAME and _ID constants for each engine.
// E.g. HIPDNN_REGISTER_ENGINE(EXAMPLE_PLUGIN_RELU_ENGINE) will create
// EXAMPLE_PLUGIN_RELU_ENGINE_NAME with the value "EXAMPLE_PLUGIN_RELU_ENGINE" and
// EXAMPLE_PLUGIN_RELU_ENGINE_ID with the hash-derived integer ID for the engine.
HIPDNN_REGISTER_ENGINE(EXAMPLE_PLUGIN_RELU_ENGINE)
HIPDNN_REGISTER_ENGINE(EXAMPLE_PLUGIN_CONV_FWD_ENGINE)

const std::vector<ExamplePluginContainer::EngineDefinition>&
    ExamplePluginContainer::getEngineDefinitions()
{
    static const std::vector<EngineDefinition> s_engineDefinitions = {
        {EXAMPLE_PLUGIN_RELU_ENGINE_ID,
         [](const IKernelCompiler& compiler) -> ExamplePluginEnginePtr {
             auto engine = std::make_unique<ExamplePluginEngine>(EXAMPLE_PLUGIN_RELU_ENGINE_ID);
             engine->addPlanBuilder(std::make_unique<ReluPlanBuilder>(compiler));
             return engine;
         }},
        {EXAMPLE_PLUGIN_CONV_FWD_ENGINE_ID,
         [](const IKernelCompiler& compiler) -> ExamplePluginEnginePtr {
             auto engine = std::make_unique<ExamplePluginEngine>(EXAMPLE_PLUGIN_CONV_FWD_ENGINE_ID);
             engine->addPlanBuilder(std::make_unique<ConvFwdPlanBuilder>(compiler));
             return engine;
         }},
    };

    return s_engineDefinitions;
}

uint32_t ExamplePluginContainer::copyEngineIds(int64_t* engineIds,
                                               uint32_t maxEngines,
                                               uint32_t& numEngines)
{
    const auto& engineDefinitions = getEngineDefinitions();
    auto totalEngines = static_cast<uint32_t>(engineDefinitions.size());

    if(maxEngines == 0)
    {
        numEngines = totalEngines;
        return totalEngines;
    }

    auto enginesToCopy = std::min(maxEngines, totalEngines);
    for(uint32_t i = 0; i < enginesToCopy; ++i)
    {
        engineIds[i] = engineDefinitions[i].id;
    }

    numEngines = enginesToCopy;

    return totalEngines;
}

ExamplePluginContainer::ExamplePluginContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Creating ExamplePluginContainer");

    _kernelCompiler = std::make_unique<HipKernelCompiler>();

    _engineManager = std::make_unique<hipdnn_plugin_sdk::EngineManager<ExamplePluginHandle,
                                                                       ExamplePluginSettings,
                                                                       ExamplePluginContext>>();

    for(const auto& engineDefinition : getEngineDefinitions())
    {
        _engineManager->addEngine(engineDefinition.createEngine(*_kernelCompiler));
    }
}

ExamplePluginContainer::~ExamplePluginContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Destroying ExamplePluginContainer");
}

hipdnn_plugin_sdk::EngineManager<ExamplePluginHandle, ExamplePluginSettings, ExamplePluginContext>&
    ExamplePluginContainer::getEngineManager()
{
    return *_engineManager;
}

} // namespace example_plugin
