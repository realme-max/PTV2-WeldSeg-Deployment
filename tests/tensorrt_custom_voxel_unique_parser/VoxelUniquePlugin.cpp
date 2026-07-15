#include "VoxelUniquePlugin.h"

#include <atomic>
#include <cstdio>
#include <new>

namespace ptv2::trt_prototype
{
namespace
{
std::atomic<int32_t> gBuildCreationCount{0};
std::atomic<int32_t> gRuntimeCreationCount{0};
}

nvinfer1::IPluginCapability* VoxelUniquePlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept
{
    switch (type)
    {
    case nvinfer1::PluginCapabilityType::kCORE:
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    case nvinfer1::PluginCapabilityType::kBUILD:
        return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    case nvinfer1::PluginCapabilityType::kRUNTIME:
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    default:
        return nullptr;
    }
}

nvinfer1::IPluginV3* VoxelUniquePlugin::clone() noexcept
{
    return new (std::nothrow) VoxelUniquePlugin(*this);
}

nvinfer1::AsciiChar const* VoxelUniquePlugin::getPluginName() const noexcept
{
    return kPluginName;
}

nvinfer1::AsciiChar const* VoxelUniquePlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

nvinfer1::AsciiChar const* VoxelUniquePlugin::getPluginNamespace() const noexcept
{
    return kPluginNamespace;
}

int32_t VoxelUniquePlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    return inputs != nullptr && outputs != nullptr && nbInputs == 1 && nbOutputs == 3 ? 0 : 1;
}

int32_t VoxelUniquePlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes, int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbInputs != 1 || nbOutputs != 3
        || inputTypes[0] != nvinfer1::DataType::kINT64)
    {
        return 1;
    }
    outputTypes[0] = nvinfer1::DataType::kINT32;
    outputTypes[1] = nvinfer1::DataType::kINT64;
    outputTypes[2] = nvinfer1::DataType::kINT64;
    return 0;
}

int32_t VoxelUniquePlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
    nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs,
    nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    (void) shapeInputs;
    if (inputs == nullptr || outputs == nullptr || nbInputs != 1 || nbShapeInputs != 0
        || nbOutputs != 3 || inputs[0].nbDims != 1)
    {
        return 1;
    }

    outputs[0].nbDims = 0;
    auto const* optimum = exprBuilder.constant(3);
    auto const* upper = inputs[0].d[0];
    if (optimum == nullptr || upper == nullptr)
    {
        return 1;
    }
    auto const* runtimeM = exprBuilder.declareSizeTensor(0, *optimum, *upper);
    if (runtimeM == nullptr)
    {
        return 1;
    }
    outputs[1].nbDims = 1;
    outputs[1].d[0] = runtimeM;
    outputs[2].nbDims = 1;
    outputs[2].d[0] = inputs[0].d[0];
    return 0;
}

bool VoxelUniquePlugin::supportsFormatCombination(
    int32_t position, nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != 1 || nbOutputs != 3 || position < 0 || position >= 4)
    {
        return false;
    }
    nvinfer1::DataType expected = nvinfer1::DataType::kINT64;
    if (position == 1)
    {
        expected = nvinfer1::DataType::kINT32;
    }
    return inOut[position].desc.format == nvinfer1::TensorFormat::kLINEAR
        && inOut[position].desc.type == expected;
}

int32_t VoxelUniquePlugin::getNbOutputs() const noexcept
{
    return 3;
}

int32_t VoxelUniquePlugin::onShapeChange(
    nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    return inputs != nullptr && outputs != nullptr && nbInputs == 1 && nbOutputs == 3 ? 0 : 1;
}

int32_t VoxelUniquePlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    // Phase 2B never executes inference. Returning success keeps the runtime
    // capability structurally complete without pretending this is the real algorithm.
    (void) inputDesc;
    (void) outputDesc;
    (void) inputs;
    (void) outputs;
    (void) workspace;
    (void) stream;
    return 0;
}

nvinfer1::IPluginV3* VoxelUniquePlugin::attachToContext(
    nvinfer1::IPluginResourceContext* context) noexcept
{
    (void) context;
    return clone();
}

nvinfer1::PluginFieldCollection const* VoxelUniquePlugin::getFieldsToSerialize() noexcept
{
    static nvinfer1::PluginFieldCollection fields{0, nullptr};
    return &fields;
}

nvinfer1::IPluginV3* VoxelUniquePluginCreator::createPlugin(
    nvinfer1::AsciiChar const* name,
    nvinfer1::PluginFieldCollection const* fields,
    nvinfer1::TensorRTPhase phase) noexcept
{
    int32_t const fieldCount = fields == nullptr ? -1 : fields->nbFields;
    if (phase == nvinfer1::TensorRTPhase::kBUILD)
    {
        ++gBuildCreationCount;
        std::fprintf(stdout, "PLUGIN_CREATOR_CREATE phase=BUILD name=%s fields=%d\n",
            name == nullptr ? "<null>" : name, fieldCount);
    }
    else
    {
        ++gRuntimeCreationCount;
        std::fprintf(stdout, "PLUGIN_CREATOR_CREATE phase=RUNTIME name=%s fields=%d\n",
            name == nullptr ? "<null>" : name, fieldCount);
    }
    return new (std::nothrow) VoxelUniquePlugin();
}

nvinfer1::PluginFieldCollection const* VoxelUniquePluginCreator::getFieldNames() noexcept
{
    static nvinfer1::PluginFieldCollection fields{0, nullptr};
    return &fields;
}

nvinfer1::AsciiChar const* VoxelUniquePluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

nvinfer1::AsciiChar const* VoxelUniquePluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

nvinfer1::AsciiChar const* VoxelUniquePluginCreator::getPluginNamespace() const noexcept
{
    return kPluginNamespace;
}

int32_t getBuildCreationCount() noexcept
{
    return gBuildCreationCount.load();
}

int32_t getRuntimeCreationCount() noexcept
{
    return gRuntimeCreationCount.load();
}

} // namespace ptv2::trt_prototype

