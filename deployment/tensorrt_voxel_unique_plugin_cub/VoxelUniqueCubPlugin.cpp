#include "VoxelUniqueCubPlugin.h"

#include <atomic>
#include <climits>
#include <new>

namespace ptv2::voxel_unique_cub
{
namespace
{
std::atomic<int32_t> gBuildCreationCount{0};
std::atomic<int32_t> gRuntimeCreationCount{0};
}

nvinfer1::IPluginCapability* VoxelUniqueCubPlugin::getCapabilityInterface(
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

nvinfer1::IPluginV3* VoxelUniqueCubPlugin::clone() noexcept
{
    return new (std::nothrow) VoxelUniqueCubPlugin(*this);
}

nvinfer1::AsciiChar const* VoxelUniqueCubPlugin::getPluginName() const noexcept { return kPluginName; }
nvinfer1::AsciiChar const* VoxelUniqueCubPlugin::getPluginVersion() const noexcept { return kPluginVersion; }
nvinfer1::AsciiChar const* VoxelUniqueCubPlugin::getPluginNamespace() const noexcept { return kPluginNamespace; }

int32_t VoxelUniqueCubPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != 1 || nbOutputs != 3
        || inputs[0].desc.dims.nbDims != 1 || inputs[0].desc.type != nvinfer1::DataType::kINT64)
    {
        return 1;
    }
    int64_t const maxN = inputs[0].max.d[0];
    return maxN >= 1 && maxN <= kMaxInputSize ? 0 : 1;
}

int32_t VoxelUniqueCubPlugin::getOutputDataTypes(
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

int32_t VoxelUniqueCubPlugin::getOutputShapes(
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
    auto const* runtimeM = exprBuilder.declareSizeTensor(0, *inputs[0].d[0], *inputs[0].d[0]);
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

bool VoxelUniqueCubPlugin::supportsFormatCombination(
    int32_t position, nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != 1 || nbOutputs != 3 || position < 0 || position >= 4)
    {
        return false;
    }
    nvinfer1::DataType expected = position == 1
        ? nvinfer1::DataType::kINT32 : nvinfer1::DataType::kINT64;
    return inOut[position].desc.format == nvinfer1::TensorFormat::kLINEAR
        && inOut[position].desc.type == expected;
}

int32_t VoxelUniqueCubPlugin::getNbOutputs() const noexcept { return 3; }

int32_t VoxelUniqueCubPlugin::onShapeChange(
    nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != 1 || nbOutputs != 3
        || inputs[0].dims.nbDims != 1)
    {
        return 1;
    }
    int64_t const n = inputs[0].dims.d[0];
    return n >= 1 && n <= kMaxInputSize ? 0 : 1;
}

nvinfer1::IPluginV3* VoxelUniqueCubPlugin::attachToContext(
    nvinfer1::IPluginResourceContext* context) noexcept
{
    (void) context;
    return clone();
}

nvinfer1::PluginFieldCollection const* VoxelUniqueCubPlugin::getFieldsToSerialize() noexcept
{
    static nvinfer1::PluginFieldCollection fields{0, nullptr};
    return &fields;
}

nvinfer1::IPluginV3* VoxelUniqueCubPluginCreator::createPlugin(
    nvinfer1::AsciiChar const* name,
    nvinfer1::PluginFieldCollection const* fields,
    nvinfer1::TensorRTPhase phase) noexcept
{
    (void) name;
    (void) fields;
    if (phase == nvinfer1::TensorRTPhase::kBUILD)
    {
        ++gBuildCreationCount;
    }
    else
    {
        ++gRuntimeCreationCount;
    }
    return new (std::nothrow) VoxelUniqueCubPlugin();
}

nvinfer1::PluginFieldCollection const* VoxelUniqueCubPluginCreator::getFieldNames() noexcept
{
    static nvinfer1::PluginFieldCollection fields{0, nullptr};
    return &fields;
}

nvinfer1::AsciiChar const* VoxelUniqueCubPluginCreator::getPluginName() const noexcept { return kPluginName; }
nvinfer1::AsciiChar const* VoxelUniqueCubPluginCreator::getPluginVersion() const noexcept { return kPluginVersion; }
nvinfer1::AsciiChar const* VoxelUniqueCubPluginCreator::getPluginNamespace() const noexcept { return kPluginNamespace; }
int32_t getBuildCreationCount() noexcept { return gBuildCreationCount.load(); }
int32_t getRuntimeCreationCount() noexcept { return gRuntimeCreationCount.load(); }

} // namespace ptv2::voxel_unique_cub
