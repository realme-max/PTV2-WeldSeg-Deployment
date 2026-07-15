#include "VoxelUniqueCorrectnessPlugin.h"

#include <atomic>
#include <climits>
#include <cstdio>
#include <new>

namespace ptv2::voxel_unique_correctness
{
namespace
{
std::atomic<int32_t> gBuildCreationCount{0};
std::atomic<int32_t> gRuntimeCreationCount{0};

// Intentionally unoptimized Phase 2C.1 implementation. One CUDA thread
// performs serial deduplication, insertion sort, and inverse lookup.
__global__ void voxelUniqueSerialKernel(
    int64_t const* keys, int32_t n, int32_t* count,
    int64_t* uniqueValues, int64_t* inverseIndices)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    int32_t uniqueCount = 0;
    for (int32_t i = 0; i < n; ++i)
    {
        bool found = false;
        for (int32_t j = 0; j < uniqueCount; ++j)
        {
            if (uniqueValues[j] == keys[i])
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            uniqueValues[uniqueCount++] = keys[i];
        }
    }

    for (int32_t i = 1; i < uniqueCount; ++i)
    {
        int64_t const value = uniqueValues[i];
        int32_t j = i - 1;
        while (j >= 0 && uniqueValues[j] > value)
        {
            uniqueValues[j + 1] = uniqueValues[j];
            --j;
        }
        uniqueValues[j + 1] = value;
    }

    for (int32_t i = 0; i < n; ++i)
    {
        int64_t inverse = -1;
        for (int32_t j = 0; j < uniqueCount; ++j)
        {
            if (uniqueValues[j] == keys[i])
            {
                inverse = j;
                break;
            }
        }
        inverseIndices[i] = inverse;
    }
    *count = uniqueCount;
}
} // namespace

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

nvinfer1::AsciiChar const* VoxelUniquePlugin::getPluginName() const noexcept { return kPluginName; }
nvinfer1::AsciiChar const* VoxelUniquePlugin::getPluginVersion() const noexcept { return kPluginVersion; }
nvinfer1::AsciiChar const* VoxelUniquePlugin::getPluginNamespace() const noexcept { return kPluginNamespace; }

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
    auto const* runtimeM = exprBuilder.declareSizeTensor(
        0, *inputs[0].d[0], *inputs[0].d[0]);
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

int32_t VoxelUniquePlugin::getNbOutputs() const noexcept { return 3; }

int32_t VoxelUniquePlugin::onShapeChange(
    nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != 1 || nbOutputs != 3
        || inputs[0].dims.nbDims != 1)
    {
        return 1;
    }
    int64_t const n = inputs[0].dims.d[0];
    return n >= 1 && n <= INT32_MAX ? 0 : 1;
}

int32_t VoxelUniquePlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    (void) outputDesc;
    (void) workspace;
    if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr
        || inputDesc[0].dims.nbDims != 1)
    {
        return 1;
    }
    int64_t const n64 = inputDesc[0].dims.d[0];
    if (n64 < 1 || n64 > INT32_MAX)
    {
        return 1;
    }
    voxelUniqueSerialKernel<<<1, 1, 0, stream>>>(
        static_cast<int64_t const*>(inputs[0]), static_cast<int32_t>(n64),
        static_cast<int32_t*>(outputs[0]),
        static_cast<int64_t*>(outputs[1]),
        static_cast<int64_t*>(outputs[2]));
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
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
    return new (std::nothrow) VoxelUniquePlugin();
}

nvinfer1::PluginFieldCollection const* VoxelUniquePluginCreator::getFieldNames() noexcept
{
    static nvinfer1::PluginFieldCollection fields{0, nullptr};
    return &fields;
}
nvinfer1::AsciiChar const* VoxelUniquePluginCreator::getPluginName() const noexcept { return kPluginName; }
nvinfer1::AsciiChar const* VoxelUniquePluginCreator::getPluginVersion() const noexcept { return kPluginVersion; }
nvinfer1::AsciiChar const* VoxelUniquePluginCreator::getPluginNamespace() const noexcept { return kPluginNamespace; }
int32_t getBuildCreationCount() noexcept { return gBuildCreationCount.load(); }
int32_t getRuntimeCreationCount() noexcept { return gRuntimeCreationCount.load(); }

} // namespace ptv2::voxel_unique_correctness

