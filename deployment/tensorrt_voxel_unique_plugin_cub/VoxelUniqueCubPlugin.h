#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cstddef>
#include <cstdint>

namespace ptv2::voxel_unique_cub
{

inline constexpr char kPluginName[]{"VoxelUniqueCub"};
inline constexpr char kPluginVersion[]{"1"};
inline constexpr char kPluginNamespace[]{"com.tensorrt.ptv2.experimental"};
inline constexpr int32_t kMaxInputSize{2048};

struct WorkspaceLayout
{
    size_t sortedKeysOffset{};
    size_t originalIndicesOffset{};
    size_t sortedOriginalIndicesOffset{};
    size_t boundaryFlagsOffset{};
    size_t uniqueIdsOffset{};
    size_t cubTemporaryOffset{};
    size_t radixSortTemporaryBytes{};
    size_t scanTemporaryBytes{};
    size_t cubTemporaryBytes{};
    size_t totalBytes{};
};

bool calculateWorkspaceLayout(int32_t n, WorkspaceLayout& layout) noexcept;

class VoxelUniqueCubPlugin final : public nvinfer1::IPluginV3,
                                   public nvinfer1::IPluginV3OneCore,
                                   public nvinfer1::IPluginV3OneBuild,
                                   public nvinfer1::IPluginV3OneRuntime
{
public:
    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;
    nvinfer1::AsciiChar const* getPluginName() const noexcept override;
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override;
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override;
    int32_t configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override;
    int32_t getOutputDataTypes(
        nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(
        nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t position, nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(
        nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t onShapeChange(
        nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override;
    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override;
    nvinfer1::IPluginV3* attachToContext(
        nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;
};

class VoxelUniqueCubPluginCreator final : public nvinfer1::IPluginCreatorV3One
{
public:
    nvinfer1::IPluginV3* createPlugin(
        nvinfer1::AsciiChar const* name,
        nvinfer1::PluginFieldCollection const* fields,
        nvinfer1::TensorRTPhase phase) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::AsciiChar const* getPluginName() const noexcept override;
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override;
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override;
};

int32_t getBuildCreationCount() noexcept;
int32_t getRuntimeCreationCount() noexcept;

} // namespace ptv2::voxel_unique_cub
