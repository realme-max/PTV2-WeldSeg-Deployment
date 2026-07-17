#include "VoxelUniqueCubPlugin.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ptv2::voxel_unique_cub
{
namespace
{
constexpr size_t kWorkspaceAlignment{256};
constexpr int32_t kThreadsPerBlock{256};

size_t alignUp(size_t value) noexcept
{
    return (value + kWorkspaceAlignment - 1U) & ~(kWorkspaceAlignment - 1U);
}

template <typename T>
T* workspacePointer(void* workspace, size_t offset) noexcept
{
    return reinterpret_cast<T*>(static_cast<std::byte*>(workspace) + offset);
}

bool queryCubTemporaryBytes(int32_t n, size_t& sortBytes, size_t& scanBytes) noexcept
{
    sortBytes = 0;
    scanBytes = 0;
    cudaError_t status = cub::DeviceRadixSort::SortPairs(
        nullptr, sortBytes,
        static_cast<int64_t const*>(nullptr), static_cast<int64_t*>(nullptr),
        static_cast<int32_t const*>(nullptr), static_cast<int32_t*>(nullptr),
        n, 0, 64, cudaStream_t{});
    if (status != cudaSuccess)
    {
        return false;
    }
    status = cub::DeviceScan::InclusiveSum(
        nullptr, scanBytes,
        static_cast<int32_t const*>(nullptr), static_cast<int32_t*>(nullptr),
        n, cudaStream_t{});
    return status == cudaSuccess;
}

__global__ void initializeOriginalIndices(int32_t* indices, int32_t n)
{
    int32_t const index = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < n)
    {
        indices[index] = index;
    }
}

__global__ void markRunBoundaries(int64_t const* sortedKeys, int32_t* flags, int32_t n)
{
    int32_t const index = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < n)
    {
        flags[index] = index == 0 || sortedKeys[index] != sortedKeys[index - 1] ? 1 : 0;
    }
}

__global__ void materializeUniqueAndInverse(
    int64_t const* sortedKeys,
    int32_t const* sortedOriginalIndices,
    int32_t const* boundaryFlags,
    int32_t const* uniqueIds,
    int32_t n,
    int32_t* count,
    int64_t* uniqueValues,
    int64_t* inverseIndices)
{
    int32_t const index = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= n)
    {
        return;
    }
    // Inclusive scan produces one-based run ids; TensorRT/PyTorch inverse
    // indices are zero-based.
    int32_t const uniqueId = uniqueIds[index] - 1;
    if (boundaryFlags[index] != 0)
    {
        uniqueValues[uniqueId] = sortedKeys[index];
    }
    inverseIndices[sortedOriginalIndices[index]] = static_cast<int64_t>(uniqueId);
    if (index == n - 1)
    {
        *count = uniqueIds[index];
    }
}
} // namespace

bool calculateWorkspaceLayout(int32_t n, WorkspaceLayout& layout) noexcept
{
    if (n < 1 || n > kMaxInputSize)
    {
        return false;
    }
    size_t sortBytes{};
    size_t scanBytes{};
    if (!queryCubTemporaryBytes(n, sortBytes, scanBytes))
    {
        return false;
    }
    size_t cursor{};
    auto allocate = [&cursor](size_t bytes) noexcept {
        size_t const offset = alignUp(cursor);
        cursor = offset + bytes;
        return offset;
    };
    layout = {};
    layout.sortedKeysOffset = allocate(static_cast<size_t>(n) * sizeof(int64_t));
    layout.originalIndicesOffset = allocate(static_cast<size_t>(n) * sizeof(int32_t));
    layout.sortedOriginalIndicesOffset = allocate(static_cast<size_t>(n) * sizeof(int32_t));
    layout.boundaryFlagsOffset = allocate(static_cast<size_t>(n) * sizeof(int32_t));
    layout.uniqueIdsOffset = allocate(static_cast<size_t>(n) * sizeof(int32_t));
    layout.cubTemporaryOffset = alignUp(cursor);
    layout.radixSortTemporaryBytes = sortBytes;
    layout.scanTemporaryBytes = scanBytes;
    layout.cubTemporaryBytes = std::max(sortBytes, scanBytes);
    layout.totalBytes = alignUp(layout.cubTemporaryOffset + layout.cubTemporaryBytes);
    return true;
}

size_t VoxelUniqueCubPlugin::getWorkspaceSize(
    nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    (void) outputs;
    if (inputs == nullptr || nbInputs != 1 || nbOutputs != 3)
    {
        return 0;
    }
    int64_t maxN = inputs[0].max.d[0];
    if (maxN < 1 || maxN > kMaxInputSize)
    {
        maxN = kMaxInputSize;
    }
    WorkspaceLayout layout{};
    return calculateWorkspaceLayout(static_cast<int32_t>(maxN), layout) ? layout.totalBytes : 0;
}

int32_t VoxelUniqueCubPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    (void) outputDesc;
    if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr || workspace == nullptr
        || inputDesc[0].dims.nbDims != 1)
    {
        return 1;
    }
    int64_t const n64 = inputDesc[0].dims.d[0];
    if (n64 < 1 || n64 > kMaxInputSize)
    {
        return 1;
    }
    int32_t const n = static_cast<int32_t>(n64);
    WorkspaceLayout layout{};
    if (!calculateWorkspaceLayout(n, layout))
    {
        return 1;
    }

    auto const* keys = static_cast<int64_t const*>(inputs[0]);
    auto* count = static_cast<int32_t*>(outputs[0]);
    auto* uniqueValues = static_cast<int64_t*>(outputs[1]);
    auto* inverseIndices = static_cast<int64_t*>(outputs[2]);
    auto* sortedKeys = workspacePointer<int64_t>(workspace, layout.sortedKeysOffset);
    auto* originalIndices = workspacePointer<int32_t>(workspace, layout.originalIndicesOffset);
    auto* sortedOriginalIndices = workspacePointer<int32_t>(workspace, layout.sortedOriginalIndicesOffset);
    auto* boundaryFlags = workspacePointer<int32_t>(workspace, layout.boundaryFlagsOffset);
    auto* uniqueIds = workspacePointer<int32_t>(workspace, layout.uniqueIdsOffset);
    void* cubTemporary = workspacePointer<std::byte>(workspace, layout.cubTemporaryOffset);

    int32_t const blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
    initializeOriginalIndices<<<blocks, kThreadsPerBlock, 0, stream>>>(originalIndices, n);
    cudaError_t status = cub::DeviceRadixSort::SortPairs(
        cubTemporary, layout.radixSortTemporaryBytes,
        keys, sortedKeys, originalIndices, sortedOriginalIndices,
        n, 0, 64, stream);
    if (status != cudaSuccess)
    {
        return 1;
    }
    markRunBoundaries<<<blocks, kThreadsPerBlock, 0, stream>>>(sortedKeys, boundaryFlags, n);
    status = cub::DeviceScan::InclusiveSum(
        cubTemporary, layout.scanTemporaryBytes,
        boundaryFlags, uniqueIds, n, stream);
    if (status != cudaSuccess)
    {
        return 1;
    }
    materializeUniqueAndInverse<<<blocks, kThreadsPerBlock, 0, stream>>>(
        sortedKeys, sortedOriginalIndices, boundaryFlags, uniqueIds, n,
        count, uniqueValues, inverseIndices);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

} // namespace ptv2::voxel_unique_cub
