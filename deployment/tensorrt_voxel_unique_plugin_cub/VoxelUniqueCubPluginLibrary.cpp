#include "VoxelUniqueCubPlugin.h"

#include <NvInferRuntimePlugin.h>

#include <cstdint>

namespace
{
ptv2::voxel_unique_cub::VoxelUniqueCubPluginCreator gCreator;
}

extern "C" __declspec(dllexport) bool initVoxelUniqueCubPlugin() noexcept
{
    auto* registry = ::getPluginRegistry();
    if (registry == nullptr)
    {
        return false;
    }
    auto* existing = registry->getCreator(
        ptv2::voxel_unique_cub::kPluginName,
        ptv2::voxel_unique_cub::kPluginVersion,
        ptv2::voxel_unique_cub::kPluginNamespace);
    if (existing != nullptr)
    {
        return true;
    }
    return registry->registerCreator(gCreator, ptv2::voxel_unique_cub::kPluginNamespace);
}

extern "C" __declspec(dllexport) int32_t getVoxelUniqueCubBuildCreationCount() noexcept
{
    return ptv2::voxel_unique_cub::getBuildCreationCount();
}

extern "C" __declspec(dllexport) int32_t getVoxelUniqueCubRuntimeCreationCount() noexcept
{
    return ptv2::voxel_unique_cub::getRuntimeCreationCount();
}

extern "C" __declspec(dllexport) bool getVoxelUniqueCubWorkspaceLayout(
    int32_t n, ptv2::voxel_unique_cub::WorkspaceLayout* layout) noexcept
{
    return layout != nullptr && ptv2::voxel_unique_cub::calculateWorkspaceLayout(n, *layout);
}
