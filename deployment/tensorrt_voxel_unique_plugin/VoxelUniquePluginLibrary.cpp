#include "VoxelUniqueCorrectnessPlugin.h"

#include <NvInferRuntimePlugin.h>

#include <cstdint>

namespace
{
ptv2::voxel_unique_correctness::VoxelUniquePluginCreator gCreator;
}

extern "C" __declspec(dllexport) bool initVoxelUniquePlugin() noexcept
{
    auto* registry = ::getPluginRegistry();
    if (registry == nullptr)
    {
        return false;
    }
    auto* existing = registry->getCreator(
        ptv2::voxel_unique_correctness::kPluginName,
        ptv2::voxel_unique_correctness::kPluginVersion,
        ptv2::voxel_unique_correctness::kPluginNamespace);
    if (existing != nullptr)
    {
        return true;
    }
    return registry->registerCreator(
        gCreator, ptv2::voxel_unique_correctness::kPluginNamespace);
}

extern "C" __declspec(dllexport) int32_t getVoxelUniqueBuildCreationCount() noexcept
{
    return ptv2::voxel_unique_correctness::getBuildCreationCount();
}

extern "C" __declspec(dllexport) int32_t getVoxelUniqueRuntimeCreationCount() noexcept
{
    return ptv2::voxel_unique_correctness::getRuntimeCreationCount();
}
