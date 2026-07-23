#pragma once

namespace ptv2::weld
{

enum class WeldStatus
{
    SUCCESS,
    INVALID_CONFIG,
    ENGINE_LOAD_FAILED,
    PLUGIN_LOAD_FAILED,
    POINTCLOUD_LOAD_FAILED,
    PREPROCESS_FAILED,
    INFERENCE_FAILED,
    POSTPROCESS_FAILED,
};

constexpr char const* toString(WeldStatus status) noexcept
{
    switch (status)
    {
    case WeldStatus::SUCCESS: return "SUCCESS";
    case WeldStatus::INVALID_CONFIG: return "INVALID_CONFIG";
    case WeldStatus::ENGINE_LOAD_FAILED: return "ENGINE_LOAD_FAILED";
    case WeldStatus::PLUGIN_LOAD_FAILED: return "PLUGIN_LOAD_FAILED";
    case WeldStatus::POINTCLOUD_LOAD_FAILED: return "POINTCLOUD_LOAD_FAILED";
    case WeldStatus::PREPROCESS_FAILED: return "PREPROCESS_FAILED";
    case WeldStatus::INFERENCE_FAILED: return "INFERENCE_FAILED";
    case WeldStatus::POSTPROCESS_FAILED: return "POSTPROCESS_FAILED";
    }
    return "UNKNOWN";
}

} // namespace ptv2::weld
