#pragma once

#include <string>
#include <vector>

namespace ptv2::weld
{

struct WeldResult
{
    bool success{};
    std::string task_id;
    int total_points{};
    int original_points{};
    int sampled_points{};
    int weld_points{};
    float weld_ratio{};
    float center[3]{};
    float bbox_min[3]{};
    float bbox_max[3]{};
    float length_mm{};
    float inference_ms{};
    float load_cloud_ms{};
    float sampling_ms{};
    float adjacency_build_ms{};
    float inference_wall_ms{};
    float postprocess_ms{};
    float total_ms{};
    int error_recorder_errors{};
    std::vector<int> labels;

    void clear() noexcept;
};

} // namespace ptv2::weld
