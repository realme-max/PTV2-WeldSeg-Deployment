#include "WeldResult.h"

#include <algorithm>

namespace ptv2::weld
{

void WeldResult::clear() noexcept
{
    success = false;
    task_id.clear();
    total_points = 0;
    original_points = 0;
    sampled_points = 0;
    weld_points = 0;
    weld_ratio = 0.0F;
    std::fill(std::begin(center), std::end(center), 0.0F);
    std::fill(std::begin(bbox_min), std::end(bbox_min), 0.0F);
    std::fill(std::begin(bbox_max), std::end(bbox_max), 0.0F);
    std::fill(std::begin(principal_direction), std::end(principal_direction), 0.0F);
    length_mm = 0.0F;
    inference_ms = 0.0F;
    load_cloud_ms = 0.0F;
    sampling_ms = 0.0F;
    adjacency_build_ms = 0.0F;
    inference_wall_ms = 0.0F;
    postprocess_ms = 0.0F;
    total_ms = 0.0F;
    error_recorder_errors = 0;
    labels.clear();
    points.clear();
}

} // namespace ptv2::weld
