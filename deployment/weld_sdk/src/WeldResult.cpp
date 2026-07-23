#include "WeldResult.h"

#include <algorithm>

namespace ptv2::weld
{

void WeldResult::clear() noexcept
{
    success = false;
    task_id.clear();
    total_points = 0;
    weld_points = 0;
    weld_ratio = 0.0F;
    std::fill(std::begin(center), std::end(center), 0.0F);
    std::fill(std::begin(bbox_min), std::end(bbox_min), 0.0F);
    std::fill(std::begin(bbox_max), std::end(bbox_max), 0.0F);
    length_mm = 0.0F;
    inference_ms = 0.0F;
    labels.clear();
}

} // namespace ptv2::weld
