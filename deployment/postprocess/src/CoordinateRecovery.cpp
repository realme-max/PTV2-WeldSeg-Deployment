#include "CoordinateRecovery.h"

#include <cmath>

namespace ptv2::postprocess
{

bool CoordinateRecovery::recover(
    std::vector<ptv2::pointcloud::PointXYZL> const& sampled,
    std::vector<ptv2::pointcloud::PointXYZL>& recovered)
{
    lastError_.clear();
    recovered.clear();
    if (sampled.empty())
    {
        lastError_ = "Cannot recover coordinates from an empty sampled point cloud";
        return false;
    }
    for (std::size_t index = 0; index < sampled.size(); ++index)
    {
        auto const& point = sampled[index];
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
        {
            lastError_ = "Sampled coordinates contain NaN/Inf at point " + std::to_string(index);
            return false;
        }
    }
    recovered = sampled;
    return true;
}

std::string const& CoordinateRecovery::lastError() const noexcept
{
    return lastError_;
}

} // namespace ptv2::postprocess
