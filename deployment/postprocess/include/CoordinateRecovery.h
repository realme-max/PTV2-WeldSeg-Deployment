#pragma once

#include "PointCloudLoader.h"

#include <string>
#include <vector>

namespace ptv2::postprocess
{

class CoordinateRecovery
{
public:
    bool recover(
        std::vector<ptv2::pointcloud::PointXYZL> const& sampled,
        std::vector<ptv2::pointcloud::PointXYZL>& recovered);

    std::string const& lastError() const noexcept;

private:
    std::string lastError_;
};

} // namespace ptv2::postprocess
