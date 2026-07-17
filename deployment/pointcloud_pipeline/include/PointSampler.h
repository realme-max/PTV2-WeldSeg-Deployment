#pragma once

#include "PointCloudLoader.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ptv2::pointcloud
{

class PointSampler
{
public:
    explicit PointSampler(std::uint32_t seed = 42U) noexcept;

    bool sample(
        std::vector<PointXYZL> const& input,
        std::vector<PointXYZL>& output,
        int targetPoints = 2048);
    bool sample(
        std::vector<PointXYZL> const& input,
        std::vector<PointXYZL>& output,
        std::vector<std::size_t>& sampledIndices,
        int targetPoints = 2048);

    std::string const& lastError() const noexcept;

private:
    std::uint32_t seed_;
    std::string lastError_;
};

} // namespace ptv2::pointcloud
