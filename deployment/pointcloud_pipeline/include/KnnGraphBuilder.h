#pragma once

#include "PointCloudLoader.h"

#include <cstddef>
#include <string>
#include <vector>

namespace ptv2::pointcloud
{

class KnnGraphBuilder
{
public:
    explicit KnnGraphBuilder(std::size_t neighbors = 6U) noexcept;

    bool build(std::vector<PointXYZL> const& points, std::vector<float>& adjacency);

    std::size_t neighbors() const noexcept;
    std::string const& lastError() const noexcept;

private:
    std::size_t neighbors_;
    std::string lastError_;
};

} // namespace ptv2::pointcloud
