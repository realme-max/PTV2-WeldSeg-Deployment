#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace ptv2::pointcloud
{

struct PointXYZL
{
    float x{};
    float y{};
    float z{};
    int label{};
};

struct PointCloudStats
{
    std::size_t pointCount{};
    std::array<float, 3> minimum{};
    std::array<float, 3> maximum{};
    std::array<std::size_t, 2> labelCounts{};
};

class PointCloudLoader
{
public:
    bool load(std::string const& path, std::vector<PointXYZL>& points);

    PointCloudStats const& stats() const noexcept;
    std::string const& lastError() const noexcept;

private:
    PointCloudStats stats_{};
    std::string lastError_;
};

} // namespace ptv2::pointcloud
