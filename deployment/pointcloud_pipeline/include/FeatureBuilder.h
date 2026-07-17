#pragma once

#include "PointCloudLoader.h"

#include <array>
#include <string>
#include <vector>

namespace ptv2::pointcloud
{

struct NormalizationStats
{
    std::array<double, 3> centroid{};
    double radius{};
};

class FeatureBuilder
{
public:
    bool buildPointsFeature(
        std::vector<PointXYZL> const& points,
        std::vector<float>& feature);
    bool buildPointsFeature(
        std::vector<PointXYZL> const& normalizationReference,
        std::vector<PointXYZL> const& sampledPoints,
        std::vector<float>& feature);

    NormalizationStats const& normalization() const noexcept;
    std::string const& lastError() const noexcept;

private:
    bool computeNormalization(std::vector<PointXYZL> const& reference);

    NormalizationStats normalization_{};
    std::string lastError_;
};

} // namespace ptv2::pointcloud
