#include "FeatureBuilder.h"

#include <cmath>

namespace ptv2::pointcloud
{

bool FeatureBuilder::computeNormalization(std::vector<PointXYZL> const& reference)
{
    if (reference.empty())
    {
        lastError_ = "Normalization reference is empty";
        return false;
    }
    normalization_ = {};
    for (PointXYZL const& point : reference)
    {
        normalization_.centroid[0] += static_cast<double>(point.x);
        normalization_.centroid[1] += static_cast<double>(point.y);
        normalization_.centroid[2] += static_cast<double>(point.z);
    }
    for (double& value : normalization_.centroid)
    {
        value /= static_cast<double>(reference.size());
    }
    for (PointXYZL const& point : reference)
    {
        double const dx = static_cast<double>(point.x) - normalization_.centroid[0];
        double const dy = static_cast<double>(point.y) - normalization_.centroid[1];
        double const dz = static_cast<double>(point.z) - normalization_.centroid[2];
        normalization_.radius = std::max(normalization_.radius, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    if (!std::isfinite(normalization_.radius) || normalization_.radius <= 0.0)
    {
        lastError_ = "Point cloud normalization radius is non-finite or zero";
        return false;
    }
    return true;
}

bool FeatureBuilder::buildPointsFeature(
    std::vector<PointXYZL> const& points,
    std::vector<float>& feature)
{
    return buildPointsFeature(points, points, feature);
}

bool FeatureBuilder::buildPointsFeature(
    std::vector<PointXYZL> const& normalizationReference,
    std::vector<PointXYZL> const& sampledPoints,
    std::vector<float>& feature)
{
    feature.clear();
    lastError_.clear();
    if (sampledPoints.size() != 2048U)
    {
        lastError_ = "FeatureBuilder requires exactly 2048 sampled points";
        return false;
    }
    if (!computeNormalization(normalizationReference))
    {
        return false;
    }
    feature.resize(sampledPoints.size() * 4U);
    for (std::size_t index = 0; index < sampledPoints.size(); ++index)
    {
        PointXYZL const& point = sampledPoints[index];
        feature[index * 4U + 0U] = static_cast<float>(
            (static_cast<double>(point.x) - normalization_.centroid[0]) / normalization_.radius);
        feature[index * 4U + 1U] = static_cast<float>(
            (static_cast<double>(point.y) - normalization_.centroid[1]) / normalization_.radius);
        feature[index * 4U + 2U] = static_cast<float>(
            (static_cast<double>(point.z) - normalization_.centroid[2]) / normalization_.radius);
        feature[index * 4U + 3U] = 1.0F;
    }
    for (float value : feature)
    {
        if (!std::isfinite(value))
        {
            lastError_ = "FeatureBuilder produced NaN/Inf";
            feature.clear();
            return false;
        }
    }
    return true;
}

NormalizationStats const& FeatureBuilder::normalization() const noexcept { return normalization_; }
std::string const& FeatureBuilder::lastError() const noexcept { return lastError_; }

} // namespace ptv2::pointcloud
