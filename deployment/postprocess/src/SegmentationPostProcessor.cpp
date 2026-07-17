#include "SegmentationPostProcessor.h"

#include <cmath>
#include <sstream>

namespace ptv2::postprocess
{

bool SegmentationPostProcessor::process(
    std::vector<ptv2::pointcloud::PointXYZL> const& recoveredPoints,
    std::vector<float> const& logits,
    std::vector<SegmentationPoint>& result)
{
    lastError_.clear();
    result.clear();
    if (recoveredPoints.size() != kExpectedPoints)
    {
        std::ostringstream stream;
        stream << "Recovered point count must be " << kExpectedPoints
               << ", got " << recoveredPoints.size();
        lastError_ = stream.str();
        return false;
    }
    if (logits.size() != kExpectedPoints * kClasses)
    {
        std::ostringstream stream;
        stream << "Logits must have shape [" << kExpectedPoints << ',' << kClasses
               << "] (" << kExpectedPoints * kClasses << " values), got " << logits.size();
        lastError_ = stream.str();
        return false;
    }

    result.reserve(kExpectedPoints);
    for (std::size_t index = 0; index < kExpectedPoints; ++index)
    {
        auto const& point = recoveredPoints[index];
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
        {
            lastError_ = "Recovered coordinates contain NaN/Inf at point " + std::to_string(index);
            result.clear();
            return false;
        }
        float const logit0 = logits[index * kClasses];
        float const logit1 = logits[index * kClasses + 1U];
        if (!std::isfinite(logit0) || !std::isfinite(logit1))
        {
            lastError_ = "Logits contain NaN/Inf at point " + std::to_string(index);
            result.clear();
            return false;
        }

        int const label = logit0 > logit1 ? 0 : 1;
        double const difference = std::abs(static_cast<double>(logit0) - static_cast<double>(logit1));
        float const confidence = static_cast<float>(1.0 / (1.0 + std::exp(-difference)));
        if (!std::isfinite(confidence))
        {
            lastError_ = "Softmax confidence is non-finite at point " + std::to_string(index);
            result.clear();
            return false;
        }
        result.push_back(SegmentationPoint{point.x, point.y, point.z, label, confidence});
    }
    return true;
}

std::string const& SegmentationPostProcessor::lastError() const noexcept
{
    return lastError_;
}

} // namespace ptv2::postprocess
