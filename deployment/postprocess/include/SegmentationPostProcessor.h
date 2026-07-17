#pragma once

#include "PointCloudLoader.h"

#include <string>
#include <vector>

namespace ptv2::postprocess
{

struct SegmentationPoint
{
    float x{};
    float y{};
    float z{};
    int label{};
    float confidence{};
};

class SegmentationPostProcessor
{
public:
    static constexpr std::size_t kExpectedPoints{2048U};
    static constexpr std::size_t kClasses{2U};

    bool process(
        std::vector<ptv2::pointcloud::PointXYZL> const& recoveredPoints,
        std::vector<float> const& logits,
        std::vector<SegmentationPoint>& result);

    std::string const& lastError() const noexcept;

private:
    std::string lastError_;
};

} // namespace ptv2::postprocess
