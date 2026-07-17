#pragma once

#include "SegmentationPostProcessor.h"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace ptv2::postprocess
{

struct WeldGeometryResult
{
    std::size_t weldPoints{};
    float weldRatio{};
    std::array<float, 3> center{};
    std::array<float, 3> bboxMin{};
    std::array<float, 3> bboxMax{};
    float lengthMm{};
};

class WeldGeometryExtractor
{
public:
    bool extract(
        std::vector<SegmentationPoint> const& segmentation,
        WeldGeometryResult& result);

    std::string const& lastError() const noexcept;

private:
    std::string lastError_;
};

} // namespace ptv2::postprocess
