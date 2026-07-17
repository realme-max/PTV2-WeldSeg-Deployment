#pragma once

#include "SegmentationPostProcessor.h"
#include "WeldGeometryExtractor.h"

#include <filesystem>
#include <string>
#include <vector>

namespace ptv2::postprocess
{

class ResultWriter
{
public:
    bool write(
        std::filesystem::path const& outputDirectory,
        std::string const& taskId,
        std::vector<SegmentationPoint> const& segmentation,
        WeldGeometryResult const& geometry,
        double inferenceMs,
        std::filesystem::path const& predictionPath = {});

    std::string const& lastError() const noexcept;

private:
    bool writeJson(
        std::filesystem::path const& path,
        std::string const& taskId,
        std::vector<SegmentationPoint> const& segmentation,
        WeldGeometryResult const& geometry,
        double inferenceMs);
    bool writePly(
        std::filesystem::path const& path,
        std::vector<SegmentationPoint> const& segmentation,
        WeldGeometryResult const& geometry);
    bool writePrediction(
        std::filesystem::path const& path,
        std::vector<SegmentationPoint> const& segmentation);

    std::string lastError_;
};

} // namespace ptv2::postprocess
