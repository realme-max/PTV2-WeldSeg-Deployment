#include "ResultWriter.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <system_error>

namespace ptv2::postprocess
{
namespace
{

std::string jsonEscape(std::string const& value)
{
    std::string result;
    for (char item : value)
    {
        if (item == '\\' || item == '"') result.push_back('\\');
        result.push_back(item);
    }
    return result;
}

} // namespace

bool ResultWriter::write(
    std::filesystem::path const& outputDirectory,
    std::string const& taskId,
    std::vector<SegmentationPoint> const& segmentation,
    WeldGeometryResult const& geometry,
    double inferenceMs,
    std::filesystem::path const& predictionPath)
{
    lastError_.clear();
    if (outputDirectory.empty() || taskId.empty() || segmentation.empty())
    {
        lastError_ = "Output directory, task id, and segmentation must be non-empty";
        return false;
    }
    if (!std::isfinite(inferenceMs) || inferenceMs < 0.0)
    {
        lastError_ = "Inference time must be finite and non-negative";
        return false;
    }
    std::size_t const actualWeld = static_cast<std::size_t>(std::count_if(
        segmentation.begin(), segmentation.end(), [](SegmentationPoint const& point) {
            return point.label == 0;
        }));
    if (actualWeld == 0U || actualWeld != geometry.weldPoints)
    {
        lastError_ = "Weld geometry count does not match segmentation";
        return false;
    }

    std::error_code error;
    if (std::filesystem::exists(outputDirectory, error)
        && !std::filesystem::is_directory(outputDirectory, error))
    {
        lastError_ = "Output path exists and is not a directory: " + outputDirectory.string();
        return false;
    }
    std::filesystem::create_directories(outputDirectory, error);
    if (error)
    {
        lastError_ = "Cannot create output directory: " + error.message();
        return false;
    }
    auto const resolvedPrediction = predictionPath.empty()
        ? outputDirectory / "prediction.txt" : predictionPath;
    if (!resolvedPrediction.parent_path().empty())
    {
        std::filesystem::create_directories(resolvedPrediction.parent_path(), error);
        if (error)
        {
            lastError_ = "Cannot create prediction output directory: " + error.message();
            return false;
        }
    }

    return writeJson(outputDirectory / "weld_result.json", taskId, segmentation, geometry, inferenceMs)
        && writePly(outputDirectory / "weld_points.ply", segmentation, geometry)
        && writePrediction(resolvedPrediction, segmentation);
}

bool ResultWriter::writeJson(
    std::filesystem::path const& path,
    std::string const& taskId,
    std::vector<SegmentationPoint> const& segmentation,
    WeldGeometryResult const& geometry,
    double inferenceMs)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output)
    {
        lastError_ = "Cannot open weld result JSON: " + path.string();
        return false;
    }
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n"
           << "  \"task_id\": \"" << jsonEscape(taskId) << "\",\n"
           << "  \"total_points\": " << segmentation.size() << ",\n"
           << "  \"weld_points\": " << geometry.weldPoints << ",\n"
           << "  \"weld_ratio\": " << geometry.weldRatio << ",\n"
           << "  \"center\": [" << geometry.center[0] << ", " << geometry.center[1]
           << ", " << geometry.center[2] << "],\n"
           << "  \"bbox\": {\n"
           << "    \"min\": [" << geometry.bboxMin[0] << ", " << geometry.bboxMin[1]
           << ", " << geometry.bboxMin[2] << "],\n"
           << "    \"max\": [" << geometry.bboxMax[0] << ", " << geometry.bboxMax[1]
           << ", " << geometry.bboxMax[2] << "]\n"
           << "  },\n"
           << "  \"length_mm\": " << geometry.lengthMm << ",\n"
           << "  \"inference_ms\": " << inferenceMs << "\n"
           << "}\n";
    if (!output)
    {
        lastError_ = "Failed to write weld result JSON: " + path.string();
        return false;
    }
    return true;
}

bool ResultWriter::writePly(
    std::filesystem::path const& path,
    std::vector<SegmentationPoint> const& segmentation,
    WeldGeometryResult const& geometry)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output)
    {
        lastError_ = "Cannot open weld PLY: " + path.string();
        return false;
    }
    output << "ply\n"
           << "format ascii 1.0\n"
           << "comment class_0 weld_seam\n"
           << "element vertex " << geometry.weldPoints << "\n"
           << "property float x\n"
           << "property float y\n"
           << "property float z\n"
           << "property int label\n"
           << "property float confidence\n"
           << "end_header\n"
           << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (auto const& point : segmentation)
    {
        if (point.label != 0) continue;
        output << point.x << ' ' << point.y << ' ' << point.z << ' '
               << point.label << ' ' << point.confidence << '\n';
    }
    if (!output)
    {
        lastError_ = "Failed to write weld PLY: " + path.string();
        return false;
    }
    return true;
}

bool ResultWriter::writePrediction(
    std::filesystem::path const& path,
    std::vector<SegmentationPoint> const& segmentation)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output)
    {
        lastError_ = "Cannot open prediction TXT: " + path.string();
        return false;
    }
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (auto const& point : segmentation)
    {
        output << point.x << ' ' << point.y << ' ' << point.z << ' ' << point.label << '\n';
    }
    if (!output)
    {
        lastError_ = "Failed to write prediction TXT: " + path.string();
        return false;
    }
    return true;
}

std::string const& ResultWriter::lastError() const noexcept
{
    return lastError_;
}

} // namespace ptv2::postprocess
