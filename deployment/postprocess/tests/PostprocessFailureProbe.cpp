#include "CoordinateRecovery.h"
#include "ResultWriter.h"
#include "SegmentationPostProcessor.h"
#include "WeldGeometryExtractor.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using ptv2::pointcloud::PointXYZL;
using ptv2::postprocess::ResultWriter;
using ptv2::postprocess::SegmentationPoint;
using ptv2::postprocess::SegmentationPostProcessor;
using ptv2::postprocess::WeldGeometryExtractor;
using ptv2::postprocess::WeldGeometryResult;

std::vector<PointXYZL> points()
{
    std::vector<PointXYZL> result(SegmentationPostProcessor::kExpectedPoints);
    for (std::size_t index = 0; index < result.size(); ++index)
    {
        result[index] = PointXYZL{
            static_cast<float>(index), static_cast<float>(index % 7U), 1.0F, 1};
    }
    return result;
}

void fail(std::string const& message)
{
    throw std::runtime_error(message);
}

void runCase(std::string const& testCase)
{
    auto const inputPoints = points();
    SegmentationPostProcessor processor;
    std::vector<SegmentationPoint> segmentation;
    std::vector<float> logits(SegmentationPostProcessor::kExpectedPoints * 2U, 0.0F);

    if (testCase == "empty_logits")
    {
        logits.clear();
        if (!processor.process(inputPoints, logits, segmentation)) fail(processor.lastError());
        throw std::runtime_error("Empty logits unexpectedly succeeded");
    }
    if (testCase == "wrong_shape")
    {
        logits.pop_back();
        if (!processor.process(inputPoints, logits, segmentation)) fail(processor.lastError());
        throw std::runtime_error("Wrong logits shape unexpectedly succeeded");
    }
    if (testCase == "nan_logits")
    {
        logits[0] = std::numeric_limits<float>::quiet_NaN();
        if (!processor.process(inputPoints, logits, segmentation)) fail(processor.lastError());
        throw std::runtime_error("NaN logits unexpectedly succeeded");
    }
    if (testCase == "no_weld")
    {
        for (std::size_t index = 0; index < inputPoints.size(); ++index)
        {
            logits[index * 2U] = -1.0F;
            logits[index * 2U + 1U] = 1.0F;
        }
        if (!processor.process(inputPoints, logits, segmentation))
            throw std::runtime_error("Setup failed: " + processor.lastError());
        WeldGeometryExtractor extractor;
        WeldGeometryResult geometry;
        if (!extractor.extract(segmentation, geometry)) fail(extractor.lastError());
        throw std::runtime_error("No-weld geometry unexpectedly succeeded");
    }
    if (testCase == "unwritable_output")
    {
        for (std::size_t index = 0; index < inputPoints.size(); ++index)
        {
            logits[index * 2U] = index == 0U ? 1.0F : -1.0F;
            logits[index * 2U + 1U] = index == 0U ? -1.0F : 1.0F;
        }
        if (!processor.process(inputPoints, logits, segmentation))
            throw std::runtime_error("Setup failed: " + processor.lastError());
        WeldGeometryExtractor extractor;
        WeldGeometryResult geometry;
        if (!extractor.extract(segmentation, geometry))
            throw std::runtime_error("Setup failed: " + extractor.lastError());
        auto const root = std::filesystem::temp_directory_path() / "ptv2_phase9c_failure_probe";
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root);
        auto const blocked = root / "not_a_directory";
        std::ofstream(blocked) << "block";
        ResultWriter writer;
        bool const succeeded = writer.write(blocked, "invalid", segmentation, geometry, 1.0);
        std::filesystem::remove_all(root);
        if (!succeeded) fail(writer.lastError());
        throw std::runtime_error("Unwritable output path unexpectedly succeeded");
    }
    throw std::runtime_error("Unknown case: " + testCase);
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 3 || std::string(argv[1]) != "--case")
    {
        std::cerr << "Usage: postprocess_failure_probe --case CASE\n";
        return 2;
    }
    try
    {
        runCase(argv[2]);
        std::cerr << "CPP_POSTPROCESS_PIPELINE_FAILED: failure probe unexpectedly returned\n";
        return 2;
    }
    catch (std::exception const& error)
    {
        std::cerr << "CPP_POSTPROCESS_PIPELINE_FAILED: " << error.what() << '\n';
        return 1;
    }
}
