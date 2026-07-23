#include "WeldConfig.h"
#include "WeldDetector.h"
#include "WeldResult.h"
#include "WeldStatus.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace
{

std::map<std::string, std::string> parseArguments(int argc, char** argv)
{
    std::map<std::string, std::string> values;
    for (int index = 1; index < argc; ++index)
    {
        std::string const key = argv[index];
        if (key.rfind("--", 0) != 0 || index + 1 >= argc)
            throw std::runtime_error("Invalid command-line argument: " + key);
        values[key] = argv[++index];
    }
    return values;
}

std::string required(std::map<std::string, std::string> const& values, char const* key)
{
    auto const found = values.find(key);
    if (found == values.end() || found->second.empty())
        throw std::runtime_error(std::string("Missing required argument: ") + key);
    return found->second;
}

void writeLabels(std::filesystem::path const& path, ptv2::weld::WeldResult const& result)
{
    if (path.empty()) return;
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::trunc);
    for (int label : result.labels) output << label << '\n';
    if (!output) throw std::runtime_error("Failed to write SDK labels: " + path.string());
}

void writeResult(std::filesystem::path const& path, ptv2::weld::WeldResult const& result)
{
    if (path.empty()) return;
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::trunc);
    output << std::setprecision(std::numeric_limits<float>::max_digits10)
           << "{\n"
           << "  \"success\": true,\n"
           << "  \"task_id\": \"" << result.task_id << "\",\n"
           << "  \"total_points\": " << result.total_points << ",\n"
           << "  \"weld_points\": " << result.weld_points << ",\n"
           << "  \"weld_ratio\": " << result.weld_ratio << ",\n"
           << "  \"center\": [" << result.center[0] << ',' << result.center[1] << ',' << result.center[2] << "],\n"
           << "  \"bbox_min\": [" << result.bbox_min[0] << ',' << result.bbox_min[1] << ',' << result.bbox_min[2] << "],\n"
           << "  \"bbox_max\": [" << result.bbox_max[0] << ',' << result.bbox_max[1] << ',' << result.bbox_max[2] << "],\n"
           << "  \"length_mm\": " << result.length_mm << ",\n"
           << "  \"inference_ms\": " << result.inference_ms << ",\n"
           << "  \"labels_count\": " << result.labels.size() << "\n"
           << "}\n";
    if (!output) throw std::runtime_error("Failed to write SDK result: " + path.string());
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        auto const values = parseArguments(argc, argv);
        ptv2::weld::WeldConfig config;
        config.engine_path = required(values, "--engine");
        config.plugin_path = required(values, "--plugin");
        if (values.count("--output") != 0) config.output_path = values.at("--output");

        ptv2::weld::WeldDetector detector;
        ptv2::weld::WeldStatus status = detector.initialize(config);
        if (status != ptv2::weld::WeldStatus::SUCCESS)
        {
            std::cerr << "STATUS=" << ptv2::weld::toString(status) << '\n'
                      << "ERROR=" << detector.lastError() << '\n';
            return 1;
        }

        ptv2::weld::WeldResult result;
        status = detector.detect(required(values, "--cloud"), result);
        if (status != ptv2::weld::WeldStatus::SUCCESS)
        {
            std::cerr << "STATUS=" << ptv2::weld::toString(status) << '\n'
                      << "ERROR=" << detector.lastError() << '\n';
            return 1;
        }
        if (!result.success || result.total_points != 2048
            || result.weld_points != 209 || result.labels.size() != 2048U
            || std::abs(result.length_mm - 57.19605255F) > 0.01F)
        {
            std::cerr << "STATUS=POSTPROCESS_FAILED\n"
                      << "ERROR=SDK result failed the weld_65 smoke contract\n";
            return 1;
        }

        if (values.count("--labels-output") != 0)
            writeLabels(values.at("--labels-output"), result);
        if (values.count("--result-output") != 0)
            writeResult(values.at("--result-output"), result);

        std::cout << "STATUS=SUCCESS\n"
                  << "TASK_ID=" << result.task_id << '\n'
                  << "TOTAL_POINTS=" << result.total_points << '\n'
                  << "WELD_POINTS=" << result.weld_points << '\n'
                  << "WELD_RATIO=" << std::setprecision(10) << result.weld_ratio << '\n'
                  << "LENGTH_MM=" << result.length_mm << '\n'
                  << "INFERENCE_MS=" << result.inference_ms << '\n'
                  << "SDK_SMOKE_TEST_PASSED\n";
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "STATUS=INVALID_CONFIG\nERROR=" << error.what() << '\n';
        return 1;
    }
}
