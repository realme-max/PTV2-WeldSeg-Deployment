#include "WeldConfig.h"
#include "WeldDetector.h"
#include "WeldResult.h"
#include "WeldStatus.h"

#include <iomanip>
#include <iostream>
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

} // namespace

int main(int argc, char** argv)
{
    try
    {
        auto const arguments = parseArguments(argc, argv);
        ptv2::weld::WeldConfig config;
        config.engine_path = required(arguments, "--engine");
        config.plugin_path = required(arguments, "--plugin");
        if (arguments.count("--output") != 0) config.output_path = arguments.at("--output");

        ptv2::weld::WeldDetector detector;
        ptv2::weld::WeldStatus status = detector.initialize(config);
        if (status != ptv2::weld::WeldStatus::SUCCESS)
        {
            std::cerr << "WELD_DETECTOR_INITIALIZE_FAILED: status="
                      << ptv2::weld::toString(status) << ", error=" << detector.lastError() << '\n';
            return 1;
        }

        ptv2::weld::WeldResult result;
        status = detector.detect(required(arguments, "--cloud"), result);
        if (status != ptv2::weld::WeldStatus::SUCCESS)
        {
            std::cerr << "WELD_DETECTOR_DETECT_FAILED: status="
                      << ptv2::weld::toString(status) << ", error=" << detector.lastError() << '\n';
            return 1;
        }

        std::cout << std::setprecision(10)
                  << "status=SUCCESS\n"
                  << "task_id=" << result.task_id << '\n'
                  << "total_points=" << result.total_points << '\n'
                  << "weld_points=" << result.weld_points << '\n'
                  << "weld_ratio=" << result.weld_ratio << '\n'
                  << "center=[" << result.center[0] << ',' << result.center[1] << ',' << result.center[2] << "]\n"
                  << "bbox_min=[" << result.bbox_min[0] << ',' << result.bbox_min[1] << ',' << result.bbox_min[2] << "]\n"
                  << "bbox_max=[" << result.bbox_max[0] << ',' << result.bbox_max[1] << ',' << result.bbox_max[2] << "]\n"
                  << "length_mm=" << result.length_mm << '\n'
                  << "inference_ms=" << result.inference_ms << '\n'
                  << "labels=" << result.labels.size() << '\n'
                  << "WELD_DETECTOR_APP_COMPLETED\n";
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "WELD_DETECTOR_APP_FAILED: " << error.what() << '\n';
        return 1;
    }
}
