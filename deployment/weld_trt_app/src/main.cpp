#include "FeatureBuilder.h"
#include "KnnGraphBuilder.h"
#include "PointCloudLoader.h"
#include "PointSampler.h"
#include "TensorRTInference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;
constexpr char kProductionEngineSha256[]{"a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299"};

double elapsedMs(Clock::time_point started)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

struct Arguments
{
    std::string cloud;
    std::string engine;
    std::string plugin;
    std::string output;
    std::string report;
    std::string engineSha256{kProductionEngineSha256};
    std::string logitsOutput;
    std::string pointsOutput;
    std::string adjOutput;
    std::string sampleIndicesOutput;
    std::uint32_t seed{42U};
};

std::string jsonEscape(std::string const& value)
{
    std::string result;
    for (char item : value)
    {
        if (item == '\\' || item == '"')
        {
            result.push_back('\\');
        }
        result.push_back(item);
    }
    return result;
}

Arguments parseArguments(int argc, char** argv)
{
    std::map<std::string, std::string> values;
    for (int index = 1; index < argc; ++index)
    {
        std::string const key = argv[index];
        if (key.rfind("--", 0) != 0 || index + 1 >= argc)
        {
            throw std::runtime_error("Invalid command-line argument: " + key);
        }
        values[key] = argv[++index];
    }
    auto required = [&](char const* key) -> std::string {
        auto const found = values.find(key);
        if (found == values.end() || found->second.empty())
        {
            throw std::runtime_error(std::string("Missing required argument: ") + key);
        }
        return found->second;
    };
    Arguments args;
    args.cloud = required("--cloud");
    args.engine = required("--engine");
    args.plugin = required("--plugin");
    args.output = required("--output");
    std::filesystem::path const outputPath(args.output);
    args.report = values.count("--report") != 0
        ? values["--report"] : (outputPath.parent_path() / "runtime_report.json").string();
    if (values.count("--engine-sha256") != 0) args.engineSha256 = values["--engine-sha256"];
    if (values.count("--logits-output") != 0) args.logitsOutput = values["--logits-output"];
    if (values.count("--points-output") != 0) args.pointsOutput = values["--points-output"];
    if (values.count("--adj-output") != 0) args.adjOutput = values["--adj-output"];
    if (values.count("--sample-indices-output") != 0) args.sampleIndicesOutput = values["--sample-indices-output"];
    if (values.count("--seed") != 0) args.seed = static_cast<std::uint32_t>(std::stoul(values["--seed"]));
    return args;
}

template <typename T>
void writeBinary(std::string const& path, std::vector<T> const& values)
{
    if (path.empty()) return;
    std::filesystem::path const file(path);
    if (!file.parent_path().empty()) std::filesystem::create_directories(file.parent_path());
    std::ofstream output(file, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<char const*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!output) throw std::runtime_error("Failed to write binary output: " + path);
}

void writePredictions(
    std::string const& path,
    std::vector<ptv2::pointcloud::PointXYZL> const& sampled,
    std::vector<float> const& logits)
{
    std::filesystem::path const file(path);
    if (!file.parent_path().empty()) std::filesystem::create_directories(file.parent_path());
    std::ofstream output(file, std::ios::trunc);
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::size_t index = 0; index < sampled.size(); ++index)
    {
        int const prediction = logits[index * 2U] >= logits[index * 2U + 1U] ? 0 : 1;
        auto const& point = sampled[index];
        output << point.x << ' ' << point.y << ' ' << point.z << ' ' << prediction << '\n';
    }
    if (!output) throw std::runtime_error("Failed to write prediction TXT: " + path);
}

void writeReport(
    Arguments const& args,
    ptv2::pointcloud::PointCloudStats const& stats,
    ptv2::pointcloud::NormalizationStats const& normalization,
    ptv2::runtime::TensorRTInference const& runtime,
    double loadMs, double sampleMs, double featureMs, double knnMs, double inferenceWallMs,
    std::vector<float> const& logits)
{
    std::filesystem::path const file(args.report);
    if (!file.parent_path().empty()) std::filesystem::create_directories(file.parent_path());
    auto const bounds = std::minmax_element(logits.begin(), logits.end());
    bool const finite = std::all_of(logits.begin(), logits.end(), [](float item) { return std::isfinite(item); });
    std::ofstream output(file, std::ios::trunc);
    output << std::setprecision(10)
           << "{\n"
           << "  \"status\": \"PASS\",\n"
           << "  \"cloud\": \"" << jsonEscape(args.cloud) << "\",\n"
           << "  \"input_points\": " << stats.pointCount << ",\n"
           << "  \"sample_points\": 2048,\n"
           << "  \"seed\": " << args.seed << ",\n"
           << "  \"feature_shape\": [1, 2048, 4],\n"
           << "  \"adj_shape\": [1, 2048, 2048],\n"
           << "  \"output_shape\": [1, 2048,2],\n"
           << "  \"normalization_centroid\": [" << normalization.centroid[0] << ','
           << normalization.centroid[1] << ',' << normalization.centroid[2] << "],\n"
           << "  \"normalization_radius\": " << normalization.radius << ",\n"
           << "  \"load_ms\": " << loadMs << ",\n"
           << "  \"sample_ms\": " << sampleMs << ",\n"
           << "  \"feature_ms\": " << featureMs << ",\n"
           << "  \"knn_ms\": " << knnMs << ",\n"
           << "  \"inference_ms\": " << runtime.lastInferenceDeviceMs() << ",\n"
           << "  \"inference_wall_ms\": " << inferenceWallMs << ",\n"
           << "  \"runtime_plugin_instances\": " << runtime.runtimePluginInstances() << ",\n"
           << "  \"error_recorder_errors\": " << runtime.errorRecorderErrors() << ",\n"
           << "  \"engine_sha256\": \"" << runtime.engineSha256() << "\",\n"
           << "  \"output_finite\": " << (finite ? "true" : "false") << ",\n"
           << "  \"logits_min\": " << *bounds.first << ",\n"
           << "  \"logits_max\": " << *bounds.second << "\n"
           << "}\n";
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        Arguments const args = parseArguments(argc, argv);
        ptv2::pointcloud::PointCloudLoader loader;
        std::vector<ptv2::pointcloud::PointXYZL> fullCloud;
        auto started = Clock::now();
        if (!loader.load(args.cloud, fullCloud)) throw std::runtime_error(loader.lastError());
        double const loadMs = elapsedMs(started);

        ptv2::pointcloud::PointSampler sampler(args.seed);
        std::vector<ptv2::pointcloud::PointXYZL> sampled;
        std::vector<std::size_t> sampleIndices;
        started = Clock::now();
        if (!sampler.sample(fullCloud, sampled, sampleIndices, 2048)) throw std::runtime_error(sampler.lastError());
        double const sampleMs = elapsedMs(started);

        ptv2::pointcloud::FeatureBuilder featureBuilder;
        std::vector<float> features;
        started = Clock::now();
        if (!featureBuilder.buildPointsFeature(fullCloud, sampled, features))
            throw std::runtime_error(featureBuilder.lastError());
        double const featureMs = elapsedMs(started);

        ptv2::pointcloud::KnnGraphBuilder graphBuilder(6U);
        std::vector<float> adjacency;
        started = Clock::now();
        if (!graphBuilder.build(sampled, adjacency)) throw std::runtime_error(graphBuilder.lastError());
        double const knnMs = elapsedMs(started);

        ptv2::runtime::TensorRTInference runtime;
        if (!runtime.initialize(args.engine, args.plugin, args.engineSha256))
            throw std::runtime_error(runtime.lastError());
        std::vector<float> logits(2048U * 2U);
        started = Clock::now();
        if (!runtime.infer(features.data(), adjacency.data(), logits.data()))
            throw std::runtime_error(runtime.lastError());
        double const inferenceWallMs = elapsedMs(started);
        if (!std::all_of(logits.begin(), logits.end(), [](float item) { return std::isfinite(item); }))
            throw std::runtime_error("TensorRT logits contain NaN/Inf");

        writePredictions(args.output, sampled, logits);
        writeBinary(args.logitsOutput, logits);
        writeBinary(args.pointsOutput, features);
        writeBinary(args.adjOutput, adjacency);
        std::vector<std::uint64_t> indices64(sampleIndices.begin(), sampleIndices.end());
        writeBinary(args.sampleIndicesOutput, indices64);
        writeReport(args, loader.stats(), featureBuilder.normalization(), runtime,
            loadMs, sampleMs, featureMs, knnMs, inferenceWallMs, logits);
        std::cout << "CPP_POINTCLOUD_PIPELINE_INFERENCE_PASSED\n";
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "CPP_POINTCLOUD_PIPELINE_FAILED: " << error.what() << '\n';
        return 1;
    }
}
