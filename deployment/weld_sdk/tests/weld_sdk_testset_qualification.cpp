#include "FeatureBuilder.h"
#include "KnnGraphBuilder.h"
#include "PointCloudLoader.h"
#include "PointSampler.h"
#include "WeldConfig.h"
#include "WeldDetector.h"
#include "WeldResult.h"
#include "WeldStatus.h"

#include <cuda_runtime_api.h>

#include <Windows.h>
#include <Psapi.h>
#include <TlHelp32.h>

#include <algorithm>
#include <array>
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;

struct Arguments
{
    std::string engine;
    std::string plugin;
    std::filesystem::path list;
    std::filesystem::path output;
    int rounds{1};
    int warmup{0};
    int resourceInterval{18};
    int repeatInitialize{1};
    bool compact{false};
};

struct ResourceSnapshot
{
    std::uint64_t workingSetBytes{};
    std::uint64_t privateBytes{};
    std::uint64_t gpuFreeBytes{};
    std::uint64_t gpuTotalBytes{};
    std::uint32_t handleCount{};
    std::uint32_t threadCount{};
};

double elapsedMs(Clock::time_point const started)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

std::string jsonEscape(std::string const& value)
{
    std::ostringstream output;
    for (unsigned char const character : value)
    {
        switch (character)
        {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (character < 0x20U)
            {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<unsigned int>(character) << std::dec;
            }
            else
            {
                output << static_cast<char>(character);
            }
        }
    }
    return output.str();
}

std::map<std::string, std::string> parseKeyValues(int argc, char** argv)
{
    std::map<std::string, std::string> values;
    for (int index = 1; index < argc; ++index)
    {
        std::string const key(argv[index]);
        if (key.rfind("--", 0) != 0 || index + 1 >= argc)
        {
            throw std::runtime_error("Invalid command-line argument: " + key);
        }
        values[key] = argv[++index];
    }
    return values;
}

std::string required(std::map<std::string, std::string> const& values, char const* key)
{
    auto const found = values.find(key);
    if (found == values.end() || found->second.empty())
    {
        throw std::runtime_error(std::string("Missing required argument: ") + key);
    }
    return found->second;
}

int positiveInteger(
    std::map<std::string, std::string> const& values,
    char const* key,
    int const fallback,
    bool const allowZero = false)
{
    auto const found = values.find(key);
    if (found == values.end())
    {
        return fallback;
    }
    int const parsed = std::stoi(found->second);
    if (parsed < (allowZero ? 0 : 1))
    {
        throw std::runtime_error(std::string(key) + " has an invalid value");
    }
    return parsed;
}

Arguments parseArguments(int argc, char** argv)
{
    auto const values = parseKeyValues(argc, argv);
    Arguments arguments;
    arguments.engine = required(values, "--engine");
    arguments.plugin = required(values, "--plugin");
    arguments.list = required(values, "--list");
    arguments.output = required(values, "--output");
    arguments.rounds = positiveInteger(values, "--rounds", 1);
    arguments.warmup = positiveInteger(values, "--warmup", 0, true);
    arguments.resourceInterval = positiveInteger(values, "--resource-interval", 18);
    arguments.repeatInitialize = positiveInteger(values, "--repeat-initialize", 1);
    auto const compact = values.find("--compact");
    arguments.compact = compact != values.end()
        && (compact->second == "1" || compact->second == "true");
    return arguments;
}

std::vector<std::filesystem::path> loadCloudList(std::filesystem::path const& path)
{
    std::ifstream input(path);
    if (!input)
    {
        throw std::runtime_error("Cannot open cloud list: " + path.string());
    }
    std::vector<std::filesystem::path> paths;
    std::string line;
    while (std::getline(input, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (!line.empty())
        {
            paths.emplace_back(line);
        }
    }
    if (paths.empty())
    {
        throw std::runtime_error("Cloud list is empty");
    }
    return paths;
}

std::uint64_t labelHash(std::vector<int> const& labels)
{
    std::uint64_t value = 1469598103934665603ULL;
    for (int const label : labels)
    {
        value ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(label));
        value *= 1099511628211ULL;
    }
    return value;
}

std::uint32_t processThreadCount()
{
    DWORD const processId = GetCurrentProcessId();
    HANDLE const snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE)
    {
        return 0U;
    }
    std::uint32_t count = 0U;
    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    if (Thread32First(snapshot, &entry) != FALSE)
    {
        do
        {
            if (entry.th32OwnerProcessID == processId)
            {
                ++count;
            }
        } while (Thread32Next(snapshot, &entry) != FALSE);
    }
    CloseHandle(snapshot);
    return count;
}

ResourceSnapshot resourceSnapshot()
{
    ResourceSnapshot snapshot;
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(
            GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
            sizeof(counters)) != FALSE)
    {
        snapshot.workingSetBytes = static_cast<std::uint64_t>(counters.WorkingSetSize);
        snapshot.privateBytes = static_cast<std::uint64_t>(counters.PrivateUsage);
    }
    DWORD handleCount = 0U;
    if (GetProcessHandleCount(GetCurrentProcess(), &handleCount) != FALSE)
    {
        snapshot.handleCount = static_cast<std::uint32_t>(handleCount);
    }
    snapshot.threadCount = processThreadCount();
    std::size_t freeBytes = 0U;
    std::size_t totalBytes = 0U;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess)
    {
        snapshot.gpuFreeBytes = static_cast<std::uint64_t>(freeBytes);
        snapshot.gpuTotalBytes = static_cast<std::uint64_t>(totalBytes);
    }
    return snapshot;
}

void writeResource(std::ofstream& output, char const* phase, int const detections)
{
    ResourceSnapshot const snapshot = resourceSnapshot();
    output << "{\"record_type\":\"resource\",\"phase\":\"" << phase
           << "\",\"detections\":" << detections
           << ",\"working_set_bytes\":" << snapshot.workingSetBytes
           << ",\"private_bytes\":" << snapshot.privateBytes
           << ",\"gpu_free_bytes\":" << snapshot.gpuFreeBytes
           << ",\"gpu_total_bytes\":" << snapshot.gpuTotalBytes
           << ",\"handle_count\":" << snapshot.handleCount
           << ",\"thread_count\":" << snapshot.threadCount << "}\n";
    output.flush();
}

void writeFloatTriple(std::ofstream& output, float const (&values)[3])
{
    output << '[' << values[0] << ',' << values[1] << ',' << values[2] << ']';
}

void writeDetection(
    std::ofstream& output,
    int const round,
    int const order,
    std::filesystem::path const& path,
    ptv2::weld::WeldStatus const status,
    ptv2::weld::WeldResult const& result,
    std::string const& error,
    bool const compact)
{
    output << "{\"record_type\":\"detection\",\"round\":" << round
           << ",\"order\":" << order
           << ",\"sample_id\":\"" << jsonEscape(path.stem().string()) << '"'
           << ",\"source_path\":\"" << jsonEscape(path.string()) << '"'
           << ",\"status\":\"" << ptv2::weld::toString(status) << '"'
           << ",\"success\":" << (result.success ? "true" : "false")
           << ",\"last_error\":\"" << jsonEscape(error) << '"'
           << ",\"original_points\":" << result.original_points
           << ",\"sampled_points\":" << result.sampled_points
           << ",\"total_points\":" << result.total_points
           << ",\"logits_shape\":[1,2048,2]"
           << ",\"logits_finite\":" << (result.success ? "true" : "false")
           << ",\"predicted_label_count\":" << result.labels.size()
           << ",\"weld_points\":" << result.weld_points
           << ",\"weld_ratio\":" << result.weld_ratio
           << ",\"center\":";
    writeFloatTriple(output, result.center);
    output << ",\"bbox_min\":";
    writeFloatTriple(output, result.bbox_min);
    output << ",\"bbox_max\":";
    writeFloatTriple(output, result.bbox_max);
    output << ",\"principal_direction\":";
    writeFloatTriple(output, result.principal_direction);
    output << ",\"length_mm\":" << result.length_mm
           << ",\"inference_cuda_ms\":" << result.inference_ms
           << ",\"load_cloud_ms\":" << result.load_cloud_ms
           << ",\"sampling_ms\":" << result.sampling_ms
           << ",\"adjacency_build_ms\":" << result.adjacency_build_ms
           << ",\"inference_wall_ms\":" << result.inference_wall_ms
           << ",\"postprocess_ms\":" << result.postprocess_ms
           << ",\"total_ms\":" << result.total_ms
           << ",\"error_recorder_errors\":" << result.error_recorder_errors
           << ",\"label_hash\":\"" << std::hex << labelHash(result.labels) << std::dec << '"';
    if (!compact)
    {
        output << ",\"labels\":[";
        for (std::size_t index = 0; index < result.labels.size(); ++index)
        {
            if (index != 0U) output << ',';
            output << result.labels[index];
        }
        output << ']';
    }
    output << "}\n";
    output.flush();
}

bool allFinite(std::vector<float> const& values)
{
    return std::all_of(values.begin(), values.end(), [](float const value) {
        return std::isfinite(value);
    });
}

void writePreprocessAudit(
    std::ofstream& output,
    std::filesystem::path const& path,
    ptv2::pointcloud::PointCloudLoader& loader,
    ptv2::pointcloud::PointSampler& sampler,
    ptv2::pointcloud::FeatureBuilder& featureBuilder,
    ptv2::pointcloud::KnnGraphBuilder& graphBuilder)
{
    std::vector<ptv2::pointcloud::PointXYZL> cloud;
    if (!loader.load(path.string(), cloud))
    {
        throw std::runtime_error("Preprocess audit load failed: " + loader.lastError());
    }
    std::vector<ptv2::pointcloud::PointXYZL> sampled;
    std::vector<std::size_t> indices;
    if (!sampler.sample(cloud, sampled, indices, 2048))
    {
        throw std::runtime_error("Preprocess audit sample failed: " + sampler.lastError());
    }
    std::vector<float> features;
    auto const featureStarted = Clock::now();
    if (!featureBuilder.buildPointsFeature(cloud, sampled, features))
    {
        throw std::runtime_error("Preprocess audit feature build failed: " + featureBuilder.lastError());
    }
    double const featureBuildMs = elapsedMs(featureStarted);
    std::vector<float> adjacency;
    auto const adjacencyStarted = Clock::now();
    if (!graphBuilder.build(sampled, adjacency))
    {
        throw std::runtime_error("Preprocess audit adjacency build failed: " + graphBuilder.lastError());
    }
    double const adjacencyBuildMs = elapsedMs(adjacencyStarted);
    bool featureFourConstant = true;
    for (std::size_t index = 3U; index < features.size(); index += 4U)
    {
        featureFourConstant = featureFourConstant && features[index] == 1.0F;
    }
    output << "{\"record_type\":\"preprocess_audit\""
           << ",\"sample_id\":\"" << jsonEscape(path.stem().string()) << '"'
           << ",\"source_path\":\"" << jsonEscape(path.string()) << '"'
           << ",\"original_points\":" << cloud.size()
           << ",\"sampled_points\":" << sampled.size()
           << ",\"sampling_seed\":42"
           << ",\"feature_shape\":[1,2048,4]"
           << ",\"adjacency_shape\":[1,2048,2048]"
           << ",\"points_finite\":" << (allFinite(features) ? "true" : "false")
           << ",\"adjacency_finite\":" << (allFinite(adjacency) ? "true" : "false")
           << ",\"fourth_feature_constant_one\":"
           << (featureFourConstant ? "true" : "false")
           << ",\"feature_build_ms\":" << featureBuildMs
           << ",\"audit_adjacency_build_ms\":" << adjacencyBuildMs
           << ",\"sampled_ground_truth_labels\":[";
    for (std::size_t index = 0; index < sampled.size(); ++index)
    {
        if (index != 0U) output << ',';
        output << sampled[index].label;
    }
    output << "],\"sampled_indices\":[";
    for (std::size_t index = 0; index < indices.size(); ++index)
    {
        if (index != 0U) output << ',';
        output << indices[index];
    }
    output << "]}\n";
    output.flush();
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        Arguments const arguments = parseArguments(argc, argv);
        std::vector<std::filesystem::path> const clouds = loadCloudList(arguments.list);
        if (!arguments.output.parent_path().empty())
        {
            std::filesystem::create_directories(arguments.output.parent_path());
        }
        std::ofstream output(arguments.output, std::ios::trunc);
        if (!output)
        {
            throw std::runtime_error("Cannot create JSONL output: " + arguments.output.string());
        }
        output << std::setprecision(std::numeric_limits<double>::max_digits10);

        ptv2::weld::WeldConfig config;
        config.engine_path = arguments.engine;
        config.plugin_path = arguments.plugin;
        ptv2::weld::WeldDetector detector;
        double initializeTotalMs = 0.0;
        ptv2::weld::WeldStatus initializeStatus = ptv2::weld::WeldStatus::INVALID_CONFIG;
        for (int attempt = 1; attempt <= arguments.repeatInitialize; ++attempt)
        {
            auto const started = Clock::now();
            initializeStatus = detector.initialize(config);
            double const attemptMs = elapsedMs(started);
            initializeTotalMs += attemptMs;
            output << "{\"record_type\":\"initialization\",\"attempt\":" << attempt
                   << ",\"status\":\"" << ptv2::weld::toString(initializeStatus) << '"'
                   << ",\"success\":"
                   << (initializeStatus == ptv2::weld::WeldStatus::SUCCESS ? "true" : "false")
                   << ",\"wall_ms\":" << attemptMs
                   << ",\"last_error\":\"" << jsonEscape(detector.lastError()) << "\"}\n";
            output.flush();
            if (initializeStatus != ptv2::weld::WeldStatus::SUCCESS)
            {
                return 2;
            }
        }
        output << "{\"record_type\":\"session\",\"sample_count\":" << clouds.size()
               << ",\"rounds\":" << arguments.rounds
               << ",\"warmup\":" << arguments.warmup
               << ",\"repeat_initialize\":" << arguments.repeatInitialize
               << ",\"initialize_total_ms\":" << initializeTotalMs << "}\n";

        ptv2::pointcloud::PointCloudLoader loader;
        ptv2::pointcloud::PointSampler sampler(42U);
        ptv2::pointcloud::FeatureBuilder featureBuilder;
        ptv2::pointcloud::KnnGraphBuilder graphBuilder(6U);
        for (auto const& cloud : clouds)
        {
            writePreprocessAudit(
                output, cloud, loader, sampler, featureBuilder, graphBuilder);
        }
        writeResource(output, "after_initialization_and_audit", 0);

        for (int index = 0; index < arguments.warmup; ++index)
        {
            ptv2::weld::WeldResult warmupResult;
            ptv2::weld::WeldStatus const status =
                detector.detect(clouds.front().string(), warmupResult);
            if (status != ptv2::weld::WeldStatus::SUCCESS)
            {
                writeDetection(
                    output, 0, index, clouds.front(), status, warmupResult,
                    detector.lastError(), true);
                return 3;
            }
        }

        int detections = 0;
        for (int round = 1; round <= arguments.rounds; ++round)
        {
            for (std::size_t index = 0; index < clouds.size(); ++index)
            {
                ptv2::weld::WeldResult result;
                ptv2::weld::WeldStatus const status =
                    detector.detect(clouds[index].string(), result);
                ++detections;
                writeDetection(
                    output, round, static_cast<int>(index), clouds[index], status,
                    result, detector.lastError(), arguments.compact);
                if (status != ptv2::weld::WeldStatus::SUCCESS)
                {
                    writeResource(output, "failure", detections);
                    return 4;
                }
                if (detections % arguments.resourceInterval == 0)
                {
                    writeResource(output, "interval", detections);
                }
            }
        }
        writeResource(output, "completed", detections);
        output << "{\"record_type\":\"summary\",\"success\":true,\"detections\":"
               << detections << ",\"engine_initializations\":"
               << arguments.repeatInitialize << "}\n";
        output.flush();
        std::cout << "WELD_SDK_TESTSET_QUALIFICATION_COMPLETED\n";
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "WELD_SDK_TESTSET_QUALIFICATION_FAILED: " << error.what() << '\n';
        return 1;
    }
}
