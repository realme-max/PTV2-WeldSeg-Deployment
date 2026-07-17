#include "CudaBufferManager.h"
#include "TensorRTInference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
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
    std::string points;
    std::string adj;
    std::string output;
    std::string engineSha256;
    std::string benchmarkJson;
    std::string runtimeJson;
    int warmup{0};
    int iterations{0};
};

std::string jsonEscape(std::string const& value)
{
    std::ostringstream output;
    for (char item : value)
    {
        switch (item)
        {
        case '\\': output << "\\\\"; break;
        case '"': output << "\\\""; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default: output << item; break;
        }
    }
    return output.str();
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
    Arguments result;
    result.engine = required("--engine");
    result.plugin = required("--plugin");
    result.points = required("--points");
    result.adj = required("--adj");
    result.output = required("--output");
    result.engineSha256 = required("--engine-sha256");
    if (values.count("--benchmark-json") != 0)
    {
        result.benchmarkJson = values["--benchmark-json"];
    }
    if (values.count("--runtime-json") != 0)
    {
        result.runtimeJson = values["--runtime-json"];
    }
    if (values.count("--warmup") != 0)
    {
        result.warmup = std::stoi(values["--warmup"]);
    }
    if (values.count("--iterations") != 0)
    {
        result.iterations = std::stoi(values["--iterations"]);
    }
    if ((result.warmup < 0 || result.iterations < 0)
        || (result.benchmarkJson.empty() != (result.iterations == 0)))
    {
        throw std::runtime_error("Benchmark requires --benchmark-json, --iterations > 0, and --warmup >= 0");
    }
    return result;
}

std::vector<float> readFloatBinary(std::string const& path, std::size_t expectedElements)
{
    std::filesystem::path const file(path);
    if (!std::filesystem::is_regular_file(file))
    {
        throw std::runtime_error("Input does not exist: " + path);
    }
    std::uintmax_t const expectedBytes = expectedElements * sizeof(float);
    if (std::filesystem::file_size(file) != expectedBytes)
    {
        throw std::runtime_error("Input size mismatch for " + path + ": expected "
            + std::to_string(expectedBytes) + " bytes, got " + std::to_string(std::filesystem::file_size(file)));
    }
    std::vector<float> values(expectedElements);
    std::ifstream input(file, std::ios::binary);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(expectedBytes));
    if (!input)
    {
        throw std::runtime_error("Failed to read complete input: " + path);
    }
    return values;
}

void writeFloatBinary(std::string const& path, std::vector<float> const& values)
{
    std::filesystem::path const file(path);
    if (!file.parent_path().empty())
    {
        std::filesystem::create_directories(file.parent_path());
    }
    std::ofstream output(file, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<char const*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output)
    {
        throw std::runtime_error("Failed to write output: " + path);
    }
}

double percentile(std::vector<double> values, double q)
{
    std::sort(values.begin(), values.end());
    double const position = q * static_cast<double>(values.size() - 1);
    std::size_t const lower = static_cast<std::size_t>(std::floor(position));
    std::size_t const upper = static_cast<std::size_t>(std::ceil(position));
    double const fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

struct Statistics
{
    double mean{};
    double p50{};
    double p95{};
    double minimum{};
    double maximum{};
};

Statistics statistics(std::vector<double> const& values)
{
    Statistics result;
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    result.p50 = percentile(values, 0.50);
    result.p95 = percentile(values, 0.95);
    auto const bounds = std::minmax_element(values.begin(), values.end());
    result.minimum = *bounds.first;
    result.maximum = *bounds.second;
    return result;
}

void writeBenchmark(
    std::string const& path,
    ptv2::runtime::InitializationTimings const& init,
    int warmup, int iterations,
    std::vector<double> const& device,
    std::vector<double> const& endToEnd)
{
    Statistics const deviceStats = statistics(device);
    Statistics const e2eStats = statistics(endToEnd);
    std::filesystem::path const file(path);
    std::filesystem::create_directories(file.parent_path());
    std::ofstream output(file);
    output << std::setprecision(10)
           << "{\n"
           << "  \"scope\": \"Phase 9A simple validation; not the Phase 8D benchmark\",\n"
           << "  \"warmup\": " << warmup << ",\n"
           << "  \"iterations\": " << iterations << ",\n"
           << "  \"init_ms\": " << init.totalMs << ",\n"
           << "  \"plugin_load_ms\": " << init.pluginLoadMs << ",\n"
           << "  \"deserialize_ms\": " << init.deserializeMs << ",\n"
           << "  \"context_creation_ms\": " << init.contextCreationMs << ",\n"
           << "  \"cuda_event_inference_ms\": {\"mean\": " << deviceStats.mean
           << ", \"p50\": " << deviceStats.p50 << ", \"p95\": " << deviceStats.p95
           << ", \"min\": " << deviceStats.minimum << ", \"max\": " << deviceStats.maximum << "},\n"
           << "  \"host_e2e_ms\": {\"mean\": " << e2eStats.mean
           << ", \"p50\": " << e2eStats.p50 << ", \"p95\": " << e2eStats.p95
           << ", \"min\": " << e2eStats.minimum << ", \"max\": " << e2eStats.maximum << "}\n"
           << "}\n";
}

void writeRuntimeSummary(
    std::string const& path,
    Arguments const& args,
    ptv2::runtime::TensorRTInference const& runtime,
    std::vector<float> const& logits)
{
    if (path.empty())
    {
        return;
    }
    std::filesystem::path const file(path);
    std::filesystem::create_directories(file.parent_path());
    auto const bounds = std::minmax_element(logits.begin(), logits.end());
    double const mean = std::accumulate(logits.begin(), logits.end(), 0.0) / static_cast<double>(logits.size());
    bool const finite = std::all_of(logits.begin(), logits.end(), [](float value) { return std::isfinite(value); });
    std::ofstream output(file);
    output << std::setprecision(10)
           << "{\n"
           << "  \"status\": \"PASS\",\n"
           << "  \"engine\": \"" << jsonEscape(args.engine) << "\",\n"
           << "  \"plugin\": \"" << jsonEscape(args.plugin) << "\",\n"
           << "  \"engine_sha256\": \"" << runtime.engineSha256() << "\",\n"
           << "  \"engine_name\": \"" << jsonEscape(runtime.engineName()) << "\",\n"
           << "  \"runtime_plugin_instances\": " << runtime.runtimePluginInstances() << ",\n"
           << "  \"error_recorder_errors\": " << runtime.errorRecorderErrors() << ",\n"
           << "  \"output_shape\": [1, 2048, 2],\n"
           << "  \"output_dtype\": \"float32\",\n"
           << "  \"output_finite\": " << (finite ? "true" : "false") << ",\n"
           << "  \"logits_min\": " << *bounds.first << ",\n"
           << "  \"logits_max\": " << *bounds.second << ",\n"
           << "  \"logits_mean\": " << mean << "\n"
           << "}\n";
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        Arguments const args = parseArguments(argc, argv);
        std::vector<float> const points = readFloatBinary(args.points, ptv2::runtime::CudaBufferManager::kPointsElements);
        std::vector<float> const adj = readFloatBinary(args.adj, ptv2::runtime::CudaBufferManager::kAdjElements);
        std::vector<float> logits(ptv2::runtime::CudaBufferManager::kLogitsElements);

        ptv2::runtime::TensorRTInference runtime;
        if (!runtime.initialize(args.engine, args.plugin, args.engineSha256))
        {
            throw std::runtime_error(runtime.lastError());
        }
        if (!runtime.infer(points.data(), adj.data(), logits.data()))
        {
            throw std::runtime_error(runtime.lastError());
        }
        if (!std::all_of(logits.begin(), logits.end(), [](float value) { return std::isfinite(value); }))
        {
            throw std::runtime_error("TensorRT output contains NaN/Inf");
        }
        writeFloatBinary(args.output, logits);

        if (args.iterations > 0)
        {
            for (int index = 0; index < args.warmup; ++index)
            {
                if (!runtime.infer(points.data(), adj.data(), logits.data()))
                {
                    throw std::runtime_error(runtime.lastError());
                }
            }
            std::vector<double> device;
            std::vector<double> endToEnd;
            device.reserve(static_cast<std::size_t>(args.iterations));
            endToEnd.reserve(static_cast<std::size_t>(args.iterations));
            for (int index = 0; index < args.iterations; ++index)
            {
                auto const started = Clock::now();
                if (!runtime.infer(points.data(), adj.data(), logits.data()))
                {
                    throw std::runtime_error(runtime.lastError());
                }
                endToEnd.push_back(std::chrono::duration<double, std::milli>(Clock::now() - started).count());
                device.push_back(runtime.lastInferenceDeviceMs());
            }
            writeBenchmark(args.benchmarkJson, runtime.initializationTimings(), args.warmup, args.iterations, device, endToEnd);
            writeFloatBinary(args.output, logits);
        }
        writeRuntimeSummary(args.runtimeJson, args, runtime, logits);
        std::cout << "TENSORRT_CPP_RUNTIME_INFERENCE_PASSED\n";
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "TENSORRT_CPP_RUNTIME_FAILED: " << error.what() << '\n';
        return 1;
    }
}
