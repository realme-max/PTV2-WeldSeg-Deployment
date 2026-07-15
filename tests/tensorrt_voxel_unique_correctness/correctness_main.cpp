#include "VoxelUniqueCorrectnessPlugin.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ptv2::voxel_unique_correctness
{

constexpr int32_t kMaxN{2048};
constexpr uint64_t kValuesCapacityBytes{64ULL * 1024ULL};

class Logger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, char const* message) noexcept override
    {
        if (severity <= Severity::kINFO)
        {
            std::cout << "[TensorRT] " << message << '\n';
        }
    }
};

template <typename T>
struct TrtDeleter
{
    void operator()(T* object) const noexcept { delete object; }
};
template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

void checkCuda(cudaError_t status, char const* expression)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(status));
    }
}
#define CUDA_CHECK(expression) checkCuda((expression), #expression)

struct Reference
{
    std::vector<int64_t> values;
    std::vector<int64_t> inverse;
};

Reference cpuUniqueReference(std::vector<int64_t> const& keys)
{
    Reference result;
    result.values = keys;
    std::sort(result.values.begin(), result.values.end());
    result.values.erase(
        std::unique(result.values.begin(), result.values.end()), result.values.end());
    result.inverse.reserve(keys.size());
    for (int64_t key : keys)
    {
        auto const it = std::lower_bound(result.values.begin(), result.values.end(), key);
        result.inverse.push_back(static_cast<int64_t>(std::distance(result.values.begin(), it)));
    }
    return result;
}

struct TestCase
{
    std::string name;
    std::vector<int64_t> keys;
};

std::vector<TestCase> makeTestCases()
{
    std::vector<TestCase> cases;
    for (int32_t n : {4, 8, 32, 2048})
    {
        std::mt19937_64 random(static_cast<uint64_t>(42 + n));
        int64_t const radius = std::max<int64_t>(2, n / 3);
        uint64_t const span = static_cast<uint64_t>(2 * radius + 1);
        for (int32_t repetition = 0; repetition < 3; ++repetition)
        {
            std::vector<int64_t> keys(static_cast<size_t>(n));
            for (int64_t& key : keys)
            {
                key = static_cast<int64_t>(random() % span) - radius;
            }
            cases.push_back({
                "random_n" + std::to_string(n) + "_r" + std::to_string(repetition),
                std::move(keys)});
        }
    }

    cases.push_back({"all_same", std::vector<int64_t>(64, -7)});

    std::vector<int64_t> allUnique(64);
    for (int32_t i = 0; i < 64; ++i) allUnique[static_cast<size_t>(i)] = i * 17 - 500;
    std::mt19937 shuffleRandom(42);
    std::shuffle(allUnique.begin(), allUnique.end(), shuffleRandom);
    cases.push_back({"all_unique", allUnique});

    std::vector<int64_t> sorted(64);
    for (int32_t i = 0; i < 64; ++i) sorted[static_cast<size_t>(i)] = i / 3 - 10;
    cases.push_back({"sorted", sorted});
    std::reverse(sorted.begin(), sorted.end());
    cases.push_back({"reversed", sorted});

    std::vector<int64_t> groups;
    for (auto const [value, count] : std::vector<std::pair<int64_t, int32_t>>{
             {9, 5}, {-3, 11}, {42, 7}, {0, 13}, {-3, 9}, {9, 19}})
    {
        groups.insert(groups.end(), static_cast<size_t>(count), value);
    }
    cases.push_back({"repeated_groups", groups});
    cases.push_back({"int64_extremes", {
        std::numeric_limits<int64_t>::min(), 0, std::numeric_limits<int64_t>::max(),
        -1, 0, std::numeric_limits<int64_t>::min(), 1, std::numeric_limits<int64_t>::max()}});
    return cases;
}

class ValuesOutputAllocator final : public nvinfer1::IOutputAllocator
{
public:
    ValuesOutputAllocator(void* memory, uint64_t capacity)
        : mMemory(memory), mCapacity(capacity) {}

    void reset() noexcept
    {
        mRequestedBytes = 0;
        mShapeNotified = false;
        mShape = nvinfer1::Dims{};
    }

    void* reallocateOutputAsync(
        char const* tensorName, void* currentMemory, uint64_t size,
        uint64_t alignment, cudaStream_t stream) override
    {
        (void) tensorName;
        (void) alignment;
        (void) stream;
        mRequestedBytes = size;
        if ((currentMemory == nullptr || currentMemory == mMemory) && size <= mCapacity)
        {
            return mMemory;
        }
        return nullptr;
    }

    void notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept override
    {
        (void) tensorName;
        mShape = dims;
        mShapeNotified = true;
    }

    uint64_t requestedBytes() const noexcept { return mRequestedBytes; }
    bool shapeNotified() const noexcept { return mShapeNotified; }
    nvinfer1::Dims shape() const noexcept { return mShape; }

private:
    void* mMemory{};
    uint64_t mCapacity{};
    uint64_t mRequestedBytes{};
    bool mShapeNotified{};
    nvinfer1::Dims mShape{};
};

struct CaseResult
{
    TestCase test;
    Reference reference;
    int32_t pluginCount{};
    std::vector<int64_t> pluginValues;
    std::vector<int64_t> pluginInverse;
    int64_t runtimeM{-1};
    uint64_t allocatorBytes{};
    bool valuesMatch{};
    bool inverseMatch{};
    bool countMatch{};
    bool shapeMatch{};
    bool passed{};
};

template <typename T>
void writeVector(std::ostream& stream, std::vector<T> const& values)
{
    stream << '[';
    for (size_t i = 0; i < values.size(); ++i)
    {
        if (i != 0) stream << ',';
        stream << values[i];
    }
    stream << ']';
}

void writeComparison(std::string const& path, std::vector<CaseResult> const& results)
{
    std::ofstream output(path);
    output << "{\n  \"algorithm\": \"serial_cuda_reference_implementation\",\n"
           << "  \"sorted\": true,\n  \"case_count\": " << results.size() << ",\n"
           << "  \"all_passed\": "
           << (std::all_of(results.begin(), results.end(), [](auto const& item) { return item.passed; }) ? "true" : "false")
           << ",\n  \"cases\": [\n";
    for (size_t index = 0; index < results.size(); ++index)
    {
        auto const& item = results[index];
        output << "    {\n      \"name\": \"" << item.test.name << "\",\n"
               << "      \"n\": " << item.test.keys.size() << ",\n"
               << "      \"keys\": ";
        writeVector(output, item.test.keys);
        output << ",\n      \"cpu_reference_count\": " << item.reference.values.size()
               << ",\n      \"cpu_reference_values\": ";
        writeVector(output, item.reference.values);
        output << ",\n      \"cpu_reference_inverse\": ";
        writeVector(output, item.reference.inverse);
        output << ",\n      \"plugin_count\": " << item.pluginCount
               << ",\n      \"plugin_values\": ";
        writeVector(output, item.pluginValues);
        output << ",\n      \"plugin_inverse\": ";
        writeVector(output, item.pluginInverse);
        output << ",\n      \"runtime_m\": " << item.runtimeM
               << ",\n      \"allocator_requested_bytes\": " << item.allocatorBytes
               << ",\n      \"values_match\": " << (item.valuesMatch ? "true" : "false")
               << ",\n      \"inverse_match\": " << (item.inverseMatch ? "true" : "false")
               << ",\n      \"count_match\": " << (item.countMatch ? "true" : "false")
               << ",\n      \"shape_match\": " << (item.shapeMatch ? "true" : "false")
               << ",\n      \"passed\": " << (item.passed ? "true" : "false") << "\n    }";
        output << (index + 1 == results.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

} // namespace ptv2::voxel_unique_correctness

int main(int argc, char** argv)
{
    using namespace ptv2::voxel_unique_correctness;
    if (argc != 4)
    {
        std::cerr << "Usage: voxel_unique_correctness <custom.onnx> <engine.plan> <comparison.json>\n";
        return 64;
    }

    try
    {
        Logger logger;
        VoxelUniquePluginCreator creator;
        auto* registry = ::getPluginRegistry();
        if (registry == nullptr || !registry->registerCreator(creator, kPluginNamespace))
        {
            throw std::runtime_error("Plugin Creator registration failed");
        }

        TrtUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
        TrtUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(0U)};
        TrtUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
        if (!builder || !network || !config) throw std::runtime_error("TensorRT builder setup failed");
        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
        if (parser == nullptr || !parser->parseFromFile(argv[1], static_cast<int32_t>(nvinfer1::ILogger::Severity::kERROR)))
        {
            if (parser != nullptr)
            {
                for (int32_t i = 0; i < parser->getNbErrors(); ++i)
                    std::cerr << "PARSER_ERROR=" << parser->getError(i)->desc() << '\n';
            }
            throw std::runtime_error("Custom correctness ONNX parse failed");
        }

        auto* profile = builder->createOptimizationProfile();
        nvinfer1::Dims minDims{1, {1}};
        nvinfer1::Dims optDims{1, {32}};
        nvinfer1::Dims maxDims{1, {kMaxN}};
        if (profile == nullptr
            || !profile->setDimensions("voxel_key", nvinfer1::OptProfileSelector::kMIN, minDims)
            || !profile->setDimensions("voxel_key", nvinfer1::OptProfileSelector::kOPT, optDims)
            || !profile->setDimensions("voxel_key", nvinfer1::OptProfileSelector::kMAX, maxDims)
            || config->addOptimizationProfile(profile) < 0)
        {
            throw std::runtime_error("Optimization profile setup failed");
        }
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 256ULL * 1024ULL * 1024ULL);
        TrtUniquePtr<nvinfer1::IHostMemory> serialized{builder->buildSerializedNetwork(*network, *config)};
        if (!serialized) throw std::runtime_error("Correctness engine build failed");
        std::ofstream plan(argv[2], std::ios::binary);
        plan.write(static_cast<char const*>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
        plan.close();

        TrtUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
        TrtUniquePtr<nvinfer1::ICudaEngine> engine{
            runtime->deserializeCudaEngine(serialized->data(), serialized->size())};
        TrtUniquePtr<nvinfer1::IExecutionContext> context{engine ? engine->createExecutionContext() : nullptr};
        if (!runtime || !engine || !context) throw std::runtime_error("Engine deserialize/context creation failed");

        int64_t* deviceKeys{};
        int32_t* deviceCount{};
        int64_t* deviceValues{};
        int64_t* deviceInverse{};
        cudaStream_t stream{};
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceKeys), kMaxN * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceCount), sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceValues), kValuesCapacityBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceInverse), kMaxN * sizeof(int64_t)));
        ValuesOutputAllocator allocator(deviceValues, kValuesCapacityBytes);
        if (!context->setTensorAddress("voxel_key", deviceKeys)
            || !context->setTensorAddress("voxel_count", deviceCount)
            || !context->setTensorAddress("unique_values", deviceValues)
            || !context->setTensorAddress("inverse_indices", deviceInverse)
            || !context->setOutputAllocator("unique_values", &allocator))
        {
            throw std::runtime_error("Tensor address setup failed");
        }

        std::vector<CaseResult> results;
        for (TestCase const& test : makeTestCases())
        {
            if (test.keys.empty() || test.keys.size() > static_cast<size_t>(kMaxN))
                throw std::runtime_error("Invalid test size");
            Reference reference = cpuUniqueReference(test.keys);
            nvinfer1::Dims inputDims{1, {static_cast<int64_t>(test.keys.size())}};
            if (!context->setInputShape("voxel_key", inputDims))
                throw std::runtime_error("setInputShape failed for " + test.name);
            allocator.reset();
            CUDA_CHECK(cudaMemcpyAsync(deviceKeys, test.keys.data(), test.keys.size() * sizeof(int64_t),
                cudaMemcpyHostToDevice, stream));
            if (!context->enqueueV3(stream))
                throw std::runtime_error("enqueueV3 failed for " + test.name);

            int32_t pluginCount{-1};
            std::vector<int64_t> values(test.keys.size());
            std::vector<int64_t> inverse(test.keys.size());
            CUDA_CHECK(cudaMemcpyAsync(&pluginCount, deviceCount, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(values.data(), deviceValues, values.size() * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(inverse.data(), deviceInverse, inverse.size() * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (pluginCount < 1 || pluginCount > static_cast<int32_t>(test.keys.size()))
                throw std::runtime_error("Invalid plugin count for " + test.name);
            values.resize(static_cast<size_t>(pluginCount));
            int64_t runtimeM = -1;
            if (allocator.shapeNotified() && allocator.shape().nbDims == 1)
                runtimeM = allocator.shape().d[0];

            CaseResult item;
            item.test = test;
            item.reference = std::move(reference);
            item.pluginCount = pluginCount;
            item.pluginValues = std::move(values);
            item.pluginInverse = std::move(inverse);
            item.runtimeM = runtimeM;
            item.allocatorBytes = allocator.requestedBytes();
            item.valuesMatch = item.pluginValues == item.reference.values;
            item.inverseMatch = item.pluginInverse == item.reference.inverse;
            item.countMatch = static_cast<size_t>(pluginCount) == item.reference.values.size();
            item.shapeMatch = runtimeM == pluginCount;
            item.passed = item.valuesMatch && item.inverseMatch && item.countMatch && item.shapeMatch;
            std::cout << "CASE name=" << test.name << " n=" << test.keys.size()
                      << " m=" << pluginCount << " passed=" << std::boolalpha << item.passed << '\n';
            results.push_back(std::move(item));
        }

        writeComparison(argv[3], results);
        bool const allPassed = std::all_of(
            results.begin(), results.end(), [](CaseResult const& item) { return item.passed; });
        std::cout << "CASE_COUNT=" << results.size() << '\n';
        std::cout << "CREATOR_BUILD_CALLS=" << getBuildCreationCount() << '\n';
        std::cout << "CREATOR_RUNTIME_CALLS=" << getRuntimeCreationCount() << '\n';

        CUDA_CHECK(cudaFree(deviceInverse));
        CUDA_CHECK(cudaFree(deviceValues));
        CUDA_CHECK(cudaFree(deviceCount));
        CUDA_CHECK(cudaFree(deviceKeys));
        CUDA_CHECK(cudaStreamDestroy(stream));

        if (!allPassed)
        {
            std::cerr << "VOXEL_UNIQUE_PLUGIN_CORRECTNESS_FAILED\n";
            return 2;
        }
        std::cout << "VOXEL_UNIQUE_PLUGIN_CORRECTNESS_PASSED\n";
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "ERROR=" << error.what() << '\n';
        std::cerr << "VOXEL_UNIQUE_PLUGIN_CORRECTNESS_FAILED\n";
        return 1;
    }
}
