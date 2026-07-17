#include "TensorRTInference.h"

#include "PluginLoader.h"

#include <NvInferPlugin.h>

#include <Windows.h>
#include <bcrypt.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <utility>

namespace ptv2::runtime
{
namespace
{
using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point started)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

bool cudaOk(cudaError_t result, char const* operation, std::string& error)
{
    if (result == cudaSuccess)
    {
        return true;
    }
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorName(result) << " - " << cudaGetErrorString(result);
    error = message.str();
    return false;
}

std::vector<char> readBinaryFile(std::string const& path, std::string& error)
{
    std::ifstream input(std::filesystem::path(path), std::ios::binary | std::ios::ate);
    if (!input)
    {
        error = "Cannot open engine: " + path;
        return {};
    }
    std::streamsize const size = input.tellg();
    if (size <= 0)
    {
        error = "Engine is empty: " + path;
        return {};
    }
    input.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<std::size_t>(size));
    if (!input.read(bytes.data(), size))
    {
        error = "Failed to read complete engine: " + path;
        return {};
    }
    return bytes;
}

std::string sha256(std::vector<char> const& bytes, std::string& error)
{
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD objectLength = 0;
    DWORD hashLength = 0;
    DWORD copied = 0;
    std::vector<UCHAR> object;
    std::vector<UCHAR> digest;

    auto cleanup = [&]() {
        if (hash != nullptr)
        {
            BCryptDestroyHash(hash);
        }
        if (algorithm != nullptr)
        {
            BCryptCloseAlgorithmProvider(algorithm, 0);
        }
    };

    NTSTATUS status = BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0);
    if (status < 0)
    {
        error = "BCryptOpenAlgorithmProvider(SHA256) failed";
        cleanup();
        return {};
    }
    status = BCryptGetProperty(
        algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&objectLength), sizeof(objectLength), &copied, 0);
    if (status < 0)
    {
        error = "BCryptGetProperty(BCRYPT_OBJECT_LENGTH) failed";
        cleanup();
        return {};
    }
    status = BCryptGetProperty(
        algorithm, BCRYPT_HASH_LENGTH, reinterpret_cast<PUCHAR>(&hashLength), sizeof(hashLength), &copied, 0);
    if (status < 0)
    {
        error = "BCryptGetProperty(BCRYPT_HASH_LENGTH) failed";
        cleanup();
        return {};
    }
    object.resize(objectLength);
    digest.resize(hashLength);
    status = BCryptCreateHash(algorithm, &hash, object.data(), objectLength, nullptr, 0, 0);
    if (status < 0)
    {
        error = "BCryptCreateHash failed";
        cleanup();
        return {};
    }
    std::size_t offset = 0;
    while (offset < bytes.size())
    {
        std::size_t const remaining = bytes.size() - offset;
        ULONG const chunk = static_cast<ULONG>(std::min<std::size_t>(remaining, 1U << 30U));
        status = BCryptHashData(
            hash, reinterpret_cast<PUCHAR>(const_cast<char*>(bytes.data() + offset)), chunk, 0);
        if (status < 0)
        {
            error = "BCryptHashData failed";
            cleanup();
            return {};
        }
        offset += chunk;
    }
    status = BCryptFinishHash(hash, digest.data(), hashLength, 0);
    if (status < 0)
    {
        error = "BCryptFinishHash failed";
        cleanup();
        return {};
    }
    cleanup();

    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (UCHAR value : digest)
    {
        output << std::setw(2) << static_cast<unsigned>(value);
    }
    return output.str();
}

std::string lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char item) {
        return static_cast<char>(std::tolower(item));
    });
    return value;
}

std::string dimsString(nvinfer1::Dims const& dims)
{
    std::ostringstream output;
    output << '[';
    for (int32_t index = 0; index < dims.nbDims; ++index)
    {
        if (index != 0)
        {
            output << ',';
        }
        output << dims.d[index];
    }
    output << ']';
    return output.str();
}

bool dimsEqual(nvinfer1::Dims const& dims, std::initializer_list<int64_t> expected)
{
    if (dims.nbDims != static_cast<int32_t>(expected.size()))
    {
        return false;
    }
    int32_t index = 0;
    for (int64_t value : expected)
    {
        if (dims.d[index++] != value)
        {
            return false;
        }
    }
    return true;
}
} // namespace

class TensorRTLogger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, char const* message) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cerr << "[TensorRT] " << (message != nullptr ? message : "") << '\n';
        }
    }
};

class RuntimeErrorRecorder final : public nvinfer1::IErrorRecorder
{
public:
    int32_t getNbErrors() const noexcept override
    {
        std::lock_guard<std::mutex> guard(mutex_);
        return static_cast<int32_t>(errors_.size());
    }

    nvinfer1::ErrorCode getErrorCode(int32_t errorIdx) const noexcept override
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (errorIdx < 0 || static_cast<std::size_t>(errorIdx) >= errors_.size())
        {
            return nvinfer1::ErrorCode::kUNSPECIFIED_ERROR;
        }
        return errors_[static_cast<std::size_t>(errorIdx)].first;
    }

    ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept override
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (errorIdx < 0 || static_cast<std::size_t>(errorIdx) >= errors_.size())
        {
            return "";
        }
        return errors_[static_cast<std::size_t>(errorIdx)].second.c_str();
    }

    bool hasOverflowed() const noexcept override
    {
        std::lock_guard<std::mutex> guard(mutex_);
        return overflowed_;
    }

    void clear() noexcept override
    {
        std::lock_guard<std::mutex> guard(mutex_);
        errors_.clear();
        overflowed_ = false;
    }

    bool reportError(nvinfer1::ErrorCode value, ErrorDesc description) noexcept override
    {
        try
        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (errors_.size() >= kMaximumErrors)
            {
                overflowed_ = true;
            }
            else
            {
                errors_.emplace_back(value, description != nullptr ? description : "");
            }
        }
        catch (...)
        {
            overflowed_ = true;
        }
        return true;
    }

    RefCount incRefCount() noexcept override { return ++referenceCount_; }
    RefCount decRefCount() noexcept override { return --referenceCount_; }

    std::string summary() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        std::ostringstream output;
        for (std::size_t index = 0; index < errors_.size(); ++index)
        {
            if (index != 0)
            {
                output << " | ";
            }
            output << static_cast<int32_t>(errors_[index].first) << ':' << errors_[index].second;
        }
        if (overflowed_)
        {
            output << " | overflowed";
        }
        return output.str();
    }

private:
    static constexpr std::size_t kMaximumErrors{64};
    mutable std::mutex mutex_;
    std::vector<std::pair<nvinfer1::ErrorCode, std::string>> errors_;
    bool overflowed_{false};
    std::atomic<RefCount> referenceCount_{0};
};

TensorRTInference::TensorRTInference()
    : logger_(std::make_unique<TensorRTLogger>()),
      errorRecorder_(std::make_unique<RuntimeErrorRecorder>()),
      pluginLoader_(std::make_unique<PluginLoader>(*logger_))
{
}

TensorRTInference::~TensorRTInference()
{
    release();
}

bool TensorRTInference::initialize(
    std::string const& enginePath,
    std::string const& pluginPath,
    std::string const& expectedEngineSha256)
{
    release();
    lastError_.clear();
    timings_ = {};
    runtimePluginInstances_ = 0;
    auto const totalStarted = Clock::now();

    std::string readError;
    std::vector<char> const engineBytes = readBinaryFile(enginePath, readError);
    if (engineBytes.empty())
    {
        return fail(readError);
    }
    engineSha256_ = sha256(engineBytes, readError);
    if (engineSha256_.empty())
    {
        return fail(readError);
    }
    if (expectedEngineSha256.empty() || lower(expectedEngineSha256) != engineSha256_)
    {
        return fail("Engine SHA-256 mismatch: actual=" + engineSha256_ + ", expected=" + expectedEngineSha256);
    }

    std::string cudaError;
    if (!cudaOk(cudaSetDevice(0), "cudaSetDevice(0)", cudaError))
    {
        return fail(cudaError);
    }

    auto const pluginStarted = Clock::now();
    if (!pluginLoader_->load(pluginPath))
    {
        return fail("Plugin load/verification failed: " + pluginLoader_->lastError());
    }
    timings_.pluginLoadMs = elapsedMs(pluginStarted);

    auto* registry = getPluginRegistry();
    if (registry == nullptr)
    {
        return fail("TensorRT plugin registry is null after plugin loading");
    }
    registry->setErrorRecorder(errorRecorder_.get());

    runtime_ = nvinfer1::createInferRuntime(*logger_);
    if (runtime_ == nullptr)
    {
        return fail("createInferRuntime returned null");
    }
    runtime_->setErrorRecorder(errorRecorder_.get());

    int32_t const pluginCountBefore = pluginLoader_->runtimeCreationCount();
    auto const deserializeStarted = Clock::now();
    engine_ = runtime_->deserializeCudaEngine(engineBytes.data(), engineBytes.size());
    timings_.deserializeMs = elapsedMs(deserializeStarted);
    if (engine_ == nullptr)
    {
        return fail("deserializeCudaEngine returned null; " + errorRecorder_->summary());
    }
    engine_->setErrorRecorder(errorRecorder_.get());
    runtimePluginInstances_ = pluginLoader_->runtimeCreationCount() - pluginCountBefore;
    if (runtimePluginInstances_ != 4)
    {
        return fail("Expected four VoxelUniqueCub runtime plugin instances, got "
            + std::to_string(runtimePluginInstances_));
    }

    auto const contextStarted = Clock::now();
    context_ = engine_->createExecutionContext();
    timings_.contextCreationMs = elapsedMs(contextStarted);
    if (context_ == nullptr)
    {
        return fail("createExecutionContext returned null; " + errorRecorder_->summary());
    }
    context_->setErrorRecorder(errorRecorder_.get());

    if (!validateEngineContract())
    {
        return false;
    }
    if (!cudaOk(cudaStreamCreate(&stream_), "cudaStreamCreate", cudaError)
        || !cudaOk(cudaEventCreate(&startEvent_), "cudaEventCreate(start)", cudaError)
        || !cudaOk(cudaEventCreate(&stopEvent_), "cudaEventCreate(stop)", cudaError))
    {
        return fail(cudaError);
    }
    if (!buffers_.allocate(cudaError))
    {
        return fail(cudaError);
    }
    if (!context_->setTensorAddress("points", buffers_.points())
        || !context_->setTensorAddress("adj", buffers_.adj())
        || !context_->setTensorAddress("logits", buffers_.logits()))
    {
        return fail("setTensorAddress failed for one or more fixed I/O tensors");
    }

    timings_.totalMs = elapsedMs(totalStarted);
    std::cout << "TensorRT version: " << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.'
              << NV_TENSORRT_PATCH << '.' << NV_TENSORRT_BUILD << '\n';
    std::cout << "Engine name: " << engineName() << '\n';
    std::cout << "Inputs: points [1,2048,4] FP32; adj [1,2048,2048] FP32\n";
    std::cout << "Output: logits [1,2048,2] FP32\n";
    std::cout << "Registered plugin creators: " << pluginLoader_->registeredCreatorCount() << '\n';
    std::cout << "VoxelUniqueCub runtime plugin instances: " << runtimePluginInstances_ << '\n';
    return true;
}

bool TensorRTInference::validateEngineContract()
{
    if (engine_->getNbIOTensors() != 3)
    {
        return fail("Engine I/O count mismatch: " + std::to_string(engine_->getNbIOTensors()) + " != 3");
    }
    struct Contract
    {
        char const* name;
        nvinfer1::TensorIOMode mode;
        std::initializer_list<int64_t> shape;
    };
    Contract const contracts[]{
        {"points", nvinfer1::TensorIOMode::kINPUT, {1, 2048, 4}},
        {"adj", nvinfer1::TensorIOMode::kINPUT, {1, 2048, 2048}},
        {"logits", nvinfer1::TensorIOMode::kOUTPUT, {1, 2048, 2}},
    };
    for (Contract const& contract : contracts)
    {
        nvinfer1::Dims const shape = engine_->getTensorShape(contract.name);
        if (engine_->getTensorIOMode(contract.name) != contract.mode
            || engine_->getTensorDataType(contract.name) != nvinfer1::DataType::kFLOAT
            || !dimsEqual(shape, contract.shape))
        {
            return fail(std::string("Engine tensor contract mismatch for ") + contract.name
                + ": shape=" + dimsString(shape));
        }
    }
    return true;
}

bool TensorRTInference::infer(float const* points, float const* adj, float* logits)
{
    return infer(
        points, CudaBufferManager::kPointsElements,
        adj, CudaBufferManager::kAdjElements,
        logits, CudaBufferManager::kLogitsElements);
}

bool TensorRTInference::infer(
    float const* points, std::size_t pointsElements,
    float const* adj, std::size_t adjElements,
    float* logits, std::size_t logitsElements)
{
    if (context_ == nullptr || stream_ == nullptr)
    {
        return fail("TensorRTInference is not initialized");
    }
    errorRecorder_->clear();
    std::string cudaError;
    if (!buffers_.copyInput(points, pointsElements, adj, adjElements, stream_, cudaError))
    {
        return fail(cudaError);
    }
    if (!cudaOk(cudaEventRecord(startEvent_, stream_), "cudaEventRecord(start)", cudaError))
    {
        return fail(cudaError);
    }
    if (!context_->enqueueV3(stream_))
    {
        return fail("enqueueV3 returned false; " + errorRecorder_->summary());
    }
    if (!cudaOk(cudaEventRecord(stopEvent_, stream_), "cudaEventRecord(stop)", cudaError)
        || !buffers_.copyOutput(logits, logitsElements, stream_, cudaError)
        || !cudaOk(cudaStreamSynchronize(stream_), "cudaStreamSynchronize", cudaError)
        || !cudaOk(cudaEventElapsedTime(&lastInferenceDeviceMs_, startEvent_, stopEvent_),
            "cudaEventElapsedTime", cudaError))
    {
        return fail(cudaError);
    }
    if (errorRecorder_->getNbErrors() != 0)
    {
        return fail("TensorRT ErrorRecorder reported: " + errorRecorder_->summary());
    }
    return true;
}

void TensorRTInference::release() noexcept
{
    buffers_.release();
    if (stopEvent_ != nullptr)
    {
        cudaEventDestroy(stopEvent_);
        stopEvent_ = nullptr;
    }
    if (startEvent_ != nullptr)
    {
        cudaEventDestroy(startEvent_);
        startEvent_ = nullptr;
    }
    if (stream_ != nullptr)
    {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    delete context_;
    context_ = nullptr;
    delete engine_;
    engine_ = nullptr;
    delete runtime_;
    runtime_ = nullptr;
    if (auto* registry = getPluginRegistry(); registry != nullptr)
    {
        registry->setErrorRecorder(nullptr);
    }
    if (pluginLoader_ != nullptr)
    {
        pluginLoader_->unload();
    }
}

bool TensorRTInference::fail(std::string message)
{
    lastError_ = std::move(message);
    std::cerr << "TENSORRT_RUNTIME_ERROR: " << lastError_ << '\n';
    return false;
}

std::string const& TensorRTInference::lastError() const noexcept { return lastError_; }
InitializationTimings const& TensorRTInference::initializationTimings() const noexcept { return timings_; }
float TensorRTInference::lastInferenceDeviceMs() const noexcept { return lastInferenceDeviceMs_; }
int32_t TensorRTInference::errorRecorderErrors() const noexcept
{
    return errorRecorder_ != nullptr ? errorRecorder_->getNbErrors() : 0;
}
int32_t TensorRTInference::runtimePluginInstances() const noexcept { return runtimePluginInstances_; }
std::string const& TensorRTInference::engineSha256() const noexcept { return engineSha256_; }
std::string TensorRTInference::engineName() const
{
    return engine_ != nullptr && engine_->getName() != nullptr ? engine_->getName() : "";
}

} // namespace ptv2::runtime
