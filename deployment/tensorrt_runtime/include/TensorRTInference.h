#pragma once

#include "CudaBufferManager.h"

#include <NvInferRuntime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ptv2::runtime
{

class PluginLoader;
class RuntimeErrorRecorder;
class TensorRTLogger;

struct InitializationTimings
{
    double totalMs{0.0};
    double pluginLoadMs{0.0};
    double deserializeMs{0.0};
    double contextCreationMs{0.0};
};

class TensorRTInference
{
public:
    TensorRTInference();
    ~TensorRTInference();

    TensorRTInference(TensorRTInference const&) = delete;
    TensorRTInference& operator=(TensorRTInference const&) = delete;

    bool initialize(
        std::string const& enginePath,
        std::string const& pluginPath,
        std::string const& expectedEngineSha256);

    bool infer(float const* points, float const* adj, float* logits);
    bool infer(
        float const* points, std::size_t pointsElements,
        float const* adj, std::size_t adjElements,
        float* logits, std::size_t logitsElements);

    void release() noexcept;

    std::string const& lastError() const noexcept;
    InitializationTimings const& initializationTimings() const noexcept;
    float lastInferenceDeviceMs() const noexcept;
    int32_t errorRecorderErrors() const noexcept;
    int32_t runtimePluginInstances() const noexcept;
    std::string const& engineSha256() const noexcept;
    std::string engineName() const;

private:
    bool validateEngineContract();
    bool fail(std::string message);

    std::unique_ptr<TensorRTLogger> logger_;
    std::unique_ptr<RuntimeErrorRecorder> errorRecorder_;
    std::unique_ptr<PluginLoader> pluginLoader_;
    nvinfer1::IRuntime* runtime_{nullptr};
    nvinfer1::ICudaEngine* engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};
    CudaBufferManager buffers_;
    cudaStream_t stream_{nullptr};
    cudaEvent_t startEvent_{nullptr};
    cudaEvent_t stopEvent_{nullptr};
    InitializationTimings timings_;
    float lastInferenceDeviceMs_{0.0F};
    int32_t runtimePluginInstances_{0};
    std::string engineSha256_;
    std::string lastError_;
};

} // namespace ptv2::runtime
