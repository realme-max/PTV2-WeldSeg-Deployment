#pragma once

#include <NvInfer.h>

#include <cstdint>
#include <string>

namespace ptv2::runtime
{

class PluginLoader
{
public:
    explicit PluginLoader(nvinfer1::ILogger& logger) noexcept;
    ~PluginLoader();

    PluginLoader(PluginLoader const&) = delete;
    PluginLoader& operator=(PluginLoader const&) = delete;

    bool load(std::string const& pluginPath);
    bool verify() const;
    void unload() noexcept;

    int32_t registeredCreatorCount() const noexcept;
    int32_t runtimeCreationCount() const noexcept;
    std::string const& lastError() const noexcept;

private:
    using RuntimeCountFunction = int32_t (*)() noexcept;

    nvinfer1::ILogger& logger_;
    void* module_{nullptr};
    nvinfer1::IPluginCreatorInterface* creator_{nullptr};
    RuntimeCountFunction runtimeCountFunction_{nullptr};
    int32_t registeredCreatorCount_{0};
    std::string lastError_;
};

} // namespace ptv2::runtime
