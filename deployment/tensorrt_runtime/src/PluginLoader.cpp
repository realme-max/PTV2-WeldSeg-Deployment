#include "PluginLoader.h"

#include <NvInferPlugin.h>
#include <NvInferRuntimePlugin.h>

#include <Windows.h>

#include <filesystem>
#include <sstream>

namespace ptv2::runtime
{
namespace
{
constexpr char kPluginName[]{"VoxelUniqueCub"};
constexpr char kPluginVersion[]{"1"};
constexpr char kPluginNamespace[]{"com.tensorrt.ptv2.experimental"};

std::string windowsError(DWORD code)
{
    LPSTR buffer = nullptr;
    DWORD const size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, 0, reinterpret_cast<LPSTR>(&buffer), 0, nullptr);
    std::string message = size != 0 && buffer != nullptr ? std::string(buffer, size) : "unknown Windows error";
    if (buffer != nullptr)
    {
        LocalFree(buffer);
    }
    return message;
}
} // namespace

PluginLoader::PluginLoader(nvinfer1::ILogger& logger) noexcept : logger_(logger) {}

PluginLoader::~PluginLoader()
{
    unload();
}

bool PluginLoader::load(std::string const& pluginPath)
{
    unload();
    lastError_.clear();

    if (!initLibNvInferPlugins(static_cast<void*>(&logger_), ""))
    {
        lastError_ = "initLibNvInferPlugins returned false";
        return false;
    }

    std::filesystem::path const path(pluginPath);
    if (!std::filesystem::is_regular_file(path))
    {
        lastError_ = "Plugin DLL does not exist: " + pluginPath;
        return false;
    }

    HMODULE const module = LoadLibraryW(path.wstring().c_str());
    if (module == nullptr)
    {
        lastError_ = "LoadLibraryW failed: " + windowsError(GetLastError());
        return false;
    }
    module_ = module;

    using InitFunction = bool (*)() noexcept;
    auto const initFunction = reinterpret_cast<InitFunction>(GetProcAddress(module, "initVoxelUniqueCubPlugin"));
    runtimeCountFunction_ = reinterpret_cast<RuntimeCountFunction>(
        GetProcAddress(module, "getVoxelUniqueCubRuntimeCreationCount"));
    if (initFunction == nullptr || runtimeCountFunction_ == nullptr)
    {
        lastError_ = "Required VoxelUniqueCub plugin exports were not found";
        unload();
        return false;
    }
    if (!initFunction())
    {
        lastError_ = "initVoxelUniqueCubPlugin returned false";
        unload();
        return false;
    }

    auto* registry = getPluginRegistry();
    if (registry == nullptr)
    {
        lastError_ = "TensorRT plugin registry is null";
        unload();
        return false;
    }
    int32_t count = 0;
    registry->getAllCreators(&count);
    registeredCreatorCount_ = count;
    creator_ = registry->getCreator(kPluginName, kPluginVersion, kPluginNamespace);
    if (!verify())
    {
        lastError_ = "VoxelUniqueCub creator was not registered with the required name/version/namespace";
        unload();
        return false;
    }
    return true;
}

bool PluginLoader::verify() const
{
    auto* registry = getPluginRegistry();
    return module_ != nullptr && registry != nullptr
        && registry->getCreator(kPluginName, kPluginVersion, kPluginNamespace) != nullptr;
}

void PluginLoader::unload() noexcept
{
    runtimeCountFunction_ = nullptr;
    registeredCreatorCount_ = 0;
    if (creator_ != nullptr)
    {
        if (auto* registry = getPluginRegistry(); registry != nullptr)
        {
            registry->deregisterCreator(*creator_);
        }
        creator_ = nullptr;
    }
    if (module_ != nullptr)
    {
        FreeLibrary(static_cast<HMODULE>(module_));
        module_ = nullptr;
    }
}

int32_t PluginLoader::registeredCreatorCount() const noexcept
{
    return registeredCreatorCount_;
}

int32_t PluginLoader::runtimeCreationCount() const noexcept
{
    return runtimeCountFunction_ != nullptr ? runtimeCountFunction_() : -1;
}

std::string const& PluginLoader::lastError() const noexcept
{
    return lastError_;
}

} // namespace ptv2::runtime
