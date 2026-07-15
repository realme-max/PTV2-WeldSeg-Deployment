#include "VoxelUniquePlugin.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <NvOnnxParser.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ptv2::trt_parser_test
{

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
    void operator()(T* object) const noexcept
    {
        delete object;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

std::string dimsToString(nvinfer1::Dims const& dims)
{
    std::ostringstream value;
    value << '[';
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        if (i != 0)
        {
            value << ',';
        }
        value << dims.d[i];
    }
    value << ']';
    return value.str();
}

void printParserErrors(nvonnxparser::IParser const& parser)
{
    for (int32_t i = 0; i < parser.getNbErrors(); ++i)
    {
        auto const* error = parser.getError(i);
        std::cerr << "PARSER_ERROR index=" << i
                  << " code=" << static_cast<int32_t>(error->code())
                  << " node=" << error->node()
                  << " node_name=" << error->nodeName()
                  << " op=" << error->nodeOperator()
                  << " description=" << error->desc() << '\n';
    }
}

} // namespace ptv2::trt_parser_test

int main(int argc, char** argv)
{
    using namespace ptv2::trt_parser_test;
    using namespace ptv2::trt_prototype;

    if (argc != 3)
    {
        std::cerr << "Usage: custom_voxel_unique_parser <custom.onnx> <engine.plan>\n";
        return 64;
    }

    try
    {
        std::string const onnxPath = argv[1];
        std::string const enginePath = argv[2];
        Logger logger;
        auto* registry = ::getPluginRegistry();
        if (registry == nullptr)
        {
            throw std::runtime_error("getPluginRegistry returned null");
        }

        VoxelUniquePluginCreator creator;
        bool const registered = registry->registerCreator(creator, kPluginNamespace);
        std::cout << "PLUGIN_CREATOR_REGISTERED=" << std::boolalpha << registered << '\n';
        auto* lookedUp = registry->getCreator(kPluginName, kPluginVersion, kPluginNamespace);
        bool const lookupPassed = lookedUp == static_cast<nvinfer1::IPluginCreatorInterface*>(&creator);
        std::cout << "PLUGIN_CREATOR_LOOKUP_PASSED=" << lookupPassed << '\n';
        std::cout << "PLUGIN_NAME=" << kPluginName << '\n';
        std::cout << "PLUGIN_VERSION=" << kPluginVersion << '\n';
        std::cout << "PLUGIN_NAMESPACE=" << kPluginNamespace << '\n';
        if (!registered || !lookupPassed)
        {
            throw std::runtime_error("Plugin Creator registration/lookup failed");
        }

        TrtUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
        if (!builder)
        {
            throw std::runtime_error("createInferBuilder failed");
        }
        TrtUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(0U)};
        TrtUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
        if (!network || !config)
        {
            throw std::runtime_error("network/config creation failed");
        }
        config->setMemoryPoolLimit(
            nvinfer1::MemoryPoolType::kWORKSPACE, 256ULL * 1024ULL * 1024ULL);

        // Deliberately use ONNX Parser. No manual addPluginV3 call exists here.
        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
        if (parser == nullptr)
        {
            throw std::runtime_error("createParser failed");
        }
        bool const parserClaimsSupport = parser->supportsOperator(kPluginName);
        std::cout << "SUPPORTS_OPERATOR_HINT=" << parserClaimsSupport << '\n';
        bool const parsePassed = parser->parseFromFile(
            onnxPath.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kVERBOSE));
        std::cout << "PARSER_ERROR_COUNT=" << parser->getNbErrors() << '\n';
        if (!parsePassed)
        {
            printParserErrors(*parser);
            std::cerr << "CUSTOM_ONNX_PLUGIN_PARSE_FAILED\n";
            return 2;
        }

        std::cout << "NETWORK_INPUTS=" << network->getNbInputs() << '\n';
        std::cout << "NETWORK_OUTPUTS=" << network->getNbOutputs() << '\n';
        std::cout << "NETWORK_LAYERS=" << network->getNbLayers() << '\n';
        for (int32_t i = 0; i < network->getNbInputs(); ++i)
        {
            auto const* tensor = network->getInput(i);
            std::cout << "INPUT name=" << tensor->getName()
                      << " dtype=" << static_cast<int32_t>(tensor->getType())
                      << " shape=" << dimsToString(tensor->getDimensions()) << '\n';
        }
        for (int32_t i = 0; i < network->getNbOutputs(); ++i)
        {
            auto const* tensor = network->getOutput(i);
            std::cout << "OUTPUT name=" << tensor->getName()
                      << " dtype=" << static_cast<int32_t>(tensor->getType())
                      << " shape=" << dimsToString(tensor->getDimensions()) << '\n';
        }
        std::cout << "PLUGIN_CREATOR_BUILD_CALLS_AFTER_PARSE="
                  << getBuildCreationCount() << '\n';
        if (getBuildCreationCount() < 1 || network->getNbLayers() != 1
            || network->getNbInputs() != 1 || network->getNbOutputs() != 3)
        {
            throw std::runtime_error("Parsed network/Creator call count did not match expectations");
        }
        std::cout << "CUSTOM_ONNX_PLUGIN_PARSE_PASSED\n";

        std::cout << "CUSTOM_ONNX_PLUGIN_ENGINE_BUILD_BEGIN\n";
        TrtUniquePtr<nvinfer1::IHostMemory> serialized{
            builder->buildSerializedNetwork(*network, *config)};
        if (!serialized)
        {
            throw std::runtime_error("buildSerializedNetwork returned null");
        }
        std::ofstream engineFile(enginePath, std::ios::binary);
        engineFile.write(static_cast<char const*>(serialized->data()),
            static_cast<std::streamsize>(serialized->size()));
        engineFile.close();
        std::cout << "ENGINE_PATH=" << enginePath << '\n';
        std::cout << "ENGINE_SIZE_BYTES=" << serialized->size() << '\n';
        std::cout << "PLUGIN_CREATOR_BUILD_CALLS_FINAL="
                  << getBuildCreationCount() << '\n';
        std::cout << "PLUGIN_CREATOR_RUNTIME_CALLS_FINAL="
                  << getRuntimeCreationCount() << '\n';
        std::cout << "CUSTOM_ONNX_PLUGIN_ENGINE_BUILD_PASSED\n";
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "ERROR=" << error.what() << '\n';
        std::cerr << "CUSTOM_ONNX_PLUGIN_PARSE_FAILED\n";
        return 1;
    }
}
