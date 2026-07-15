#include "VoxelUniqueCorrectnessPlugin.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntimePlugin.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ptv2::gcn_res_plugin_parser
{

class Logger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, char const* message) noexcept override
    {
        if (severity <= Severity::kVERBOSE)
        {
            std::cout << "[TensorRT] " << (message == nullptr ? "" : message) << '\n';
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

struct CreatorInfo
{
    std::string name;
    std::string version;
    std::string pluginNamespace;
    std::string interfaceKind;
};

CreatorInfo describeCreator(nvinfer1::IPluginCreatorInterface* creator)
{
    if (creator == nullptr) return {};
    CreatorInfo result;
    auto const interfaceInfo = creator->getInterfaceInfo();
    result.interfaceKind = interfaceInfo.kind == nullptr ? "" : interfaceInfo.kind;
    if (auto* creatorV3 = dynamic_cast<nvinfer1::IPluginCreatorV3One*>(creator))
    {
        result.name = creatorV3->getPluginName();
        result.version = creatorV3->getPluginVersion();
        result.pluginNamespace = creatorV3->getPluginNamespace();
    }
    else if (auto* creatorV2 = dynamic_cast<nvinfer1::IPluginCreator*>(creator))
    {
        result.name = creatorV2->getPluginName();
        result.version = creatorV2->getPluginVersion();
        result.pluginNamespace = creatorV2->getPluginNamespace();
    }
    return result;
}

std::vector<CreatorInfo> collectScatterCreators(nvinfer1::IPluginRegistry& registry,
    int32_t& registryCreatorCount, bool& exactScatterReductionFound)
{
    nvinfer1::IPluginCreatorInterface* const* creators
        = registry.getAllCreators(&registryCreatorCount);
    std::vector<CreatorInfo> scatterCreators;
    exactScatterReductionFound = false;
    for (int32_t index = 0; index < registryCreatorCount; ++index)
    {
        CreatorInfo info = describeCreator(creators[index]);
        std::string lowerName = info.name;
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(),
            [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
        if (lowerName.find("scatter") != std::string::npos)
        {
            exactScatterReductionFound
                = exactScatterReductionFound || info.name == "ScatterReduction";
            scatterCreators.push_back(std::move(info));
        }
    }
    return scatterCreators;
}

std::string jsonEscape(char const* value)
{
    std::ostringstream result;
    std::string const source = value == nullptr ? "" : value;
    for (unsigned char character : source)
    {
        switch (character)
        {
        case '\\': result << "\\\\"; break;
        case '"': result << "\\\""; break;
        case '\n': result << "\\n"; break;
        case '\r': result << "\\r"; break;
        case '\t': result << "\\t"; break;
        default:
            if (character < 0x20)
            {
                result << "?";
            }
            else
            {
                result << character;
            }
        }
    }
    return result.str();
}

std::string dimsJson(nvinfer1::Dims const& dims)
{
    std::ostringstream result;
    result << '[';
    for (int32_t index = 0; index < dims.nbDims; ++index)
    {
        if (index != 0) result << ',';
        result << dims.d[index];
    }
    result << ']';
    return result.str();
}

void writeTensorArray(
    std::ostream& output, nvinfer1::INetworkDefinition const& network, bool inputs)
{
    int32_t const count = inputs ? network.getNbInputs() : network.getNbOutputs();
    output << '[';
    for (int32_t index = 0; index < count; ++index)
    {
        if (index != 0) output << ',';
        auto const* tensor = inputs ? network.getInput(index) : network.getOutput(index);
        output << "{\"name\":\"" << jsonEscape(tensor->getName())
               << "\",\"dtype_code\":" << static_cast<int32_t>(tensor->getType())
               << ",\"shape\":" << dimsJson(tensor->getDimensions()) << '}';
    }
    output << ']';
}

void writeSummary(
    std::string const& path, bool creatorRegistered, bool creatorLookup,
    bool parserSuccess, nvonnxparser::IParser const& parser,
    nvinfer1::INetworkDefinition const& network, int32_t buildCreationCalls,
    bool standardPluginsInitialized, int32_t registryCreatorCount,
    std::vector<CreatorInfo> const& scatterCreators,
    bool exactScatterReductionFound)
{
    std::ofstream output(path, std::ios::binary);
    output << "{\n"
           << "  \"phase\": \"TensorRT Phase 3 parser-only audit\",\n"
           << "  \"engine_build_called\": false,\n"
           << "  \"inference_called\": false,\n"
           << "  \"standard_plugins_initialized\": " << std::boolalpha
           << standardPluginsInitialized << ",\n"
           << "  \"registry_creator_count_after_standard_init\": "
           << registryCreatorCount << ",\n"
           << "  \"scatter_reduction_creator_found\": "
           << exactScatterReductionFound << ",\n"
           << "  \"scatter_related_creators\": [";
    for (size_t index = 0; index < scatterCreators.size(); ++index)
    {
        if (index != 0) output << ',';
        CreatorInfo const& item = scatterCreators[index];
        output << "{\"name\":\"" << jsonEscape(item.name.c_str())
               << "\",\"version\":\"" << jsonEscape(item.version.c_str())
               << "\",\"namespace\":\"" << jsonEscape(item.pluginNamespace.c_str())
               << "\",\"interface_kind\":\"" << jsonEscape(item.interfaceKind.c_str())
               << "\"}";
    }
    output << "],\n"
           << "  \"plugin_creator_registered\": " << std::boolalpha << creatorRegistered << ",\n"
           << "  \"plugin_creator_lookup_passed\": " << creatorLookup << ",\n"
           << "  \"plugin_creator_build_calls\": " << buildCreationCalls << ",\n"
           << "  \"parser_success\": " << parserSuccess << ",\n"
           << "  \"parser_error_count\": " << parser.getNbErrors() << ",\n"
           << "  \"network_layer_count\": " << network.getNbLayers() << ",\n"
           << "  \"network_inputs\": ";
    writeTensorArray(output, network, true);
    output << ",\n  \"network_outputs\": ";
    writeTensorArray(output, network, false);
    output << ",\n  \"errors\": [";
    for (int32_t index = 0; index < parser.getNbErrors(); ++index)
    {
        if (index != 0) output << ',';
        auto const* error = parser.getError(index);
        output << "{\"index\":" << index
               << ",\"error_code\":" << static_cast<int32_t>(error->code())
               << ",\"node_index\":" << error->node()
               << ",\"node_name\":\"" << jsonEscape(error->nodeName())
               << "\",\"op_type\":\"" << jsonEscape(error->nodeOperator())
               << "\",\"description\":\"" << jsonEscape(error->desc())
               << "\",\"source_file\":\"" << jsonEscape(error->file())
               << "\",\"source_function\":\"" << jsonEscape(error->func())
               << "\",\"source_line\":" << error->line() << '}';
    }
    output << "]\n}\n";
}

bool ioMatchesExpected(nvinfer1::INetworkDefinition const& network)
{
    if (network.getNbInputs() != 2 || network.getNbOutputs() != 1) return false;
    auto const* points = network.getInput(0);
    auto const* adj = network.getInput(1);
    auto const* logits = network.getOutput(0);
    if (std::string(points->getName()) != "points"
        || std::string(adj->getName()) != "adj"
        || std::string(logits->getName()) != "logits")
    {
        return false;
    }
    auto const pointsDims = points->getDimensions();
    auto const adjDims = adj->getDimensions();
    auto const logitsDims = logits->getDimensions();
    return pointsDims.nbDims == 3 && pointsDims.d[0] == 1 && pointsDims.d[1] == 2048
        && pointsDims.d[2] == 4 && adjDims.nbDims == 3 && adjDims.d[0] == 1
        && adjDims.d[1] == 2048 && adjDims.d[2] == 2048 && logitsDims.nbDims == 3
        && logitsDims.d[0] == 1 && logitsDims.d[1] == 2048 && logitsDims.d[2] == 2;
}

} // namespace ptv2::gcn_res_plugin_parser

int main(int argc, char** argv)
{
    using namespace ptv2::gcn_res_plugin_parser;
    using namespace ptv2::voxel_unique_correctness;
    if (argc != 3)
    {
        std::cerr << "Usage: gcn_res_voxel_unique_parser <rewritten.onnx> <parser_summary.json>\n";
        return 64;
    }

    try
    {
        Logger logger;
        bool const standardPluginsInitialized
            = initLibNvInferPlugins(static_cast<void*>(&logger), "");
        std::cout << "STANDARD_PLUGINS_INITIALIZED=" << std::boolalpha
                  << standardPluginsInitialized << '\n';
        if (!standardPluginsInitialized)
            throw std::runtime_error("initLibNvInferPlugins returned false");

        auto* registry = ::getPluginRegistry();
        if (registry == nullptr) throw std::runtime_error("Plugin Registry is null");
        int32_t registryCreatorCount{};
        bool exactScatterReductionFound{};
        std::vector<CreatorInfo> const scatterCreators = collectScatterCreators(
            *registry, registryCreatorCount, exactScatterReductionFound);
        std::cout << "REGISTRY_CREATOR_COUNT_AFTER_STANDARD_INIT="
                  << registryCreatorCount << '\n';
        std::cout << "SCATTER_REDUCTION_CREATOR_FOUND="
                  << exactScatterReductionFound << '\n';
        std::cout << "SCATTER_RELATED_CREATOR_COUNT=" << scatterCreators.size() << '\n';
        for (CreatorInfo const& item : scatterCreators)
        {
            std::cout << "SCATTER_CREATOR name=" << item.name
                      << " version=" << item.version
                      << " namespace=" << item.pluginNamespace
                      << " interface=" << item.interfaceKind << '\n';
        }

        VoxelUniquePluginCreator creator;
        bool const creatorRegistered = registry->registerCreator(creator, kPluginNamespace);
        auto* creatorLookup = registry->getCreator(kPluginName, kPluginVersion, kPluginNamespace);
        bool const creatorLookupPassed
            = creatorLookup == static_cast<nvinfer1::IPluginCreatorInterface*>(&creator);
        std::cout << "PLUGIN_CREATOR_REGISTERED=" << std::boolalpha << creatorRegistered << '\n';
        std::cout << "PLUGIN_CREATOR_LOOKUP_PASSED=" << creatorLookupPassed << '\n';
        if (!creatorRegistered || !creatorLookupPassed)
            throw std::runtime_error("Plugin Creator registration/lookup failed");

        TrtUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
        if (!builder) throw std::runtime_error("createInferBuilder failed");
        TrtUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(0U)};
        if (!network) throw std::runtime_error("createNetworkV2 failed");
        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
        if (parser == nullptr) throw std::runtime_error("createParser failed");

        std::cout << "PARSER_ONLY=true\n";
        std::cout << "ENGINE_BUILD_CALLED=false\n";
        std::cout << "PARSER_BEGIN\n";
        bool const parserSuccess = parser->parseFromFile(
            argv[1], static_cast<int32_t>(nvinfer1::ILogger::Severity::kVERBOSE));
        std::cout << "PARSER_END success=" << parserSuccess << '\n';
        std::cout << "PARSER_ERROR_COUNT=" << parser->getNbErrors() << '\n';
        std::cout << "PLUGIN_CREATOR_BUILD_CALLS=" << getBuildCreationCount() << '\n';
        writeSummary(argv[2], creatorRegistered, creatorLookupPassed, parserSuccess,
            *parser, *network, getBuildCreationCount(), standardPluginsInitialized,
            registryCreatorCount, scatterCreators, exactScatterReductionFound);

        if (!parserSuccess)
        {
            for (int32_t index = 0; index < parser->getNbErrors(); ++index)
            {
                auto const* error = parser->getError(index);
                std::cerr << "PARSER_ERROR index=" << index
                          << " code=" << static_cast<int32_t>(error->code())
                          << " node=" << error->node()
                          << " node_name=" << error->nodeName()
                          << " op=" << error->nodeOperator()
                          << " description=" << error->desc() << '\n';
            }
            if (parser->getNbErrors() > 0)
            {
                auto const* first = parser->getError(0);
                std::cerr << "FIRST_BLOCKING_NODE=" << first->nodeName() << '\n';
                std::cerr << "FIRST_BLOCKING_OPERATOR=" << first->nodeOperator() << '\n';
                std::cerr << "FIRST_BLOCKING_ERROR=" << first->desc() << '\n';
            }
            std::cerr << "TENSORRT_GCN_RES_PLUGIN_PARSER_FAILED\n";
            return 2;
        }

        bool const ioPassed = ioMatchesExpected(*network);
        bool const creatorsPassed = getBuildCreationCount() == 4;
        std::cout << "NETWORK_LAYERS=" << network->getNbLayers() << '\n';
        std::cout << "NETWORK_INPUTS=" << network->getNbInputs() << '\n';
        std::cout << "NETWORK_OUTPUTS=" << network->getNbOutputs() << '\n';
        std::cout << "IO_MATCHES_EXPECTED=" << ioPassed << '\n';
        std::cout << "FOUR_PLUGIN_INSTANCES_CREATED=" << creatorsPassed << '\n';
        if (!ioPassed || !creatorsPassed)
            throw std::runtime_error("Parser returned success but network contract validation failed");
        std::cout << "TENSORRT_GCN_RES_PLUGIN_PARSER_PASSED\n";
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "ERROR=" << error.what() << '\n';
        std::cerr << "TENSORRT_GCN_RES_PLUGIN_PARSER_FAILED\n";
        return 1;
    }
}
