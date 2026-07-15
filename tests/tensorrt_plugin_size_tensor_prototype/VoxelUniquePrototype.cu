#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace voxel_proto
{

using namespace nvinfer1;

constexpr char const* kPluginName{"VoxelUniquePrototype"};
constexpr char const* kPluginVersion{"1"};
constexpr int32_t kInputLength{4};
constexpr size_t kDynamicOutputCapacityBytes{4096};

class Logger final : public ILogger
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

void checkCuda(cudaError_t status, char const* expression)
{
    if (status != cudaSuccess)
    {
        std::ostringstream message;
        message << expression << " failed: " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression)

// Deliberately minimal fixed-input prototype kernel. It demonstrates the
// runtime size tensor contract; it is not the production Unique algorithm.
__global__ void prototypeUniqueKernel(
    int64_t const* keys, int32_t n, int32_t* count, int64_t* values, int64_t* inverse)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    int32_t uniqueCount = 0;
    for (int32_t i = 0; i < n; ++i)
    {
        bool found = false;
        for (int32_t j = 0; j < uniqueCount; ++j)
        {
            if (values[j] == keys[i])
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            values[uniqueCount++] = keys[i];
        }
    }

    // Match ONNX Unique(sorted=1) semantics for this prototype.
    for (int32_t i = 1; i < uniqueCount; ++i)
    {
        int64_t value = values[i];
        int32_t j = i - 1;
        while (j >= 0 && values[j] > value)
        {
            values[j + 1] = values[j];
            --j;
        }
        values[j + 1] = value;
    }

    for (int32_t i = 0; i < n; ++i)
    {
        inverse[i] = -1;
        for (int32_t j = 0; j < uniqueCount; ++j)
        {
            if (values[j] == keys[i])
            {
                inverse[i] = j;
                break;
            }
        }
    }
    *count = uniqueCount;
}

class VoxelUniquePrototype final : public IPluginV3,
                                   public IPluginV3OneCore,
                                   public IPluginV3OneBuild,
                                   public IPluginV3OneRuntime
{
public:
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        switch (type)
        {
        case PluginCapabilityType::kCORE:
            return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD:
            return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME:
            return static_cast<IPluginV3OneRuntime*>(this);
        default:
            return nullptr;
        }
    }

    IPluginV3* clone() noexcept override
    {
        return new (std::nothrow) VoxelUniquePrototype(*this);
    }

    AsciiChar const* getPluginName() const noexcept override
    {
        return kPluginName;
    }

    AsciiChar const* getPluginVersion() const noexcept override
    {
        return kPluginVersion;
    }

    AsciiChar const* getPluginNamespace() const noexcept override
    {
        return "";
    }

    int32_t configurePlugin(
        DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != 1 || nbOutputs != 3)
        {
            return 1;
        }
        return 0;
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs,
        DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        if (outputTypes == nullptr || inputTypes == nullptr || nbInputs != 1 || nbOutputs != 3
            || inputTypes[0] != DataType::kINT64)
        {
            return 1;
        }
        outputTypes[0] = DataType::kINT32; // Runtime size tensor M.
        outputTypes[1] = DataType::kINT64; // values[M].
        outputTypes[2] = DataType::kINT64; // inverse[N].
        return 0;
    }

    int32_t getOutputShapes(
        DimsExprs const* inputs, int32_t nbInputs,
        DimsExprs const* shapeInputs, int32_t nbShapeInputs,
        DimsExprs* outputs, int32_t nbOutputs,
        IExprBuilder& exprBuilder) noexcept override
    {
        (void) shapeInputs;
        if (inputs == nullptr || outputs == nullptr || nbInputs != 1 || nbShapeInputs != 0
            || nbOutputs != 3 || inputs[0].nbDims != 1)
        {
            return 1;
        }

        // Output 0 is a 0D INT32 size tensor. The runtime kernel writes M.
        outputs[0].nbDims = 0;
        IDimensionExpr const* optimum = exprBuilder.constant(3);
        IDimensionExpr const* upper = inputs[0].d[0];
        IDimensionExpr const* runtimeM = exprBuilder.declareSizeTensor(0, *optimum, *upper);
        if (runtimeM == nullptr)
        {
            return 1;
        }

        outputs[1].nbDims = 1;
        outputs[1].d[0] = runtimeM;
        outputs[2].nbDims = 1;
        outputs[2].d[0] = inputs[0].d[0];
        return 0;
    }

    bool supportsFormatCombination(
        int32_t position, DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (inOut == nullptr || nbInputs != 1 || nbOutputs != 3 || position < 0 || position >= 4)
        {
            return false;
        }
        DataType expected = DataType::kINT64;
        if (position == 1)
        {
            expected = DataType::kINT32;
        }
        return inOut[position].desc.format == TensorFormat::kLINEAR
            && inOut[position].desc.type == expected;
    }

    int32_t getNbOutputs() const noexcept override
    {
        return 3;
    }

    int32_t onShapeChange(
        PluginTensorDesc const* inputs, int32_t nbInputs,
        PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != 1 || nbOutputs != 3
            || inputs[0].dims.nbDims != 1)
        {
            return 1;
        }
        return 0;
    }

    int32_t enqueue(
        PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override
    {
        (void) outputDesc;
        (void) workspace;
        if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr
            || inputDesc[0].dims.nbDims != 1)
        {
            return 1;
        }
        int64_t const n64 = inputDesc[0].dims.d[0];
        if (n64 < 0 || n64 > INT32_MAX)
        {
            return 1;
        }
        int32_t const n = static_cast<int32_t>(n64);
        prototypeUniqueKernel<<<1, 1, 0, stream>>>(
            static_cast<int64_t const*>(inputs[0]), n,
            static_cast<int32_t*>(outputs[0]),
            static_cast<int64_t*>(outputs[1]),
            static_cast<int64_t*>(outputs[2]));
        return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
    }

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override
    {
        (void) context;
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        static PluginFieldCollection fields{0, nullptr};
        return &fields;
    }
};

class PreallocatedOutputAllocator final : public IOutputAllocator
{
public:
    PreallocatedOutputAllocator(void* memory, uint64_t capacity)
        : mMemory(memory)
        , mCapacity(capacity)
    {
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

    void notifyShape(char const* tensorName, Dims const& dims) noexcept override
    {
        mTensorName = tensorName == nullptr ? "" : tensorName;
        mShape = dims;
        mShapeNotified = true;
    }

    bool shapeNotified() const noexcept
    {
        return mShapeNotified;
    }

    Dims shape() const noexcept
    {
        return mShape;
    }

    uint64_t requestedBytes() const noexcept
    {
        return mRequestedBytes;
    }

private:
    void* mMemory{};
    uint64_t mCapacity{};
    uint64_t mRequestedBytes{};
    bool mShapeNotified{};
    Dims mShape{};
    std::string mTensorName;
};

std::string dimsToString(Dims const& dims)
{
    std::ostringstream result;
    result << '[';
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        if (i != 0)
        {
            result << ',';
        }
        result << dims.d[i];
    }
    result << ']';
    return result.str();
}

template <typename T>
bool equals(std::vector<T> const& actual, std::vector<T> const& expected)
{
    return actual == expected;
}

} // namespace voxel_proto

int main(int argc, char** argv)
{
    using namespace voxel_proto;
    try
    {
        std::string enginePath = argc > 1 ? argv[1] : "voxel_unique_prototype.plan";
        Logger logger;
        TrtUniquePtr<IBuilder> builder{createInferBuilder(logger)};
        if (!builder)
        {
            throw std::runtime_error("createInferBuilder failed");
        }
        TrtUniquePtr<INetworkDefinition> network{builder->createNetworkV2(0U)};
        TrtUniquePtr<IBuilderConfig> config{builder->createBuilderConfig()};
        if (!network || !config)
        {
            throw std::runtime_error("TensorRT network/config creation failed");
        }
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 256ULL * 1024ULL * 1024ULL);

        Dims keysDims{};
        keysDims.nbDims = 1;
        keysDims.d[0] = kInputLength;
        ITensor* keys = network->addInput("keys", DataType::kINT64, keysDims);
        if (keys == nullptr)
        {
            throw std::runtime_error("addInput(keys) failed");
        }

        auto plugin = std::make_unique<VoxelUniquePrototype>();
        ITensor* pluginInputs[]{keys};
        IPluginV3Layer* layer = network->addPluginV3(pluginInputs, 1, nullptr, 0, *plugin);
        if (layer == nullptr)
        {
            throw std::runtime_error("addPluginV3 failed");
        }
        layer->setName(kPluginName);
        char const* outputNames[]{"count", "values", "inverse"};
        for (int32_t i = 0; i < 3; ++i)
        {
            ITensor* output = layer->getOutput(i);
            if (output == nullptr)
            {
                throw std::runtime_error("Plugin output is null");
            }
            output->setName(outputNames[i]);
            network->markOutput(*output);
        }

        std::cout << "ENGINE_BUILD_BEGIN\n";
        TrtUniquePtr<ICudaEngine> engine{builder->buildEngineWithConfig(*network, *config)};
        if (!engine)
        {
            throw std::runtime_error("buildEngineWithConfig returned null");
        }
        std::cout << "ENGINE_BUILD_PASSED\n";

        TrtUniquePtr<IHostMemory> serialized{engine->serialize()};
        if (!serialized)
        {
            throw std::runtime_error("engine serialization failed");
        }
        std::ofstream engineFile(enginePath, std::ios::binary);
        engineFile.write(static_cast<char const*>(serialized->data()),
            static_cast<std::streamsize>(serialized->size()));
        engineFile.close();
        std::cout << "ENGINE_PATH=" << enginePath << '\n';
        std::cout << "ENGINE_SIZE_BYTES=" << serialized->size() << '\n';

        for (int32_t i = 0; i < engine->getNbIOTensors(); ++i)
        {
            char const* name = engine->getIOTensorName(i);
            std::cout << "IO name=" << name
                      << " mode=" << static_cast<int32_t>(engine->getTensorIOMode(name))
                      << " dtype=" << static_cast<int32_t>(engine->getTensorDataType(name))
                      << " shape=" << dimsToString(engine->getTensorShape(name)) << '\n';
        }

        TrtUniquePtr<IExecutionContext> context{engine->createExecutionContext()};
        if (!context)
        {
            throw std::runtime_error("createExecutionContext failed");
        }

        std::vector<int64_t> const hostKeys{3, 1, 3, 2};
        std::vector<int64_t> hostValues(kInputLength, -1);
        std::vector<int64_t> hostInverse(kInputLength, -1);
        int32_t hostCount{-1};

        int64_t* deviceKeys{};
        int32_t* deviceCount{};
        int64_t* deviceValues{};
        int64_t* deviceInverse{};
        cudaStream_t stream{};
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceKeys), hostKeys.size() * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceCount), sizeof(int32_t)));
        // TensorRT may request an alignment-rounded allocation larger than the
        // logical upper bound (4 * sizeof(int64_t)); retain a guarded prototype
        // buffer and reject any request beyond it in IOutputAllocator.
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceValues), kDynamicOutputCapacityBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceInverse), hostInverse.size() * sizeof(int64_t)));
        CUDA_CHECK(cudaMemcpyAsync(deviceKeys, hostKeys.data(), hostKeys.size() * sizeof(int64_t),
            cudaMemcpyHostToDevice, stream));

        PreallocatedOutputAllocator valuesAllocator{
            deviceValues, kDynamicOutputCapacityBytes};
        if (!context->setTensorAddress("keys", deviceKeys)
            || !context->setTensorAddress("count", deviceCount)
            || !context->setTensorAddress("values", deviceValues)
            || !context->setTensorAddress("inverse", deviceInverse)
            || !context->setOutputAllocator("values", &valuesAllocator))
        {
            throw std::runtime_error("setTensorAddress/setOutputAllocator failed");
        }

        if (!context->enqueueV3(stream))
        {
            throw std::runtime_error("enqueueV3 failed");
        }
        CUDA_CHECK(cudaMemcpyAsync(&hostCount, deviceCount, sizeof(int32_t),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(hostValues.data(), deviceValues,
            hostValues.size() * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(hostInverse.data(), deviceInverse,
            hostInverse.size() * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        Dims runtimeValuesShape = context->getTensorShape("values");
        if (valuesAllocator.shapeNotified())
        {
            runtimeValuesShape = valuesAllocator.shape();
        }
        std::vector<int64_t> actualValues(
            hostValues.begin(), hostValues.begin() + std::max(0, hostCount));
        std::vector<int64_t> const expectedValues{1, 2, 3};
        std::vector<int64_t> const expectedInverse{2, 0, 2, 1};
        bool const passed = hostCount == 3
            && equals(actualValues, expectedValues)
            && equals(hostInverse, expectedInverse)
            && runtimeValuesShape.nbDims == 1
            && runtimeValuesShape.d[0] == 3;

        std::cout << "RUNTIME_COUNT=" << hostCount << '\n';
        std::cout << "RUNTIME_VALUES=";
        for (int64_t value : actualValues)
        {
            std::cout << value << ' ';
        }
        std::cout << "\nRUNTIME_INVERSE=";
        for (int64_t value : hostInverse)
        {
            std::cout << value << ' ';
        }
        std::cout << "\nRUNTIME_VALUES_SHAPE=" << dimsToString(runtimeValuesShape) << '\n';
        std::cout << "OUTPUT_ALLOCATOR_REQUESTED_BYTES=" << valuesAllocator.requestedBytes() << '\n';

        CUDA_CHECK(cudaFree(deviceInverse));
        CUDA_CHECK(cudaFree(deviceValues));
        CUDA_CHECK(cudaFree(deviceCount));
        CUDA_CHECK(cudaFree(deviceKeys));
        CUDA_CHECK(cudaStreamDestroy(stream));

        if (!passed)
        {
            std::cerr << "PLUGIN_SIZE_TENSOR_PROTOTYPE_FAILED\n";
            return 2;
        }
        std::cout << "PLUGIN_SIZE_TENSOR_PROTOTYPE_PASSED\n";
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "ERROR=" << error.what() << '\n';
        std::cerr << "PLUGIN_SIZE_TENSOR_PROTOTYPE_FAILED\n";
        return 1;
    }
}
