#include "CudaBufferManager.h"

#include <sstream>

namespace ptv2::runtime
{
namespace
{
bool checkCuda(cudaError_t result, char const* operation, std::string& error)
{
    if (result == cudaSuccess)
    {
        return true;
    }
    std::ostringstream stream;
    stream << operation << " failed: " << cudaGetErrorName(result) << " - " << cudaGetErrorString(result);
    error = stream.str();
    return false;
}
} // namespace

CudaBufferManager::~CudaBufferManager()
{
    release();
}

bool CudaBufferManager::allocate(std::string& error)
{
    release();
    if (!checkCuda(cudaMalloc(&points_, pointsBytes()), "cudaMalloc(points)", error)
        || !checkCuda(cudaMalloc(&adj_, adjBytes()), "cudaMalloc(adj)", error)
        || !checkCuda(cudaMalloc(&logits_, logitsBytes()), "cudaMalloc(logits)", error))
    {
        release();
        return false;
    }
    return true;
}

bool CudaBufferManager::copyInput(
    float const* points, std::size_t pointsElements,
    float const* adj, std::size_t adjElements,
    cudaStream_t stream, std::string& error) const
{
    if (points == nullptr || adj == nullptr)
    {
        error = "Input host pointer is null";
        return false;
    }
    if (pointsElements != kPointsElements || adjElements != kAdjElements)
    {
        std::ostringstream message;
        message << "Input element count mismatch: points=" << pointsElements << "/" << kPointsElements
                << ", adj=" << adjElements << "/" << kAdjElements;
        error = message.str();
        return false;
    }
    return checkCuda(
               cudaMemcpyAsync(points_, points, pointsBytes(), cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(points H2D)", error)
        && checkCuda(
               cudaMemcpyAsync(adj_, adj, adjBytes(), cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(adj H2D)", error);
}

bool CudaBufferManager::copyOutput(
    float* logits, std::size_t logitsElements,
    cudaStream_t stream, std::string& error) const
{
    if (logits == nullptr)
    {
        error = "Output host pointer is null";
        return false;
    }
    if (logitsElements != kLogitsElements)
    {
        std::ostringstream message;
        message << "Output element count mismatch: logits=" << logitsElements << "/" << kLogitsElements;
        error = message.str();
        return false;
    }
    return checkCuda(
        cudaMemcpyAsync(logits, logits_, logitsBytes(), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync(logits D2H)", error);
}

void CudaBufferManager::release() noexcept
{
    if (logits_ != nullptr)
    {
        cudaFree(logits_);
        logits_ = nullptr;
    }
    if (adj_ != nullptr)
    {
        cudaFree(adj_);
        adj_ = nullptr;
    }
    if (points_ != nullptr)
    {
        cudaFree(points_);
        points_ = nullptr;
    }
}

void* CudaBufferManager::points() const noexcept { return points_; }
void* CudaBufferManager::adj() const noexcept { return adj_; }
void* CudaBufferManager::logits() const noexcept { return logits_; }

} // namespace ptv2::runtime
