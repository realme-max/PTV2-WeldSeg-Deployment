#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>

namespace ptv2::runtime
{

class CudaBufferManager
{
public:
    static constexpr std::size_t kPointsElements{1U * 2048U * 4U};
    static constexpr std::size_t kAdjElements{1U * 2048U * 2048U};
    static constexpr std::size_t kLogitsElements{1U * 2048U * 2U};

    CudaBufferManager() = default;
    ~CudaBufferManager();

    CudaBufferManager(CudaBufferManager const&) = delete;
    CudaBufferManager& operator=(CudaBufferManager const&) = delete;

    bool allocate(std::string& error);
    bool copyInput(
        float const* points, std::size_t pointsElements,
        float const* adj, std::size_t adjElements,
        cudaStream_t stream, std::string& error) const;
    bool copyOutput(
        float* logits, std::size_t logitsElements,
        cudaStream_t stream, std::string& error) const;
    void release() noexcept;

    void* points() const noexcept;
    void* adj() const noexcept;
    void* logits() const noexcept;

    static constexpr std::size_t pointsBytes() noexcept { return kPointsElements * sizeof(float); }
    static constexpr std::size_t adjBytes() noexcept { return kAdjElements * sizeof(float); }
    static constexpr std::size_t logitsBytes() noexcept { return kLogitsElements * sizeof(float); }

private:
    void* points_{nullptr};
    void* adj_{nullptr};
    void* logits_{nullptr};
};

} // namespace ptv2::runtime
