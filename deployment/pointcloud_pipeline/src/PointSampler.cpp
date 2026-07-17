#include "PointSampler.h"

#include <algorithm>
#include <numeric>
#include <random>

namespace ptv2::pointcloud
{

PointSampler::PointSampler(std::uint32_t seed) noexcept : seed_(seed) {}

bool PointSampler::sample(
    std::vector<PointXYZL> const& input,
    std::vector<PointXYZL>& output,
    int targetPoints)
{
    std::vector<std::size_t> ignored;
    return sample(input, output, ignored, targetPoints);
}

bool PointSampler::sample(
    std::vector<PointXYZL> const& input,
    std::vector<PointXYZL>& output,
    std::vector<std::size_t>& sampledIndices,
    int targetPoints)
{
    output.clear();
    sampledIndices.clear();
    lastError_.clear();
    if (targetPoints <= 0)
    {
        lastError_ = "targetPoints must be positive";
        return false;
    }
    if (input.size() < static_cast<std::size_t>(targetPoints))
    {
        lastError_ = "Input point count " + std::to_string(input.size())
            + " is smaller than target " + std::to_string(targetPoints);
        return false;
    }

    sampledIndices.resize(input.size());
    std::iota(sampledIndices.begin(), sampledIndices.end(), 0U);
    std::mt19937 generator(seed_);
    std::shuffle(sampledIndices.begin(), sampledIndices.end(), generator);
    sampledIndices.resize(static_cast<std::size_t>(targetPoints));
    output.reserve(static_cast<std::size_t>(targetPoints));
    for (std::size_t index : sampledIndices)
    {
        output.push_back(input[index]);
    }
    return true;
}

std::string const& PointSampler::lastError() const noexcept { return lastError_; }

} // namespace ptv2::pointcloud
