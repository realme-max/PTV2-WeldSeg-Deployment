#include "KnnGraphBuilder.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace ptv2::pointcloud
{

KnnGraphBuilder::KnnGraphBuilder(std::size_t neighbors) noexcept : neighbors_(neighbors) {}

bool KnnGraphBuilder::build(std::vector<PointXYZL> const& points, std::vector<float>& adjacency)
{
    adjacency.clear();
    lastError_.clear();
    if (points.size() != 2048U)
    {
        lastError_ = "KnnGraphBuilder requires exactly 2048 points";
        return false;
    }
    if (neighbors_ == 0U || neighbors_ >= points.size())
    {
        lastError_ = "Invalid KNN neighbor count: " + std::to_string(neighbors_);
        return false;
    }

    std::size_t const count = points.size();
    adjacency.assign(count * count, 0.0F);
    std::vector<std::pair<double, std::size_t>> distances;
    distances.reserve(count - 1U);
    auto const less = [](auto const& left, auto const& right) {
        if (left.first != right.first)
        {
            return left.first < right.first;
        }
        return left.second < right.second;
    };
    for (std::size_t row = 0; row < count; ++row)
    {
        distances.clear();
        PointXYZL const& source = points[row];
        for (std::size_t column = 0; column < count; ++column)
        {
            if (column == row)
            {
                continue;
            }
            PointXYZL const& target = points[column];
            double const dx = static_cast<double>(source.x) - target.x;
            double const dy = static_cast<double>(source.y) - target.y;
            double const dz = static_cast<double>(source.z) - target.z;
            distances.emplace_back(dx * dx + dy * dy + dz * dz, column);
        }
        std::partial_sort(distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(neighbors_),
            distances.end(), less);
        for (std::size_t neighbor = 0; neighbor < neighbors_; ++neighbor)
        {
            adjacency[row * count + distances[neighbor].second] = 1.0F;
        }
    }
    return true;
}

std::size_t KnnGraphBuilder::neighbors() const noexcept { return neighbors_; }
std::string const& KnnGraphBuilder::lastError() const noexcept { return lastError_; }

} // namespace ptv2::pointcloud
