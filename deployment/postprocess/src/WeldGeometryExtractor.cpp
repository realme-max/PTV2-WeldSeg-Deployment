#include "WeldGeometryExtractor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace ptv2::postprocess
{
namespace
{

using Matrix3 = std::array<std::array<double, 3>, 3>;

std::array<double, 3> principalDirection(Matrix3 matrix)
{
    Matrix3 eigenvectors{{
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}},
    }};
    for (int iteration = 0; iteration < 64; ++iteration)
    {
        std::size_t p = 0U;
        std::size_t q = 1U;
        double largest = std::abs(matrix[p][q]);
        for (std::size_t row = 0; row < 3U; ++row)
        {
            for (std::size_t column = row + 1U; column < 3U; ++column)
            {
                double const value = std::abs(matrix[row][column]);
                if (value > largest)
                {
                    largest = value;
                    p = row;
                    q = column;
                }
            }
        }
        if (largest <= 1.0e-15) break;

        double const app = matrix[p][p];
        double const aqq = matrix[q][q];
        double const apq = matrix[p][q];
        double const angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
        double const cosine = std::cos(angle);
        double const sine = std::sin(angle);

        for (std::size_t index = 0; index < 3U; ++index)
        {
            if (index == p || index == q) continue;
            double const aip = matrix[index][p];
            double const aiq = matrix[index][q];
            matrix[index][p] = cosine * aip - sine * aiq;
            matrix[p][index] = matrix[index][p];
            matrix[index][q] = sine * aip + cosine * aiq;
            matrix[q][index] = matrix[index][q];
        }
        matrix[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;

        for (std::size_t row = 0; row < 3U; ++row)
        {
            double const vip = eigenvectors[row][p];
            double const viq = eigenvectors[row][q];
            eigenvectors[row][p] = cosine * vip - sine * viq;
            eigenvectors[row][q] = sine * vip + cosine * viq;
        }
    }

    std::size_t principal = 0U;
    if (matrix[1][1] > matrix[principal][principal]) principal = 1U;
    if (matrix[2][2] > matrix[principal][principal]) principal = 2U;
    std::array<double, 3> direction{
        eigenvectors[0][principal], eigenvectors[1][principal], eigenvectors[2][principal]};
    double const norm = std::sqrt(
        direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
    if (!(norm > 0.0) || !std::isfinite(norm)) return {1.0, 0.0, 0.0};
    for (double& value : direction) value /= norm;
    return direction;
}

} // namespace

bool WeldGeometryExtractor::extract(
    std::vector<SegmentationPoint> const& segmentation,
    WeldGeometryResult& result)
{
    lastError_.clear();
    result = {};
    if (segmentation.empty())
    {
        lastError_ = "Cannot extract weld geometry from an empty segmentation";
        return false;
    }

    std::vector<SegmentationPoint const*> weld;
    weld.reserve(segmentation.size());
    for (std::size_t index = 0; index < segmentation.size(); ++index)
    {
        auto const& point = segmentation[index];
        if (point.label != 0 && point.label != 1)
        {
            lastError_ = "Segmentation label must be 0 or 1 at point " + std::to_string(index);
            return false;
        }
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)
            || !std::isfinite(point.confidence))
        {
            lastError_ = "Segmentation contains non-finite data at point " + std::to_string(index);
            return false;
        }
        if (point.label == 0) weld.push_back(&point);
    }
    if (weld.empty())
    {
        lastError_ = "Segmentation contains no weld_seam (class 0) points";
        return false;
    }

    std::array<double, 3> minimum{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    std::array<double, 3> maximum{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
    std::array<double, 3> center{};
    for (auto const* point : weld)
    {
        std::array<double, 3> const coordinate{point->x, point->y, point->z};
        for (std::size_t axis = 0; axis < 3U; ++axis)
        {
            center[axis] += coordinate[axis];
            minimum[axis] = std::min(minimum[axis], coordinate[axis]);
            maximum[axis] = std::max(maximum[axis], coordinate[axis]);
        }
    }
    for (double& value : center) value /= static_cast<double>(weld.size());

    Matrix3 covariance{};
    for (auto const* point : weld)
    {
        std::array<double, 3> const delta{
            static_cast<double>(point->x) - center[0],
            static_cast<double>(point->y) - center[1],
            static_cast<double>(point->z) - center[2]};
        for (std::size_t row = 0; row < 3U; ++row)
        {
            for (std::size_t column = 0; column < 3U; ++column)
            {
                covariance[row][column] += delta[row] * delta[column];
            }
        }
    }
    double const divisor = static_cast<double>(weld.size());
    for (auto& row : covariance)
    {
        for (double& value : row) value /= divisor;
    }
    auto const direction = principalDirection(covariance);

    double projectionMin = std::numeric_limits<double>::infinity();
    double projectionMax = -std::numeric_limits<double>::infinity();
    for (auto const* point : weld)
    {
        double const projection =
            (static_cast<double>(point->x) - center[0]) * direction[0]
            + (static_cast<double>(point->y) - center[1]) * direction[1]
            + (static_cast<double>(point->z) - center[2]) * direction[2];
        projectionMin = std::min(projectionMin, projection);
        projectionMax = std::max(projectionMax, projection);
    }

    result.weldPoints = weld.size();
    result.weldRatio = static_cast<float>(
        static_cast<double>(weld.size()) / static_cast<double>(segmentation.size()));
    for (std::size_t axis = 0; axis < 3U; ++axis)
    {
        result.center[axis] = static_cast<float>(center[axis]);
        result.bboxMin[axis] = static_cast<float>(minimum[axis]);
        result.bboxMax[axis] = static_cast<float>(maximum[axis]);
        result.principalDirection[axis] = static_cast<float>(direction[axis]);
    }
    result.lengthMm = static_cast<float>(projectionMax - projectionMin);
    float const directionNorm = std::sqrt(
        result.principalDirection[0] * result.principalDirection[0]
        + result.principalDirection[1] * result.principalDirection[1]
        + result.principalDirection[2] * result.principalDirection[2]);
    if (!std::isfinite(result.weldRatio) || !std::isfinite(result.lengthMm)
        || !std::isfinite(directionNorm) || std::abs(directionNorm - 1.0F) > 1.0e-5F)
    {
        lastError_ = "Computed weld geometry contains NaN/Inf";
        result = {};
        return false;
    }
    return true;
}

std::string const& WeldGeometryExtractor::lastError() const noexcept
{
    return lastError_;
}

} // namespace ptv2::postprocess
