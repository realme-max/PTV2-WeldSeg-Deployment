#include "PointCloudLoader.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

namespace ptv2::pointcloud
{
namespace
{
void skipWhitespace(char const*& cursor) noexcept
{
    while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n')
    {
        ++cursor;
    }
}

bool parseFloat(char const*& cursor, float& value) noexcept
{
    skipWhitespace(cursor);
    if (*cursor == '\0')
    {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    value = std::strtof(cursor, &end);
    if (end == cursor || errno == ERANGE)
    {
        return false;
    }
    cursor = end;
    return true;
}
} // namespace

bool PointCloudLoader::load(std::string const& path, std::vector<PointXYZL>& points)
{
    points.clear();
    stats_ = {};
    lastError_.clear();

    std::filesystem::path const file(path);
    if (!std::filesystem::is_regular_file(file))
    {
        lastError_ = "Point cloud TXT does not exist: " + path;
        return false;
    }
    std::ifstream input(file);
    if (!input)
    {
        lastError_ = "Cannot open point cloud TXT: " + path;
        return false;
    }

    stats_.minimum.fill(std::numeric_limits<float>::infinity());
    stats_.maximum.fill(-std::numeric_limits<float>::infinity());
    points.reserve(2048);
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line))
    {
        ++lineNumber;
        char const* cursor = line.c_str();
        skipWhitespace(cursor);
        if (*cursor == '\0')
        {
            continue;
        }

        float values[4]{};
        bool valid = true;
        for (float& value : values)
        {
            valid = valid && parseFloat(cursor, value);
        }
        skipWhitespace(cursor);
        valid = valid && *cursor == '\0';
        for (float value : values)
        {
            valid = valid && std::isfinite(value);
        }
        float const roundedLabel = std::round(values[3]);
        valid = valid && roundedLabel == values[3] && (roundedLabel == 0.0F || roundedLabel == 1.0F);
        if (!valid)
        {
            lastError_ = "Invalid four-column finite x y z label row at line " + std::to_string(lineNumber);
            points.clear();
            return false;
        }

        PointXYZL const point{values[0], values[1], values[2], static_cast<int>(roundedLabel)};
        points.push_back(point);
        float const xyz[3]{point.x, point.y, point.z};
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            stats_.minimum[axis] = std::min(stats_.minimum[axis], xyz[axis]);
            stats_.maximum[axis] = std::max(stats_.maximum[axis], xyz[axis]);
        }
        ++stats_.labelCounts[static_cast<std::size_t>(point.label)];
    }
    if (!input.eof())
    {
        lastError_ = "I/O error while reading point cloud TXT: " + path;
        points.clear();
        return false;
    }
    if (points.empty())
    {
        lastError_ = "Point cloud TXT contains no points: " + path;
        return false;
    }
    stats_.pointCount = points.size();
    std::cout << "PointCloudLoader: count=" << stats_.pointCount
              << ", min=[" << stats_.minimum[0] << ',' << stats_.minimum[1] << ',' << stats_.minimum[2] << ']'
              << ", max=[" << stats_.maximum[0] << ',' << stats_.maximum[1] << ',' << stats_.maximum[2] << ']'
              << ", labels={0:" << stats_.labelCounts[0] << ",1:" << stats_.labelCounts[1] << "}\n";
    return true;
}

PointCloudStats const& PointCloudLoader::stats() const noexcept { return stats_; }
std::string const& PointCloudLoader::lastError() const noexcept { return lastError_; }

} // namespace ptv2::pointcloud
