#pragma once

#include <string>

namespace ptv2::weld
{

struct WeldConfig
{
    std::string engine_path;
    std::string plugin_path;
    int input_points{2048};
    int num_classes{2};
    bool enable_geometry{true};

    // Optional Phase 9C compatibility output. Empty means result-only SDK use.
    std::string output_path;

    bool validate(std::string& error) const;
};

} // namespace ptv2::weld
