#include "WeldConfig.h"

namespace ptv2::weld
{

bool WeldConfig::validate(std::string& error) const
{
    error.clear();
    if (engine_path.empty())
    {
        error = "engine_path must not be empty";
        return false;
    }
    if (plugin_path.empty())
    {
        error = "plugin_path must not be empty";
        return false;
    }
    if (input_points != 2048)
    {
        error = "input_points must be 2048 for the production Engine";
        return false;
    }
    if (num_classes != 2)
    {
        error = "num_classes must be 2 for the production Engine";
        return false;
    }
    return true;
}

} // namespace ptv2::weld
