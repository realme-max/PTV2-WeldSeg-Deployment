#pragma once

#include <QString>

namespace ptv2::qtui
{

class ProductInfo
{
public:
    static QString applicationName();
    static QString applicationVersion();
    static QString buildType();
    static QString buildTimestamp();
    static QString gitCommit();
    static QString compiler();
};

} // namespace ptv2::qtui
