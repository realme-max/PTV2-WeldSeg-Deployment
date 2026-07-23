#pragma once

#include "WeldConfig.h"
#include "WeldResult.h"
#include "WeldStatus.h"

#include <memory>
#include <string>

namespace ptv2::weld
{

class WeldDetector
{
public:
    WeldDetector();
    ~WeldDetector();

    WeldDetector(WeldDetector const&) = delete;
    WeldDetector& operator=(WeldDetector const&) = delete;

    WeldStatus initialize(WeldConfig const& config);
    WeldStatus detect(std::string const& cloudPath, WeldResult& result);
    std::string lastError() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ptv2::weld
