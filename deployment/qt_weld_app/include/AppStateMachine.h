#pragma once

#include <QString>

namespace ptv2::qtui
{

enum class AppState
{
    kStarting,
    kConfigurationInvalid,
    kInitializing,
    kReady,
    kCloudSelected,
    kDetecting,
    kDetectionSucceeded,
    kDetectionFailed,
    kExporting,
    kShuttingDown
};

class AppStateMachine
{
public:
    AppState state() const noexcept;
    bool transition(AppState next, QString& error);
    bool canSelectCloud() const noexcept;
    bool canDetect(bool validCloud) const noexcept;
    bool canExport() const noexcept;
    bool canOpenSettings() const noexcept;
    QString stateName() const;

private:
    bool allowed(AppState next) const noexcept;
    AppState state_{AppState::kStarting};
};

} // namespace ptv2::qtui
