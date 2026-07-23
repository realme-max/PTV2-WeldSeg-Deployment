#include "AppStateMachine.h"

namespace ptv2::qtui
{

AppState AppStateMachine::state() const noexcept
{
    return state_;
}

bool AppStateMachine::allowed(AppState next) const noexcept
{
    if (next == AppState::kShuttingDown) return state_ != AppState::kExporting;
    switch (state_)
    {
    case AppState::kStarting:
        return next == AppState::kConfigurationInvalid || next == AppState::kInitializing;
    case AppState::kConfigurationInvalid:
        return next == AppState::kInitializing || next == AppState::kConfigurationInvalid;
    case AppState::kInitializing:
        return next == AppState::kReady || next == AppState::kConfigurationInvalid;
    case AppState::kReady:
        return next == AppState::kCloudSelected || next == AppState::kInitializing;
    case AppState::kCloudSelected:
        return next == AppState::kDetecting || next == AppState::kReady
            || next == AppState::kCloudSelected || next == AppState::kInitializing;
    case AppState::kDetecting:
        return next == AppState::kDetectionSucceeded || next == AppState::kDetectionFailed;
    case AppState::kDetectionSucceeded:
        return next == AppState::kExporting || next == AppState::kDetecting
            || next == AppState::kCloudSelected || next == AppState::kInitializing;
    case AppState::kDetectionFailed:
        return next == AppState::kDetecting || next == AppState::kCloudSelected
            || next == AppState::kReady || next == AppState::kInitializing;
    case AppState::kExporting:
        return next == AppState::kDetectionSucceeded || next == AppState::kDetectionFailed;
    case AppState::kShuttingDown:
        return false;
    }
    return false;
}

bool AppStateMachine::transition(AppState next, QString& error)
{
    if (!allowed(next))
    {
        error = QStringLiteral("Invalid application state transition: %1 -> %2")
            .arg(stateName(), [next]() {
                AppStateMachine temporary;
                temporary.state_ = next;
                return temporary.stateName();
            }());
        return false;
    }
    state_ = next;
    error.clear();
    return true;
}

bool AppStateMachine::canSelectCloud() const noexcept
{
    return state_ != AppState::kDetecting && state_ != AppState::kExporting
        && state_ != AppState::kInitializing && state_ != AppState::kShuttingDown;
}

bool AppStateMachine::canDetect(bool validCloud) const noexcept
{
    return validCloud && (state_ == AppState::kCloudSelected
        || state_ == AppState::kDetectionSucceeded || state_ == AppState::kDetectionFailed);
}

bool AppStateMachine::canExport() const noexcept
{
    return state_ == AppState::kDetectionSucceeded;
}

bool AppStateMachine::canOpenSettings() const noexcept
{
    return state_ != AppState::kDetecting && state_ != AppState::kExporting
        && state_ != AppState::kInitializing && state_ != AppState::kShuttingDown;
}

QString AppStateMachine::stateName() const
{
    switch (state_)
    {
    case AppState::kStarting: return QStringLiteral("STARTING");
    case AppState::kConfigurationInvalid: return QStringLiteral("CONFIGURATION_INVALID");
    case AppState::kInitializing: return QStringLiteral("INITIALIZING");
    case AppState::kReady: return QStringLiteral("READY");
    case AppState::kCloudSelected: return QStringLiteral("CLOUD_SELECTED");
    case AppState::kDetecting: return QStringLiteral("DETECTING");
    case AppState::kDetectionSucceeded: return QStringLiteral("DETECTION_SUCCEEDED");
    case AppState::kDetectionFailed: return QStringLiteral("DETECTION_FAILED");
    case AppState::kExporting: return QStringLiteral("EXPORTING");
    case AppState::kShuttingDown: return QStringLiteral("SHUTTING_DOWN");
    }
    return QStringLiteral("UNKNOWN");
}

} // namespace ptv2::qtui
