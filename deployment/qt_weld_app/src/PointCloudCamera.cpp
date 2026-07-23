#include "PointCloudCamera.h"

#include <QtMath>

#include <algorithm>
#include <cmath>

namespace ptv2::qtui
{

void PointCloudCamera::resetView()
{
    yaw_ = -35.0F;
    pitch_ = 25.0F;
    distance_ = std::max(2.5F * fitRadius_, 0.01F);
    panOffset_ = {};
    clampState();
}

void PointCloudCamera::fitToBounds(QVector3D const& minimum, QVector3D const& maximum)
{
    center_ = (minimum + maximum) * 0.5F;
    QVector3D const extent = maximum - minimum;
    fitRadius_ = std::max({extent.x(), extent.y(), extent.z(), 1.0e-3F}) * 0.5F;
    resetView();
}

void PointCloudCamera::rotate(float deltaYaw, float deltaPitch)
{
    yaw_ += deltaYaw;
    pitch_ += deltaPitch;
    clampState();
}

void PointCloudCamera::pan(float deltaX, float deltaY)
{
    float const scale = std::max(distance_, 0.01F) * 0.0015F;
    panOffset_.setX(panOffset_.x() - deltaX * scale);
    panOffset_.setY(panOffset_.y() + deltaY * scale);
}

void PointCloudCamera::zoom(float wheelSteps)
{
    distance_ *= std::pow(0.88F, wheelSteps);
    clampState();
}

QMatrix4x4 PointCloudCamera::viewMatrix() const
{
    QMatrix4x4 view;
    view.translate(panOffset_);
    view.translate(0.0F, 0.0F, -distance_);
    view.rotate(pitch_, 1.0F, 0.0F, 0.0F);
    view.rotate(yaw_, 0.0F, 0.0F, 1.0F);
    view.translate(-center_);
    return view;
}

QMatrix4x4 PointCloudCamera::projectionMatrix(float aspectRatio) const
{
    QMatrix4x4 projection;
    float const safeAspect = std::isfinite(aspectRatio) && aspectRatio > 0.0F ? aspectRatio : 1.0F;
    float const nearPlane = std::max(0.001F, distance_ - fitRadius_ * 4.0F);
    float const farPlane = std::max(nearPlane + 1.0F, distance_ + fitRadius_ * 8.0F);
    projection.perspective(45.0F, safeAspect, nearPlane, farPlane);
    return projection;
}

float PointCloudCamera::yaw() const noexcept { return yaw_; }
float PointCloudCamera::pitch() const noexcept { return pitch_; }
float PointCloudCamera::distance() const noexcept { return distance_; }
QVector3D PointCloudCamera::center() const noexcept { return center_; }
QVector3D PointCloudCamera::panOffset() const noexcept { return panOffset_; }

void PointCloudCamera::clampState()
{
    pitch_ = std::max(-89.0F, std::min(89.0F, pitch_));
    float const minimum = std::max(fitRadius_ * 0.05F, 0.001F);
    float const maximum = std::max(fitRadius_ * 100.0F, 1.0F);
    if (!std::isfinite(distance_)) distance_ = std::max(2.5F * fitRadius_, 0.01F);
    distance_ = std::max(minimum, std::min(maximum, distance_));
    if (!std::isfinite(yaw_)) yaw_ = -35.0F;
    if (!std::isfinite(pitch_)) pitch_ = 25.0F;
}

} // namespace ptv2::qtui
