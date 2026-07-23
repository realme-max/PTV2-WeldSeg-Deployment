#pragma once

#include <QMatrix4x4>
#include <QPointF>
#include <QVector3D>

namespace ptv2::qtui
{

class PointCloudCamera
{
public:
    void resetView();
    void fitToBounds(QVector3D const& minimum, QVector3D const& maximum);
    void rotate(float deltaYaw, float deltaPitch);
    void pan(float deltaX, float deltaY);
    void zoom(float wheelSteps);

    QMatrix4x4 viewMatrix() const;
    QMatrix4x4 projectionMatrix(float aspectRatio) const;

    float yaw() const noexcept;
    float pitch() const noexcept;
    float distance() const noexcept;
    QVector3D center() const noexcept;
    QVector3D panOffset() const noexcept;

private:
    void clampState();

    float yaw_{-35.0F};
    float pitch_{25.0F};
    float distance_{10.0F};
    float fitRadius_{1.0F};
    QVector3D center_{};
    QVector3D panOffset_{};
};

} // namespace ptv2::qtui
