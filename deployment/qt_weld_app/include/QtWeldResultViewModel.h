#pragma once

#include "WeldResult.h"

#include <QMetaType>
#include <QString>
#include <QVector>

namespace ptv2::qtui
{

struct QtWeldPointViewModel
{
    float x{};
    float y{};
    float z{};
    int label{};
    float confidence{};
};

struct QtWeldResultViewModel
{
    bool success{};
    QString status;
    QString sourcePath;
    QString taskId;
    int originalPoints{};
    int sampledPoints{};
    int weldPoints{};
    double weldRatio{};
    double lengthMm{};
    double centerX{};
    double centerY{};
    double centerZ{};
    double bboxMinX{};
    double bboxMinY{};
    double bboxMinZ{};
    double bboxMaxX{};
    double bboxMaxY{};
    double bboxMaxZ{};
    double principalDirectionX{};
    double principalDirectionY{};
    double principalDirectionZ{};
    double loadCloudMs{};
    double samplingMs{};
    double adjacencyBuildMs{};
    double inferenceCudaMs{};
    double inferenceWallMs{};
    double postprocessMs{};
    double totalMs{};
    int errorRecorderErrors{};
    QVector<QtWeldPointViewModel> points;

    static QtWeldResultViewModel fromSdk(
        ptv2::weld::WeldResult const& result,
        QString const& sourcePath);
};

} // namespace ptv2::qtui

Q_DECLARE_METATYPE(ptv2::qtui::QtWeldResultViewModel)
