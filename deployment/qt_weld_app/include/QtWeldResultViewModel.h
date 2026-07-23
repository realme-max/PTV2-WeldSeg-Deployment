#pragma once

#include "WeldResult.h"

#include <QMetaType>
#include <QString>

namespace ptv2::qtui
{

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
    double loadCloudMs{};
    double samplingMs{};
    double adjacencyBuildMs{};
    double inferenceCudaMs{};
    double inferenceWallMs{};
    double postprocessMs{};
    double totalMs{};
    int errorRecorderErrors{};

    static QtWeldResultViewModel fromSdk(
        ptv2::weld::WeldResult const& result,
        QString const& sourcePath);
};

} // namespace ptv2::qtui

Q_DECLARE_METATYPE(ptv2::qtui::QtWeldResultViewModel)
