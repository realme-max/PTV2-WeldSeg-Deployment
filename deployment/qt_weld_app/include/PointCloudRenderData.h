#pragma once

#include "QtWeldResultViewModel.h"

#include <QString>
#include <QVector>
#include <QVector3D>

namespace ptv2::qtui
{

struct RenderPoint
{
    QVector3D position;
    QVector3D color;
    float confidence{};
    int label{};
};

struct PointCloudRenderData
{
    QVector<RenderPoint> points;
    QVector3D bboxMin;
    QVector3D bboxMax;
    QVector3D weldCenter;
    QVector3D pcaStart;
    QVector3D pcaEnd;
    bool valid{false};
    QString error;
    double conversionMs{};

    static PointCloudRenderData fromResult(QtWeldResultViewModel const& result);
    bool validate(QString& detail) const;

    static QVector3D weldColor();
    static QVector3D backgroundColor();
};

} // namespace ptv2::qtui
