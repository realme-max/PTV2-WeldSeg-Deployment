#include "PointCloudRenderData.h"

#include <QElapsedTimer>

#include <cmath>

namespace ptv2::qtui
{
namespace
{

bool finite(QVector3D const& value)
{
    return std::isfinite(value.x()) && std::isfinite(value.y()) && std::isfinite(value.z());
}

} // namespace

QVector3D PointCloudRenderData::weldColor()
{
    return {1.0F, 0.25F, 0.08F};
}

QVector3D PointCloudRenderData::backgroundColor()
{
    return {0.18F, 0.55F, 0.95F};
}

PointCloudRenderData PointCloudRenderData::fromResult(QtWeldResultViewModel const& result)
{
    QElapsedTimer timer;
    timer.start();
    PointCloudRenderData data;
    if (!result.success)
    {
        data.error = QStringLiteral("Visualization requires a successful WeldDetector result");
        return data;
    }
    if (result.points.size() != result.sampledPoints)
    {
        data.error = QStringLiteral("Sampled point count and visualization vector size differ");
        return data;
    }

    data.points.reserve(result.points.size());
    for (QtWeldPointViewModel const& point : result.points)
    {
        data.points.append(RenderPoint{
            QVector3D(point.x, point.y, point.z),
            point.label == 0 ? weldColor() : backgroundColor(),
            point.confidence,
            point.label});
    }
    data.bboxMin = QVector3D(
        static_cast<float>(result.bboxMinX),
        static_cast<float>(result.bboxMinY),
        static_cast<float>(result.bboxMinZ));
    data.bboxMax = QVector3D(
        static_cast<float>(result.bboxMaxX),
        static_cast<float>(result.bboxMaxY),
        static_cast<float>(result.bboxMaxZ));
    data.weldCenter = QVector3D(
        static_cast<float>(result.centerX),
        static_cast<float>(result.centerY),
        static_cast<float>(result.centerZ));
    QVector3D const direction(
        static_cast<float>(result.principalDirectionX),
        static_cast<float>(result.principalDirectionY),
        static_cast<float>(result.principalDirectionZ));
    float const halfLength = static_cast<float>(result.lengthMm * 0.5);
    data.pcaStart = data.weldCenter - direction * halfLength;
    data.pcaEnd = data.weldCenter + direction * halfLength;
    QString detail;
    data.valid = data.validate(detail);
    data.error = detail;
    data.conversionMs = static_cast<double>(timer.nsecsElapsed()) / 1.0e6;
    return data;
}

bool PointCloudRenderData::validate(QString& detail) const
{
    if (points.isEmpty())
    {
        detail = QStringLiteral("Visualization point vector is empty");
        return false;
    }
    for (int index = 0; index < points.size(); ++index)
    {
        RenderPoint const& point = points[index];
        if (!finite(point.position) || !finite(point.color))
        {
            detail = QStringLiteral("Visualization point %1 contains NaN/Inf").arg(index);
            return false;
        }
        if (point.label != 0 && point.label != 1)
        {
            detail = QStringLiteral("Visualization point %1 has an invalid label").arg(index);
            return false;
        }
        if (!std::isfinite(point.confidence)
            || point.confidence < 0.0F || point.confidence > 1.0F)
        {
            detail = QStringLiteral("Visualization point %1 has invalid confidence").arg(index);
            return false;
        }
    }
    if (!finite(bboxMin) || !finite(bboxMax) || !finite(weldCenter)
        || !finite(pcaStart) || !finite(pcaEnd))
    {
        detail = QStringLiteral("Visualization geometry contains NaN/Inf");
        return false;
    }
    QVector3D const direction = pcaEnd - pcaStart;
    if (!(direction.length() > 0.0F) || !std::isfinite(direction.length()))
    {
        detail = QStringLiteral("Visualization PCA direction is invalid");
        return false;
    }
    detail.clear();
    return true;
}

} // namespace ptv2::qtui
