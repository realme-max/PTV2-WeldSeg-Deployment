#include "QtWeldResultViewModel.h"

#include <QDir>

namespace ptv2::qtui
{

QtWeldResultViewModel QtWeldResultViewModel::fromSdk(
    ptv2::weld::WeldResult const& result,
    QString const& sourcePath)
{
    QtWeldResultViewModel view;
    view.success = result.success;
    view.status = result.success ? QStringLiteral("SUCCESS") : QStringLiteral("POSTPROCESS_FAILED");
    view.sourcePath = QDir::toNativeSeparators(sourcePath);
    view.taskId = QString::fromStdString(result.task_id);
    view.originalPoints = result.original_points;
    view.sampledPoints = result.sampled_points;
    view.weldPoints = result.weld_points;
    view.weldRatio = result.weld_ratio;
    view.lengthMm = result.length_mm;
    view.centerX = result.center[0];
    view.centerY = result.center[1];
    view.centerZ = result.center[2];
    view.bboxMinX = result.bbox_min[0];
    view.bboxMinY = result.bbox_min[1];
    view.bboxMinZ = result.bbox_min[2];
    view.bboxMaxX = result.bbox_max[0];
    view.bboxMaxY = result.bbox_max[1];
    view.bboxMaxZ = result.bbox_max[2];
    view.principalDirectionX = result.principal_direction[0];
    view.principalDirectionY = result.principal_direction[1];
    view.principalDirectionZ = result.principal_direction[2];
    view.loadCloudMs = result.load_cloud_ms;
    view.samplingMs = result.sampling_ms;
    view.adjacencyBuildMs = result.adjacency_build_ms;
    view.inferenceCudaMs = result.inference_ms;
    view.inferenceWallMs = result.inference_wall_ms;
    view.postprocessMs = result.postprocess_ms;
    view.totalMs = result.total_ms;
    view.errorRecorderErrors = result.error_recorder_errors;
    view.points.reserve(static_cast<int>(result.points.size()));
    for (auto const& point : result.points)
    {
        view.points.append(QtWeldPointViewModel{
            point.x, point.y, point.z, point.label, point.confidence});
    }
    return view;
}

} // namespace ptv2::qtui
