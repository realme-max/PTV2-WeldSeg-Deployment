#include "DetectionExportService.h"

#include "AppConfig.h"
#include "ScreenshotExportService.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <QTextStream>

namespace ptv2::qtui
{
namespace
{

bool writeJson(QString const& path, QJsonObject const& object, QString& error)
{
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        error = file.errorString();
        return false;
    }
    if (file.write(QJsonDocument(object).toJson(QJsonDocument::Indented)) < 0
        || !file.commit())
    {
        error = file.errorString();
        return false;
    }
    return true;
}

bool writePly(
    QString const& path,
    QtWeldResultViewModel const& result,
    int& vertexCount,
    QString& error)
{
    vertexCount = 0;
    for (auto const& point : result.points)
        if (point.label == 0) ++vertexCount;
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        error = file.errorString();
        return false;
    }
    QTextStream stream(&file);
    stream.setRealNumberNotation(QTextStream::SmartNotation);
    stream.setRealNumberPrecision(9);
    stream << "ply\nformat ascii 1.0\n";
    stream << "element vertex " << vertexCount << '\n';
    stream << "property float x\nproperty float y\nproperty float z\n";
    stream << "property int label\nproperty float confidence\nend_header\n";
    for (auto const& point : result.points)
    {
        if (point.label != 0) continue;
        stream << point.x << ' ' << point.y << ' ' << point.z << ' '
               << point.label << ' ' << point.confidence << '\n';
    }
    stream.flush();
    if (stream.status() != QTextStream::Ok || !file.commit())
    {
        error = file.errorString();
        return false;
    }
    return true;
}

bool writePrediction(
    QString const& path,
    QtWeldResultViewModel const& result,
    int& rows,
    QString& error)
{
    rows = 0;
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        error = file.errorString();
        return false;
    }
    QTextStream stream(&file);
    stream.setRealNumberNotation(QTextStream::SmartNotation);
    stream.setRealNumberPrecision(9);
    for (auto const& point : result.points)
    {
        stream << point.x << ' ' << point.y << ' ' << point.z << ' ' << point.label << '\n';
        ++rows;
    }
    stream.flush();
    if (stream.status() != QTextStream::Ok || !file.commit())
    {
        error = file.errorString();
        return false;
    }
    return true;
}

QJsonArray vector3(double x, double y, double z)
{
    return QJsonArray{QJsonValue(x), QJsonValue(y), QJsonValue(z)};
}

QString safeTaskId(QString taskId)
{
    if (taskId.isEmpty()) taskId = QStringLiteral("task");
    for (QChar& ch : taskId)
    {
        if (!ch.isLetterOrNumber() && ch != QLatin1Char('-') && ch != QLatin1Char('_'))
            ch = QLatin1Char('_');
    }
    return taskId;
}

} // namespace

DetectionExportResult DetectionExportService::exportTask(
    QtWeldResultViewModel const& result,
    QImage const& screenshot,
    QString const& exportRoot,
    DetectionExportIdentity const& identity)
{
    DetectionExportResult outcome;
    if (!result.success || result.points.size() != result.sampledPoints
        || result.sampledPoints <= 0)
    {
        outcome.error = QStringLiteral("No validated successful detection result is available");
        return outcome;
    }
    QString const timestamp = QDateTime::currentDateTime().toString(
        QStringLiteral("yyyyMMdd_HHmmss_zzz"));
    QString const directoryName = QStringLiteral("%1_%2").arg(safeTaskId(result.taskId), timestamp);
    QDir root(QFileInfo(exportRoot).absoluteFilePath());
    if (!QDir().mkpath(root.absolutePath()))
    {
        outcome.error = QStringLiteral("Cannot create export root: %1").arg(root.absolutePath());
        return outcome;
    }
    QString const temporaryPath = root.filePath(QStringLiteral(".%1.tmp").arg(directoryName));
    QString const finalPath = root.filePath(directoryName);
    QDir(temporaryPath).removeRecursively();
    if (!QDir().mkpath(temporaryPath))
    {
        outcome.error = QStringLiteral("Cannot create temporary export directory");
        return outcome;
    }
    auto fail = [&](QString const& file, QString const& detail) {
        outcome.failingFile = file;
        outcome.error = detail;
        QDir(temporaryPath).removeRecursively();
        return outcome;
    };

    QString error;
    QJsonObject resultJson;
    resultJson.insert(QStringLiteral("task_id"), result.taskId);
    resultJson.insert(QStringLiteral("source_cloud"), result.sourcePath);
    resultJson.insert(QStringLiteral("timestamp"), QDateTime::currentDateTime().toString(Qt::ISODateWithMs));
    resultJson.insert(QStringLiteral("application_version"), identity.applicationVersion);
    resultJson.insert(QStringLiteral("sdk_version"), identity.sdkVersion);
    resultJson.insert(QStringLiteral("engine_sha256"), identity.engineSha256);
    resultJson.insert(QStringLiteral("plugin_sha256"), identity.pluginSha256);
    resultJson.insert(QStringLiteral("original_point_count"), result.originalPoints);
    resultJson.insert(QStringLiteral("sampled_point_count"), result.sampledPoints);
    resultJson.insert(QStringLiteral("weld_point_count"), result.weldPoints);
    resultJson.insert(QStringLiteral("background_point_count"), result.sampledPoints - result.weldPoints);
    resultJson.insert(QStringLiteral("weld_ratio"), result.weldRatio);
    resultJson.insert(QStringLiteral("center"), vector3(result.centerX, result.centerY, result.centerZ));
    QJsonObject bbox;
    bbox.insert(QStringLiteral("min"), vector3(result.bboxMinX, result.bboxMinY, result.bboxMinZ));
    bbox.insert(QStringLiteral("max"), vector3(result.bboxMaxX, result.bboxMaxY, result.bboxMaxZ));
    resultJson.insert(QStringLiteral("bbox"), bbox);
    resultJson.insert(QStringLiteral("principal_direction"),
        vector3(result.principalDirectionX, result.principalDirectionY, result.principalDirectionZ));
    resultJson.insert(QStringLiteral("pca_length_mm"), result.lengthMm);
    QJsonObject timing;
    timing.insert(QStringLiteral("load_cloud_ms"), result.loadCloudMs);
    timing.insert(QStringLiteral("sampling_ms"), result.samplingMs);
    timing.insert(QStringLiteral("adjacency_build_ms"), result.adjacencyBuildMs);
    timing.insert(QStringLiteral("inference_cuda_ms"), result.inferenceCudaMs);
    timing.insert(QStringLiteral("inference_wall_ms"), result.inferenceWallMs);
    timing.insert(QStringLiteral("postprocess_ms"), result.postprocessMs);
    timing.insert(QStringLiteral("total_ms"), result.totalMs);
    resultJson.insert(QStringLiteral("timing"), timing);
    resultJson.insert(QStringLiteral("error_recorder_error_count"), result.errorRecorderErrors);
    resultJson.insert(QStringLiteral("result_status"), result.status);

    QString const resultPath = QDir(temporaryPath).filePath(QStringLiteral("weld_result.json"));
    if (!writeJson(resultPath, resultJson, error))
        return fail(resultPath, error);
    int plyVertices = 0;
    QString const plyPath = QDir(temporaryPath).filePath(QStringLiteral("weld_points.ply"));
    if (!writePly(plyPath, result, plyVertices, error))
        return fail(plyPath, error);
    int predictionRows = 0;
    QString const predictionPath = QDir(temporaryPath).filePath(QStringLiteral("prediction.txt"));
    if (!writePrediction(predictionPath, result, predictionRows, error))
        return fail(predictionPath, error);
    QString const imagePath = QDir(temporaryPath).filePath(QStringLiteral("detection_view.png"));
    ScreenshotExportResult const screenshotResult =
        ScreenshotExportService::savePng(screenshot, imagePath);
    if (!screenshotResult.success)
        return fail(imagePath, screenshotResult.error);

    QStringList const payloadNames{
        QStringLiteral("weld_result.json"),
        QStringLiteral("weld_points.ply"),
        QStringLiteral("prediction.txt"),
        QStringLiteral("detection_view.png")};
    QJsonArray files;
    for (QString const& name : payloadNames)
    {
        QString const path = QDir(temporaryPath).filePath(name);
        QString hashError;
        QString const hash = AppConfig::sha256File(path, hashError);
        if (!hashError.isEmpty()) return fail(path, hashError);
        QJsonObject entry;
        entry.insert(QStringLiteral("name"), name);
        entry.insert(QStringLiteral("size_bytes"), static_cast<double>(QFileInfo(path).size()));
        entry.insert(QStringLiteral("sha256"), hash);
        files.append(entry);
    }
    QJsonObject manifest;
    manifest.insert(QStringLiteral("export_status"), QStringLiteral("SUCCESS"));
    manifest.insert(QStringLiteral("task_id"), result.taskId);
    manifest.insert(QStringLiteral("source_cloud"), result.sourcePath);
    manifest.insert(QStringLiteral("engine_sha256"), identity.engineSha256);
    manifest.insert(QStringLiteral("plugin_sha256"), identity.pluginSha256);
    manifest.insert(QStringLiteral("ply_vertex_count"), plyVertices);
    manifest.insert(QStringLiteral("prediction_rows"), predictionRows);
    manifest.insert(QStringLiteral("screenshot_width"), screenshotResult.width);
    manifest.insert(QStringLiteral("screenshot_height"), screenshotResult.height);
    manifest.insert(QStringLiteral("files"), files);
    QString const manifestPath =
        QDir(temporaryPath).filePath(QStringLiteral("task_manifest.json"));
    if (!writeJson(manifestPath, manifest, error))
        return fail(manifestPath, error);

    if (QFileInfo(finalPath).exists() || !root.rename(QFileInfo(temporaryPath).fileName(), directoryName))
        return fail(finalPath, QStringLiteral("Cannot atomically promote export directory"));
    outcome.success = true;
    outcome.directory = finalPath;
    outcome.files = payloadNames;
    outcome.files.append(QStringLiteral("task_manifest.json"));
    return outcome;
}

bool DetectionExportService::verifyTask(QString const& taskDirectory, QString& error)
{
    error.clear();
    QFile manifestFile(QDir(taskDirectory).filePath(QStringLiteral("task_manifest.json")));
    if (!manifestFile.open(QIODevice::ReadOnly))
    {
        error = QStringLiteral("Cannot read task_manifest.json: %1").arg(manifestFile.errorString());
        return false;
    }
    QJsonParseError parseError;
    QJsonDocument const document = QJsonDocument::fromJson(manifestFile.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject())
    {
        error = QStringLiteral("Invalid task manifest: %1").arg(parseError.errorString());
        return false;
    }
    QJsonArray const files = document.object().value(QStringLiteral("files")).toArray();
    if (files.isEmpty())
    {
        error = QStringLiteral("Task manifest has no payload files");
        return false;
    }
    for (QJsonValue const& value : files)
    {
        QJsonObject const entry = value.toObject();
        QString const name = entry.value(QStringLiteral("name")).toString();
        QString const expectedHash = entry.value(QStringLiteral("sha256")).toString();
        QString const path = QDir(taskDirectory).filePath(name);
        if (!QFileInfo(path).isFile())
        {
            error = QStringLiteral("Exported payload is missing: %1").arg(name);
            return false;
        }
        QString hashError;
        QString const actualHash = AppConfig::sha256File(path, hashError);
        if (!hashError.isEmpty() || actualHash != expectedHash)
        {
            error = QStringLiteral("Exported payload hash mismatch: %1").arg(name);
            return false;
        }
    }
    return true;
}

} // namespace ptv2::qtui
