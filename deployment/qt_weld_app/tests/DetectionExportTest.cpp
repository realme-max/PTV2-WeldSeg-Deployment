#include "DetectionExportService.h"

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <QtTest>

class DetectionExportTest final : public QObject
{
    Q_OBJECT

private slots:
    void exportsValidatedContract()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        ptv2::qtui::QtWeldResultViewModel result;
        result.success = true;
        result.status = QStringLiteral("SUCCESS");
        result.taskId = QStringLiteral("weld_65");
        result.sourcePath = QStringLiteral("weld_65.txt");
        result.originalPoints = 4096;
        result.sampledPoints = 2048;
        result.weldPoints = 209;
        result.weldRatio = 0.10205078125;
        result.lengthMm = 57.1960525513;
        result.points.reserve(2048);
        for (int index = 0; index < 2048; ++index)
            result.points.append({float(index), 0.0F, 0.0F, index < 209 ? 0 : 1, 0.9F});
        QImage image(128, 96, QImage::Format_RGBA8888);
        image.fill(Qt::black);
        ptv2::qtui::DetectionExportIdentity identity;
        identity.applicationVersion = QStringLiteral("0.1.0");
        identity.sdkVersion = QStringLiteral("Phase 9D");
        identity.engineSha256 = QString(64, QLatin1Char('a'));
        identity.pluginSha256 = QString(64, QLatin1Char('b'));
        auto const exported = ptv2::qtui::DetectionExportService::exportTask(
            result, image, temporary.path(), identity);
        QVERIFY2(exported.success, qPrintable(exported.error));
        QCOMPARE(exported.files.size(), 5);
        QFile manifest(exported.directory + QStringLiteral("/task_manifest.json"));
        QVERIFY(manifest.open(QIODevice::ReadOnly));
        QJsonObject const json = QJsonDocument::fromJson(manifest.readAll()).object();
        QCOMPARE(json.value(QStringLiteral("ply_vertex_count")).toInt(), 209);
        QCOMPARE(json.value(QStringLiteral("prediction_rows")).toInt(), 2048);
        QCOMPARE(json.value(QStringLiteral("screenshot_width")).toInt(), 128);
        QCOMPARE(json.value(QStringLiteral("screenshot_height")).toInt(), 96);
        QString verifyError;
        QVERIFY(ptv2::qtui::DetectionExportService::verifyTask(
            exported.directory, verifyError));
        QFile corrupt(exported.directory + QStringLiteral("/prediction.txt"));
        QVERIFY(corrupt.open(QIODevice::Append));
        corrupt.write("corrupt\n");
        corrupt.close();
        QVERIFY(!ptv2::qtui::DetectionExportService::verifyTask(
            exported.directory, verifyError)); // Manifest corruption detected.
    }

    void invalidResultAndScreenshotFailClosed()
    {
        QTemporaryDir temporary;
        ptv2::qtui::QtWeldResultViewModel result;
        auto failed = ptv2::qtui::DetectionExportService::exportTask(
            result, QImage(), temporary.path(), {});
        QVERIFY(!failed.success); // 8: no successful result.
        result.success = true;
        result.sampledPoints = 1;
        result.points.append({0, 0, 0, 0, 1});
        failed = ptv2::qtui::DetectionExportService::exportTask(
            result, QImage(), temporary.path(), {});
        QVERIFY(!failed.success); // 9: invalid screenshot.
        QFile notDirectory(temporary.filePath(QStringLiteral("not_directory")));
        QVERIFY(notDirectory.open(QIODevice::WriteOnly));
        notDirectory.write("x");
        notDirectory.close();
        QImage image(8, 8, QImage::Format_RGBA8888);
        image.fill(Qt::black);
        failed = ptv2::qtui::DetectionExportService::exportTask(
            result, image, notDirectory.fileName(), {});
        QVERIFY(!failed.success); // Non-writable/invalid export root.
    }
};

QTEST_GUILESS_MAIN(DetectionExportTest)
#include "DetectionExportTest.moc"
