#include "PointCloudRenderData.h"
#include "QtWeldResultViewModel.h"
#include "WeldConfig.h"
#include "WeldDetector.h"
#include "WeldResult.h"
#include "WeldStatus.h"

#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QtTest>

#include <cmath>

namespace
{

struct Options
{
    QString engine;
    QString plugin;
    QString cloud;
    QString output;
};

QString required(QCommandLineParser const& parser, QCommandLineOption const& option)
{
    QString const value = parser.value(option).trimmed();
    if (value.isEmpty()) qFatal("Missing --%s", qPrintable(option.names().first()));
    return QFileInfo(value).absoluteFilePath();
}

class PointCloudRenderDataTest final : public QObject
{
    Q_OBJECT

public:
    explicit PointCloudRenderDataTest(Options options) : options_(std::move(options)) {}

private slots:
    void buildFromRealSdkResult()
    {
        ptv2::weld::WeldConfig config;
        config.engine_path = options_.engine.toStdString();
        config.plugin_path = options_.plugin.toStdString();
        ptv2::weld::WeldDetector detector;
        QVERIFY(detector.initialize(config) == ptv2::weld::WeldStatus::SUCCESS);
        ptv2::weld::WeldResult sdkResult;
        QVERIFY(detector.detect(options_.cloud.toStdString(), sdkResult)
            == ptv2::weld::WeldStatus::SUCCESS);

        auto view = ptv2::qtui::QtWeldResultViewModel::fromSdk(sdkResult, options_.cloud);
        ptv2::qtui::PointCloudRenderData data =
            ptv2::qtui::PointCloudRenderData::fromResult(view);
        QVERIFY2(data.valid, qPrintable(data.error));
        QCOMPARE(data.points.size(), 2048);
        int weld = 0;
        int background = 0;
        for (auto const& point : data.points)
        {
            QVERIFY(std::isfinite(point.position.x()));
            QVERIFY(std::isfinite(point.position.y()));
            QVERIFY(std::isfinite(point.position.z()));
            QVERIFY(std::isfinite(point.color.x()));
            QVERIFY(std::isfinite(point.color.y()));
            QVERIFY(std::isfinite(point.color.z()));
            QVERIFY(std::isfinite(point.confidence));
            QVERIFY(point.confidence >= 0.0F && point.confidence <= 1.0F);
            QVERIFY(point.label == 0 || point.label == 1);
            if (point.label == 0) ++weld;
            else ++background;
        }
        QCOMPARE(weld, 209);
        QCOMPARE(background, 1839);
        QVERIFY((data.bboxMin - QVector3D(
            -25.9417991638F, -5.2817997932F, 268.566711426F)).length() < 1.0e-5F);
        QVERIFY((data.bboxMax - QVector3D(
            30.8199996948F, 6.1363000870F, 277.464599609F)).length() < 1.0e-5F);
        QVERIFY((data.weldCenter - QVector3D(
            3.5052173138F, 0.2649071813F, 272.750274658F)).length() < 1.0e-5F);
        QVERIFY(std::isfinite(data.pcaStart.x()) && std::isfinite(data.pcaEnd.x()));
        QVERIFY(std::abs((data.pcaEnd - data.pcaStart).length() - 57.19605255F) < 1.0e-4F);

        QVector3D const copiedPosition = data.points[0].position;
        view.points[0].x += 1000.0F;
        QCOMPARE(data.points[0].position, copiedPosition);

        QJsonObject report;
        report.insert(QStringLiteral("status"), QStringLiteral("PASS"));
        report.insert(QStringLiteral("render_point_count"), data.points.size());
        report.insert(QStringLiteral("weld_point_count"), weld);
        report.insert(QStringLiteral("background_point_count"), background);
        report.insert(QStringLiteral("positions_finite"), true);
        report.insert(QStringLiteral("colors_finite"), true);
        report.insert(QStringLiteral("confidence_contract"), true);
        report.insert(QStringLiteral("geometry_within_phase9d_tolerance"), true);
        report.insert(QStringLiteral("pca_endpoints_finite"), true);
        report.insert(QStringLiteral("copy_has_no_source_alias"), true);
        report.insert(QStringLiteral("conversion_ms"), data.conversionMs);
        QFile output(options_.output);
        QVERIFY(output.open(QIODevice::WriteOnly | QIODevice::Truncate));
        QVERIFY(output.write(QJsonDocument(report).toJson(QJsonDocument::Indented)) > 0);
    }

private:
    Options options_;
};

} // namespace

int main(int argc, char** argv)
{
    QStringList raw;
    for (int index = 0; index < argc; ++index) raw.append(QString::fromLocal8Bit(argv[index]));
    QCoreApplication application(argc, argv);
    QCommandLineParser parser;
    QCommandLineOption engine(QStringList() << "engine", {}, "path");
    QCommandLineOption plugin(QStringList() << "plugin", {}, "path");
    QCommandLineOption cloud(QStringList() << "cloud", {}, "path");
    QCommandLineOption output(QStringList() << "output", {}, "path");
    parser.addOption(engine);
    parser.addOption(plugin);
    parser.addOption(cloud);
    parser.addOption(output);
    parser.process(raw);
    Options options{required(parser, engine), required(parser, plugin),
                    required(parser, cloud), required(parser, output)};
    PointCloudRenderDataTest test(std::move(options));
    int testArgc = 1;
    char* testArgv[] = {argv[0], nullptr};
    return QTest::qExec(&test, testArgc, testArgv);
}

#include "PointCloudRenderDataTest.moc"
