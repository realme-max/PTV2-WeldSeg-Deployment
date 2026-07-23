#include "MainWindow.h"
#include "PointCloudView.h"
#include "QtWeldResultViewModel.h"
#include "WeldConfig.h"
#include "WeldDetectionWorker.h"

#include <QApplication>
#include <QCheckBox>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QPushButton>
#include <QSignalSpy>
#include <QSurfaceFormat>
#include <QTextStream>
#include <QThread>
#include <QTimer>
#include <QtTest>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>

namespace
{

using ptv2::qtui::QtWeldResultViewModel;
using ptv2::qtui::WeldDetectionWorker;

struct TestOptions
{
    QString engine;
    QString plugin;
    QString engineSha256;
    QString cloud;
    QString smallCloud;
    QString artifactDir;
};

QString required(
    QCommandLineParser const& parser,
    QCommandLineOption const& option)
{
    QString const value = parser.value(option).trimmed();
    if (value.isEmpty())
        qFatal("Missing required option --%s", qPrintable(option.names().first()));
    return QFileInfo(value).absoluteFilePath();
}

ptv2::weld::WeldConfig makeConfig(QString const& engine, QString const& plugin)
{
    ptv2::weld::WeldConfig config;
    config.engine_path = QFileInfo(engine).absoluteFilePath().toStdString();
    config.plugin_path = QFileInfo(plugin).absoluteFilePath().toStdString();
    return config;
}

bool writeJson(QString const& path, QJsonObject const& object)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
        return false;
    return file.write(QJsonDocument(object).toJson(QJsonDocument::Indented)) >= 0;
}

bool makeSmallCloud(QString const& source, QString const& destination)
{
    QFile input(source);
    if (!input.open(QIODevice::ReadOnly | QIODevice::Text))
        return false;
    QFile output(destination);
    QDir().mkpath(QFileInfo(destination).absolutePath());
    if (!output.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate))
        return false;
    QTextStream reader(&input);
    QTextStream writer(&output);
    for (int index = 0; index < 2047; ++index)
    {
        if (reader.atEnd()) return false;
        writer << reader.readLine() << '\n';
    }
    return writer.status() == QTextStream::Ok;
}

class WorkerHarness final
{
public:
    WorkerHarness(ptv2::weld::WeldConfig config, QString expectedSha256)
    {
        worker = new WeldDetectionWorker(std::move(config), std::move(expectedSha256));
        worker->moveToThread(&thread);
        QObject::connect(&thread, &QThread::started, worker, &WeldDetectionWorker::initialize);
        QObject::connect(&thread, &QThread::finished, worker, &QObject::deleteLater);
    }

    WorkerHarness(WorkerHarness const&) = delete;
    WorkerHarness& operator=(WorkerHarness const&) = delete;

    ~WorkerHarness()
    {
        stop();
    }

    void start()
    {
        thread.start();
    }

    void stop()
    {
        if (!thread.isRunning()) return;
        QMetaObject::invokeMethod(worker, "shutdown", Qt::BlockingQueuedConnection);
        thread.quit();
        thread.wait();
        worker = nullptr;
    }

    QThread thread;
    WeldDetectionWorker* worker{};
};

class QtSdkIntegrationSmoke final : public QObject
{
    Q_OBJECT

public:
    explicit QtSdkIntegrationSmoke(TestOptions options)
        : options_(std::move(options))
    {
    }

private slots:
    void initTestCase()
    {
        qRegisterMetaType<QtWeldResultViewModel>("ptv2::qtui::QtWeldResultViewModel");
        QVERIFY2(QFileInfo(options_.engine).isFile(), qPrintable(options_.engine));
        QVERIFY2(QFileInfo(options_.plugin).isFile(), qPrintable(options_.plugin));
        QVERIFY2(QFileInfo(options_.cloud).isFile(), qPrintable(options_.cloud));
        QVERIFY(QDir().mkpath(options_.artifactDir));
        if (!QFileInfo(options_.smallCloud).isFile())
            QVERIFY2(makeSmallCloud(options_.cloud, options_.smallCloud), "Failed to create 2047-point cloud");
    }

    void mainWindowConstructsAndShutsDown()
    {
        auto window = std::make_unique<ptv2::qtui::MainWindow>(
            makeConfig(options_.engine, options_.plugin),
            options_.engineSha256,
            options_.cloud);
        window->show();
        QVERIFY(window->isVisible());

        auto* status = window->findChild<QLabel*>(QStringLiteral("initializeStatus"));
        auto* detect = window->findChild<QPushButton*>(QStringLiteral("detectButton"));
        auto* sampled = window->findChild<QLabel*>(QStringLiteral("sampled_points"));
        auto* weld = window->findChild<QLabel*>(QStringLiteral("weld_points"));
        auto* ratio = window->findChild<QLabel*>(QStringLiteral("weld_ratio"));
        auto* length = window->findChild<QLabel*>(QStringLiteral("length_mm"));
        auto* errors = window->findChild<QLabel*>(QStringLiteral("error_recorder_errors"));
        auto* pointCloud = window->findChild<ptv2::qtui::PointCloudView*>(
            QStringLiteral("pointCloudView"));
        auto* resetView = window->findChild<QPushButton*>(QStringLiteral("resetViewButton"));
        auto* bboxToggle = window->findChild<QCheckBox*>(QStringLiteral("showBboxCheck"));
        auto* pcaToggle = window->findChild<QCheckBox*>(QStringLiteral("showPcaCheck"));
        QVERIFY(status != nullptr);
        QVERIFY(detect != nullptr);
        QVERIFY(sampled != nullptr);
        QVERIFY(weld != nullptr);
        QVERIFY(ratio != nullptr);
        QVERIFY(length != nullptr);
        QVERIFY(errors != nullptr);
        QVERIFY(pointCloud != nullptr);
        QVERIFY(resetView != nullptr);
        QVERIFY(bboxToggle != nullptr);
        QVERIFY(pcaToggle != nullptr);
        QVERIFY2(waitUntil([&] {
            return pointCloud->openGLInitialized() || !pointCloud->visualizationError().isEmpty();
        }, 30000), "OpenGL initialization timed out");
        QVERIFY2(pointCloud->openGLInitialized(), qPrintable(pointCloud->visualizationError()));
        QVERIFY2(waitUntil([&] { return status->text() == QStringLiteral("SUCCESS"); }, 120000),
            "MainWindow SDK initialization timed out");
        QVERIFY(detect->isEnabled());

        bool responsiveEventProcessed = false;
        QTimer::singleShot(1, [&] { responsiveEventProcessed = true; });
        QElapsedTimer firstRefreshTimer;
        firstRefreshTimer.start();
        QTest::mouseClick(detect, Qt::LeftButton);
        QVERIFY2(waitUntil([&] { return !detect->isEnabled(); }, 5000),
            "Detect did not enter active state");
        QVERIFY2(waitUntil([&] {
            return detect->isEnabled() && weld->text() == QStringLiteral("209");
        }, 120000), "First MainWindow detection timed out");
        QVERIFY(responsiveEventProcessed);
        QCOMPARE(sampled->text(), QStringLiteral("2048"));
        QCOMPARE(errors->text(), QStringLiteral("0"));
        QCOMPARE(pointCloud->renderedPointCount(), 2048);
        manual_.insert(QStringLiteral("first_detection_refresh_ms"),
            static_cast<double>(firstRefreshTimer.nsecsElapsed()) / 1.0e6);
        QCOMPARE(pointCloud->weldPointCount(), 209);
        QCOMPARE(pointCloud->backgroundPointCount(), 1839);
        QCOMPARE(pointCloud->lastGlError(), 0U);
        QVERIFY(std::abs(ratio->text().toDouble() - 0.10205078125) < 1.0e-7);
        QVERIFY(std::abs(length->text().toDouble() - 57.19605255) < 1.0e-3);

        QElapsedTimer secondRefreshTimer;
        secondRefreshTimer.start();
        QTest::mouseClick(detect, Qt::LeftButton);
        QVERIFY2(waitUntil([&] { return !detect->isEnabled(); }, 5000),
            "Second Detect did not enter active state");
        QVERIFY2(waitUntil([&] { return detect->isEnabled(); }, 120000),
            "Second MainWindow detection timed out");
        QCOMPARE(weld->text(), QStringLiteral("209"));
        QCOMPARE(errors->text(), QStringLiteral("0"));
        QCOMPARE(pointCloud->renderedPointCount(), 2048);
        manual_.insert(QStringLiteral("second_detection_refresh_ms"),
            static_cast<double>(secondRefreshTimer.nsecsElapsed()) / 1.0e6);
        QVERIFY(QMetaObject::invokeMethod(
            window.get(), "onDetectionFailed", Qt::DirectConnection,
            Q_ARG(QString, QStringLiteral("POINTCLOUD_LOAD_FAILED")),
            Q_ARG(QString, QStringLiteral("missing cloud test"))));
        QCOMPARE(pointCloud->renderedPointCount(), 2048);
        manual_.insert(QStringLiteral("missing_cloud_preserved_previous_view"), true);
        QVERIFY(QMetaObject::invokeMethod(
            window.get(), "onDetectionFailed", Qt::DirectConnection,
            Q_ARG(QString, QStringLiteral("PREPROCESS_FAILED")),
            Q_ARG(QString, QStringLiteral("2047 point test"))));
        QCOMPARE(pointCloud->renderedPointCount(), 2048);
        manual_.insert(QStringLiteral("small_cloud_preserved_previous_view"), true);
        QTest::mouseClick(resetView, Qt::LeftButton);
        QTest::mouseClick(bboxToggle, Qt::LeftButton);
        QVERIFY(!bboxToggle->isChecked());
        QTest::mouseClick(bboxToggle, Qt::LeftButton);
        QTest::mouseClick(pcaToggle, Qt::LeftButton);
        QVERIFY(!pcaToggle->isChecked());
        QTest::mouseClick(pcaToggle, Qt::LeftButton);

        manual_.insert(QStringLiteral("application_started"), true);
        manual_.insert(QStringLiteral("initialization_status_displayed"), true);
        manual_.insert(QStringLiteral("ui_responsive_during_detection"), responsiveEventProcessed);
        manual_.insert(QStringLiteral("first_detection_succeeded"), true);
        manual_.insert(QStringLiteral("second_detection_succeeded"), true);
        manual_.insert(QStringLiteral("sampled_points"), sampled->text().toInt());
        manual_.insert(QStringLiteral("weld_points"), weld->text().toInt());
        manual_.insert(QStringLiteral("weld_ratio"), ratio->text().toDouble());
        manual_.insert(QStringLiteral("length_mm"), length->text().toDouble());
        manual_.insert(QStringLiteral("error_recorder_errors"), errors->text().toInt());
        manual_.insert(QStringLiteral("rendered_points"), pointCloud->renderedPointCount());
        manual_.insert(QStringLiteral("weld_colored_points"), pointCloud->weldPointCount());
        manual_.insert(QStringLiteral("background_colored_points"), pointCloud->backgroundPointCount());
        manual_.insert(QStringLiteral("opengl_version"), pointCloud->openGLVersion());
        manual_.insert(QStringLiteral("opengl_renderer"), pointCloud->openGLRenderer());
        manual_.insert(QStringLiteral("opengl_vendor"), pointCloud->openGLVendor());
        manual_.insert(QStringLiteral("bbox_toggle"), true);
        manual_.insert(QStringLiteral("pca_toggle"), true);
        manual_.insert(QStringLiteral("reset_view"), true);
        manual_.insert(QStringLiteral("second_detection_replaced_not_duplicated"),
            pointCloud->renderedPointCount() == 2048);
        window.reset();
        manual_.insert(QStringLiteral("clean_close"), true);
        automated_.insert(QStringLiteral("main_window_construct_and_shutdown"), true);
    }

    void validInitializationAndTwoDetections()
    {
        validHarness_ = std::make_unique<WorkerHarness>(
            makeConfig(options_.engine, options_.plugin), options_.engineSha256);
        QSignalSpy initialized(validHarness_->worker, &WeldDetectionWorker::initializationFinished);
        validHarness_->start();
        QVERIFY2(initialized.wait(120000), "Timed out waiting for SDK initialization");
        QCOMPARE(initialized.count(), 1);
        QCOMPARE(initialized.at(0).at(0).toString(), QStringLiteral("SUCCESS"));
        automated_.insert(QStringLiteral("sdk_initialize_once"), true);

        QtWeldResultViewModel first = detectOnce(*validHarness_, options_.cloud);
        verifyWeld65(first);
        QtWeldResultViewModel second = detectOnce(*validHarness_, options_.cloud);
        verifyWeld65(second);
        QCOMPARE(first.weldPoints, second.weldPoints);
        QVERIFY(std::abs(first.lengthMm - second.lengthMm) < 1.0e-6);
        automated_.insert(QStringLiteral("two_sequential_detections"), true);
        referenceResult_ = first;
    }

    void concurrentRequestIsRejected()
    {
        QVERIFY(validHarness_ != nullptr);
        QSignalSpy success(validHarness_->worker, &WeldDetectionWorker::detectionSucceeded);
        bool const firstAccepted = validHarness_->worker->requestDetection(options_.cloud);
        bool const secondAccepted = validHarness_->worker->requestDetection(options_.cloud);
        QVERIFY(firstAccepted);
        QVERIFY(!secondAccepted);
        QVERIFY2(success.wait(120000), "Timed out waiting for concurrent-request control detection");
        QCOMPARE(success.count(), 1);
        automated_.insert(QStringLiteral("concurrent_request_rejected"), true);
        appendFailureCase(
            QStringLiteral("second_detect_while_active"),
            QStringLiteral("REQUEST_REJECTED"),
            QStringLiteral("REQUEST_REJECTED"),
            true);
    }

    void detectionFailuresFailClosed()
    {
        QVERIFY(validHarness_ != nullptr);
        verifyDetectionFailure(
            *validHarness_,
            QFileInfo(options_.artifactDir, QStringLiteral("missing_cloud.txt")).absoluteFilePath(),
            QStringLiteral("POINTCLOUD_LOAD_FAILED"),
            QStringLiteral("missing_cloud"));
        verifyDetectionFailure(
            *validHarness_,
            options_.smallCloud,
            QStringLiteral("PREPROCESS_FAILED"),
            QStringLiteral("point_count_below_2048"));
    }

    void initializationFailuresFailClosed()
    {
        QString const missingEngine =
            QFileInfo(options_.artifactDir, QStringLiteral("missing.plan")).absoluteFilePath();
        QString const missingPlugin =
            QFileInfo(options_.artifactDir, QStringLiteral("missing.dll")).absoluteFilePath();
        verifyInitializationFailure(
            makeConfig(missingEngine, options_.plugin),
            options_.engineSha256,
            QStringLiteral("ENGINE_LOAD_FAILED"),
            QStringLiteral("missing_engine"));
        verifyInitializationFailure(
            makeConfig(options_.engine, missingPlugin),
            options_.engineSha256,
            QStringLiteral("PLUGIN_LOAD_FAILED"),
            QStringLiteral("missing_plugin"));
        verifyInitializationFailure(
            makeConfig(options_.engine, options_.plugin),
            QString(64, QLatin1Char('0')),
            QStringLiteral("ENGINE_LOAD_FAILED"),
            QStringLiteral("wrong_engine_sha256"));
    }

    void preInitializationDetectionFailsClosed()
    {
        WeldDetectionWorker worker(
            makeConfig(options_.engine, options_.plugin), options_.engineSha256);
        QSignalSpy failure(&worker, &WeldDetectionWorker::detectionFailed);
        QVERIFY(worker.requestDetection(options_.cloud));
        QVERIFY2(failure.wait(5000), "Pre-initialization request was not rejected");
        QCOMPARE(failure.at(0).at(0).toString(), QStringLiteral("INVALID_CONFIG"));
        appendFailureCase(
            QStringLiteral("detection_before_initialize"),
            QStringLiteral("INVALID_CONFIG"),
            failure.at(0).at(0).toString(),
            true);
    }

    void cleanWorkerShutdown()
    {
        QVERIFY(validHarness_ != nullptr);
        QSignalSpy shutdown(validHarness_->worker, &WeldDetectionWorker::shutdownFinished);
        QVERIFY(QMetaObject::invokeMethod(
            validHarness_->worker, "shutdown", Qt::BlockingQueuedConnection));
        QCOMPARE(shutdown.count(), 1);
        validHarness_->thread.quit();
        QVERIFY(validHarness_->thread.wait(30000));
        validHarness_->worker = nullptr;
        validHarness_.reset();
        automated_.insert(QStringLiteral("worker_clean_shutdown"), true);
    }

    void cleanupTestCase()
    {
        if (validHarness_ != nullptr) validHarness_.reset();

        QJsonObject automatedReport;
        automatedReport.insert(QStringLiteral("status"), allObjectFlagsTrue(automated_)
            ? QStringLiteral("PASS") : QStringLiteral("FAILED"));
        automatedReport.insert(QStringLiteral("checks"), automated_);
        automatedReport.insert(QStringLiteral("sample_id"), QStringLiteral("weld_65"));
        automatedReport.insert(QStringLiteral("engine_sha256"), options_.engineSha256);
        QVERIFY(writeJson(
            QDir(options_.artifactDir).filePath(QStringLiteral("automated_smoke_result.json")),
            automatedReport));

        QJsonObject manualReport = manual_;
        manualReport.insert(QStringLiteral("status"),
            manual_.value(QStringLiteral("application_started")).toBool()
                && manual_.value(QStringLiteral("first_detection_succeeded")).toBool()
                && manual_.value(QStringLiteral("second_detection_succeeded")).toBool()
                && manual_.value(QStringLiteral("clean_close")).toBool()
            ? QStringLiteral("PASS") : QStringLiteral("FAILED"));
        manualReport.insert(QStringLiteral("method"),
            QStringLiteral("scripted visible Qt Widgets smoke using actual MainWindow controls"));
        QVERIFY(writeJson(
            QDir(options_.artifactDir).filePath(QStringLiteral("manual_smoke_result.json")),
            manualReport));

        int passedCases = 0;
        for (QJsonValue const& value : failClosed_)
        {
            if (value.toObject().value(QStringLiteral("passed")).toBool()) ++passedCases;
        }
        QJsonObject failClosedReport;
        failClosedReport.insert(QStringLiteral("status"),
            passedCases == failClosed_.size() ? QStringLiteral("PASS") : QStringLiteral("FAILED"));
        failClosedReport.insert(QStringLiteral("tested_cases"), failClosed_.size());
        failClosedReport.insert(QStringLiteral("passed_cases"), passedCases);
        failClosedReport.insert(QStringLiteral("cases"), failClosed_);
        QVERIFY(writeJson(
            QDir(options_.artifactDir).filePath(QStringLiteral("fail_closed_result.json")),
            failClosedReport));

        QJsonObject compatibility;
        compatibility.insert(QStringLiteral("status"), referenceResult_.success
            ? QStringLiteral("PASS") : QStringLiteral("FAILED"));
        compatibility.insert(QStringLiteral("original_points"), referenceResult_.originalPoints);
        compatibility.insert(QStringLiteral("sampled_points"), referenceResult_.sampledPoints);
        compatibility.insert(QStringLiteral("weld_points"), referenceResult_.weldPoints);
        compatibility.insert(QStringLiteral("weld_ratio"), referenceResult_.weldRatio);
        compatibility.insert(QStringLiteral("length_mm"), referenceResult_.lengthMm);
        compatibility.insert(QStringLiteral("error_recorder_errors"), referenceResult_.errorRecorderErrors);
        compatibility.insert(QStringLiteral("expected_weld_points"), 209);
        compatibility.insert(QStringLiteral("expected_length_mm"), 57.19605255);
        compatibility.insert(QStringLiteral("length_abs_error"),
            std::abs(referenceResult_.lengthMm - 57.19605255));
        QJsonArray center;
        center.append(referenceResult_.centerX);
        center.append(referenceResult_.centerY);
        center.append(referenceResult_.centerZ);
        compatibility.insert(QStringLiteral("center"), center);
        QJsonArray bboxMin;
        bboxMin.append(referenceResult_.bboxMinX);
        bboxMin.append(referenceResult_.bboxMinY);
        bboxMin.append(referenceResult_.bboxMinZ);
        compatibility.insert(QStringLiteral("bbox_min"), bboxMin);
        QJsonArray bboxMax;
        bboxMax.append(referenceResult_.bboxMaxX);
        bboxMax.append(referenceResult_.bboxMaxY);
        bboxMax.append(referenceResult_.bboxMaxZ);
        compatibility.insert(QStringLiteral("bbox_max"), bboxMax);
        double const geometryMaxError = std::max({
            std::abs(referenceResult_.centerX - 3.5052173137664795),
            std::abs(referenceResult_.centerY - 0.26490718126296997),
            std::abs(referenceResult_.centerZ - 272.75027465820312),
            std::abs(referenceResult_.bboxMinX - -25.941799163818359),
            std::abs(referenceResult_.bboxMinY - -5.2817997932434082),
            std::abs(referenceResult_.bboxMinZ - 268.56671142578125),
            std::abs(referenceResult_.bboxMaxX - 30.819999694824219),
            std::abs(referenceResult_.bboxMaxY - 6.1363000869750977),
            std::abs(referenceResult_.bboxMaxZ - 277.464599609375),
            std::abs(referenceResult_.lengthMm - 57.196052551269531),
        });
        compatibility.insert(QStringLiteral("maximum_geometry_error"), geometryMaxError);
        compatibility.insert(QStringLiteral("geometry_within_1e-5"), geometryMaxError < 1.0e-5);
        QVERIFY(writeJson(
            QDir(options_.artifactDir).filePath(QStringLiteral("phase9d_compatibility.json")),
            compatibility));
    }

private:
    QtWeldResultViewModel detectOnce(WorkerHarness& harness, QString const& cloud)
    {
        QSignalSpy success(harness.worker, &WeldDetectionWorker::detectionSucceeded);
        QSignalSpy failure(harness.worker, &WeldDetectionWorker::detectionFailed);
        if (!harness.worker->requestDetection(cloud))
        {
            QTest::qFail("Detection request was rejected", __FILE__, __LINE__);
            return {};
        }
        if (!success.wait(120000))
        {
            QTest::qFail("Timed out waiting for weld detection", __FILE__, __LINE__);
            return {};
        }
        if (failure.count() != 0 || success.count() != 1)
        {
            QTest::qFail("Unexpected success/failure signal counts", __FILE__, __LINE__);
            return {};
        }
        return qvariant_cast<QtWeldResultViewModel>(success.at(0).at(0));
    }

    void verifyWeld65(QtWeldResultViewModel const& result)
    {
        QVERIFY(result.success);
        QCOMPARE(result.status, QStringLiteral("SUCCESS"));
        QCOMPARE(result.originalPoints, 2048);
        QCOMPARE(result.sampledPoints, 2048);
        QCOMPARE(result.weldPoints, 209);
        QVERIFY(std::abs(result.weldRatio - 0.10205078125) < 1.0e-7);
        QVERIFY(std::abs(result.lengthMm - 57.19605255) < 1.0e-3);
        QCOMPARE(result.errorRecorderErrors, 0);
        QVERIFY(std::isfinite(result.totalMs));
        QVERIFY(result.totalMs > 0.0);
    }

    void verifyDetectionFailure(
        WorkerHarness& harness,
        QString const& cloud,
        QString const& expectedStatus,
        QString const& caseName)
    {
        QSignalSpy failure(harness.worker, &WeldDetectionWorker::detectionFailed);
        QSignalSpy success(harness.worker, &WeldDetectionWorker::detectionSucceeded);
        QVERIFY(harness.worker->requestDetection(cloud));
        QVERIFY2(failure.wait(30000), qPrintable(caseName));
        QCOMPARE(success.count(), 0);
        QString const actualStatus = failure.at(0).at(0).toString();
        QCOMPARE(actualStatus, expectedStatus);
        appendFailureCase(caseName, expectedStatus, actualStatus, true);
    }

    void verifyInitializationFailure(
        ptv2::weld::WeldConfig config,
        QString const& expectedSha,
        QString const& expectedStatus,
        QString const& caseName)
    {
        WorkerHarness harness(std::move(config), expectedSha);
        QSignalSpy initialized(harness.worker, &WeldDetectionWorker::initializationFinished);
        harness.start();
        QVERIFY2(initialized.wait(30000), qPrintable(caseName));
        QCOMPARE(initialized.count(), 1);
        QString const actualStatus = initialized.at(0).at(0).toString();
        QCOMPARE(actualStatus, expectedStatus);
        appendFailureCase(caseName, expectedStatus, actualStatus, true);
        harness.stop();
    }

    void appendFailureCase(
        QString const& name,
        QString const& expected,
        QString const& actual,
        bool passed)
    {
        QJsonObject item;
        item.insert(QStringLiteral("case"), name);
        item.insert(QStringLiteral("expected_status"), expected);
        item.insert(QStringLiteral("reported_status"), actual);
        item.insert(QStringLiteral("passed"), passed);
        item.insert(QStringLiteral("fallback"), QStringLiteral("NONE"));
        failClosed_.append(item);
    }

    static bool allObjectFlagsTrue(QJsonObject const& object)
    {
        return std::all_of(
            object.constBegin(),
            object.constEnd(),
            [](QJsonValue const& value) { return value.toBool(); });
    }

    static bool waitUntil(std::function<bool()> const& predicate, int timeoutMs)
    {
        QElapsedTimer timer;
        timer.start();
        while (timer.elapsed() < timeoutMs)
        {
            if (predicate()) return true;
            QCoreApplication::processEvents(QEventLoop::AllEvents, 25);
            QTest::qWait(10);
        }
        return predicate();
    }

    TestOptions options_;
    std::unique_ptr<WorkerHarness> validHarness_;
    QJsonObject automated_;
    QJsonObject manual_;
    QJsonArray failClosed_;
    QtWeldResultViewModel referenceResult_;
};

} // namespace

int main(int argc, char** argv)
{
    QStringList rawArguments;
    rawArguments.reserve(argc);
    for (int index = 0; index < argc; ++index)
        rawArguments.append(QString::fromLocal8Bit(argv[index]));
    QCoreApplication::setAttribute(Qt::AA_Use96Dpi, true);
    QCoreApplication::setAttribute(Qt::AA_UseDesktopOpenGL, true);
    QSurfaceFormat format;
    format.setRenderableType(QSurfaceFormat::OpenGL);
    format.setVersion(3, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    QSurfaceFormat::setDefaultFormat(format);
    QApplication application(argc, argv);
    application.setApplicationName(QStringLiteral("QtSdkIntegrationSmoke"));

    QCommandLineParser parser;
    parser.addHelpOption();
    QCommandLineOption engine(QStringList() << QStringLiteral("engine"), {}, QStringLiteral("path"));
    QCommandLineOption plugin(QStringList() << QStringLiteral("plugin"), {}, QStringLiteral("path"));
    QCommandLineOption sha(
        QStringList() << QStringLiteral("engine-sha256"), {}, QStringLiteral("sha256"));
    QCommandLineOption cloud(QStringList() << QStringLiteral("cloud"), {}, QStringLiteral("path"));
    QCommandLineOption smallCloud(
        QStringList() << QStringLiteral("small-cloud"), {}, QStringLiteral("path"));
    QCommandLineOption artifactDir(
        QStringList() << QStringLiteral("artifact-dir"), {}, QStringLiteral("path"));
    parser.addOption(engine);
    parser.addOption(plugin);
    parser.addOption(sha);
    parser.addOption(cloud);
    parser.addOption(smallCloud);
    parser.addOption(artifactDir);
    parser.process(rawArguments);

    QString const engineValue = required(parser, engine);
    QString const pluginValue = required(parser, plugin);
    QString const shaValue = parser.value(sha).trimmed().toLower();
    QString const cloudValue = required(parser, cloud);
    QString const smallCloudValue = required(parser, smallCloud);
    QString const artifactDirValue = required(parser, artifactDir);
    TestOptions options{
        engineValue,
        pluginValue,
        shaValue,
        cloudValue,
        smallCloudValue,
        artifactDirValue,
    };
    if (options.engineSha256.size() != 64)
        qFatal("engine-sha256 must contain exactly 64 hexadecimal characters");

    QtSdkIntegrationSmoke test(std::move(options));
    int const testArgc = 1;
    char* testArgv[] = {argv[0], nullptr};
    return QTest::qExec(&test, testArgc, testArgv);
}

#include "QtSdkIntegrationSmoke.moc"
