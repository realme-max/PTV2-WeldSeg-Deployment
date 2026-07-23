#include "AppConfig.h"
#include "MainWindow.h"
#include "PointCloudCamera.h"
#include "PointCloudView.h"
#include "QtWeldResultViewModel.h"

#include <QApplication>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QGroupBox>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QList>
#include <QPoint>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QSettings>
#include <QSignalSpy>
#include <QScreen>
#include <QSplitter>
#include <QSurfaceFormat>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <utility>

namespace
{

struct Options
{
    QString engine;
    QString plugin;
    QString engineSha256;
    QString pluginSha256;
    QString cloud;
    QString artifactDirectory;
};

QString required(
    QCommandLineParser const& parser,
    QCommandLineOption const& option)
{
    QString const value = parser.value(option);
    if (value.trimmed().isEmpty())
        qFatal("Required option is missing");
    return QFileInfo(value).absoluteFilePath();
}

bool writeJson(QString const& path, QJsonObject const& object)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
        return false;
    return file.write(QJsonDocument(object).toJson(QJsonDocument::Indented)) >= 0;
}

bool finiteCamera(ptv2::qtui::PointCloudCamera const& camera)
{
    QVector3D const center = camera.center();
    QVector3D const pan = camera.panOffset();
    return std::isfinite(camera.yaw())
        && std::isfinite(camera.pitch())
        && std::isfinite(camera.distance())
        && std::isfinite(center.x())
        && std::isfinite(center.y())
        && std::isfinite(center.z())
        && std::isfinite(pan.x())
        && std::isfinite(pan.y())
        && std::isfinite(pan.z());
}

bool sameCamera(
    ptv2::qtui::PointCloudCamera const& left,
    float yaw,
    float pitch,
    float distance,
    QVector3D const& center,
    QVector3D const& pan)
{
    constexpr float epsilon = 1.0e-6F;
    return std::abs(left.yaw() - yaw) <= epsilon
        && std::abs(left.pitch() - pitch) <= epsilon
        && std::abs(left.distance() - distance) <= epsilon
        && (left.center() - center).length() <= epsilon
        && (left.panOffset() - pan).length() <= epsilon;
}

class QtLayoutResizeStabilityTest final : public QObject
{
    Q_OBJECT

public:
    explicit QtLayoutResizeStabilityTest(Options options)
        : options_(std::move(options))
    {
    }

private slots:
    void initTestCase()
    {
        QVERIFY(QDir().mkpath(options_.artifactDirectory));
        ptv2::qtui::AppConfig config = ptv2::qtui::AppConfig::defaults();
        config.enginePath = options_.engine;
        config.pluginPath = options_.plugin;
        config.engineSha256 = options_.engineSha256;
        config.pluginSha256 = options_.pluginSha256;
        config.logDirectory = QDir(options_.artifactDirectory).filePath(QStringLiteral("logs"));
        config.defaultExportDirectory =
            QDir(options_.artifactDirectory).filePath(QStringLiteral("exports"));
        config.rememberWindowGeometry = true;
        settingsPath_ =
            QDir(options_.artifactDirectory).filePath(QStringLiteral("layout_test.ini"));
        QFile::remove(settingsPath_);
        {
            QSettings settings(settingsPath_, QSettings::IniFormat);
            settings.setValue(
                QStringLiteral("Window/geometry"), QByteArray("invalid-geometry"));
            settings.sync();
        }

        window_ = std::make_unique<ptv2::qtui::MainWindow>(
            config, options_.cloud, settingsPath_);
        window_->show();
        QVERIFY2(waitUntil([this] {
            auto* status =
                window_->findChild<QLabel*>(QStringLiteral("initializeStatus"));
            return status != nullptr && status->text() == QStringLiteral("SUCCESS");
        }, 120000), "SDK initialization timed out");
    }

    void scrollAreaArchitecture()
    {
        auto* scroll =
            window_->findChild<QScrollArea*>(QStringLiteral("rightScrollArea"));
        auto* splitter =
            window_->findChild<QSplitter*>(QStringLiteral("mainContentSplitter"));
        auto* content =
            window_->findChild<QWidget*>(QStringLiteral("rightScrollContent"));
        auto* bottom =
            window_->findChild<QGroupBox*>(QStringLiteral("runtimeLogGroup"));
        QVERIFY(scroll != nullptr);
        QVERIFY(splitter != nullptr);
        QVERIFY(content != nullptr);
        QVERIFY(bottom != nullptr);
        QVERIFY(scroll->widgetResizable());
        QCOMPARE(scroll->verticalScrollBarPolicy(), Qt::ScrollBarAsNeeded);
        QCOMPARE(scroll->horizontalScrollBarPolicy(), Qt::ScrollBarAlwaysOff);
        QCOMPARE(scroll->widget(), content);
        bool geometryVisible = false;
        for (QScreen* screen : QGuiApplication::screens())
        {
            QRect const intersection =
                window_->frameGeometry().intersected(screen->availableGeometry());
            if (intersection.width() >= 160 && intersection.height() >= 100)
            {
                geometryVisible = true;
                break;
            }
        }
        QVERIFY(geometryVisible);

        window_->resize(800, 500);
        QTest::qWait(250);
        QVERIFY(scroll->verticalScrollBar()->maximum() > 0);
        QCOMPARE(scroll->horizontalScrollBar()->maximum(), 0);
        scroll->verticalScrollBar()->setValue(
            scroll->verticalScrollBar()->maximum());
        scroll->ensureWidgetVisible(bottom, 0, 0);
        QCoreApplication::processEvents(QEventLoop::AllEvents, 100);
        QPoint const bottomInViewport =
            bottom->mapTo(scroll->viewport(), QPoint(0, bottom->height() - 1));
        QVERIFY(bottomInViewport.y() >= 0);
        QVERIFY(bottomInViewport.y() < scroll->viewport()->height());

        QJsonObject hierarchy;
        hierarchy.insert(QStringLiteral("status"), QStringLiteral("PASS"));
        hierarchy.insert(QStringLiteral("root"), QStringLiteral("QMainWindow/centralWidget/QVBoxLayout"));
        hierarchy.insert(QStringLiteral("main_splitter"), QStringLiteral("QSplitter(Horizontal)"));
        hierarchy.insert(QStringLiteral("left"), QStringLiteral("visualizationGroup/PointCloudView"));
        hierarchy.insert(QStringLiteral("right"), QStringLiteral("QScrollArea/rightScrollContent"));
        hierarchy.insert(QStringLiteral("right_children"),
            QStringLiteral("resultGroup,recentTasksGroup,runtimeLogGroup"));
        hierarchy.insert(QStringLiteral("widget_resizable"), scroll->widgetResizable());
        hierarchy.insert(QStringLiteral("vertical_scroll_maximum"),
            scroll->verticalScrollBar()->maximum());
        hierarchy.insert(QStringLiteral("horizontal_scroll_maximum"),
            scroll->horizontalScrollBar()->maximum());
        hierarchy.insert(QStringLiteral("bottom_content_reachable"), true);
        hierarchy.insert(QStringLiteral("invalid_saved_geometry_fell_back_visible"), true);
        QVERIFY(writeJson(QDir(options_.artifactDirectory)
            .filePath(QStringLiteral("layout_hierarchy_after.json")), hierarchy));
        QVERIFY(writeJson(QDir(options_.artifactDirectory)
            .filePath(QStringLiteral("scroll_area_test.json")), hierarchy));
    }

    void resizeDuringDetectionAndStress()
    {
        auto* view =
            window_->findChild<ptv2::qtui::PointCloudView*>(
                QStringLiteral("pointCloudView"));
        QVERIFY(view != nullptr);

        qint64 maxHeartbeatDelayMs = 0;
        qint64 lastHeartbeatMs = -1;
        int heartbeatCount = 0;
        QElapsedTimer heartbeatClock;
        heartbeatClock.start();
        QTimer heartbeat;
        heartbeat.setInterval(10);
        connect(&heartbeat, &QTimer::timeout, this, [&] {
            qint64 const now = heartbeatClock.elapsed();
            if (lastHeartbeatMs >= 0)
                maxHeartbeatDelayMs =
                    std::max(maxHeartbeatDelayMs, now - lastHeartbeatMs);
            lastHeartbeatMs = now;
            ++heartbeatCount;
        });
        heartbeat.start();

        QSignalSpy completed(
            window_.get(), &ptv2::qtui::MainWindow::productSmokeCompleted);
        QString const exportRoot =
            QDir(options_.artifactDirectory).filePath(QStringLiteral("resize_export"));
        window_->startProductSmoke(exportRoot);

        QList<QSize> const sequence{
            QSize(800, 600),
            QSize(1200, 700),
            QSize(900, 500),
            QSize(1600, 900),
            QSize(700, 500),
        };
        QElapsedTimer detectionTimer;
        detectionTimer.start();
        int resizeIndex = 0;
        for (; resizeIndex < 20 && completed.count() == 0; ++resizeIndex)
        {
            window_->resize(sequence.at(resizeIndex % sequence.size()));
            QTest::qWait(20);
        }
        QVERIFY2(waitUntil([&completed] { return completed.count() == 1; }, 90000),
            "Detection/export did not finish after the finite resize overlap");
        QVERIFY(completed.at(0).at(0).toBool());
        QString const exportDirectory = completed.at(0).at(1).toString();
        QVERIFY(QFileInfo(QDir(exportDirectory)
            .filePath(QStringLiteral("weld_result.json"))).isFile());
        QVERIFY(QFileInfo(QDir(exportDirectory)
            .filePath(QStringLiteral("weld_points.ply"))).isFile());
        QVERIFY(QFileInfo(QDir(exportDirectory)
            .filePath(QStringLiteral("prediction.txt"))).isFile());

        QVERIFY2(waitUntil([view] {
            return view->renderedPointCount() == 2048
                && view->bufferUploadCount() > 0;
        }, 30000), "Point-cloud render data did not settle");
        QCOMPARE(view->weldPointCount(), 209);
        QCOMPARE(view->backgroundPointCount(), 1839);
        QCOMPARE(view->lastGlError(), 0U);

        ptv2::qtui::PointCloudCamera const& camera = view->camera();
        float const yaw = camera.yaw();
        float const pitch = camera.pitch();
        float const distance = camera.distance();
        QVector3D const center = camera.center();
        QVector3D const pan = camera.panOffset();
        quint64 const uploadCountBefore = view->bufferUploadCount();
        quint64 const resizeCountBefore = view->resizeGlCount();

        QElapsedTimer stressTimer;
        stressTimer.start();
        for (int iteration = 0; iteration < 100; ++iteration)
        {
            window_->resize(sequence.at(iteration % sequence.size()));
            QTest::qWait(8);
        }
        window_->showMaximized();
        QTest::qWait(100);
        window_->showNormal();
        QTest::qWait(200);

        QVERIFY(heartbeatCount > 0);
        QVERIFY2(maxHeartbeatDelayMs < 1000,
            qPrintable(QStringLiteral("UI heartbeat delay was %1 ms")
                .arg(maxHeartbeatDelayMs)));
        QVERIFY(view->resizeGlCount() > resizeCountBefore);
        QCOMPARE(view->bufferUploadCount(), uploadCountBefore);
        quint64 const uploadCountAfterResize = view->bufferUploadCount();
        QCOMPARE(view->renderedPointCount(), 2048);
        QCOMPARE(view->weldPointCount() + view->backgroundPointCount(), 2048);
        QCOMPARE(view->lastGlError(), 0U);
        QVERIFY(std::isfinite(view->aspectRatio()));
        QVERIFY(view->aspectRatio() > 0.0F);
        QVERIFY(finiteCamera(view->camera()));
        QVERIFY(sameCamera(
            view->camera(), yaw, pitch, distance, center, pan));

        auto* detect =
            window_->findChild<QPushButton*>(QStringLiteral("detectButton"));
        QVERIFY(detect != nullptr);
        QVERIFY(detect->isEnabled());
        QTest::mouseClick(detect, Qt::LeftButton);
        QVERIFY(!detect->isEnabled());
        for (int iteration = 0; iteration < 20; ++iteration)
        {
            window_->resize(sequence.at(iteration % sequence.size()));
            QTest::qWait(15);
        }
        QVERIFY2(waitUntil([detect] { return detect->isEnabled(); }, 90000),
            "Second detection did not complete after resize overlap");
        QVERIFY2(waitUntil([view, uploadCountBefore] {
            return view->bufferUploadCount() > uploadCountBefore;
        }, 30000), "Second detection render data did not upload");
        heartbeat.stop();

        auto* sampled = window_->findChild<QLabel*>(QStringLiteral("sampled_points"));
        auto* weld = window_->findChild<QLabel*>(QStringLiteral("weld_points"));
        auto* ratio = window_->findChild<QLabel*>(QStringLiteral("weld_ratio"));
        auto* length = window_->findChild<QLabel*>(QStringLiteral("length_mm"));
        auto* errors =
            window_->findChild<QLabel*>(QStringLiteral("error_recorder_errors"));
        QVERIFY(sampled != nullptr);
        QVERIFY(weld != nullptr);
        QVERIFY(ratio != nullptr);
        QVERIFY(length != nullptr);
        QVERIFY(errors != nullptr);
        QCOMPARE(sampled->text().toInt(), 2048);
        QCOMPARE(weld->text().toInt(), 209);
        QVERIFY(std::abs(ratio->text().toDouble() - 0.10205078125) < 1.0e-12);
        QVERIFY(std::abs(length->text().toDouble() - 57.1960525513) < 1.0e-6);
        QCOMPARE(errors->text().toInt(), 0);

        QJsonObject resizeReport;
        resizeReport.insert(QStringLiteral("status"), QStringLiteral("PASS"));
        resizeReport.insert(QStringLiteral("resize_operations"), resizeIndex + 122);
        resizeReport.insert(QStringLiteral("stress_elapsed_ms"), stressTimer.elapsed());
        resizeReport.insert(QStringLiteral("rendered_points"), view->renderedPointCount());
        resizeReport.insert(QStringLiteral("weld_points"), view->weldPointCount());
        resizeReport.insert(QStringLiteral("background_points"), view->backgroundPointCount());
        resizeReport.insert(QStringLiteral("buffer_uploads_before"), static_cast<double>(uploadCountBefore));
        resizeReport.insert(QStringLiteral("buffer_uploads_after_resize_only"),
            static_cast<double>(uploadCountAfterResize));
        resizeReport.insert(QStringLiteral("buffer_uploads_after_second_detection"),
            static_cast<double>(view->bufferUploadCount()));
        resizeReport.insert(QStringLiteral("resize_gl_calls"),
            static_cast<double>(view->resizeGlCount() - resizeCountBefore));
        resizeReport.insert(QStringLiteral("camera_finite"), finiteCamera(view->camera()));
        resizeReport.insert(QStringLiteral("camera_preserved"), true);
        resizeReport.insert(QStringLiteral("opengl_error"), static_cast<int>(view->lastGlError()));
        resizeReport.insert(QStringLiteral("detection_export_pass"), true);
        resizeReport.insert(QStringLiteral("second_detection_during_resize_pass"), true);
        QVERIFY(writeJson(QDir(options_.artifactDirectory)
            .filePath(QStringLiteral("resize_stress_test.json")), resizeReport));

        QJsonObject heartbeatReport;
        heartbeatReport.insert(QStringLiteral("status"), QStringLiteral("PASS"));
        heartbeatReport.insert(QStringLiteral("timer_interval_ms"), 10);
        heartbeatReport.insert(QStringLiteral("heartbeat_count"), heartbeatCount);
        heartbeatReport.insert(QStringLiteral("maximum_delay_ms"),
            static_cast<double>(maxHeartbeatDelayMs));
        heartbeatReport.insert(QStringLiteral("threshold_ms"), 1000);
        QVERIFY(writeJson(QDir(options_.artifactDirectory)
            .filePath(QStringLiteral("ui_heartbeat_test.json")), heartbeatReport));

        QJsonObject glReport;
        glReport.insert(QStringLiteral("status"), QStringLiteral("PASS"));
        glReport.insert(QStringLiteral("rendered_points"), view->renderedPointCount());
        glReport.insert(QStringLiteral("buffer_reuploaded_during_resize"), false);
        glReport.insert(QStringLiteral("camera_state_finite"), true);
        glReport.insert(QStringLiteral("camera_state_preserved"), true);
        glReport.insert(QStringLiteral("last_gl_error"), static_cast<int>(view->lastGlError()));
        glReport.insert(QStringLiteral("aspect_ratio"), view->aspectRatio());
        QVERIFY(writeJson(QDir(options_.artifactDirectory)
            .filePath(QStringLiteral("opengl_resize_test.json")), glReport));
    }

    void cleanupTestCase()
    {
        QVERIFY(window_ != nullptr);
        window_->close();
        QCoreApplication::processEvents(QEventLoop::AllEvents, 200);
        window_.reset();
    }

private:
    static bool waitUntil(
        std::function<bool()> const& predicate,
        int timeoutMs)
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

    Options options_;
    QString settingsPath_;
    std::unique_ptr<ptv2::qtui::MainWindow> window_;
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
    application.setApplicationName(QStringLiteral("QtLayoutResizeStabilityTest"));
    qRegisterMetaType<ptv2::qtui::QtWeldResultViewModel>(
        "ptv2::qtui::QtWeldResultViewModel");

    QCommandLineParser parser;
    parser.addHelpOption();
    QCommandLineOption engine(QStringList() << QStringLiteral("engine"), {}, QStringLiteral("path"));
    QCommandLineOption plugin(
        QStringList() << QStringLiteral("trt-plugin"), {}, QStringLiteral("path"));
    QCommandLineOption engineSha(
        QStringList() << QStringLiteral("engine-sha256"), {}, QStringLiteral("sha256"));
    QCommandLineOption pluginSha(
        QStringList() << QStringLiteral("plugin-sha256"), {}, QStringLiteral("sha256"));
    QCommandLineOption cloud(QStringList() << QStringLiteral("cloud"), {}, QStringLiteral("path"));
    QCommandLineOption artifactDir(
        QStringList() << QStringLiteral("artifact-dir"), {}, QStringLiteral("path"));
    parser.addOption(engine);
    parser.addOption(plugin);
    parser.addOption(engineSha);
    parser.addOption(pluginSha);
    parser.addOption(cloud);
    parser.addOption(artifactDir);
    parser.process(rawArguments);

    Options options{
        required(parser, engine),
        required(parser, plugin),
        parser.value(engineSha).trimmed().toLower(),
        parser.value(pluginSha).trimmed().toLower(),
        required(parser, cloud),
        required(parser, artifactDir),
    };
    if (options.engineSha256.size() != 64 || options.pluginSha256.size() != 64)
        qFatal("Engine and Plugin SHA-256 values must contain 64 hexadecimal characters");

    QtLayoutResizeStabilityTest test(std::move(options));
    int const testArgc = 1;
    char* testArgv[] = {argv[0], nullptr};
    return QTest::qExec(&test, testArgc, testArgv);
}

#include "QtLayoutResizeStabilityTest.moc"
