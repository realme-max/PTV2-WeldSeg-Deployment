#include "PointCloudRenderData.h"
#include "PointCloudView.h"
#include "QtWeldResultViewModel.h"
#include "WeldConfig.h"
#include "WeldDetector.h"
#include "WeldResult.h"
#include "WeldStatus.h"

#include <QApplication>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMouseEvent>
#include <QSurfaceFormat>
#include <QWheelEvent>
#include <QtTest>

#include <functional>
#include <limits>
#include <memory>

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

bool waitUntil(std::function<bool()> const& predicate, int timeoutMs)
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

class PointCloudViewSmokeTest final : public QObject
{
    Q_OBJECT

public:
    explicit PointCloudViewSmokeTest(Options options) : options_(std::move(options)) {}

private slots:
    void initTestCase()
    {
        ptv2::weld::WeldConfig config;
        config.engine_path = options_.engine.toStdString();
        config.plugin_path = options_.plugin.toStdString();
        detector_ = std::make_unique<ptv2::weld::WeldDetector>();
        QVERIFY(detector_->initialize(config) == ptv2::weld::WeldStatus::SUCCESS);
        ptv2::weld::WeldResult sdkResult;
        QVERIFY(detector_->detect(options_.cloud.toStdString(), sdkResult)
            == ptv2::weld::WeldStatus::SUCCESS);
        viewModel_ = ptv2::qtui::QtWeldResultViewModel::fromSdk(sdkResult, options_.cloud);
        renderData_ = ptv2::qtui::PointCloudRenderData::fromResult(viewModel_);
        QVERIFY(renderData_.valid);
    }

    void realOpenGlWidgetLifecycle()
    {
        auto view = std::make_unique<ptv2::qtui::PointCloudView>();
        view->resize(900, 650);
        view->show();
        QVERIFY2(waitUntil([&] {
            return view->openGLInitialized() || !view->visualizationError().isEmpty();
        }, 30000), "OpenGL initialization timed out");
        QVERIFY2(view->openGLInitialized(), qPrintable(view->visualizationError()));
        QVERIFY(view->shaderLinked());
        glVersion_ = view->openGLVersion();
        glRenderer_ = view->openGLRenderer();
        glVendor_ = view->openGLVendor();
        QString error;
        QVERIFY2(view->setPointCloud(renderData_, error), qPrintable(error));
        QVERIFY(waitUntil([&] {
            return view->renderedPointCount() == 2048
                && view->lastUploadMs() > 0.0 && view->lastPaintMs() > 0.0;
        }, 10000));
        QCOMPARE(view->renderedPointCount(), 2048);
        QCOMPARE(view->weldPointCount(), 209);
        QCOMPARE(view->backgroundPointCount(), 1839);
        QCOMPARE(view->lastGlError(), 0U);
        double const uploadMs = view->lastUploadMs();
        double const firstPaintMs = view->lastPaintMs();
        view->update();
        QTest::qWait(100);
        double const repaintMs = view->lastPaintMs();

        float const yawBefore = view->camera().yaw();
        QMouseEvent press(QEvent::MouseButtonPress, QPointF(200, 200),
            Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(view.get(), &press);
        QMouseEvent move(QEvent::MouseMove, QPointF(250, 225),
            Qt::NoButton, Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(view.get(), &move);
        QVERIFY(view->camera().yaw() != yawBefore);

        float const distanceBefore = view->camera().distance();
        QWheelEvent wheel(
            QPointF(250, 225), QPointF(250, 225), QPoint(), QPoint(0, 120),
            120, Qt::Vertical, Qt::NoButton, Qt::NoModifier);
        QApplication::sendEvent(view.get(), &wheel);
        QVERIFY(view->camera().distance() < distanceBefore);

        QMouseEvent panPress(QEvent::MouseButtonPress, QPointF(300, 300),
            Qt::RightButton, Qt::RightButton, Qt::NoModifier);
        QApplication::sendEvent(view.get(), &panPress);
        QMouseEvent panMove(QEvent::MouseMove, QPointF(330, 320),
            Qt::NoButton, Qt::RightButton, Qt::NoModifier);
        QApplication::sendEvent(view.get(), &panMove);
        QVERIFY(view->camera().panOffset().length() > 0.0F);

        view->resetView();
        view->setShowBoundingBox(false);
        view->setShowPcaDirection(false);
        view->setShowBoundingBox(true);
        view->setShowPcaDirection(true);
        QVERIFY(view->setPointCloud(renderData_, error));
        QCOMPARE(view->renderedPointCount(), 2048);

        ptv2::qtui::PointCloudRenderData invalid = renderData_;
        invalid.points[0].position.setX(std::numeric_limits<float>::quiet_NaN());
        invalid.valid = true;
        QVERIFY(!view->setPointCloud(invalid, error));
        QCOMPARE(view->renderedPointCount(), 2048);
        auto mismatchedView = viewModel_;
        mismatchedView.points.removeLast();
        auto mismatched = ptv2::qtui::PointCloudRenderData::fromResult(mismatchedView);
        QVERIFY(!mismatched.valid);
        QVERIFY(!view->setPointCloud(mismatched, error));
        QCOMPARE(view->renderedPointCount(), 2048);
        ptv2::qtui::PointCloudRenderData invalidPca = renderData_;
        invalidPca.pcaEnd = invalidPca.pcaStart;
        invalidPca.valid = true;
        QVERIFY(!view->setPointCloud(invalidPca, error));
        QCOMPARE(view->renderedPointCount(), 2048);
        view->clearPointCloud();
        QCOMPARE(view->renderedPointCount(), 0);
        view.reset();

        report_.insert(QStringLiteral("status"), QStringLiteral("PASS"));
        report_.insert(QStringLiteral("opengl_initialized"), true);
        report_.insert(QStringLiteral("shader_linked"), true);
        report_.insert(QStringLiteral("opengl_version"), glVersion_);
        report_.insert(QStringLiteral("opengl_renderer"), glRenderer_);
        report_.insert(QStringLiteral("opengl_vendor"), glVendor_);
        report_.insert(QStringLiteral("rendered_points"), 2048);
        report_.insert(QStringLiteral("weld_points"), 209);
        report_.insert(QStringLiteral("background_points"), 1839);
        report_.insert(QStringLiteral("gl_error"), 0);
        report_.insert(QStringLiteral("gpu_buffer_upload_ms"), uploadMs);
        report_.insert(QStringLiteral("first_paint_ms"), firstPaintMs);
        report_.insert(QStringLiteral("subsequent_repaint_ms"), repaintMs);
        report_.insert(QStringLiteral("reset_view"), true);
        report_.insert(QStringLiteral("rotation"), true);
        report_.insert(QStringLiteral("zoom"), true);
        report_.insert(QStringLiteral("pan_event"), true);
        report_.insert(QStringLiteral("second_upload"), true);
        report_.insert(QStringLiteral("clear"), true);
        report_.insert(QStringLiteral("invalid_data_preserved_previous_view"), true);
        report_.insert(QStringLiteral("mismatched_count_rejected"), true);
        report_.insert(QStringLiteral("invalid_pca_rejected"), true);
    }

    void shaderFailureFailsClosed()
    {
        auto view = std::make_unique<ptv2::qtui::PointCloudView>();
        view->setForceShaderFailureForTest(true);
        view->resize(320, 240);
        view->show();
        QVERIFY(waitUntil([&] { return !view->visualizationError().isEmpty(); }, 10000));
        QVERIFY(!view->openGLInitialized());
        QVERIFY(!view->shaderLinked());
        report_.insert(QStringLiteral("shader_failure_fail_closed"), true);
        view.reset();
    }

    void cleanupTestCase()
    {
        QFile output(options_.output);
        QVERIFY(output.open(QIODevice::WriteOnly | QIODevice::Truncate));
        QVERIFY(output.write(QJsonDocument(report_).toJson(QJsonDocument::Indented)) > 0);
    }

private:
    Options options_;
    std::unique_ptr<ptv2::weld::WeldDetector> detector_;
    ptv2::qtui::PointCloudRenderData renderData_;
    ptv2::qtui::QtWeldResultViewModel viewModel_;
    QJsonObject report_;
    QString glVersion_;
    QString glRenderer_;
    QString glVendor_;
};

} // namespace

int main(int argc, char** argv)
{
    QStringList raw;
    for (int index = 0; index < argc; ++index) raw.append(QString::fromLocal8Bit(argv[index]));
    QCoreApplication::setAttribute(Qt::AA_UseDesktopOpenGL, true);
    QSurfaceFormat format;
    format.setRenderableType(QSurfaceFormat::OpenGL);
    format.setVersion(3, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    QSurfaceFormat::setDefaultFormat(format);
    QApplication application(argc, argv);
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
    PointCloudViewSmokeTest test(std::move(options));
    int testArgc = 1;
    char* testArgv[] = {argv[0], nullptr};
    return QTest::qExec(&test, testArgc, testArgv);
}

#include "PointCloudViewSmokeTest.moc"
