#include "AppConfig.h"
#include "MainWindow.h"
#include "PointCloudView.h"
#include "QtWeldResultViewModel.h"
#include "WeldDetectionWorker.h"

#include <QApplication>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QLineEdit>
#include <QPushButton>
#include <QSettings>
#include <QSignalSpy>
#include <QSurfaceFormat>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>

#include <memory>
#include <utility>

namespace ptv2::qtui
{
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

QString normalizedDirectory(QString const& path)
{
    return QDir::toNativeSeparators(QFileInfo(path).absoluteFilePath());
}

QFileDialog* activeFileDialog()
{
    for (QWidget* widget : QApplication::topLevelWidgets())
    {
        if (auto* dialog = qobject_cast<QFileDialog*>(widget))
            return dialog;
    }
    return nullptr;
}

bool createTextFile(QString const& path)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
        return false;
    return file.write("0 0 0 0\n") > 0;
}

} // namespace

class QtCloudBrowseDirectoryTest final : public QObject
{
    Q_OBJECT

public:
    explicit QtCloudBrowseDirectoryTest(Options options)
        : options_(std::move(options))
    {
    }

private slots:
    void initTestCase()
    {
        QVERIFY(temporary_.isValid());
        QVERIFY(QDir().mkpath(options_.artifactDirectory));
        settingsPath_ = temporary_.filePath(QStringLiteral("qt_weld_app.ini"));

        AppConfig config = AppConfig::defaults();
        config.enginePath = options_.engine;
        config.pluginPath = options_.plugin;
        config.engineSha256 = options_.engineSha256;
        config.pluginSha256 = options_.pluginSha256;
        config.defaultCloudDirectory.clear();
        config.defaultExportDirectory =
            temporary_.filePath(QStringLiteral("exports"));
        config.logDirectory = temporary_.filePath(QStringLiteral("logs"));
        config.rememberLastCloud = true;
        config.rememberWindowGeometry = false;

        window_ = std::make_unique<MainWindow>(
            config, options_.cloud, settingsPath_);
        QVERIFY(window_ != nullptr);

        QElapsedTimer timer;
        timer.start();
        while (!window_->initialized_ && timer.elapsed() < 10000)
        {
            QCoreApplication::processEvents(QEventLoop::AllEvents, 25);
            QTest::qWait(10);
        }
        QVERIFY2(window_->initialized_, "SDK initialization timed out");
        QVERIFY(window_->browseButton_->isEnabled());
    }

    void noHistoryUsesDevelopmentDefault()
    {
        QSettings settings(settingsPath_, QSettings::IniFormat);
        settings.remove(QStringLiteral("Application/last_cloud_directory"));
        settings.sync();
        window_->appConfig_.rememberLastCloud = true;
        window_->appConfig_.defaultCloudDirectory.clear();

        QString const expected =
            normalizedDirectory(QStringLiteral("E:/GRP-PTv2/data/weld/000001"));
        QVERIFY2(QDir(expected).exists(), qPrintable(expected));
        QCOMPARE(window_->resolveInitialCloudDirectory(), expected);
    }

    void lastDirectoryHasHighestPriority()
    {
        QString const lastDirectory =
            temporary_.filePath(QStringLiteral("last_cloud"));
        QString const configuredDirectory =
            temporary_.filePath(QStringLiteral("configured_cloud"));
        QVERIFY(QDir().mkpath(lastDirectory));
        QVERIFY(QDir().mkpath(configuredDirectory));

        QSettings settings(settingsPath_, QSettings::IniFormat);
        settings.setValue(
            QStringLiteral("Application/last_cloud_directory"), lastDirectory);
        settings.sync();
        window_->appConfig_.rememberLastCloud = true;
        window_->appConfig_.defaultCloudDirectory = configuredDirectory;

        QCOMPARE(
            window_->resolveInitialCloudDirectory(),
            normalizedDirectory(lastDirectory));
    }

    void missingLastDirectoryFallsBack()
    {
        QString const configuredDirectory =
            temporary_.filePath(QStringLiteral("configured_fallback"));
        QVERIFY(QDir().mkpath(configuredDirectory));

        QSettings settings(settingsPath_, QSettings::IniFormat);
        settings.setValue(
            QStringLiteral("Application/last_cloud_directory"),
            temporary_.filePath(QStringLiteral("missing_last")));
        settings.sync();
        window_->appConfig_.rememberLastCloud = true;
        window_->appConfig_.defaultCloudDirectory = configuredDirectory;

        QCOMPARE(
            window_->resolveInitialCloudDirectory(),
            normalizedDirectory(configuredDirectory));
    }

    void configuredDirectoryPrecedesDevelopmentDefault()
    {
        QString const configuredDirectory =
            temporary_.filePath(QStringLiteral("configured_priority"));
        QVERIFY(QDir().mkpath(configuredDirectory));

        QSettings settings(settingsPath_, QSettings::IniFormat);
        settings.remove(QStringLiteral("Application/last_cloud_directory"));
        settings.sync();
        window_->appConfig_.rememberLastCloud = true;
        window_->appConfig_.defaultCloudDirectory = configuredDirectory;

        QCOMPARE(
            window_->resolveInitialCloudDirectory(),
            normalizedDirectory(configuredDirectory));
    }

    void browseDialogUsesDevelopmentDefaultAndCancelPreservesPath()
    {
        QSettings settings(settingsPath_, QSettings::IniFormat);
        settings.remove(QStringLiteral("Application/last_cloud_directory"));
        settings.sync();
        window_->appConfig_.rememberLastCloud = true;
        window_->appConfig_.defaultCloudDirectory.clear();
        QString const before = window_->cloudPathEdit_->text();
        AppState const stateBefore = window_->stateMachine_.state();
        bool const detectingBefore = window_->detectionActive_;
        bool const hasResultBefore = window_->hasResult_;
        QString observedDirectory;
        QString observedFilter;

        QTimer::singleShot(0, this, [&]() {
            QFileDialog* dialog = activeFileDialog();
            QVERIFY(dialog != nullptr);
            observedDirectory = normalizedDirectory(dialog->directory().absolutePath());
            observedFilter = dialog->nameFilters().join(QStringLiteral(";"));
            dialog->reject();
        });
        QTest::mouseClick(window_->browseButton_, Qt::LeftButton);

        QCOMPARE(
            observedDirectory,
            normalizedDirectory(QStringLiteral("E:/GRP-PTv2/data/weld/000001")));
        QVERIFY(observedFilter.contains(QStringLiteral("*.txt")));
        QCOMPARE(window_->cloudPathEdit_->text(), before);
        QCOMPARE(window_->stateMachine_.state(), stateBefore);
        QCOMPARE(window_->detectionActive_, detectingBefore);
        QCOMPARE(window_->hasResult_, hasResultBefore);
    }

    void acceptedSelectionPersistsParentWithoutDetectAndReopensThere()
    {
        QString const selectedDirectory =
            temporary_.filePath(QStringLiteral("accepted"));
        QString const selectedFile =
            QDir(selectedDirectory).filePath(QStringLiteral("accepted.txt"));
        QVERIFY(createTextFile(selectedFile));
        window_->appConfig_.rememberLastCloud = true;
        QSignalSpy detectionStarted(
            window_->worker_, SIGNAL(detectionStarted(QString)));

        window_->applyCloudSelection(selectedFile);
        QCoreApplication::processEvents();

        QCOMPARE(
            window_->cloudPathEdit_->text(),
            QDir::toNativeSeparators(QFileInfo(selectedFile).absoluteFilePath()));
        QSettings settings(settingsPath_, QSettings::IniFormat);
        QCOMPARE(
            normalizedDirectory(settings.value(
                QStringLiteral("Application/last_cloud_directory")).toString()),
            normalizedDirectory(selectedDirectory));
        QCOMPARE(
            window_->resolveInitialCloudDirectory(),
            normalizedDirectory(selectedDirectory));
        QCOMPARE(detectionStarted.count(), 0);
        QVERIFY(!window_->detectionActive_);
        QVERIFY(!window_->hasResult_);
        QCOMPARE(window_->pointCloudView_->renderedPointCount(), 0);
        QVERIFY(!window_->exportResultButton_->isEnabled());
        QVERIFY(!window_->exportScreenshotButton_->isEnabled());

        QString reopenedDirectory;
        QString const selectedPathBeforeCancel = window_->cloudPathEdit_->text();
        QTimer::singleShot(0, this, [&]() {
            QFileDialog* dialog = activeFileDialog();
            QVERIFY(dialog != nullptr);
            reopenedDirectory =
                normalizedDirectory(dialog->directory().absolutePath());
            dialog->reject();
        });
        QTest::mouseClick(window_->browseButton_, Qt::LeftButton);
        QCOMPARE(reopenedDirectory, normalizedDirectory(selectedDirectory));
        QCOMPARE(window_->cloudPathEdit_->text(), selectedPathBeforeCancel);
    }

    void rememberDisabledDoesNotUpdateLastDirectory()
    {
        QString const retainedDirectory =
            temporary_.filePath(QStringLiteral("retained"));
        QString const selectedDirectory =
            temporary_.filePath(QStringLiteral("not_remembered"));
        QString const selectedFile =
            QDir(selectedDirectory).filePath(QStringLiteral("selected.txt"));
        QString const configuredDirectory =
            temporary_.filePath(QStringLiteral("configured_when_disabled"));
        QVERIFY(QDir().mkpath(retainedDirectory));
        QVERIFY(QDir().mkpath(configuredDirectory));
        QVERIFY(createTextFile(selectedFile));

        QSettings settings(settingsPath_, QSettings::IniFormat);
        settings.setValue(
            QStringLiteral("Application/last_cloud_directory"), retainedDirectory);
        settings.sync();
        window_->appConfig_.rememberLastCloud = false;
        window_->appConfig_.defaultCloudDirectory = configuredDirectory;

        window_->applyCloudSelection(selectedFile);
        settings.sync();

        QCOMPARE(
            normalizedDirectory(settings.value(
                QStringLiteral("Application/last_cloud_directory")).toString()),
            normalizedDirectory(retainedDirectory));
        QCOMPARE(
            window_->resolveInitialCloudDirectory(),
            normalizedDirectory(configuredDirectory));
    }

    void cleanupTestCase()
    {
        window_.reset();
    }

private:
    Options options_;
    QTemporaryDir temporary_;
    QString settingsPath_;
    std::unique_ptr<MainWindow> window_;
};

} // namespace ptv2::qtui

int main(int argc, char** argv)
{
    QStringList rawArguments;
    rawArguments.reserve(argc);
    for (int index = 0; index < argc; ++index)
        rawArguments.append(QString::fromLocal8Bit(argv[index]));

    QCoreApplication::setAttribute(Qt::AA_UseDesktopOpenGL, true);
    QCoreApplication::setAttribute(Qt::AA_DontUseNativeDialogs, true);
    QSurfaceFormat format;
    format.setRenderableType(QSurfaceFormat::OpenGL);
    format.setVersion(3, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    QSurfaceFormat::setDefaultFormat(format);
    QApplication application(argc, argv);
    application.setApplicationName(QStringLiteral("QtCloudBrowseDirectoryTest"));
    qRegisterMetaType<ptv2::qtui::QtWeldResultViewModel>(
        "ptv2::qtui::QtWeldResultViewModel");

    QCommandLineParser parser;
    parser.addHelpOption();
    QCommandLineOption engine(
        QStringList() << QStringLiteral("engine"), {}, QStringLiteral("path"));
    QCommandLineOption plugin(
        QStringList() << QStringLiteral("trt-plugin"), {}, QStringLiteral("path"));
    QCommandLineOption engineSha(
        QStringList() << QStringLiteral("engine-sha256"), {}, QStringLiteral("sha256"));
    QCommandLineOption pluginSha(
        QStringList() << QStringLiteral("plugin-sha256"), {}, QStringLiteral("sha256"));
    QCommandLineOption cloud(
        QStringList() << QStringLiteral("cloud"), {}, QStringLiteral("path"));
    QCommandLineOption artifactDirectory(
        QStringList() << QStringLiteral("artifact-dir"), {}, QStringLiteral("path"));
    parser.addOption(engine);
    parser.addOption(plugin);
    parser.addOption(engineSha);
    parser.addOption(pluginSha);
    parser.addOption(cloud);
    parser.addOption(artifactDirectory);
    parser.process(rawArguments);

    ptv2::qtui::Options options{
        ptv2::qtui::required(parser, engine),
        ptv2::qtui::required(parser, plugin),
        parser.value(engineSha).trimmed().toLower(),
        parser.value(pluginSha).trimmed().toLower(),
        ptv2::qtui::required(parser, cloud),
        ptv2::qtui::required(parser, artifactDirectory),
    };
    if (options.engineSha256.size() != 64 || options.pluginSha256.size() != 64)
        qFatal("Engine and Plugin SHA-256 values must contain 64 hexadecimal characters");

    ptv2::qtui::QtCloudBrowseDirectoryTest test(std::move(options));
    int const testArgc = 1;
    char* testArgv[] = {argv[0], nullptr};
    return QTest::qExec(&test, testArgc, testArgv);
}

#include "QtCloudBrowseDirectoryTest.moc"
