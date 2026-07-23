#include "MainWindow.h"
#include "AppConfig.h"
#include "ProductInfo.h"
#include "QtWeldResultViewModel.h"

#include <QApplication>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QFileInfo>
#include <QDir>
#include <QMap>
#include <QMessageBox>
#include <QStandardPaths>
#include <QSurfaceFormat>
#include <QStringList>

int main(int argc, char** argv)
{
    QStringList rawArguments;
    rawArguments.reserve(argc);
    for (int index = 0; index < argc; ++index)
        rawArguments.append(QString::fromLocal8Bit(argv[index]));

    QCoreApplication::setAttribute(Qt::AA_UseDesktopOpenGL, true);
    QSurfaceFormat format;
    format.setRenderableType(QSurfaceFormat::OpenGL);
    format.setVersion(3, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    format.setSamples(4);
    QSurfaceFormat::setDefaultFormat(format);
    QApplication application(argc, argv);
    application.setOrganizationName(QStringLiteral("PTV2"));
    application.setApplicationName(ptv2::qtui::ProductInfo::applicationName());
    application.setApplicationVersion(ptv2::qtui::ProductInfo::applicationVersion());
    qRegisterMetaType<ptv2::qtui::QtWeldResultViewModel>(
        "ptv2::qtui::QtWeldResultViewModel");

    QCommandLineParser parser;
    parser.setApplicationDescription(QStringLiteral("PTV2 Weld Segmentation product application"));
    parser.addHelpOption();
    parser.addVersionOption();
    QCommandLineOption engineOption(QStringList() << QStringLiteral("engine"),
        QStringLiteral("Production TensorRT Engine path"), QStringLiteral("path"));
    QCommandLineOption pluginOption(QStringList() << QStringLiteral("plugin"),
        QStringLiteral("VoxelUniqueCub Plugin path"), QStringLiteral("path"));
    QCommandLineOption shaOption(QStringList() << QStringLiteral("engine-sha256"),
        QStringLiteral("Expected production Engine SHA-256"), QStringLiteral("sha256"));
    QCommandLineOption pluginShaOption(QStringList() << QStringLiteral("plugin-sha256"),
        QStringLiteral("Expected VoxelUniqueCub Plugin SHA-256"), QStringLiteral("sha256"));
    QCommandLineOption cloudOption(QStringList() << QStringLiteral("cloud"),
        QStringLiteral("Optional initial weld TXT path"), QStringLiteral("path"));
    QCommandLineOption configOption(QStringList() << QStringLiteral("config"),
        QStringLiteral("Default/package INI path"), QStringLiteral("path"));
    QCommandLineOption userConfigOption(QStringList() << QStringLiteral("user-config"),
        QStringLiteral("Persistent user INI path"), QStringLiteral("path"));
    QCommandLineOption smokeExportOption(QStringList() << QStringLiteral("product-smoke-export"),
        QStringLiteral("Run one detection/export smoke then exit"), QStringLiteral("directory"));
    parser.addOption(engineOption);
    parser.addOption(pluginOption);
    parser.addOption(shaOption);
    parser.addOption(pluginShaOption);
    parser.addOption(cloudOption);
    parser.addOption(configOption);
    parser.addOption(userConfigOption);
    parser.addOption(smokeExportOption);
    // QApplication consumes its own -plugin option. Parse the preserved raw
    // arguments so the WeldDetector SDK --plugin contract remains available.
    parser.process(rawArguments);

    QString const defaultConfig = parser.isSet(configOption)
        ? QFileInfo(parser.value(configOption)).absoluteFilePath()
        : QDir(QCoreApplication::applicationDirPath())
            .filePath(QStringLiteral("config/qt_weld_app.ini"));
    QString const userConfig = parser.isSet(userConfigOption)
        ? QFileInfo(parser.value(userConfigOption)).absoluteFilePath()
        : QDir(QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation))
            .filePath(QStringLiteral("qt_weld_app.ini"));
    QMap<QString, QString> overrides;
    if (parser.isSet(engineOption))
        overrides.insert(QStringLiteral("engine_path"), parser.value(engineOption));
    if (parser.isSet(pluginOption))
        overrides.insert(QStringLiteral("plugin_path"), parser.value(pluginOption));
    if (parser.isSet(shaOption))
        overrides.insert(QStringLiteral("engine_sha256"), parser.value(shaOption));
    if (parser.isSet(pluginShaOption))
        overrides.insert(QStringLiteral("plugin_sha256"), parser.value(pluginShaOption));
    QString error;
    ptv2::qtui::AppConfig config =
        ptv2::qtui::AppConfig::loadLayered(defaultConfig, userConfig, overrides, error);
    QString const configBase = QFileInfo(defaultConfig).absolutePath();
    config.enginePath = config.resolvedPath(config.enginePath, configBase);
    config.pluginPath = config.resolvedPath(config.pluginPath, configBase);
    config.defaultCloudDirectory =
        config.resolvedPath(config.defaultCloudDirectory, configBase);
    config.defaultExportDirectory =
        config.resolvedPath(config.defaultExportDirectory, configBase);
    config.logDirectory = config.resolvedPath(config.logDirectory, configBase);
    if (!error.isEmpty())
    {
        QMessageBox::critical(nullptr, QStringLiteral("Startup configuration error"), error);
        return 2;
    }
    ptv2::qtui::AppConfigValidation const validation = config.validateRuntime();
    if (!validation.valid)
    {
        QMessageBox::critical(nullptr, QStringLiteral("Runtime integrity failed"),
            QStringLiteral("RUNTIME_INTEGRITY_FAILED\n%1\n"
                           "Engine: %2\nPlugin: %3\n"
                           "Check package paths and SHA-256 identities.")
                .arg(validation.error, config.enginePath, config.pluginPath));
        return 2;
    }

    ptv2::qtui::MainWindow window(config,
        parser.isSet(cloudOption) ? parser.value(cloudOption) : QString(), userConfig);
    int smokeExitCode = 0;
    if (parser.isSet(smokeExportOption))
    {
        if (!parser.isSet(cloudOption))
        {
            QMessageBox::critical(nullptr, QStringLiteral("Product smoke configuration error"),
                QStringLiteral("--product-smoke-export requires --cloud"));
            return 2;
        }
        QObject::connect(&window, &ptv2::qtui::MainWindow::productSmokeCompleted,
            &application, [&](bool success, QString const&, QString const&) {
                smokeExitCode = success ? 0 : 3;
                application.quit();
            });
    }
    window.show();
    if (parser.isSet(smokeExportOption))
        window.startProductSmoke(parser.value(smokeExportOption));
    application.exec();
    return smokeExitCode;
}
