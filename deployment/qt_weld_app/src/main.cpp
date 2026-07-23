#include "MainWindow.h"
#include "QtWeldResultViewModel.h"
#include "WeldConfig.h"

#include <QApplication>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QFileInfo>
#include <QMessageBox>
#include <QRegularExpression>
#include <QSurfaceFormat>
#include <QStringList>

namespace
{

bool requireOption(
    QCommandLineParser const& parser,
    QCommandLineOption const& option,
    QString& error)
{
    if (!parser.isSet(option) || parser.value(option).trimmed().isEmpty())
    {
        error = QStringLiteral("Missing required option --%1").arg(option.names().first());
        return false;
    }
    return true;
}

} // namespace

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
    application.setApplicationName(QStringLiteral("ptv2_weld_qt_smoke"));
    application.setApplicationVersion(QStringLiteral("Phase 10A"));
    qRegisterMetaType<ptv2::qtui::QtWeldResultViewModel>(
        "ptv2::qtui::QtWeldResultViewModel");

    QCommandLineParser parser;
    parser.setApplicationDescription(QStringLiteral("PTV2 WeldDetector SDK Qt integration smoke"));
    parser.addHelpOption();
    parser.addVersionOption();
    QCommandLineOption engineOption(QStringList() << QStringLiteral("engine"),
        QStringLiteral("Production TensorRT Engine path"), QStringLiteral("path"));
    QCommandLineOption pluginOption(QStringList() << QStringLiteral("plugin"),
        QStringLiteral("VoxelUniqueCub Plugin path"), QStringLiteral("path"));
    QCommandLineOption shaOption(QStringList() << QStringLiteral("engine-sha256"),
        QStringLiteral("Expected production Engine SHA-256"), QStringLiteral("sha256"));
    QCommandLineOption cloudOption(QStringList() << QStringLiteral("cloud"),
        QStringLiteral("Optional initial weld TXT path"), QStringLiteral("path"));
    parser.addOption(engineOption);
    parser.addOption(pluginOption);
    parser.addOption(shaOption);
    parser.addOption(cloudOption);
    // QApplication consumes its own -plugin option. Parse the preserved raw
    // arguments so the WeldDetector SDK --plugin contract remains available.
    parser.process(rawArguments);

    QString error;
    if (!requireOption(parser, engineOption, error)
        || !requireOption(parser, pluginOption, error)
        || !requireOption(parser, shaOption, error))
    {
        QMessageBox::critical(nullptr, QStringLiteral("Startup configuration error"), error);
        return 2;
    }
    QString const engine = QFileInfo(parser.value(engineOption)).absoluteFilePath();
    QString const plugin = QFileInfo(parser.value(pluginOption)).absoluteFilePath();
    QString const sha = parser.value(shaOption).trimmed().toLower();
    QRegularExpression const shaPattern(QStringLiteral("^[0-9a-f]{64}$"));
    if (!QFileInfo(engine).isFile() || !QFileInfo(plugin).isFile()
        || !shaPattern.match(sha).hasMatch())
    {
        QMessageBox::critical(nullptr, QStringLiteral("Startup configuration error"),
            QStringLiteral("Engine and Plugin must be existing files and engine-sha256 must be 64 hex characters."));
        return 2;
    }

    ptv2::weld::WeldConfig config;
    config.engine_path = engine.toStdString();
    config.plugin_path = plugin.toStdString();

    ptv2::qtui::MainWindow window(
        config, sha, parser.isSet(cloudOption) ? parser.value(cloudOption) : QString());
    window.show();
    return application.exec();
}
