#include "AppConfig.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QSettings>

namespace ptv2::qtui
{
namespace
{

void readSettings(QSettings& settings, AppConfig& config)
{
    auto text = [&](char const* key, QString const& fallback) {
        return settings.value(QString::fromLatin1(key), fallback).toString().trimmed();
    };
    config.enginePath = text("Runtime/engine_path", config.enginePath);
    config.pluginPath = text("Runtime/plugin_path", config.pluginPath);
    config.engineSha256 = text("Runtime/engine_sha256", config.engineSha256).toLower();
    config.pluginSha256 = text("Runtime/plugin_sha256", config.pluginSha256).toLower();
    config.defaultCloudDirectory =
        text("Application/default_cloud_directory", config.defaultCloudDirectory);
    config.defaultExportDirectory =
        text("Application/default_export_directory", config.defaultExportDirectory);
    config.rememberLastCloud = settings.value(
        QStringLiteral("Application/remember_last_cloud"), config.rememberLastCloud).toBool();
    config.rememberWindowGeometry = settings.value(
        QStringLiteral("Application/remember_window_geometry"), config.rememberWindowGeometry).toBool();
    config.autoInitialize = settings.value(
        QStringLiteral("Application/auto_initialize"), config.autoInitialize).toBool();
    config.showBoundingBox = settings.value(
        QStringLiteral("Visualization/show_bbox"), config.showBoundingBox).toBool();
    config.showCentroid = settings.value(
        QStringLiteral("Visualization/show_centroid"), config.showCentroid).toBool();
    config.showPcaDirection = settings.value(
        QStringLiteral("Visualization/show_pca"), config.showPcaDirection).toBool();
    config.pointSize = settings.value(
        QStringLiteral("Visualization/point_size"), config.pointSize).toDouble();
    config.logDirectory = text("Logging/log_directory", config.logDirectory);
    config.maximumLogFiles = settings.value(
        QStringLiteral("Logging/maximum_log_files"), config.maximumLogFiles).toInt();
}

void writeSettings(QSettings& settings, AppConfig const& config)
{
    settings.setValue(QStringLiteral("Runtime/engine_path"), config.enginePath);
    settings.setValue(QStringLiteral("Runtime/plugin_path"), config.pluginPath);
    settings.setValue(QStringLiteral("Runtime/engine_sha256"), config.engineSha256);
    settings.setValue(QStringLiteral("Runtime/plugin_sha256"), config.pluginSha256);
    settings.setValue(
        QStringLiteral("Application/default_cloud_directory"), config.defaultCloudDirectory);
    settings.setValue(
        QStringLiteral("Application/default_export_directory"), config.defaultExportDirectory);
    settings.setValue(QStringLiteral("Application/remember_last_cloud"), config.rememberLastCloud);
    settings.setValue(
        QStringLiteral("Application/remember_window_geometry"), config.rememberWindowGeometry);
    settings.setValue(QStringLiteral("Application/auto_initialize"), config.autoInitialize);
    settings.setValue(QStringLiteral("Visualization/show_bbox"), config.showBoundingBox);
    settings.setValue(QStringLiteral("Visualization/show_centroid"), config.showCentroid);
    settings.setValue(QStringLiteral("Visualization/show_pca"), config.showPcaDirection);
    settings.setValue(QStringLiteral("Visualization/point_size"), config.pointSize);
    settings.setValue(QStringLiteral("Logging/log_directory"), config.logDirectory);
    settings.setValue(QStringLiteral("Logging/maximum_log_files"), config.maximumLogFiles);
}

} // namespace

AppConfig AppConfig::defaults()
{
    return AppConfig{};
}

AppConfig AppConfig::loadLayered(
    QString const& defaultIni,
    QString const& userIni,
    QMap<QString, QString> const& processOverrides,
    QString& error)
{
    error.clear();
    AppConfig config = defaults();
    if (!defaultIni.isEmpty() && QFileInfo(defaultIni).isFile())
    {
        QSettings defaultsSettings(defaultIni, QSettings::IniFormat);
        readSettings(defaultsSettings, config);
        if (defaultsSettings.status() != QSettings::NoError)
            error = QStringLiteral("Failed to read default INI: %1").arg(defaultIni);
    }
    if (!userIni.isEmpty() && QFileInfo(userIni).isFile())
    {
        QSettings userSettings(userIni, QSettings::IniFormat);
        readSettings(userSettings, config);
        if (userSettings.status() != QSettings::NoError)
            error = QStringLiteral("Failed to read user INI: %1").arg(userIni);
    }
    auto apply = [&](QString const& key, QString& destination) {
        auto const found = processOverrides.constFind(key);
        if (found != processOverrides.constEnd() && !found.value().trimmed().isEmpty())
            destination = found.value().trimmed();
    };
    apply(QStringLiteral("engine_path"), config.enginePath);
    apply(QStringLiteral("plugin_path"), config.pluginPath);
    apply(QStringLiteral("engine_sha256"), config.engineSha256);
    apply(QStringLiteral("plugin_sha256"), config.pluginSha256);
    apply(QStringLiteral("cloud_directory"), config.defaultCloudDirectory);
    apply(QStringLiteral("export_directory"), config.defaultExportDirectory);
    config.engineSha256 = config.engineSha256.toLower();
    config.pluginSha256 = config.pluginSha256.toLower();
    if (config.pointSize < 1.0 || config.pointSize > 12.0
        || config.maximumLogFiles < 1 || config.maximumLogFiles > 100)
    {
        error = QStringLiteral("Unsupported point_size or maximum_log_files value");
    }
    return config;
}

bool AppConfig::saveUser(QString const& userIni, QString& error) const
{
    error.clear();
    if (!QDir().mkpath(QFileInfo(userIni).absolutePath()))
    {
        error = QStringLiteral("Cannot create user configuration directory");
        return false;
    }
    QSettings settings(userIni, QSettings::IniFormat);
    writeSettings(settings, *this);
    settings.sync();
    if (settings.status() != QSettings::NoError)
    {
        error = QStringLiteral("QSettings failed to save %1").arg(userIni);
        return false;
    }
    return true;
}

QString AppConfig::sha256File(QString const& path, QString& error)
{
    error.clear();
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        error = file.errorString();
        return {};
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd())
    {
        QByteArray const block = file.read(4 * 1024 * 1024);
        if (block.isEmpty() && file.error() != QFile::NoError)
        {
            error = file.errorString();
            return {};
        }
        hash.addData(block);
    }
    return QString::fromLatin1(hash.result().toHex());
}

AppConfigValidation AppConfig::validateRuntime() const
{
    AppConfigValidation result;
    result.engineExists = QFileInfo(enginePath).isFile();
    result.pluginExists = QFileInfo(pluginPath).isFile();
    QRegularExpression const shaPattern(QStringLiteral("^[0-9a-f]{64}$"));
    if (!result.engineExists || !result.pluginExists)
    {
        result.error = QStringLiteral("Engine or Plugin path does not reference an existing file");
        return result;
    }
    if (!shaPattern.match(engineSha256).hasMatch()
        || !shaPattern.match(pluginSha256).hasMatch())
    {
        result.error = QStringLiteral("Expected SHA-256 must contain exactly 64 lowercase hex characters");
        return result;
    }
    QString hashError;
    result.engineActualSha256 = sha256File(enginePath, hashError);
    if (!hashError.isEmpty())
    {
        result.error = QStringLiteral("Engine hash failed: %1").arg(hashError);
        return result;
    }
    result.pluginActualSha256 = sha256File(pluginPath, hashError);
    if (!hashError.isEmpty())
    {
        result.error = QStringLiteral("Plugin hash failed: %1").arg(hashError);
        return result;
    }
    result.engineHashMatches = result.engineActualSha256 == engineSha256;
    result.pluginHashMatches = result.pluginActualSha256 == pluginSha256;
    result.valid = result.engineHashMatches && result.pluginHashMatches;
    if (!result.valid)
        result.error = QStringLiteral("Runtime SHA-256 validation failed");
    return result;
}

QString AppConfig::resolvedPath(QString const& value, QString const& baseDirectory) const
{
    if (value.isEmpty()) return {};
    QFileInfo const info(value);
    return QDir::cleanPath(info.isAbsolute() ? info.absoluteFilePath()
                                             : QDir(baseDirectory).absoluteFilePath(value));
}

} // namespace ptv2::qtui
