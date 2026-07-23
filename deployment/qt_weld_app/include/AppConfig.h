#pragma once

#include <QMap>
#include <QString>

namespace ptv2::qtui
{

struct AppConfigValidation
{
    bool valid{false};
    bool engineExists{false};
    bool pluginExists{false};
    bool engineHashMatches{false};
    bool pluginHashMatches{false};
    QString engineActualSha256;
    QString pluginActualSha256;
    QString error;
};

struct AppConfig
{
    QString enginePath;
    QString pluginPath;
    QString engineSha256{
        QStringLiteral("a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299")};
    QString pluginSha256{
        QStringLiteral("6641ec147e8eac10206a5c60ba1c1390c398d4b59e32a6a618a37046360ec348")};
    QString defaultCloudDirectory;
    QString defaultExportDirectory;
    QString logDirectory{QStringLiteral("logs")};
    bool rememberLastCloud{true};
    bool rememberWindowGeometry{true};
    bool autoInitialize{true};
    bool showBoundingBox{true};
    bool showCentroid{true};
    bool showPcaDirection{true};
    double pointSize{3.0};
    int maximumLogFiles{20};

    static AppConfig defaults();
    static AppConfig loadLayered(
        QString const& defaultIni,
        QString const& userIni,
        QMap<QString, QString> const& processOverrides,
        QString& error);
    bool saveUser(QString const& userIni, QString& error) const;
    AppConfigValidation validateRuntime() const;
    QString resolvedPath(QString const& value, QString const& baseDirectory) const;
    static QString sha256File(QString const& path, QString& error);
};

} // namespace ptv2::qtui
