#include "AppConfig.h"

#include <QFile>
#include <QSettings>
#include <QTemporaryDir>
#include <QtTest>

class AppConfigTest final : public QObject
{
    Q_OBJECT

private slots:
    void layeredPriorityAndPersistence()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        QString const defaultsPath = temporary.filePath(QStringLiteral("defaults.ini"));
        QString const userPath = temporary.filePath(QStringLiteral("user.ini"));
        {
            QSettings settings(defaultsPath, QSettings::IniFormat);
            settings.setValue(QStringLiteral("Visualization/point_size"), 2.0);
            settings.setValue(QStringLiteral("Runtime/engine_path"), QStringLiteral("default.plan"));
        }
        {
            QSettings settings(userPath, QSettings::IniFormat);
            settings.setValue(QStringLiteral("Visualization/point_size"), 4.0);
            settings.setValue(QStringLiteral("Runtime/engine_path"), QStringLiteral("user.plan"));
        }
        QMap<QString, QString> overrides;
        overrides.insert(QStringLiteral("engine_path"), QStringLiteral("cli.plan"));
        QString error;
        ptv2::qtui::AppConfig config =
            ptv2::qtui::AppConfig::loadLayered(defaultsPath, userPath, overrides, error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(config.enginePath, QStringLiteral("cli.plan"));
        QCOMPARE(config.pointSize, 4.0);
        config.defaultExportDirectory = QStringLiteral("exports");
        QVERIFY(config.saveUser(userPath, error));
        QVERIFY(error.isEmpty());
    }

    void runtimeIntegrityFailClosed()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        QString const engine = temporary.filePath(QStringLiteral("engine.plan"));
        QString const plugin = temporary.filePath(QStringLiteral("plugin.dll"));
        QFile engineFile(engine);
        QVERIFY(engineFile.open(QIODevice::WriteOnly));
        engineFile.write("engine");
        engineFile.close();
        QFile pluginFile(plugin);
        QVERIFY(pluginFile.open(QIODevice::WriteOnly));
        pluginFile.write("plugin");
        pluginFile.close();
        QString error;
        ptv2::qtui::AppConfig config;
        config.enginePath = engine;
        config.pluginPath = plugin;
        config.engineSha256 = ptv2::qtui::AppConfig::sha256File(engine, error);
        config.pluginSha256 = ptv2::qtui::AppConfig::sha256File(plugin, error);
        QVERIFY(config.validateRuntime().valid);

        config.engineSha256 = QString(64, QLatin1Char('0'));
        QVERIFY(!config.validateRuntime().valid); // 1: Engine mismatch.
        config.engineSha256 = QStringLiteral("invalid");
        QVERIFY(!config.validateRuntime().valid); // 2: malformed SHA.
        config.enginePath = temporary.filePath(QStringLiteral("missing.plan"));
        QVERIFY(!config.validateRuntime().valid); // 3: missing Engine.
        config.enginePath = engine;
        config.pluginPath = temporary.filePath(QStringLiteral("missing.dll"));
        QVERIFY(!config.validateRuntime().valid); // 4: missing Plugin.
        config.pluginPath = plugin;
        config.engineSha256 = ptv2::qtui::AppConfig::sha256File(engine, error);
        config.pluginSha256 = QString(64, QLatin1Char('0'));
        QVERIFY(!config.validateRuntime().valid); // Wrong Plugin hash.

        QSettings malformed(temporary.filePath(QStringLiteral("malformed.ini")), QSettings::IniFormat);
        malformed.setValue(QStringLiteral("Visualization/point_size"), 999.0);
        malformed.sync();
        QString loadError;
        ptv2::qtui::AppConfig::loadLayered(
            temporary.filePath(QStringLiteral("malformed.ini")), {}, {}, loadError);
        QVERIFY(!loadError.isEmpty()); // Unsupported configuration value.
    }
};

QTEST_GUILESS_MAIN(AppConfigTest)
#include "AppConfigTest.moc"
