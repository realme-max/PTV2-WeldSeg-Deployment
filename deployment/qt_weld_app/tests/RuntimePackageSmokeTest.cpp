#include "RuntimePackageValidator.h"

#include <QDir>
#include <QFile>
#include <QTemporaryDir>
#include <QtTest>

namespace
{

void touch(QString const& root, QString const& relative, QByteArray const& content = QByteArray("x"))
{
    QString const path = QDir(root).filePath(relative);
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly)) qFatal("Cannot create %s", qPrintable(path));
    file.write(content);
}

}

class RuntimePackageSmokeTest final : public QObject
{
    Q_OBJECT

private slots:
    void validatesExternalPackageWhenProvided()
    {
        QString const package = QString::fromLocal8Bit(qgetenv("PTV2_PACKAGE_ROOT"));
        if (package.isEmpty()) QSKIP("PTV2_PACKAGE_ROOT not provided");
        auto const result = ptv2::qtui::RuntimePackageValidator::validate(package);
        QVERIFY2(result.valid, qPrintable(result.error
            + QStringLiteral("; missing=") + result.missing.join(QStringLiteral(","))
            + QStringLiteral("; forbidden=") + result.forbidden.join(QStringLiteral(","))
            + QStringLiteral("; absolute=")
            + result.absoluteSourceReferences.join(QStringLiteral(","))));
    }

    void validatesRelocatablePackageAndFailsClosed()
    {
        QTemporaryDir temporary;
        QString const root = temporary.path();
        QStringList const required{
            QStringLiteral("ptv2_weld_qt.exe"),
            QStringLiteral("config/qt_weld_app.ini"),
            QStringLiteral("engine/strict_fp32_voxelunique_cub.plan"),
            QStringLiteral("plugins/VoxelUniqueCubPlugin.dll"),
            QStringLiteral("platforms/qwindows.dll"),
            QStringLiteral("launch.bat"),
            QStringLiteral("runtime_inventory.json"),
            QStringLiteral("checksums.sha256")};
        for (QString const& file : required) touch(root, file);
        QVERIFY(ptv2::qtui::RuntimePackageValidator::validate(root).valid);

        QFile::remove(QDir(root).filePath(QStringLiteral("platforms/qwindows.dll")));
        QVERIFY(!ptv2::qtui::RuntimePackageValidator::validate(root).valid); // 11.
        touch(root, QStringLiteral("platforms/qwindows.dll"));
        touch(root, QStringLiteral("source.cpp"));
        QVERIFY(!ptv2::qtui::RuntimePackageValidator::validate(root).valid); // 12.
        QFile::remove(QDir(root).filePath(QStringLiteral("source.cpp")));
        touch(root, QStringLiteral("config/bad.ini"), QByteArray("E:\\GRP-PTv2"));
        QVERIFY(!ptv2::qtui::RuntimePackageValidator::validate(root).valid); // 13.
        QFile::remove(QDir(root).filePath(QStringLiteral("config/bad.ini")));
        QFile::remove(QDir(root).filePath(QStringLiteral("plugins/VoxelUniqueCubPlugin.dll")));
        QVERIFY(!ptv2::qtui::RuntimePackageValidator::validate(root).valid); // 14.
        touch(root, QStringLiteral("plugins/VoxelUniqueCubPlugin.dll"));
        QFile::remove(QDir(root).filePath(QStringLiteral("engine/strict_fp32_voxelunique_cub.plan")));
        QVERIFY(!ptv2::qtui::RuntimePackageValidator::validate(root).valid); // 15.
    }
};

QTEST_GUILESS_MAIN(RuntimePackageSmokeTest)
#include "RuntimePackageSmokeTest.moc"
