#include "RuntimePackageValidator.h"

#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>

namespace ptv2::qtui
{

RuntimePackageValidation RuntimePackageValidator::validate(QString const& packageRoot)
{
    RuntimePackageValidation result;
    QDir const root(packageRoot);
    QStringList const required{
        QStringLiteral("ptv2_weld_qt.exe"),
        QStringLiteral("config/qt_weld_app.ini"),
        QStringLiteral("engine/strict_fp32_voxelunique_cub.plan"),
        QStringLiteral("plugins/VoxelUniqueCubPlugin.dll"),
        QStringLiteral("platforms/qwindows.dll"),
        QStringLiteral("launch.bat"),
        QStringLiteral("runtime_inventory.json"),
        QStringLiteral("checksums.sha256")};
    for (QString const& relative : required)
    {
        if (!QFileInfo(root.filePath(relative)).isFile()) result.missing.append(relative);
    }
    QDirIterator iterator(packageRoot, QDir::Files, QDirIterator::Subdirectories);
    while (iterator.hasNext())
    {
        QString const path = iterator.next();
        QString const suffix = QFileInfo(path).suffix().toLower();
        if (suffix == QStringLiteral("cpp") || suffix == QStringLiteral("h")
            || suffix == QStringLiteral("obj") || suffix == QStringLiteral("pdb"))
            result.forbidden.append(root.relativeFilePath(path));
        if (suffix == QStringLiteral("ini") || suffix == QStringLiteral("bat")
            || suffix == QStringLiteral("ps1") || suffix == QStringLiteral("json")
            || suffix == QStringLiteral("txt"))
        {
            QFile file(path);
            if (file.open(QIODevice::ReadOnly))
            {
                QByteArray const content = file.readAll().toLower();
                if (content.contains("e:\\grp-ptv2") || content.contains("e:/grp-ptv2"))
                    result.absoluteSourceReferences.append(root.relativeFilePath(path));
            }
        }
    }
    result.valid = result.missing.isEmpty() && result.forbidden.isEmpty()
        && result.absoluteSourceReferences.isEmpty();
    if (!result.valid)
        result.error = QStringLiteral("Runtime package contract validation failed");
    return result;
}

} // namespace ptv2::qtui
