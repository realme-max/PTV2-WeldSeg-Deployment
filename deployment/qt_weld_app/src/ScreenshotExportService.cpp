#include "ScreenshotExportService.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>

namespace ptv2::qtui
{

ScreenshotExportResult ScreenshotExportService::savePng(
    QImage const& image,
    QString const& path)
{
    ScreenshotExportResult result;
    result.path = QFileInfo(path).absoluteFilePath();
    result.width = image.width();
    result.height = image.height();
    if (image.isNull() || image.width() <= 0 || image.height() <= 0)
    {
        result.error = QStringLiteral("Visualization framebuffer is empty");
        return result;
    }
    if (!QDir().mkpath(QFileInfo(result.path).absolutePath()))
    {
        result.error = QStringLiteral("Cannot create screenshot directory");
        return result;
    }
    QString const temporary = result.path + QStringLiteral(".tmp.png");
    QFile::remove(temporary);
    if (!image.save(temporary, "PNG"))
    {
        result.error = QStringLiteral("QImage failed to write PNG: %1").arg(temporary);
        return result;
    }
    QFile::remove(result.path);
    if (!QFile::rename(temporary, result.path))
    {
        QFile::remove(temporary);
        result.error = QStringLiteral("Cannot atomically promote PNG: %1").arg(result.path);
        return result;
    }
    result.success = true;
    return result;
}

} // namespace ptv2::qtui
