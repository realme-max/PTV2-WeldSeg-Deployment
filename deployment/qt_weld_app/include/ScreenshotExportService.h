#pragma once

#include <QImage>
#include <QString>

namespace ptv2::qtui
{

struct ScreenshotExportResult
{
    bool success{false};
    QString path;
    int width{};
    int height{};
    QString error;
};

class ScreenshotExportService
{
public:
    static ScreenshotExportResult savePng(QImage const& image, QString const& path);
};

} // namespace ptv2::qtui
