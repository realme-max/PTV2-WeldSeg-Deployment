#pragma once

#include "QtWeldResultViewModel.h"

#include <QImage>
#include <QString>
#include <QStringList>

namespace ptv2::qtui
{

struct DetectionExportIdentity
{
    QString applicationVersion;
    QString sdkVersion;
    QString engineSha256;
    QString pluginSha256;
};

struct DetectionExportResult
{
    bool success{false};
    QString directory;
    QStringList files;
    QString failingFile;
    QString error;
};

class DetectionExportService
{
public:
    static DetectionExportResult exportTask(
        QtWeldResultViewModel const& result,
        QImage const& screenshot,
        QString const& exportRoot,
        DetectionExportIdentity const& identity);
    static bool verifyTask(QString const& taskDirectory, QString& error);
};

} // namespace ptv2::qtui
