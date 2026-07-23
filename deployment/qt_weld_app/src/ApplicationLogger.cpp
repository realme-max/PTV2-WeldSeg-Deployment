#include "ApplicationLogger.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfoList>
#include <QTextStream>

namespace ptv2::qtui
{

ApplicationLogger::ApplicationLogger(QObject* parent)
    : QObject(parent)
{
}

ApplicationLogger::~ApplicationLogger()
{
    delete file_;
}

void ApplicationLogger::rotate(QString const& directory, int maximumFiles)
{
    QDir dir(directory);
    QFileInfoList files = dir.entryInfoList(
        QStringList() << QStringLiteral("ptv2_weld_*.log"),
        QDir::Files, QDir::Time);
    while (files.size() >= maximumFiles)
    {
        QFile::remove(files.takeLast().absoluteFilePath());
    }
}

bool ApplicationLogger::initialize(
    QString const& directory,
    int maximumFiles,
    QString& error)
{
    error.clear();
    if (!QDir().mkpath(directory))
    {
        error = QStringLiteral("Cannot create log directory: %1").arg(directory);
        return false;
    }
    rotate(directory, maximumFiles);
    path_ = QDir(directory).filePath(QStringLiteral("ptv2_weld_%1.log")
        .arg(QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss_zzz"))));
    file_ = new QFile(path_, this);
    if (!file_->open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text))
    {
        error = file_->errorString();
        delete file_;
        file_ = nullptr;
        return false;
    }
    return true;
}

void ApplicationLogger::log(
    QString const& level,
    QString const& category,
    QString const& message)
{
    QString const sanitized = QString(message).replace('\r', ' ').replace('\n', ' ');
    QString const line = QStringLiteral("%1 [%2] [%3] %4")
        .arg(QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss.zzz")),
            level, category, sanitized);
    if (file_ != nullptr)
    {
        QTextStream stream(file_);
        stream << line << '\n';
        stream.flush();
    }
    emit lineReady(line);
}

QString ApplicationLogger::currentLogPath() const
{
    return path_;
}

} // namespace ptv2::qtui
