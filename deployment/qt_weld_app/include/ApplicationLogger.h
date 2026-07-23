#pragma once

#include <QObject>
#include <QString>

class QFile;

namespace ptv2::qtui
{

class ApplicationLogger final : public QObject
{
    Q_OBJECT

public:
    explicit ApplicationLogger(QObject* parent = nullptr);
    ~ApplicationLogger() override;

    bool initialize(QString const& directory, int maximumFiles, QString& error);
    void log(QString const& level, QString const& category, QString const& message);
    QString currentLogPath() const;

signals:
    void lineReady(QString line);

private:
    void rotate(QString const& directory, int maximumFiles);
    QFile* file_{nullptr};
    QString path_;
};

} // namespace ptv2::qtui
