#pragma once

#include <QList>
#include <QString>

namespace ptv2::qtui
{

struct RecentTask
{
    QString taskId;
    QString sourceCloud;
    QString timestamp;
    int weldPoints{};
    double weldRatio{};
    double lengthMm{};
    double totalMs{};
    QString exportedDirectory;
    QString status;
    bool sourceMissing{false};
};

class RecentTaskStore
{
public:
    explicit RecentTaskStore(QString settingsPath, int maximumTasks = 20);
    QList<RecentTask> load(QString& error) const;
    bool add(RecentTask const& task, QString& error);
    bool updateExport(QString const& taskId, QString const& exportDirectory, QString& error);
    bool clear(QString& error);

private:
    bool save(QList<RecentTask> const& tasks, QString& error) const;
    QString settingsPath_;
    int maximumTasks_{20};
};

} // namespace ptv2::qtui
