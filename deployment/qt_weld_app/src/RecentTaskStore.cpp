#include "RecentTaskStore.h"

#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSettings>

#include <utility>

namespace ptv2::qtui
{
namespace
{

QJsonObject toJson(RecentTask const& task)
{
    QJsonObject object;
    object.insert(QStringLiteral("task_id"), task.taskId);
    object.insert(QStringLiteral("source_cloud"), task.sourceCloud);
    object.insert(QStringLiteral("timestamp"), task.timestamp);
    object.insert(QStringLiteral("weld_points"), task.weldPoints);
    object.insert(QStringLiteral("weld_ratio"), task.weldRatio);
    object.insert(QStringLiteral("length_mm"), task.lengthMm);
    object.insert(QStringLiteral("total_ms"), task.totalMs);
    object.insert(QStringLiteral("exported_directory"), task.exportedDirectory);
    object.insert(QStringLiteral("status"), task.status);
    return object;
}

RecentTask fromJson(QJsonObject const& object)
{
    RecentTask task;
    task.taskId = object.value(QStringLiteral("task_id")).toString();
    task.sourceCloud = object.value(QStringLiteral("source_cloud")).toString();
    task.timestamp = object.value(QStringLiteral("timestamp")).toString();
    task.weldPoints = object.value(QStringLiteral("weld_points")).toInt();
    task.weldRatio = object.value(QStringLiteral("weld_ratio")).toDouble();
    task.lengthMm = object.value(QStringLiteral("length_mm")).toDouble();
    task.totalMs = object.value(QStringLiteral("total_ms")).toDouble();
    task.exportedDirectory = object.value(QStringLiteral("exported_directory")).toString();
    task.status = object.value(QStringLiteral("status")).toString();
    task.sourceMissing = !QFileInfo(task.sourceCloud).isFile();
    return task;
}

} // namespace

RecentTaskStore::RecentTaskStore(QString settingsPath, int maximumTasks)
    : settingsPath_(std::move(settingsPath)), maximumTasks_(maximumTasks)
{
}

QList<RecentTask> RecentTaskStore::load(QString& error) const
{
    error.clear();
    QSettings settings(settingsPath_, QSettings::IniFormat);
    QList<RecentTask> result;
    int const count = settings.beginReadArray(QStringLiteral("RecentTasks/items"));
    for (int index = 0; index < count; ++index)
    {
        settings.setArrayIndex(index);
        QJsonParseError parseError;
        QJsonDocument const document = QJsonDocument::fromJson(
            settings.value(QStringLiteral("json")).toByteArray(), &parseError);
        if (parseError.error != QJsonParseError::NoError || !document.isObject())
        {
            error = QStringLiteral("Recent task %1 is corrupt: %2")
                .arg(index).arg(parseError.errorString());
            settings.endArray();
            return {};
        }
        result.append(fromJson(document.object()));
    }
    settings.endArray();
    if (settings.status() != QSettings::NoError)
        error = QStringLiteral("Cannot read recent task settings");
    return result;
}

bool RecentTaskStore::save(QList<RecentTask> const& tasks, QString& error) const
{
    error.clear();
    QSettings settings(settingsPath_, QSettings::IniFormat);
    settings.remove(QStringLiteral("RecentTasks"));
    settings.beginWriteArray(QStringLiteral("RecentTasks/items"), tasks.size());
    for (int index = 0; index < tasks.size(); ++index)
    {
        settings.setArrayIndex(index);
        settings.setValue(QStringLiteral("json"),
            QJsonDocument(toJson(tasks.at(index))).toJson(QJsonDocument::Compact));
    }
    settings.endArray();
    settings.sync();
    if (settings.status() != QSettings::NoError)
    {
        error = QStringLiteral("Cannot save recent task settings");
        return false;
    }
    return true;
}

bool RecentTaskStore::add(RecentTask const& task, QString& error)
{
    QList<RecentTask> tasks = load(error);
    if (!error.isEmpty()) return false;
    for (int index = tasks.size() - 1; index >= 0; --index)
    {
        if (tasks.at(index).taskId == task.taskId)
            tasks.removeAt(index);
    }
    tasks.prepend(task);
    while (tasks.size() > maximumTasks_) tasks.removeLast();
    return save(tasks, error);
}

bool RecentTaskStore::updateExport(
    QString const& taskId,
    QString const& exportDirectory,
    QString& error)
{
    QList<RecentTask> tasks = load(error);
    if (!error.isEmpty()) return false;
    for (RecentTask& task : tasks)
    {
        if (task.taskId == taskId)
        {
            task.exportedDirectory = exportDirectory;
            return save(tasks, error);
        }
    }
    error = QStringLiteral("Recent task not found: %1").arg(taskId);
    return false;
}

bool RecentTaskStore::clear(QString& error)
{
    return save({}, error);
}

} // namespace ptv2::qtui
