#include "RecentTaskStore.h"

#include <QTemporaryDir>
#include <QtTest>

class RecentTaskStoreTest final : public QObject
{
    Q_OBJECT

private slots:
    void storesDedupeCapsAndClears()
    {
        QTemporaryDir temporary;
        ptv2::qtui::RecentTaskStore store(
            temporary.filePath(QStringLiteral("settings.ini")), 20);
        QString error;
        for (int index = 0; index < 25; ++index)
        {
            ptv2::qtui::RecentTask task;
            task.taskId = QStringLiteral("task_%1").arg(index);
            task.sourceCloud = temporary.filePath(QStringLiteral("missing_%1.txt").arg(index));
            task.timestamp = QString::number(index);
            QVERIFY(store.add(task, error));
        }
        auto tasks = store.load(error);
        QCOMPARE(tasks.size(), 20);
        QVERIFY(tasks.first().sourceMissing);
        auto duplicate = tasks.first();
        duplicate.weldPoints = 209;
        QVERIFY(store.add(duplicate, error));
        tasks = store.load(error);
        QCOMPARE(tasks.size(), 20);
        QCOMPARE(tasks.first().weldPoints, 209);
        QVERIFY(!store.updateExport(QStringLiteral("unknown"), QStringLiteral("x"), error)); // 10.
        QVERIFY(store.clear(error));
        QVERIFY(store.load(error).isEmpty());
    }
};

QTEST_GUILESS_MAIN(RecentTaskStoreTest)
#include "RecentTaskStoreTest.moc"
