#pragma once

#include "QtWeldResultViewModel.h"
#include "WeldConfig.h"
#include "WeldDetector.h"

#include <QObject>
#include <QString>

#include <atomic>
#include <memory>

namespace ptv2::qtui
{

class WeldDetectionWorker final : public QObject
{
    Q_OBJECT

public:
    WeldDetectionWorker(
        ptv2::weld::WeldConfig config,
        QString expectedEngineSha256,
        QObject* parent = nullptr);
    ~WeldDetectionWorker() override;

    Q_INVOKABLE bool requestDetection(QString const& cloudPath);

public slots:
    void initialize();
    void shutdown();

private slots:
    void detectImpl(QString cloudPath);

signals:
    void initializationFinished(QString status, QString message);
    void detectionStarted(QString cloudPath);
    void detectionSucceeded(ptv2::qtui::QtWeldResultViewModel result);
    void detectionFailed(QString status, QString message);
    void workerLog(QString message);
    void shutdownFinished();

private:
    QString validateEngineSha256() const;

    ptv2::weld::WeldConfig config_;
    QString expectedEngineSha256_;
    std::unique_ptr<ptv2::weld::WeldDetector> detector_;
    std::atomic_bool detectionScheduled_{false};
    bool initialized_{false};
};

} // namespace ptv2::qtui
