#include "WeldDetectionWorker.h"

#include "WeldResult.h"
#include "WeldStatus.h"

#include <QCryptographicHash>
#include <QFile>
#include <QMetaObject>

#include <utility>

namespace ptv2::qtui
{

WeldDetectionWorker::WeldDetectionWorker(
    ptv2::weld::WeldConfig config,
    QString expectedEngineSha256,
    QObject* parent)
    : QObject(parent),
      config_(std::move(config)),
      expectedEngineSha256_(expectedEngineSha256.trimmed().toLower())
{
}

WeldDetectionWorker::~WeldDetectionWorker() = default;

bool WeldDetectionWorker::requestDetection(QString const& cloudPath)
{
    bool expected = false;
    if (!detectionScheduled_.compare_exchange_strong(expected, true))
        return false;
    bool const queued = QMetaObject::invokeMethod(
        this, "detectImpl", Qt::QueuedConnection, Q_ARG(QString, cloudPath));
    if (!queued) detectionScheduled_.store(false);
    return queued;
}

QString WeldDetectionWorker::validateEngineSha256() const
{
    QFile engine(QString::fromStdString(config_.engine_path));
    if (!engine.open(QIODevice::ReadOnly))
        return QStringLiteral("Cannot open Engine for SHA-256 validation: %1").arg(engine.errorString());
    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (!hash.addData(&engine))
        return QStringLiteral("Failed to read Engine during SHA-256 validation");
    QString const actual = QString::fromLatin1(hash.result().toHex()).toLower();
    if (actual != expectedEngineSha256_)
    {
        return QStringLiteral("Engine SHA-256 mismatch: actual=%1, expected=%2")
            .arg(actual, expectedEngineSha256_);
    }
    return {};
}

void WeldDetectionWorker::initialize()
{
    emit workerLog(QStringLiteral("WeldDetector initialization started"));
    QString const shaError = validateEngineSha256();
    if (!shaError.isEmpty())
    {
        initialized_ = false;
        emit workerLog(shaError);
        emit initializationFinished(QStringLiteral("ENGINE_LOAD_FAILED"), shaError);
        return;
    }

    detector_ = std::make_unique<ptv2::weld::WeldDetector>();
    ptv2::weld::WeldStatus const status = detector_->initialize(config_);
    QString const symbolic = QString::fromLatin1(ptv2::weld::toString(status));
    QString const detail = QString::fromStdString(detector_->lastError());
    initialized_ = status == ptv2::weld::WeldStatus::SUCCESS;
    emit workerLog(initialized_
        ? QStringLiteral("WeldDetector initialization succeeded")
        : QStringLiteral("WeldDetector initialization failed: %1: %2").arg(symbolic, detail));
    emit initializationFinished(symbolic, detail);
}

void WeldDetectionWorker::detectImpl(QString cloudPath)
{
    emit detectionStarted(cloudPath);
    emit workerLog(QStringLiteral("Detection started: %1").arg(cloudPath));
    if (!initialized_ || !detector_)
    {
        detectionScheduled_.store(false);
        QString const message = QStringLiteral("Detection requested before successful SDK initialization");
        emit workerLog(message);
        emit detectionFailed(QStringLiteral("INVALID_CONFIG"), message);
        return;
    }

    ptv2::weld::WeldResult result;
    ptv2::weld::WeldStatus const status = detector_->detect(cloudPath.toStdString(), result);
    QString const symbolic = QString::fromLatin1(ptv2::weld::toString(status));
    if (status != ptv2::weld::WeldStatus::SUCCESS)
    {
        detectionScheduled_.store(false);
        QString const detail = QString::fromStdString(detector_->lastError());
        emit workerLog(QStringLiteral("Detection failed: %1: %2").arg(symbolic, detail));
        emit detectionFailed(symbolic, detail);
        return;
    }

    QtWeldResultViewModel view = QtWeldResultViewModel::fromSdk(result, cloudPath);
    detectionScheduled_.store(false);
    emit workerLog(QStringLiteral("Detection succeeded: %1").arg(cloudPath));
    emit detectionSucceeded(view);
}

void WeldDetectionWorker::shutdown()
{
    detector_.reset();
    initialized_ = false;
    detectionScheduled_.store(false);
    emit workerLog(QStringLiteral("WeldDetector worker shut down"));
    emit shutdownFinished();
}

} // namespace ptv2::qtui
