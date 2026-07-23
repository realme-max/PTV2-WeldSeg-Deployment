#include "MainWindow.h"

#include "PointCloudRenderData.h"
#include "PointCloudView.h"
#include "WeldDetectionWorker.h"

#include <QCheckBox>
#include <QDateTime>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMetaObject>
#include <QPushButton>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QWidget>

#include <utility>

namespace ptv2::qtui
{

MainWindow::MainWindow(
    ptv2::weld::WeldConfig config,
    QString expectedEngineSha256,
    QString initialCloudPath,
    QWidget* parent)
    : QMainWindow(parent)
{
    buildUi();
    if (!initialCloudPath.isEmpty())
        cloudPathEdit_->setText(normalizedFilePath(initialCloudPath));

    worker_ = new WeldDetectionWorker(std::move(config), std::move(expectedEngineSha256));
    worker_->moveToThread(&workerThread_);
    connect(&workerThread_, &QThread::started, worker_, &WeldDetectionWorker::initialize);
    connect(&workerThread_, &QThread::finished, worker_, &QObject::deleteLater);
    connect(worker_, &WeldDetectionWorker::initializationFinished,
        this, &MainWindow::onInitializationFinished);
    connect(worker_, &WeldDetectionWorker::detectionStarted,
        this, &MainWindow::onDetectionStarted);
    connect(worker_, &WeldDetectionWorker::detectionSucceeded,
        this, &MainWindow::onDetectionSucceeded);
    connect(worker_, &WeldDetectionWorker::detectionFailed,
        this, &MainWindow::onDetectionFailed);
    connect(worker_, &WeldDetectionWorker::workerLog,
        this, &MainWindow::appendLog);
    workerThread_.start();
    appendLog(QStringLiteral("Worker thread started; SDK initialization queued"));
    updateControls();
}

MainWindow::~MainWindow()
{
    if (workerThread_.isRunning() && worker_ != nullptr)
        QMetaObject::invokeMethod(worker_, "shutdown", Qt::BlockingQueuedConnection);
    workerThread_.quit();
    workerThread_.wait();
    worker_ = nullptr;
}

void MainWindow::buildUi()
{
    setWindowTitle(QStringLiteral("PTV2 WeldDetector SDK Smoke"));
    resize(1320, 980);

    auto* central = new QWidget(this);
    auto* root = new QVBoxLayout(central);

    auto* inputGroup = new QGroupBox(QStringLiteral("Input and SDK initialization"), central);
    auto* inputLayout = new QFormLayout(inputGroup);
    auto* pathRow = new QWidget(inputGroup);
    auto* pathLayout = new QHBoxLayout(pathRow);
    pathLayout->setContentsMargins(0, 0, 0, 0);
    cloudPathEdit_ = new QLineEdit(pathRow);
    cloudPathEdit_->setObjectName(QStringLiteral("cloudPathEdit"));
    cloudPathEdit_->setReadOnly(true);
    browseButton_ = new QPushButton(QStringLiteral("Browse..."), pathRow);
    browseButton_->setObjectName(QStringLiteral("browseButton"));
    detectButton_ = new QPushButton(QStringLiteral("Detect"), pathRow);
    detectButton_->setObjectName(QStringLiteral("detectButton"));
    pathLayout->addWidget(cloudPathEdit_, 1);
    pathLayout->addWidget(browseButton_);
    pathLayout->addWidget(detectButton_);
    initializeStatus_ = new QLabel(QStringLiteral("INITIALIZING"), inputGroup);
    initializeStatus_->setObjectName(QStringLiteral("initializeStatus"));
    inputLayout->addRow(QStringLiteral("Cloud TXT"), pathRow);
    inputLayout->addRow(QStringLiteral("Initialize status"), initializeStatus_);
    root->addWidget(inputGroup);

    auto* visualizationGroup = new QGroupBox(QStringLiteral("Segmented point cloud"), central);
    auto* visualizationLayout = new QVBoxLayout(visualizationGroup);
    auto* visualizationControls = new QHBoxLayout();
    resetViewButton_ = new QPushButton(QStringLiteral("Reset View"), visualizationGroup);
    resetViewButton_->setObjectName(QStringLiteral("resetViewButton"));
    showBboxCheck_ = new QCheckBox(QStringLiteral("Show Bounding Box"), visualizationGroup);
    showBboxCheck_->setObjectName(QStringLiteral("showBboxCheck"));
    showBboxCheck_->setChecked(true);
    showPcaCheck_ = new QCheckBox(QStringLiteral("Show PCA Direction"), visualizationGroup);
    showPcaCheck_->setObjectName(QStringLiteral("showPcaCheck"));
    showPcaCheck_->setChecked(true);
    pointSizeSpin_ = new QDoubleSpinBox(visualizationGroup);
    pointSizeSpin_->setObjectName(QStringLiteral("pointSizeSpin"));
    pointSizeSpin_->setRange(1.0, 12.0);
    pointSizeSpin_->setValue(3.0);
    pointSizeSpin_->setSingleStep(0.5);
    visualizationControls->addWidget(resetViewButton_);
    visualizationControls->addWidget(showBboxCheck_);
    visualizationControls->addWidget(showPcaCheck_);
    visualizationControls->addWidget(new QLabel(QStringLiteral("Point size"), visualizationGroup));
    visualizationControls->addWidget(pointSizeSpin_);
    visualizationControls->addStretch(1);
    auto* legend = new QLabel(
        QStringLiteral("<span style='color:#ff4014'>● Weld seam (class 0)</span>"
                       "&nbsp;&nbsp;<span style='color:#2e8cf2'>● Background (class 1)</span>"),
        visualizationGroup);
    pointCloudView_ = new PointCloudView(visualizationGroup);
    visualizationLayout->addLayout(visualizationControls);
    visualizationLayout->addWidget(legend);
    visualizationLayout->addWidget(pointCloudView_, 1);
    root->addWidget(visualizationGroup, 2);

    auto* resultGroup = new QGroupBox(QStringLiteral("Weld detection result"), central);
    auto* resultLayout = new QFormLayout(resultGroup);
    auto addResult = [&](char const* key, QString const& label) {
        auto* value = new QLabel(QStringLiteral("-"), resultGroup);
        value->setTextInteractionFlags(Qt::TextSelectableByMouse);
        value->setObjectName(QString::fromLatin1(key));
        resultLabels_.emplace(key, value);
        resultLayout->addRow(label, value);
    };
    addResult("source", QStringLiteral("Task / source file"));
    addResult("sdk_status", QStringLiteral("SDK status"));
    addResult("original_points", QStringLiteral("Original point count"));
    addResult("sampled_points", QStringLiteral("Sampled point count"));
    addResult("weld_points", QStringLiteral("Weld point count"));
    addResult("weld_ratio", QStringLiteral("Weld ratio"));
    addResult("length_mm", QStringLiteral("PCA length (mm)"));
    addResult("center", QStringLiteral("Centroid X / Y / Z"));
    addResult("bbox_min", QStringLiteral("BBox min X / Y / Z"));
    addResult("bbox_max", QStringLiteral("BBox max X / Y / Z"));
    addResult("load_cloud_ms", QStringLiteral("Load-cloud time (ms)"));
    addResult("sampling_ms", QStringLiteral("Sampling time (ms)"));
    addResult("adjacency_build_ms", QStringLiteral("Adjacency-build time (ms)"));
    addResult("inference_cuda_ms", QStringLiteral("Inference CUDA time (ms)"));
    addResult("inference_wall_ms", QStringLiteral("Inference wall time (ms)"));
    addResult("postprocess_ms", QStringLiteral("Post-process time (ms)"));
    addResult("total_ms", QStringLiteral("Total time (ms)"));
    addResult("error_recorder_errors", QStringLiteral("ErrorRecorder errors"));
    root->addWidget(resultGroup);

    auto* logGroup = new QGroupBox(QStringLiteral("Runtime log"), central);
    auto* logLayout = new QVBoxLayout(logGroup);
    logEdit_ = new QTextEdit(logGroup);
    logEdit_->setObjectName(QStringLiteral("runtimeLog"));
    logEdit_->setReadOnly(true);
    logLayout->addWidget(logEdit_);
    root->addWidget(logGroup, 1);

    setCentralWidget(central);
    connect(browseButton_, &QPushButton::clicked, this, &MainWindow::browseCloud);
    connect(detectButton_, &QPushButton::clicked, this, &MainWindow::startDetection);
    connect(resetViewButton_, &QPushButton::clicked, this, &MainWindow::resetVisualization);
    connect(showBboxCheck_, &QCheckBox::toggled, pointCloudView_, &PointCloudView::setShowBoundingBox);
    connect(showPcaCheck_, &QCheckBox::toggled, pointCloudView_, &PointCloudView::setShowPcaDirection);
    connect(pointSizeSpin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        [this](double value) { pointCloudView_->setPointSize(static_cast<float>(value)); });
    connect(pointCloudView_, &PointCloudView::visualizationLog, this, &MainWindow::appendLog);
    connect(pointCloudView_, &PointCloudView::openGLStatusChanged,
        [this](bool ready, QString const& message) {
            appendLog(QStringLiteral("Visualization %1: %2")
                .arg(ready ? QStringLiteral("ready") : QStringLiteral("failed"), message));
        });
}

QString MainWindow::normalizedFilePath(QString const& path) const
{
    return QDir::toNativeSeparators(QFileInfo(path).absoluteFilePath());
}

void MainWindow::browseCloud()
{
    QString const selected = QFileDialog::getOpenFileName(
        this, QStringLiteral("Select weld point cloud"), {}, QStringLiteral("Point cloud TXT (*.txt)"));
    if (selected.isEmpty()) return;
    cloudPathEdit_->setText(normalizedFilePath(selected));
    appendLog(QStringLiteral("Cloud selected: %1").arg(cloudPathEdit_->text()));
    updateControls();
}

void MainWindow::startDetection()
{
    QString const path = cloudPathEdit_->text();
    QFileInfo const file(path);
    if (!file.isFile())
    {
        appendLog(QStringLiteral("POINTCLOUD_LOAD_FAILED: selected cloud does not exist"));
        QMessageBox::warning(this, QStringLiteral("Invalid point cloud"),
            QStringLiteral("Select an existing TXT point-cloud file before detection."));
        updateControls();
        return;
    }
    if (!initialized_ || detectionActive_)
    {
        appendLog(QStringLiteral("INVALID_CONFIG: SDK is not ready or detection is already active"));
        return;
    }
    if (!worker_->requestDetection(normalizedFilePath(path)))
    {
        appendLog(QStringLiteral("INVALID_CONFIG: concurrent Detect request rejected"));
        return;
    }
    detectionActive_ = true;
    updateControls();
}

void MainWindow::onInitializationFinished(QString status, QString message)
{
    initialized_ = status == QStringLiteral("SUCCESS");
    initializeStatus_->setText(status);
    appendLog(message.isEmpty()
        ? QStringLiteral("SDK initialization status: %1").arg(status)
        : QStringLiteral("SDK initialization status: %1; %2").arg(status, message));
    updateControls();
}

void MainWindow::onDetectionStarted(QString cloudPath)
{
    detectionActive_ = true;
    appendLog(QStringLiteral("Detection started: %1").arg(cloudPath));
    updateControls();
}

void MainWindow::onDetectionSucceeded(QtWeldResultViewModel result)
{
    detectionActive_ = false;
    PointCloudRenderData const renderData = PointCloudRenderData::fromResult(result);
    QString visualizationError;
    if (!pointCloudView_->setPointCloud(renderData, visualizationError))
    {
        appendLog(QStringLiteral("VISUALIZATION_DATA_FAILED: %1; previous successful view preserved")
            .arg(visualizationError));
    }
    else
    {
        appendLog(QStringLiteral("Visualization updated: points=%1, conversion=%2 ms")
            .arg(renderData.points.size()).arg(renderData.conversionMs, 0, 'f', 4));
    }
    setResult(QStringLiteral("source"), QStringLiteral("%1 | %2").arg(result.taskId, result.sourcePath));
    setResult(QStringLiteral("sdk_status"), result.status);
    setResult(QStringLiteral("original_points"), QString::number(result.originalPoints));
    setResult(QStringLiteral("sampled_points"), QString::number(result.sampledPoints));
    setResult(QStringLiteral("weld_points"), QString::number(result.weldPoints));
    setResult(QStringLiteral("weld_ratio"), QString::number(result.weldRatio, 'g', 12));
    setResult(QStringLiteral("length_mm"), QString::number(result.lengthMm, 'g', 12));
    setResult(QStringLiteral("center"), QStringLiteral("%1 / %2 / %3")
        .arg(result.centerX, 0, 'g', 10).arg(result.centerY, 0, 'g', 10).arg(result.centerZ, 0, 'g', 10));
    setResult(QStringLiteral("bbox_min"), QStringLiteral("%1 / %2 / %3")
        .arg(result.bboxMinX, 0, 'g', 10).arg(result.bboxMinY, 0, 'g', 10).arg(result.bboxMinZ, 0, 'g', 10));
    setResult(QStringLiteral("bbox_max"), QStringLiteral("%1 / %2 / %3")
        .arg(result.bboxMaxX, 0, 'g', 10).arg(result.bboxMaxY, 0, 'g', 10).arg(result.bboxMaxZ, 0, 'g', 10));
    setResult(QStringLiteral("load_cloud_ms"), QString::number(result.loadCloudMs, 'f', 4));
    setResult(QStringLiteral("sampling_ms"), QString::number(result.samplingMs, 'f', 4));
    setResult(QStringLiteral("adjacency_build_ms"), QString::number(result.adjacencyBuildMs, 'f', 4));
    setResult(QStringLiteral("inference_cuda_ms"), QString::number(result.inferenceCudaMs, 'f', 4));
    setResult(QStringLiteral("inference_wall_ms"), QString::number(result.inferenceWallMs, 'f', 4));
    setResult(QStringLiteral("postprocess_ms"), QString::number(result.postprocessMs, 'f', 4));
    setResult(QStringLiteral("total_ms"), QString::number(result.totalMs, 'f', 4));
    setResult(QStringLiteral("error_recorder_errors"), QString::number(result.errorRecorderErrors));
    appendLog(QStringLiteral("Detection succeeded: weld points=%1, length=%2 mm")
        .arg(result.weldPoints).arg(result.lengthMm, 0, 'g', 10));
    updateControls();
}

void MainWindow::resetVisualization()
{
    pointCloudView_->resetView();
    appendLog(QStringLiteral("Visualization view reset"));
}

void MainWindow::onDetectionFailed(QString status, QString message)
{
    detectionActive_ = false;
    setResult(QStringLiteral("sdk_status"), status);
    appendLog(QStringLiteral("Detection failed: %1: %2").arg(status, message));
    updateControls();
}

void MainWindow::appendLog(QString message)
{
    logEdit_->append(QStringLiteral("[%1] %2")
        .arg(QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss.zzz")), message));
}

void MainWindow::setResult(QString const& key, QString const& value)
{
    auto const found = resultLabels_.find(key.toStdString());
    if (found != resultLabels_.end()) found->second->setText(value);
}

void MainWindow::updateControls()
{
    bool const validCloud = QFileInfo(cloudPathEdit_->text()).isFile();
    browseButton_->setEnabled(!detectionActive_);
    detectButton_->setEnabled(initialized_ && validCloud && !detectionActive_);
}

} // namespace ptv2::qtui
