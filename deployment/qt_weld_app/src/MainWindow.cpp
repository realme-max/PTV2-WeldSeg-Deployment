#include "MainWindow.h"

#include "ApplicationLogger.h"
#include "DetectionExportService.h"
#include "PointCloudRenderData.h"
#include "PointCloudView.h"
#include "ProductInfo.h"
#include "RecentTaskStore.h"
#include "ScreenshotExportService.h"
#include "SettingsDialog.h"
#include "WeldDetectionWorker.h"

#include <QCheckBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFutureWatcher>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QMetaObject>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QGuiApplication>
#include <QScreen>
#include <QScrollArea>
#include <QSettings>
#include <QSizePolicy>
#include <QSplitter>
#include <QStandardPaths>
#include <QStatusBar>
#include <QTextDocument>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>
#include <QtConcurrent/QtConcurrentRun>
#include <QCloseEvent>

#include <algorithm>
#include <utility>

namespace ptv2::qtui
{
namespace
{

AppConfig legacyAppConfig(
    ptv2::weld::WeldConfig const& config,
    QString const& engineSha256)
{
    AppConfig app = AppConfig::defaults();
    app.enginePath = QString::fromStdString(config.engine_path);
    app.pluginPath = QString::fromStdString(config.plugin_path);
    app.engineSha256 = engineSha256;
    QString hashError;
    app.pluginSha256 = AppConfig::sha256File(app.pluginPath, hashError);
    return app;
}

} // namespace

MainWindow::MainWindow(
    ptv2::weld::WeldConfig config,
    QString expectedEngineSha256,
    QString initialCloudPath,
    QWidget* parent)
    : MainWindow(
        legacyAppConfig(config, expectedEngineSha256),
        std::move(initialCloudPath),
        QDir(QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation))
            .filePath(QStringLiteral("qt_weld_app.ini")),
        parent)
{
}

MainWindow::MainWindow(
    AppConfig config,
    QString initialCloudPath,
    QString userSettingsPath,
    QWidget* parent)
    : QMainWindow(parent),
      appConfig_(std::move(config)),
      userSettingsPath_(std::move(userSettingsPath))
{
    buildUi();
    restoreWindowLayout();
    if (!initialCloudPath.isEmpty())
        cloudPathEdit_->setText(normalizedFilePath(initialCloudPath));

    logger_ = std::make_unique<ApplicationLogger>();
    connect(logger_.get(), &ApplicationLogger::lineReady,
        logEdit_, &QPlainTextEdit::appendPlainText);
    QString logError;
    QString const logPath = QFileInfo(appConfig_.logDirectory).isAbsolute()
        ? appConfig_.logDirectory
        : QDir(QFileInfo(userSettingsPath_).absolutePath()).filePath(appConfig_.logDirectory);
    if (!logger_->initialize(logPath, appConfig_.maximumLogFiles, logError))
        logEdit_->appendPlainText(QStringLiteral("[LOGGER FAILED] %1").arg(logError));
    recentStore_ = std::make_unique<RecentTaskStore>(userSettingsPath_, 20);
    showBboxCheck_->setChecked(appConfig_.showBoundingBox);
    showPcaCheck_->setChecked(appConfig_.showPcaDirection);
    pointSizeSpin_->setValue(appConfig_.pointSize);
    refreshRecentTasks();
    setResult(QStringLiteral("engine_integrity"),
        QStringLiteral("PASS | %1").arg(appConfig_.engineSha256.left(12)));
    setResult(QStringLiteral("plugin_integrity"),
        QStringLiteral("PASS | %1").arg(appConfig_.pluginSha256.left(12)));
    setState(AppState::kInitializing);
    startWorker();
    appendLog(QStringLiteral("Product startup; validated Engine=%1 Plugin=%2")
        .arg(appConfig_.engineSha256.left(12), appConfig_.pluginSha256.left(12)));
    updateControls();
}

void MainWindow::startWorker()
{
    ptv2::weld::WeldConfig config;
    config.engine_path = appConfig_.enginePath.toStdString();
    config.plugin_path = appConfig_.pluginPath.toStdString();
    worker_ = new WeldDetectionWorker(std::move(config), appConfig_.engineSha256);
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
}

MainWindow::~MainWindow()
{
    QString ignored;
    stateMachine_.transition(AppState::kShuttingDown, ignored);
    appendLog(QStringLiteral("Application shutdown requested"));
    stopWorker();
}

void MainWindow::stopWorker()
{
    if (workerThread_.isRunning() && worker_ != nullptr)
        QMetaObject::invokeMethod(worker_, "shutdown", Qt::BlockingQueuedConnection);
    workerThread_.quit();
    workerThread_.wait();
    worker_ = nullptr;
}

void MainWindow::buildUi()
{
    setWindowTitle(QStringLiteral("%1 %2")
        .arg(ProductInfo::applicationName(), ProductInfo::applicationVersion()));
    resize(1200, 760);

    auto* central = new QWidget(this);
    central->setObjectName(QStringLiteral("centralWidget"));
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
    exportResultButton_ = new QPushButton(QStringLiteral("Export Result"), pathRow);
    exportResultButton_->setObjectName(QStringLiteral("exportResultButton"));
    exportScreenshotButton_ = new QPushButton(QStringLiteral("Export Screenshot"), pathRow);
    exportScreenshotButton_->setObjectName(QStringLiteral("exportScreenshotButton"));
    settingsButton_ = new QPushButton(QStringLiteral("Settings"), pathRow);
    settingsButton_->setObjectName(QStringLiteral("settingsButton"));
    productInfoButton_ = new QPushButton(QStringLiteral("Product Info"), pathRow);
    productInfoButton_->setObjectName(QStringLiteral("productInfoButton"));
    pathLayout->addWidget(cloudPathEdit_, 1);
    pathLayout->addWidget(browseButton_);
    initializeStatus_ = new QLabel(QStringLiteral("INITIALIZING"), inputGroup);
    initializeStatus_->setObjectName(QStringLiteral("initializeStatus"));
    auto* actionRow = new QWidget(inputGroup);
    actionRow->setObjectName(QStringLiteral("actionRow"));
    auto* actionLayout = new QHBoxLayout(actionRow);
    actionLayout->setContentsMargins(0, 0, 0, 0);
    actionLayout->addWidget(detectButton_);
    actionLayout->addWidget(exportResultButton_);
    actionLayout->addWidget(exportScreenshotButton_);
    actionLayout->addWidget(settingsButton_);
    actionLayout->addWidget(productInfoButton_);
    actionLayout->addStretch(1);
    inputLayout->addRow(QStringLiteral("Cloud TXT"), pathRow);
    inputLayout->addRow(QStringLiteral("Actions"), actionRow);
    inputLayout->addRow(QStringLiteral("Initialize status"), initializeStatus_);
    root->addWidget(inputGroup);

    auto* visualizationGroup = new QGroupBox(QStringLiteral("Segmented point cloud"), central);
    visualizationGroup->setObjectName(QStringLiteral("visualizationGroup"));
    visualizationGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
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

    auto* rightContent = new QWidget();
    rightContent->setObjectName(QStringLiteral("rightScrollContent"));
    rightContent->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
    auto* rightContentLayout = new QVBoxLayout(rightContent);
    rightContentLayout->setAlignment(Qt::AlignTop);

    auto* resultGroup = new QGroupBox(QStringLiteral("Weld detection result"), rightContent);
    resultGroup->setObjectName(QStringLiteral("resultGroup"));
    auto* resultLayout = new QFormLayout(resultGroup);
    resultLayout->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    resultLayout->setRowWrapPolicy(QFormLayout::WrapLongRows);
    auto addResult = [&](char const* key, QString const& label) {
        auto* value = new QLabel(QStringLiteral("-"), resultGroup);
        value->setTextInteractionFlags(Qt::TextSelectableByMouse);
        value->setWordWrap(true);
        value->setMinimumWidth(0);
        value->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
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
    addResult("engine_integrity", QStringLiteral("Engine integrity"));
    addResult("plugin_integrity", QStringLiteral("Plugin integrity"));
    rightContentLayout->addWidget(resultGroup);

    auto* recentGroup = new QGroupBox(QStringLiteral("Recent successful tasks"), rightContent);
    recentGroup->setObjectName(QStringLiteral("recentTasksGroup"));
    auto* recentLayout = new QVBoxLayout(recentGroup);
    recentTasks_ = new QListWidget(recentGroup);
    recentTasks_->setObjectName(QStringLiteral("recentTasks"));
    recentTasks_->setMinimumHeight(120);
    recentTasks_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    recentTasks_->setTextElideMode(Qt::ElideMiddle);
    recentLayout->addWidget(recentTasks_);
    auto* clearHistory = new QPushButton(QStringLiteral("Clear History"), recentGroup);
    recentLayout->addWidget(clearHistory);
    rightContentLayout->addWidget(recentGroup);

    auto* logGroup = new QGroupBox(QStringLiteral("Runtime log"), rightContent);
    logGroup->setObjectName(QStringLiteral("runtimeLogGroup"));
    auto* logLayout = new QVBoxLayout(logGroup);
    logEdit_ = new QPlainTextEdit(logGroup);
    logEdit_->setObjectName(QStringLiteral("runtimeLog"));
    logEdit_->setReadOnly(true);
    logEdit_->setMinimumHeight(160);
    logEdit_->setLineWrapMode(QPlainTextEdit::WidgetWidth);
    logEdit_->document()->setMaximumBlockCount(2000);
    logLayout->addWidget(logEdit_);
    rightContentLayout->addWidget(logGroup);
    rightContentLayout->addStretch(1);

    rightScrollArea_ = new QScrollArea(central);
    rightScrollArea_->setObjectName(QStringLiteral("rightScrollArea"));
    rightScrollArea_->setWidgetResizable(true);
    rightScrollArea_->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    rightScrollArea_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    rightScrollArea_->setFrameShape(QFrame::StyledPanel);
    rightScrollArea_->setMinimumWidth(280);
    rightScrollArea_->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    rightScrollArea_->setWidget(rightContent);

    mainContentSplitter_ = new QSplitter(Qt::Horizontal, central);
    mainContentSplitter_->setObjectName(QStringLiteral("mainContentSplitter"));
    mainContentSplitter_->setChildrenCollapsible(false);
    mainContentSplitter_->addWidget(visualizationGroup);
    mainContentSplitter_->addWidget(rightScrollArea_);
    mainContentSplitter_->setStretchFactor(0, 3);
    mainContentSplitter_->setStretchFactor(1, 1);
    mainContentSplitter_->setSizes(QList<int>() << 880 << 320);
    root->addWidget(mainContentSplitter_, 1);

    setCentralWidget(central);
    stateStatus_ = new QLabel(QStringLiteral("STARTING"), this);
    progress_ = new QProgressBar(this);
    progress_->setRange(0, 0);
    progress_->setMaximumWidth(160);
    progress_->setVisible(false);
    statusBar()->addPermanentWidget(stateStatus_);
    statusBar()->addPermanentWidget(progress_);
    connect(browseButton_, &QPushButton::clicked, this, &MainWindow::browseCloud);
    connect(detectButton_, &QPushButton::clicked, this, &MainWindow::startDetection);
    connect(exportResultButton_, &QPushButton::clicked, this, &MainWindow::exportResult);
    connect(exportScreenshotButton_, &QPushButton::clicked, this, &MainWindow::exportScreenshot);
    connect(settingsButton_, &QPushButton::clicked, this, &MainWindow::openSettings);
    connect(productInfoButton_, &QPushButton::clicked, this, &MainWindow::showProductInfo);
    connect(recentTasks_, &QListWidget::itemDoubleClicked, this, &MainWindow::loadRecentTask);
    connect(clearHistory, &QPushButton::clicked, this, [this] {
        if (QMessageBox::question(this, QStringLiteral("Clear recent tasks"),
                QStringLiteral("Clear all locally stored recent-task entries?"))
            != QMessageBox::Yes)
            return;
        QString error;
        if (!recentStore_ || !recentStore_->clear(error))
            appendLog(QStringLiteral("Recent task clear failed: %1").arg(error));
        refreshRecentTasks();
    });
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

QString MainWindow::resolveInitialCloudDirectory() const
{
    auto existingDirectory = [](QString const& candidate) {
        QString const trimmed = candidate.trimmed();
        if (trimmed.isEmpty()) return QString();
        QFileInfo const info(trimmed);
        if (!info.isDir()) return QString();
        return QDir::toNativeSeparators(info.absoluteFilePath());
    };

    if (appConfig_.rememberLastCloud && !userSettingsPath_.isEmpty())
    {
        QSettings settings(userSettingsPath_, QSettings::IniFormat);
        QString const lastDirectory = existingDirectory(
            settings.value(QStringLiteral("Application/last_cloud_directory")).toString());
        if (!lastDirectory.isEmpty()) return lastDirectory;
    }

    QString const configuredDirectory =
        existingDirectory(appConfig_.defaultCloudDirectory);
    if (!configuredDirectory.isEmpty()) return configuredDirectory;

    QString const applicationDirectory = QCoreApplication::applicationDirPath();
    QString const packageDirectory = existingDirectory(
        QDir(applicationDirectory).filePath(QStringLiteral("data/weld/000001")));
    if (!packageDirectory.isEmpty()) return packageDirectory;

    QString const developmentDirectory =
        existingDirectory(QStringLiteral("E:/GRP-PTv2/data/weld/000001"));
    if (!developmentDirectory.isEmpty()) return developmentDirectory;

    QString const executableDirectory = existingDirectory(applicationDirectory);
    if (!executableDirectory.isEmpty()) return executableDirectory;

    return QDir::homePath();
}

void MainWindow::persistLastCloudDirectory(QString const& selectedFile)
{
    if (!appConfig_.rememberLastCloud || userSettingsPath_.isEmpty()) return;

    QFileInfo const selectedInfo(selectedFile);
    QString const selectedDirectory =
        QDir::toNativeSeparators(selectedInfo.absolutePath());
    if (!QDir(selectedDirectory).exists()) return;

    QSettings settings(userSettingsPath_, QSettings::IniFormat);
    settings.setValue(
        QStringLiteral("Application/last_cloud_directory"), selectedDirectory);
    settings.sync();
    if (settings.status() != QSettings::NoError)
    {
        appendLog(QStringLiteral("LAST_CLOUD_DIRECTORY_SAVE_FAILED: %1")
            .arg(userSettingsPath_));
    }
}

void MainWindow::applyCloudSelection(QString const& selectedFile)
{
    if (selectedFile.isEmpty()) return;
    cloudPathEdit_->setText(normalizedFilePath(selectedFile));
    persistLastCloudDirectory(selectedFile);
    appendLog(QStringLiteral("Cloud selected: %1").arg(cloudPathEdit_->text()));
    setState(AppState::kCloudSelected);
    updateControls();
}

void MainWindow::browseCloud()
{
    QString const selected = QFileDialog::getOpenFileName(
        this, QStringLiteral("Select weld point cloud"), resolveInitialCloudDirectory(),
        QStringLiteral("Point cloud TXT (*.txt)"));
    applyCloudSelection(selected);
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
    setState(AppState::kDetecting);
    updateControls();
}

void MainWindow::onInitializationFinished(QString status, QString message)
{
    initialized_ = status == QStringLiteral("SUCCESS");
    initializeStatus_->setText(status);
    appendLog(message.isEmpty()
        ? QStringLiteral("SDK initialization status: %1").arg(status)
        : QStringLiteral("SDK initialization status: %1; %2").arg(status, message));
    if (initialized_)
    {
        setState(AppState::kReady);
        if (QFileInfo(cloudPathEdit_->text()).isFile())
            setState(AppState::kCloudSelected);
    }
    else
    {
        setState(AppState::kConfigurationInvalid);
        if (smokePending_)
        {
            smokePending_ = false;
            emit productSmokeCompleted(false, {}, message);
        }
    }
    if (initialized_ && smokePending_)
        QTimer::singleShot(0, this, &MainWindow::startDetection);
}

void MainWindow::onDetectionStarted(QString cloudPath)
{
    detectionActive_ = true;
    appendLog(QStringLiteral("Detection started: %1").arg(cloudPath));
    if (stateMachine_.state() != AppState::kDetecting)
        setState(AppState::kDetecting);
    updateControls();
}

void MainWindow::onDetectionSucceeded(QtWeldResultViewModel result)
{
    detectionActive_ = false;
    lastResult_ = result;
    hasResult_ = true;
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
    if (recentStore_)
    {
        RecentTask task;
        task.taskId = result.taskId;
        task.sourceCloud = result.sourcePath;
        task.timestamp = QDateTime::currentDateTime().toString(Qt::ISODateWithMs);
        task.weldPoints = result.weldPoints;
        task.weldRatio = result.weldRatio;
        task.lengthMm = result.lengthMm;
        task.totalMs = result.totalMs;
        task.status = result.status;
        QString historyError;
        if (!recentStore_->add(task, historyError))
            appendLog(QStringLiteral("Recent task storage failed: %1").arg(historyError));
        refreshRecentTasks();
    }
    setState(AppState::kDetectionSucceeded);
    if (smokePending_)
    {
        QTimer::singleShot(250, this, [this] {
            performExport(smokeExportRoot_, true);
        });
    }
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
    setState(AppState::kDetectionFailed);
    if (smokePending_)
    {
        smokePending_ = false;
        emit productSmokeCompleted(false, {}, QStringLiteral("%1: %2").arg(status, message));
    }
}

void MainWindow::appendLog(QString message)
{
    if (logger_)
    {
        QString category = QStringLiteral("Application");
        if (message.contains(QStringLiteral("startup"), Qt::CaseInsensitive))
            category = QStringLiteral("Startup");
        else if (message.contains(QStringLiteral("configuration"), Qt::CaseInsensitive)
            || message.contains(QStringLiteral("settings"), Qt::CaseInsensitive))
            category = QStringLiteral("Configuration");
        else if (message.contains(QStringLiteral("SDK"), Qt::CaseInsensitive)
            || message.contains(QStringLiteral("worker"), Qt::CaseInsensitive))
            category = QStringLiteral("SDKInitialization");
        else if (message.contains(QStringLiteral("Detection"), Qt::CaseInsensitive))
            category = QStringLiteral("Detection");
        else if (message.contains(QStringLiteral("Visualization"), Qt::CaseInsensitive))
            category = QStringLiteral("Visualization");
        else if (message.contains(QStringLiteral("Export"), Qt::CaseInsensitive)
            || message.contains(QStringLiteral("Screenshot"), Qt::CaseInsensitive))
            category = QStringLiteral("Export");
        else if (message.contains(QStringLiteral("shutdown"), Qt::CaseInsensitive))
            category = QStringLiteral("Shutdown");
        logger_->log(message.contains(QStringLiteral("FAILED"), Qt::CaseInsensitive)
                ? QStringLiteral("ERROR") : QStringLiteral("INFO"),
            category, message);
    }
    else
        logEdit_->appendPlainText(QStringLiteral("[%1] %2")
            .arg(QDateTime::currentDateTime().toString(
                QStringLiteral("yyyy-MM-dd HH:mm:ss.zzz")), message));
}

void MainWindow::setResult(QString const& key, QString const& value)
{
    auto const found = resultLabels_.find(key.toStdString());
    if (found != resultLabels_.end())
    {
        found->second->setText(value);
        found->second->setToolTip(value);
    }
}

void MainWindow::updateControls()
{
    bool const validCloud = QFileInfo(cloudPathEdit_->text()).isFile();
    browseButton_->setEnabled(stateMachine_.canSelectCloud());
    detectButton_->setEnabled(initialized_ && stateMachine_.canDetect(validCloud));
    exportResultButton_->setEnabled(hasResult_ && stateMachine_.canExport());
    exportScreenshotButton_->setEnabled(hasResult_ && stateMachine_.canExport());
    settingsButton_->setEnabled(stateMachine_.canOpenSettings());
    productInfoButton_->setEnabled(stateMachine_.state() != AppState::kShuttingDown);
    progress_->setVisible(stateMachine_.state() == AppState::kInitializing
        || stateMachine_.state() == AppState::kDetecting
        || stateMachine_.state() == AppState::kExporting);
}

void MainWindow::setState(AppState state)
{
    QString error;
    if (!stateMachine_.transition(state, error))
    {
        appendLog(error);
        return;
    }
    stateStatus_->setText(stateMachine_.stateName());
    statusBar()->showMessage(stateMachine_.stateName());
    updateControls();
}

void MainWindow::exportResult()
{
    if (!hasResult_ || !stateMachine_.canExport()) return;
    QString root = QFileDialog::getExistingDirectory(
        this, QStringLiteral("Select export directory"), appConfig_.defaultExportDirectory);
    if (root.isEmpty()) return;
    performExport(root, false);
}

void MainWindow::performExport(QString const& root, bool automated)
{
    setState(AppState::kExporting);
    DetectionExportIdentity identity;
    identity.applicationVersion = ProductInfo::applicationVersion();
    identity.sdkVersion = QStringLiteral("Phase 9D");
    identity.engineSha256 = appConfig_.engineSha256;
    identity.pluginSha256 = appConfig_.pluginSha256;
    QtWeldResultViewModel const resultSnapshot = lastResult_;
    QImage const screenshot = pointCloudView_->grabFramebuffer();
    auto* watcher = new QFutureWatcher<DetectionExportResult>(this);
    connect(watcher, &QFutureWatcher<DetectionExportResult>::finished,
        this, [this, watcher, automated] {
            DetectionExportResult const result = watcher->result();
            watcher->deleteLater();
            if (!result.success)
            {
                appendLog(QStringLiteral("EXPORT_FAILED: %1 (%2)")
                    .arg(result.error, result.failingFile));
                setState(AppState::kDetectionFailed);
                if (automated)
                {
                    smokePending_ = false;
                    emit productSmokeCompleted(false, {}, result.error);
                }
                else
                {
                    QMessageBox::critical(this, QStringLiteral("Export failed"),
                        QStringLiteral("EXPORT_FAILED\n%1\nFile: %2")
                            .arg(result.error, result.failingFile));
                }
                return;
            }
            appendLog(QStringLiteral("Export succeeded: %1").arg(result.directory));
            if (recentStore_)
            {
                QString historyError;
                recentStore_->updateExport(lastResult_.taskId, result.directory, historyError);
                refreshRecentTasks();
            }
            setState(AppState::kDetectionSucceeded);
            if (automated)
            {
                smokePending_ = false;
                emit productSmokeCompleted(true, result.directory, {});
            }
        });
    watcher->setFuture(QtConcurrent::run(
        [resultSnapshot, screenshot, root, identity] {
            return DetectionExportService::exportTask(
                resultSnapshot, screenshot, root, identity);
        }));
}

void MainWindow::exportScreenshot()
{
    if (!hasResult_ || !stateMachine_.canExport()) return;
    QString const suggested = QDir(appConfig_.defaultExportDirectory).filePath(
        QStringLiteral("%1_%2.png").arg(lastResult_.taskId,
            QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss"))));
    QString const path = QFileDialog::getSaveFileName(
        this, QStringLiteral("Export viewport PNG"), suggested, QStringLiteral("PNG (*.png)"));
    if (path.isEmpty()) return;
    ScreenshotExportResult const result =
        ScreenshotExportService::savePng(pointCloudView_->grabFramebuffer(), path);
    appendLog(result.success
        ? QStringLiteral("Screenshot exported: %1 (%2x%3)")
            .arg(result.path).arg(result.width).arg(result.height)
        : QStringLiteral("SCREENSHOT_EXPORT_FAILED: %1").arg(result.error));
}

void MainWindow::openSettings()
{
    SettingsDialog dialog(appConfig_, this);
    if (dialog.exec() != QDialog::Accepted || !dialog.validated()) return;
    AppConfig const updated = dialog.config();
    QString saveError;
    if (!updated.saveUser(userSettingsPath_, saveError))
    {
        QMessageBox::critical(this, QStringLiteral("Settings save failed"), saveError);
        return;
    }
    bool const runtimeChanged = updated.enginePath != appConfig_.enginePath
        || updated.pluginPath != appConfig_.pluginPath
        || updated.engineSha256 != appConfig_.engineSha256
        || updated.pluginSha256 != appConfig_.pluginSha256;
    appConfig_ = updated;
    showBboxCheck_->setChecked(appConfig_.showBoundingBox);
    showPcaCheck_->setChecked(appConfig_.showPcaDirection);
    pointSizeSpin_->setValue(appConfig_.pointSize);
    if (runtimeChanged)
    {
        appendLog(QStringLiteral("Validated runtime settings changed; controlled SDK reinitialization"));
        initialized_ = false;
        stopWorker();
        setState(AppState::kInitializing);
        startWorker();
    }
}

void MainWindow::showProductInfo()
{
    QMessageBox::information(this, QStringLiteral("Product Information"),
        QStringLiteral(
            "%1\nVersion: %2\nBuild: %3 (%4)\nBuilt: %5\nGit: %6\n"
            "Qt: %7\nCompiler: %8\nTensorRT: 11.1.0.106\nCUDA Toolkit: 12.8\n"
            "Engine: %9\nPlugin: %10\nOpenGL: %11\nVendor: %12\nRenderer: %13")
            .arg(ProductInfo::applicationName())
            .arg(ProductInfo::applicationVersion())
            .arg(ProductInfo::buildType())
            .arg(QStringLiteral("x64"))
            .arg(ProductInfo::buildTimestamp())
            .arg(ProductInfo::gitCommit())
            .arg(QString::fromLatin1(qVersion()))
            .arg(ProductInfo::compiler())
            .arg(appConfig_.engineSha256)
            .arg(appConfig_.pluginSha256)
            .arg(pointCloudView_->openGLVersion())
            .arg(pointCloudView_->openGLVendor())
            .arg(pointCloudView_->openGLRenderer()));
}

void MainWindow::refreshRecentTasks()
{
    if (!recentTasks_ || !recentStore_) return;
    recentTasks_->clear();
    QString error;
    for (RecentTask const& task : recentStore_->load(error))
    {
        auto* item = new QListWidgetItem(QStringLiteral("%1 | %2 | weld=%3 | %4")
            .arg(task.timestamp, task.taskId).arg(task.weldPoints)
            .arg(task.sourceMissing ? QStringLiteral("SOURCE MISSING") : task.sourceCloud),
            recentTasks_);
        item->setData(Qt::UserRole, task.sourceCloud);
        if (task.sourceMissing) item->setForeground(Qt::red);
    }
    if (!error.isEmpty()) appendLog(QStringLiteral("Recent task load failed: %1").arg(error));
}

void MainWindow::loadRecentTask()
{
    if (!recentTasks_->currentItem()) return;
    QString const source = recentTasks_->currentItem()->data(Qt::UserRole).toString();
    if (!QFileInfo(source).isFile())
    {
        appendLog(QStringLiteral("Recent source is missing: %1").arg(source));
        return;
    }
    cloudPathEdit_->setText(normalizedFilePath(source));
    setState(AppState::kCloudSelected);
}

void MainWindow::startProductSmoke(QString exportRoot)
{
    smokeExportRoot_ = QFileInfo(exportRoot).absoluteFilePath();
    smokePending_ = true;
    appendLog(QStringLiteral("Product deployment smoke armed: %1").arg(smokeExportRoot_));
    if (initialized_)
        QTimer::singleShot(0, this, &MainWindow::startDetection);
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    if (stateMachine_.state() == AppState::kExporting)
    {
        appendLog(QStringLiteral("Shutdown rejected while export is active"));
        event->ignore();
        return;
    }
    saveWindowLayout();
    event->accept();
}

bool MainWindow::geometryHasVisibleArea(QRect const& geometry) const
{
    constexpr int kRequiredVisibleWidth = 160;
    constexpr int kRequiredVisibleHeight = 100;
    for (QScreen* screen : QGuiApplication::screens())
    {
        QRect const visible = geometry.intersected(screen->availableGeometry());
        if (visible.width() >= kRequiredVisibleWidth
            && visible.height() >= kRequiredVisibleHeight)
            return true;
    }
    return false;
}

void MainWindow::applySafeDefaultGeometry()
{
    QScreen* screen = QGuiApplication::primaryScreen();
    QRect const available = screen != nullptr
        ? screen->availableGeometry()
        : QRect(0, 0, 1200, 760);
    QSize const target(
        std::max(520, std::min(1200, available.width() - 80)),
        std::max(420, std::min(760, available.height() - 80)));
    resize(target);
    move(available.center() - rect().center());
}

void MainWindow::restoreWindowLayout()
{
    if (!appConfig_.rememberWindowGeometry || userSettingsPath_.isEmpty())
    {
        applySafeDefaultGeometry();
        return;
    }
    QSettings settings(userSettingsPath_, QSettings::IniFormat);
    QByteArray const geometry = settings.value(QStringLiteral("Window/geometry")).toByteArray();
    bool const restored = !geometry.isEmpty() && restoreGeometry(geometry);
    if (!restored || !geometryHasVisibleArea(frameGeometry()))
        applySafeDefaultGeometry();
    QByteArray const splitter =
        settings.value(QStringLiteral("Window/main_splitter")).toByteArray();
    if (!splitter.isEmpty() && mainContentSplitter_ != nullptr)
        mainContentSplitter_->restoreState(splitter);
}

void MainWindow::saveWindowLayout()
{
    if (!appConfig_.rememberWindowGeometry || userSettingsPath_.isEmpty()) return;
    QSettings settings(userSettingsPath_, QSettings::IniFormat);
    settings.setValue(QStringLiteral("Window/geometry"), saveGeometry());
    if (mainContentSplitter_ != nullptr)
        settings.setValue(
            QStringLiteral("Window/main_splitter"), mainContentSplitter_->saveState());
    settings.sync();
}

} // namespace ptv2::qtui
