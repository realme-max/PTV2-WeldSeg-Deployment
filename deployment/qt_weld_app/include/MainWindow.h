#pragma once

#include "AppConfig.h"
#include "AppStateMachine.h"
#include "QtWeldResultViewModel.h"
#include "WeldConfig.h"

#include <QMainWindow>
#include <QString>
#include <QThread>

#include <map>
#include <memory>
#include <string>

class QLabel;
class QLineEdit;
class QPushButton;
class QCheckBox;
class QDoubleSpinBox;
class QPlainTextEdit;
class QProgressBar;
class QListWidget;
class QCloseEvent;
class QScrollArea;
class QSplitter;
class QRect;

namespace ptv2::qtui
{

class WeldDetectionWorker;
class PointCloudView;
class ApplicationLogger;
class RecentTaskStore;
class QtCloudBrowseDirectoryTest;

class MainWindow final : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(
        ptv2::weld::WeldConfig config,
        QString expectedEngineSha256,
        QString initialCloudPath,
        QWidget* parent = nullptr);
    MainWindow(
        AppConfig config,
        QString initialCloudPath,
        QString userSettingsPath,
        QWidget* parent = nullptr);
    ~MainWindow() override;
    void startProductSmoke(QString exportRoot);

signals:
    void productSmokeCompleted(bool success, QString exportDirectory, QString error);

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    void browseCloud();
    void startDetection();
    void onInitializationFinished(QString status, QString message);
    void onDetectionStarted(QString cloudPath);
    void onDetectionSucceeded(ptv2::qtui::QtWeldResultViewModel result);
    void onDetectionFailed(QString status, QString message);
    void appendLog(QString message);
    void resetVisualization();
    void exportResult();
    void exportScreenshot();
    void openSettings();
    void showProductInfo();
    void loadRecentTask();

private:
    void buildUi();
    void startWorker();
    void stopWorker();
    void setState(AppState state);
    void refreshRecentTasks();
    void performExport(QString const& root, bool automated);
    void updateControls();
    void setResult(QString const& key, QString const& value);
    QString normalizedFilePath(QString const& path) const;
    QString resolveInitialCloudDirectory() const;
    void applyCloudSelection(QString const& selectedFile);
    void persistLastCloudDirectory(QString const& selectedFile);
    void restoreWindowLayout();
    void saveWindowLayout();
    void applySafeDefaultGeometry();
    bool geometryHasVisibleArea(QRect const& geometry) const;

    friend class QtCloudBrowseDirectoryTest;

    QThread workerThread_;
    WeldDetectionWorker* worker_{nullptr};
    QLineEdit* cloudPathEdit_{nullptr};
    QPushButton* browseButton_{nullptr};
    QPushButton* detectButton_{nullptr};
    QLabel* initializeStatus_{nullptr};
    QPlainTextEdit* logEdit_{nullptr};
    PointCloudView* pointCloudView_{nullptr};
    QScrollArea* rightScrollArea_{nullptr};
    QSplitter* mainContentSplitter_{nullptr};
    QPushButton* resetViewButton_{nullptr};
    QPushButton* exportResultButton_{nullptr};
    QPushButton* exportScreenshotButton_{nullptr};
    QPushButton* settingsButton_{nullptr};
    QPushButton* productInfoButton_{nullptr};
    QCheckBox* showBboxCheck_{nullptr};
    QCheckBox* showPcaCheck_{nullptr};
    QDoubleSpinBox* pointSizeSpin_{nullptr};
    QProgressBar* progress_{nullptr};
    QLabel* stateStatus_{nullptr};
    QListWidget* recentTasks_{nullptr};
    std::map<std::string, QLabel*> resultLabels_;
    AppConfig appConfig_;
    QString userSettingsPath_;
    AppStateMachine stateMachine_;
    QtWeldResultViewModel lastResult_;
    bool hasResult_{false};
    QString smokeExportRoot_;
    bool smokePending_{false};
    std::unique_ptr<ApplicationLogger> logger_;
    std::unique_ptr<RecentTaskStore> recentStore_;
    bool initialized_{false};
    bool detectionActive_{false};
};

} // namespace ptv2::qtui
