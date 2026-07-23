#pragma once

#include "QtWeldResultViewModel.h"
#include "WeldConfig.h"

#include <QMainWindow>
#include <QString>
#include <QThread>

#include <map>
#include <string>

class QLabel;
class QLineEdit;
class QPushButton;
class QCheckBox;
class QDoubleSpinBox;
class QTextEdit;

namespace ptv2::qtui
{

class WeldDetectionWorker;
class PointCloudView;

class MainWindow final : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(
        ptv2::weld::WeldConfig config,
        QString expectedEngineSha256,
        QString initialCloudPath,
        QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    void browseCloud();
    void startDetection();
    void onInitializationFinished(QString status, QString message);
    void onDetectionStarted(QString cloudPath);
    void onDetectionSucceeded(ptv2::qtui::QtWeldResultViewModel result);
    void onDetectionFailed(QString status, QString message);
    void appendLog(QString message);
    void resetVisualization();

private:
    void buildUi();
    void updateControls();
    void setResult(QString const& key, QString const& value);
    QString normalizedFilePath(QString const& path) const;

    QThread workerThread_;
    WeldDetectionWorker* worker_{nullptr};
    QLineEdit* cloudPathEdit_{nullptr};
    QPushButton* browseButton_{nullptr};
    QPushButton* detectButton_{nullptr};
    QLabel* initializeStatus_{nullptr};
    QTextEdit* logEdit_{nullptr};
    PointCloudView* pointCloudView_{nullptr};
    QPushButton* resetViewButton_{nullptr};
    QCheckBox* showBboxCheck_{nullptr};
    QCheckBox* showPcaCheck_{nullptr};
    QDoubleSpinBox* pointSizeSpin_{nullptr};
    std::map<std::string, QLabel*> resultLabels_;
    bool initialized_{false};
    bool detectionActive_{false};
};

} // namespace ptv2::qtui
