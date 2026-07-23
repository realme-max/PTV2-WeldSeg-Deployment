#pragma once

#include "AppConfig.h"

#include <QDialog>

class QCheckBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;

namespace ptv2::qtui
{

class SettingsDialog final : public QDialog
{
    Q_OBJECT

public:
    explicit SettingsDialog(AppConfig config, QWidget* parent = nullptr);
    AppConfig config() const;
    bool validated() const noexcept;

private slots:
    void browseEngine();
    void browsePlugin();
    void restoreDefaults();
    void validate();
    void acceptValidated();

private:
    void populate(AppConfig const& config);
    AppConfig collect() const;

    QLineEdit* engine_{nullptr};
    QLineEdit* plugin_{nullptr};
    QLineEdit* engineSha_{nullptr};
    QLineEdit* pluginSha_{nullptr};
    QLineEdit* cloudDirectory_{nullptr};
    QLineEdit* exportDirectory_{nullptr};
    QDoubleSpinBox* pointSize_{nullptr};
    QCheckBox* showBbox_{nullptr};
    QCheckBox* showPca_{nullptr};
    QCheckBox* autoInitialize_{nullptr};
    QLabel* validation_{nullptr};
    bool validated_{false};
};

} // namespace ptv2::qtui
