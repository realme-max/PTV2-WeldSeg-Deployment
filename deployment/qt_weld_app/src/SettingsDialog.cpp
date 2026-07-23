#include "SettingsDialog.h"

#include <QCheckBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>

namespace ptv2::qtui
{

SettingsDialog::SettingsDialog(AppConfig config, QWidget* parent)
    : QDialog(parent)
{
    setWindowTitle(QStringLiteral("PTV2 Weld Segmentation Settings"));
    resize(760, 460);
    auto* root = new QVBoxLayout(this);
    auto* form = new QFormLayout();
    auto pathRow = [&](QLineEdit*& edit, QString const& buttonText, auto slot) {
        auto* widget = new QWidget(this);
        auto* layout = new QHBoxLayout(widget);
        layout->setContentsMargins(0, 0, 0, 0);
        edit = new QLineEdit(widget);
        auto* button = new QPushButton(buttonText, widget);
        layout->addWidget(edit, 1);
        layout->addWidget(button);
        connect(button, &QPushButton::clicked, this, slot);
        return widget;
    };
    form->addRow(QStringLiteral("Engine"),
        pathRow(engine_, QStringLiteral("Browse Engine"), &SettingsDialog::browseEngine));
    form->addRow(QStringLiteral("Plugin"),
        pathRow(plugin_, QStringLiteral("Browse Plugin"), &SettingsDialog::browsePlugin));
    engineSha_ = new QLineEdit(this);
    pluginSha_ = new QLineEdit(this);
    cloudDirectory_ = new QLineEdit(this);
    exportDirectory_ = new QLineEdit(this);
    pointSize_ = new QDoubleSpinBox(this);
    pointSize_->setRange(1.0, 12.0);
    pointSize_->setSingleStep(0.5);
    showBbox_ = new QCheckBox(QStringLiteral("Show bounding box"), this);
    showPca_ = new QCheckBox(QStringLiteral("Show PCA direction"), this);
    autoInitialize_ = new QCheckBox(QStringLiteral("Initialize SDK automatically"), this);
    form->addRow(QStringLiteral("Engine SHA-256"), engineSha_);
    form->addRow(QStringLiteral("Plugin SHA-256"), pluginSha_);
    form->addRow(QStringLiteral("Default cloud directory"), cloudDirectory_);
    form->addRow(QStringLiteral("Default export directory"), exportDirectory_);
    form->addRow(QStringLiteral("Point size"), pointSize_);
    form->addRow(QStringLiteral("Visualization"), showBbox_);
    form->addRow(QString(), showPca_);
    form->addRow(QStringLiteral("Runtime"), autoInitialize_);
    root->addLayout(form);
    validation_ = new QLabel(QStringLiteral("NOT CHECKED"), this);
    validation_->setWordWrap(true);
    root->addWidget(validation_);
    auto* actions = new QHBoxLayout();
    auto* defaults = new QPushButton(QStringLiteral("Restore Defaults"), this);
    auto* validateButton = new QPushButton(QStringLiteral("Validate"), this);
    actions->addWidget(defaults);
    actions->addWidget(validateButton);
    actions->addStretch(1);
    auto* buttons = new QDialogButtonBox(
        QDialogButtonBox::Save | QDialogButtonBox::Cancel, this);
    actions->addWidget(buttons);
    root->addLayout(actions);
    connect(defaults, &QPushButton::clicked, this, &SettingsDialog::restoreDefaults);
    connect(validateButton, &QPushButton::clicked, this, &SettingsDialog::validate);
    connect(buttons, &QDialogButtonBox::accepted, this, &SettingsDialog::acceptValidated);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    populate(config);
}

void SettingsDialog::populate(AppConfig const& config)
{
    engine_->setText(config.enginePath);
    plugin_->setText(config.pluginPath);
    engineSha_->setText(config.engineSha256);
    pluginSha_->setText(config.pluginSha256);
    cloudDirectory_->setText(config.defaultCloudDirectory);
    exportDirectory_->setText(config.defaultExportDirectory);
    pointSize_->setValue(config.pointSize);
    showBbox_->setChecked(config.showBoundingBox);
    showPca_->setChecked(config.showPcaDirection);
    autoInitialize_->setChecked(config.autoInitialize);
    validated_ = false;
    validation_->setText(QStringLiteral("NOT CHECKED"));
}

AppConfig SettingsDialog::collect() const
{
    AppConfig result = AppConfig::defaults();
    result.enginePath = engine_->text().trimmed();
    result.pluginPath = plugin_->text().trimmed();
    result.engineSha256 = engineSha_->text().trimmed().toLower();
    result.pluginSha256 = pluginSha_->text().trimmed().toLower();
    result.defaultCloudDirectory = cloudDirectory_->text().trimmed();
    result.defaultExportDirectory = exportDirectory_->text().trimmed();
    result.pointSize = pointSize_->value();
    result.showBoundingBox = showBbox_->isChecked();
    result.showPcaDirection = showPca_->isChecked();
    result.autoInitialize = autoInitialize_->isChecked();
    return result;
}

AppConfig SettingsDialog::config() const { return collect(); }
bool SettingsDialog::validated() const noexcept { return validated_; }

void SettingsDialog::browseEngine()
{
    QString const path = QFileDialog::getOpenFileName(
        this, QStringLiteral("Select TensorRT Engine"), {}, QStringLiteral("TensorRT Engine (*.plan *.engine)"));
    if (!path.isEmpty()) { engine_->setText(path); validated_ = false; }
}

void SettingsDialog::browsePlugin()
{
    QString const path = QFileDialog::getOpenFileName(
        this, QStringLiteral("Select VoxelUnique Plugin"), {}, QStringLiteral("Plugin DLL (*.dll)"));
    if (!path.isEmpty()) { plugin_->setText(path); validated_ = false; }
}

void SettingsDialog::restoreDefaults() { populate(AppConfig::defaults()); }

void SettingsDialog::validate()
{
    AppConfigValidation const result = collect().validateRuntime();
    validated_ = result.valid;
    validation_->setText(result.valid
        ? QStringLiteral("PASS - Engine and Plugin paths/hashes satisfy the runtime contract")
        : QStringLiteral("FAILED - %1").arg(result.error));
}

void SettingsDialog::acceptValidated()
{
    validate();
    if (validated_) accept();
}

} // namespace ptv2::qtui
