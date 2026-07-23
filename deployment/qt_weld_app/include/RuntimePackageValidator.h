#pragma once

#include <QString>
#include <QStringList>

namespace ptv2::qtui
{

struct RuntimePackageValidation
{
    bool valid{false};
    QStringList missing;
    QStringList forbidden;
    QStringList absoluteSourceReferences;
    QString error;
};

class RuntimePackageValidator
{
public:
    static RuntimePackageValidation validate(QString const& packageRoot);
};

} // namespace ptv2::qtui
