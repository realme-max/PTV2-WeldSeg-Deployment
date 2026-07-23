#include "ProductInfo.h"

namespace ptv2::qtui
{

QString ProductInfo::applicationName() { return QStringLiteral("PTV2 Weld Segmentation"); }
QString ProductInfo::applicationVersion() { return QStringLiteral("0.1.1"); }

QString ProductInfo::buildType()
{
#ifdef NDEBUG
    return QStringLiteral("Release");
#else
    return QStringLiteral("Debug");
#endif
}

QString ProductInfo::buildTimestamp()
{
    return QStringLiteral(__DATE__ " " __TIME__);
}

QString ProductInfo::gitCommit()
{
#ifdef PTV2_GIT_COMMIT
    return QStringLiteral(PTV2_GIT_COMMIT);
#else
    return QStringLiteral("unknown");
#endif
}

QString ProductInfo::compiler()
{
#ifdef _MSC_VER
    return QStringLiteral("MSVC %1").arg(_MSC_VER);
#else
    return QStringLiteral("unknown");
#endif
}

} // namespace ptv2::qtui
