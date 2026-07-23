#include "PointCloudView.h"

#include <QElapsedTimer>
#include <QLabel>
#include <QMouseEvent>
#include <QOpenGLContext>
#include <QSizePolicy>
#include <QSurfaceFormat>
#include <QVBoxLayout>
#include <QWheelEvent>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace ptv2::qtui
{
namespace
{

char const* kVertexShader = R"(
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
uniform mat4 mvp;
uniform float pointSize;
out vec3 vertexColor;
void main()
{
    gl_Position = mvp * vec4(position, 1.0);
    gl_PointSize = pointSize;
    vertexColor = color;
}
)";

char const* kFragmentShader = R"(
#version 330 core
in vec3 vertexColor;
out vec4 fragmentColor;
void main()
{
    fragmentColor = vec4(vertexColor, 1.0);
}
)";

PointCloudView::Vertex vertex(QVector3D const& position, QVector3D const& color)
{
    return {
        position.x(), position.y(), position.z(),
        color.x(), color.y(), color.z()};
}

} // namespace

PointCloudView::PointCloudView(QWidget* parent)
    : QOpenGLWidget(parent)
{
    setObjectName(QStringLiteral("pointCloudView"));
    setMinimumSize(400, 300);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setFocusPolicy(Qt::StrongFocus);
    emptyStateLabel_ = new QLabel(
        QStringLiteral("Select a point cloud and run detection"), this);
    emptyStateLabel_->setObjectName(QStringLiteral("pointCloudEmptyState"));
    emptyStateLabel_->setAlignment(Qt::AlignCenter);
    emptyStateLabel_->setAttribute(Qt::WA_TransparentForMouseEvents);
    emptyStateLabel_->setStyleSheet(QStringLiteral(
        "QLabel { color: rgb(205, 215, 225); background: transparent; }"));
    auto* overlayLayout = new QVBoxLayout(this);
    overlayLayout->setContentsMargins(0, 0, 0, 0);
    overlayLayout->addWidget(emptyStateLabel_);
}

PointCloudView::~PointCloudView()
{
    if (context() != nullptr)
    {
        makeCurrent();
        releaseGlResources();
        doneCurrent();
    }
}

bool PointCloudView::setPointCloud(PointCloudRenderData const& renderData, QString& error)
{
    QString detail;
    if (!renderData.valid || !renderData.validate(detail))
    {
        error = detail.isEmpty() ? renderData.error : detail;
        emit visualizationLog(QStringLiteral("VISUALIZATION_DATA_FAILED: %1").arg(error));
        return false;
    }
    data_ = renderData;
    rebuildCpuVertices();

    QVector3D minimum(
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity());
    QVector3D maximum(
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity());
    for (RenderPoint const& point : data_.points)
    {
        minimum.setX(std::min(minimum.x(), point.position.x()));
        minimum.setY(std::min(minimum.y(), point.position.y()));
        minimum.setZ(std::min(minimum.z(), point.position.z()));
        maximum.setX(std::max(maximum.x(), point.position.x()));
        maximum.setY(std::max(maximum.y(), point.position.y()));
        maximum.setZ(std::max(maximum.z(), point.position.z()));
    }
    camera_.fitToBounds(minimum, maximum);
    buffersDirty_ = true;
    emptyStateLabel_->hide();
    error.clear();
    update();
    return true;
}

void PointCloudView::clearPointCloud()
{
    data_ = {};
    pointVertices_.clear();
    lineVertices_.clear();
    centerVertices_.clear();
    weldPointCount_ = 0;
    backgroundPointCount_ = 0;
    buffersDirty_ = true;
    emptyStateLabel_->show();
    update();
}

void PointCloudView::resetView()
{
    camera_.resetView();
    update();
}

void PointCloudView::setShowBoundingBox(bool enabled)
{
    showBoundingBox_ = enabled;
    rebuildCpuVertices();
    buffersDirty_ = true;
    update();
}

void PointCloudView::setShowPcaDirection(bool enabled)
{
    showPcaDirection_ = enabled;
    rebuildCpuVertices();
    buffersDirty_ = true;
    update();
}

void PointCloudView::setPointSize(float size)
{
    pointSize_ = std::max(1.0F, std::min(12.0F, size));
    update();
}

void PointCloudView::setForceShaderFailureForTest(bool enabled)
{
    forceShaderFailure_ = enabled;
}

bool PointCloudView::openGLInitialized() const noexcept { return initialized_; }
bool PointCloudView::shaderLinked() const noexcept { return shaderLinked_; }
int PointCloudView::renderedPointCount() const noexcept { return pointVertices_.size(); }
int PointCloudView::weldPointCount() const noexcept { return weldPointCount_; }
int PointCloudView::backgroundPointCount() const noexcept { return backgroundPointCount_; }
unsigned int PointCloudView::lastGlError() const noexcept { return lastGlError_; }
double PointCloudView::lastUploadMs() const noexcept { return lastUploadMs_; }
double PointCloudView::lastPaintMs() const noexcept { return lastPaintMs_; }
quint64 PointCloudView::bufferUploadCount() const noexcept { return bufferUploadCount_; }
quint64 PointCloudView::resizeGlCount() const noexcept { return resizeGlCount_; }
float PointCloudView::aspectRatio() const noexcept { return aspectRatio_; }
QString PointCloudView::openGLVersion() const { return glVersion_; }
QString PointCloudView::openGLRenderer() const { return glRenderer_; }
QString PointCloudView::openGLVendor() const { return glVendor_; }
QString PointCloudView::visualizationError() const { return error_; }
PointCloudCamera const& PointCloudView::camera() const noexcept { return camera_; }

void PointCloudView::initializeGL()
{
    initializeOpenGLFunctions();
    glVersion_ = glString(GL_VERSION);
    glRenderer_ = glString(GL_RENDERER);
    glVendor_ = glString(GL_VENDOR);
    QSurfaceFormat const actual = context()->format();
    if (actual.majorVersion() < 3
        || (actual.majorVersion() == 3 && actual.minorVersion() < 3))
    {
        error_ = QStringLiteral("OpenGL 3.3 core is required; actual context is %1.%2")
            .arg(actual.majorVersion()).arg(actual.minorVersion());
        emit openGLStatusChanged(false, error_);
        return;
    }

    QString shaderError;
    if (!buildShader(shaderError))
    {
        error_ = shaderError;
        emit openGLStatusChanged(false, error_);
        return;
    }
    if (!vao_.create() || !pointsBuffer_.create()
        || !linesBuffer_.create() || !centerBuffer_.create())
    {
        error_ = QStringLiteral("Failed to create OpenGL buffers/VAO");
        emit openGLStatusChanged(false, error_);
        return;
    }
    initialized_ = true;
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glClearColor(0.035F, 0.045F, 0.06F, 1.0F);
    emit visualizationLog(QStringLiteral("OpenGL %1 | %2 | %3 | shader linked")
        .arg(glVersion_, glRenderer_, glVendor_));
    emit openGLStatusChanged(true, QStringLiteral("OpenGL visualization ready"));
}

void PointCloudView::resizeGL(int width, int height)
{
    int const safeWidth = std::max(width, 1);
    int const safeHeight = std::max(height, 1);
    glViewport(0, 0, safeWidth, safeHeight);
    aspectRatio_ = static_cast<float>(safeWidth) / static_cast<float>(safeHeight);
    ++resizeGlCount_;
}

void PointCloudView::paintGL()
{
    QElapsedTimer timer;
    timer.start();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (!initialized_ || !shaderLinked_)
    {
        lastPaintMs_ = static_cast<double>(timer.nsecsElapsed()) / 1.0e6;
        return;
    }
    if (buffersDirty_) uploadPendingBuffers();
    QMatrix4x4 const mvp =
        camera_.projectionMatrix(aspectRatio_) * camera_.viewMatrix();
    program_.bind();
    program_.setUniformValue("mvp", mvp);
    drawBuffer(pointsBuffer_, pointVertices_.size(), GL_POINTS, pointSize_);
    drawBuffer(linesBuffer_, lineVertices_.size(), GL_LINES, 1.0F);
    drawBuffer(centerBuffer_, centerVertices_.size(), GL_POINTS, pointSize_ * 2.5F);
    program_.release();
    lastGlError_ = glGetError();
    lastPaintMs_ = static_cast<double>(timer.nsecsElapsed()) / 1.0e6;

}

void PointCloudView::mousePressEvent(QMouseEvent* event)
{
    lastMousePosition_ = event->pos();
    event->accept();
}

void PointCloudView::mouseMoveEvent(QMouseEvent* event)
{
    QPoint const delta = event->pos() - lastMousePosition_;
    lastMousePosition_ = event->pos();
    if (event->buttons() & Qt::LeftButton)
        camera_.rotate(delta.x() * 0.4F, delta.y() * 0.4F);
    else if (event->buttons() & (Qt::RightButton | Qt::MiddleButton))
        camera_.pan(static_cast<float>(delta.x()), static_cast<float>(delta.y()));
    update();
    event->accept();
}

void PointCloudView::wheelEvent(QWheelEvent* event)
{
    camera_.zoom(static_cast<float>(event->angleDelta().y()) / 120.0F);
    update();
    event->accept();
}

void PointCloudView::mouseDoubleClickEvent(QMouseEvent* event)
{
    resetView();
    event->accept();
}

void PointCloudView::releaseGlResources()
{
    pointsBuffer_.destroy();
    linesBuffer_.destroy();
    centerBuffer_.destroy();
    vao_.destroy();
    program_.removeAllShaders();
    initialized_ = false;
    shaderLinked_ = false;
}

bool PointCloudView::buildShader(QString& error)
{
    char const* vertexSource = forceShaderFailure_
        ? "#version 330 core\nthis is intentionally invalid"
        : kVertexShader;
    if (!program_.addShaderFromSourceCode(QOpenGLShader::Vertex, vertexSource))
    {
        error = QStringLiteral("Point shader vertex compilation failed: %1").arg(program_.log());
        return false;
    }
    if (!program_.addShaderFromSourceCode(QOpenGLShader::Fragment, kFragmentShader))
    {
        error = QStringLiteral("Point shader fragment compilation failed: %1").arg(program_.log());
        return false;
    }
    if (!program_.link())
    {
        error = QStringLiteral("Point shader link failed: %1").arg(program_.log());
        return false;
    }
    shaderLinked_ = true;
    return true;
}

void PointCloudView::rebuildCpuVertices()
{
    pointVertices_.clear();
    lineVertices_.clear();
    centerVertices_.clear();
    weldPointCount_ = 0;
    backgroundPointCount_ = 0;
    for (RenderPoint const& point : data_.points)
    {
        pointVertices_.append(vertex(point.position, point.color));
        if (point.label == 0) ++weldPointCount_;
        else ++backgroundPointCount_;
    }
    QVector3D const overlayColor(1.0F, 0.9F, 0.2F);
    if (showBoundingBox_ && data_.valid)
    {
        QVector3D const lo = data_.bboxMin;
        QVector3D const hi = data_.bboxMax;
        std::array<QVector3D, 8> const corners{{
            {lo.x(), lo.y(), lo.z()}, {hi.x(), lo.y(), lo.z()},
            {hi.x(), hi.y(), lo.z()}, {lo.x(), hi.y(), lo.z()},
            {lo.x(), lo.y(), hi.z()}, {hi.x(), lo.y(), hi.z()},
            {hi.x(), hi.y(), hi.z()}, {lo.x(), hi.y(), hi.z()},
        }};
        std::array<std::array<int, 2>, 12> const edges{{
            {{0,1}}, {{1,2}}, {{2,3}}, {{3,0}},
            {{4,5}}, {{5,6}}, {{6,7}}, {{7,4}},
            {{0,4}}, {{1,5}}, {{2,6}}, {{3,7}},
        }};
        for (auto const& edge : edges)
        {
            lineVertices_.append(vertex(corners[edge[0]], overlayColor));
            lineVertices_.append(vertex(corners[edge[1]], overlayColor));
        }
    }
    if (showPcaDirection_ && data_.valid)
    {
        QVector3D const pcaColor(0.2F, 1.0F, 0.35F);
        lineVertices_.append(vertex(data_.pcaStart, pcaColor));
        lineVertices_.append(vertex(data_.pcaEnd, pcaColor));
    }
    if (data_.valid)
        centerVertices_.append(vertex(data_.weldCenter, QVector3D(1.0F, 1.0F, 1.0F)));
}

void PointCloudView::uploadPendingBuffers()
{
    QElapsedTimer timer;
    timer.start();
    auto upload = [](QOpenGLBuffer& buffer, QVector<Vertex> const& vertices) {
        buffer.bind();
        buffer.setUsagePattern(QOpenGLBuffer::DynamicDraw);
        buffer.allocate(vertices.constData(), vertices.size() * static_cast<int>(sizeof(Vertex)));
        buffer.release();
    };
    upload(pointsBuffer_, pointVertices_);
    upload(linesBuffer_, lineVertices_);
    upload(centerBuffer_, centerVertices_);
    buffersDirty_ = false;
    ++bufferUploadCount_;
    lastUploadMs_ = static_cast<double>(timer.nsecsElapsed()) / 1.0e6;
}

void PointCloudView::drawBuffer(
    QOpenGLBuffer& buffer, int vertices, unsigned int mode, float pointSize)
{
    if (vertices <= 0) return;
    QOpenGLVertexArrayObject::Binder binder(&vao_);
    buffer.bind();
    program_.enableAttributeArray(0);
    program_.enableAttributeArray(1);
    program_.setAttributeBuffer(0, GL_FLOAT, 0, 3, sizeof(Vertex));
    program_.setAttributeBuffer(1, GL_FLOAT, 3 * sizeof(float), 3, sizeof(Vertex));
    program_.setUniformValue("pointSize", pointSize);
    glDrawArrays(mode, 0, vertices);
    program_.disableAttributeArray(0);
    program_.disableAttributeArray(1);
    buffer.release();
}

QString PointCloudView::glString(unsigned int name)
{
    auto const* value = glGetString(name);
    return value == nullptr ? QStringLiteral("UNAVAILABLE")
                            : QString::fromLatin1(reinterpret_cast<char const*>(value));
}

} // namespace ptv2::qtui
