#pragma once

#include "PointCloudCamera.h"
#include "PointCloudRenderData.h"

#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QPoint>
#include <QString>

namespace ptv2::qtui
{

class PointCloudView final : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    struct Vertex
    {
        float x;
        float y;
        float z;
        float r;
        float g;
        float b;
    };

    explicit PointCloudView(QWidget* parent = nullptr);
    ~PointCloudView() override;

    bool setPointCloud(PointCloudRenderData const& renderData, QString& error);
    void clearPointCloud();
    void resetView();
    void setShowBoundingBox(bool enabled);
    void setShowPcaDirection(bool enabled);
    void setPointSize(float size);
    void setForceShaderFailureForTest(bool enabled);

    bool openGLInitialized() const noexcept;
    bool shaderLinked() const noexcept;
    int renderedPointCount() const noexcept;
    int weldPointCount() const noexcept;
    int backgroundPointCount() const noexcept;
    unsigned int lastGlError() const noexcept;
    double lastUploadMs() const noexcept;
    double lastPaintMs() const noexcept;
    QString openGLVersion() const;
    QString openGLRenderer() const;
    QString openGLVendor() const;
    QString visualizationError() const;
    PointCloudCamera const& camera() const noexcept;

signals:
    void openGLStatusChanged(bool ready, QString message);
    void visualizationLog(QString message);

protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
    void releaseGlResources();
    bool buildShader(QString& error);
    void rebuildCpuVertices();
    void uploadPendingBuffers();
    void drawBuffer(QOpenGLBuffer& buffer, int vertices, unsigned int mode, float pointSize);
    QString glString(unsigned int name);

    PointCloudRenderData data_;
    PointCloudCamera camera_;
    QOpenGLShaderProgram program_;
    QOpenGLVertexArrayObject vao_;
    QOpenGLBuffer pointsBuffer_{QOpenGLBuffer::VertexBuffer};
    QOpenGLBuffer linesBuffer_{QOpenGLBuffer::VertexBuffer};
    QOpenGLBuffer centerBuffer_{QOpenGLBuffer::VertexBuffer};
    QVector<Vertex> pointVertices_;
    QVector<Vertex> lineVertices_;
    QVector<Vertex> centerVertices_;
    QPoint lastMousePosition_;
    QString glVersion_;
    QString glRenderer_;
    QString glVendor_;
    QString error_;
    bool initialized_{false};
    bool shaderLinked_{false};
    bool buffersDirty_{false};
    bool showBoundingBox_{true};
    bool showPcaDirection_{true};
    bool forceShaderFailure_{false};
    float pointSize_{3.0F};
    int weldPointCount_{0};
    int backgroundPointCount_{0};
    unsigned int lastGlError_{0};
    double lastUploadMs_{0.0};
    double lastPaintMs_{0.0};
};

} // namespace ptv2::qtui
