# TensorRT Phase 10B: Qt Point-Cloud Visualization Smoke

## Outcome

```text
PHASE_10B_QT_POINTCLOUD_VISUALIZATION_COMPLETED
```

Evidence: `artifacts/gcn_res_tensorrt/20260723_161213_760000_phase10b_qt_pointcloud_visualization/`.

Phase 10B adds visualization only. The checkpoint, ONNX, production Engine,
VoxelUniqueCub Plugin, TensorRT Runtime, sampling, adjacency, task mathematics,
labels and numerical thresholds are unchanged.

## Decision and architecture

The renderer uses Qt 5.9.1 `QOpenGLWidget`, `QOpenGLFunctions`, a compact
OpenGL 3.3 shader, VAO and vertex buffers. No PCL, VTK, QVTK, scene graph or
plotting dependency was added.

```text
WeldDetector worker
  -> public WeldResult value
  -> QtWeldResultViewModel copy
  -> PointCloudRenderData copy (GUI thread)
  -> PointCloudView buffers (GUI thread)
```

The source audit checked 16 Qt files and found no TensorRT/CUDA/internal
pipeline/PCL/VTK include. OpenGL updates occur only on the GUI thread.

## Public SDK extension

The public result lacked already-computed display data. The minimal,
backward-compatible extension adds:

- `WeldPointResult {x,y,z,label,confidence}` and `WeldResult.points`;
- `WeldResult.principal_direction`;
- `WeldGeometryResult.principalDirection`.

`WeldDetector` copies existing recovered segmentation values.
`WeldGeometryExtractor` stores the normalized eigenvector it already uses for
PCA length. Qt neither parses generated files nor recomputes PCA.

For weld_65, 2048 points are exposed and the direction is
`[-0.989224315,-0.0287930481,0.143548802]`. Phase 9D regression passed:
2048/2048 labels, unchanged task values, maximum geometry error
`4.2578125203363015e-7`, ErrorRecorder zero and 4/4 original failures.

## Rendering contract and interaction

Render data owns copied values and validates finite positions/colors/geometry,
labels `{0,1}`, confidence `[0,1]`, vector size and PCA line. Colors are:

- class 0 / weld seam: `(1.0,0.25,0.08)`;
- class 1 / background: `(0.18,0.55,0.95)`.

Confidence does not modify class color. Exactly 2048 points render: 209 weld
and 1839 background. Camera fitting uses all sampled original XYZ without
altering SDK coordinates.

Overlays use the SDK AABB, weld centroid and
`center ± principal_direction × length_mm/2`. Controls are left-drag rotate,
right/middle-drag pan, wheel zoom, reset/double-click, bbox/PCA toggles and
clamped point size.

## OpenGL and runtime

- Requested/actual: OpenGL 3.3 core / `3.3.0 NVIDIA 610.74`.
- Vendor/renderer: NVIDIA Corporation / RTX 5060.
- Shader: PASS; controlled `glGetError`: 0.

CMake links `Qt5::Widgets`, `Qt5::OpenGL` and `ptv2_weld_sdk`.
QOpenGLWidget/Functions are provided by Qt5Widgets/Gui; the linker retained no
runtime need for `Qt5OpenGL.dll`, so windeployqt did not copy it. Qt
Core/Gui/Widgets, qwindows, TensorRT and CUDA DLLs are present. VS2022 x64
Release C++17 `/W4 /WX` and windeployqt passed.

## Validation

CTest: 3/3 PASS: Phase 10A integration regression, render-data test and real
OpenGL widget smoke.

Render-data checks passed for 2048/209/1839 points, finite data, confidence,
Phase 9D geometry, actual PCA endpoints and no source aliasing. Widget checks
passed for context/shader/upload/paint, rotate/zoom/pan/reset, bbox/PCA,
second upload, clear, GL error zero and clean destruction. MainWindow remained
responsive, detected twice, replaced rather than duplicated points, and shut
down worker/GL resources cleanly.

Visualization fail-closed: 6/6 PASS:

1. success then missing cloud—SDK error and previous view preserved;
2. success then 2047 points—SDK error and previous view preserved;
3. NaN coordinate rejected;
4. mismatched point count rejected;
5. invalid PCA rejected;
6. simulated shader failure rejected.

No fake visualization, partial new upload or fallback was produced.

## UI timing smoke

This is not an inference benchmark. Approximate recorded values:

- conversion `0.068 ms`;
- GPU upload `0.021 ms`;
- first paint `0.28 ms`;
- repaint `0.11 ms`.

Refresh values are in `visualization_timing.json`; the GUI remained responsive.

## Stop point

No PCL/VTK, robot/trajectory, database/login, FP16 or INT8 work was performed.
`CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED` remains explicit. Stop after
Phase 10B.
