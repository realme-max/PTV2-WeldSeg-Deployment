# Qt WeldDetector SDK integration smoke

This Phase 10A Qt Widgets application depends only on the public `ptv2_weld_sdk` API. A dedicated `WeldDetectionWorker` owns one initialized `WeldDetector` in a `QThread`; the GUI never includes or accesses TensorRT, CUDA, point-cloud preprocessing, or post-processing implementation headers.

The UI is intentionally limited to file selection, status/log presentation, task fields, geometry, and timing. It contains no PCL, VTK, OpenGL point-cloud visualization, robot interface, database, login, or Python binding.

Runtime paths are command-line only:

```text
--engine <production .plan>
--plugin <VoxelUniqueCubPlugin.dll>
--engine-sha256 <expected SHA-256>
--cloud <optional initial TXT>
```

The detector is initialized once and reused. Concurrent requests on the same worker are rejected; the SDK does not claim concurrent inference support.

## Build

Use a Visual Studio x64 generator and point CMake at the selected Qt and
TensorRT SDK installations:

```powershell
cmake -S deployment/qt_weld_app -B <build-dir> `
  -G "Visual Studio 17 2022" -A x64 `
  -DTENSORRT_ROOT=<TensorRT-SDK-root> `
  -DCUDAToolkit_ROOT="<CUDA-12.8-root>" `
  -DCMAKE_PREFIX_PATH=<Qt-msvc-x64-root>

cmake --build <build-dir> --config Release --parallel 4
```

Deploy Qt runtime files after the Release build:

```powershell
<Qt-root>\bin\windeployqt.exe --release --no-translations `
  <build-dir>\Release\ptv2_weld_qt_smoke.exe
```

The tested local configuration uses Qt 5.9.1 `msvc2015_64`, VS2022 v143
x64, CUDA Toolkit 12.8.93, and TensorRT 11.1.0.106. CMake also accepts a
compatible Qt 6 Widgets/Test installation.

## Run

```powershell
<build-dir>\Release\ptv2_weld_qt_smoke.exe `
  --engine <production-package>\engine\strict_fp32_voxelunique_cub.plan `
  --plugin <production-package>\plugins\VoxelUniqueCubPlugin.dll `
  --engine-sha256 <expected-engine-sha256> `
  --cloud <optional-initial-weld-txt>
```

`--engine`, `--plugin`, and `--engine-sha256` are required. The application
does not fall back to another artifact when validation or SDK initialization
fails.

## Threading and scope

Initialization and detection execute in one `WeldDetectionWorker` moved to a
dedicated `QThread`. The worker owns and reuses one `WeldDetector`; a second
request while detection is active is rejected. Closing the window performs a
blocking worker shutdown followed by `QThread::quit()`/`wait()`.

Phase 10A has no point-cloud visualization. It does not add PCL, VTK,
OpenGL, robot interfaces, FP16, INT8, or concurrent SDK inference.

## Phase 10B visualization

The application now renders the 2048 public SDK points with
`QOpenGLWidget`/`QOpenGLFunctions`; it never reads prediction/PLY/logit files
or TensorRT/CUDA buffers. Class 0 `weld_seam` is orange-red and class 1
`background` is blue. Confidence is retained and validated but does not alter
the class color.

Controls: left-drag rotate, right/middle-drag pan, wheel zoom, double-click or
**Reset View** to reset, plus bbox/PCA toggles and point-size control. The
centroid, bbox and PCA line use SDK geometry; Qt does not recompute PCA. The
previous successful view is preserved on later detection/render failure.
