# TensorRT Phase 10A: Qt WeldDetector SDK Integration Smoke

## 1. Purpose and result

Phase 10A adds a minimal Qt Widgets application over the public Phase 9D
`WeldDetector` SDK. The Qt layer performs configuration, file selection,
status/result presentation and worker-thread orchestration only. It does not
access TensorRT, CUDA, the Plugin Registry, point-cloud preprocessing or
post-processing internals.

Final state:

```text
PHASE_10A_QT_SDK_INTEGRATION_SMOKE_COMPLETED
```

The successful evidence root is:

```text
artifacts/gcn_res_tensorrt/20260723_145741_755831_phase10a_qt_sdk_smoke/
```

## 2. Selected local Qt and compiler environment

The read-only discovery selected the existing installation:

- Qt: `5.9.1`, 64-bit MSVC build.
- Qt root: `D:\Qt\Qt5.9.1\5.9.1\msvc2015_64`.
- qmake: `D:\Qt\Qt5.9.1\5.9.1\msvc2015_64\bin\qmake.exe`.
- windeployqt: `D:\Qt\Qt5.9.1\5.9.1\msvc2015_64\bin\windeployqt.exe`.
- CMake package: `lib\cmake\Qt5Widgets\Qt5WidgetsConfig.cmake`.
- Build: Visual Studio 2022 17.8.2, MSVC 19.38.33130, v143 x64 Release.
- CUDA Toolkit: 12.8.93.
- TensorRT: 11.1.0.106.
- GPU: NVIDIA GeForce RTX 5060, compute capability 12.0, driver 610.74.

Qt's `msvc2015_64` and VS2022 use the compatible Microsoft VS2015–2022
binary ABI family. The actual final link uses the VS2022 v143 x64 toolset.
No Qt package or compiler toolset was installed, removed or switched.

## 3. Application architecture

```text
MainWindow (GUI thread)
  -> WeldDetectionWorker (dedicated QThread)
  -> public WeldDetector SDK
  -> existing Phase 9D deployment chain
```

`WeldDetectionWorker` owns exactly one `WeldDetector`. Initialization runs
once when its thread starts. The initialized instance is reused for every
accepted Detect request. An atomic scheduled flag rejects a second request
while a detection is queued or active; concurrent inference is not claimed.

The worker emits copied `QtWeldResultViewModel` values. No raw SDK
references, CUDA objects, TensorRT objects or internal pointers cross the Qt
thread boundary. Window destruction queues a blocking SDK shutdown in the
worker thread, then calls `quit()` and `wait()`.

## 4. SDK-only dependency boundary

The audit script is
`scripts/audit_qt_weld_sdk_dependency_boundary.py`. It scanned eight
Phase 10A headers/sources/tests and found zero forbidden includes.

Qt application sources include only Qt/standard headers, Phase 10A
controller/view headers, and the public SDK headers `WeldConfig.h`,
`WeldDetector.h`, `WeldResult.h`, and `WeldStatus.h`. They do not include:

- `NvInfer.h`, `NvInferPlugin.h`, or `cuda_runtime_api.h`;
- Windows Plugin-loading APIs;
- `PointCloudLoader`, sampling/feature/KNN headers;
- `TensorRTInference`;
- segmentation or geometry implementation headers.

The CMake application/frontend targets link to `ptv2_weld_sdk`; internal
libraries are reached only through the existing static SDK build.

```text
SDK_ONLY_DEPENDENCY_AUDIT_PASSED
```

## 5. Minimal public SDK extension

The existing Phase 9D public result did not expose all UI-required point
counts, stage timings or ErrorRecorder count. This was a real public-interface
defect, so a backward-compatible extension was made before Qt integration:

- `original_points` and `sampled_points`;
- load, sampling, adjacency, inference-wall, post-process and total times;
- `error_recorder_errors`.

The existing `inference_ms` remains the CUDA inference time. No task,
sampling, model, Engine, Plugin, precision or label behavior changed.
Phase 9D was rebuilt and its existing validator passed before Qt work
continued: 2048 labels, 209 weld points, geometry within its existing
tolerance, ErrorRecorder zero, and 4/4 original fail-closed cases.

## 6. UI and configuration contract

The hand-written Widgets UI provides:

- a read-only TXT path, Browse button, initialization status and Detect;
- source/task, symbolic SDK status and original/sampled/weld point counts;
- weld ratio, PCA length, centroid and bounding box;
- load, sample, adjacency, CUDA inference, wall inference, post-process and
  total times;
- ErrorRecorder count and timestamped logs.

There is no point-cloud visualization.

Runtime configuration is supplied only through:

```text
--engine <path>
--plugin <path>
--engine-sha256 <sha256>
--cloud <optional path>
```

`QApplication` has its own `-plugin` option and consumes it during
construction. Phase 10A therefore preserves the raw argument list before
constructing `QApplication`, then parses that preserved list. This keeps the
required SDK `--plugin` CLI contract intact without renaming it.

Missing/invalid required values cause a clear startup error and nonzero exit.
The worker verifies the exact Engine SHA-256 before SDK initialization.
There is no fallback path.

## 7. Build and deployment

Configure and build:

```powershell
cmake -S deployment/qt_weld_app -B <build-dir> `
  -G "Visual Studio 17 2022" -A x64 `
  -DTENSORRT_ROOT="D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106" `
  -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" `
  -DCMAKE_PREFIX_PATH="D:\Qt\Qt5.9.1\5.9.1\msvc2015_64"

cmake --build <build-dir> --config Release --parallel 4
```

Both `ptv2_weld_qt_smoke.exe` and `QtSdkIntegrationSmoke.exe` built
successfully with C++17 and `/W4 /WX`.

Qt deployment:

```powershell
D:\Qt\Qt5.9.1\5.9.1\msvc2015_64\bin\windeployqt.exe `
  --release --no-translations --no-system-d3d-compiler --no-opengl-sw `
  <build-dir>\Release\ptv2_weld_qt_smoke.exe
```

The Release runtime inventory confirms `Qt5Core`, `Qt5Gui`, `Qt5Widgets`,
`qwindows.dll`, TensorRT runtime/plugin DLLs and CUDA Runtime. `Qt5Test` is
present for the integration executable. windeployqt warned that
`VCINSTALLDIR` was not set, but returned success; all required local DLLs
were found and the application/test executed successfully.

The production Engine and VoxelUniqueCub Plugin remain in the versioned
Phase 8D package and are passed by command line. They were not copied into
the Git source tree.

## 8. Launch command

```powershell
ptv2_weld_qt_smoke.exe `
  --engine <phase8d-package>\engine\strict_fp32_voxelunique_cub.plan `
  --plugin <phase8d-package>\plugins\VoxelUniqueCubPlugin.dll `
  --engine-sha256 a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299 `
  --cloud E:\GRP-PTv2\data\weld\000001\weld_65.txt
```

## 9. MainWindow smoke result

A scripted, visible Qt Widgets smoke exercised the actual MainWindow
controls and event loop:

- application start and SDK initialization: PASS;
- initialization status displayed: PASS;
- event-loop callback processed while detection was active: PASS;
- first weld_65 detection: PASS;
- second detection on the same initialized worker/SDK: PASS;
- sampled points: 2048;
- weld points: 209;
- weld ratio: 0.10205078125;
- PCA length: 57.1960525513 mm;
- ErrorRecorder errors: 0;
- close and worker shutdown: PASS.

No visual-only assertion was used for acceptance.

## 10. Automated Qt integration and fail-closed tests

`QtSdkIntegrationSmoke` uses `QSignalSpy`, real worker threads, the
production package and the actual `weld_65.txt`. It verifies construction,
single initialization, two detections, the result contract, active-request
rejection and clean shutdown.

Automated result: PASS.

All seven required fail-closed cases passed with no fake result or fallback:

1. missing Engine path -> `ENGINE_LOAD_FAILED`;
2. missing Plugin path -> `PLUGIN_LOAD_FAILED`;
3. wrong Engine SHA -> `ENGINE_LOAD_FAILED`;
4. missing cloud -> `POINTCLOUD_LOAD_FAILED`;
5. 2047-point cloud -> `PREPROCESS_FAILED`;
6. detection before initialization -> `INVALID_CONFIG`;
7. second active detection -> explicit `REQUEST_REJECTED`.

## 11. Phase 9D compatibility

The Qt-facing copied result exactly preserves the fixed Phase 9D weld_65
contract:

| Field | Phase 10A |
|---|---:|
| Original points | 2048 |
| Sampled points | 2048 |
| Weld points | 209 |
| Weld ratio | 0.10205078125 |
| PCA length | 57.19605255126953 mm |
| Geometry maximum absolute error | 0 |
| ErrorRecorder errors | 0 |

Centroid, bbox and PCA length are within `1e-5`; no tolerance was changed.

Integrity:

- Engine SHA-256:
  `a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299`;
- Plugin SHA-256:
  `6641ec147e8eac10206a5c60ba1c1390c398d4b59e32a6a618a37046360ec348`.

The Engine and Plugin hashes are unchanged. No Engine was rebuilt.

## 12. Known limitations and stop point

- The local Qt version is 5.9.1; the CMake project also supports a compatible
  Qt 6 Widgets/Test installation, but Qt 6 was not tested here.
- For arbitrary non-ASCII Windows command-line paths, a future production
  launcher should use an explicitly UTF-16 argument capture path. The tested
  production and cloud paths are represented correctly by the current local
  Windows code page.
- The application serializes one SDK instance and rejects concurrency.
- No PCL, VTK, OpenGL visualization, robot interface, database/login,
  Python binding, FP16 or INT8 work was started.
- `CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED` remains explicit. Phase 10A
  confirms task compatibility; it does not claim strict PyTorch/Engine
  numerical equivalence.

Stop after Phase 10A. Do not enter Phase 10B without new authorization.
