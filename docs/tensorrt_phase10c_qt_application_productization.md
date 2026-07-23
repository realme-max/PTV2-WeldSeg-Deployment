# TensorRT Phase 10C: Qt Weld Application Productization

## Outcome

```text
PHASE_10C_QT_APPLICATION_PRODUCTIZATION_COMPLETED
```

Evidence root:
`artifacts/gcn_res_tensorrt/20260723_172448_360408_phase10c_qt_productization/`.

No checkpoint, ONNX, production Engine, VoxelUniqueCub algorithm, sampling,
KNN, label, geometry or numerical-threshold semantics changed. No Engine was
rebuilt. PCL, VTK, FP16, INT8, robot control and trajectory work remain out of
scope.

`CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED` remains explicit; Phase 10C does
not relabel task-level equivalence as strict numerical equivalence.

## Product architecture

The existing Qt application remains the only GUI project. Phase 10C adds:

- `AppConfig`: default INI, user INI and process-only CLI override priority;
- `AppStateMachine`: ten states and centralized action gating;
- `DetectionExportService` and `ScreenshotExportService`;
- `ApplicationLogger`, `RecentTaskStore`, `ProductInfo`;
- `RuntimePackageValidator` and `SettingsDialog`.

SDK detection stays on its worker thread. Framebuffer capture occurs on the
GUI thread; atomic file export runs through QtConcurrent. Initialization,
detection and export show an indeterminate busy state and disable conflicts.
Close is rejected while export is active.

## Configuration and integrity

QSettings uses INI format with:

```text
command line > user INI > package/default INI
```

Save is refused until Engine and Plugin exist and both hashes match. Changing
runtime settings stops the old worker and performs controlled SDK
reinitialization.

- Engine: `a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299`;
- Plugin: `6641ec147e8eac10206a5c60ba1c1390c398d4b59e32a6a618a37046360ec348`.

Product Information reports version/build/compiler/Qt/TensorRT/CUDA/OpenGL.
Unavailable Git metadata is `unknown`.

## State, export, logs and history

The state machine prevents Detect before readiness, concurrent Detect, Export
before success, Settings during detection and shutdown during export.

The export service consumes the copied Phase 9D result and never recomputes
geometry. It writes a temporary directory, verifies files, then renames it.
Manifest verification detects a deliberately corrupted payload.

Qualified `weld_65`:

| Item | Result |
|---|---:|
| original / sampled | 2048 / 2048 |
| weld / background | 209 / 1839 |
| weld ratio | 0.10205078125 |
| PCA length | 57.1960525513 mm |
| ErrorRecorder | 0 |
| PLY vertices | 209 |
| prediction rows | 2048 |
| screenshot | 1274 x 420 PNG |
| manifest hashes | PASS |

PLY remains weld-only ASCII `x y z label confidence`, label `0`. Prediction
remains all 2048 points in sampling order.

Logs contain timestamp, level and category. The relocated smoke produced
Startup, SDKInitialization, Detection, Visualization, Export and Shutdown.
Rotation retains 20 files and excludes raw logits, adjacency and full points.

QSettings retains 20 task summaries, deduplicates task IDs, marks missing
sources, reloads paths without rerunning detection and confirms clear-history.

## Release and relocation

`scripts/package_release.ps1` runs `windeployqt`, copies TensorRT/CUDA DLLs
and frozen Engine/Plugin, writes a relative INI/launcher, inventory and
checksums, and fails on missing dependencies.

The qualified package has `qwindows.dll`, no source/PDB/intermediate files,
no duplicate DLL names and no `E:\GRP-PTv2` references. All 29 checksum
entries verified.

It was copied to `D:\PTV2_Weld_App_0.1.0_Phase10C_RELEASE_*`, outside
source/build/artifacts. `launch.bat` used package-local runtime/model files,
detected package-local `weld_65`, rendered 2048 points, exported all five
files and exited zero. Visual Studio environment variables were not required.
Remaining dependencies: Windows x64, compatible NVIDIA driver and GPU.

## Validation

- VS2022 x64 Release C++17 `/W4 /WX`: PASS.
- CTest: 8/8 PASS.
- Phase 9D: 2048/2048 labels, geometry max error
  `4.2578125203363015e-7`, fail-closed 4/4: PASS.
- Phase 10A integration: PASS.
- Phase 10B render-data/OpenGL: PASS.
- Phase 10C productization fail-closed: 15/15 PASS, no fallback.
- Engine and Plugin hashes unchanged.

## Known limitations

The production contract remains B=1, N=2048, FP32, two classes, deterministic
Phase 9B sampling and CPU k=6 KNN. The app is local/single-task and adds no
batch monitoring, database, login, cloud service, PCL/VTK, robot or trajectory
features. Stop after Phase 10C.
