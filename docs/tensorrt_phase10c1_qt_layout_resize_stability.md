# TensorRT Phase 10C.1: Qt Layout, Scrolling and Resize Stability

## Outcome

```text
PHASE_10C1_QT_LAYOUT_SCROLL_RESIZE_STABILITY_COMPLETED
```

Evidence root:
`artifacts/gcn_res_tensorrt/20260723_184946_689058_phase10c1_qt_layout_resize_stability/`.

Qualified package:
`D:\PTV2_Weld_App_0.1.1_Phase10C1_QUALIFIED_20260723_184946`.

No checkpoint, ONNX, TensorRT Engine, VoxelUniqueCub Plugin, SDK inference,
preprocessing, label, post-processing geometry, tolerance or export-data
semantics changed.

## Investigation and root cause

The original `MainWindow::buildUi()` hierarchy was:

```text
QMainWindow
└─ centralWidget / QVBoxLayout
   ├─ inputGroup
   ├─ visualizationGroup
   │  └─ PointCloudView (minimum 520 x 420)
   ├─ resultGroup (20 rows)
   ├─ recentTasksGroup
   └─ runtimeLogGroup / QTextEdit
```

The code had no `QScrollArea`, no `QSplitter`, and no alternate route to the
lower controls. The missing scrollbar therefore had a direct structural
cause: the complete page was one non-scrollable vertical layout.

The resize symptom had the same primary layout cause. The vertical sum of the
OpenGL minimum height, result form, recent-task widget and log editor propagated
to the main-window minimum size hint. Once a border drag reached that layout
floor, the window could not shrink further and appeared stuck. The source audit
found no `MainWindow::resizeEvent`, recursive geometry change, `adjustSize`,
`repaint`, settings/history reload, mutex acquisition or worker wait in the
resize path.

`PointCloudView::resizeGL()` only called `glViewport`; point buffers were
uploaded only when `buffersDirty_` was set by new render data or a visualization
toggle. As a secondary Qt 5.9 safety improvement, empty-state text is now a
transparent child `QLabel`, not a `QPainter` created inside `paintGL()` during
every empty-view redraw. This keeps `paintGL()` limited to existing GL state and
buffers.

## Layout and usability changes

`MainWindow::buildUi()` now creates:

```text
QMainWindow
└─ centralWidget / QVBoxLayout
   ├─ inputGroup
   └─ mainContentSplitter / QSplitter(Qt::Horizontal)
      ├─ visualizationGroup / PointCloudView
      └─ rightScrollArea / QScrollArea
         └─ rightScrollContent / QVBoxLayout(Qt::AlignTop)
            ├─ resultGroup
            ├─ recentTasksGroup
            ├─ runtimeLogGroup
            └─ stretch
```

The scroll area is widget-resizable, uses an as-needed vertical bar, disables
unnecessary horizontal scrolling, and has no fixed height. The splitter uses
3:1 stretch factors and can be adjusted by the user.

Other changes:

- the cloud path and action buttons use separate form rows to reduce the
  minimum width;
- result values wrap, are selectable, and retain the full value as a tooltip;
- the history list elides long display paths but retains the stored path;
- the log view is a `QPlainTextEdit` with a 2,000-block visible-document cap;
- `PointCloudView` is expanding with a `400 x 300` minimum instead of
  `520 x 420`;
- splitter state and window geometry are persisted;
- invalid/off-screen saved geometry falls back to a centered visible default;
- the Release/package version is `0.1.1`.

## OpenGL resize contract

`PointCloudView::resizeGL()` now clamps width and height to at least one,
updates the viewport and stores a finite aspect ratio. `paintGL()` uses that
ratio and does not fit/reset the camera, rebuild CPU points, upload buffers,
compile shaders or call nested `makeCurrent()` during resize.

Instrumentation used by the regression test records `resizeGlCount()` and
`bufferUploadCount()`. It proved that 122 OpenGL resize calls did not upload a
buffer. A second successful detection caused exactly one expected new upload.

## Automated validation

VS2022 x64 Release, C++17, `/W4 /WX`: PASS.

CTest:

```text
9/9 PASS
```

The new `QtLayoutResizeStabilityTest` verified:

- actual `QScrollArea` ownership and `widgetResizable=true`;
- small-window vertical scrollbar maximum `651`;
- horizontal scrollbar maximum `0`;
- bottom log section reachable after scrolling;
- invalid saved geometry falls back to a visible screen;
- detection/export while resize events are being delivered;
- a second detection while resize events are being delivered;
- 135 total resize operations including the requested size sequence,
  maximize and restore;
- maximum 10 ms heartbeat-timer delay observed: `41 ms`;
- render count `2048`, weld `209`, background `1839`;
- no resize-triggered VBO upload;
- finite and unchanged camera state;
- OpenGL error `0`;
- clean close.

Existing regressions also passed:

- Phase 9D SDK behavior through `QtSdkIntegrationSmoke`;
- Phase 10A worker/integration behavior;
- Phase 10B render-data and OpenGL smoke;
- Phase 10C configuration, state-machine, export, recent-task and runtime
  package tests.

Qualified `weld_65` remains:

| Field | Result |
|---|---:|
| sampled points | 2048 |
| weld points | 209 |
| background points | 1839 |
| weld ratio | 0.10205078125 |
| PCA length | 57.1960525513 mm |
| ErrorRecorder errors | 0 |

## Release-package smoke

`package_release.ps1` produced a new package without overwriting the qualified
0.1.0 package. Package-local `launch.bat` loaded the package-local Engine,
Plugin and `sample\weld_65.txt`, completed detection, exported
JSON/PLY/TXT/PNG plus manifest and exited zero.

Package audit:

- checksum failures: 0;
- source/PDB/build/intermediate files: 0;
- absolute `E:\GRP-PTv2` references: 0;
- Engine SHA-256 unchanged:
  `a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299`;
- Plugin SHA-256 unchanged:
  `6641ec147e8eac10206a5c60ba1c1390c398d4b59e32a6a618a37046360ec348`.

The Windows automation helper identified the packaged Qt window but could not
capture it because the host returned `0x80004002` for the Qt 5.9.1 window
capture interface. Therefore no automated screenshot or synthetic mouse-border
recording is claimed. The same scrolling, resize, maximize/restore, heartbeat,
OpenGL and detection-overlap behavior was exercised directly by the Qt widget
test, and the package-local launch/detect/export path was executed separately.

## Files

Added:

- `deployment/qt_weld_app/tests/QtLayoutResizeStabilityTest.cpp`;
- `docs/tensorrt_phase10c1_qt_layout_resize_stability.md`.

Modified:

- `deployment/qt_weld_app/CMakeLists.txt`;
- `deployment/qt_weld_app/include/MainWindow.h`;
- `deployment/qt_weld_app/src/MainWindow.cpp`;
- `deployment/qt_weld_app/include/PointCloudView.h`;
- `deployment/qt_weld_app/src/PointCloudView.cpp`;
- `deployment/qt_weld_app/src/ProductInfo.cpp`;
- `deployment/qt_weld_app/scripts/package_release.ps1`;
- `deployment/qt_weld_app/README.md`;
- `deployment/qt_weld_app/QUICK_START.md`;
- `.gitignore`;
- `docs/context_handoff.md`.

## Known limitations

The production contract remains B=1, N=2048, FP32, two classes, deterministic
sampling and CPU k=6 KNN. The visible log document retains 2,000 blocks while
full rotating file logs remain independent. The right panel is desktop-oriented
and not redesigned for mobile/touch.
