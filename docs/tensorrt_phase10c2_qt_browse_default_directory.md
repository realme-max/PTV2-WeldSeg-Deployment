# Phase 10C.2: Qt Browse Default Directory

Status:

```text
PHASE_10C2_QT_BROWSE_DEFAULT_DIRECTORY_COMPLETED
```

Evidence root:
`artifacts/gcn_res_tensorrt/20260723_194208_270011_phase10c2_qt_browse_default_directory/`.

Qualified Release package:
`D:\PTV2_Weld_App_0.1.1_Phase10C2_QUALIFIED_20260723_194357`.

## Scope

This change is limited to the Qt point-cloud Browse workflow. It does not
modify the WeldDetector SDK, TensorRT Engine, VoxelUnique Plugin, checkpoint,
ONNX, preprocessing, detection, visualization, post-processing or export
semantics.

## Directory priority

`MainWindow::resolveInitialCloudDirectory()` returns the first existing
directory in this order:

1. `[Application] last_cloud_directory` when `remember_last_cloud=true`;
2. configured `default_cloud_directory`;
3. package-local `<applicationDir>/data/weld/000001`;
4. development directory `E:/GRP-PTv2/data/weld/000001`;
5. the application executable directory;
6. `QDir::homePath()`.

Missing candidates are skipped. The file dialog remains restricted to
`*.txt`. Cancelling leaves the current point-cloud path and application state
unchanged. A successful selection stores the selected file's parent directory
in the user INI only when `remember_last_cloud=true`; it does not start
detection.

## Validation

VS2022 x64 Release with C++17, `/W4` and `/WX`: PASS.

The new `QtCloudBrowseDirectoryTest` passed 9/9 checks:

- development default without history;
- recent-directory priority;
- missing-history fallback;
- configured-directory priority;
- real `QFileDialog` initial directory and `*.txt` filter;
- cancel preserves the current path and state;
- accepted selection persists its parent directory;
- a second real Browse opens the persisted directory;
- `remember_last_cloud=false` neither updates nor consumes recent history;
- Browse does not trigger detection, visualization or export.

Full CTest regression: 10/10 PASS, including SDK integration, rendering,
OpenGL, layout/resize, configuration, history and export tests.

The new Release package launched successfully and remained responsive, then
closed cleanly. The Windows UI capture helper could not attach to the
Qt/OpenGL window and returned `SetIsBorderRequired failed (0x80004002)`.
Therefore no claim is made that the native dialog was manually clicked through
that helper. The Browse behavior itself was exercised through the real Qt file
dialog in the integration test, including first open, cancel, accepted
selection persistence and second open.

