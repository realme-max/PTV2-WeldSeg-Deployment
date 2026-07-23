# PTV2 Weld Segmentation 0.1.1 — Quick Start

## Run the packaged application

1. Keep the package directory intact.
2. Double-click `launch.bat`.
3. Wait until the status is `READY`.
4. Select a four-column weld point-cloud TXT with at least 2048 valid points.
5. Click **Detect**.
6. Inspect the colored point cloud and geometry, then use **Export Result**.

Class `0` is `weld_seam`; class `1` is `background`.

The launcher resolves Engine, Plugin and configuration paths relative to its
own directory. Do not move only the EXE. The application verifies both
production SHA-256 identities before SDK initialization.

Each successful export contains `weld_result.json`, `weld_points.ply`,
`prediction.txt`, `detection_view.png` and `task_manifest.json`.

The visualization and the right information panel are separated by a movable
splitter. When the window is short, use the right-side vertical scrollbar to
reach detection, integrity, history and log sections.

Runtime requirements are Windows x64, a compatible NVIDIA driver and an NVIDIA
GPU. Qt, TensorRT/CUDA runtime files, Engine and VoxelUniqueCub Plugin are
included in the qualified package. Inspect `logs/` when startup fails.
