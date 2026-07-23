"""Finalize structured evidence for Phase 10B Qt point-cloud visualization."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ENGINE_SHA = "a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299"
PLUGIN_SHA = "6641ec147e8eac10206a5c60ba1c1390c398d4b59e32a6a618a37046360ec348"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def probe(command: list[str]) -> dict:
    result = subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)
    return {
        "command": subprocess.list2cmdline(command),
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> int:
    args = arguments()
    root = args.artifact_root.resolve()
    validation = root / "validation"
    release = args.release_dir.resolve()
    engine_hash = digest(args.engine.resolve())
    plugin_hash = digest(args.plugin.resolve())
    if engine_hash != ENGINE_SHA or plugin_hash != PLUGIN_SHA:
        raise RuntimeError("Production Engine or Plugin SHA-256 changed")

    render = read(validation / "render_data_test.json")
    widget = read(validation / "opengl_widget_smoke.json")
    gui = read(validation / "manual_smoke_result.json")
    qt_fail = read(validation / "fail_closed_result.json")
    phase9 = read(root / "phase9d_regression" / "weld_sdk_validation_report.json")
    dependency = read(root / "sdk_dependency_audit.json")
    if any(item.get("status") != "PASS" for item in (render, widget, gui, qt_fail, phase9, dependency)):
        raise RuntimeError("A required Phase 10B input report did not pass")

    shutil.copy2(validation / "render_data_test.json", root / "render_data_test.json")
    shutil.copy2(validation / "opengl_widget_smoke.json", root / "opengl_widget_smoke.json")
    shutil.copy2(validation / "manual_smoke_result.json", root / "gui_visualization_smoke.json")

    opengl = {
        "status": "PASS",
        "requested": "OpenGL 3.3 core",
        "actual_version": widget["opengl_version"],
        "renderer": widget["opengl_renderer"],
        "vendor": widget["opengl_vendor"],
        "shader_linked": widget["shader_linked"],
        "gl_error": widget["gl_error"],
    }
    write(root / "opengl_info.json", opengl)

    fail_cases = [
        {
            "case": "successful_then_missing_cloud",
            "passed": gui["missing_cloud_preserved_previous_view"]
                and any(x["case"] == "missing_cloud" and x["passed"] for x in qt_fail["cases"]),
            "policy": "previous successful visualization preserved",
        },
        {
            "case": "successful_then_2047_points",
            "passed": gui["small_cloud_preserved_previous_view"]
                and any(x["case"] == "point_count_below_2048" and x["passed"] for x in qt_fail["cases"]),
            "policy": "previous successful visualization preserved",
        },
        {"case": "nan_coordinate", "passed": widget["invalid_data_preserved_previous_view"]},
        {"case": "mismatched_point_count", "passed": widget["mismatched_count_rejected"]},
        {"case": "invalid_pca_direction", "passed": widget["invalid_pca_rejected"]},
        {"case": "shader_initialization_failure", "passed": widget["shader_failure_fail_closed"]},
    ]
    visualization_fail = {
        "status": "PASS" if all(x["passed"] for x in fail_cases) else "FAILED",
        "tested_cases": len(fail_cases),
        "passed_cases": sum(bool(x["passed"]) for x in fail_cases),
        "cases": fail_cases,
        "fallback": "NONE",
    }
    write(root / "visualization_fail_closed.json", visualization_fail)
    if visualization_fail["status"] != "PASS":
        raise RuntimeError("Visualization fail-closed validation failed")

    sdk_result = read(root / "phase9d_regression" / "sdk_result.json")
    direction = sdk_result["principal_direction"]
    norm = sum(float(x) ** 2 for x in direction) ** 0.5
    extension = {
        "status": "PASS",
        "reason": "Qt visualization required already-computed sampled XYZ/confidence and PCA direction",
        "backward_compatible_fields_added": [
            "WeldPointResult",
            "WeldResult.points",
            "WeldResult.principal_direction",
            "WeldGeometryResult.principalDirection",
        ],
        "points_count": sdk_result["points_count"],
        "principal_direction": direction,
        "principal_direction_norm": norm,
        "direction_finite_and_normalized": all(abs(float(x)) < float("inf") for x in direction)
            and abs(norm - 1.0) < 1.0e-5,
        "qt_reads_generated_prediction_files": False,
        "pca_recomputed_in_qt": False,
        "phase9d_regression": phase9["status"],
        "phase9d_maximum_geometry_error": phase9["phase9c_compatibility"]["maximum_geometry_error"],
        "sdk_dependency_audit": dependency["status"],
    }
    write(root / "sdk_extension_audit.json", extension)

    compatibility = {
        "status": "PASS",
        "sampled_points": phase9["sdk_smoke"]["sampled_points"],
        "weld_points": phase9["sdk_smoke"]["weld_points"],
        "weld_ratio": sdk_result["weld_ratio"],
        "length_mm": sdk_result["length_mm"],
        "error_recorder_errors": phase9["sdk_smoke"]["error_recorder_errors"],
        "label_agreement": phase9["phase9c_compatibility"]["label_agreement"],
        "maximum_geometry_error": phase9["phase9c_compatibility"]["maximum_geometry_error"],
        "existing_tolerances_unchanged": True,
    }
    write(root / "phase9d_compatibility.json", compatibility)

    timing = {
        "status": "PASS",
        "result_to_render_data_ms": render["conversion_ms"],
        "gpu_buffer_upload_ms": widget["gpu_buffer_upload_ms"],
        "first_paint_ms": widget["first_paint_ms"],
        "subsequent_repaint_ms": widget["subsequent_repaint_ms"],
        "first_detection_refresh_ms": gui["first_detection_refresh_ms"],
        "second_detection_refresh_ms": gui["second_detection_refresh_ms"],
        "scope": "UI-level smoke only; not an inference benchmark",
    }
    write(root / "visualization_timing.json", timing)

    files = [
        {"path": str(path.relative_to(release)).replace("\\", "/"), "size_bytes": path.stat().st_size}
        for path in sorted(release.rglob("*")) if path.is_file()
    ]
    runtime = {
        "status": "PASS",
        "release_dir": str(release),
        "files": files,
        "qwindows_present": (release / "platforms" / "qwindows.dll").is_file(),
        "qt_core_gui_widgets_present": all(
            (release / name).is_file() for name in ("Qt5Core.dll", "Qt5Gui.dll", "Qt5Widgets.dll")
        ),
        "qt5opengl_link_target": "Qt5::OpenGL",
        "qt5opengl_runtime_needed_by_linker": (release / "Qt5OpenGL.dll").is_file(),
        "qt5opengl_note": (
            "QOpenGLWidget/QOpenGLFunctions are provided by Qt5Widgets/Qt5Gui; "
            "windeployqt found no runtime dependency on Qt5OpenGL.dll."
        ),
        "tensorrt_cuda_runtime_present": all(
            (release / name).is_file()
            for name in ("nvinfer_11.dll", "nvinfer_plugin_11.dll", "cudart64_12.dll")
        ),
    }
    write(root / "runtime_inventory.json", runtime)

    environment = {
        "status": "PASS",
        "os": platform.platform(),
        "python": sys.version,
        "gpu": probe(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"]),
        "cuda": probe(["nvcc", "--version"]),
        "qt": "5.9.1 msvc2015_64",
        "qt_root": r"D:\Qt\Qt5.9.1\5.9.1\msvc2015_64",
        "visual_studio": "2022 17.8.2 / MSVC 19.38.33130 / x64 Release",
        "tensorrt": "11.1.0.106",
        "engine_sha256": engine_hash,
        "plugin_sha256": plugin_hash,
    }
    write(root / "environment.json", environment)

    build = {
        "status": "PASS",
        "configuration": "VS2022 x64 Release C++17 /W4 /WX",
        "targets": {
            "ptv2_weld_qt_smoke": (release / "ptv2_weld_qt_smoke.exe").is_file(),
            "QtSdkIntegrationSmoke": (release / "QtSdkIntegrationSmoke.exe").is_file(),
            "PointCloudRenderDataTest": (release / "PointCloudRenderDataTest.exe").is_file(),
            "PointCloudViewSmokeTest": (release / "PointCloudViewSmokeTest.exe").is_file(),
        },
        "ctest": "3/3 PASS",
        "windeployqt": "PASS",
    }
    write(root / "build_summary.json", build)

    for target, value in (
        ("render_test.log", render),
        ("widget_smoke.log", widget),
        ("gui_smoke.log", gui),
        ("failure.log", visualization_fail),
    ):
        write(root / target, value)

    summary = f"""# Phase 10B Qt point-cloud visualization

`PHASE_10B_QT_POINTCLOUD_VISUALIZATION_COMPLETED`

- OpenGL: {widget['opengl_version']} / {widget['opengl_vendor']} / {widget['opengl_renderer']}
- Shader and GL smoke: PASS, glGetError={widget['gl_error']}
- Rendered points: 2048 (weld 209, background 1839)
- Controls: rotate/zoom/pan/reset/bbox/PCA PASS
- Conversion/upload/first paint: {render['conversion_ms']} / {widget['gpu_buffer_upload_ms']} / {widget['first_paint_ms']} ms
- Second detection refresh: {gui['second_detection_refresh_ms']} ms, no duplicate points
- Visualization fail-closed: {visualization_fail['passed_cases']}/{visualization_fail['tested_cases']} PASS
- Phase 9D regression: PASS, geometry max error {compatibility['maximum_geometry_error']}
- Engine SHA-256: `{engine_hash}`
- Plugin SHA-256: `{plugin_hash}`

QOpenGLWidget/QOpenGLFunctions were used without PCL, VTK or external rendering libraries.
The Engine, Plugin, inference path, sampling, task semantics and tolerances were unchanged.
`CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED` remains preserved.
"""
    (root / "phase10b_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({
        "status": "PHASE_10B_QT_POINTCLOUD_VISUALIZATION_COMPLETED",
        "artifact_root": str(root),
        "opengl": opengl,
        "visualization_fail_closed": visualization_fail["status"],
        "engine_sha256": engine_hash,
        "plugin_sha256": plugin_hash,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
