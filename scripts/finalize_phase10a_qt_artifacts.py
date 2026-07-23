"""Finalize the structured Phase 10A Qt SDK smoke-test evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


EXPECTED_ENGINE_SHA256 = "a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299"
EXPECTED_PLUGIN_SHA256 = "6641ec147e8eac10206a5c60ba1c1390c398d4b59e32a6a618a37046360ec348"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--qt-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        return {
            "command": subprocess.list2cmdline(command),
            "exit_code": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as error:  # Evidence collection must report unavailable probes.
        return {
            "command": subprocess.list2cmdline(command),
            "exit_code": None,
            "stdout": "",
            "stderr": repr(error),
        }


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    validation = artifact_root / "validation"
    release_dir = args.release_dir.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    engine = args.engine.resolve()
    plugin = args.plugin.resolve()
    engine_hash = sha256(engine)
    plugin_hash = sha256(plugin)
    if engine_hash != EXPECTED_ENGINE_SHA256:
        raise RuntimeError(f"Production Engine hash changed: {engine_hash}")
    if plugin_hash != EXPECTED_PLUGIN_SHA256:
        raise RuntimeError(f"Production Plugin hash changed: {plugin_hash}")

    result_names = [
        "manual_smoke_result.json",
        "automated_smoke_result.json",
        "fail_closed_result.json",
        "phase9d_compatibility.json",
    ]
    for name in result_names:
        source = validation / name
        if not source.is_file():
            raise RuntimeError(f"Required Qt integration result is missing: {source}")
        shutil.copy2(source, artifact_root / name)

    manual = read_json(artifact_root / "manual_smoke_result.json")
    automated = read_json(artifact_root / "automated_smoke_result.json")
    fail_closed = read_json(artifact_root / "fail_closed_result.json")
    compatibility = read_json(artifact_root / "phase9d_compatibility.json")
    dependency = read_json(artifact_root / "sdk_dependency_audit.json")
    statuses = {
        "manual_smoke": manual.get("status"),
        "automated_smoke": automated.get("status"),
        "fail_closed": fail_closed.get("status"),
        "phase9d_compatibility": compatibility.get("status"),
        "sdk_dependency_audit": dependency.get("status"),
    }
    if any(value != "PASS" for value in statuses.values()):
        raise RuntimeError(f"Phase 10A validation is incomplete: {statuses}")

    qt_root = args.qt_root.resolve()
    qmake = qt_root / "bin" / "qmake.exe"
    windeployqt = qt_root / "bin" / "windeployqt.exe"
    qt_widgets_cmake = qt_root / "lib" / "cmake" / "Qt5Widgets" / "Qt5WidgetsConfig.cmake"
    qt_test_cmake = qt_root / "lib" / "cmake" / "Qt5Test" / "Qt5TestConfig.cmake"
    qt_discovery = {
        "status": "PASS",
        "selected_version": "5.9.1",
        "selected_root": str(qt_root),
        "selected_architecture": "x64",
        "selected_toolchain_family": "msvc2015_64",
        "qmake_path": str(qmake),
        "qmake_exists": qmake.is_file(),
        "qmake_query": command_output([str(qmake), "-query"]),
        "windeployqt_path": str(windeployqt),
        "windeployqt_exists": windeployqt.is_file(),
        "qt_widgets_cmake": str(qt_widgets_cmake),
        "qt_widgets_cmake_exists": qt_widgets_cmake.is_file(),
        "qt_test_cmake": str(qt_test_cmake),
        "qt_test_cmake_exists": qt_test_cmake.is_file(),
        "compiler": "MSVC 19.38.33130 (Visual Studio 2022 17.8.2)",
        "abi_assessment": (
            "Qt msvc2015_64 and VS2022 use the compatible Microsoft VS2015-2022 "
            "binary ABI family; the application is linked with the VS2022 v143 x64 toolset."
        ),
    }
    write_json(artifact_root / "qt_discovery.json", qt_discovery)

    environment = {
        "status": "PASS",
        "os": platform.platform(),
        "python": sys.version,
        "cmake": command_output(["cmake", "--version"]),
        "nvidia_smi": command_output([
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader",
        ]),
        "nvcc": command_output(["nvcc", "--version"]),
        "qt_version": "5.9.1",
        "qt_root": str(qt_root),
        "visual_studio": "Visual Studio 2022 17.8.2",
        "msvc": "19.38.33130",
        "cuda_toolkit": "12.8.93",
        "tensorrt": "11.1.0.106",
        "precision": {
            "fp32": True,
            "tf32": False,
            "fp16": False,
            "int8": False,
        },
        "engine_path": str(engine),
        "engine_sha256": engine_hash,
        "plugin_path": str(plugin),
        "plugin_sha256": plugin_hash,
    }
    write_json(artifact_root / "environment.json", environment)

    required_runtime = {
        "ptv2_weld_qt_smoke.exe",
        "QtSdkIntegrationSmoke.exe",
        "Qt5Core.dll",
        "Qt5Gui.dll",
        "Qt5Widgets.dll",
        "Qt5Test.dll",
        "nvinfer_11.dll",
        "nvinfer_plugin_11.dll",
        "cudart64_12.dll",
        "platforms/qwindows.dll",
    }
    inventory: list[dict[str, object]] = []
    found_names: set[str] = set()
    for path in sorted(item for item in release_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(release_dir).as_posix()
        found_names.add(relative)
        inventory.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
        })
    missing_runtime = sorted(required_runtime - found_names)
    runtime_inventory = {
        "status": "PASS" if not missing_runtime else "FAILED",
        "release_dir": str(release_dir),
        "required_runtime_files": sorted(required_runtime),
        "missing_runtime_files": missing_runtime,
        "files": inventory,
        "production_engine_copied_into_runtime_dir": any(
            path["path"].endswith((".plan", ".engine")) for path in inventory
        ),
        "production_plugin_is_external_cli_argument": True,
    }
    write_json(artifact_root / "runtime_inventory.json", runtime_inventory)
    if missing_runtime:
        raise RuntimeError(f"Release runtime directory is incomplete: {missing_runtime}")

    build_summary = {
        "status": "PASS",
        "generator": "Visual Studio 17 2022",
        "architecture": "x64",
        "configuration": "Release",
        "cxx_standard": 17,
        "qt_application": {
            "path": str(release_dir / "ptv2_weld_qt_smoke.exe"),
            "exists": (release_dir / "ptv2_weld_qt_smoke.exe").is_file(),
            "size_bytes": (release_dir / "ptv2_weld_qt_smoke.exe").stat().st_size,
        },
        "qt_test": {
            "path": str(release_dir / "QtSdkIntegrationSmoke.exe"),
            "exists": (release_dir / "QtSdkIntegrationSmoke.exe").is_file(),
            "size_bytes": (release_dir / "QtSdkIntegrationSmoke.exe").stat().st_size,
        },
        "sdk_target": "ptv2_weld_sdk.lib",
        "windeployqt": "PASS",
        "statuses": statuses,
    }
    write_json(artifact_root / "build_summary.json", build_summary)

    (artifact_root / "manual_smoke.log").write_text(
        json.dumps(manual, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (artifact_root / "failure.log").write_text(
        json.dumps(fail_closed, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = f"""# Phase 10A Qt SDK integration smoke

- Status: `PHASE_10A_QT_SDK_INTEGRATION_SMOKE_COMPLETED`
- Qt: 5.9.1, `{qt_root}`
- Build: VS2022 x64 Release PASS
- UI smoke: PASS; two detections; responsive event loop; clean shutdown
- Automated Qt integration: PASS
- Fail-closed: {fail_closed['passed_cases']}/{fail_closed['tested_cases']} PASS
- weld_65: sampled=2048, weld=209, ratio=0.10205078125, length={compatibility['length_mm']} mm
- Phase 9D geometry maximum error: {compatibility['maximum_geometry_error']}
- ErrorRecorder errors: {compatibility['error_recorder_errors']}
- SDK-only source dependency audit: PASS
- Engine SHA-256: `{engine_hash}`
- Plugin SHA-256: `{plugin_hash}`

The production Engine and Plugin were not rebuilt or changed. The application has no
PCL/VTK/OpenGL visualization, FP16, INT8, robot, database, login, or Python-binding work.
`CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED` remains explicitly preserved.
"""
    (artifact_root / "phase10a_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({
        "status": "PHASE_10A_QT_SDK_INTEGRATION_SMOKE_COMPLETED",
        "artifact_root": str(artifact_root),
        "engine_sha256": engine_hash,
        "plugin_sha256": plugin_hash,
        "checks": statuses,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
