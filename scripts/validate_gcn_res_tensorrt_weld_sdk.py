"""Validate the Phase 9D WeldDetector SDK against the frozen Phase 9C result."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260717_173128_144483_phase8d_production_baseline"
)
MANIFEST = BASELINE_ROOT / "package" / "manifests" / "deployment_manifest.json"
CLOUD = PROJECT_ROOT / "data" / "weld" / "000001" / "weld_65.txt"
PHASE9C_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260717_221445_024382_phase9c_cpp_postprocess"
    / "result"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-smoke", type=Path, required=True)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--phase9c-root", type=Path, default=PHASE9C_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], log_path: Path, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=1800)
    log_path.write_text(
        f"COMMAND: {subprocess.list2cmdline(command)}\n\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        encoding="utf-8",
    )
    if expect_success and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {log_path}")
    return result


def max_error(left: object, right: object) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def parse_prediction(path: Path) -> np.ndarray:
    prediction = np.loadtxt(path, dtype=np.float64)
    if prediction.shape != (2048, 4):
        raise RuntimeError(f"Unexpected prediction contract at {path}: {prediction.shape}")
    return prediction


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (args.output_dir or (
        PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt" / f"{timestamp}_phase9d_weld_sdk"
    )).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    package = MANIFEST.parent.parent
    engine = (package / manifest["artifacts"]["engine"]).resolve()
    plugin = (package / manifest["artifacts"]["plugin"]).resolve()
    if sha256(engine) != manifest["engine_sha256"] or sha256(plugin) != manifest["plugin_sha256"]:
        raise RuntimeError("Production Engine/Plugin hash validation failed")

    phase9c_result_path = args.phase9c_root / "weld_result.json"
    phase9c_prediction_path = args.phase9c_root / "prediction.txt"
    if not phase9c_result_path.is_file() or not phase9c_prediction_path.is_file():
        raise RuntimeError(f"Frozen Phase 9C baseline is incomplete: {args.phase9c_root}")
    phase9c_result = json.loads(phase9c_result_path.read_text(encoding="utf-8"))
    phase9c_prediction = parse_prediction(phase9c_prediction_path)
    phase9c_labels = np.rint(phase9c_prediction[:, 3]).astype(np.int64)

    sdk_output = output_dir / "sdk_output"
    sdk_labels_path = output_dir / "sdk_labels.txt"
    sdk_result_path = output_dir / "sdk_result.json"
    sdk_command = [
        str(args.sdk_smoke.resolve()),
        "--engine", str(engine),
        "--plugin", str(plugin),
        "--cloud", str(CLOUD),
        "--output", str(sdk_output),
        "--labels-output", str(sdk_labels_path),
        "--result-output", str(sdk_result_path),
    ]
    sdk_run = run(sdk_command, output_dir / "sdk_smoke.log")
    if "STATUS=SUCCESS" not in sdk_run.stdout or "SDK_SMOKE_TEST_PASSED" not in sdk_run.stdout:
        raise RuntimeError("SDK smoke success markers are missing")

    app_output = output_dir / "app_output"
    app_command = [
        str(args.app.resolve()),
        "--engine", str(engine),
        "--plugin", str(plugin),
        "--cloud", str(CLOUD),
        "--output", str(app_output),
    ]
    app_run = run(app_command, output_dir / "weld_trt_app.log")
    if "WELD_DETECTOR_APP_COMPLETED" not in app_run.stdout:
        raise RuntimeError("SDK-only weld_trt_app completion marker is missing")

    sdk_result = json.loads(sdk_result_path.read_text(encoding="utf-8"))
    sdk_labels = np.loadtxt(sdk_labels_path, dtype=np.int64)
    sdk_prediction = parse_prediction(sdk_output / "prediction.txt")
    app_prediction = parse_prediction(app_output / "prediction.txt")
    app_result = json.loads((app_output / "weld_result.json").read_text(encoding="utf-8"))
    if sdk_labels.shape != (2048,):
        raise RuntimeError(f"Unexpected WeldResult.labels contract: {sdk_labels.shape}")

    geometry_errors = {
        "weld_ratio": max_error(sdk_result["weld_ratio"], phase9c_result["weld_ratio"]),
        "center": max_error(sdk_result["center"], phase9c_result["center"]),
        "bbox_min": max_error(sdk_result["bbox_min"], phase9c_result["bbox"]["min"]),
        "bbox_max": max_error(sdk_result["bbox_max"], phase9c_result["bbox"]["max"]),
        "length_mm": max_error(sdk_result["length_mm"], phase9c_result["length_mm"]),
    }
    app_geometry_errors = {
        "weld_ratio": max_error(app_result["weld_ratio"], sdk_result["weld_ratio"]),
        "center": max_error(app_result["center"], sdk_result["center"]),
        "bbox_min": max_error(app_result["bbox"]["min"], sdk_result["bbox_min"]),
        "bbox_max": max_error(app_result["bbox"]["max"], sdk_result["bbox_max"]),
        "length_mm": max_error(app_result["length_mm"], sdk_result["length_mm"]),
    }

    too_small = output_dir / "too_small.txt"
    lines = CLOUD.read_text(encoding="utf-8").splitlines()
    too_small.write_text("\n".join(lines[:2047]) + "\n", encoding="utf-8")
    missing_engine = output_dir / "missing.plan"
    missing_plugin = output_dir / "missing.dll"
    missing_cloud = output_dir / "missing.txt"
    failure_specs = [
        ("missing_engine", "ENGINE_LOAD_FAILED", missing_engine, plugin, CLOUD),
        ("missing_plugin", "PLUGIN_LOAD_FAILED", engine, missing_plugin, CLOUD),
        ("missing_cloud", "POINTCLOUD_LOAD_FAILED", engine, plugin, missing_cloud),
        ("point_count_below_2048", "PREPROCESS_FAILED", engine, plugin, too_small),
    ]
    failure_results = []
    for case, expected_status, case_engine, case_plugin, case_cloud in failure_specs:
        command = [
            str(args.sdk_smoke.resolve()),
            "--engine", str(case_engine),
            "--plugin", str(case_plugin),
            "--cloud", str(case_cloud),
        ]
        failure = run(command, output_dir / f"fail_{case}.log", expect_success=False)
        combined = failure.stdout + failure.stderr
        failure_results.append({
            "case": case,
            "expected_status": expected_status,
            "reported_status": next(
                (line.split("=", 1)[1] for line in combined.splitlines() if line.startswith("STATUS=")),
                None,
            ),
            "exit_code": int(failure.returncode),
            "passed": failure.returncode != 0 and f"STATUS={expected_status}" in combined,
            "fallback": "NONE",
        })
    fail_closed = {
        "status": "PASS" if all(item["passed"] for item in failure_results) else "FAILED",
        "tested_cases": len(failure_results),
        "passed_cases": sum(bool(item["passed"]) for item in failure_results),
        "cases": failure_results,
    }
    (output_dir / "fail_closed_report.json").write_text(
        json.dumps(fail_closed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    phase9c_matching = int(np.count_nonzero(sdk_labels == phase9c_labels))
    sdk_prediction_labels = np.rint(sdk_prediction[:, 3]).astype(np.int64)
    app_labels = np.rint(app_prediction[:, 3]).astype(np.int64)
    report = {
        "status": "PASS",
        "sample_id": "weld_65",
        "sdk_smoke": {
            "exit_code": int(sdk_run.returncode),
            "success": bool(sdk_result["success"]),
            "task_id": sdk_result["task_id"],
            "total_points": int(sdk_result["total_points"]),
            "original_points": int(sdk_result["original_points"]),
            "sampled_points": int(sdk_result["sampled_points"]),
            "labels_count": int(sdk_result["labels_count"]),
            "points_count": int(sdk_result["points_count"]),
            "weld_points": int(sdk_result["weld_points"]),
            "length_mm": float(sdk_result["length_mm"]),
            "error_recorder_errors": int(sdk_result["error_recorder_errors"]),
        },
        "phase9c_compatibility": {
            "matching_labels": phase9c_matching,
            "total_labels": 2048,
            "label_agreement": phase9c_matching / 2048.0,
            "weld_points_equal": int(sdk_result["weld_points"]) == int(phase9c_result["weld_points"]),
            "geometry_errors": geometry_errors,
            "maximum_geometry_error": max(geometry_errors.values()),
        },
        "sdk_output_consistency": {
            "weld_result_json": (sdk_output / "weld_result.json").is_file(),
            "weld_points_ply": (sdk_output / "weld_points.ply").is_file(),
            "prediction_txt": (sdk_output / "prediction.txt").is_file(),
            "weld_result_labels_equal": bool(np.array_equal(sdk_labels, sdk_prediction_labels)),
        },
        "sdk_only_app": {
            "exit_code": int(app_run.returncode),
            "labels_equal_sdk": bool(np.array_equal(app_labels, sdk_labels)),
            "geometry_errors": app_geometry_errors,
            "maximum_geometry_error": max(app_geometry_errors.values()),
        },
        "fail_closed": fail_closed,
        "engine_sha256": sha256(engine),
        "plugin_sha256": sha256(plugin),
    }
    passed = all([
        report["sdk_smoke"]["success"],
        report["sdk_smoke"]["task_id"] == "weld_65",
        report["sdk_smoke"]["total_points"] == 2048,
        report["sdk_smoke"]["original_points"] == 2048,
        report["sdk_smoke"]["sampled_points"] == 2048,
        report["sdk_smoke"]["labels_count"] == 2048,
        report["sdk_smoke"]["points_count"] == 2048,
        report["sdk_smoke"]["weld_points"] == 209,
        report["sdk_smoke"]["error_recorder_errors"] == 0,
        abs(report["sdk_smoke"]["length_mm"] - 57.19605255) < 1.0e-3,
        phase9c_matching == 2048,
        report["phase9c_compatibility"]["weld_points_equal"],
        max(geometry_errors.values()) < 1.0e-5,
        all(report["sdk_output_consistency"].values()),
        report["sdk_only_app"]["labels_equal_sdk"],
        max(app_geometry_errors.values()) < 1.0e-5,
        fail_closed["status"] == "PASS",
    ])
    report["status"] = "PASS" if passed else "FAILED"
    (output_dir / "weld_sdk_validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        print("WELD_DETECTOR_SDK_BLOCKED", file=sys.stderr)
        return 1
    print("WELD_DETECTOR_SDK_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
