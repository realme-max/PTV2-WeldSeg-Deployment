"""Validate Phase 9C C++ segmentation post-processing against an independent NumPy reference."""

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
PHASE9B_PREDICTION = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260717_215224_557179_phase9b_cpp_pointcloud_pipeline"
    / "cpp_run_1"
    / "prediction.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--failure-probe", type=Path, required=True)
    parser.add_argument("--phase9b-prediction", type=Path, default=PHASE9B_PREDICTION)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=1800)
    log_path.write_text(
        f"COMMAND: {subprocess.list2cmdline(command)}\n\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {log_path}")
    return result


def load_ply(path: Path) -> tuple[int, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    vertex_line = next(line for line in lines if line.startswith("element vertex "))
    expected = int(vertex_line.rsplit(" ", 1)[1])
    header_end = lines.index("end_header")
    data_lines = lines[header_end + 1 :]
    if len(data_lines) != expected:
        raise RuntimeError(f"PLY declared {expected} vertices but contains {len(data_lines)} rows")
    if not data_lines:
        return expected, np.empty((0, 5), dtype=np.float64)
    data = np.loadtxt(data_lines, dtype=np.float64)
    return expected, np.atleast_2d(data)


def geometry_reference(
    points: np.ndarray, labels: np.ndarray
) -> tuple[dict[str, object], dict[str, object], dict[str, float]]:
    weld = points[labels == 0].astype(np.float64)
    if weld.shape[0] == 0:
        raise RuntimeError("Python reference contains no weld points")
    center = weld.mean(axis=0)
    centered = weld - center
    covariance = (centered.T @ centered) / weld.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    projection = centered @ direction
    reference_fp64 = {
        "weld_points": int(weld.shape[0]),
        "weld_ratio": float(weld.shape[0] / points.shape[0]),
        "center": center.tolist(),
        "bbox_min": weld.min(axis=0).tolist(),
        "bbox_max": weld.max(axis=0).tolist(),
        "length_mm": float(projection.max() - projection.min()),
    }
    reference_fp32 = {
        "weld_points": reference_fp64["weld_points"],
        "weld_ratio": float(np.float32(reference_fp64["weld_ratio"])),
        "center": np.asarray(reference_fp64["center"], dtype=np.float32).astype(np.float64).tolist(),
        "bbox_min": np.asarray(reference_fp64["bbox_min"], dtype=np.float32).astype(np.float64).tolist(),
        "bbox_max": np.asarray(reference_fp64["bbox_max"], dtype=np.float32).astype(np.float64).tolist(),
        "length_mm": float(np.float32(reference_fp64["length_mm"])),
    }
    quantization = {
        key: maximum_error(reference_fp32[key], reference_fp64[key])
        for key in ("weld_ratio", "center", "bbox_min", "bbox_max", "length_mm")
    }
    return reference_fp32, reference_fp64, quantization


def maximum_error(actual: object, reference: object) -> float:
    return float(np.max(np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64))))


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (args.output_dir or (
        PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt" / f"{timestamp}_phase9c_cpp_postprocess"
    )).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    result_dir = output_dir / "result"

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    package = MANIFEST.parent.parent
    engine = (package / manifest["artifacts"]["engine"]).resolve()
    plugin = (package / manifest["artifacts"]["plugin"]).resolve()
    if sha256(engine) != manifest["engine_sha256"] or sha256(plugin) != manifest["plugin_sha256"]:
        raise RuntimeError("Production Engine/Plugin hash validation failed")
    if not args.phase9b_prediction.is_file():
        raise RuntimeError(f"Phase 9B prediction baseline is missing: {args.phase9b_prediction}")

    logits_path = output_dir / "cpp_logits.bin"
    indices_path = output_dir / "sample_indices.bin"
    command = [
        str(args.exe.resolve()),
        "--cloud", str(CLOUD),
        "--engine", str(engine),
        "--plugin", str(plugin),
        "--output", str(result_dir),
        "--report", str(output_dir / "runtime_report.json"),
        "--logits-output", str(logits_path),
        "--sample-indices-output", str(indices_path),
        "--seed", "42",
    ]
    run(command, output_dir / "weld_trt_demo.log")

    required = [result_dir / "weld_result.json", result_dir / "weld_points.ply", result_dir / "prediction.txt"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required result files are missing: {missing}")

    raw = np.loadtxt(CLOUD, dtype=np.float32)
    indices = np.fromfile(indices_path, dtype=np.uint64).astype(np.int64)
    logits = np.fromfile(logits_path, dtype=np.float32).reshape(2048, 2)
    sampled_xyz = np.ascontiguousarray(raw[indices, :3], dtype=np.float32)
    python_labels = np.where(logits[:, 0] > logits[:, 1], 0, 1).astype(np.int64)
    differences = np.abs(logits[:, 0].astype(np.float64) - logits[:, 1].astype(np.float64))
    python_confidence = (1.0 / (1.0 + np.exp(-differences))).astype(np.float32)
    reference, reference_fp64, fp32_quantization = geometry_reference(sampled_xyz, python_labels)

    prediction = np.loadtxt(result_dir / "prediction.txt", dtype=np.float64)
    phase9b_prediction = np.loadtxt(args.phase9b_prediction, dtype=np.float64)
    if prediction.shape != (2048, 4) or phase9b_prediction.shape != (2048, 4):
        raise RuntimeError("Prediction TXT shape contract failed")
    cpp_labels = np.rint(prediction[:, 3]).astype(np.int64)
    phase9b_labels = np.rint(phase9b_prediction[:, 3]).astype(np.int64)

    result = json.loads((result_dir / "weld_result.json").read_text(encoding="utf-8"))
    geometry_errors = {
        "weld_ratio": maximum_error(result["weld_ratio"], reference["weld_ratio"]),
        "center": maximum_error(result["center"], reference["center"]),
        "bbox_min": maximum_error(result["bbox"]["min"], reference["bbox_min"]),
        "bbox_max": maximum_error(result["bbox"]["max"], reference["bbox_max"]),
        "length_mm": maximum_error(result["length_mm"], reference["length_mm"]),
    }
    ply_count, ply = load_ply(result_dir / "weld_points.ply")
    weld_mask = python_labels == 0
    ply_coordinate_error = maximum_error(ply[:, :3], sampled_xyz[weld_mask])
    ply_confidence_error = maximum_error(ply[:, 4], python_confidence[weld_mask])

    cases = ["empty_logits", "wrong_shape", "nan_logits", "no_weld", "unwritable_output"]
    failure_results = []
    for test_case in cases:
        failure = subprocess.run(
            [str(args.failure_probe.resolve()), "--case", test_case],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=120,
        )
        failure_results.append({
            "case": test_case,
            "passed": failure.returncode == 1 and "CPP_POSTPROCESS_PIPELINE_FAILED:" in failure.stderr,
            "exit_code": int(failure.returncode),
            "stdout": failure.stdout,
            "stderr": failure.stderr,
        })
    fail_closed = {
        "status": "PASS" if all(item["passed"] for item in failure_results) else "FAILED",
        "tested_cases": len(failure_results),
        "passed_cases": sum(bool(item["passed"]) for item in failure_results),
        "fallback": "NONE",
        "cases": failure_results,
    }
    (output_dir / "fail_closed_report.json").write_text(
        json.dumps(fail_closed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    coordinate_error = maximum_error(prediction[:, :3], sampled_xyz)
    phase9b_matching = int(np.count_nonzero(cpp_labels == phase9b_labels))
    python_matching = int(np.count_nonzero(cpp_labels == python_labels))
    report = {
        "status": "PASS",
        "sample_id": "weld_65",
        "task_id": result["task_id"],
        "output_files": {path.name: {"exists": True, "bytes": path.stat().st_size} for path in required},
        "contracts": {
            "logits_shape": list(logits.shape),
            "logits_finite": bool(np.isfinite(logits).all()),
            "prediction_shape": list(prediction.shape),
            "coordinate_recovery_max_abs_error": coordinate_error,
        },
        "classification": {
            "phase9b_matching_points": phase9b_matching,
            "phase9b_label_agreement": phase9b_matching / 2048.0,
            "python_matching_points": python_matching,
            "python_label_agreement": python_matching / 2048.0,
            "python_weld_points": int(reference["weld_points"]),
            "cpp_weld_points": int(result["weld_points"]),
        },
        "python_geometry": reference,
        "python_geometry_fp64_diagnostic": reference_fp64,
        "fp32_output_contract_quantization": fp32_quantization,
        "cpp_geometry": {
            "weld_points": int(result["weld_points"]),
            "weld_ratio": float(result["weld_ratio"]),
            "center": result["center"],
            "bbox_min": result["bbox"]["min"],
            "bbox_max": result["bbox"]["max"],
            "length_mm": float(result["length_mm"]),
        },
        "geometry_max_abs_errors": geometry_errors,
        "maximum_geometry_error": max(geometry_errors.values()),
        "ply": {
            "declared_vertices": ply_count,
            "rows": int(ply.shape[0]),
            "all_labels_weld_seam": bool(np.all(np.rint(ply[:, 3]).astype(np.int64) == 0)),
            "coordinate_max_abs_error": ply_coordinate_error,
            "confidence_max_abs_error": ply_confidence_error,
        },
        "fail_closed": fail_closed,
        "runtime_report": json.loads((output_dir / "runtime_report.json").read_text(encoding="utf-8")),
        "engine_sha256": sha256(engine),
        "plugin_sha256": sha256(plugin),
    }
    passed = all([
        report["contracts"]["logits_finite"],
        coordinate_error < 1.0e-6,
        phase9b_matching == 2048,
        python_matching == 2048,
        int(result["weld_points"]) == int(reference["weld_points"]),
        max(geometry_errors.values()) < 1.0e-5,
        ply_count == int(reference["weld_points"]),
        ply.shape[0] == int(reference["weld_points"]),
        report["ply"]["all_labels_weld_seam"],
        ply_coordinate_error < 1.0e-6,
        ply_confidence_error < 1.0e-6,
        fail_closed["status"] == "PASS",
    ])
    report["status"] = "PASS" if passed else "FAILED"
    (output_dir / "postprocess_parity.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        print("CPP_POSTPROCESS_PIPELINE_BLOCKED", file=sys.stderr)
        return 1
    print("CPP_POSTPROCESS_PIPELINE_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
