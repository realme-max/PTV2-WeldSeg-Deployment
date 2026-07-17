"""Validate the Phase 9B TXT-to-TensorRT C++ pipeline against NumPy/sklearn and Python TensorRT."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.neighbors import kneighbors_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv_ptv2" / "Scripts" / "python.exe"
BASELINE_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260717_173128_144483_phase8d_production_baseline"
)
MANIFEST = BASELINE_ROOT / "package" / "manifests" / "deployment_manifest.json"
CLOUD = PROJECT_ROOT / "data" / "weld" / "000001" / "weld_65.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], log_path: Path) -> None:
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=1800)
    log_path.write_text(
        f"COMMAND: {subprocess.list2cmdline(command)}\n\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {log_path}")


def run_expect_failure(command: list[str], case: str) -> dict[str, object]:
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=120)
    return {
        "case": case,
        "passed": result.returncode != 0,
        "exit_code": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def cpp_command(exe: Path, engine: Path, plugin: Path, cloud: Path, output_dir: Path) -> list[str]:
    return [
        str(exe),
        "--cloud", str(cloud),
        "--engine", str(engine),
        "--plugin", str(plugin),
        "--output", str(output_dir / "prediction.txt"),
        "--report", str(output_dir / "runtime_report.json"),
        "--logits-output", str(output_dir / "cpp_logits.bin"),
        "--points-output", str(output_dir / "cpp_points.bin"),
        "--adj-output", str(output_dir / "cpp_adj.bin"),
        "--sample-indices-output", str(output_dir / "sample_indices.bin"),
        "--seed", "42",
    ]


def segmentation_metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    for truth, pred in zip(labels.reshape(-1), prediction.reshape(-1), strict=True):
        confusion[int(truth), int(pred)] += 1
    iou = []
    for cls in range(2):
        tp = int(confusion[cls, cls])
        fp = int(confusion[:, cls].sum() - tp)
        fn = int(confusion[cls, :].sum() - tp)
        denominator = tp + fp + fn
        iou.append(tp / denominator if denominator else 0.0)
    tp = int(confusion[0, 0])
    fp = int(confusion[1, 0])
    fn = int(confusion[0, 1])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "label_semantics": {"0": "weld_seam", "1": "background"},
        "confusion_matrix": confusion.tolist(),
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "weld_seam_iou": float(iou[0]),
        "background_iou": float(iou[1]),
        "miou": float(np.mean(iou)),
        "weld_seam_precision": float(precision),
        "weld_seam_recall": float(recall),
        "weld_seam_f1": float(f1),
    }


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (args.output_dir or (
        PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt" / f"{timestamp}_phase9b_cpp_pointcloud_pipeline"
    )).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    package = MANIFEST.parent.parent
    engine = (package / manifest["artifacts"]["engine"]).resolve()
    plugin = (package / manifest["artifacts"]["plugin"]).resolve()
    if sha256(engine) != manifest["engine_sha256"] or sha256(plugin) != manifest["plugin_sha256"]:
        raise RuntimeError("Production package hash check failed")

    run1 = output_dir / "cpp_run_1"
    run2 = output_dir / "cpp_run_2"
    run1.mkdir()
    run2.mkdir()
    run(cpp_command(args.exe.resolve(), engine, plugin, CLOUD, run1), output_dir / "cpp_run_1.log")
    run(cpp_command(args.exe.resolve(), engine, plugin, CLOUD, run2), output_dir / "cpp_run_2.log")

    raw = np.loadtxt(CLOUD, dtype=np.float32)
    if raw.shape != (2048, 4) or not np.isfinite(raw).all():
        raise RuntimeError(f"Unexpected weld_65 TXT contract: {raw.shape}")
    indices1 = np.fromfile(run1 / "sample_indices.bin", dtype=np.uint64).astype(np.int64)
    indices2 = np.fromfile(run2 / "sample_indices.bin", dtype=np.uint64).astype(np.int64)
    if indices1.shape != (2048,) or np.unique(indices1).size != 2048:
        raise RuntimeError("C++ sample indices are not a 2048-element sample without replacement")

    xyz64 = raw[:, :3].astype(np.float64)
    centroid = xyz64.sum(axis=0) / len(xyz64)
    centered = xyz64 - centroid
    radius = np.sqrt(np.sum(centered * centered, axis=1)).max()
    normalized = centered / radius
    python_points = np.ascontiguousarray(
        np.concatenate(
            [normalized[indices1].astype(np.float32), np.ones((2048, 1), dtype=np.float32)], axis=1
        )[None, ...],
        dtype=np.float32,
    )
    python_adj = np.ascontiguousarray(
        kneighbors_graph(
            raw[indices1, :3], n_neighbors=6, mode="connectivity", include_self=False
        ).toarray().astype(np.float32)[None, ...],
        dtype=np.float32,
    )
    cpp_points = np.fromfile(run1 / "cpp_points.bin", dtype=np.float32).reshape(1, 2048, 4)
    cpp_adj = np.fromfile(run1 / "cpp_adj.bin", dtype=np.float32).reshape(1, 2048, 2048)
    points_difference = np.abs(cpp_points - python_points)
    adjacency_mismatches = int(np.count_nonzero(cpp_adj != python_adj))

    points_npy = output_dir / "python_points.npy"
    adj_npy = output_dir / "python_adj.npy"
    np.save(points_npy, python_points, allow_pickle=False)
    np.save(adj_npy, python_adj, allow_pickle=False)
    python_logits_npy = output_dir / "python_logits.npy"
    run(
        [
            str(PYTHON),
            str(PROJECT_ROOT / "scripts" / "run_gcn_res_tensorrt_production.py"),
            "--manifest", str(MANIFEST),
            "--points", str(points_npy),
            "--adj", str(adj_npy),
            "--output", str(python_logits_npy),
        ],
        output_dir / "python_tensorrt.log",
    )
    python_logits = np.ascontiguousarray(np.load(python_logits_npy, allow_pickle=False), dtype=np.float32)
    cpp_logits = np.fromfile(run1 / "cpp_logits.bin", dtype=np.float32).reshape(1, 2048, 2)
    cpp_logits_second = np.fromfile(run2 / "cpp_logits.bin", dtype=np.float32).reshape(1, 2048, 2)
    logits_difference = np.abs(cpp_logits - python_logits)
    repeat_difference = np.abs(cpp_logits - cpp_logits_second)
    python_prediction = np.argmax(python_logits, axis=-1)
    cpp_prediction = np.argmax(cpp_logits, axis=-1)
    cpp_prediction_second = np.argmax(cpp_logits_second, axis=-1)
    sampled_labels = np.rint(raw[indices1, 3]).astype(np.int64)[None, ...]
    matching = int(np.count_nonzero(cpp_prediction == python_prediction))
    metrics_python = segmentation_metrics(sampled_labels, python_prediction)
    metrics_cpp = segmentation_metrics(sampled_labels, cpp_prediction)

    invalid_dir = output_dir / "negative_inputs"
    invalid_dir.mkdir()
    too_small = invalid_dir / "too_small.txt"
    malformed = invalid_dir / "malformed.txt"
    nan_cloud = invalid_dir / "nan.txt"
    np.savetxt(too_small, raw[:2047], fmt="%.9g")
    malformed.write_text("1.0 2.0 3.0\n", encoding="utf-8")
    nan_data = raw.copy()
    nan_data[0, 0] = np.nan
    np.savetxt(nan_cloud, nan_data, fmt="%.9g")

    base = cpp_command(args.exe.resolve(), engine, plugin, CLOUD, output_dir / "negative_output")
    cloud_position = base.index("--cloud") + 1
    engine_position = base.index("--engine") + 1
    plugin_position = base.index("--plugin") + 1
    negative_commands: list[tuple[str, list[str]]] = []
    for case, position, value in (
        ("missing_txt", cloud_position, output_dir / "missing.txt"),
        ("point_count_below_2048", cloud_position, too_small),
        ("malformed_txt", cloud_position, malformed),
        ("nan_coordinate", cloud_position, nan_cloud),
        ("missing_engine", engine_position, output_dir / "missing.plan"),
        ("missing_plugin", plugin_position, output_dir / "missing.dll"),
    ):
        command = base.copy()
        command[position] = str(value)
        negative_commands.append((case, command))
    negative_results = [run_expect_failure(command, case) for case, command in negative_commands]
    negative_report = {
        "status": "PASS" if all(item["passed"] for item in negative_results) else "FAILED",
        "tested_cases": len(negative_results),
        "passed_cases": sum(bool(item["passed"]) for item in negative_results),
        "fallback": "NONE",
        "cases": negative_results,
    }
    (output_dir / "negative_test_report.json").write_text(
        json.dumps(negative_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = {
        "status": "PASS",
        "deployment_id": manifest["deployment_id"],
        "sample_id": "weld_65",
        "cloud_path": str(CLOUD.resolve()),
        "input_points": int(raw.shape[0]),
        "sample_points": 2048,
        "sampling": {"algorithm": "std::mt19937 + std::shuffle", "seed": 42, "replacement": False},
        "sampling_indices_repeat_exact": bool(np.array_equal(indices1, indices2)),
        "sampling_unique_indices": int(np.unique(indices1).size),
        "points_shape": list(cpp_points.shape),
        "adj_shape": list(cpp_adj.shape),
        "logits_shape": list(cpp_logits.shape),
        "preprocessing_parity": {
            "points_exact": bool(np.array_equal(cpp_points, python_points)),
            "points_max_abs_error": float(points_difference.max()),
            "adjacency_exact": adjacency_mismatches == 0,
            "adjacency_mismatches": adjacency_mismatches,
        },
        "runtime_parity": {
            "python_finite": bool(np.isfinite(python_logits).all()),
            "cpp_finite": bool(np.isfinite(cpp_logits).all()),
            "max_abs_error": float(logits_difference.max()),
            "mean_abs_error": float(logits_difference.mean()),
            "matching_points": matching,
            "total_points": 2048,
            "label_agreement": matching / 2048.0,
            "cpp_repeat_logits_max_abs_error": float(repeat_difference.max()),
            "cpp_repeat_labels_exact": bool(np.array_equal(cpp_prediction, cpp_prediction_second)),
        },
        "python_metrics": metrics_python,
        "cpp_metrics": metrics_cpp,
        "miou_delta": float(metrics_cpp["miou"] - metrics_python["miou"]),
        "weld_seam_f1_delta": float(metrics_cpp["weld_seam_f1"] - metrics_python["weld_seam_f1"]),
        "negative_tests": negative_report,
        "runtime_report": json.loads((run1 / "runtime_report.json").read_text(encoding="utf-8")),
        "engine_sha256": sha256(engine),
        "plugin_sha256": sha256(plugin),
    }
    passed = all(
        [
            report["sampling_indices_repeat_exact"],
            report["preprocessing_parity"]["points_exact"],
            report["preprocessing_parity"]["adjacency_exact"],
            report["runtime_parity"]["python_finite"],
            report["runtime_parity"]["cpp_finite"],
            report["runtime_parity"]["label_agreement"] == 1.0,
            report["runtime_parity"]["cpp_repeat_labels_exact"],
            report["miou_delta"] == 0.0,
            report["weld_seam_f1_delta"] == 0.0,
            negative_report["status"] == "PASS",
        ]
    )
    report["status"] = "PASS" if passed else "FAILED"
    (output_dir / "pipeline_parity.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not passed:
        print("CPP_PIPELINE_PARITY_FAILED", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("CPP_POINTCLOUD_PIPELINE_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
