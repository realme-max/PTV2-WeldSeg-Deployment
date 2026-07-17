"""Validate the Phase 9A C++ TensorRT runtime against the existing Python production runner."""

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
PYTHON = PROJECT_ROOT / ".venv_ptv2" / "Scripts" / "python.exe"
BASELINE_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260717_173128_144483_phase8d_production_baseline"
)
MANIFEST = BASELINE_ROOT / "package" / "manifests" / "deployment_manifest.json"
SAMPLE = BASELINE_ROOT / "validation_inputs" / "00_weld_65.npz"
POINTS_NPY = BASELINE_ROOT / "input_arrays" / "weld_65_points.npy"
ADJ_NPY = BASELINE_ROOT / "input_arrays" / "weld_65_adj.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
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


def run_expect_failure(command: list[str], label: str) -> dict[str, object]:
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=120)
    return {
        "case": label,
        "passed": result.returncode != 0,
        "exit_code": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float | list[list[int]]]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    for truth, pred in zip(labels.reshape(-1), prediction.reshape(-1), strict=True):
        confusion[int(truth), int(pred)] += 1
    ious = []
    for cls in range(2):
        tp = int(confusion[cls, cls])
        fp = int(confusion[:, cls].sum() - tp)
        fn = int(confusion[cls, :].sum() - tp)
        denominator = tp + fp + fn
        ious.append(tp / denominator if denominator else 0.0)
    tp = int(confusion[0, 0])
    fp = int(confusion[1, 0])
    fn = int(confusion[0, 1])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "confusion_matrix": confusion.tolist(),
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "weld_seam_iou": float(ious[0]),
        "background_iou": float(ious[1]),
        "miou": float(np.mean(ious)),
        "weld_seam_precision": float(precision),
        "weld_seam_recall": float(recall),
        "weld_seam_f1": float(f1),
    }


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (args.output_dir or (
        PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt" / f"{timestamp}_phase9a_cpp_runtime"
    )).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    package = MANIFEST.parent.parent
    engine = (package / manifest["artifacts"]["engine"]).resolve()
    plugin = (package / manifest["artifacts"]["plugin"]).resolve()
    if sha256(engine) != manifest["engine_sha256"] or sha256(plugin) != manifest["plugin_sha256"]:
        raise RuntimeError("Phase 8D production Engine/Plugin hash validation failed")

    points = np.ascontiguousarray(np.load(POINTS_NPY, allow_pickle=False), dtype=np.float32)
    adj = np.ascontiguousarray(np.load(ADJ_NPY, allow_pickle=False), dtype=np.float32)
    with np.load(SAMPLE, allow_pickle=False) as sample:
        labels = np.asarray(sample["labels"], dtype=np.int64)
    if points.shape != (1, 2048, 4) or adj.shape != (1, 2048, 2048) or labels.size != 2048:
        raise RuntimeError("Frozen weld_65 input contract mismatch")

    points_bin = output_dir / "weld_65_points.bin"
    adj_bin = output_dir / "weld_65_adj.bin"
    python_logits_bin = output_dir / "python_logits.bin"
    cpp_logits_bin = output_dir / "cpp_logits.bin"
    points.tofile(points_bin)
    adj.tofile(adj_bin)

    python_logits_npy = output_dir / "python_logits.npy"
    run(
        [
            str(PYTHON),
            str(PROJECT_ROOT / "scripts" / "run_gcn_res_tensorrt_production.py"),
            "--manifest", str(MANIFEST),
            "--points", str(POINTS_NPY),
            "--adj", str(ADJ_NPY),
            "--output", str(python_logits_npy),
        ],
        output_dir / "python_runtime.log",
    )
    python_logits = np.ascontiguousarray(np.load(python_logits_npy, allow_pickle=False), dtype=np.float32)
    python_logits.tofile(python_logits_bin)

    runtime_json = output_dir / "cpp_runtime_summary.json"
    benchmark_json = output_dir / "cpp_runtime_benchmark.json"
    run(
        [
            str(args.exe.resolve()),
            "--engine", str(engine),
            "--plugin", str(plugin),
            "--engine-sha256", manifest["engine_sha256"],
            "--points", str(points_bin),
            "--adj", str(adj_bin),
            "--output", str(cpp_logits_bin),
            "--runtime-json", str(runtime_json),
            "--benchmark-json", str(benchmark_json),
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
        ],
        output_dir / "cpp_runtime.log",
    )

    base_command = [
        str(args.exe.resolve()),
        "--engine", str(engine),
        "--plugin", str(plugin),
        "--engine-sha256", manifest["engine_sha256"],
        "--points", str(points_bin),
        "--adj", str(adj_bin),
        "--output", str(output_dir / "negative_should_not_exist.bin"),
    ]
    invalid_points = output_dir / "invalid_points_size.bin"
    points.reshape(-1)[:-1].tofile(invalid_points)
    invalid_plugin = args.exe.resolve().parent / "nvinfer_11.dll"
    negative_commands = [
        ("engine_missing", base_command[:2] + [str(output_dir / "missing.plan")] + base_command[3:]),
        ("plugin_missing", base_command[:4] + [str(output_dir / "missing.dll")] + base_command[5:]),
        ("plugin_load_failed", base_command[:4] + [str(invalid_plugin)] + base_command[5:]),
        (
            "engine_hash_mismatch",
            base_command[:6] + ["0" * 64] + base_command[7:],
        ),
        (
            "input_size_mismatch",
            base_command[:8] + [str(invalid_points)] + base_command[9:],
        ),
    ]
    negative_results = [run_expect_failure(command, label) for label, command in negative_commands]
    negative_report = {
        "status": "PASS" if all(item["passed"] for item in negative_results) else "FAILED",
        "tested_cases": len(negative_results),
        "passed_cases": sum(bool(item["passed"]) for item in negative_results),
        "cases": negative_results,
        "cuda_error_handling": "Implemented at every CUDA API call; no synthetic GPU fault was injected.",
        "fallback": "NONE",
    }
    (output_dir / "negative_test_report.json").write_text(
        json.dumps(negative_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if negative_report["status"] != "PASS":
        raise RuntimeError("One or more C++ fail-closed negative paths did not fail")

    cpp_logits = np.fromfile(cpp_logits_bin, dtype=np.float32).reshape(1, 2048, 2)
    difference = np.abs(cpp_logits - python_logits)
    python_prediction = np.argmax(python_logits, axis=-1)
    cpp_prediction = np.argmax(cpp_logits, axis=-1)
    matching = int(np.count_nonzero(python_prediction == cpp_prediction))
    python_metrics = metrics(labels, python_prediction)
    cpp_metrics = metrics(labels, cpp_prediction)
    report = {
        "status": "PASS" if matching == 2048 and np.isfinite(cpp_logits).all() else "FAILED",
        "deployment_id": manifest["deployment_id"],
        "sample_id": "weld_65",
        "engine_sha256": sha256(engine),
        "plugin_sha256": sha256(plugin),
        "points_shape": list(points.shape),
        "adj_shape": list(adj.shape),
        "logits_shape": list(cpp_logits.shape),
        "python_finite": bool(np.isfinite(python_logits).all()),
        "cpp_finite": bool(np.isfinite(cpp_logits).all()),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "matching_points": matching,
        "total_points": 2048,
        "label_agreement": matching / 2048.0,
        "python_metrics": python_metrics,
        "cpp_metrics": cpp_metrics,
        "miou_delta": float(cpp_metrics["miou"] - python_metrics["miou"]),
        "weld_seam_f1_delta": float(cpp_metrics["weld_seam_f1"] - python_metrics["weld_seam_f1"]),
        "python_logits_bin": str(python_logits_bin),
        "cpp_logits_bin": str(cpp_logits_bin),
        "cpp_runtime_benchmark": str(benchmark_json),
        "negative_tests": negative_report,
    }
    (output_dir / "runtime_compare.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if report["status"] != "PASS":
        raise RuntimeError("C++ vs Python TensorRT task-level parity failed")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("TENSORRT_CPP_RUNTIME_MINIMAL_INFERENCE_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
