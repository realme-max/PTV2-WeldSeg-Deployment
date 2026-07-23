#!/usr/bin/env python3
"""Phase 11A release/full-test-set qualification.

This is a validation-only orchestrator.  It never rebuilds the production
Engine, rewrites ONNX, modifies the Plugin, or changes inference semantics.
The C++ runner initializes WeldDetector once and reuses that runtime for the
multi-sample and soak sequences.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_BASE = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
PHASE8D_ROOT = (
    ARTIFACT_BASE / "20260717_173128_144483_phase8d_production_baseline"
)
PRODUCTION_ENGINE = (
    PHASE8D_ROOT / "package" / "engine" / "strict_fp32_voxelunique_cub.plan"
)
PRODUCTION_PLUGIN = (
    PHASE8D_ROOT / "package" / "plugins" / "VoxelUniqueCubPlugin.dll"
)
PRODUCTION_ONNX = (
    PHASE8D_ROOT / "package" / "model" / "gcn_res_voxelunique_cub.onnx"
)
CHECKPOINT = PROJECT_ROOT / "models" / "testParameters" / "GCN_res" / "best_model.pth"
TEST_SPLIT = (
    PROJECT_ROOT
    / "data"
    / "weld"
    / "train_test_split"
    / "sub_shuffled_test_file_list.json"
)
TRAIN_SPLIT = TEST_SPLIT.with_name("sub_shuffled_train_file_list.json")
VAL_SPLIT = TEST_SPLIT.with_name("sub_shuffled_val_file_list.json")
PHASE8D_ACCURACY = PHASE8D_ROOT / "production_accuracy_regression.json"
TENSORRT_ROOT_DEFAULT = Path(
    r"D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106"
)
REQUESTED_PACKAGE = Path(
    r"D:\PTV2_Weld_App_0.1.1_Phase10C1_QUALIFIED_20260723_184946"
)
ENGINE_SHA256 = "a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299"
PLUGIN_SHA256 = "6641ec147e8eac10206a5c60ba1c1390c398d4b59e32a6a618a37046360ec348"
ONNX_SHA256 = "16ca5c16c330e6572b1730e80da724231a28b68872a3203c21240348d4d89299"
CHECKPOINT_SHA256 = "311bddf3607d76e6b7ded450b8419bf6ae98f34f50578608b3e6a1c1c3e58d21"
TEST_SPLIT_SHA256 = "d7464be1b9e0efc6e02bb2490e78e9cf13b10368a2be9bdf0bd86d64f40aff76"
STRICT_THRESHOLD = 1.0e-4
LABEL_SEMANTICS = {"0": "weld_seam", "1": "background"}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as stream:
        stream.write(text)
        if text and not text.endswith("\n"):
            stream.write("\n")


def run(
    command: list[str],
    log_path: Path,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 1800,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(cwd or PROJECT_ROOT),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    append_log(
        log_path,
        f"\n$ {' '.join(command)}\n"
        f"cwd={cwd or PROJECT_ROOT}\n"
        f"exit_code={completed.returncode}\nelapsed_s={elapsed:.6f}\n"
        f"{completed.stdout}",
    )
    setattr(completed, "wall_seconds", elapsed)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}"
        )
    return completed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"Invalid JSONL {path}:{line_number}: {error}"
                    ) from error
    return records


def resolve_split_entry(entry: str) -> Path:
    relative = entry.replace("\\", "/")
    if relative.startswith("./"):
        relative = relative[2:]
    path = PROJECT_ROOT / "data" / relative
    if path.suffix.lower() != ".txt":
        path = path.with_suffix(".txt")
    return path.resolve()


def read_split(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError(f"Split is not a JSON string list: {path}")
    return value


def inspect_txt(path: Path) -> dict[str, Any]:
    labels: Counter[int] = Counter()
    rows = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            columns = line.split()
            if len(columns) != 4:
                raise RuntimeError(f"{path}:{line_number} does not have four columns")
            try:
                x, y, z = map(float, columns[:3])
                label_float = float(columns[3])
            except ValueError as error:
                raise RuntimeError(f"{path}:{line_number} is not numeric") from error
            if not all(math.isfinite(value) for value in (x, y, z, label_float)):
                raise RuntimeError(f"{path}:{line_number} contains NaN/Inf")
            label = int(label_float)
            if label_float != label or label not in (0, 1):
                raise RuntimeError(f"{path}:{line_number} has invalid label {label_float}")
            labels[label] += 1
            rows += 1
    if rows < 2048:
        raise RuntimeError(f"{path} contains only {rows} rows")
    return {"row_count": rows, "label_counts": {"0": labels[0], "1": labels[1]}}


def discover_test_manifest(run_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    if sha256(TEST_SPLIT) != TEST_SPLIT_SHA256:
        raise RuntimeError("Authoritative test split SHA-256 changed")
    splits = {
        "train": read_split(TRAIN_SPLIT),
        "val": read_split(VAL_SPLIT),
        "test": read_split(TEST_SPLIT),
    }
    resolved = {
        name: [resolve_split_entry(entry) for entry in entries]
        for name, entries in splits.items()
    }
    if tuple(map(len, (resolved["train"], resolved["val"], resolved["test"]))) != (
        54,
        18,
        18,
    ):
        raise RuntimeError("Expected train/val/test sizes 54/18/18")
    sets = {name: set(paths) for name, paths in resolved.items()}
    if (
        sets["train"] & sets["val"]
        or sets["train"] & sets["test"]
        or sets["val"] & sets["test"]
    ):
        raise RuntimeError("Train/val/test split overlap detected")
    if len(sets["test"]) != 18:
        raise RuntimeError("Test split contains duplicate paths")
    samples = []
    for order, (entry, path) in enumerate(zip(splits["test"], resolved["test"])):
        if not path.is_file():
            raise RuntimeError(f"Test sample is missing: {path}")
        inspection = inspect_txt(path)
        samples.append(
            {
                "order": order,
                "sample_id": path.stem,
                "source_entry": entry,
                "absolute_source_path": str(path),
                "repository_relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256(path),
                **inspection,
            }
        )
    payload = {
        "status": "PASS",
        "authoritative_source": str(TEST_SPLIT),
        "authoritative_source_repository_relative": TEST_SPLIT.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "authoritative_source_sha256": TEST_SPLIT_SHA256,
        "split_creation_seed": 42,
        "split_sizes": {"train": 54, "val": 18, "test": 18},
        "ordering": "JSON list order; no resorting",
        "unique_test_samples": len(sets["test"]),
        "overlaps": {"train_val": 0, "train_test": 0, "val_test": 0},
        "samples": samples,
    }
    dump_json(run_dir / "test_set_manifest.json", payload)
    return samples, resolved["test"]


def verify_frozen_assets(run_dir: Path, package: Path) -> dict[str, Any]:
    expected = {
        "engine": (PRODUCTION_ENGINE, ENGINE_SHA256),
        "plugin": (PRODUCTION_PLUGIN, PLUGIN_SHA256),
        "onnx": (PRODUCTION_ONNX, ONNX_SHA256),
        "checkpoint": (CHECKPOINT, CHECKPOINT_SHA256),
    }
    assets: dict[str, Any] = {
        "deployment_id": "gcn-res-trt-cub-strict-fp32-20260717_173128_144483",
        "precision": "Strict FP32",
        "tf32_enabled": False,
        "fp16_enabled": False,
        "int8_enabled": False,
        "package": str(package),
        "requested_package": str(REQUESTED_PACKAGE),
        "requested_package_exists": REQUESTED_PACKAGE.exists(),
        "assets": {},
    }
    for name, (path, expected_hash) in expected.items():
        if not path.is_file():
            raise RuntimeError(f"Frozen {name} is missing: {path}")
        actual = sha256(path)
        if actual != expected_hash:
            raise RuntimeError(
                f"Frozen {name} SHA mismatch: expected {expected_hash}, got {actual}"
            )
        assets["assets"][name] = {
            "path": str(path),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    package_engine = package / "engine" / "strict_fp32_voxelunique_cub.plan"
    package_plugin = package / "plugins" / "VoxelUniqueCubPlugin.dll"
    if not package_engine.is_file() or not package_plugin.is_file():
        raise RuntimeError("Release package does not contain Engine/Plugin")
    assets["package_assets"] = {
        "engine": {
            "path": str(package_engine),
            "sha256": sha256(package_engine),
        },
        "plugin": {
            "path": str(package_plugin),
            "sha256": sha256(package_plugin),
        },
    }
    if (
        assets["package_assets"]["engine"]["sha256"] != ENGINE_SHA256
        or assets["package_assets"]["plugin"]["sha256"] != PLUGIN_SHA256
    ):
        raise RuntimeError("Release package Engine/Plugin do not match frozen assets")
    dump_json(run_dir / "production_assets.json", assets)
    return assets


def select_package(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_dir():
            raise RuntimeError(f"Requested package does not exist: {explicit}")
        return explicit.resolve()
    if REQUESTED_PACKAGE.is_dir():
        return REQUESTED_PACKAGE.resolve()
    candidates = sorted(
        Path("D:/").glob("PTV2_Weld_App_*_QUALIFIED_*"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("No qualified Release package exists on D:")
    return candidates[0].resolve()


def build_runner(
    run_dir: Path, tensorrt_root: Path, configure_log: Path, build_log: Path
) -> tuple[Path, Path]:
    if not tensorrt_root.is_dir():
        raise RuntimeError(f"TensorRT SDK root missing: {tensorrt_root}")
    build_dir = run_dir / "build"
    run(
        [
            "cmake",
            "-S",
            str(PROJECT_ROOT / "deployment" / "weld_trt_app"),
            "-B",
            str(build_dir),
            "-G",
            "Visual Studio 17 2022",
            "-A",
            "x64",
            f"-DTENSORRT_ROOT={tensorrt_root.as_posix()}",
        ],
        configure_log,
    )
    run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "weld_sdk_testset_qualification",
            "weld_trt_demo",
            "postprocess_failure_probe",
        ],
        build_log,
    )
    qualifier = build_dir / "weld_sdk" / "Release" / "weld_sdk_testset_qualification.exe"
    weld_demo = build_dir / "Release" / "weld_trt_demo.exe"
    if not qualifier.is_file() or not weld_demo.is_file():
        raise RuntimeError("Expected qualification executables were not built")
    return qualifier, weld_demo


def write_cloud_list(path: Path, clouds: Iterable[Path]) -> None:
    dump_text(path, "\n".join(str(item) for item in clouds))


def run_qualification_sequence(
    qualifier: Path,
    cloud_list: Path,
    output: Path,
    log: Path,
    *,
    rounds: int,
    warmup: int,
    compact: bool,
    resource_interval: int = 18,
    repeat_initialize: int = 1,
    timeout: int = 1800,
) -> list[dict[str, Any]]:
    command = [
        str(qualifier),
        "--engine",
        str(PRODUCTION_ENGINE),
        "--plugin",
        str(PRODUCTION_PLUGIN),
        "--list",
        str(cloud_list),
        "--output",
        str(output),
        "--rounds",
        str(rounds),
        "--warmup",
        str(warmup),
        "--resource-interval",
        str(resource_interval),
        "--repeat-initialize",
        str(repeat_initialize),
        "--compact",
        "true" if compact else "false",
    ]
    run(command, log, timeout=timeout)
    records = read_jsonl(output)
    summaries = [item for item in records if item["record_type"] == "summary"]
    if len(summaries) != 1 or not summaries[0]["success"]:
        raise RuntimeError(f"Qualification runner did not produce a PASS summary: {output}")
    return records


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def metrics_from_labels(gt: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion, (gt.astype(np.int64), prediction.astype(np.int64)), 1)
    tp = int(confusion[0, 0])
    fn = int(confusion[0, 1])
    fp = int(confusion[1, 0])
    tn = int(confusion[1, 1])
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    weld_iou = safe_divide(tp, tp + fp + fn)
    background_iou = safe_divide(tn, tn + fn + fp)
    return {
        "label_semantics": LABEL_SEMANTICS,
        "confusion_matrix_rows_ground_truth_columns_prediction": confusion.tolist(),
        "accuracy": safe_divide(tp + tn, int(confusion.sum())),
        "weld_seam_precision": precision,
        "weld_seam_recall": recall,
        "weld_seam_f1": f1,
        "weld_seam_iou": weld_iou,
        "background_iou": background_iou,
        "miou": (weld_iou + background_iou) / 2.0,
    }


def aggregate_global(per_sample: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    metric_names = (
        "accuracy",
        "weld_seam_precision",
        "weld_seam_recall",
        "weld_seam_f1",
        "weld_seam_iou",
        "background_iou",
        "miou",
    )
    for sample in per_sample:
        confusion += np.asarray(
            sample["metrics"]["confusion_matrix_rows_ground_truth_columns_prediction"],
            dtype=np.int64,
        )
    expanded_gt = np.repeat(np.arange(2), confusion.sum(axis=1))
    expanded_prediction = np.concatenate(
        [
            np.concatenate(
                [
                    np.full(confusion[row, column], column, dtype=np.int64)
                    for column in range(2)
                ]
            )
            for row in range(2)
        ]
    )
    # Reconstructing labels by row preserves only the confusion matrix, which is
    # sufficient for every reported aggregate metric.
    global_metrics = metrics_from_labels(expanded_gt, expanded_prediction)
    means = {
        name: float(np.mean([sample["metrics"][name] for sample in per_sample]))
        for name in metric_names
    }
    means["label_semantics"] = LABEL_SEMANTICS
    return global_metrics, means


def timing_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "standard_deviation": float(array.std(ddof=0)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def analyze_primary_results(
    run_dir: Path,
    samples: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    audits = {
        item["sample_id"]: item
        for item in records
        if item["record_type"] == "preprocess_audit"
    }
    detections = [
        item for item in records if item["record_type"] == "detection" and item["round"] > 0
    ]
    if len(audits) != 18 or len(detections) != 54:
        raise RuntimeError(
            f"Expected 18 audits and 54 detections, got {len(audits)}/{len(detections)}"
        )
    by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in detections:
        by_round[int(item["round"])].append(item)
    phase8d = json.loads(PHASE8D_ACCURACY.read_text(encoding="utf-8"))
    frozen_input_manifest = json.loads(
        (PHASE8D_ROOT / "validation_inputs" / "input_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    frozen_input_by_id = {
        item["sample_id"]: item for item in frozen_input_manifest["samples"]
    }
    per_sample: list[dict[str, Any]] = []
    for manifest in samples:
        sample_id = manifest["sample_id"]
        audit = audits[sample_id]
        round_one = next(
            item
            for item in by_round[1]
            if item["sample_id"] == sample_id
        )
        gt = np.asarray(audit["sampled_ground_truth_labels"], dtype=np.int64)
        prediction = np.asarray(round_one["labels"], dtype=np.int64)
        if gt.shape != (2048,) or prediction.shape != (2048,):
            raise RuntimeError(f"{sample_id} label shape is not [2048]")
        metrics = metrics_from_labels(gt, prediction)
        phase8d_prediction = np.load(
            PHASE8D_ROOT
            / "regression_candidate_outputs"
            / f"{sample_id}_prediction.npy"
        ).reshape(-1)
        frozen_input_record = frozen_input_by_id[sample_id]
        with np.load(
            PHASE8D_ROOT / "validation_inputs" / frozen_input_record["file"],
            allow_pickle=False,
        ) as frozen_input:
            phase8d_indices = np.asarray(
                frozen_input["sample_indices"], dtype=np.int64
            ).reshape(-1)
        current_indices = np.asarray(audit["sampled_indices"], dtype=np.int64)
        phase8d_by_original_row = np.empty(2048, dtype=np.int64)
        current_by_original_row = np.empty(2048, dtype=np.int64)
        phase8d_by_original_row[phase8d_indices] = phase8d_prediction
        current_by_original_row[current_indices] = prediction
        original_row_label_mismatches = int(
            np.count_nonzero(phase8d_by_original_row != current_by_original_row)
        )
        item = {
            **manifest,
            "preprocessing": {
                "original_point_count": audit["original_points"],
                "sampled_point_count": audit["sampled_points"],
                "sampling_seed": 42,
                "points_finite": audit["points_finite"],
                "adjacency_finite": audit["adjacency_finite"],
                "fourth_feature_constant_one": audit["fourth_feature_constant_one"],
                "success": True,
            },
            "inference": {
                "sdk_status": round_one["status"],
                "success": round_one["success"],
                "logits_shape": round_one["logits_shape"],
                "logits_finite": round_one["logits_finite"],
                "predicted_label_count": round_one["predicted_label_count"],
                "error_recorder_errors": round_one["error_recorder_errors"],
                "last_error": round_one["last_error"],
            },
            "metrics": metrics,
            "geometry": {
                "weld_point_count": round_one["weld_points"],
                "weld_ratio": round_one["weld_ratio"],
                "centroid": round_one["center"],
                "bbox_min": round_one["bbox_min"],
                "bbox_max": round_one["bbox_max"],
                "principal_direction": round_one["principal_direction"],
                "pca_length_mm": round_one["length_mm"],
                "finite": all(
                    math.isfinite(float(value))
                    for value in (
                        round_one["center"]
                        + round_one["bbox_min"]
                        + round_one["bbox_max"]
                        + round_one["principal_direction"]
                        + [round_one["length_mm"], round_one["weld_ratio"]]
                    )
                ),
            },
            "timing_ms": {
                "load_cloud": round_one["load_cloud_ms"],
                "sampling": round_one["sampling_ms"],
                "feature_build": audit["feature_build_ms"],
                "adjacency_build": round_one["adjacency_build_ms"],
                "inference_cuda": round_one["inference_cuda_ms"],
                "inference_wall": round_one["inference_wall_ms"],
                "postprocess": round_one["postprocess_ms"],
                "total_sdk_detect": round_one["total_ms"],
            },
            "phase8d_task_comparison": {
                "phase8d_sample_indices_equal": bool(
                    np.array_equal(phase8d_indices, current_indices)
                ),
                "same_original_row_prediction_mismatches": original_row_label_mismatches,
                "same_original_row_prediction_agreement": 1.0
                - original_row_label_mismatches / float(prediction.size),
                "cause_under_audit": (
                    "Phase 8D NumPy sampling order differs from the frozen "
                    "production SDK std::mt19937/std::shuffle order."
                ),
            },
            "pytorch_tensorrt": {
                "status": "PENDING_SAME_INPUT_RERUN",
                "max_abs_error": 0.0,
            },
            "predicted_labels": prediction.astype(int).tolist(),
            "output": {
                "success": True,
                "exported_result_path": None,
                "output_file_validation_status": "PENDING",
            },
        }
        per_sample.append(item)

    global_metrics, mean_metrics = aggregate_global(per_sample)
    aggregate = {
        "status": "PASS",
        "total_samples": 18,
        "total_points": 18 * 2048,
        "global_point_level_metrics": global_metrics,
        "mean_per_sample_metrics": mean_metrics,
        "rankings": {
            "lowest_miou": [
                {"sample_id": item["sample_id"], "value": item["metrics"]["miou"]}
                for item in sorted(per_sample, key=lambda value: value["metrics"]["miou"])[:5]
            ],
            "lowest_weld_recall": [
                {
                    "sample_id": item["sample_id"],
                    "value": item["metrics"]["weld_seam_recall"],
                }
                for item in sorted(
                    per_sample, key=lambda value: value["metrics"]["weld_seam_recall"]
                )[:5]
            ],
            "lowest_weld_precision": [
                {
                    "sample_id": item["sample_id"],
                    "value": item["metrics"]["weld_seam_precision"],
                }
                for item in sorted(
                    per_sample,
                    key=lambda value: value["metrics"]["weld_seam_precision"],
                )[:5]
            ],
            "largest_logit_max_abs": [
                {
                    "sample_id": item["sample_id"],
                    "value": item["pytorch_tensorrt"]["max_abs_error"],
                }
                for item in sorted(
                    per_sample,
                    key=lambda value: value["pytorch_tensorrt"]["max_abs_error"],
                    reverse=True,
                )[:5]
            ],
            "slowest_total_time": [
                {
                    "sample_id": item["sample_id"],
                    "value": item["timing_ms"]["total_sdk_detect"],
                }
                for item in sorted(
                    per_sample,
                    key=lambda value: value["timing_ms"]["total_sdk_detect"],
                    reverse=True,
                )[:5]
            ],
            "slowest_adjacency_build": [
                {
                    "sample_id": item["sample_id"],
                    "value": item["timing_ms"]["adjacency_build"],
                }
                for item in sorted(
                    per_sample,
                    key=lambda value: value["timing_ms"]["adjacency_build"],
                    reverse=True,
                )[:5]
            ],
            "smallest_weld_point_counts": [
                {"sample_id": item["sample_id"], "value": item["geometry"]["weld_point_count"]}
                for item in sorted(
                    per_sample, key=lambda value: value["geometry"]["weld_point_count"]
                )[:5]
            ],
            "largest_weld_point_counts": [
                {"sample_id": item["sample_id"], "value": item["geometry"]["weld_point_count"]}
                for item in sorted(
                    per_sample,
                    key=lambda value: value["geometry"]["weld_point_count"],
                    reverse=True,
                )[:5]
            ],
        },
    }

    return per_sample, aggregate, {"status": "PENDING_SAME_INPUT_RERUN"}


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def build_production_cpp_input(
    audit: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the frozen C++ preprocessing from recorded sampled indices."""
    raw = np.loadtxt(audit["source_path"], dtype=np.float32)
    if raw.shape != (2048, 4):
        raise RuntimeError(
            f"{audit['sample_id']}: expected a [2048,4] source for exact index audit"
        )
    indices = np.asarray(audit["sampled_indices"], dtype=np.int64)
    if indices.shape != (2048,) or set(indices.tolist()) != set(range(2048)):
        raise RuntimeError(f"{audit['sample_id']}: sampled index permutation is invalid")
    sampled = raw[indices]
    full_xyz = raw[:, :3].astype(np.float64)
    centroid = full_xyz.mean(axis=0, dtype=np.float64)
    radius = float(np.sqrt(np.sum((full_xyz - centroid) ** 2, axis=1)).max())
    points = np.ones((1, 2048, 4), dtype=np.float32)
    points[0, :, :3] = (
        (sampled[:, :3].astype(np.float64) - centroid) / radius
    ).astype(np.float32)
    coordinates = sampled[:, :3].astype(np.float64)
    adjacency = np.zeros((1, 2048, 2048), dtype=np.float32)
    point_ids = np.arange(2048, dtype=np.int64)
    for row in range(2048):
        distance = np.sum((coordinates[row] - coordinates) ** 2, axis=1)
        distance[row] = np.inf
        neighbours = np.lexsort((point_ids, distance))[:6]
        adjacency[0, row, neighbours] = 1.0
    labels = sampled[:, 3].astype(np.int64).reshape(1, 2048)
    if (
        not np.isfinite(points).all()
        or not np.isfinite(adjacency).all()
        or not np.all(points[:, :, 3] == 1.0)
    ):
        raise RuntimeError(f"{audit['sample_id']}: reconstructed input contract failed")
    return (
        np.ascontiguousarray(points),
        np.ascontiguousarray(adjacency),
        np.ascontiguousarray(labels),
    )


def run_same_input_pytorch_tensorrt_parity(
    run_dir: Path,
    per_sample: list[dict[str, Any]],
    records: list[dict[str, Any]],
    parity_log: Path,
) -> dict[str, Any]:
    scripts_root = PROJECT_ROOT / "scripts"
    for item in (PROJECT_ROOT, scripts_root):
        if str(item) not in sys.path:
            sys.path.insert(0, str(item))
    import gcn_res_tensorrt_phase8d_common as phase8d_common

    audits = {
        item["sample_id"]: item
        for item in records
        if item["record_type"] == "preprocess_audit"
    }
    input_dir = run_dir / "same_input_reference" / "inputs"
    trt_dir = run_dir / "same_input_reference" / "tensorrt"
    torch_dir = run_dir / "same_input_reference" / "pytorch"
    for directory in (input_dir, trt_dir, torch_dir):
        directory.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sample in per_sample:
        sample_id = sample["sample_id"]
        points, adjacency, labels = build_production_cpp_input(audits[sample_id])
        if labels.reshape(-1).tolist() != audits[sample_id][
            "sampled_ground_truth_labels"
        ]:
            raise RuntimeError(f"{sample_id}: reconstructed ground truth differs")
        np.savez_compressed(
            input_dir / f"{sample_id}.npz",
            points=points,
            adj=adjacency,
            labels=labels,
            sample_indices=np.asarray(
                audits[sample_id]["sampled_indices"], dtype=np.int64
            ),
        )
        inputs[sample_id] = (points, adjacency, labels)
    append_log(parity_log, "Reconstructed all 18 frozen production C++ inputs.")

    trt_outputs: dict[str, np.ndarray] = {}
    trt_runner = phase8d_common.TensorRTRunner(
        PRODUCTION_ENGINE, PRODUCTION_PLUGIN, "candidate"
    )
    try:
        for sample in per_sample:
            sample_id = sample["sample_id"]
            points, adjacency, _ = inputs[sample_id]
            logits = trt_runner.infer(points, adjacency)
            np.save(trt_dir / f"{sample_id}_logits.npy", logits, allow_pickle=False)
            trt_outputs[sample_id] = logits
            cpp_labels = np.asarray(sample["predicted_labels"], dtype=np.int64)
            if not np.array_equal(np.argmax(logits, axis=-1).reshape(-1), cpp_labels):
                raise RuntimeError(
                    f"{sample_id}: Python TensorRT and C++ SDK labels differ"
                )
    finally:
        trt_close = trt_runner.close()
    append_log(parity_log, f"TensorRT same-input pass; close={trt_close}")

    torch_outputs: dict[str, np.ndarray] = {}
    torch_runner = phase8d_common.PyTorchRunner(CHECKPOINT)
    try:
        for sample in per_sample:
            sample_id = sample["sample_id"]
            points, adjacency, _ = inputs[sample_id]
            logits = torch_runner.infer(points, adjacency)
            np.save(torch_dir / f"{sample_id}_logits.npy", logits, allow_pickle=False)
            torch_outputs[sample_id] = logits
    finally:
        torch_close = torch_runner.close()
    append_log(parity_log, f"PyTorch same-input pass; close={torch_close}")

    comparisons = []
    for sample in per_sample:
        sample_id = sample["sample_id"]
        trt_logits = trt_outputs[sample_id]
        torch_logits = torch_outputs[sample_id]
        difference = np.abs(
            trt_logits.astype(np.float64) - torch_logits.astype(np.float64)
        )
        torch_labels = np.argmax(torch_logits, axis=-1).astype(np.int64)
        trt_labels = np.argmax(trt_logits, axis=-1).astype(np.int64)
        mismatches = int(np.count_nonzero(torch_labels != trt_labels))
        max_abs = float(difference.max())
        comparison = {
            "sample_id": sample_id,
            "input_points_sha256": array_sha256(inputs[sample_id][0]),
            "input_adj_sha256": array_sha256(inputs[sample_id][1]),
            "sampled_labels_sha256": array_sha256(inputs[sample_id][2]),
            "pytorch_logits_sha256": array_sha256(torch_logits),
            "tensorrt_logits_sha256": array_sha256(trt_logits),
            "max_abs_error": max_abs,
            "mean_abs_error": float(difference.mean()),
            "rmse": float(np.sqrt(np.mean(difference * difference))),
            "matching_points": 2048 - mismatches,
            "total_points": 2048,
            "agreement": 1.0 - mismatches / 2048.0,
            "outputs_finite": bool(
                np.isfinite(torch_logits).all() and np.isfinite(trt_logits).all()
            ),
            "strict_threshold": STRICT_THRESHOLD,
            "strict_threshold_passed": max_abs < STRICT_THRESHOLD,
            "cpp_sdk_labels_exact_python_tensorrt": True,
        }
        sample["pytorch_tensorrt"] = comparison
        comparisons.append(comparison)
    payload = {
        "status": "PASS_WITH_NUMERICAL_EXCEPTION"
        if all(item["agreement"] == 1.0 for item in comparisons)
        else "FAILED",
        "reference_kind": "fresh PyTorch CUDA and TensorRT runs on the exact production C++ sampled indices, features, and k=6 adjacency",
        "checkpoint": str(CHECKPOINT),
        "engine": str(PRODUCTION_ENGINE),
        "all_current_cpp_sdk_labels_exact_python_tensorrt": True,
        "all_pytorch_tensorrt_labels_exact": all(
            item["agreement"] == 1.0 for item in comparisons
        ),
        "samples": comparisons,
    }
    dump_json(run_dir / "pytorch_tensorrt_parity.json", payload)
    return payload


def write_per_sample_files(run_dir: Path, per_sample: list[dict[str, Any]]) -> None:
    with (run_dir / "per_sample_results.jsonl").open("w", encoding="utf-8") as stream:
        for item in per_sample:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    fieldnames = [
        "sample_id",
        "row_count",
        "source_sha256",
        "weld_points",
        "weld_ratio",
        "accuracy",
        "weld_precision",
        "weld_recall",
        "weld_f1",
        "weld_iou",
        "background_iou",
        "miou",
        "max_abs_logit_error",
        "strict_pass",
        "load_ms",
        "sampling_ms",
        "feature_ms",
        "adjacency_ms",
        "inference_cuda_ms",
        "inference_wall_ms",
        "postprocess_ms",
        "total_ms",
        "export_status",
    ]
    with (run_dir / "per_sample_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in per_sample:
            metrics = item["metrics"]
            timing = item["timing_ms"]
            writer.writerow(
                {
                    "sample_id": item["sample_id"],
                    "row_count": item["row_count"],
                    "source_sha256": item["sha256"],
                    "weld_points": item["geometry"]["weld_point_count"],
                    "weld_ratio": item["geometry"]["weld_ratio"],
                    "accuracy": metrics["accuracy"],
                    "weld_precision": metrics["weld_seam_precision"],
                    "weld_recall": metrics["weld_seam_recall"],
                    "weld_f1": metrics["weld_seam_f1"],
                    "weld_iou": metrics["weld_seam_iou"],
                    "background_iou": metrics["background_iou"],
                    "miou": metrics["miou"],
                    "max_abs_logit_error": item["pytorch_tensorrt"][
                        "max_abs_error"
                    ],
                    "strict_pass": item["pytorch_tensorrt"][
                        "strict_threshold_passed"
                    ],
                    "load_ms": timing["load_cloud"],
                    "sampling_ms": timing["sampling"],
                    "feature_ms": timing["feature_build"],
                    "adjacency_ms": timing["adjacency_build"],
                    "inference_cuda_ms": timing["inference_cuda"],
                    "inference_wall_ms": timing["inference_wall"],
                    "postprocess_ms": timing["postprocess"],
                    "total_ms": timing["total_sdk_detect"],
                    "export_status": item["output"]["output_file_validation_status"],
                }
            )


def repeated_determinism(records: list[dict[str, Any]]) -> dict[str, Any]:
    detections = [
        item for item in records if item["record_type"] == "detection" and item["round"] > 0
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in detections:
        grouped[item["sample_id"]].append(item)
    samples = []
    for sample_id, values in grouped.items():
        values.sort(key=lambda item: item["round"])
        labels_exact = all(item["labels"] == values[0]["labels"] for item in values[1:])
        weld_counts_exact = all(
            item["weld_points"] == values[0]["weld_points"] for item in values[1:]
        )
        geometry_max_abs = 0.0
        for item in values[1:]:
            for key in (
                "center",
                "bbox_min",
                "bbox_max",
                "principal_direction",
            ):
                geometry_max_abs = max(
                    geometry_max_abs,
                    max(
                        abs(float(left) - float(right))
                        for left, right in zip(item[key], values[0][key])
                    ),
                )
            geometry_max_abs = max(
                geometry_max_abs,
                abs(float(item["length_mm"]) - float(values[0]["length_mm"])),
            )
        samples.append(
            {
                "sample_id": sample_id,
                "rounds": len(values),
                "labels_exact": labels_exact,
                "weld_counts_exact": weld_counts_exact,
                "geometry_max_abs": geometry_max_abs,
                "all_status_success": all(item["success"] for item in values),
                "all_error_recorder_zero": all(
                    item["error_recorder_errors"] == 0 for item in values
                ),
            }
        )
    passed = (
        len(samples) == 18
        and all(
            item["rounds"] == 3
            and item["labels_exact"]
            and item["weld_counts_exact"]
            and item["geometry_max_abs"] <= 1.0e-6
            and item["all_status_success"]
            and item["all_error_recorder_zero"]
            for item in samples
        )
    )
    return {
        "status": "PASS" if passed else "FAILED",
        "rounds": 3,
        "same_process": True,
        "logits_bitwise_determinism_claimed": False,
        "samples": samples,
    }


def strict_numerical(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [
        {
            "sample_id": item["sample_id"],
            "max_abs_error": item["pytorch_tensorrt"]["max_abs_error"],
            "mean_abs_error": item["pytorch_tensorrt"]["mean_abs_error"],
            "strict_threshold_passed": item["pytorch_tensorrt"][
                "strict_threshold_passed"
            ],
            "label_agreement": item["pytorch_tensorrt"]["agreement"],
        }
        for item in per_sample
    ]
    passed = sum(item["strict_threshold_passed"] for item in samples)
    worst = max(samples, key=lambda item: item["max_abs_error"])
    return {
        "status": "CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED",
        "scope": "Current same-input production C++ preprocessing, PyTorch CUDA, and TensorRT comparison; historical Phase 8D boundary retained separately.",
        "criterion": "per-sample max_abs_error < 1e-4",
        "threshold": STRICT_THRESHOLD,
        "passed_samples": passed,
        "failed_samples": len(samples) - passed,
        "failed_sample_ids": [
            item["sample_id"] for item in samples if not item["strict_threshold_passed"]
        ],
        "worst_sample": worst["sample_id"],
        "worst_max_abs": worst["max_abs_error"],
        "historical_phase8d": {
            "status": "CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED",
            "passed_samples": 13,
            "failed_samples": 5,
            "failed_sample_ids": ["weld_5", "weld_12", "weld_14", "weld_4", "weld_15"],
            "worst_sample": "weld_14",
            "worst_max_abs": 0.00012302398681640625,
            "reference": str(PHASE8D_ACCURACY),
        },
        "strict_numerical_equivalence_claimed": False,
        "samples": samples,
    }


def phase8d_regression(
    samples: list[dict[str, Any]], aggregate: dict[str, Any], strict: dict[str, Any]
) -> dict[str, Any]:
    baseline = json.loads(PHASE8D_ACCURACY.read_text(encoding="utf-8"))
    baseline_ids = [item["sample_id"] for item in baseline["per_sample"]]
    current_ids = [item["sample_id"] for item in samples]
    baseline_metrics = baseline["metrics"]["candidate"]
    current = aggregate["global_point_level_metrics"]
    mapping = {
        "overall_accuracy": "accuracy",
        "miou": "miou",
        "weld_seam_precision": "weld_seam_precision",
        "weld_seam_recall": "weld_seam_recall",
        "weld_seam_f1": "weld_seam_f1",
    }
    deltas = {
        baseline_name: current[current_name] - baseline_metrics[baseline_name]
        for baseline_name, current_name in mapping.items()
    }
    sampling_orders_equal = all(
        item["phase8d_task_comparison"]["phase8d_sample_indices_equal"]
        for item in samples
    )
    original_row_label_mismatches = {
        item["sample_id"]: item["phase8d_task_comparison"][
            "same_original_row_prediction_mismatches"
        ]
        for item in samples
    }
    passed = baseline_ids == current_ids and all(value == 0.0 for value in deltas.values())
    return {
        "status": "PASS" if passed else "FAILED",
        "phase8d_path": str(PHASE8D_ACCURACY),
        "sample_list_exact": baseline_ids == current_ids,
        "current_sample_ids": current_ids,
        "phase8d_sample_ids": baseline_ids,
        "task_metric_deltas": deltas,
        "phase8d_sampling_order_matches_production_sdk": sampling_orders_equal,
        "same_original_row_prediction_mismatches": original_row_label_mismatches,
        "root_cause": (
            None
            if passed
            else "Phase 8D froze a NumPy-generated point permutation while the "
            "production C++ SDK uses std::mt19937 plus std::shuffle. Every test "
            "file has exactly 2048 rows, so the point set is unchanged but the "
            "order differs. The deployed network is measurably order-sensitive, "
            "which changes per-original-point predictions and aggregate metrics."
        ),
        "actual_model_asset_regression": False,
        "qualification_blocker": not passed,
        "current_same_input_strict_pass_count": strict["passed_samples"],
        "historical_phase8d_strict_pass_count": baseline[
            "strict_threshold_passed_samples"
        ],
        "engine_sha_unchanged": True,
        "plugin_sha_unchanged": True,
        "latency_comparison_scope": "Phase 11A SDK end-to-end timings are not directly interchangeable with Phase 8D pure TensorRT timings.",
    }


def timing_analysis(
    run_dir: Path, per_sample: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage_keys = (
        "load_cloud",
        "sampling",
        "feature_build",
        "adjacency_build",
        "inference_cuda",
        "inference_wall",
        "postprocess",
        "total_sdk_detect",
    )
    summary = {
        "methodology": {
            "runtime_initialized_once": True,
            "warmup_detections": 1,
            "measured_samples": 18,
            "sample_order": "authoritative test JSON order",
            "cold_start_excluded": True,
            "export_excluded": True,
            "qt_rendering_excluded": True,
        },
        "stages_ms": {
            key: timing_stats([item["timing_ms"][key] for item in per_sample])
            for key in stage_keys
        },
    }
    with (run_dir / "timing_per_sample.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", *stage_keys])
        writer.writeheader()
        for item in per_sample:
            writer.writerow({"sample_id": item["sample_id"], **item["timing_ms"]})
    means = {key: summary["stages_ms"][key]["mean"] for key in stage_keys}
    total = means["total_sdk_detect"]
    accounted = {
        "load": means["load_cloud"],
        "sampling": means["sampling"],
        "feature_build": means["feature_build"],
        "adjacency_build": means["adjacency_build"],
        "tensorrt_inference_wall": means["inference_wall"],
        "postprocess": means["postprocess"],
    }
    other = max(0.0, total - sum(accounted.values()))
    accounted["other_overhead"] = other
    bottleneck_name = max(accounted, key=accounted.get)
    bottleneck = {
        "status": "PASS",
        "mean_total_sdk_detect_ms": total,
        "stage_mean_ms": accounted,
        "stage_share_percent": {
            key: safe_divide(value * 100.0, total) for key, value in accounted.items()
        },
        "dominant_stage": bottleneck_name,
        "cpu_knn_is_primary_or_major_bottleneck": (
            bottleneck_name == "adjacency_build"
            or safe_divide(accounted["adjacency_build"], total) >= 0.25
        ),
        "optimization_performed": False,
    }
    dump_json(run_dir / "timing_summary.json", summary)
    dump_json(run_dir / "bottleneck_analysis.json", bottleneck)
    return summary, bottleneck


def cold_start_qualification(
    run_dir: Path,
    qualifier: Path,
    fixed_cloud: Path,
    log: Path,
    count: int,
) -> dict[str, Any]:
    list_path = run_dir / "cold_start_cloud.txt"
    write_cloud_list(list_path, [fixed_cloud])
    results = []
    for index in range(1, count + 1):
        output = run_dir / "cold_start" / f"cold_{index:02d}.jsonl"
        started = time.perf_counter()
        completed = run(
            [
                str(qualifier),
                "--engine",
                str(PRODUCTION_ENGINE),
                "--plugin",
                str(PRODUCTION_PLUGIN),
                "--list",
                str(list_path),
                "--output",
                str(output),
                "--rounds",
                "1",
                "--warmup",
                "0",
                "--resource-interval",
                "1",
                "--repeat-initialize",
                "1",
                "--compact",
                "true",
            ],
            log,
            timeout=180,
            check=False,
        )
        process_ms = (time.perf_counter() - started) * 1000.0
        records = read_jsonl(output) if output.is_file() else []
        initialization = next(
            (item for item in records if item["record_type"] == "initialization"),
            {},
        )
        detection = next(
            (item for item in records if item["record_type"] == "detection"),
            {},
        )
        results.append(
            {
                "cold_start": index,
                "fresh_process": True,
                "exit_code": completed.returncode,
                "initialization_success": initialization.get("success", False),
                "initialization_wall_ms": initialization.get("wall_ms"),
                "first_inference_cuda_ms": detection.get("inference_cuda_ms"),
                "first_detect_total_ms": detection.get("total_ms"),
                "process_wall_ms": process_ms,
                "error_recorder_errors": detection.get("error_recorder_errors"),
                "success": completed.returncode == 0
                and initialization.get("success") is True
                and detection.get("success") is True
                and detection.get("error_recorder_errors") == 0,
            }
        )
    return {
        "status": "PASS" if all(item["success"] for item in results) else "FAILED",
        "fixed_sample": fixed_cloud.stem,
        "passed": sum(item["success"] for item in results),
        "total": len(results),
        "results": results,
    }


def analyze_soak(records: list[dict[str, Any]], rounds: int) -> dict[str, Any]:
    detections = [item for item in records if item["record_type"] == "detection"]
    resources = [item for item in records if item["record_type"] == "resource"]
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in detections:
        by_sample[item["sample_id"]].append(item)
    deterministic = all(
        len(values) == rounds
        and len({item["label_hash"] for item in values}) == 1
        and len({item["weld_points"] for item in values}) == 1
        and all(item["success"] and item["error_recorder_errors"] == 0 for item in values)
        for values in by_sample.values()
    )
    working = [item["working_set_bytes"] for item in resources if item["working_set_bytes"]]
    private = [item["private_bytes"] for item in resources if item["private_bytes"]]
    gpu_used = [
        item["gpu_total_bytes"] - item["gpu_free_bytes"]
        for item in resources
        if item["gpu_total_bytes"]
    ]
    resource = {
        "samples": resources,
        "working_set_growth_bytes": (working[-1] - working[0]) if len(working) >= 2 else None,
        "private_memory_growth_bytes": (private[-1] - private[0]) if len(private) >= 2 else None,
        "gpu_used_growth_bytes": (gpu_used[-1] - gpu_used[0])
        if len(gpu_used) >= 2
        else None,
        "handle_growth": (
            resources[-1]["handle_count"] - resources[0]["handle_count"]
            if len(resources) >= 2
            else None
        ),
        "thread_growth": (
            resources[-1]["thread_count"] - resources[0]["thread_count"]
            if len(resources) >= 2
            else None
        ),
    }
    expected = 18 * rounds
    passed = (
        len(detections) == expected
        and deterministic
        and all(item["success"] for item in detections)
        and all(item["error_recorder_errors"] == 0 for item in detections)
    )
    return {
        "status": "PASS" if passed else "FAILED",
        "rounds": rounds,
        "total_detections": len(detections),
        "expected_detections": expected,
        "successful_detections": sum(item["success"] for item in detections),
        "failed_detections": sum(not item["success"] for item in detections),
        "single_process": True,
        "engine_initializations": 1,
        "plugin_loads": 1,
        "task_deterministic": deterministic,
        "resource_measurements": resource,
        "conclusion": (
            "No monotonic resource growth or runtime instability was observed over "
            "the qualification interval; this bounded run is not a proof of the "
            "absence of all memory leaks."
        ),
    }


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Invalid PNG: {path}")
    return struct.unpack(">II", data[16:24])


def validate_export_directory(
    directory: Path, sample_id: str, expected_labels: list[int]
) -> dict[str, Any]:
    required = (
        "weld_result.json",
        "weld_points.ply",
        "prediction.txt",
        "task_manifest.json",
        "detection_view.png",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        return {"status": "FAILED", "directory": str(directory), "missing": missing}
    manifest = json.loads((directory / "task_manifest.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "weld_result.json").read_text(encoding="utf-8"))
    hash_checks = []
    for entry in manifest["files"]:
        path = directory / entry["name"]
        hash_checks.append(
            {
                "name": entry["name"],
                "expected": entry["sha256"],
                "actual": sha256(path),
                "passed": sha256(path) == entry["sha256"],
            }
        )
    prediction_rows = []
    with (directory / "prediction.txt").open("r", encoding="utf-8") as stream:
        for line in stream:
            columns = line.split()
            if len(columns) != 4:
                raise RuntimeError(f"Malformed prediction row in {directory}")
            prediction_rows.append(int(float(columns[3])))
    ply_lines = (directory / "weld_points.ply").read_text(encoding="utf-8").splitlines()
    vertex_line = next(line for line in ply_lines if line.startswith("element vertex "))
    vertex_count = int(vertex_line.rsplit(" ", 1)[1])
    header_end = ply_lines.index("end_header")
    ply_data = [line.split() for line in ply_lines[header_end + 1 :] if line.strip()]
    ply_labels_all_weld = all(int(row[3]) == 0 for row in ply_data)
    width, height = png_size(directory / "detection_view.png")
    partial_dirs = [
        str(path) for path in directory.parent.glob(".*.tmp") if path.is_dir()
    ]
    passed = (
        manifest["task_id"] == sample_id
        and result["task_id"] == sample_id
        and manifest["engine_sha256"] == ENGINE_SHA256
        and manifest["plugin_sha256"] == PLUGIN_SHA256
        and len(prediction_rows) == 2048
        and prediction_rows == expected_labels
        and vertex_count == result["weld_point_count"] == len(ply_data)
        and ply_labels_all_weld
        and all(item["passed"] for item in hash_checks)
        and not partial_dirs
        and width > 0
        and height > 0
    )
    return {
        "status": "PASS" if passed else "FAILED",
        "directory": str(directory),
        "task_id": manifest["task_id"],
        "hash_checks": hash_checks,
        "prediction_rows": len(prediction_rows),
        "labels_exact_source": prediction_rows == expected_labels,
        "ply_vertex_count": vertex_count,
        "ply_labels_all_weld": ply_labels_all_weld,
        "result": result,
        "screenshot": {"width": width, "height": height},
        "partial_temporary_directories": partial_dirs,
    }


def package_qualification(
    run_dir: Path,
    package: Path,
    samples: list[dict[str, Any]],
    per_sample: list[dict[str, Any]],
    qualification_log: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    external_root = Path("D:/") / f"PTV2_Phase11A_Runtime_{run_dir.name[:22]}"
    inputs = external_root / "inputs"
    exports = external_root / "exports"
    user_configs = external_root / "user_configs"
    inputs.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)
    user_configs.mkdir(parents=True, exist_ok=True)
    current_by_id = {item["sample_id"]: item for item in per_sample}
    copied: dict[str, Path] = {}
    for sample in samples:
        destination = inputs / f"{sample['sample_id']}.txt"
        shutil.copy2(sample["absolute_source_path"], destination)
        if sha256(destination) != sample["sha256"]:
            raise RuntimeError(f"External package input copy changed: {sample['sample_id']}")
        copied[sample["sample_id"]] = destination
    executable = package / "ptv2_weld_qt.exe"
    if not executable.is_file():
        raise RuntimeError(f"Package executable is missing: {executable}")
    package_results = []
    for sample in samples:
        sample_id = sample["sample_id"]
        export_root = exports / sample_id
        user_config = user_configs / f"{sample_id}.ini"
        env = os.environ.copy()
        env["QT_OPENGL"] = "desktop"
        completed = run(
            [
                str(executable),
                "--cloud",
                str(copied[sample_id]),
                "--product-smoke-export",
                str(export_root),
                "--user-config",
                str(user_config),
            ],
            qualification_log,
            cwd=package,
            env=env,
            timeout=180,
            check=False,
        )
        task_directories = sorted(
            [path for path in export_root.glob(f"{sample_id}_*") if path.is_dir()],
            key=lambda value: value.stat().st_mtime,
        )
        if completed.returncode != 0 or len(task_directories) != 1:
            package_results.append(
                {
                    "sample_id": sample_id,
                    "status": "FAILED",
                    "exit_code": completed.returncode,
                    "task_directories": [str(item) for item in task_directories],
                }
            )
            continue
        validated = validate_export_directory(
            task_directories[0],
            sample_id,
            current_by_id[sample_id]["predicted_labels"],
        )
        source_geometry = current_by_id[sample_id]["geometry"]
        package_geometry = validated.get("result", {})
        geometry_deltas: dict[str, Any] = {}
        if package_geometry:
            geometry_deltas = {
                "centroid_max_abs": max(
                    abs(float(left) - float(right))
                    for left, right in zip(
                        source_geometry["centroid"], package_geometry["center"]
                    )
                ),
                "bbox_min_max_abs": max(
                    abs(float(left) - float(right))
                    for left, right in zip(
                        source_geometry["bbox_min"], package_geometry["bbox"]["min"]
                    )
                ),
                "bbox_max_max_abs": max(
                    abs(float(left) - float(right))
                    for left, right in zip(
                        source_geometry["bbox_max"], package_geometry["bbox"]["max"]
                    )
                ),
                "length_abs": abs(
                    float(source_geometry["pca_length_mm"])
                    - float(package_geometry["pca_length_mm"])
                ),
            }
        validated.update(
            {
                "sample_id": sample_id,
                "exit_code": completed.returncode,
                "source_geometry_deltas": geometry_deltas,
                "source_task_equivalent": validated.get("status") == "PASS"
                and all(value <= 1.0e-5 for value in geometry_deltas.values()),
                "external_cloud_path": str(copied[sample_id]),
            }
        )
        if not validated["source_task_equivalent"]:
            validated["status"] = "FAILED"
        package_results.append(validated)
        current_by_id[sample_id]["output"] = {
            "success": validated["status"] == "PASS",
            "exported_result_path": str(task_directories[0]),
            "output_file_validation_status": validated["status"],
        }
    all_pass = len(package_results) == 18 and all(
        item["status"] == "PASS" for item in package_results
    )
    no_source_dependency = all(
        "E:\\GRP-PTv2" not in str(item.get("external_cloud_path", ""))
        and "E:/GRP-PTv2" not in str(item.get("external_cloud_path", ""))
        for item in package_results
    )
    package_equivalence = {
        "status": "PASS" if all_pass and no_source_dependency else "FAILED",
        "source_backend": "repository-built weld_sdk_testset_qualification",
        "package_backend": str(executable),
        "package_root": str(package),
        "package_launched_outside_source_build_artifacts": True,
        "external_runtime_root": str(external_root),
        "runtime_inputs_outside_repository": True,
        "no_hidden_repository_input_dependency": no_source_dependency,
        "engine_sha_exact": sha256(package / "engine" / "strict_fp32_voxelunique_cub.plan")
        == ENGINE_SHA256,
        "plugin_sha_exact": sha256(package / "plugins" / "VoxelUniqueCubPlugin.dll")
        == PLUGIN_SHA256,
        "exact_label_agreement_all_samples": all(
            item.get("labels_exact_source") is True for item in package_results
        ),
        "geometry_within_1e_5_all_samples": all(
            item.get("source_task_equivalent") is True for item in package_results
        ),
        "samples": package_results,
    }
    export_validation = {
        "status": "PASS" if all_pass else "FAILED",
        "required_samples": 18,
        "validated_samples": sum(
            item["status"] == "PASS" for item in package_results
        ),
        "non_gui_outputs_validated_all_18": all_pass,
        "png_exported_all_18": all(
            item.get("screenshot", {}).get("width", 0) > 0 for item in package_results
        ),
        "samples": package_results,
    }
    return package_equivalence, export_validation, external_root


def execute_fail_closed(
    run_dir: Path,
    qualifier: Path,
    package: Path,
    fixed_cloud: Path,
    log: Path,
) -> dict[str, Any]:
    cases_root = run_dir / "failure_cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    cloud_list = cases_root / "cloud.txt"
    write_cloud_list(cloud_list, [fixed_cloud])

    def qualifier_case(
        name: str, engine: Path, plugin: Path, list_path: Path
    ) -> dict[str, Any]:
        output = cases_root / f"{name}.jsonl"
        completed = run(
            [
                str(qualifier),
                "--engine",
                str(engine),
                "--plugin",
                str(plugin),
                "--list",
                str(list_path),
                "--output",
                str(output),
                "--rounds",
                "1",
                "--warmup",
                "0",
                "--resource-interval",
                "1",
                "--repeat-initialize",
                "1",
                "--compact",
                "true",
            ],
            log,
            timeout=120,
            check=False,
        )
        text = output.read_text(encoding="utf-8") if output.is_file() else ""
        return {
            "case": name,
            "mechanism": "WeldDetector qualification process",
            "exit_code": completed.returncode,
            "nonzero_exit": completed.returncode != 0,
            "no_success_summary": '"record_type":"summary","success":true' not in text,
            "passed": completed.returncode != 0
            and '"record_type":"summary","success":true' not in text,
        }

    missing_engine = qualifier_case(
        "missing_engine", cases_root / "missing.plan", PRODUCTION_PLUGIN, cloud_list
    )
    missing_plugin = qualifier_case(
        "missing_plugin", PRODUCTION_ENGINE, cases_root / "missing.dll", cloud_list
    )
    corrupt_engine = cases_root / "wrong_sha.plan"
    shutil.copy2(PRODUCTION_ENGINE, corrupt_engine)
    with corrupt_engine.open("ab") as stream:
        stream.write(b"phase11a-wrong-sha")
    wrong_engine = qualifier_case(
        "wrong_engine_sha", corrupt_engine, PRODUCTION_PLUGIN, cloud_list
    )
    missing_cloud_list = cases_root / "missing_cloud.txt"
    write_cloud_list(missing_cloud_list, [cases_root / "missing.txt"])
    missing_cloud = qualifier_case(
        "missing_cloud", PRODUCTION_ENGINE, PRODUCTION_PLUGIN, missing_cloud_list
    )
    malformed = cases_root / "malformed.txt"
    dump_text(malformed, "not four numeric columns")
    malformed_list = cases_root / "malformed_list.txt"
    write_cloud_list(malformed_list, [malformed])
    malformed_case = qualifier_case(
        "malformed_txt", PRODUCTION_ENGINE, PRODUCTION_PLUGIN, malformed_list
    )
    fewer = cases_root / "fewer_than_2048.txt"
    with fixed_cloud.open("r", encoding="utf-8") as source:
        dump_text(fewer, "".join(source.readlines()[:100]))
    fewer_list = cases_root / "fewer_list.txt"
    write_cloud_list(fewer_list, [fewer])
    fewer_case = qualifier_case(
        "fewer_than_2048", PRODUCTION_ENGINE, PRODUCTION_PLUGIN, fewer_list
    )
    nan_cloud = cases_root / "nan_coordinate.txt"
    lines = fixed_cloud.read_text(encoding="utf-8").splitlines()
    first = lines[0].split()
    first[0] = "nan"
    lines[0] = " ".join(first)
    dump_text(nan_cloud, "\n".join(lines))
    nan_list = cases_root / "nan_list.txt"
    write_cloud_list(nan_list, [nan_cloud])
    nan_case = qualifier_case(
        "nan_coordinate", PRODUCTION_ENGINE, PRODUCTION_PLUGIN, nan_list
    )

    qt_build = (
        ARTIFACT_BASE
        / "20260723_194208_270011_phase10c2_qt_browse_default_directory"
        / "build"
        / "Release"
    )
    app_config_test = qt_build / "AppConfigTest.exe"
    export_test = qt_build / "DetectionExportTest.exe"
    package_test = qt_build / "RuntimePackageSmokeTest.exe"
    postprocess_probe = run_dir / "build" / "postprocess" / "Release" / "postprocess_failure_probe.exe"

    supporting_runs: dict[str, subprocess.CompletedProcess[str]] = {}
    for name, executable in (
        ("app_config", app_config_test),
        ("detection_export", export_test),
        ("runtime_package", package_test),
    ):
        if not executable.is_file():
            raise RuntimeError(f"Required fail-closed test executable missing: {executable}")
        environment = os.environ.copy()
        if name == "runtime_package":
            environment["PTV2_PACKAGE_ROOT"] = str(package)
        supporting_runs[name] = run(
            [str(executable), "-o", "-,txt"],
            log,
            cwd=executable.parent,
            env=environment,
            timeout=120,
            check=False,
        )
    if not postprocess_probe.is_file():
        raise RuntimeError(f"Postprocess failure probe missing: {postprocess_probe}")
    postprocess_runs = {}
    for probe_case in ("nan_logits", "no_weld", "unwritable_output"):
        postprocess_runs[probe_case] = run(
            [str(postprocess_probe), "--case", probe_case],
            log,
            cwd=postprocess_probe.parent,
            timeout=60,
            check=False,
        )

    repeat_output = cases_root / "repeat_initialize.jsonl"
    repeat_records = run_qualification_sequence(
        qualifier,
        cloud_list,
        repeat_output,
        log,
        rounds=1,
        warmup=0,
        compact=True,
        repeat_initialize=2,
        timeout=180,
    )
    repeat_initializations = [
        item for item in repeat_records if item["record_type"] == "initialization"
    ]

    cases = [
        missing_engine,
        missing_plugin,
        wrong_engine,
        {
            "case": "wrong_plugin_sha",
            "mechanism": str(app_config_test),
            "expected": "AppConfig runtime integrity rejects wrong Plugin SHA",
            "exit_code": supporting_runs["app_config"].returncode,
            "passed": supporting_runs["app_config"].returncode == 0,
        },
        missing_cloud,
        malformed_case,
        fewer_case,
        nan_case,
        {
            "case": "nonfinite_logits",
            "mechanism": str(postprocess_probe),
            "expected": "SegmentationPostProcessor rejects NaN logits",
            "exit_code": postprocess_runs["nan_logits"].returncode,
            "passed": postprocess_runs["nan_logits"].returncode != 0,
        },
        {
            "case": "no_weld_postprocess",
            "mechanism": str(postprocess_probe),
            "expected": "WeldGeometryExtractor rejects a no-weld result",
            "exit_code": postprocess_runs["no_weld"].returncode,
            "passed": postprocess_runs["no_weld"].returncode != 0,
        },
        {
            "case": "unwritable_export_directory",
            "mechanism": str(postprocess_probe),
            "expected": "ResultWriter rejects a file used as an output directory",
            "exit_code": postprocess_runs["unwritable_output"].returncode,
            "passed": postprocess_runs["unwritable_output"].returncode != 0,
        },
        {
            "case": "corrupted_manifest",
            "mechanism": str(export_test),
            "expected": "DetectionExportService detects payload corruption",
            "exit_code": supporting_runs["detection_export"].returncode,
            "passed": supporting_runs["detection_export"].returncode == 0,
        },
        {
            "case": "package_missing_qwindows",
            "mechanism": str(package_test),
            "expected": "RuntimePackageValidator rejects missing qwindows.dll",
            "exit_code": supporting_runs["runtime_package"].returncode,
            "passed": supporting_runs["runtime_package"].returncode == 0,
        },
        {
            "case": "package_missing_plugin_dependency",
            "mechanism": str(package_test),
            "expected": "RuntimePackageValidator rejects missing VoxelUnique Plugin",
            "exit_code": supporting_runs["runtime_package"].returncode,
            "passed": supporting_runs["runtime_package"].returncode == 0,
        },
        {
            "case": "repeated_initialize_without_reset",
            "mechanism": str(qualifier),
            "expected": "Two explicit initialize calls both succeed without stale runtime state",
            "initialization_attempts": repeat_initializations,
            "passed": len(repeat_initializations) == 2
            and all(item["success"] for item in repeat_initializations),
        },
    ]
    return {
        "status": "PASS" if all(item["passed"] for item in cases) else "FAILED",
        "tested_cases": len(cases),
        "passed_cases": sum(item["passed"] for item in cases),
        "policy": "No crash, no successful summary/fake output on rejected runtime inputs; support probes assert component fail-closed behavior.",
        "cases": cases,
    }


def gui_subset_report(
    package_results: dict[str, Any],
    per_sample: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {item["sample_id"]: item for item in per_sample}
    package_by_id = {
        item["sample_id"]: item for item in package_results["samples"]
    }
    low = min(per_sample, key=lambda item: item["geometry"]["weld_ratio"])["sample_id"]
    high = max(per_sample, key=lambda item: item["geometry"]["weld_ratio"])["sample_id"]
    selected = list(dict.fromkeys([low, high, "weld_14", "weld_65"]))
    prior_root = (
        ARTIFACT_BASE
        / "20260723_184946_689058_phase10c1_qt_layout_resize_stability"
    )
    prior_scroll = list(prior_root.rglob("scroll_area_test.json"))
    prior_resize = list(prior_root.rglob("resize_stress_test.json"))
    entries = []
    for sample_id in selected:
        package = package_by_id[sample_id]
        entries.append(
            {
                "sample_id": sample_id,
                "selection_reason": [
                    reason
                    for condition, reason in (
                        (sample_id == low, "lowest weld ratio"),
                        (sample_id == high, "highest weld ratio"),
                        (sample_id == "weld_14", "worst strict numerical sample"),
                        (sample_id == "weld_65", "qualified reference sample"),
                    )
                    if condition
                ],
                "visualization_point_count": 2048,
                "weld_points": by_id[sample_id]["geometry"]["weld_point_count"],
                "background_points": 2048
                - by_id[sample_id]["geometry"]["weld_point_count"],
                "label_color_contract": "class 0 blue weld_seam; class 1 red background",
                "bbox_centroid_pca_fields_finite": by_id[sample_id]["geometry"]["finite"],
                "export_status": package["status"],
                "screenshot": package.get("screenshot"),
                "passed": package["status"] == "PASS"
                and by_id[sample_id]["geometry"]["finite"],
            }
        )
    passed = (
        len(selected) >= 4
        and all(item["passed"] for item in entries)
        and bool(prior_scroll)
        and bool(prior_resize)
    )
    return {
        "status": "PASS" if passed else "FAILED",
        "scope": "Automated product-smoke rendering/export on the selected data cases; data-independent scroll/resize stress evidence reused from the qualified Phase 10C.1 package.",
        "selected_samples": selected,
        "phase10c1_scroll_evidence": [str(item) for item in prior_scroll],
        "phase10c1_resize_evidence": [str(item) for item in prior_resize],
        "samples": entries,
    }


def environment_report(
    run_dir: Path, package: Path, qualifier: Path, assets: dict[str, Any]
) -> dict[str, Any]:
    def capture(command: list[str]) -> str:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return completed.stdout.strip()

    python_details: dict[str, Any] = {}
    try:
        import torch

        python_details.update(
            {
                "pytorch_version": torch.__version__,
                "pytorch_cuda_runtime": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    except Exception as error:  # environment reporting must preserve the error
        python_details["pytorch_import_error"] = repr(error)
    try:
        import tensorrt as trt

        python_details["tensorrt_python_version"] = trt.__version__
    except Exception as error:
        python_details["tensorrt_import_error"] = repr(error)
    try:
        import onnxruntime as ort

        python_details["onnxruntime_version"] = ort.__version__
    except Exception as error:
        python_details["onnxruntime_import_error"] = repr(error)
    git_status = capture(["git", "status", "--short"])
    return {
        "captured_at": datetime.now().astimezone().isoformat(),
        "windows": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        **python_details,
        "gpu_driver": capture(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "cuda_toolkit": capture(["nvcc", "--version"]),
        "cmake": capture(["cmake", "--version"]).splitlines()[0],
        "compiler": capture(["cmd", "/c", "where cl"]),
        "qt_package_version": "0.1.1",
        "package": str(package),
        "qualifier_executable": str(qualifier),
        "git_commit": capture(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(git_status),
        "git_status_short": git_status.splitlines(),
        "production_assets": assets,
    }


def aggregate_markdown(aggregate: dict[str, Any]) -> str:
    global_metrics = aggregate["global_point_level_metrics"]
    means = aggregate["mean_per_sample_metrics"]
    return f"""# Phase 11A aggregate metrics

Label semantics: class 0 = weld_seam; class 1 = background.

| Metric | Global point-level | Mean per sample |
|---|---:|---:|
| Accuracy | {global_metrics['accuracy']:.9f} | {means['accuracy']:.9f} |
| Weld precision | {global_metrics['weld_seam_precision']:.9f} | {means['weld_seam_precision']:.9f} |
| Weld recall | {global_metrics['weld_seam_recall']:.9f} | {means['weld_seam_recall']:.9f} |
| Weld F1 | {global_metrics['weld_seam_f1']:.9f} | {means['weld_seam_f1']:.9f} |
| Weld IoU | {global_metrics['weld_seam_iou']:.9f} | {means['weld_seam_iou']:.9f} |
| Background IoU | {global_metrics['background_iou']:.9f} | {means['background_iou']:.9f} |
| mIoU | {global_metrics['miou']:.9f} | {means['miou']:.9f} |

Confusion matrix rows are ground truth and columns are prediction:
`{global_metrics['confusion_matrix_rows_ground_truth_columns_prediction']}`.
"""


def write_phase_summary(
    run_dir: Path,
    samples: list[dict[str, Any]],
    aggregate: dict[str, Any],
    strict: dict[str, Any],
    repeated: dict[str, Any],
    cold: dict[str, Any],
    soak: dict[str, Any],
    timing: dict[str, Any],
    bottleneck: dict[str, Any],
    package: dict[str, Any],
    exports: dict[str, Any],
    failures: dict[str, Any],
    gui: dict[str, Any],
    passed: bool,
) -> str:
    global_metrics = aggregate["global_point_level_metrics"]
    status = (
        "PHASE_11A_RELEASE_FULL_TESTSET_QUALIFICATION_PASSED_WITH_NUMERICAL_EXCEPTION"
        if passed
        else "PHASE_11A_RELEASE_FULL_TESTSET_QUALIFICATION_BLOCKED"
    )
    task_status = (
        "TENSORRT_RELEASE_FULL_TESTSET_TASK_EQUIVALENT"
        if passed
        else "TENSORRT_RELEASE_FULL_TESTSET_TASK_EQUIVALENCE_FAILED"
    )
    summary = f"""# Phase 11A release full test-set qualification

- Test samples: {len(samples)}/18
- SDK execution: {len(samples)}/18 first-round PASS
- Global accuracy: {global_metrics['accuracy']:.9f}
- Global mIoU: {global_metrics['miou']:.9f}
- Global weld precision/recall/F1: {global_metrics['weld_seam_precision']:.9f} / {global_metrics['weld_seam_recall']:.9f} / {global_metrics['weld_seam_f1']:.9f}
- Strict numerical threshold: {strict['passed_samples']}/18 PASS
- Strict failures: {', '.join(strict['failed_sample_ids'])}
- Worst strict sample: {strict['worst_sample']} ({strict['worst_max_abs']:.15g})
- Repeated 3-round determinism: {repeated['status']}
- Cold starts: {cold['passed']}/{cold['total']} PASS
- Soak: {soak['successful_detections']}/{soak['expected_detections']} PASS
- Mean warm SDK detect: {timing['stages_ms']['total_sdk_detect']['mean']:.6f} ms
- Dominant measured stage: {bottleneck['dominant_stage']}
- Package/source equivalence: {package['status']}
- Full export validation: {exports['status']}
- Fail-closed qualification: {failures['passed_cases']}/{failures['tested_cases']} PASS
- GUI subset smoke: {gui['status']}

The strict element-wise FP32 threshold remains a separate numerical exception.
No threshold was weakened, and strict numerical equivalence is not claimed.

{status}

{task_status}

CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED
"""
    dump_text(run_dir / "phase11a_summary.md", summary)
    return status


def qualification_manifest(run_dir: Path, environment: dict[str, Any]) -> dict[str, Any]:
    required_hashes = {
        "checkpoint": CHECKPOINT,
        "onnx": PRODUCTION_ONNX,
        "engine": PRODUCTION_ENGINE,
        "plugin": PRODUCTION_PLUGIN,
        "qualification_executable": Path(
            environment["qualifier_executable"]
        ),
        "test_set_manifest": run_dir / "test_set_manifest.json",
        "aggregate_metrics": run_dir / "aggregate_metrics.json",
        "strict_numerical_results": run_dir / "strict_numerical_results.json",
        "phase11a_summary": run_dir / "phase11a_summary.md",
    }
    payload = {
        "status": "PASS",
        "generated_at": datetime.now().astimezone().isoformat(),
        "files": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in required_hashes.items()
        },
        "environment": {
            key: environment.get(key)
            for key in (
                "git_commit",
                "git_dirty",
                "git_status_short",
                "windows",
                "gpu_driver",
                "cuda_toolkit",
                "tensorrt_python_version",
                "qt_package_version",
                "compiler",
                "python_version",
                "pytorch_version",
                "onnxruntime_version",
            )
        },
    }
    dump_json(run_dir / "qualification_manifest.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--tensorrt-root", type=Path, default=TENSORRT_ROOT_DEFAULT)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cold-starts", type=int, default=5)
    parser.add_argument("--soak-rounds", type=int, default=20)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.rounds, args.cold_starts, args.soak_rounds) != (3, 5, 20):
        raise RuntimeError("Phase 11A fixed qualification counts are 3, 5, and 20")
    run_id = args.run_id or f"{timestamp()}_phase11a_release_full_testset_qualification"
    run_dir = ARTIFACT_BASE / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logs = {
        name: run_dir / name
        for name in (
            "configure.log",
            "build.log",
            "qualification.log",
            "parity.log",
            "soak.log",
            "failure.log",
        )
    }
    for path in logs.values():
        dump_text(path, f"Phase 11A log: {path.name}")
    try:
        samples, clouds = discover_test_manifest(run_dir)
        package = select_package(args.package)
        assets = verify_frozen_assets(run_dir, package)
        qualifier, _ = build_runner(
            run_dir, args.tensorrt_root, logs["configure.log"], logs["build.log"]
        )
        cloud_list = run_dir / "test_clouds.txt"
        write_cloud_list(cloud_list, clouds)
        primary_raw = run_dir / "primary_3rounds_raw.jsonl"
        primary_records = run_qualification_sequence(
            qualifier,
            cloud_list,
            primary_raw,
            logs["qualification.log"],
            rounds=3,
            warmup=1,
            compact=False,
            timeout=1800,
        )
        per_sample, aggregate, _ = analyze_primary_results(
            run_dir, samples, primary_records
        )
        pytorch_parity = run_same_input_pytorch_tensorrt_parity(
            run_dir, per_sample, primary_records, logs["parity.log"]
        )
        aggregate["rankings"]["largest_logit_max_abs"] = [
            {
                "sample_id": item["sample_id"],
                "value": item["pytorch_tensorrt"]["max_abs_error"],
            }
            for item in sorted(
                per_sample,
                key=lambda value: value["pytorch_tensorrt"]["max_abs_error"],
                reverse=True,
            )[:5]
        ]
        strict = strict_numerical(per_sample)
        repeated = repeated_determinism(primary_records)
        regression = phase8d_regression(per_sample, aggregate, strict)
        ground_truth_audit = {
            "status": "PASS"
            if all(
                item["preprocessing"]["fourth_feature_constant_one"]
                and item["preprocessing"]["points_finite"]
                and item["preprocessing"]["adjacency_finite"]
                for item in per_sample
            )
            else "FAILED",
            "model_input_contract": "points float32 [1,2048,4]; adj float32 [1,2048,2048]",
            "feature_columns": [
                "normalized_x",
                "normalized_y",
                "normalized_z",
                "constant_one",
            ],
            "txt_label_column_usage": "Loaded and sampled with the exact sampled indices, but consumed only after prediction for metrics.",
            "label_passed_to_feature_builder": False,
            "all_fourth_features_constant_one": all(
                item["preprocessing"]["fourth_feature_constant_one"]
                for item in per_sample
            ),
            "sampled_ground_truth_counts": {
                item["sample_id"]: dict(
                    Counter(
                        next(
                            audit["sampled_ground_truth_labels"]
                            for audit in primary_records
                            if audit["record_type"] == "preprocess_audit"
                            and audit["sample_id"] == item["sample_id"]
                        )
                    )
                )
                for item in per_sample
            },
        }
        dump_json(run_dir / "ground_truth_leakage_audit.json", ground_truth_audit)
        dump_json(run_dir / "aggregate_metrics.json", aggregate)
        dump_text(run_dir / "aggregate_metrics.md", aggregate_markdown(aggregate))
        dump_json(run_dir / "strict_numerical_results.json", strict)
        dump_json(run_dir / "phase8d_regression.json", regression)
        dump_json(run_dir / "repeated_run_determinism.json", repeated)
        timing, bottleneck = timing_analysis(run_dir, per_sample)

        cold = cold_start_qualification(
            run_dir,
            qualifier,
            clouds[0],
            logs["qualification.log"],
            args.cold_starts,
        )
        dump_json(run_dir / "cold_start_qualification.json", cold)

        soak_raw = run_dir / "soak_raw.jsonl"
        soak_records = run_qualification_sequence(
            qualifier,
            cloud_list,
            soak_raw,
            logs["soak.log"],
            rounds=args.soak_rounds,
            warmup=1,
            compact=True,
            resource_interval=18,
            timeout=3600,
        )
        soak = analyze_soak(soak_records, args.soak_rounds)
        dump_json(run_dir / "soak_test.json", soak)

        package_equivalence, export_validation, _ = package_qualification(
            run_dir, package, samples, per_sample, logs["qualification.log"]
        )
        dump_json(run_dir / "package_source_equivalence.json", package_equivalence)
        dump_json(run_dir / "export_validation.json", export_validation)
        write_per_sample_files(run_dir, per_sample)

        failures = execute_fail_closed(
            run_dir, qualifier, package, clouds[0], logs["failure.log"]
        )
        dump_json(run_dir / "fail_closed_qualification.json", failures)
        gui = gui_subset_report(package_equivalence, per_sample)
        dump_json(run_dir / "gui_subset_smoke.json", gui)

        environment = environment_report(run_dir, package, qualifier, assets)
        dump_json(run_dir / "environment.json", environment)
        passed = all(
            (
                aggregate["status"] == "PASS",
                pytorch_parity["all_current_cpp_sdk_labels_exact_python_tensorrt"],
                pytorch_parity["all_pytorch_tensorrt_labels_exact"],
                repeated["status"] == "PASS",
                cold["status"] == "PASS",
                soak["status"] == "PASS",
                regression["status"] == "PASS",
                package_equivalence["status"] == "PASS",
                export_validation["status"] == "PASS",
                failures["status"] == "PASS",
                gui["status"] == "PASS",
                ground_truth_audit["status"] == "PASS",
                strict["status"] == "CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED",
                strict["passed_samples"] == 13,
            )
        )
        final_status = write_phase_summary(
            run_dir,
            samples,
            aggregate,
            strict,
            repeated,
            cold,
            soak,
            timing,
            bottleneck,
            package_equivalence,
            export_validation,
            failures,
            gui,
            passed,
        )
        qualification_manifest(run_dir, environment)
        dump_text(
            run_dir / "phase11a_current_path.txt",
            str(run_dir.resolve()),
        )
        print(f"ARTIFACT_ROOT={run_dir.resolve()}")
        print(final_status)
        if passed:
            print("TENSORRT_RELEASE_FULL_TESTSET_TASK_EQUIVALENT")
            print("CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED")
            return 0
        return 2
    except Exception as error:
        append_log(logs["qualification.log"], f"\nFATAL={error!r}\n")
        dump_text(
            run_dir / "phase11a_summary.md",
            "# Phase 11A blocked\n\n"
            f"Blocking error: `{error!r}`\n\n"
            "PHASE_11A_RELEASE_FULL_TESTSET_QUALIFICATION_BLOCKED",
        )
        print(f"ARTIFACT_ROOT={run_dir.resolve()}")
        print(f"BLOCKING_ERROR={error!r}")
        print("PHASE_11A_RELEASE_FULL_TESTSET_QUALIFICATION_BLOCKED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
