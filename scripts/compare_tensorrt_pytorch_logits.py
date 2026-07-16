"""Compare one fixed-shape TensorRT FP32 output with a PyTorch reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def segmentation_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if labels.shape != predictions.shape:
        raise ValueError(f"Label shape mismatch: {labels.shape} != {predictions.shape}")
    if not np.isin(labels, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Expected binary labels 0=weld_seam, 1=background")
    confusion = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    ious: list[float | None] = []
    for class_index in (0, 1):
        true_positive = int(confusion[class_index, class_index])
        false_negative = int(confusion[class_index, :].sum() - true_positive)
        false_positive = int(confusion[:, class_index].sum() - true_positive)
        ious.append(safe_ratio(true_positive, true_positive + false_negative + false_positive))

    # Weld seam is class 0 and is the positive class for precision/recall/F1.
    weld_tp = int(confusion[0, 0])
    weld_fn = int(confusion[0, 1])
    weld_fp = int(confusion[1, 0])
    precision = safe_ratio(weld_tp, weld_tp + weld_fp)
    recall = safe_ratio(weld_tp, weld_tp + weld_fn)
    f1 = (
        safe_ratio(2.0 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None
    )
    valid_ious = [item for item in ious if item is not None]
    return {
        "label_semantics": {"0": "weld_seam", "1": "background"},
        "confusion_matrix_rows_ground_truth_columns_prediction": confusion.tolist(),
        "overall_accuracy": float((labels == predictions).mean()),
        "weld_seam_iou": ious[0],
        "background_iou": ious[1],
        "miou": float(np.mean(valid_ious)) if valid_ious else None,
        "weld_seam_precision": precision,
        "weld_seam_recall": recall,
        "weld_seam_f1": f1,
    }


def compare(
    tensorrt_path: Path,
    pytorch_path: Path,
    labels_path: Path | None,
    max_abs_threshold: float,
    cosine_threshold: float,
) -> dict[str, Any]:
    tensorrt_logits = np.load(tensorrt_path, allow_pickle=False)
    pytorch_logits = np.load(pytorch_path, allow_pickle=False)
    if tensorrt_logits.shape != (1, 2048, 2):
        raise ValueError(f"Unexpected TensorRT shape: {tensorrt_logits.shape}")
    if pytorch_logits.shape != tensorrt_logits.shape:
        raise ValueError(
            f"PyTorch/TensorRT shape mismatch: {pytorch_logits.shape} != {tensorrt_logits.shape}"
        )
    if tensorrt_logits.dtype != np.float32 or pytorch_logits.dtype != np.float32:
        raise TypeError(
            f"Expected FP32 logits, got TRT={tensorrt_logits.dtype}, PT={pytorch_logits.dtype}"
        )

    finite = bool(np.isfinite(tensorrt_logits).all() and np.isfinite(pytorch_logits).all())
    difference = tensorrt_logits.astype(np.float64) - pytorch_logits.astype(np.float64)
    absolute = np.abs(difference)
    denominator = np.maximum(np.abs(pytorch_logits.astype(np.float64)), 1.0e-8)
    relative = absolute / denominator
    trt_flat = tensorrt_logits.astype(np.float64).reshape(-1)
    torch_flat = pytorch_logits.astype(np.float64).reshape(-1)
    cosine_denominator = float(np.linalg.norm(trt_flat) * np.linalg.norm(torch_flat))
    cosine = safe_ratio(float(np.dot(trt_flat, torch_flat)), cosine_denominator)
    relative_l2 = safe_ratio(float(np.linalg.norm(difference)), float(np.linalg.norm(torch_flat)))

    trt_labels = np.argmax(tensorrt_logits, axis=-1)
    torch_labels = np.argmax(pytorch_logits, axis=-1)
    matching = int((trt_labels == torch_labels).sum())
    total = int(trt_labels.size)
    max_absolute = float(absolute.max())
    passed = bool(
        finite
        and cosine is not None
        and max_absolute < max_abs_threshold
        and cosine > cosine_threshold
    )
    result: dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": (
            "TENSORRT_FP32_INFERENCE_PARITY_PASSED"
            if passed
            else "TENSORRT_FP32_NUMERICAL_PARITY_FAILED"
        ),
        "tensorrt_logits": {
            "path": str(tensorrt_path),
            "sha256": sha256(tensorrt_path),
            "shape": list(tensorrt_logits.shape),
            "dtype": str(tensorrt_logits.dtype),
            "finite": bool(np.isfinite(tensorrt_logits).all()),
            "min": float(tensorrt_logits.min()),
            "max": float(tensorrt_logits.max()),
            "mean": float(tensorrt_logits.mean()),
            "std": float(tensorrt_logits.std()),
        },
        "pytorch_logits": {
            "path": str(pytorch_path),
            "sha256": sha256(pytorch_path),
            "shape": list(pytorch_logits.shape),
            "dtype": str(pytorch_logits.dtype),
            "finite": bool(np.isfinite(pytorch_logits).all()),
            "min": float(pytorch_logits.min()),
            "max": float(pytorch_logits.max()),
            "mean": float(pytorch_logits.mean()),
            "std": float(pytorch_logits.std()),
        },
        "numerical_comparison": {
            "max_absolute_error": max_absolute,
            "mean_absolute_error": float(absolute.mean()),
            "rmse": float(np.sqrt(np.mean(np.square(difference)))),
            "cosine_similarity": cosine,
            "max_relative_error_denominator_clamped_1e-8": float(relative.max()),
            "mean_relative_error_denominator_clamped_1e-8": float(relative.mean()),
            "relative_l2_error": relative_l2,
            "outputs_finite": finite,
        },
        "classification_agreement": {
            "matching_points": matching,
            "total_points": total,
            "agreement": float(matching / total),
        },
        "acceptance": {
            "max_absolute_error_strictly_less_than": max_abs_threshold,
            "cosine_similarity_strictly_greater_than": cosine_threshold,
            "outputs_must_be_finite": True,
            "passed": passed,
        },
    }
    if labels_path is not None:
        with np.load(labels_path, allow_pickle=False) as input_archive:
            if "ground_truth_labels" not in input_archive:
                raise KeyError(f"ground_truth_labels missing from {labels_path}")
            labels = np.asarray(input_archive["ground_truth_labels"])
        result["ground_truth"] = {
            "source": str(labels_path),
            "source_sha256": sha256(labels_path),
            "tensorrt_metrics": segmentation_metrics(labels, trt_labels),
            "pytorch_metrics": segmentation_metrics(labels, torch_labels),
        }
    return result


def main(args: argparse.Namespace) -> int:
    tensorrt_path = args.tensorrt.resolve()
    pytorch_path = args.pytorch.resolve()
    labels_path = args.labels.resolve() if args.labels else None
    for path in (tensorrt_path, pytorch_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if labels_path is not None and not labels_path.is_file():
        raise FileNotFoundError(labels_path)
    result = compare(
        tensorrt_path,
        pytorch_path,
        labels_path,
        args.max_abs_threshold,
        args.cosine_threshold,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    comparison = result["numerical_comparison"]
    agreement = result["classification_agreement"]
    print(f"MAX_ABSOLUTE_ERROR={comparison['max_absolute_error']:.9e}")
    print(f"MEAN_ABSOLUTE_ERROR={comparison['mean_absolute_error']:.9e}")
    print(f"RMSE={comparison['rmse']:.9e}")
    print(f"COSINE_SIMILARITY={comparison['cosine_similarity']:.12f}")
    print(f"LABEL_AGREEMENT={agreement['agreement']:.12f}")
    print(result["status"])
    return 0 if result["acceptance"]["passed"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorrt", type=Path, required=True)
    parser.add_argument("--pytorch", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-abs-threshold", type=float, default=1.0e-4)
    parser.add_argument("--cosine-threshold", type=float, default=0.9999)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
