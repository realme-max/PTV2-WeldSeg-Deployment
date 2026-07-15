"""Evaluate the historical GCN_res checkpoint on fixed weld val/test splits."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

assert (PROJECT_ROOT / "models").is_dir(), PROJECT_ROOT / "models"
assert (PROJECT_ROOT / "config").is_dir(), PROJECT_ROOT / "config"
assert (PROJECT_ROOT / "data").is_dir(), PROJECT_ROOT / "data"

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.neighbors import kneighbors_graph
from torch.utils.data import DataLoader, Dataset


SEED = 42
NUM_POINT = 2048
K_NEIGHBORS = 6
BATCH_SIZE = 1
DEVICE = "cuda:0"
NUM_WORKERS = 0
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 50
EXAMPLE_COUNT_PER_SPLIT = 3

DATA_ROOT = PROJECT_ROOT / "data" / "weld"
SPLIT_ROOT = DATA_ROOT / "train_test_split"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "testParameters" / "GCN_res" / "best_model.pth"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_evaluation"

# Evidence-backed mapping only. The repository does not name 0/1 as background/weld.
LABEL_MAPPING = {
    0: "raw_txt_label_0 (semantic name not documented in repository)",
    1: "raw_txt_label_1 (positive class for binary metrics; semantic name not documented)",
}


@dataclass(frozen=True)
class SampleRecord:
    logical_path: str
    file_path: Path
    sample_name: str


class FixedWeldEvaluationDataset(Dataset):
    """Deterministically sample one fixed 2048-point view per JSON entry."""

    def __init__(self, split: str) -> None:
        if split not in ("val", "test"):
            raise ValueError(f"Evaluation split must be val or test, got {split!r}")
        self.split = split
        self.split_file = SPLIT_ROOT / f"sub_shuffled_{split}_file_list.json"
        if not self.split_file.is_file():
            raise FileNotFoundError(f"Split JSON not found: {self.split_file}")
        entries = json.loads(self.split_file.read_text(encoding="utf-8"))
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Expected a non-empty JSON list: {self.split_file}")
        if len(entries) != len(set(entries)):
            raise ValueError(f"Duplicate entries found in {self.split_file}")
        self.records = [self._resolve_entry(str(entry)) for entry in entries]
        missing = [str(item.file_path) for item in self.records if not item.file_path.is_file()]
        if missing:
            raise FileNotFoundError(f"{self.split_file} references missing files: {missing}")

    def _resolve_entry(self, entry: str) -> SampleRecord:
        parts = [part for part in PurePosixPath(entry.replace("\\", "/")).parts if part != "."]
        if parts and parts[0].lower() == DATA_ROOT.name.lower():
            parts = parts[1:]
        if len(parts) < 2:
            raise ValueError(f"Invalid weld split entry: {entry!r}")
        relative = Path(*parts)
        if relative.suffix.lower() != ".txt":
            relative = relative.with_suffix(".txt")
        file_path = (DATA_ROOT / relative).resolve()
        try:
            file_path.relative_to(DATA_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"Split entry escapes data root: {entry!r}") from exc
        return SampleRecord(entry, file_path, file_path.stem)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        data = np.loadtxt(record.file_path, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != 4:
            raise ValueError(f"Expected four TXT columns [x,y,z,label], got {data.shape}: {record.file_path}")
        if not np.isfinite(data).all():
            raise ValueError(f"Non-finite input data: {record.file_path}")

        original_xyz = data[:, :3].copy()
        raw_labels = data[:, 3]
        rounded_labels = np.rint(raw_labels)
        if not np.array_equal(raw_labels, rounded_labels):
            raise ValueError(f"Non-integral labels found in {record.file_path}")
        labels = rounded_labels.astype(np.int64)
        unique_labels = set(np.unique(labels).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Labels outside {{0,1}} in {record.file_path}: {sorted(unique_labels)}")

        normalized_xyz = original_xyz - original_xyz.mean(axis=0, keepdims=True)
        radius = np.sqrt(np.sum(normalized_xyz**2, axis=1)).max()
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError(f"Degenerate normalization radius {radius}: {record.file_path}")
        normalized_xyz /= radius

        split_offset = 1_000_000 if self.split == "val" else 2_000_000
        rng = np.random.default_rng(SEED + split_offset + index)
        choice = rng.choice(len(labels), NUM_POINT, replace=True)
        return {
            "normalized_xyz": torch.from_numpy(normalized_xyz[choice].astype(np.float32, copy=False)),
            "original_xyz": torch.from_numpy(original_xyz[choice].astype(np.float32, copy=False)),
            "labels": torch.from_numpy(labels[choice]),
            "sample_indices": torch.from_numpy(choice.astype(np.int64, copy=False)),
            "sample_name": record.sample_name,
            "logical_path": record.logical_path,
            "split": self.split,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def seed_everything() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def make_run_directory(requested_run_id: str | None) -> Path:
    run_id = requested_run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_historical_checkpoint"
    if any(token in run_id for token in ("/", "\\", "..")):
        raise ValueError(f"Unsafe run ID: {run_id!r}")
    run_dir = ARTIFACTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "predictions").mkdir()
    return run_dir


def make_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"gcn_res_evaluation.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def resolved_config(run_dir: Path) -> dict[str, Any]:
    return {
        "project_root": str(PROJECT_ROOT),
        "data_root": str(DATA_ROOT),
        "val_split": str(SPLIT_ROOT / "sub_shuffled_val_file_list.json"),
        "test_split": str(SPLIT_ROOT / "sub_shuffled_test_file_list.json"),
        "model_source": str(PROJECT_ROOT / "models" / "testParameters" / "GCN_res" / "model.py"),
        "checkpoint": str(CHECKPOINT_PATH),
        "run_directory": str(run_dir),
        "seed": SEED,
        "num_point": NUM_POINT,
        "k_neighbors": K_NEIGHBORS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "device": DEVICE,
        "warmup_iterations": WARMUP_ITERATIONS,
        "benchmark_iterations": BENCHMARK_ITERATIONS,
        "prediction_examples_per_split": EXAMPLE_COUNT_PER_SPLIT,
        "positive_class": 1,
        "label_mapping": {str(key): value for key, value in LABEL_MAPPING.items()},
    }


def save_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def build_adjacency_cpu(normalized_xyz: torch.Tensor) -> tuple[torch.Tensor, float]:
    if normalized_xyz.device.type != "cpu" or normalized_xyz.shape != (BATCH_SIZE, NUM_POINT, 3):
        raise ValueError(f"Expected CPU XYZ [{BATCH_SIZE},{NUM_POINT},3], got {normalized_xyz.shape} on {normalized_xyz.device}")
    start = time.perf_counter()
    matrices = []
    for sample in normalized_xyz.numpy():
        matrix = kneighbors_graph(
            sample,
            n_neighbors=K_NEIGHBORS,
            mode="connectivity",
            include_self=False,
        ).toarray()
        matrices.append(matrix.astype(np.float32, copy=False))
    adjacency = torch.from_numpy(np.stack(matrices, axis=0))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if adjacency.shape != (BATCH_SIZE, NUM_POINT, NUM_POINT):
        raise RuntimeError(f"Unexpected adjacency shape: {adjacency.shape}")
    return adjacency, elapsed_ms


def make_model_input(normalized_xyz: torch.Tensor) -> torch.Tensor:
    category_one_hot = torch.ones(
        (*normalized_xyz.shape[:2], 1), dtype=normalized_xyz.dtype, device=normalized_xyz.device
    )
    points = torch.cat([normalized_xyz, category_one_hot], dim=-1)
    if points.shape != (BATCH_SIZE, NUM_POINT, 4):
        raise RuntimeError(f"Unexpected points shape: {points.shape}")
    return points


def checkpoint_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("epoch", "train_acc", "test_acc", "class_avg_iou", "inctance_avg_iou"):
        value = checkpoint.get(key)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        else:
            result[key] = repr(value)
    return result


def load_historical_model(logger: logging.Logger) -> tuple[torch.nn.Module, dict[str, Any]]:
    from models.testParameters.GCN_res.model import PTV2Segmentation

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Historical checkpoint not found: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise KeyError(f"model_state_dict missing from historical checkpoint: {CHECKPOINT_PATH}")
    state_dict = checkpoint["model_state_dict"]
    if tuple(state_dict["linear_1.weight"].shape) != (48, 4):
        raise RuntimeError(f"linear_1.weight mismatch: {state_dict['linear_1.weight'].shape}")
    if tuple(state_dict["mlp.weight"].shape) != (2, 48):
        raise RuntimeError(f"mlp.weight mismatch: {state_dict['mlp.weight'].shape}")
    non_finite = [name for name, tensor in state_dict.items() if torch.is_tensor(tensor) and not torch.isfinite(tensor).all()]
    if non_finite:
        raise FloatingPointError(f"Non-finite tensors in checkpoint: {non_finite}")

    model = PTV2Segmentation(SimpleNamespace(num_class=2), in_dim=4)
    strict_result = model.load_state_dict(state_dict, strict=True)
    model = model.to(torch.device(DEVICE)).eval()
    metadata = checkpoint_metadata(checkpoint)
    metadata.update(
        {
            "strict_load": str(strict_result),
            "linear_1_weight_shape": list(state_dict["linear_1.weight"].shape),
            "mlp_weight_shape": list(state_dict["mlp.weight"].shape),
            "all_checkpoint_tensors_finite": True,
        }
    )
    logger.info("Historical checkpoint strict=True load: %s", strict_result)
    logger.info("Checkpoint metadata: %s", metadata)
    return model, metadata


def forward_checked(
    model: torch.nn.Module, points_cuda: torch.Tensor, adjacency_cuda: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(points_cuda, adjacency_cuda)
    if not isinstance(output, tuple) or len(output) != 2:
        raise RuntimeError(f"Expected model tuple(points_xyz, logits), got {type(output)}")
    points_xyz, logits = output
    if points_xyz.shape != (BATCH_SIZE, NUM_POINT, 3):
        raise RuntimeError(f"Unexpected output XYZ shape: {points_xyz.shape}")
    if logits.shape != (BATCH_SIZE, NUM_POINT, 2):
        raise RuntimeError(f"Unexpected logits shape: {logits.shape}")
    if not torch.isfinite(points_xyz).all() or not torch.isfinite(logits).all():
        raise FloatingPointError("Model output contains NaN or Inf")
    return points_xyz, logits


def update_confusion(confusion: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> None:
    encoded = target.astype(np.int64) * 2 + prediction.astype(np.int64)
    confusion += np.bincount(encoded, minlength=4).reshape(2, 2)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, Any]:
    confusion = confusion.astype(np.int64, copy=False)
    true_count = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    diagonal = np.diag(confusion)
    ious: list[float] = []
    precision_per_class: list[float] = []
    recall_per_class: list[float] = []
    f1_per_class: list[float] = []
    for class_index in range(2):
        union = true_count[class_index] + predicted_count[class_index] - diagonal[class_index]
        # Matches the legacy training evaluator's convention for an absent class.
        iou = 1.0 if union == 0 else safe_ratio(diagonal[class_index], union)
        precision = safe_ratio(diagonal[class_index], predicted_count[class_index])
        recall = safe_ratio(diagonal[class_index], true_count[class_index])
        f1 = safe_ratio(2.0 * precision * recall, precision + recall)
        ious.append(iou)
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
    return {
        "overall_accuracy": safe_ratio(diagonal.sum(), confusion.sum()),
        "class_0_iou": ious[0],
        "class_1_iou": ious[1],
        "miou": float(np.mean(ious)),
        "confusion_matrix": confusion.tolist(),
        "positive_class": 1,
        "precision": precision_per_class[1],
        "recall": recall_per_class[1],
        "f1": f1_per_class[1],
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "macro_precision": float(np.mean(precision_per_class)),
        "macro_recall": float(np.mean(recall_per_class)),
        "macro_f1": float(np.mean(f1_per_class)),
    }


def evaluate_split(
    split: str,
    dataset: FixedWeldEvaluationDataset,
    loader: DataLoader,
    model: torch.nn.Module,
    predictions_dir: Path,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    total_loss = 0.0
    total_points = 0
    total_adjacency_ms = 0.0
    confusion = np.zeros((2, 2), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    device = torch.device(DEVICE)

    with torch.inference_mode():
        for sample_index, batch in enumerate(loader):
            normalized_xyz = batch["normalized_xyz"].to(torch.float32)
            labels_cpu = batch["labels"].to(torch.long)
            adjacency_cpu, adjacency_ms = build_adjacency_cpu(normalized_xyz)
            points_cpu = make_model_input(normalized_xyz)
            points_cuda = points_cpu.to(device, non_blocking=True)
            adjacency_cuda = adjacency_cpu.to(device, non_blocking=True)
            labels_cuda = labels_cpu.to(device, non_blocking=True)
            _, logits = forward_checked(model, points_cuda, adjacency_cuda)
            loss = F.cross_entropy(logits.reshape(-1, 2), labels_cuda.reshape(-1), reduction="mean")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for {split} sample {sample_index}")
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError(f"Non-finite probabilities for {split} sample {sample_index}")

            target_np = labels_cpu[0].numpy()
            prediction_np = prediction[0].cpu().numpy()
            probability_np = probabilities[0, :, 1].cpu().numpy()
            sample_confusion = np.zeros((2, 2), dtype=np.int64)
            update_confusion(sample_confusion, target_np, prediction_np)
            update_confusion(confusion, target_np, prediction_np)
            sample_metrics = metrics_from_confusion(sample_confusion)
            sample_name = str(batch["sample_name"][0])
            row = {
                "sample_name": sample_name,
                "split": split,
                "loss": float(loss.item()),
                "accuracy": sample_metrics["overall_accuracy"],
                "class_0_iou": sample_metrics["class_0_iou"],
                "class_1_iou": sample_metrics["class_1_iou"],
                "miou": sample_metrics["miou"],
                "predicted_positive_points": int((prediction_np == 1).sum()),
                "ground_truth_positive_points": int((target_np == 1).sum()),
                "precision": sample_metrics["precision"],
                "recall": sample_metrics["recall"],
                "f1": sample_metrics["f1"],
            }
            rows.append(row)
            total_loss += float(loss.item()) * int(target_np.size)
            total_points += int(target_np.size)
            total_adjacency_ms += adjacency_ms

            if sample_index < EXAMPLE_COUNT_PER_SPLIT:
                output_path = predictions_dir / f"{split}_{sample_index:02d}_{sample_name}.npz"
                np.savez_compressed(
                    output_path,
                    original_xyz=batch["original_xyz"][0].numpy(),
                    normalized_xyz=normalized_xyz[0].numpy(),
                    ground_truth_labels=target_np,
                    predicted_labels=prediction_np,
                    class_1_probability=probability_np,
                    logits=logits[0].cpu().numpy(),
                    sample_indices=batch["sample_indices"][0].numpy(),
                    sample_name=np.asarray(sample_name),
                    split=np.asarray(split),
                )
            logger.info(
                "%s %02d/%02d sample=%s loss=%.6f acc=%.6f mIoU=%.6f pred_pos=%d gt_pos=%d adj=%.3fms",
                split,
                sample_index + 1,
                len(dataset),
                sample_name,
                row["loss"],
                row["accuracy"],
                row["miou"],
                row["predicted_positive_points"],
                row["ground_truth_positive_points"],
                adjacency_ms,
            )

    split_metrics = metrics_from_confusion(confusion)
    split_metrics["average_loss"] = total_loss / total_points
    split_metrics["sample_count"] = len(dataset)
    split_metrics["point_count"] = total_points
    split_metrics["cpu_adjacency_total_ms"] = total_adjacency_ms
    split_metrics["cpu_adjacency_mean_ms"] = total_adjacency_ms / len(dataset)
    logger.info("%s aggregate metrics: %s", split, split_metrics)
    return split_metrics, rows, confusion


def write_per_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_name",
        "split",
        "loss",
        "accuracy",
        "class_0_iou",
        "class_1_iou",
        "miou",
        "predicted_positive_points",
        "ground_truth_positive_points",
        "precision",
        "recall",
        "f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(path: Path, matrices: dict[str, np.ndarray]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "ground_truth_class", "predicted_class_0", "predicted_class_1"])
        for split, matrix in matrices.items():
            writer.writerow([split, 0, int(matrix[0, 0]), int(matrix[0, 1])])
            writer.writerow([split, 1, int(matrix[1, 0]), int(matrix[1, 1])])


def distribution_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size != BENCHMARK_ITERATIONS or not np.isfinite(array).all():
        raise RuntimeError(f"Invalid benchmark values: count={array.size}, finite={np.isfinite(array).all()}")
    return {
        "mean": float(np.mean(array)),
        "median_p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def benchmark_once(
    model: torch.nn.Module, dataset: FixedWeldEvaluationDataset, sample_index: int
) -> dict[str, float]:
    total_start = time.perf_counter()
    sample = dataset[sample_index]
    normalized_xyz = sample["normalized_xyz"].unsqueeze(0).to(torch.float32)
    points_cpu = make_model_input(normalized_xyz)
    adjacency_cpu, adjacency_ms = build_adjacency_cpu(normalized_xyz)
    points_cpu = points_cpu.pin_memory()
    adjacency_cpu = adjacency_cpu.pin_memory()

    h2d_start = torch.cuda.Event(enable_timing=True)
    h2d_end = torch.cuda.Event(enable_timing=True)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    h2d_start.record()
    points_cuda = points_cpu.to(DEVICE, non_blocking=True)
    adjacency_cuda = adjacency_cpu.to(DEVICE, non_blocking=True)
    h2d_end.record()
    forward_start.record()
    with torch.inference_mode():
        output = model(points_cuda, adjacency_cuda)
    forward_end.record()
    if not isinstance(output, tuple) or len(output) != 2:
        raise RuntimeError(f"Expected model tuple(points_xyz, logits), got {type(output)}")
    _, logits = output
    if logits.shape != (BATCH_SIZE, NUM_POINT, 2):
        raise RuntimeError(f"Unexpected benchmark logits shape: {logits.shape}")
    _ = torch.argmax(logits, dim=-1).cpu()
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - total_start) * 1000.0
    return {
        "cpu_adjacency_ms": adjacency_ms,
        "host_to_device_ms": float(h2d_start.elapsed_time(h2d_end)),
        "cuda_forward_ms": float(forward_start.elapsed_time(forward_end)),
        "end_to_end_ms": total_ms,
    }


def run_benchmark(
    model: torch.nn.Module, dataset: FixedWeldEvaluationDataset, logger: logging.Logger
) -> dict[str, Any]:
    for _ in range(WARMUP_ITERATIONS):
        benchmark_once(model, dataset, 0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(torch.device(DEVICE))

    samples = {
        "cpu_adjacency_ms": [],
        "host_to_device_ms": [],
        "cuda_forward_ms": [],
        "end_to_end_ms": [],
    }
    for _ in range(BENCHMARK_ITERATIONS):
        measured = benchmark_once(model, dataset, 0)
        for key, value in measured.items():
            samples[key].append(value)
    peak_mib = torch.cuda.max_memory_allocated(torch.device(DEVICE)) / (1024**2)
    result = {
        "benchmark_sample": dataset.records[0].sample_name,
        "benchmark_split": dataset.split,
        "warmup_iterations": WARMUP_ITERATIONS,
        "measured_iterations": BENCHMARK_ITERATIONS,
        "units": "milliseconds",
        "timing_definition": {
            "cpu_adjacency": "sklearn kneighbors_graph plus dense float32 conversion",
            "host_to_device": "CUDA-event time for points and dense adjacency copies",
            "cuda_forward": "CUDA-event time for GCN_res model forward only",
            "end_to_end": "wall time including TXT load, deterministic sampling, normalization, adjacency, pinning, H2D, forward, argmax and D2H",
        },
        "cpu_adjacency": distribution_stats(samples["cpu_adjacency_ms"]),
        "host_to_device": distribution_stats(samples["host_to_device_ms"]),
        "cuda_forward": distribution_stats(samples["cuda_forward_ms"]),
        "end_to_end": distribution_stats(samples["end_to_end_ms"]),
        "gpu_peak_memory_mib": float(peak_mib),
    }
    logger.info("Benchmark result: %s", result)
    return result


def verify_reproducibility(
    dataset: FixedWeldEvaluationDataset, logger: logging.Logger
) -> dict[str, Any]:
    reloaded_model, reloaded_metadata = load_historical_model(logger)
    sample = dataset[0]
    normalized_xyz = sample["normalized_xyz"].unsqueeze(0).to(torch.float32)
    points_cuda = make_model_input(normalized_xyz).to(DEVICE)
    adjacency_cuda = build_adjacency_cpu(normalized_xyz)[0].to(DEVICE)
    with torch.inference_mode():
        _, first = forward_checked(reloaded_model, points_cuda, adjacency_cuda)
        _, second = forward_checked(reloaded_model, points_cuda, adjacency_cuda)
    torch.cuda.synchronize()
    first_cpu = first.cpu()
    second_cpu = second.cpu()
    finite = bool(torch.isfinite(first_cpu).all() and torch.isfinite(second_cpu).all())
    exact = bool(torch.equal(first_cpu, second_cpu))
    max_absolute_difference = float(torch.max(torch.abs(first_cpu - second_cpu)).item())
    allclose = bool(torch.allclose(first_cpu, second_cpu, rtol=1e-5, atol=1e-6))
    if not finite or not allclose:
        raise RuntimeError(
            f"Reproducibility check failed: finite={finite}, exact={exact}, "
            f"allclose={allclose}, max_abs={max_absolute_difference}"
        )
    result = {
        "sample_name": dataset.records[0].sample_name,
        "split": dataset.split,
        "checkpoint_reloaded_strict": reloaded_metadata["strict_load"],
        "outputs_finite": finite,
        "exact_equal": exact,
        "allclose_rtol_1e-5_atol_1e-6": allclose,
        "max_absolute_difference": max_absolute_difference,
    }
    logger.info("Reproducibility result: %s", result)
    del reloaded_model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    run_dir = make_run_directory(args.run_id)
    logger = make_logger(run_dir)
    config = resolved_config(run_dir)
    OmegaConf.save(OmegaConf.create(config), run_dir / "config_resolved.yaml", resolve=True)
    try:
        seed_everything()
        logger.info(
            "Startup PROJECT_ROOT=%s sys.path[0]=%s cwd=%s python=%s",
            PROJECT_ROOT,
            sys.path[0],
            Path.cwd(),
            sys.executable,
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        logger.info(
            "Environment torch=%s cuda_runtime=%s gpu=%s capability=%s",
            torch.__version__,
            torch.version.cuda,
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_capability(0),
        )
        logger.info("Label mapping: %s", LABEL_MAPPING)

        val_dataset = FixedWeldEvaluationDataset("val")
        test_dataset = FixedWeldEvaluationDataset("test")
        if len(val_dataset) != 18 or len(test_dataset) != 18:
            raise RuntimeError(f"Expected val/test counts 18/18, got {len(val_dataset)}/{len(test_dataset)}")
        overlap = {item.logical_path for item in val_dataset.records} & {
            item.logical_path for item in test_dataset.records
        }
        if overlap:
            raise RuntimeError(f"Val/test split overlap: {sorted(overlap)}")
        logger.info("Fixed split counts val=%d test=%d overlap=0", len(val_dataset), len(test_dataset))

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        model, metadata = load_historical_model(logger)
        torch.cuda.reset_peak_memory_stats(torch.device(DEVICE))

        predictions_dir = run_dir / "predictions"
        val_metrics, val_rows, val_confusion = evaluate_split(
            "val", val_dataset, val_loader, model, predictions_dir, logger
        )
        test_metrics, test_rows, test_confusion = evaluate_split(
            "test", test_dataset, test_loader, model, predictions_dir, logger
        )
        evaluation_peak_mib = torch.cuda.max_memory_allocated(torch.device(DEVICE)) / (1024**2)

        all_rows = val_rows + test_rows
        write_per_sample_csv(run_dir / "per_sample_metrics.csv", all_rows)
        write_confusion_csv(
            run_dir / "confusion_matrix.csv", {"val": val_confusion, "test": test_confusion}
        )

        benchmark = run_benchmark(model, val_dataset, logger)
        save_json(run_dir / "benchmark.json", benchmark)
        reproducibility = verify_reproducibility(val_dataset, logger)
        overall_peak_mib = max(evaluation_peak_mib, float(benchmark["gpu_peak_memory_mib"]))
        metrics = {
            "status": "GCN_RES_CHECKPOINT_EVALUATION_PASSED",
            "checkpoint": str(CHECKPOINT_PATH),
            "checkpoint_metadata": metadata,
            "label_mapping": {str(key): value for key, value in LABEL_MAPPING.items()},
            "positive_class_for_precision_recall_f1": 1,
            "val": val_metrics,
            "test": test_metrics,
            "evaluation_gpu_peak_memory_mib": float(evaluation_peak_mib),
            "overall_gpu_peak_memory_mib": float(overall_peak_mib),
            "prediction_npz_count": len(list(predictions_dir.glob("*.npz"))),
            "reproducibility": reproducibility,
        }
        save_json(run_dir / "metrics.json", metrics)

        required = [
            "metrics.json",
            "confusion_matrix.csv",
            "per_sample_metrics.csv",
            "benchmark.json",
            "config_resolved.yaml",
            "run.log",
        ]
        missing_outputs = [name for name in required if not (run_dir / name).is_file()]
        if missing_outputs:
            raise RuntimeError(f"Required outputs missing: {missing_outputs}")
        if metrics["prediction_npz_count"] != 6:
            raise RuntimeError(f"Expected six prediction NPZ files, got {metrics['prediction_npz_count']}")
        logger.info("GCN_RES_CHECKPOINT_EVALUATION_PASSED artifact_dir=%s", run_dir)
        print(f"GCN_RES_CHECKPOINT_EVALUATION_PASSED\nARTIFACT_DIR={run_dir}")
        return 0
    except torch.cuda.OutOfMemoryError:
        logger.exception("CUDA_OUT_OF_MEMORY: evaluation stopped without changing fixed settings")
        raise
    except Exception:
        logger.exception("GCN_RES_CHECKPOINT_EVALUATION_FAILED")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
