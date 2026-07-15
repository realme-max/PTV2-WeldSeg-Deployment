"""Independent GCN_res smoke trainer for the validated weld sub-dataset.

This entry point intentionally does not use or modify the legacy WeldDataset,
training scripts, model sources, or original shuffled_*.json splits.
"""

from __future__ import annotations

import argparse
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
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

assert (PROJECT_ROOT / "models").is_dir(), f"Project models directory not found: {PROJECT_ROOT / 'models'}"
assert (PROJECT_ROOT / "config").is_dir(), f"Project config directory not found: {PROJECT_ROOT / 'config'}"
assert (PROJECT_ROOT / "data").is_dir(), f"Project data directory not found: {PROJECT_ROOT / 'data'}"

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn.neighbors import kneighbors_graph
from torch.utils.data import DataLoader, Dataset


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "gcn_res_subdataset_smoke.yaml"


@dataclass(frozen=True)
class SampleRecord:
    logical_path: str
    file_path: Path


class WeldSubDataset(Dataset):
    """Read only the validated sub_shuffled_{split}_file_list.json files."""

    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        root: Path,
        split: str,
        num_point: int,
        seed: int,
        split_directory: str = "train_test_split",
        split_prefix: str = "sub_shuffled",
    ) -> None:
        if split not in self.SPLITS:
            raise ValueError(f"Unsupported split {split!r}; expected one of {self.SPLITS}")
        if num_point <= 0:
            raise ValueError(f"num_point must be positive, got {num_point}")

        self.root = root.resolve()
        self.split = split
        self.num_point = int(num_point)
        self.seed = int(seed)
        self.epoch = 0
        self.split_file = (
            self.root
            / split_directory
            / f"{split_prefix}_{split}_file_list.json"
        )
        if not self.split_file.is_file():
            raise FileNotFoundError(f"Sub-dataset split file not found: {self.split_file}")

        entries = json.loads(self.split_file.read_text(encoding="utf-8"))
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Split must contain a non-empty JSON list: {self.split_file}")
        if len(entries) != len(set(entries)):
            raise ValueError(f"Duplicate entries inside split: {self.split_file}")

        self.records = [self._resolve_entry(str(entry)) for entry in entries]
        missing = [str(record.file_path) for record in self.records if not record.file_path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Split {self.split_file} references {len(missing)} missing files: {missing}"
            )

    def _resolve_entry(self, entry: str) -> SampleRecord:
        # Current JSON format is ./weld/000001/weld_59 (without .txt).
        parts = [part for part in PurePosixPath(entry.replace("\\", "/")).parts if part != "."]
        if not parts:
            raise ValueError(f"Invalid empty split entry in {self.split_file}: {entry!r}")
        if parts[0].lower() == self.root.name.lower():
            parts = parts[1:]
        if len(parts) < 2:
            raise ValueError(
                f"Split entry does not match ./weld/<category>/<sample>: {entry!r}"
            )
        relative = Path(*parts)
        if relative.suffix.lower() != ".txt":
            relative = relative.with_suffix(".txt")
        file_path = (self.root / relative).resolve()
        try:
            file_path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Split entry escapes data root: {entry!r}") from exc
        return SampleRecord(logical_path=entry, file_path=file_path)

    @property
    def logical_paths(self) -> set[str]:
        return {record.logical_path for record in self.records}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        data = np.loadtxt(record.file_path, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != 4:
            raise ValueError(
                f"Expected exactly four columns [x,y,z,label] in {record.file_path}; "
                f"got shape {data.shape}"
            )
        if not np.isfinite(data).all():
            raise ValueError(f"Non-finite data found in {record.file_path}")

        labels = data[:, 3].astype(np.int64)
        unique_labels = set(np.unique(labels).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                f"Labels outside {{0,1}} in {record.file_path}: {sorted(unique_labels)}"
            )

        xyz = data[:, :3].copy()
        xyz -= xyz.mean(axis=0, keepdims=True)
        radius = np.sqrt(np.sum(xyz**2, axis=1)).max()
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError(f"Degenerate point cloud radius in {record.file_path}: {radius}")
        xyz /= radius

        # Deterministic per-sample/per-epoch resampling. Validation/test stay fixed.
        epoch_component = self.epoch if self.split == "train" else 0
        split_offset = {"train": 0, "val": 1_000_000, "test": 2_000_000}[self.split]
        rng = np.random.default_rng(
            self.seed + split_offset + epoch_component * len(self.records) + index
        )
        choice = rng.choice(len(labels), self.num_point, replace=True)

        return {
            "points": torch.from_numpy(xyz[choice].astype(np.float32, copy=False)),
            "cls": torch.tensor([0], dtype=torch.long),
            "seg": torch.from_numpy(labels[choice]),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "logical_path": record.logical_path,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=("from_scratch", "resume_checkpoint"),
        default=None,
        help="Override config mode.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(args: argparse.Namespace) -> DictConfig:
    config_path = resolve_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    if args.mode is not None:
        cfg.mode = args.mode
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    cfg.dry_run = bool(args.dry_run)
    cfg.config_source = str(config_path)
    cfg.project_root = str(PROJECT_ROOT)
    cfg.data_root = str(resolve_path(cfg.data_root))
    cfg.checkpoint_path = str(resolve_path(cfg.checkpoint_path))
    cfg.artifacts_root = str(resolve_path(cfg.artifacts_root))

    required_positive = (
        "batch_size",
        "epochs",
        "num_point",
        "k_neighbors",
        "learning_rate",
        "num_class",
        "in_dim",
    )
    for key in required_positive:
        if float(cfg[key]) <= 0:
            raise ValueError(f"Config {key} must be positive, got {cfg[key]}")
    if cfg.optimizer != "Adam":
        raise ValueError(f"Only Adam is supported by this smoke entry, got {cfg.optimizer!r}")
    if cfg.mode not in ("from_scratch", "resume_checkpoint"):
        raise ValueError(f"Unsupported mode: {cfg.mode!r}")
    if int(cfg.num_class) != 2 or int(cfg.in_dim) != 4:
        raise ValueError(
            f"GCN_res weld contract requires num_class=2 and in_dim=4; "
            f"got {cfg.num_class}, {cfg.in_dim}"
        )
    if int(cfg.num_point) != 2048:
        raise ValueError(f"This smoke run is locked to num_point=2048, got {cfg.num_point}")
    if int(cfg.k_neighbors) != 6:
        raise ValueError(f"This smoke run is locked to k_neighbors=6, got {cfg.k_neighbors}")
    if int(cfg.num_workers) != 0:
        raise ValueError(f"Reproducible smoke config requires num_workers=0, got {cfg.num_workers}")
    return cfg


def make_run_directory(cfg: DictConfig, requested_run_id: str | None) -> Path:
    suffix = "dryrun" if cfg.dry_run else "train"
    run_id = requested_run_id or (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{cfg.mode}_{suffix}"
    )
    if any(char in run_id for char in ("/", "\\", "..")):
        raise ValueError(f"Unsafe run_id: {run_id!r}")
    run_dir = Path(cfg.artifacts_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg.run_id = run_id
    cfg.run_directory = str(run_dir)
    OmegaConf.save(cfg, run_dir / "config_resolved.yaml", resolve=True)
    return run_dir


def make_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"gcn_res_smoke.{run_dir.name}")
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


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def validate_split_disjointness(datasets: Iterable[WeldSubDataset]) -> None:
    datasets = list(datasets)
    for i, left in enumerate(datasets):
        for right in datasets[i + 1 :]:
            overlap = left.logical_paths & right.logical_paths
            if overlap:
                raise ValueError(
                    f"Split overlap between {left.split} and {right.split}: {sorted(overlap)}"
                )


def build_dense_adjacency_cpu(xyz: torch.Tensor, k: int) -> tuple[torch.Tensor, float]:
    if xyz.device.type != "cpu" or xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected CPU XYZ [B,N,3], got {xyz.shape} on {xyz.device}")
    if k >= xyz.shape[1]:
        raise ValueError(f"k={k} must be smaller than N={xyz.shape[1]}")
    start = time.perf_counter()
    matrices = []
    xyz_numpy = xyz.detach().numpy()
    for sample in xyz_numpy:
        matrix = kneighbors_graph(
            sample,
            n_neighbors=k,
            mode="connectivity",
            include_self=False,
        ).toarray()
        matrices.append(matrix.astype(np.float32, copy=False))
    adjacency = torch.from_numpy(np.stack(matrices, axis=0))
    elapsed = time.perf_counter() - start
    return adjacency, elapsed


def confusion_update(confusion: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor) -> None:
    encoded = target.to(torch.int64) * 2 + prediction.to(torch.int64)
    confusion += torch.bincount(encoded.cpu(), minlength=4).reshape(2, 2)


def mean_iou(confusion: torch.Tensor) -> tuple[float, list[float]]:
    confusion = confusion.to(torch.float64)
    intersection = confusion.diag()
    union = confusion.sum(dim=1) + confusion.sum(dim=0) - intersection
    valid = union > 0
    per_class = torch.full((2,), float("nan"), dtype=torch.float64)
    per_class[valid] = intersection[valid] / union[valid]
    return float(per_class[valid].mean().item()), per_class.tolist()


def make_model_and_optimizer(cfg: DictConfig, logger: logging.Logger):
    from models.testParameters.GCN_res.model import PTV2Segmentation

    model = PTV2Segmentation(SimpleNamespace(num_class=2), in_dim=4)
    checkpoint_metadata: dict[str, Any] | None = None
    if cfg.mode == "resume_checkpoint":
        checkpoint_path = Path(cfg.checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"model_state_dict missing from {checkpoint_path}")
        result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        logger.info("MODE=resume_checkpoint strict_load=%s source=%s", result, checkpoint_path)
        checkpoint_metadata = {
            key: checkpoint.get(key)
            for key in (
                "epoch",
                "train_acc",
                "test_acc",
                "class_avg_iou",
                "inctance_avg_iou",
            )
        }
    else:
        checkpoint = None
        logger.info("MODE=from_scratch random initialization seed=%d", int(cfg.seed))

    device = torch.device(str(cfg.device))
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError(f"Smoke run requires cuda:0, got {device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    if cfg.mode == "resume_checkpoint":
        if "optimizer_state_dict" not in checkpoint:
            raise KeyError(f"optimizer_state_dict missing from {cfg.checkpoint_path}")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = float(cfg.learning_rate)
            group["weight_decay"] = float(cfg.weight_decay)
        logger.info(
            "Resume optimizer state loaded; lr overridden to %.6g, weight_decay=%.6g",
            float(cfg.learning_rate),
            float(cfg.weight_decay),
        )
    return model, optimizer, device, checkpoint_metadata


def make_network_input(points_cpu: torch.Tensor, cls_cpu: torch.Tensor, device: torch.device):
    points_xyz = points_cpu.to(device, non_blocking=True)
    cls = cls_cpu.squeeze(-1).to(device, non_blocking=True)
    if not torch.all(cls == 0):
        raise ValueError(f"Weld category indices must all be zero, got {cls.tolist()}")
    one_hot = F.one_hot(cls, num_classes=1).to(dtype=points_xyz.dtype)
    one_hot = one_hot.unsqueeze(1).expand(-1, points_xyz.shape[1], -1)
    network_input = torch.cat([points_xyz, one_hot], dim=-1)
    if network_input.shape[-1] != 4:
        raise RuntimeError(f"Expected four input features, got {network_input.shape}")
    return network_input


def finite_or_raise(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"Non-finite values detected in {name}")


def run_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    batch: dict[str, Any],
    device: torch.device,
    k_neighbors: int,
    training: bool,
) -> dict[str, Any]:
    points_cpu = batch["points"]
    seg = batch["seg"].to(device, non_blocking=True)
    adjacency_cpu, adjacency_seconds = build_dense_adjacency_cpu(
        points_cpu[:, :, :3], k_neighbors
    )
    points = make_network_input(points_cpu, batch["cls"], device)
    adjacency = adjacency_cpu.to(device, non_blocking=True)

    if training:
        if optimizer is None:
            raise ValueError("Training batch requires an optimizer")
        optimizer.zero_grad(set_to_none=True)
        _, logits = model(points, adjacency)
        finite_or_raise("logits", logits)
        loss = F.cross_entropy(logits.reshape(-1, 2), seg.reshape(-1))
        finite_or_raise("loss", loss)
        loss.backward()
        optimizer.step()
    else:
        with torch.inference_mode():
            _, logits = model(points, adjacency)
            finite_or_raise("logits", logits)
            loss = F.cross_entropy(logits.reshape(-1, 2), seg.reshape(-1))
            finite_or_raise("loss", loss)

    prediction = logits.argmax(dim=-1)
    correct = int((prediction == seg).sum().item())
    point_count = int(seg.numel())
    return {
        "loss": float(loss.item()),
        "correct": correct,
        "point_count": point_count,
        "prediction": prediction.detach(),
        "target": seg.detach(),
        "adjacency_seconds": adjacency_seconds,
        "points_shape": tuple(points.shape),
        "adjacency_shape": tuple(adjacency.shape),
        "logits_shape": tuple(logits.shape),
        "points_device": str(points.device),
        "adjacency_device": str(adjacency.device),
        "logits_device": str(logits.device),
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    k_neighbors: int,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_points = 0
    adjacency_seconds = 0.0
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    first_shapes = None

    for batch in loader:
        result = run_batch(
            model=model,
            optimizer=optimizer,
            batch=batch,
            device=device,
            k_neighbors=k_neighbors,
            training=training,
        )
        total_loss += result["loss"] * result["point_count"]
        total_correct += result["correct"]
        total_points += result["point_count"]
        adjacency_seconds += result["adjacency_seconds"]
        confusion_update(confusion, result["target"].reshape(-1), result["prediction"].reshape(-1))
        if first_shapes is None:
            first_shapes = {
                key: result[key]
                for key in (
                    "points_shape",
                    "adjacency_shape",
                    "logits_shape",
                    "points_device",
                    "adjacency_device",
                    "logits_device",
                )
            }

    if total_points == 0:
        raise RuntimeError("DataLoader produced no points")
    miou, per_class_iou = mean_iou(confusion)
    return {
        "loss": total_loss / total_points,
        "accuracy": total_correct / total_points,
        "miou": miou,
        "per_class_iou": per_class_iou,
        "adjacency_cpu_seconds": adjacency_seconds,
        "point_count": total_points,
        "confusion_matrix": confusion.tolist(),
        "first_batch": first_shapes,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def atomic_json_dump(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    cfg: DictConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "mode": str(cfg.mode),
        "dry_run": bool(cfg.dry_run),
        "source_checkpoint": str(cfg.checkpoint_path) if cfg.mode == "resume_checkpoint" else None,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": json_safe(metrics),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


def verify_saved_checkpoint(path: Path) -> str:
    from models.testParameters.GCN_res.model import PTV2Segmentation

    saved = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in saved:
        raise KeyError(f"Saved checkpoint has no model_state_dict: {path}")
    reloaded = PTV2Segmentation(SimpleNamespace(num_class=2), in_dim=4)
    result = reloaded.load_state_dict(saved["model_state_dict"], strict=True)
    for key, value in saved["model_state_dict"].items():
        if torch.is_tensor(value) and not torch.isfinite(value).all().item():
            raise FloatingPointError(f"Non-finite saved tensor {key} in {path}")
    return str(result)


def make_datasets_and_loaders(cfg: DictConfig):
    common = {
        "root": Path(cfg.data_root),
        "num_point": int(cfg.num_point),
        "seed": int(cfg.seed),
        "split_directory": str(cfg.split_directory),
        "split_prefix": str(cfg.split_prefix),
    }
    train_dataset = WeldSubDataset(split="train", **common)
    val_dataset = WeldSubDataset(split="val", **common)
    test_dataset = WeldSubDataset(split="test", **common)
    validate_split_disjointness((train_dataset, val_dataset, test_dataset))

    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))
    loader_common = {
        "batch_size": int(cfg.batch_size),
        "num_workers": int(cfg.num_workers),
        "pin_memory": bool(cfg.pin_memory),
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **loader_common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_common)
    return (train_dataset, val_dataset, test_dataset), (train_loader, val_loader)


def execute_dry_run(
    cfg: DictConfig,
    run_dir: Path,
    logger: logging.Logger,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_dataset: WeldSubDataset,
    train_loader: DataLoader,
) -> dict[str, Any]:
    logger.info("DRY_RUN begin: exactly one batch and one optimizer step")
    train_dataset.set_epoch(0)
    model.train()
    batch = next(iter(train_loader))
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = run_batch(
        model=model,
        optimizer=optimizer,
        batch=batch,
        device=device,
        k_neighbors=int(cfg.k_neighbors),
        training=True,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    confusion_update(confusion, result["target"].reshape(-1), result["prediction"].reshape(-1))
    miou, per_class = mean_iou(confusion)
    metrics = {
        "status": "DRY_RUN_PASSED",
        "mode": str(cfg.mode),
        "loss": result["loss"],
        "accuracy": result["correct"] / result["point_count"],
        "miou": miou,
        "per_class_iou": per_class,
        "elapsed_seconds": elapsed,
        "adjacency_cpu_seconds": result["adjacency_seconds"],
        "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "gpu_peak_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "points_shape": result["points_shape"],
        "adjacency_shape": result["adjacency_shape"],
        "logits_shape": result["logits_shape"],
        "points_device": result["points_device"],
        "adjacency_device": result["adjacency_device"],
        "logits_device": result["logits_device"],
    }
    payload = checkpoint_payload(cfg, model, optimizer, epoch=0, metrics=metrics)
    save_checkpoint(run_dir / "last_model.pth", payload)
    save_checkpoint(run_dir / "best_model.pth", payload)
    atomic_json_dump(run_dir / "metrics.json", metrics)
    reload_last = verify_saved_checkpoint(run_dir / "last_model.pth")
    reload_best = verify_saved_checkpoint(run_dir / "best_model.pth")
    metrics["last_model_reload"] = reload_last
    metrics["best_model_reload"] = reload_best
    atomic_json_dump(run_dir / "metrics.json", metrics)
    logger.info(
        "DRY_RUN result mode=%s loss=%.6f accuracy=%.6f mIoU=%.6f "
        "elapsed=%.3fs adjacency_cpu=%.3fs gpu_peak=%.2fMiB shapes=%s/%s/%s",
        cfg.mode,
        metrics["loss"],
        metrics["accuracy"],
        metrics["miou"],
        metrics["elapsed_seconds"],
        metrics["adjacency_cpu_seconds"],
        metrics["gpu_peak_memory_mib"],
        metrics["points_shape"],
        metrics["adjacency_shape"],
        metrics["logits_shape"],
    )
    logger.info("DRY_RUN checkpoints reloaded: last=%s best=%s", reload_last, reload_best)
    logger.info("DRY_RUN_PASSED artifact_dir=%s", run_dir)
    print(f"DRY_RUN_PASSED\nARTIFACT_DIR={run_dir}")
    return metrics


def execute_training(
    cfg: DictConfig,
    run_dir: Path,
    logger: logging.Logger,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_dataset: WeldSubDataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    best_miou = -float("inf")
    for epoch in range(int(cfg.epochs)):
        epoch_start = time.perf_counter()
        train_dataset.set_epoch(epoch)
        torch.cuda.reset_peak_memory_stats(device)
        train_metrics = run_epoch(
            model, train_loader, device, int(cfg.k_neighbors), optimizer
        )
        val_metrics = run_epoch(
            model, val_loader, device, int(cfg.k_neighbors), optimizer=None
        )
        torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start
        peak_bytes = torch.cuda.max_memory_allocated(device)
        record = {
            "epoch": epoch + 1,
            "mode": str(cfg.mode),
            "train": train_metrics,
            "val": val_metrics,
            "epoch_seconds": epoch_seconds,
            "adjacency_cpu_seconds": (
                train_metrics["adjacency_cpu_seconds"]
                + val_metrics["adjacency_cpu_seconds"]
            ),
            "gpu_peak_memory_bytes": peak_bytes,
            "gpu_peak_memory_mib": peak_bytes / 1024**2,
        }
        history.append(record)
        payload = checkpoint_payload(cfg, model, optimizer, epoch + 1, record)
        save_checkpoint(run_dir / "last_model.pth", payload)
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            save_checkpoint(run_dir / "best_model.pth", payload)
        atomic_json_dump(
            run_dir / "metrics.json",
            {"status": "RUNNING", "best_val_miou": best_miou, "epochs": history},
        )
        logger.info(
            "EPOCH %d/%d mode=%s train_loss=%.6f train_acc=%.6f "
            "val_loss=%.6f val_acc=%.6f mIoU=%.6f epoch=%.3fs "
            "adjacency_cpu=%.3fs (train=%.3fs,val=%.3fs) gpu_peak=%.2fMiB",
            epoch + 1,
            int(cfg.epochs),
            cfg.mode,
            train_metrics["loss"],
            train_metrics["accuracy"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["miou"],
            epoch_seconds,
            record["adjacency_cpu_seconds"],
            train_metrics["adjacency_cpu_seconds"],
            val_metrics["adjacency_cpu_seconds"],
            record["gpu_peak_memory_mib"],
        )

    reload_last = verify_saved_checkpoint(run_dir / "last_model.pth")
    reload_best = verify_saved_checkpoint(run_dir / "best_model.pth")
    final_metrics = {
        "status": "SMOKE_TRAINING_PASSED",
        "mode": str(cfg.mode),
        "best_val_miou": best_miou,
        "epochs": history,
        "last_model_reload": reload_last,
        "best_model_reload": reload_best,
    }
    atomic_json_dump(run_dir / "metrics.json", final_metrics)
    logger.info("Saved checkpoints reloaded: last=%s best=%s", reload_last, reload_best)
    logger.info("SMOKE_TRAINING_PASSED artifact_dir=%s", run_dir)
    print(f"SMOKE_TRAINING_PASSED\nARTIFACT_DIR={run_dir}")
    return history


def main() -> int:
    args = parse_args()
    cfg = load_config(args)
    run_dir = make_run_directory(cfg, args.run_id)
    logger = make_logger(run_dir)
    try:
        seed_everything(int(cfg.seed))
        logger.info(
            "Startup paths PROJECT_ROOT=%s sys.path[0]=%s cwd=%s",
            PROJECT_ROOT,
            sys.path[0],
            Path.cwd(),
        )
        logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
        logger.info(
            "Environment python=%s torch=%s cuda_runtime=%s gpu=%s capability=%s",
            sys.executable,
            torch.__version__,
            torch.version.cuda,
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        )
        datasets, loaders = make_datasets_and_loaders(cfg)
        train_dataset, val_dataset, test_dataset = datasets
        train_loader, val_loader = loaders
        logger.info(
            "Sub-dataset splits train=%d val=%d test=%d; files are disjoint",
            len(train_dataset),
            len(val_dataset),
            len(test_dataset),
        )
        model, optimizer, device, checkpoint_metadata = make_model_and_optimizer(cfg, logger)
        if checkpoint_metadata is not None:
            logger.info("Resume checkpoint metadata=%s", checkpoint_metadata)

        if cfg.dry_run:
            execute_dry_run(
                cfg,
                run_dir,
                logger,
                model,
                optimizer,
                device,
                train_dataset,
                train_loader,
            )
        else:
            execute_training(
                cfg,
                run_dir,
                logger,
                model,
                optimizer,
                device,
                train_dataset,
                train_loader,
                val_loader,
            )
        return 0
    except torch.cuda.OutOfMemoryError:
        logger.exception(
            "CUDA_OUT_OF_MEMORY: run stopped; batch size, N, K, model and precision were not changed"
        )
        raise
    except Exception:
        logger.exception("RUN_FAILED")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
