"""Audit and evaluate every models/partseg/*/best_model.pth on fixed weld splits.

The benchmark is inference-only.  It never changes model sources, checkpoints,
dataset files, split JSON files, or deployment code.  A checkpoint is evaluated
only after its 4-input/2-output contract and strict state_dict load are verified.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import random
import sys
import time
import traceback
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.evaluate_gcn_res_checkpoint import (
    FixedWeldEvaluationDataset,
    build_adjacency_cpu,
    make_model_input,
)


SEED = 42
DEVICE = "cuda:0"
EXPECTED_INPUT_DIM = 4
EXPECTED_OUTPUT_DIM = 2
PARTSEG_ROOT = PROJECT_ROOT / "models" / "partseg"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "partseg_checkpoint_benchmark"
GCN_RES_METRICS = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_evaluation"
    / "20260714_160831_945091_historical_checkpoint"
    / "metrics.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    return parser.parse_args()


def make_run_dir(run_id: str | None) -> Path:
    identifier = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_fixed_sub_splits"
    if any(token in identifier for token in ("/", "\\", "..")):
        raise ValueError(f"Unsafe run id: {identifier!r}")
    path = ARTIFACTS_ROOT / identifier
    path.mkdir(parents=True, exist_ok=False)
    return path


def make_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"partseg_checkpoint_benchmark.{run_dir.name}")
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


def seed_everything() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def source_classes(model_path: Path) -> list[str]:
    tree = ast.parse(model_path.read_text(encoding="utf-8", errors="replace"))
    return [node.name for node in tree.body if isinstance(node, ast.ClassDef)]


def choose_utility_source(model_name: str, saved_dir: Path) -> tuple[Path, str]:
    saved = saved_dir / "ptv2_utils.py"
    if saved.is_file():
        return saved, "checkpoint_directory"
    training_module = PROJECT_ROOT / "models" / model_name / "ptv2_utils.py"
    if training_module.is_file():
        return training_module, "corresponding_training_module"
    # The historical Res_LFA root module is absent.  Its saved model uses the
    # same PTV2 utility API and tensor dimensions as the sibling GCN_LFA model.
    if model_name == "Nico_v2_GCN_Res_LFA":
        fallback = PROJECT_ROOT / "models" / "Nico_v2_GCN_LFA" / "ptv2_utils.py"
        if fallback.is_file():
            return fallback, "inferred_sibling_GCN_LFA_utility_root_module_missing"
    raise FileNotFoundError(
        f"No ptv2_utils.py available for saved model {model_name} in {saved_dir}"
    )


def load_saved_model_module(model_name: str, model_path: Path, utility_path: Path) -> Any:
    """Load saved model.py with an isolated package and explicit sibling utils."""

    safe = "".join(character if character.isalnum() else "_" for character in model_name)
    package_name = f"_partseg_benchmark_{safe}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_path.parent)]
    package.__package__ = package_name
    sys.modules[package_name] = package

    utility_name = f"{package_name}.ptv2_utils"
    utility_spec = importlib.util.spec_from_file_location(utility_name, utility_path)
    if utility_spec is None or utility_spec.loader is None:
        raise ImportError(f"Cannot create import spec for {utility_path}")
    utility_module = importlib.util.module_from_spec(utility_spec)
    sys.modules[utility_name] = utility_module
    utility_spec.loader.exec_module(utility_module)

    module_name = f"{package_name}.model"
    model_spec = importlib.util.spec_from_file_location(module_name, model_path)
    if model_spec is None or model_spec.loader is None:
        raise ImportError(f"Cannot create import spec for {model_path}")
    model_module = importlib.util.module_from_spec(model_spec)
    sys.modules[module_name] = model_module
    model_spec.loader.exec_module(model_module)
    return model_module


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def extract_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and len(output) >= 2 and torch.is_tensor(output[1]):
        return output[1]
    raise TypeError(f"Unsupported model output: {type(output).__name__}")


def run_model(
    model: torch.nn.Module,
    forward_parameter_count: int,
    points: torch.Tensor,
    adjacency: torch.Tensor,
) -> torch.Tensor:
    if forward_parameter_count == 1:
        return extract_logits(model(points))
    if forward_parameter_count == 2:
        return extract_logits(model(points, adjacency))
    raise TypeError(
        f"Expected forward(points) or forward(points, adj), got {forward_parameter_count} parameters"
    )


def evaluate_split(
    model: torch.nn.Module,
    forward_parameter_count: int,
    split: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    dataset = FixedWeldEvaluationDataset(split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    confusion = np.zeros((2, 2), dtype=np.int64)
    losses: list[float] = []
    start = time.perf_counter()
    for index, batch in enumerate(loader, start=1):
        xyz_cpu = batch["normalized_xyz"]
        labels = batch["labels"].to(DEVICE)
        adjacency_cpu, _ = build_adjacency_cpu(xyz_cpu)
        points = make_model_input(xyz_cpu).to(DEVICE)
        adjacency = adjacency_cpu.to(DEVICE)
        with torch.inference_mode():
            logits = run_model(model, forward_parameter_count, points, adjacency)
        if tuple(logits.shape) != (1, 2048, 2):
            raise RuntimeError(f"{split} sample {index}: unexpected logits {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"{split} sample {index}: logits contain NaN/Inf")
        loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{split} sample {index}: loss is NaN/Inf")
        losses.append(float(loss.item()))
        prediction = logits.argmax(dim=-1)
        ground_truth_cpu = labels.reshape(-1).cpu().numpy()
        prediction_cpu = prediction.reshape(-1).cpu().numpy()
        np.add.at(confusion, (ground_truth_cpu, prediction_cpu), 1)
        logger.info(
            "%s %02d/%02d sample=%s loss=%.6f",
            split,
            index,
            len(dataset),
            batch["sample_name"][0],
            losses[-1],
        )

    tp = int(confusion[0, 0])
    fn = int(confusion[0, 1])
    fp = int(confusion[1, 0])
    tn = int(confusion[1, 1])
    weld_iou = safe_divide(tp, tp + fn + fp)
    background_iou = safe_divide(tn, tn + fp + fn)
    weld_precision = safe_divide(tp, tp + fp)
    weld_recall = safe_divide(tp, tp + fn)
    weld_f1 = safe_divide(
        2.0 * weld_precision * weld_recall, weld_precision + weld_recall
    )
    return {
        "sample_count": len(dataset),
        "point_count": int(confusion.sum()),
        "loss": float(np.mean(losses)),
        "accuracy": safe_divide(int(np.trace(confusion)), int(confusion.sum())),
        "weld_seam_iou": weld_iou,
        "background_iou": background_iou,
        "miou": (weld_iou + background_iou) / 2.0,
        "weld_precision": weld_precision,
        "weld_recall": weld_recall,
        "weld_f1": weld_f1,
        "confusion_matrix": confusion.tolist(),
        "elapsed_seconds": time.perf_counter() - start,
    }


def deployment_ease(model_name: str, model_text: str, utility_text: str) -> dict[str, Any]:
    reasons: list[str] = []
    score = 2
    combined = model_text + "\n" + utility_text
    if "sklearn" in combined or ".cpu().numpy()" in combined:
        score = 0
        reasons.append("Python/sklearn or CPU NumPy round-trip is inside the model path")
    if "torch_cluster" in utility_text:
        score = min(score, 1)
        reasons.append("voxel pooling depends on torch_cluster custom operators")
    if "for i in range(N)" in model_text or "for b in range(B)" in model_text:
        score = 0
        reasons.append("LocalFeatureAggregation contains Python point/batch loops")
    if "GridCluster" in utility_text:
        score = min(score, 1)
        reasons.append("custom grid implementation still requires export/parity validation")
    labels = {0: "very_difficult", 1: "difficult", 2: "moderate"}
    return {"score": score, "label": labels[score], "reasons": reasons or ["no extra static blocker found"]}


def audit_and_evaluate(
    checkpoint_path: Path, logger: logging.Logger
) -> dict[str, Any]:
    model_name = checkpoint_path.parent.name
    model_path = checkpoint_path.parent / "model.py"
    result: dict[str, Any] = {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "model_path": str(model_path),
        "status": "audit_started",
    }
    try:
        if not model_path.is_file():
            result["status"] = "missing_model_py"
            return result
        result["model_sha256"] = sha256(model_path)
        result["source_classes"] = source_classes(model_path)
        result["class_name"] = (
            "PTV2Segmentation" if "PTV2Segmentation" in result["source_classes"] else None
        )
        training_model_path = PROJECT_ROOT / "models" / model_name / "model.py"
        result["corresponding_training_model_path"] = (
            str(training_model_path) if training_model_path.is_file() else None
        )
        result["saved_model_matches_corresponding_training_model"] = (
            sha256(model_path) == sha256(training_model_path)
            if training_model_path.is_file()
            else None
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise KeyError("checkpoint has no model_state_dict")
        state_dict = checkpoint["model_state_dict"]
        first = state_dict.get("linear_1.weight")
        last = state_dict.get("mlp.weight")
        if not torch.is_tensor(first) or not torch.is_tensor(last):
            raise KeyError("linear_1.weight or mlp.weight is missing")
        input_dim = int(first.shape[1])
        output_dim = int(last.shape[0])
        non_finite = [
            name
            for name, tensor in state_dict.items()
            if torch.is_tensor(tensor)
            and (tensor.is_floating_point() or tensor.is_complex())
            and not torch.isfinite(tensor).all()
        ]
        result.update(
            {
                "checkpoint_sha256": sha256(checkpoint_path),
                "checkpoint_fields": sorted(checkpoint.keys()),
                "checkpoint_key_count": len(state_dict),
                "state_dict_numel": sum(
                    tensor.numel() for tensor in state_dict.values() if torch.is_tensor(tensor)
                ),
                "input_dim": input_dim,
                "first_layer_weight_shape": list(first.shape),
                "output_dim": output_dim,
                "last_layer_weight_shape": list(last.shape),
                "checkpoint_non_finite_tensors": non_finite,
                "checkpoint_metadata": {
                    key: scalar(checkpoint.get(key))
                    for key in (
                        "epoch",
                        "train_acc",
                        "test_acc",
                        "class_avg_iou",
                        "inctance_avg_iou",
                    )
                },
            }
        )
        if non_finite:
            result["status"] = "checkpoint_contains_non_finite"
            return result
        if input_dim != EXPECTED_INPUT_DIM:
            # The incompatible historical ShapeNet checkpoint has a copied
            # model.py that no longer matches its 19-D/50-class state_dict.
            # Audit its parameter count against the corresponding training
            # module, but never adapt or evaluate it on weld input.
            if training_model_path.is_file():
                try:
                    training_module = __import__(
                        f"models.{model_name}.model", fromlist=["PTV2Segmentation"]
                    )
                    audit_model = training_module.PTV2Segmentation(
                        SimpleNamespace(num_class=output_dim), in_dim=input_dim
                    )
                    result["strict_load"] = str(
                        audit_model.load_state_dict(state_dict, strict=True)
                    )
                    result["parameter_count"] = sum(
                        parameter.numel() for parameter in audit_model.parameters()
                    )
                    result["parameter_tensor_count"] = sum(
                        1 for _ in audit_model.parameters()
                    )
                    result["parameter_count_source"] = "corresponding_training_module"
                except Exception as audit_exc:
                    result["incompatible_model_audit_error"] = (
                        f"{type(audit_exc).__name__}: {audit_exc}"
                    )
            result["status"] = "incompatible_with_weld_input"
            result["incompatibility_reason"] = (
                f"checkpoint requires {input_dim} input channels; weld contract is 4"
            )
            if output_dim != EXPECTED_OUTPUT_DIM:
                result["incompatibility_reason"] += (
                    f"; checkpoint also outputs {output_dim} classes instead of 2"
                )
            return result
        if output_dim != EXPECTED_OUTPUT_DIM:
            result["status"] = "incompatible_with_weld_output"
            result["incompatibility_reason"] = (
                f"checkpoint outputs {output_dim} classes; weld contract is 2"
            )
            return result
        if result["class_name"] is None:
            result["status"] = "missing_ptv2_segmentation_class"
            return result

        utility_path, utility_resolution = choose_utility_source(
            model_name, checkpoint_path.parent
        )
        result["ptv2_utils_path"] = str(utility_path)
        result["ptv2_utils_sha256"] = sha256(utility_path)
        result["ptv2_utils_resolution"] = utility_resolution
        model_text = model_path.read_text(encoding="utf-8", errors="replace")
        utility_text = utility_path.read_text(encoding="utf-8", errors="replace")
        result["onnx_deployment_ease"] = deployment_ease(
            model_name, model_text, utility_text
        )

        module = load_saved_model_module(model_name, model_path, utility_path)
        model_class = getattr(module, "PTV2Segmentation")
        model = model_class(SimpleNamespace(num_class=2), in_dim=4)
        strict_result = model.load_state_dict(state_dict, strict=True)
        result["strict_load"] = str(strict_result)
        result["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        result["parameter_tensor_count"] = sum(1 for _ in model.parameters())
        forward_parameters = list(inspect.signature(model.forward).parameters.values())
        result["forward_parameters"] = [parameter.name for parameter in forward_parameters]
        result["forward_parameter_count"] = len(forward_parameters)
        if len(forward_parameters) not in (1, 2):
            result["status"] = "unsupported_forward_signature"
            return result

        seed_everything()
        model = model.to(DEVICE).eval()
        result["val"] = evaluate_split(
            model, len(forward_parameters), "val", logger
        )
        result["test"] = evaluate_split(
            model, len(forward_parameters), "test", logger
        )
        result["status"] = "evaluated"
        logger.info(
            "MODEL_RESULT name=%s val_mIoU=%.9f test_mIoU=%.9f test_weld_f1=%.9f",
            model_name,
            result["val"]["miou"],
            result["test"]["miou"],
            result["test"]["weld_f1"],
        )
        del model
        torch.cuda.empty_cache()
        return result
    except Exception as exc:
        result.update(
            {
                "status": "evaluation_failed",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        logger.exception("MODEL_FAILED name=%s", model_name)
        torch.cuda.empty_cache()
        return result


def gcn_res_baseline() -> dict[str, Any] | None:
    if not GCN_RES_METRICS.is_file():
        return None
    payload = json.loads(GCN_RES_METRICS.read_text(encoding="utf-8"))
    return {
        "model_name": "GCN_res_reference_baseline",
        "checkpoint_path": payload["checkpoint"],
        "input_dim": 4,
        "output_dim": 2,
        "parameter_count": 6_193_202,
        "checkpoint_key_count": 434,
        "status": "existing_verified_baseline",
        "val": {
            "loss": payload["val"]["average_loss"],
            "accuracy": payload["val"]["overall_accuracy"],
            "weld_seam_iou": payload["val"]["weld_seam_iou"],
            "background_iou": payload["val"]["background_iou"],
            "miou": payload["val"]["miou"],
            "weld_precision": payload["val"]["weld_seam_precision"],
            "weld_recall": payload["val"]["weld_seam_recall"],
            "weld_f1": payload["val"]["weld_seam_f1"],
        },
        "test": {
            "loss": payload["test"]["average_loss"],
            "accuracy": payload["test"]["overall_accuracy"],
            "weld_seam_iou": payload["test"]["weld_seam_iou"],
            "background_iou": payload["test"]["background_iou"],
            "miou": payload["test"]["miou"],
            "weld_precision": payload["test"]["weld_seam_precision"],
            "weld_recall": payload["test"]["weld_seam_recall"],
            "weld_f1": payload["test"]["weld_seam_f1"],
        },
        "onnx_deployment_ease": {
            "score": 1,
            "label": "difficult",
            "reasons": [
                "historical source contains torch_cluster voxel pooling",
                "standard-ops deployment model has not passed all layer-wise parity checks",
            ],
        },
        "metrics_source": str(GCN_RES_METRICS),
    }


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row.get("status") in {"evaluated", "existing_verified_baseline"}
    ]
    return sorted(
        eligible,
        key=lambda row: (
            -float(row["test"]["miou"]),
            -float(row["test"]["weld_f1"]),
            -int(row.get("onnx_deployment_ease", {}).get("score", -1)),
            row["model_name"],
        ),
    )


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model_name",
        "checkpoint_path",
        "input_dim",
        "output_dim",
        "parameter_count",
        "checkpoint_key_count",
        "val_miou",
        "test_miou",
        "test_weld_f1",
        "onnx_deployment_ease",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model_name": row["model_name"],
                    "checkpoint_path": row["checkpoint_path"],
                    "input_dim": row.get("input_dim"),
                    "output_dim": row.get("output_dim"),
                    "parameter_count": row.get("parameter_count"),
                    "checkpoint_key_count": row.get("checkpoint_key_count"),
                    "val_miou": row.get("val", {}).get("miou"),
                    "test_miou": row.get("test", {}).get("miou"),
                    "test_weld_f1": row.get("test", {}).get("weld_f1"),
                    "onnx_deployment_ease": row.get("onnx_deployment_ease", {}).get("label"),
                    "status": row["status"],
                }
            )


def main() -> int:
    args = parse_args()
    run_dir = make_run_dir(args.run_id)
    logger = make_logger(run_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    seed_everything()
    checkpoints = sorted(PARTSEG_ROOT.glob("*/best_model.pth"), key=lambda path: path.parent.name)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {PARTSEG_ROOT}")
    logger.info(
        "START project=%s checkpoints=%d python=%s torch=%s gpu=%s",
        PROJECT_ROOT,
        len(checkpoints),
        sys.executable,
        torch.__version__,
        torch.cuda.get_device_name(0),
    )
    rows: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        logger.info("AUDIT_MODEL %d/%d name=%s", index, len(checkpoints), checkpoint.parent.name)
        rows.append(audit_and_evaluate(checkpoint, logger))
    baseline = gcn_res_baseline()
    all_rows = rows + ([baseline] if baseline is not None else [])
    ranking = rank_rows(all_rows)
    payload = {
        "status": "PARTSEG_CHECKPOINT_BENCHMARK_COMPLETED",
        "project_root": str(PROJECT_ROOT),
        "run_directory": str(run_dir),
        "seed": SEED,
        "device": DEVICE,
        "label_mapping": {"0": "weld_seam", "1": "background"},
        "val_split": str(
            PROJECT_ROOT / "data" / "weld" / "train_test_split" / "sub_shuffled_val_file_list.json"
        ),
        "test_split": str(
            PROJECT_ROOT / "data" / "weld" / "train_test_split" / "sub_shuffled_test_file_list.json"
        ),
        "partseg_checkpoint_count": len(checkpoints),
        "partseg_results": rows,
        "gcn_res_reference": baseline,
        "ranking_order": [row["model_name"] for row in ranking],
        "ranking_rule": ["test_miou_desc", "test_weld_f1_desc", "onnx_deployment_ease_desc"],
    }
    (run_dir / "benchmark.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_summary_csv(run_dir / "summary.csv", all_rows)
    logger.info("RANKING %s", payload["ranking_order"])
    print("PARTSEG_CHECKPOINT_BENCHMARK_COMPLETED")
    print(f"ARTIFACT_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
