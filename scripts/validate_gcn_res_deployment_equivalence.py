"""Validate the standard-operator GCN_res deployment model against the source."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph

from deployment.gcn_res_onnx_model import GCNResStandardOps
from deployment.onnx_voxel_pool import standard_voxel_pool_with_metadata
from models.testParameters.GCN_res.model import PTV2Segmentation
from scripts.validate_voxel_pool_equivalence import reference_pool


SEED = 42
NUM_POINTS = 2048
K_NEIGHBORS = 6
DEVICE = "cuda:0"
POOL_RTOL = 1e-5
POOL_ATOL = 1e-6
LOGITS_RTOL = 1e-4
LOGITS_ATOL = 1e-5
MIN_LABEL_AGREEMENT = 0.9999

CHECKPOINT = PROJECT_ROOT / "models" / "testParameters" / "GCN_res" / "best_model.pth"
PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_evaluation"
    / "20260714_160831_945091_historical_checkpoint"
    / "predictions"
)
SAMPLES = [
    "val_00_weld_7",
    "val_01_weld_61",
    "val_02_weld_49",
    "test_00_weld_65",
    "test_01_weld_30",
    "test_02_weld_28",
]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_standard_ops"
STAGES = [
    "linear_1",
    "ptb_0",
    "gcn_0",
    "tdb_1",
    "ptb_1",
    "tdb_2",
    "ptb_2",
    "tdb_3",
    "ptb_3",
    "tdb_4",
    "ptb_4",
    "tub_6",
    "ptb_6",
    "tub_7",
    "ptb_7",
    "tub_8",
    "ptb_8",
    "tub_9",
    "ptb_9",
    "mlp",
]
TRANSITIONS = ["tdb_1", "tdb_2", "tdb_3", "tdb_4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", default=DEVICE)
    return parser.parse_args()


def make_run_dir(requested: Path | None) -> Path:
    if requested is not None:
        path = requested.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = ARTIFACTS_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_deployment_parity"
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def make_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"gcn_res_deployment_parity.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "deployment_equivalence.log", encoding="utf-8")
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


def load_models(device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint has no model_state_dict")
    state_dict = checkpoint["model_state_dict"]
    source = PTV2Segmentation(SimpleNamespace(num_class=2), in_dim=4)
    deployment = GCNResStandardOps(SimpleNamespace(num_class=2), in_dim=4)
    source_result = source.load_state_dict(state_dict, strict=True)
    deployment_result = deployment.load_state_dict(state_dict, strict=True)
    source_state = source.state_dict()
    deployment_state = deployment.state_dict()
    if source_state.keys() != deployment_state.keys():
        raise RuntimeError("source/deployment state_dict keys differ")
    unequal_parameters = [
        name
        for name in source_state
        if not torch.equal(source_state[name], deployment_state[name])
    ]
    if unequal_parameters:
        raise RuntimeError(f"checkpoint tensors differ after load: {unequal_parameters}")
    non_finite = [
        name
        for name, tensor in state_dict.items()
        if torch.is_tensor(tensor) and not torch.isfinite(tensor).all()
    ]
    if non_finite:
        raise FloatingPointError(f"non-finite checkpoint tensors: {non_finite}")
    metadata = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "source_strict_load": str(source_result),
        "deployment_strict_load": str(deployment_result),
        "state_dict_key_count": len(state_dict),
        "state_dict_keys_identical": True,
        "state_dict_tensors_bitwise_identical": True,
        "state_dict_key_remapping": False,
        "linear_1_weight_shape": list(state_dict["linear_1.weight"].shape),
        "mlp_weight_shape": list(state_dict["mlp.weight"].shape),
        "all_checkpoint_tensors_finite": True,
    }
    return source.to(device).eval(), deployment.to(device).eval(), metadata


def load_sample(name: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    path = PREDICTIONS_DIR / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        xyz = np.asarray(data["normalized_xyz"], dtype=np.float32)
        baseline_logits = np.asarray(data["logits"], dtype=np.float32)
    if xyz.shape != (NUM_POINTS, 3) or baseline_logits.shape != (NUM_POINTS, 2):
        raise RuntimeError(
            f"{name}: unexpected xyz/logits shapes {xyz.shape}, {baseline_logits.shape}"
        )
    points_np = np.concatenate(
        [xyz, np.ones((NUM_POINTS, 1), dtype=np.float32)], axis=1
    )[None]
    adj_np = kneighbors_graph(
        xyz,
        n_neighbors=K_NEIGHBORS,
        mode="connectivity",
        include_self=False,
    ).toarray().astype(np.float32, copy=False)[None]
    return (
        torch.from_numpy(points_np).to(device),
        torch.from_numpy(adj_np).to(device),
        baseline_logits,
    )


def clone_output(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(clone_output(item) for item in value)
    raise TypeError(f"unsupported hook output type: {type(value).__name__}")


def register_capture_hooks(
    model: torch.nn.Module,
    outputs: dict[str, Any],
    transition_inputs: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> list[Any]:
    handles: list[Any] = []
    modules = dict(model.named_modules())
    for stage_name in STAGES:
        module = modules[stage_name]

        def output_hook(_module: torch.nn.Module, _args: tuple[Any, ...], result: Any, name: str = stage_name) -> None:
            outputs[name] = clone_output(result)

        handles.append(module.register_forward_hook(output_hook))
    for stage_name in TRANSITIONS:
        module = modules[stage_name]

        def pre_hook(_module: torch.nn.Module, args: tuple[Any, ...], name: str = stage_name) -> None:
            transition_inputs[name] = (args[0].detach().clone(), args[1].detach().clone())

        handles.append(module.register_forward_pre_hook(pre_hook))
    return handles


def tensor_error(reference: torch.Tensor, actual: torch.Tensor, rtol: float, atol: float) -> dict[str, Any]:
    if reference.shape != actual.shape:
        return {
            "passed": False,
            "reference_shape": list(reference.shape),
            "deployment_shape": list(actual.shape),
            "reason": "shape_mismatch",
        }
    absolute = (reference - actual).abs()
    relative = absolute / reference.abs().clamp_min(1e-12)
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
    return {
        "passed": bool(finite and torch.allclose(reference, actual, rtol=rtol, atol=atol)),
        "reference_shape": list(reference.shape),
        "deployment_shape": list(actual.shape),
        "max_abs_error": float(absolute.max().item()) if absolute.numel() else 0.0,
        "mean_abs_error": float(absolute.mean().item()) if absolute.numel() else 0.0,
        "max_relative_error": float(relative.max().item()) if relative.numel() else 0.0,
        "all_finite": finite,
        "rtol": rtol,
        "atol": atol,
    }


def compare_stage(reference: Any, actual: Any) -> dict[str, Any]:
    if torch.is_tensor(reference) and torch.is_tensor(actual):
        return tensor_error(reference, actual, POOL_RTOL, POOL_ATOL)
    if isinstance(reference, tuple) and isinstance(actual, tuple) and len(reference) == len(actual):
        components = [
            tensor_error(ref, got, POOL_RTOL, POOL_ATOL)
            for ref, got in zip(reference, actual)
        ]
        return {"passed": all(item["passed"] for item in components), "components": components}
    return {"passed": False, "reason": "output_structure_mismatch"}


def compare_transition_pool(
    stage_name: str,
    source_model: torch.nn.Module,
    source_input: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    module = dict(source_model.named_modules())[stage_name]
    xyz, input_features = source_input
    # Hook captures created inside inference_mode remain inference tensors.  This
    # diagnostic replay is inference-only as well; keeping it in the same mode
    # prevents autograd from trying to save those tensors for backward.
    with torch.inference_mode():
        transformed = module.linear(input_features)
        transformed = module.norm(transformed.permute(0, 2, 1)).permute(0, 2, 1)
        transformed = module.relu(transformed)
        voxel_size = torch.tensor(module.grid_size, device=xyz.device, dtype=xyz.dtype)
        reference = reference_pool(xyz, transformed, voxel_size)
        actual = standard_voxel_pool_with_metadata(xyz, transformed, voxel_size)

    membership = []
    key_match = []
    for batch_index, reference_inverse in enumerate(reference["inverse"]):
        actual_inverse = actual.point_to_voxel[batch_index]
        membership.append(
            torch.equal(
                reference_inverse[:, None] == reference_inverse[None, :],
                actual_inverse[:, None] == actual_inverse[None, :],
            )
        )
        actual_keys = actual.unique_local_keys[actual.unique_batch_ids == batch_index]
        key_match.append(torch.equal(reference["unique_keys"][batch_index], actual_keys))

    xyz_error = tensor_error(reference["pooled_points"], actual.pooled_points, POOL_RTOL, POOL_ATOL)
    feature_error = tensor_error(
        reference["pooled_features"], actual.pooled_features, POOL_RTOL, POOL_ATOL
    )
    discrete = {
        "voxel_count_exact": bool(
            torch.equal(reference["voxel_counts"], actual.voxel_count_per_batch)
        ),
        "voxel_coordinates_exact": bool(
            torch.equal(reference["retained_coordinates"], actual.retained_voxel_coordinates)
        ),
        "voxel_point_counts_exact": bool(
            torch.equal(reference["retained_counts"], actual.retained_voxel_counts)
        ),
        "cluster_membership_semantic_exact": all(membership),
        "sorted_unique_keys_exact": all(key_match),
    }
    passed = all(discrete.values()) and xyz_error["passed"] and feature_error["passed"]
    return {
        "passed": passed,
        "input_xyz_shape": list(xyz.shape),
        "input_features_shape": list(transformed.shape),
        "output_xyz_shape": list(actual.pooled_points.shape),
        "output_features_shape": list(actual.pooled_features.shape),
        "voxel_size": module.grid_size,
        "voxel_count_per_batch": actual.voxel_count_per_batch.detach().cpu().tolist(),
        "discrete": discrete,
        "pooled_xyz": xyz_error,
        "pooled_features": feature_error,
    }


def validate_sample(
    name: str,
    source: torch.nn.Module,
    deployment: torch.nn.Module,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, Any]:
    points, adj, baseline_logits = load_sample(name, device)
    source_outputs: dict[str, Any] = {}
    deployment_outputs: dict[str, Any] = {}
    source_transition_inputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    deployment_transition_inputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = register_capture_hooks(source, source_outputs, source_transition_inputs)
    handles += register_capture_hooks(deployment, deployment_outputs, deployment_transition_inputs)
    try:
        with torch.inference_mode():
            source_xyz, source_logits = source(points, adj)
            deployment_xyz, deployment_logits = deployment(points, adj)
        torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()

    stage_results = {
        stage_name: compare_stage(source_outputs[stage_name], deployment_outputs[stage_name])
        for stage_name in STAGES
    }
    transition_results = {
        stage_name: compare_transition_pool(
            stage_name, source, source_transition_inputs[stage_name]
        )
        for stage_name in TRANSITIONS
    }
    transition_input_results = {
        stage_name: {
            "xyz": tensor_error(
                source_transition_inputs[stage_name][0],
                deployment_transition_inputs[stage_name][0],
                POOL_RTOL,
                POOL_ATOL,
            ),
            "features": tensor_error(
                source_transition_inputs[stage_name][1],
                deployment_transition_inputs[stage_name][1],
                POOL_RTOL,
                POOL_ATOL,
            ),
        }
        for stage_name in TRANSITIONS
    }
    logits_error = tensor_error(
        source_logits, deployment_logits, LOGITS_RTOL, LOGITS_ATOL
    )
    label_agreement = float(
        (source_logits.argmax(dim=-1) == deployment_logits.argmax(dim=-1))
        .float()
        .mean()
        .item()
    )
    source_probability = torch.softmax(source_logits, dim=-1)
    deployment_probability = torch.softmax(deployment_logits, dim=-1)
    probability_error = (source_probability - deployment_probability).abs()
    baseline = torch.from_numpy(baseline_logits).to(device).unsqueeze(0)
    baseline_error = tensor_error(source_logits, baseline, LOGITS_RTOL, LOGITS_ATOL)
    xyz_exact = torch.equal(source_xyz, deployment_xyz)
    passed = bool(
        xyz_exact
        and logits_error["passed"]
        and label_agreement >= MIN_LABEL_AGREEMENT
        and all(item["passed"] for item in stage_results.values())
        and all(item["passed"] for item in transition_results.values())
        and all(
            item[axis]["passed"]
            for item in transition_input_results.values()
            for axis in ("xyz", "features")
        )
    )
    result = {
        "sample": name,
        "passed": passed,
        "input_shapes": {"points": list(points.shape), "adj": list(adj.shape)},
        "output_shape": list(deployment_logits.shape),
        "returned_points_xyz_exact": xyz_exact,
        "stages": stage_results,
        "transition_inputs": transition_input_results,
        "voxel_pooling": transition_results,
        "final_logits": logits_error,
        "predicted_label_agreement": label_agreement,
        "weld_seam_probability_max_abs_error": float(probability_error[..., 0].max().item()),
        "weld_seam_probability_mean_abs_error": float(probability_error[..., 0].mean().item()),
        "background_probability_max_abs_error": float(probability_error[..., 1].max().item()),
        "background_probability_mean_abs_error": float(probability_error[..., 1].mean().item()),
        "source_vs_saved_baseline_logits": baseline_error,
        "all_logits_finite": bool(
            torch.isfinite(source_logits).all() and torch.isfinite(deployment_logits).all()
        ),
    }
    logger.info(
        "%s passed=%s logits_max_abs=%.9g label_agreement=%.8f voxel_counts=%s",
        name,
        passed,
        logits_error["max_abs_error"],
        label_agreement,
        {
            stage: item["voxel_count_per_batch"]
            for stage, item in transition_results.items()
        },
    )
    return result


def main() -> int:
    args = parse_args()
    run_dir = make_run_dir(args.run_dir)
    logger = make_logger(run_dir)
    output_path = run_dir / "deployment_equivalence.json"
    payload: dict[str, Any] = {
        "status": "started",
        "project_root": str(PROJECT_ROOT),
        "source_model": str(
            PROJECT_ROOT / "models" / "testParameters" / "GCN_res" / "model.py"
        ),
        "deployment_model": str(PROJECT_ROOT / "deployment" / "gcn_res_onnx_model.py"),
        "voxel_implementation": str(PROJECT_ROOT / "deployment" / "onnx_voxel_pool.py"),
        "samples": SAMPLES,
        "thresholds": {
            "pool_rtol": POOL_RTOL,
            "pool_atol": POOL_ATOL,
            "logits_rtol": LOGITS_RTOL,
            "logits_atol": LOGITS_ATOL,
            "minimum_predicted_label_agreement": MIN_LABEL_AGREEMENT,
        },
        "results": [],
    }
    try:
        seed_everything()
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA deployment validation requires an available CUDA device")
        logger.info(
            "PROJECT_ROOT=%s cwd=%s python=%s torch=%s GPU=%s capability=%s",
            PROJECT_ROOT,
            Path.cwd(),
            sys.executable,
            torch.__version__,
            torch.cuda.get_device_name(device),
            torch.cuda.get_device_capability(device),
        )
        source, deployment, checkpoint_metadata = load_models(device)
        payload["checkpoint"] = checkpoint_metadata
        logger.info("checkpoint strict load source/deployment passed with no key mapping")
        for sample_name in SAMPLES:
            sample_result = validate_sample(sample_name, source, deployment, device, logger)
            payload["results"].append(sample_result)
            # Persist the first failing sample's full stage diagnostics before
            # enforcing the stop condition.  No later sample is evaluated.
            if not sample_result["passed"]:
                output_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                failed_stages = [
                    key
                    for key, value in sample_result["stages"].items()
                    if not value["passed"]
                ]
                failed_pools = [
                    key
                    for key, value in sample_result["voxel_pooling"].items()
                    if not value["passed"]
                ]
                raise AssertionError(
                    f"{sample_name} deployment parity failed: stages={failed_stages}, "
                    f"pools={failed_pools}, logits={sample_result['final_logits']}, "
                    f"agreement={sample_result['predicted_label_agreement']}"
                )
        payload["status"] = "GCN_RES_DEPLOYMENT_MODEL_PARITY_PASSED"
        payload["all_six_samples_passed"] = True
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("GCN_RES_DEPLOYMENT_MODEL_PARITY_PASSED")
        print(f"ARTIFACT_DIR={run_dir}")
        return 0
    except Exception as exc:
        payload.update(
            {
                "status": "GCN_RES_DEPLOYMENT_MODEL_PARITY_FAILED",
                "all_six_samples_passed": False,
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.exception("GCN_RES_DEPLOYMENT_MODEL_PARITY_FAILED")
        print("GCN_RES_DEPLOYMENT_MODEL_PARITY_FAILED")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
