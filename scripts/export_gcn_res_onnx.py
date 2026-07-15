"""Export the standard-operator GCN_res deployment model to fixed FP32 ONNX."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

assert (PROJECT_ROOT / "models").is_dir(), PROJECT_ROOT / "models"
assert (PROJECT_ROOT / "deployment").is_dir(), PROJECT_ROOT / "deployment"

import numpy as np
import onnx
import torch
from omegaconf import OmegaConf
from onnx import TensorProto, numpy_helper, shape_inference
from sklearn.neighbors import kneighbors_graph

from deployment.gcn_res_onnx_wrapper import GCNResOnnxWrapper


SEED = 42
BATCH_SIZE = 1
NUM_POINTS = 2048
INPUT_DIM = 4
NUM_CLASSES = 2
K_NEIGHBORS = 6
OPSET = 18
DEVICE = "cuda:0"
DTYPE = np.float32

CHECKPOINT = PROJECT_ROOT / "models" / "testParameters" / "GCN_res" / "best_model.pth"
BASELINE_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_evaluation"
    / "20260714_160831_945091_historical_checkpoint"
)
SOURCE_NPZ = BASELINE_DIR / "predictions" / "val_00_weld_7.npz"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_onnx"
MODEL_NAME = "gcn_res_deploy_fp32_opset18.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def make_run_directory(run_id: str | None) -> Path:
    resolved_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_fixed_b1_n2048"
    if any(token in resolved_id for token in ("/", "\\", "..")):
        raise ValueError(f"Unsafe run ID: {resolved_id!r}")
    run_dir = ARTIFACTS_ROOT / resolved_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"gcn_res_onnx_export.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "export.log", encoding="utf-8")
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


def build_fixed_inputs() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if not SOURCE_NPZ.is_file():
        raise FileNotFoundError(f"Fixed baseline NPZ not found: {SOURCE_NPZ}")
    with np.load(SOURCE_NPZ) as source:
        normalized_xyz = np.asarray(source["normalized_xyz"], dtype=DTYPE)
        sample_indices = np.asarray(source["sample_indices"], dtype=np.int64)
        ground_truth = np.asarray(source["ground_truth_labels"], dtype=np.int64)
        baseline_logits = np.asarray(source["logits"], dtype=DTYPE)
    if normalized_xyz.shape != (NUM_POINTS, 3):
        raise RuntimeError(f"Unexpected normalized XYZ shape: {normalized_xyz.shape}")
    points = np.concatenate(
        [normalized_xyz, np.ones((NUM_POINTS, 1), dtype=DTYPE)], axis=1
    )[None, ...]
    adjacency = kneighbors_graph(
        normalized_xyz,
        n_neighbors=K_NEIGHBORS,
        mode="connectivity",
        include_self=False,
    ).toarray().astype(DTYPE, copy=False)[None, ...]
    if points.shape != (BATCH_SIZE, NUM_POINTS, INPUT_DIM):
        raise RuntimeError(f"Unexpected points shape: {points.shape}")
    if adjacency.shape != (BATCH_SIZE, NUM_POINTS, NUM_POINTS):
        raise RuntimeError(f"Unexpected adjacency shape: {adjacency.shape}")
    if not np.isfinite(points).all() or not np.isfinite(adjacency).all():
        raise FloatingPointError("Export inputs contain NaN or Inf")
    metadata = {
        "normalized_xyz": normalized_xyz,
        "sample_indices": sample_indices,
        "ground_truth_labels": ground_truth,
        "evaluation_baseline_logits": baseline_logits,
    }
    return points, adjacency, metadata


def load_wrapper(logger: logging.Logger) -> tuple[GCNResOnnxWrapper, dict[str, Any]]:
    from deployment.gcn_res_onnx_model import GCNResStandardOps

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise KeyError(f"model_state_dict missing from {CHECKPOINT}")
    state_dict = checkpoint["model_state_dict"]
    if tuple(state_dict["linear_1.weight"].shape) != (48, 4):
        raise RuntimeError(f"linear_1.weight mismatch: {state_dict['linear_1.weight'].shape}")
    if tuple(state_dict["mlp.weight"].shape) != (2, 48):
        raise RuntimeError(f"mlp.weight mismatch: {state_dict['mlp.weight'].shape}")
    non_finite = [
        name
        for name, tensor in state_dict.items()
        if torch.is_tensor(tensor) and not torch.isfinite(tensor).all()
    ]
    if non_finite:
        raise FloatingPointError(f"Non-finite checkpoint tensors: {non_finite}")
    model = GCNResStandardOps(SimpleNamespace(num_class=2), in_dim=4)
    strict_result = model.load_state_dict(state_dict, strict=True)
    wrapper = GCNResOnnxWrapper(model).to(DEVICE).eval()
    metadata = {
        "epoch": int(checkpoint["epoch"]),
        "strict_load": str(strict_result),
        "linear_1_weight_shape": [48, 4],
        "mlp_weight_shape": [2, 48],
        "checkpoint_tensors_finite": True,
    }
    logger.info("Checkpoint strict=True load: %s", strict_result)
    return wrapper, metadata


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    dimensions: list[int | str | None] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dimensions.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dimensions.append(dim.dim_param)
        else:
            dimensions.append(None)
    return dimensions


def graph_input_dependencies(model: onnx.ModelProto, output_name: str) -> set[str]:
    graph_inputs = {item.name for item in model.graph.input}
    producer = {output: node for node in model.graph.node for output in node.output}
    found: set[str] = set()
    pending = [output_name]
    visited: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        if name in graph_inputs:
            found.add(name)
            continue
        node = producer.get(name)
        if node is not None:
            pending.extend(node.input)
    return found


def inspect_onnx(model_path: Path, logger: logging.Logger) -> dict[str, Any]:
    model = onnx.load(model_path)
    onnx.checker.check_model(model, full_check=True)
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(inferred, full_check=True)
    onnx.save(inferred, model_path)

    inputs = [
        {
            "name": item.name,
            "shape": tensor_shape(item),
            "dtype": TensorProto.DataType.Name(item.type.tensor_type.elem_type),
        }
        for item in inferred.graph.input
    ]
    outputs = [
        {
            "name": item.name,
            "shape": tensor_shape(item),
            "dtype": TensorProto.DataType.Name(item.type.tensor_type.elem_type),
        }
        for item in inferred.graph.output
    ]
    python_ops = [node.name for node in inferred.graph.node if node.op_type == "PythonOp"]
    aten_fallback = [
        node.name
        for node in inferred.graph.node
        if node.op_type in {"ATen", "ATenOp"} or "aten" in node.domain.lower()
    ]
    standard_domains = {"", "ai.onnx", "ai.onnx.ml"}
    custom_domains = sorted({node.domain for node in inferred.graph.node if node.domain not in standard_domains})
    torch_cluster_nodes = [
        {"name": node.name, "domain": node.domain, "op_type": node.op_type}
        for node in inferred.graph.node
        if "torch_cluster" in node.domain.lower()
        or "torch_cluster" in node.op_type.lower()
        or "torch_cluster" in node.name.lower()
    ]

    constants: list[dict[str, Any]] = []
    for node in inferred.graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                array = numpy_helper.to_array(attribute.t)
                constants.append({"name": node.name, "shape": list(array.shape), "bytes": int(array.nbytes)})
    initializer_constants = [
        {
            "name": item.name,
            "shape": list(numpy_helper.to_array(item).shape),
            "bytes": int(numpy_helper.to_array(item).nbytes),
        }
        for item in inferred.graph.initializer
    ]
    large_constants = [
        item
        for item in constants + initializer_constants
        if item["bytes"] >= 16 * 1024 * 1024
    ]
    initializer_names = {item.name for item in inferred.graph.initializer}
    dependencies = graph_input_dependencies(inferred, inferred.graph.output[0].name)
    result = {
        "checker": "passed",
        "shape_inference": "passed",
        "opset_imports": {item.domain or "ai.onnx": int(item.version) for item in inferred.opset_import},
        "inputs": inputs,
        "outputs": outputs,
        "node_count": len(inferred.graph.node),
        "python_ops": python_ops,
        "aten_fallback": aten_fallback,
        "custom_domains": custom_domains,
        "torch_cluster_nodes": torch_cluster_nodes,
        "large_constants_threshold_bytes": 16 * 1024 * 1024,
        "large_constants": large_constants,
        "output_input_dependencies": sorted(dependencies),
        "adj_is_graph_input": "adj" in {item["name"] for item in inputs},
        "adj_is_initializer": "adj" in initializer_names,
        "output_depends_on_adj": "adj" in dependencies,
        "output_depends_on_points": "points" in dependencies,
    }
    expected_inputs = [
        {"name": "points", "shape": [1, 2048, 4], "dtype": "FLOAT"},
        {"name": "adj", "shape": [1, 2048, 2048], "dtype": "FLOAT"},
    ]
    expected_outputs = [{"name": "logits", "shape": [1, 2048, 2], "dtype": "FLOAT"}]
    if inputs != expected_inputs or outputs != expected_outputs:
        raise RuntimeError(f"ONNX interface mismatch: inputs={inputs}, outputs={outputs}")
    if python_ops or aten_fallback or custom_domains or torch_cluster_nodes:
        raise RuntimeError(
            "Forbidden ONNX nodes: "
            f"PythonOp={python_ops}, ATen={aten_fallback}, custom_domains={custom_domains}, "
            f"torch_cluster={torch_cluster_nodes}"
        )
    if large_constants:
        raise RuntimeError(f"Large Constant nodes found: {large_constants}")
    if (
        not result["adj_is_graph_input"]
        or result["adj_is_initializer"]
        or not result["output_depends_on_adj"]
        or not result["output_depends_on_points"]
    ):
        raise RuntimeError(f"Adjacency was removed, frozen, or disconnected: {result}")
    logger.info("ONNX inspection: %s", json.dumps(result, ensure_ascii=False))
    return result


def main() -> int:
    args = parse_args()
    run_dir = make_run_directory(args.run_id)
    logger = make_logger(run_dir)
    model_path = run_dir / MODEL_NAME
    config: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "original_model_source": str(PROJECT_ROOT / "models" / "testParameters" / "GCN_res" / "model.py"),
        "model_source": str(PROJECT_ROOT / "deployment" / "gcn_res_onnx_model.py"),
        "voxel_pool_source": str(PROJECT_ROOT / "deployment" / "onnx_voxel_pool.py"),
        "checkpoint": str(CHECKPOINT),
        "baseline_directory": str(BASELINE_DIR),
        "source_npz": str(SOURCE_NPZ),
        "onnx_path": str(model_path),
        "batch_size": BATCH_SIZE,
        "num_points": NUM_POINTS,
        "input_dim": INPUT_DIM,
        "num_classes": NUM_CLASSES,
        "k_neighbors": K_NEIGHBORS,
        "dtype": "float32",
        "device": DEVICE,
        "opset": OPSET,
        "dynamic_axes": False,
        "aten_fallback": False,
        "custom_symbolics_registered": False,
        "deployment_math": "standard-operator equivalent voxel pooling",
        "status": "started",
    }
    OmegaConf.save(OmegaConf.create(config), run_dir / "config_resolved.yaml", resolve=True)
    try:
        seed_everything()
        logger.info(
            "Startup PROJECT_ROOT=%s cwd=%s python=%s torch=%s cuda_runtime=%s",
            PROJECT_ROOT,
            Path.cwd(),
            sys.executable,
            torch.__version__,
            torch.version.cuda,
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        logger.info("GPU=%s capability=%s", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
        points_np, adjacency_np, input_metadata = build_fixed_inputs()
        np.savez_compressed(
            run_dir / "export_input.npz",
            points=points_np,
            adj=adjacency_np,
            normalized_xyz=input_metadata["normalized_xyz"],
            sample_indices=input_metadata["sample_indices"],
            ground_truth_labels=input_metadata["ground_truth_labels"],
            source_npz=np.asarray(str(SOURCE_NPZ)),
        )
        wrapper, checkpoint_info = load_wrapper(logger)
        points = torch.from_numpy(points_np).to(DEVICE)
        adjacency = torch.from_numpy(adjacency_np).to(DEVICE)
        with torch.inference_mode():
            logits = wrapper(points, adjacency)
        torch.cuda.synchronize()
        if logits.shape != (1, 2048, 2) or not torch.isfinite(logits).all():
            raise RuntimeError(f"Invalid PyTorch reference logits: {logits.shape}")
        np.savez_compressed(
            run_dir / "pytorch_deploy_reference.npz",
            logits=logits.cpu().numpy(),
            evaluation_baseline_logits=input_metadata["evaluation_baseline_logits"],
            source_npz=np.asarray(str(SOURCE_NPZ)),
        )
        logger.info(
            "Fixed export input points=%s adj=%s output logits=%s",
            tuple(points.shape),
            tuple(adjacency.shape),
            tuple(logits.shape),
        )

        export_start = time.perf_counter()
        torch.onnx.export(
            wrapper,
            (points, adjacency),
            model_path,
            export_params=True,
            opset_version=OPSET,
            do_constant_folding=True,
            input_names=["points", "adj"],
            output_names=["logits"],
            dynamic_axes=None,
            dynamo=False,
            verbose=False,
        )
        export_seconds = time.perf_counter() - export_start
        if not model_path.is_file() or model_path.stat().st_size == 0:
            raise RuntimeError("torch.onnx.export did not create a non-empty model")
        inspection = inspect_onnx(model_path, logger)
        config.update(
            {
                "status": "GCN_RES_ONNX_EXPORT_PASSED",
                "export_seconds": export_seconds,
                "onnx_size_bytes": model_path.stat().st_size,
                "checkpoint_metadata": checkpoint_info,
                "onnx_inspection": inspection,
            }
        )
        OmegaConf.save(OmegaConf.create(config), run_dir / "config_resolved.yaml", resolve=True)
        logger.info("GCN_RES_ONNX_EXPORT_PASSED artifact_dir=%s", run_dir)
        print(f"GCN_RES_ONNX_EXPORT_PASSED\nARTIFACT_DIR={run_dir}")
        return 0
    except Exception as exc:
        config.update(
            {
                "status": "GCN_RES_ONNX_EXPORT_FAILED",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "traceback": traceback.format_exc(),
                "stop_policy": "Stopped at the first export/check/inspection failure; no model rewrite attempted.",
            }
        )
        OmegaConf.save(OmegaConf.create(config), run_dir / "config_resolved.yaml", resolve=True)
        logger.exception("GCN_RES_ONNX_EXPORT_FAILED; validation must not run")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
