"""Validate the strict-FP32 GCN_res TensorRT engine on the fixed test split.

The script performs correctness validation only.  It does not rebuild or modify
the engine, ONNX graph, VoxelUnique plugin, checkpoint or dataset.  TF32 is
disabled for the PyTorch CUDA reference, and no FP16, INT8, warmup or benchmark
path is used.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import build_gcn_res_tensorrt_fp32 as phase4  # noqa: E402
import compare_tensorrt_pytorch_logits as parity  # noqa: E402
import evaluate_gcn_res_checkpoint as evaluation  # noqa: E402
import run_gcn_res_tensorrt_fp32_inference as phase5  # noqa: E402


SEED = 42
NUM_POINTS = 2048
K_NEIGHBORS = 6
EXPECTED_TEST_SAMPLES = 18
MAX_ABS_THRESHOLD = 1.0e-4
COSINE_THRESHOLD = 0.99999
AGREEMENT_THRESHOLD = 0.999
METRIC_DELTA_THRESHOLD = 1.0e-5

DEFAULT_ENGINE = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260716_224643_531592_strict_fp32"
    / "strict_fp32.plan"
)
DEFAULT_ONNX = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260716_190125_699274_dds_reshape_rewrite"
    / "dds_reshape_rewritten.onnx"
)
DEFAULT_PLUGIN_LIBRARY = phase4.DEFAULT_PLUGIN_LIBRARY
DEFAULT_CHECKPOINT = phase5.DEFAULT_CHECKPOINT
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "data" / "weld" / "train_test_split"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
DEFAULT_TENSORRT_ROOT = phase5.DEFAULT_TENSORRT_ROOT
DEFAULT_CUDA_ROOT = phase5.DEFAULT_CUDA_ROOT

EXPECTED_ENGINE_SHA256 = "b76563580089bbc0684e7a8e1edda6028c2d89fd91397c22b4efbd89ecd7fc2c"
EXPECTED_ONNX_SHA256 = "f71a585ded3f348a193e37395a124ab8a425ff72fc09ae92114d41872c460b98"


def dump_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def make_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"tensorrt_phase6.{run_dir.name}")
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


def array_sha256(array: np.ndarray) -> str:
    return phase5.array_sha256(np.ascontiguousarray(array))


def split_entries(split_root: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    entries: dict[str, list[str]] = {}
    metadata: dict[str, Any] = {}
    all_entries: list[str] = []
    expected_counts = {"train": 54, "val": 18, "test": 18}
    for split, expected_count in expected_counts.items():
        path = split_root / f"sub_shuffled_{split}_file_list.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or len(payload) != expected_count:
            raise ValueError(
                f"Expected {expected_count} {split} entries, got "
                f"{len(payload) if isinstance(payload, list) else type(payload)}"
            )
        normalized = [str(item).replace("\\", "/") for item in payload]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"Duplicate entries in {path}")
        entries[split] = normalized
        all_entries.extend(normalized)
        metadata[split] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "count": len(normalized),
            "entries": normalized,
        }
    if len(all_entries) != len(set(all_entries)):
        raise ValueError("The fixed train/val/test sub splits overlap")
    metadata["sets_disjoint"] = True
    metadata["only_test_executed"] = True
    return entries, metadata


def segmentation_metric_deltas(
    pytorch_metrics: dict[str, Any], tensorrt_metrics: dict[str, Any]
) -> dict[str, float | None]:
    keys = (
        "overall_accuracy",
        "weld_seam_iou",
        "background_iou",
        "miou",
        "weld_seam_precision",
        "weld_seam_recall",
        "weld_seam_f1",
    )
    result: dict[str, float | None] = {}
    for key in keys:
        left = pytorch_metrics.get(key)
        right = tensorrt_metrics.get(key)
        result[key] = None if left is None or right is None else abs(float(left) - float(right))
    return result


def numerical_comparison(
    pytorch_logits: np.ndarray, tensorrt_logits: np.ndarray
) -> dict[str, Any]:
    if pytorch_logits.shape != (1, NUM_POINTS, 2):
        raise ValueError(f"Unexpected PyTorch logits shape: {pytorch_logits.shape}")
    if tensorrt_logits.shape != pytorch_logits.shape:
        raise ValueError(
            f"TensorRT/PyTorch shape mismatch: {tensorrt_logits.shape} != {pytorch_logits.shape}"
        )
    if pytorch_logits.dtype != np.float32 or tensorrt_logits.dtype != np.float32:
        raise TypeError(
            f"Expected FP32 logits, got PT={pytorch_logits.dtype}, TRT={tensorrt_logits.dtype}"
        )
    finite = bool(
        np.isfinite(pytorch_logits).all() and np.isfinite(tensorrt_logits).all()
    )
    difference = tensorrt_logits.astype(np.float64) - pytorch_logits.astype(np.float64)
    absolute = np.abs(difference)
    torch_flat = pytorch_logits.astype(np.float64).reshape(-1)
    trt_flat = tensorrt_logits.astype(np.float64).reshape(-1)
    denominator = float(np.linalg.norm(torch_flat) * np.linalg.norm(trt_flat))
    cosine = (
        float(np.dot(torch_flat, trt_flat) / denominator) if denominator > 0.0 else None
    )
    torch_labels = np.argmax(pytorch_logits, axis=-1)
    trt_labels = np.argmax(tensorrt_logits, axis=-1)
    matching = int((torch_labels == trt_labels).sum())
    total = int(torch_labels.size)
    result = {
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "cosine_similarity": cosine,
        "matching_points": matching,
        "total_points": total,
        "agreement": float(matching / total),
        "outputs_finite": finite,
    }
    result["parity_passed"] = bool(
        finite
        and result["max_abs_error"] < MAX_ABS_THRESHOLD
        and cosine is not None
        and cosine > COSINE_THRESHOLD
        and result["agreement"] > AGREEMENT_THRESHOLD
    )
    return result


class PyTorchStrictFP32Runner:
    def __init__(self, checkpoint_path: Path) -> None:
        import torch

        from deployment.gcn_res_onnx_model import GCNResStandardOps
        from deployment.gcn_res_onnx_wrapper import GCNResOnnxWrapper

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise KeyError("checkpoint.model_state_dict missing")
        state_dict = checkpoint["model_state_dict"]
        if tuple(state_dict["linear_1.weight"].shape) != (48, 4):
            raise RuntimeError("linear_1.weight is not (48,4)")
        if tuple(state_dict["mlp.weight"].shape) != (2, 48):
            raise RuntimeError("mlp.weight is not (2,48)")
        non_finite = [
            name
            for name, tensor in state_dict.items()
            if torch.is_tensor(tensor) and not torch.isfinite(tensor).all()
        ]
        if non_finite:
            raise FloatingPointError(f"Checkpoint contains non-finite tensors: {non_finite}")
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model = GCNResStandardOps(SimpleNamespace(num_class=2), in_dim=4)
        strict_result = model.load_state_dict(state_dict, strict=True)
        self.wrapper = GCNResOnnxWrapper(model).to("cuda:0").eval()
        self.metadata = {
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "strict_load_result": str(strict_result),
            "linear_1_weight_shape": list(state_dict["linear_1.weight"].shape),
            "mlp_weight_shape": list(state_dict["mlp.weight"].shape),
            "device": "cuda:0",
        }
        del checkpoint, state_dict, model

    def infer(self, points: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        import torch

        points_tensor = torch.from_numpy(points).to("cuda:0")
        adjacency_tensor = torch.from_numpy(adjacency).to("cuda:0")
        with torch.inference_mode():
            logits_tensor = self.wrapper(points_tensor, adjacency_tensor)
        logits = np.ascontiguousarray(logits_tensor.detach().cpu().numpy(), dtype=np.float32)
        if logits.shape != (1, NUM_POINTS, 2) or not np.isfinite(logits).all():
            raise RuntimeError(f"Invalid PyTorch logits: {logits.shape}")
        del logits_tensor, adjacency_tensor, points_tensor
        return logits

    def close(self) -> None:
        import torch

        del self.wrapper
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(0)


class TensorRTRunner:
    def __init__(self, trt: Any, cudart: Any, engine_path: Path, plugin_path: Path) -> None:
        self.trt = trt
        self.cudart = cudart
        self.logger = trt.Logger(trt.Logger.INFO)
        if not trt.init_libnvinfer_plugins(self.logger, ""):
            raise RuntimeError("trt.init_libnvinfer_plugins returned false")
        self.plugin_library, plugin_info = phase4.load_plugin_library(plugin_path)
        if not plugin_info["registration_function_returned"]:
            raise RuntimeError("VoxelUnique plugin registration failed")
        registry = phase4.collect_registry(trt, trt.get_plugin_registry())
        if not registry["voxel_unique_creator_found"]:
            raise RuntimeError("VoxelUnique Creator not found")
        self.registry = registry
        ErrorRecorder = phase5.make_error_recorder_class(trt)
        self.recorder = ErrorRecorder()
        self.runtime = trt.Runtime(self.logger)
        self.runtime.error_recorder = self.recorder
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError("deserialize_cuda_engine returned None")
        self.engine.error_recorder = self.recorder
        runtime_instances = int(self.plugin_library.getVoxelUniqueRuntimeCreationCount())
        if runtime_instances != 4:
            raise RuntimeError(
                f"Expected 4 VoxelUnique runtime instances, got {runtime_instances}"
            )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("create_execution_context returned None")
        self.context.error_recorder = self.recorder
        self.io_records = phase5.engine_io_records(trt, self.engine, self.context)
        self.logits = np.empty((1, NUM_POINTS, 2), dtype=np.float32, order="C")
        self.device_pointers: dict[str, int] = {}
        self.stream = None
        phase5.cuda_call(cudart, "cudaSetDevice(0)", cudart.cudaSetDevice(0))
        self.stream = phase5.cuda_call(
            cudart, "cudaStreamCreate", cudart.cudaStreamCreate()
        )[0]
        buffer_sizes = {
            "points": 1 * NUM_POINTS * 4 * np.dtype(np.float32).itemsize,
            "adj": 1 * NUM_POINTS * NUM_POINTS * np.dtype(np.float32).itemsize,
            "logits": self.logits.nbytes,
        }
        for name, byte_count in buffer_sizes.items():
            self.device_pointers[name] = int(
                phase5.cuda_call(
                    cudart, f"cudaMalloc({name})", cudart.cudaMalloc(byte_count)
                )[0]
            )
            if not self.context.set_tensor_address(name, self.device_pointers[name]):
                raise RuntimeError(f"set_tensor_address failed for {name}")
        self.metadata = {
            "plugin_info": plugin_info,
            "plugin_creator": registry["voxel_unique"],
            "runtime_instances": runtime_instances,
            "engine_io": self.io_records,
            "buffer_bytes": buffer_sizes,
            "context_reused_across_samples": True,
            "single_stream_reused_across_samples": True,
        }

    def infer(self, points: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        if points.shape != (1, NUM_POINTS, 4) or adjacency.shape != (
            1,
            NUM_POINTS,
            NUM_POINTS,
        ):
            raise ValueError(
                f"Input shape mismatch: points={points.shape}, adj={adjacency.shape}"
            )
        for name, array in (("points", points), ("adj", adjacency)):
            phase5.cuda_call(
                self.cudart,
                f"cudaMemcpyAsync H2D {name}",
                self.cudart.cudaMemcpyAsync(
                    self.device_pointers[name],
                    int(array.ctypes.data),
                    int(array.nbytes),
                    self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                ),
            )
        if not self.context.execute_async_v3(stream_handle=int(self.stream)):
            raise RuntimeError("execute_async_v3 returned false")
        phase5.cuda_call(
            self.cudart,
            "cudaMemcpyAsync D2H logits",
            self.cudart.cudaMemcpyAsync(
                int(self.logits.ctypes.data),
                self.device_pointers["logits"],
                int(self.logits.nbytes),
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            ),
        )
        phase5.cuda_call(
            self.cudart,
            "cudaStreamSynchronize",
            self.cudart.cudaStreamSynchronize(self.stream),
        )
        if self.recorder.num_errors:
            raise RuntimeError(
                f"TensorRT ErrorRecorder contains errors: {self.recorder.serializable()}"
            )
        if not np.isfinite(self.logits).all():
            raise FloatingPointError("TensorRT logits contain NaN/Inf")
        return np.ascontiguousarray(self.logits.copy(), dtype=np.float32)

    def close(self) -> dict[str, Any]:
        for name, pointer in reversed(list(self.device_pointers.items())):
            phase5.cuda_call(
                self.cudart, f"cudaFree({name})", self.cudart.cudaFree(pointer)
            )
        self.device_pointers.clear()
        if self.stream is not None:
            phase5.cuda_call(
                self.cudart,
                "cudaStreamDestroy",
                self.cudart.cudaStreamDestroy(self.stream),
            )
            self.stream = None
        error_summary = {
            "num_errors": self.recorder.num_errors,
            "has_overflowed": self.recorder.has_overflowed(),
            "errors": self.recorder.serializable(),
        }
        del self.context, self.engine, self.runtime, self.recorder, self.plugin_library
        gc.collect()
        return error_summary


def collect_environment(
    trt: Any,
    cudart: Any,
    args: argparse.Namespace,
    split_metadata: dict[str, Any],
    plugin_creator: dict[str, Any] | None,
) -> dict[str, Any]:
    import torch

    runtime_version = phase5.cuda_call(
        cudart, "cudaRuntimeGetVersion", cudart.cudaRuntimeGetVersion()
    )[0]
    driver_version = phase5.cuda_call(
        cudart, "cudaDriverGetVersion", cudart.cudaDriverGetVersion()
    )[0]
    properties = torch.cuda.get_device_properties(0)
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "tensorrt_version": trt.__version__,
        "cuda_python_version": package_version("cuda-python"),
        "cuda_bindings_version": package_version("cuda-bindings"),
        "cuda_runtime_api_version": int(runtime_version),
        "cuda_driver_api_version": int(driver_version),
        "cuda_toolkit_root": str(args.cuda_root.resolve()),
        "tensorrt_root": str(args.tensorrt_root.resolve()),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(properties.total_memory),
        },
        "engine": {
            "path": str(args.engine.resolve()),
            "size_bytes": args.engine.stat().st_size,
            "sha256": sha256(args.engine),
        },
        "onnx": {
            "path": str(args.onnx.resolve()),
            "size_bytes": args.onnx.stat().st_size,
            "sha256": sha256(args.onnx),
        },
        "plugin": {
            "path": str(args.plugin_library.resolve()),
            "size_bytes": args.plugin_library.stat().st_size,
            "sha256": sha256(args.plugin_library),
            "creator": plugin_creator,
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "size_bytes": args.checkpoint.stat().st_size,
            "sha256": sha256(args.checkpoint),
        },
        "splits": split_metadata,
        "validation_policy": {
            "test_samples_only": True,
            "seed": SEED,
            "num_points": NUM_POINTS,
            "k_neighbors": K_NEIGHBORS,
            "tensorrt_tf32": False,
            "pytorch_matmul_tf32": False,
            "pytorch_cudnn_tf32": False,
            "fp16": False,
            "int8": False,
            "benchmark": False,
        },
        "pip_check": phase4.command_output([sys.executable, "-m", "pip", "check"]),
    }


def write_worst_case(path: Path, report: dict[str, Any]) -> None:
    worst = report["worst_case_sample"]
    path.write_text(
        f"""# Strict FP32 multi-sample worst case

- Sample: `{worst['sample_id']}`
- Source: `{worst['logical_path']}`
- max absolute error: `{worst['max_abs_error']:.12e}`
- mean absolute error: `{worst['mean_abs_error']:.12e}`
- RMSE: `{worst['rmse']:.12e}`
- cosine similarity: `{worst['cosine_similarity']:.15f}`
- point agreement: `{worst['matching_points']}/{worst['total_points']} = {worst['agreement']:.12f}`
- sample parity passed: `{worst['passed']}`

The worst case is selected by maximum absolute logits error.  No engine, ONNX,
Plugin, checkpoint, dataset, tolerance or precision setting was modified.

`{report['status']}`
""",
        encoding="utf-8",
    )


def main(args: argparse.Namespace) -> int:
    source_paths = {
        "engine": args.engine.resolve(),
        "onnx": args.onnx.resolve(),
        "plugin": args.plugin_library.resolve(),
        "checkpoint": args.checkpoint.resolve(),
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(source_paths["engine"]) != EXPECTED_ENGINE_SHA256:
        raise RuntimeError("Strict FP32 Engine SHA-256 mismatch")
    if sha256(source_paths["onnx"]) != EXPECTED_ONNX_SHA256:
        raise RuntimeError("Source ONNX SHA-256 mismatch")
    _, split_metadata = split_entries(args.split_root.resolve())
    run_dir = (
        args.output_root.resolve()
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_strict_fp32_multisample"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir()
    logger = make_logger(run_dir)
    initial = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "TENSORRT_STRICT_FP32_MULTISAMPLE_VALIDATION_FAILED",
        "run_dir": str(run_dir),
        "source_hashes_before": {name: sha256(path) for name, path in source_paths.items()},
        "test_samples_only": True,
        "fp16": False,
        "int8": False,
        "benchmark": False,
    }
    dump_json(run_dir / "strict_fp32_validation_report.json", initial)

    dll_handles: list[Any] = []
    torch_runner: PyTorchStrictFP32Runner | None = None
    trt_runner: TensorRTRunner | None = None
    old_matmul_tf32: bool | None = None
    old_cudnn_tf32: bool | None = None
    old_precision: str | None = None
    try:
        dll_handles = phase5.configure_dll_search(
            args.tensorrt_root.resolve(),
            args.cuda_root.resolve(),
            source_paths["plugin"],
        )
        import tensorrt as trt
        import torch
        from cuda.bindings import runtime as cudart

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        old_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        old_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
        old_precision = torch.get_float32_matmul_precision()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

        dataset = evaluation.FixedWeldEvaluationDataset("test")
        if len(dataset) != EXPECTED_TEST_SAMPLES:
            raise RuntimeError(f"Expected 18 test samples, got {len(dataset)}")
        expected_names = [Path(item).stem for item in split_metadata["test"]["entries"]]
        actual_names = [record.sample_name for record in dataset.records]
        if actual_names != expected_names:
            raise RuntimeError(
                f"Dataset order differs from fixed test JSON: {actual_names} != {expected_names}"
            )

        torch_runner = PyTorchStrictFP32Runner(source_paths["checkpoint"])
        trt_runner = TensorRTRunner(
            trt, cudart, source_paths["engine"], source_paths["plugin"]
        )
        environment = collect_environment(
            trt, cudart, args, split_metadata, trt_runner.registry["voxel_unique"]
        )
        environment["tensorrt_runtime"] = trt_runner.metadata
        environment["pytorch_model"] = torch_runner.metadata
        dump_json(run_dir / "environment.json", environment)

        per_sample: list[dict[str, Any]] = []
        all_labels: list[np.ndarray] = []
        all_pytorch_predictions: list[np.ndarray] = []
        all_tensorrt_predictions: list[np.ndarray] = []

        for index in range(len(dataset)):
            sample = dataset[index]
            sample_id = str(sample["sample_name"])
            normalized_xyz = sample["normalized_xyz"].unsqueeze(0).to(torch.float32)
            points_tensor = evaluation.make_model_input(normalized_xyz)
            adjacency_tensor, _ = evaluation.build_adjacency_cpu(normalized_xyz)
            points = np.ascontiguousarray(points_tensor.numpy(), dtype=np.float32)
            adjacency = np.ascontiguousarray(adjacency_tensor.numpy(), dtype=np.float32)
            labels = np.ascontiguousarray(
                sample["labels"].unsqueeze(0).numpy(), dtype=np.int64
            )
            if points.shape != (1, NUM_POINTS, 4) or adjacency.shape != (
                1,
                NUM_POINTS,
                NUM_POINTS,
            ):
                raise RuntimeError(
                    f"{sample_id}: invalid input shape points={points.shape}, adj={adjacency.shape}"
                )
            if labels.shape != (1, NUM_POINTS):
                raise RuntimeError(f"{sample_id}: invalid label shape {labels.shape}")
            if not np.isfinite(points).all() or not np.isfinite(adjacency).all():
                raise FloatingPointError(f"{sample_id}: non-finite model input")

            pytorch_logits = torch_runner.infer(points, adjacency)
            tensorrt_logits = trt_runner.infer(points, adjacency)
            sample_dir = predictions_dir / f"{index:02d}_{sample_id}"
            sample_dir.mkdir()
            pytorch_path = sample_dir / "pytorch_logits.npy"
            tensorrt_path = sample_dir / "tensorrt_logits.npy"
            np.save(pytorch_path, pytorch_logits, allow_pickle=False)
            np.save(tensorrt_path, tensorrt_logits, allow_pickle=False)

            comparison = numerical_comparison(pytorch_logits, tensorrt_logits)
            pytorch_prediction = np.argmax(pytorch_logits, axis=-1)
            tensorrt_prediction = np.argmax(tensorrt_logits, axis=-1)
            pytorch_metrics = parity.segmentation_metrics(labels, pytorch_prediction)
            tensorrt_metrics = parity.segmentation_metrics(labels, tensorrt_prediction)
            metric_deltas = segmentation_metric_deltas(pytorch_metrics, tensorrt_metrics)
            record = {
                "sample_index": index,
                "sample_id": sample_id,
                "logical_path": str(sample["logical_path"]),
                "sample_indices_sha256": array_sha256(sample["sample_indices"].numpy()),
                "points_sha256": array_sha256(points),
                "adj_sha256": array_sha256(adjacency),
                "pytorch_logits_path": str(pytorch_path),
                "pytorch_logits_sha256": sha256(pytorch_path),
                "tensorrt_logits_path": str(tensorrt_path),
                "tensorrt_logits_sha256": sha256(tensorrt_path),
                **comparison,
                "cosine": comparison["cosine_similarity"],
                "miou": tensorrt_metrics["miou"],
                "f1": tensorrt_metrics["weld_seam_f1"],
                "pytorch_metrics": pytorch_metrics,
                "tensorrt_metrics": tensorrt_metrics,
                "metric_deltas": metric_deltas,
                "passed": comparison["parity_passed"],
            }
            per_sample.append(record)
            all_labels.append(labels.reshape(-1))
            all_pytorch_predictions.append(pytorch_prediction.reshape(-1))
            all_tensorrt_predictions.append(tensorrt_prediction.reshape(-1))
            dump_json(run_dir / "per_sample_results.json", per_sample)
            logger.info(
                "%02d/%02d %-8s max_abs=%.9e mean_abs=%.9e rmse=%.9e "
                "cosine=%.12f agreement=%d/%d passed=%s",
                index + 1,
                len(dataset),
                sample_id,
                record["max_abs_error"],
                record["mean_abs_error"],
                record["rmse"],
                record["cosine_similarity"],
                record["matching_points"],
                record["total_points"],
                record["passed"],
            )

        labels_all = np.concatenate(all_labels)
        pytorch_predictions_all = np.concatenate(all_pytorch_predictions)
        tensorrt_predictions_all = np.concatenate(all_tensorrt_predictions)
        pytorch_metrics = parity.segmentation_metrics(labels_all, pytorch_predictions_all)
        tensorrt_metrics = parity.segmentation_metrics(labels_all, tensorrt_predictions_all)
        metric_deltas = segmentation_metric_deltas(pytorch_metrics, tensorrt_metrics)
        required_metric_deltas = {
            "miou": metric_deltas["miou"],
            "weld_seam_f1": metric_deltas["weld_seam_f1"],
        }
        metrics_passed = all(
            value is not None and value < METRIC_DELTA_THRESHOLD
            for value in required_metric_deltas.values()
        )
        passed_samples = sum(bool(item["passed"]) for item in per_sample)
        worst_max = max(per_sample, key=lambda item: item["max_abs_error"])
        worst_cosine = min(per_sample, key=lambda item: item["cosine_similarity"])
        worst_agreement = min(per_sample, key=lambda item: item["agreement"])
        status = (
            "TENSORRT_STRICT_FP32_MULTISAMPLE_VALIDATION_PASSED"
            if passed_samples == len(dataset) and metrics_passed
            else "TENSORRT_STRICT_FP32_MULTISAMPLE_VALIDATION_FAILED"
        )

        error_summary = trt_runner.close()
        trt_runner = None
        torch_runner.close()
        torch_runner = None
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.set_float32_matmul_precision(old_precision)
        settings_restored = bool(
            torch.backends.cuda.matmul.allow_tf32 == old_matmul_tf32
            and torch.backends.cudnn.allow_tf32 == old_cudnn_tf32
            and torch.get_float32_matmul_precision() == old_precision
        )

        source_hashes_after = {name: sha256(path) for name, path in source_paths.items()}
        source_hashes_unchanged = initial["source_hashes_before"] == source_hashes_after
        report = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "status": status,
            "run_dir": str(run_dir),
            "total_samples": len(dataset),
            "passed_samples": passed_samples,
            "total_points": int(labels_all.size),
            "worst_max_abs_error": float(worst_max["max_abs_error"]),
            "average_max_abs_error": float(
                np.mean([item["max_abs_error"] for item in per_sample])
            ),
            "worst_cosine_similarity": float(worst_cosine["cosine_similarity"]),
            "mean_point_agreement": float(
                np.mean([item["agreement"] for item in per_sample])
            ),
            "pytorch_metrics": pytorch_metrics,
            "tensorrt_metrics": tensorrt_metrics,
            "metric_deltas": metric_deltas,
            "required_metric_deltas": required_metric_deltas,
            "metrics_delta_passed": metrics_passed,
            "worst_case_sample": worst_max,
            "worst_cosine_sample": {
                "sample_id": worst_cosine["sample_id"],
                "cosine_similarity": worst_cosine["cosine_similarity"],
            },
            "worst_agreement_sample": {
                "sample_id": worst_agreement["sample_id"],
                "agreement": worst_agreement["agreement"],
                "matching_points": worst_agreement["matching_points"],
                "total_points": worst_agreement["total_points"],
            },
            "acceptance": {
                "every_sample_max_abs_error_strictly_less_than": MAX_ABS_THRESHOLD,
                "every_sample_cosine_strictly_greater_than": COSINE_THRESHOLD,
                "every_sample_agreement_strictly_greater_than": AGREEMENT_THRESHOLD,
                "aggregate_miou_and_f1_delta_strictly_less_than": METRIC_DELTA_THRESHOLD,
                "all_outputs_finite": True,
                "passed": status.endswith("_PASSED"),
            },
            "engine_sha256": initial["source_hashes_before"]["engine"],
            "onnx_sha256": initial["source_hashes_before"]["onnx"],
            "plugin_sha256": initial["source_hashes_before"]["plugin"],
            "checkpoint_sha256": initial["source_hashes_before"]["checkpoint"],
            "test_split_sha256": split_metadata["test"]["sha256"],
            "tensorrt_error_recorder": error_summary,
            "pytorch_strict_fp32_policy": {
                "matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "float32_matmul_precision": "highest",
                "settings_restored": settings_restored,
            },
            "source_hashes_before": initial["source_hashes_before"],
            "source_hashes_after": source_hashes_after,
            "source_hashes_unchanged": source_hashes_unchanged,
            "only_test_split_executed": True,
            "engine_modified": False,
            "onnx_modified": False,
            "plugin_modified": False,
            "checkpoint_modified": False,
            "fp16": False,
            "int8": False,
            "benchmark": False,
        }
        if error_summary["num_errors"] != 0:
            raise RuntimeError(f"TensorRT ErrorRecorder contains errors: {error_summary}")
        if not settings_restored:
            raise RuntimeError("PyTorch precision settings were not restored")
        if not source_hashes_unchanged:
            raise RuntimeError("A read-only source hash changed during validation")
        dump_json(run_dir / "strict_fp32_validation_report.json", report)
        dump_json(run_dir / "per_sample_results.json", per_sample)
        write_worst_case(run_dir / "worst_case_sample.md", report)
        logger.info(
            "aggregate status=%s passed=%d/%d worst_max=%.9e avg_max=%.9e "
            "worst_cosine=%.12f mean_agreement=%.12f miou_delta=%.9e f1_delta=%.9e",
            status,
            passed_samples,
            len(dataset),
            report["worst_max_abs_error"],
            report["average_max_abs_error"],
            report["worst_cosine_similarity"],
            report["mean_point_agreement"],
            required_metric_deltas["miou"],
            required_metric_deltas["weld_seam_f1"],
        )
        print(f"RUN_DIR={run_dir}")
        print(f"TOTAL_SAMPLES={len(dataset)}")
        print(f"PASSED_SAMPLES={passed_samples}")
        print(f"WORST_CASE_SAMPLE={worst_max['sample_id']}")
        print(f"WORST_MAX_ABS_ERROR={report['worst_max_abs_error']:.12e}")
        print(f"AVERAGE_MAX_ABS_ERROR={report['average_max_abs_error']:.12e}")
        print(f"WORST_COSINE_SIMILARITY={report['worst_cosine_similarity']:.15f}")
        print(f"MEAN_POINT_AGREEMENT={report['mean_point_agreement']:.12f}")
        print(status)
        del dll_handles
        gc.collect()
        return 0 if status.endswith("_PASSED") else 2
    except Exception:
        failure = dict(initial)
        failure.update(
            {
                "timestamp": datetime.now().astimezone().isoformat(),
                "traceback": traceback.format_exc(),
                "source_hashes_after": {
                    name: sha256(path) for name, path in source_paths.items()
                },
            }
        )
        failure["source_hashes_unchanged"] = (
            failure["source_hashes_before"] == failure["source_hashes_after"]
        )
        dump_json(run_dir / "strict_fp32_validation_report.json", failure)
        logger.exception("Strict FP32 multi-sample validation failed")
        print(f"RUN_DIR={run_dir}")
        print("TENSORRT_STRICT_FP32_MULTISAMPLE_VALIDATION_FAILED")
        return 1
    finally:
        if trt_runner is not None:
            try:
                trt_runner.close()
            except Exception:
                logger.exception("TensorRT cleanup failed")
        if torch_runner is not None:
            try:
                torch_runner.close()
            except Exception:
                logger.exception("PyTorch cleanup failed")
        try:
            import torch

            if old_matmul_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            if old_cudnn_tf32 is not None:
                torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            if old_precision is not None:
                torch.set_float32_matmul_precision(old_precision)
        except Exception:
            logger.exception("Precision-setting restoration failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--plugin-library", type=Path, default=DEFAULT_PLUGIN_LIBRARY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TENSORRT_ROOT)
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
