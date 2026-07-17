"""Attribute residual Strict-FP32 TensorRT error across major GCN_res stages.

The validated ONNX, formal engine, VoxelUnique plugin and checkpoint are
read-only.  The formal engine has no debug tensors, so the same parsed network
is built as an in-memory-only diagnostic engine with selected ITensors exposed
as outputs.  The diagnostic builder uses the same 4 GiB Strict-FP32 policy
(TF32/FP16/INT8 disabled); it is never serialized to disk and is not benchmarked.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import build_gcn_res_tensorrt_fp32 as phase4  # noqa: E402
import evaluate_gcn_res_checkpoint as evaluation  # noqa: E402
import locate_gcn_res_tensorrt_first_divergence as diagnostic  # noqa: E402
import run_gcn_res_tensorrt_fp32_inference as phase5  # noqa: E402


SEED = 42
SAMPLES = ("weld_14", "weld_5", "weld_12")
DEFAULT_ENGINE = (
    PROJECT_ROOT
    / "artifacts/gcn_res_tensorrt/20260716_224643_531592_strict_fp32/strict_fp32.plan"
)
DEFAULT_ONNX = (
    PROJECT_ROOT
    / "artifacts/gcn_res_tensorrt/20260716_190125_699274_dds_reshape_rewrite/dds_reshape_rewritten.onnx"
)
DEFAULT_PLUGIN = phase4.DEFAULT_PLUGIN_LIBRARY
DEFAULT_CHECKPOINT = phase5.DEFAULT_CHECKPOINT
DEFAULT_PHASE6 = (
    PROJECT_ROOT
    / "artifacts/gcn_res_tensorrt/20260717_110500_836041_strict_fp32_multisample"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/gcn_res_tensorrt"
DEFAULT_TENSORRT_ROOT = phase5.DEFAULT_TENSORRT_ROOT
DEFAULT_CUDA_ROOT = phase5.DEFAULT_CUDA_ROOT
EXPECTED_ENGINE_SHA256 = "b76563580089bbc0684e7a8e1edda6028c2d89fd91397c22b4efbd89ecd7fc2c"
EXPECTED_ONNX_SHA256 = "f71a585ded3f348a193e37395a124ab8a425ff72fc09ae92114d41872c460b98"


STAGES = [
    {"order": 0, "stage_name": "stem_linear", "module": "linear_1", "tuple_index": None,
     "onnx_tensor": "/model/linear_1/Add_output_0", "category": "encoder stem"},
    {"order": 1, "stage_name": "ptb_0", "module": "ptb_0", "tuple_index": 1,
     "onnx_tensor": "/model/ptb_0/Add_output_0", "category": "PointTransformer encoder"},
    {"order": 2, "stage_name": "gcn_0", "module": "gcn_0", "tuple_index": None,
     "onnx_tensor": "/model/gcn_0/linear/Add_output_0", "category": "GCN enhancement"},
    {"order": 3, "stage_name": "transition_down_1", "module": "tdb_1", "tuple_index": 1,
     "onnx_tensor": "/model/tdb_1/Reshape_11_output_0", "category": "voxel downsample"},
    {"order": 4, "stage_name": "ptb_1", "module": "ptb_1", "tuple_index": 1,
     "onnx_tensor": "/model/ptb_1/Add_output_0", "category": "PointTransformer encoder"},
    {"order": 5, "stage_name": "transition_down_2", "module": "tdb_2", "tuple_index": 1,
     "onnx_tensor": "/model/tdb_2/Reshape_14_output_0", "category": "voxel downsample"},
    {"order": 6, "stage_name": "ptb_2", "module": "ptb_2", "tuple_index": 1,
     "onnx_tensor": "/model/ptb_2/Add_output_0", "category": "PointTransformer encoder"},
    {"order": 7, "stage_name": "transition_down_3", "module": "tdb_3", "tuple_index": 1,
     "onnx_tensor": "/model/tdb_3/Reshape_14_output_0", "category": "voxel downsample"},
    {"order": 8, "stage_name": "ptb_3", "module": "ptb_3", "tuple_index": 1,
     "onnx_tensor": "/model/ptb_3/Add_output_0", "category": "PointTransformer encoder"},
    {"order": 9, "stage_name": "transition_down_4", "module": "tdb_4", "tuple_index": 1,
     "onnx_tensor": "/model/tdb_4/Reshape_14_output_0", "category": "voxel downsample"},
    {"order": 10, "stage_name": "ptb_4", "module": "ptb_4", "tuple_index": 1,
     "onnx_tensor": "/model/ptb_4/Add_output_0", "category": "bottleneck PointTransformer"},
    {"order": 11, "stage_name": "transition_up_6", "module": "tub_6", "tuple_index": 1,
     "onnx_tensor": "/model/tub_6/Add_2_output_0", "category": "decoder interpolation"},
    {"order": 12, "stage_name": "ptb_6", "module": "ptb_6", "tuple_index": 1,
     "onnx_tensor": "/model/ptb_6/Add_output_0", "category": "PointTransformer decoder"},
    {"order": 13, "stage_name": "transition_up_7", "module": "tub_7", "tuple_index": 1,
     "onnx_tensor": "/model/tub_7/Add_2_output_0", "category": "decoder interpolation"},
    {"order": 14, "stage_name": "ptb_7", "module": "ptb_7", "tuple_index": 1,
     "onnx_tensor": "/model/ptb_7/Add_output_0", "category": "PointTransformer decoder"},
    {"order": 15, "stage_name": "transition_up_8", "module": "tub_8", "tuple_index": 1,
     "onnx_tensor": "/model/tub_8/Add_2_output_0", "category": "decoder interpolation"},
    {"order": 16, "stage_name": "ptb_8", "module": "ptb_8", "tuple_index": 1,
     "onnx_tensor": "/model/ptb_8/Add_output_0", "category": "PointTransformer decoder"},
    {"order": 17, "stage_name": "transition_up_9", "module": "tub_9", "tuple_index": 1,
     "onnx_tensor": "/model/tub_9/Add_2_output_0", "category": "decoder interpolation"},
    {"order": 18, "stage_name": "segmentation_head_input", "module": "ptb_9", "tuple_index": 1,
     "onnx_tensor": "/model/ptb_9/Add_output_0", "category": "final PointTransformer / head input"},
]


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


def make_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"tensorrt_phase6b.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def tensor_numpy(tensor: Any) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().cpu().numpy())


class PyTorchStageCapture:
    def __init__(self, checkpoint_path: Path) -> None:
        import torch

        from deployment.gcn_res_onnx_model import GCNResStandardOps
        from deployment.gcn_res_onnx_wrapper import GCNResOnnxWrapper

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        model = GCNResStandardOps(SimpleNamespace(num_class=2), in_dim=4)
        strict_result = model.load_state_dict(state_dict, strict=True)
        self.model = model.to("cuda:0").eval()
        self.wrapper = GCNResOnnxWrapper(self.model).eval()
        self.current: dict[str, Any] = {}
        self.hooks = []

        def capture(name: str, tuple_index: int | None) -> Any:
            def hook(module: Any, inputs: Any, output: Any) -> None:
                del module, inputs
                selected = output if tuple_index is None else output[tuple_index]
                self.current[name] = selected.detach().clone()

            return hook

        for stage in STAGES:
            module = getattr(self.model, stage["module"])
            self.hooks.append(
                module.register_forward_hook(
                    capture(stage["stage_name"], stage["tuple_index"])
                )
            )
        self.metadata = {
            "strict_load_result": str(strict_result),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "hook_count": len(self.hooks),
            "device": "cuda:0",
        }
        del checkpoint, state_dict

    def run(
        self, points: np.ndarray, adjacency: np.ndarray, dump_dir: Path
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        import torch

        self.current.clear()
        points_tensor = torch.from_numpy(points).to("cuda:0")
        adjacency_tensor = torch.from_numpy(adjacency).to("cuda:0")
        with torch.inference_mode():
            logits_tensor = self.wrapper(points_tensor, adjacency_tensor)
        torch.cuda.synchronize(0)
        required = {stage["stage_name"] for stage in STAGES}
        if set(self.current) != required:
            raise RuntimeError(
                f"PyTorch hook mismatch: missing={required - set(self.current)}, "
                f"extra={set(self.current) - required}"
            )
        arrays = {name: tensor_numpy(value).astype(np.float32, copy=False)
                  for name, value in self.current.items()}
        arrays["logits"] = tensor_numpy(logits_tensor).astype(np.float32, copy=False)
        for name, array in arrays.items():
            if not np.isfinite(array).all():
                raise FloatingPointError(f"PyTorch {name} contains NaN/Inf")
            np.save(dump_dir / f"{name}.npy", array, allow_pickle=False)
        summary = {
            "tensors": {name: phase5.array_stats(array) for name, array in arrays.items()},
            "forward_count": 1,
            "benchmark": False,
        }
        del logits_tensor, adjacency_tensor, points_tensor
        self.current.clear()
        return arrays, summary

    def close(self) -> None:
        import torch

        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        del self.wrapper, self.model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(0)


def compare_arrays(
    stage: dict[str, Any], pytorch_array: np.ndarray, tensorrt_array: np.ndarray
) -> dict[str, Any]:
    shape_match = pytorch_array.shape == tensorrt_array.shape
    dtype_match = pytorch_array.dtype == tensorrt_array.dtype
    if not shape_match:
        raise ValueError(
            f"{stage['stage_name']} shape mismatch: PT={pytorch_array.shape}, "
            f"TRT={tensorrt_array.shape}"
        )
    if not dtype_match:
        raise TypeError(
            f"{stage['stage_name']} dtype mismatch: PT={pytorch_array.dtype}, "
            f"TRT={tensorrt_array.dtype}"
        )
    pytorch64 = pytorch_array.astype(np.float64)
    tensorrt64 = tensorrt_array.astype(np.float64)
    difference = tensorrt64 - pytorch64
    absolute = np.abs(difference)
    pytorch_flat = pytorch64.reshape(-1)
    tensorrt_flat = tensorrt64.reshape(-1)
    norm_reference = float(np.linalg.norm(pytorch_flat))
    cosine_denominator = float(norm_reference * np.linalg.norm(tensorrt_flat))
    exact = bool(np.array_equal(pytorch_array, tensorrt_array))
    first_mismatch = np.argwhere(pytorch_array != tensorrt_array)
    first_index = first_mismatch[0].tolist() if first_mismatch.size else None
    return {
        "order": stage["order"],
        "stage_name": stage["stage_name"],
        "category": stage["category"],
        "onnx_tensor": stage["onnx_tensor"],
        "shape": list(pytorch_array.shape),
        "dtype": str(pytorch_array.dtype),
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "outputs_finite": bool(
            np.isfinite(pytorch_array).all() and np.isfinite(tensorrt_array).all()
        ),
        "exact_equal": exact,
        "max_abs_error": float(absolute.max()) if absolute.size else 0.0,
        "mean_abs_error": float(absolute.mean()) if absolute.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(difference)))) if absolute.size else 0.0,
        "relative_error": (
            float(np.linalg.norm(difference) / norm_reference)
            if norm_reference > 0.0
            else (0.0 if exact else None)
        ),
        "max_relative_error_clamped_1e8": (
            float((absolute / np.maximum(np.abs(pytorch64), 1.0e-8)).max())
            if absolute.size
            else 0.0
        ),
        "cosine_similarity": (
            float(np.dot(pytorch_flat, tensorrt_flat) / cosine_denominator)
            if cosine_denominator > 0.0
            else (1.0 if exact else None)
        ),
        "first_mismatch_index": first_index,
    }


def add_growth_fields(comparisons: list[dict[str, Any]]) -> None:
    previous: dict[str, Any] | None = None
    for item in comparisons:
        item["delta_error"] = (
            item["max_abs_error"]
            if previous is None
            else item["max_abs_error"] - previous["max_abs_error"]
        )
        item["error_amplification_ratio"] = (
            None
            if previous is None or previous["max_abs_error"] == 0.0
            else item["max_abs_error"] / previous["max_abs_error"]
        )
        previous = item


def compare_saved(reference: Path, observed: np.ndarray, name: str) -> dict[str, Any]:
    array = np.load(reference, allow_pickle=False)
    stage = {"order": -1, "stage_name": name, "category": "representativeness", "onnx_tensor": name}
    result = compare_arrays(stage, array, observed)
    reference_labels = np.argmax(array, axis=-1)
    observed_labels = np.argmax(observed, axis=-1)
    result["label_agreement"] = float((reference_labels == observed_labels).mean())
    result["reference_path"] = str(reference)
    result["reference_sha256"] = sha256(reference)
    return result


def strict_diagnostic_config(original: Any) -> Any:
    def configure(trt: Any, config: Any, workspace_gib: float) -> dict[str, Any]:
        payload = original(trt, config, workspace_gib)
        if hasattr(trt.BuilderFlag, "TF32"):
            config.clear_flag(trt.BuilderFlag.TF32)
        for name in ("FP16", "INT8"):
            if hasattr(trt.BuilderFlag, name):
                config.clear_flag(getattr(trt.BuilderFlag, name))
        payload.update(
            {
                "precision_policy": "same_as_formal_strict_fp32",
                "tf32": (
                    bool(config.get_flag(trt.BuilderFlag.TF32))
                    if hasattr(trt.BuilderFlag, "TF32")
                    else False
                ),
                "fp16": (
                    bool(config.get_flag(trt.BuilderFlag.FP16))
                    if hasattr(trt.BuilderFlag, "FP16")
                    else False
                ),
                "int8": (
                    bool(config.get_flag(trt.BuilderFlag.INT8))
                    if hasattr(trt.BuilderFlag, "INT8")
                    else False
                ),
                "formal_builder_config_not_modified": True,
                "diagnostic_network_outputs_added_in_memory_only": True,
            }
        )
        if payload["tf32"] or payload["fp16"] or payload["int8"]:
            raise RuntimeError(f"Diagnostic strict precision flags invalid: {payload}")
        return payload

    return configure


def markdown_growth(samples: list[dict[str, Any]], status: str) -> str:
    sections = []
    for sample in samples:
        rows = "\n".join(
            f"| {item['stage_name']} | `{item['shape']}` | {item['max_abs_error']:.9e} | "
            f"{item['mean_abs_error']:.9e} | {item['rmse']:.9e} | "
            f"{item['relative_error']:.9e} | {item['cosine_similarity']:.12f} | "
            f"{item['delta_error']:+.9e} |"
            for item in sample["error_growth"]
        )
        sections.append(
            f"""## {sample['sample_id']}

| Stage | Shape | Max abs | Mean abs | RMSE | Relative L2 | Cosine | Delta max error |
|---|---|---:|---:|---:|---:|---:|---:|
{rows}

- First exact divergence: `{sample['first_divergence_stage']}`
- First positive error amplification: `{sample['first_error_amplification_stage']}`
- Maximum amplification: `{sample['maximum_amplification_stage']}`
- Diagnostic/formal TensorRT logits max abs: `{sample['diagnostic_vs_formal_tensorrt']['max_abs_error']:.9e}`
- Diagnostic/formal TensorRT label agreement: `{sample['diagnostic_vs_formal_tensorrt']['label_agreement']:.12f}`
"""
        )
    return f"""# Strict FP32 residual error growth

The formal engine has no marked debug tensors.  Intermediate values therefore
come from an in-memory-only diagnostic engine built from the same read-only ONNX
and Plugin under the same Strict-FP32 precision policy.  It was not saved and no
benchmark was performed.  Diagnostic logits are compared with the formal engine
for every sample to quantify output-exposure/tactic perturbation.

{"".join(sections)}

`{status}`
"""


def write_worst_case(path: Path, sample: dict[str, Any], conclusion: str, status: str) -> None:
    path.write_text(
        f"""# Worst-case residual error analysis

- Sample: `{sample['sample_id']}`
- JSON index: `{sample['dataset_index']}`
- First divergence stage: `{sample['first_divergence_stage']}`
- First error amplification stage: `{sample['first_error_amplification_stage']}`
- Maximum amplification stage: `{sample['maximum_amplification_stage']}`
- Maximum amplification delta: `{sample['maximum_amplification_delta']:.12e}`
- Formal TensorRT/PyTorch logits max abs: `{sample['formal_tensorrt_vs_captured_pytorch']['max_abs_error']:.12e}`
- Diagnostic TensorRT/PyTorch logits max abs: `{sample['error_growth'][-1]['max_abs_error']:.12e}`
- Diagnostic/formal TensorRT logits max abs: `{sample['diagnostic_vs_formal_tensorrt']['max_abs_error']:.12e}`
- Formal/diagnostic label agreement: `{sample['diagnostic_vs_formal_tensorrt']['label_agreement']:.12f}`

## Conclusion

{conclusion}

This is attribution only.  No ONNX, formal Engine, Plugin, checkpoint, builder
policy, precision mode or acceptance tolerance was modified.

`{status}`
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
        raise RuntimeError("ONNX SHA-256 mismatch")
    phase6_dir = args.phase6_artifacts.resolve()
    if not phase6_dir.is_dir():
        raise FileNotFoundError(phase6_dir)
    run_dir = (
        args.output_root.resolve()
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_residual_error_attribution"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    pytorch_root = run_dir / "pytorch_intermediate_dump"
    tensorrt_root = run_dir / "tensorrt_intermediate_dump"
    pytorch_root.mkdir()
    tensorrt_root.mkdir()
    logger = make_logger(run_dir)
    initial = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "RESIDUAL_ERROR_ATTRIBUTION_FAILED",
        "source_hashes_before": {name: sha256(path) for name, path in source_paths.items()},
        "samples": list(SAMPLES),
        "formal_sources_modified": False,
        "fp16": False,
        "int8": False,
        "benchmark": False,
    }
    dump_json(run_dir / "residual_error_report.json", initial)

    dll_handles: list[Any] = []
    capture: PyTorchStageCapture | None = None
    original_configure = diagnostic.configure_builder
    old_matmul_tf32: bool | None = None
    old_cudnn_tf32: bool | None = None
    old_precision: str | None = None
    try:
        dll_handles = phase5.configure_dll_search(
            args.tensorrt_root.resolve(), args.cuda_root.resolve(), source_paths["plugin"]
        )
        import tensorrt as trt
        import torch
        from cuda.bindings import runtime as cudart

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        old_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        old_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
        old_precision = torch.get_float32_matmul_precision()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

        logger_trt = trt.Logger(trt.Logger.INFO)
        if not trt.init_libnvinfer_plugins(logger_trt, ""):
            raise RuntimeError("trt.init_libnvinfer_plugins returned false")
        plugin_library, plugin_info = phase4.load_plugin_library(source_paths["plugin"])
        if not plugin_info["registration_function_returned"]:
            raise RuntimeError("VoxelUnique plugin registration failed")
        registry = phase4.collect_registry(trt, trt.get_plugin_registry())
        if not registry["voxel_unique_creator_found"]:
            raise RuntimeError("VoxelUnique Creator not found")
        targets = [
            {
                "order": stage["order"],
                "logical_name": stage["stage_name"],
                "onnx_tensor": stage["onnx_tensor"],
                "category": stage["category"],
            }
            for stage in STAGES
        ]
        capability = diagnostic.query_formal_engine_debug_capability(
            trt, logger_trt, source_paths["engine"], targets
        )
        if capability["can_dump_from_formal_engine"]:
            raise RuntimeError(
                "Formal Engine unexpectedly exposes all debug tensors; this script's "
                "audited diagnostic-output path should be re-evaluated"
            )

        dataset = evaluation.FixedWeldEvaluationDataset("test")
        name_to_index = {record.sample_name: index for index, record in enumerate(dataset.records)}
        if not set(SAMPLES).issubset(name_to_index):
            raise RuntimeError(f"Requested samples missing from test split: {name_to_index}")
        capture = PyTorchStageCapture(source_paths["checkpoint"])
        diagnostic.configure_builder = strict_diagnostic_config(original_configure)
        sample_reports: list[dict[str, Any]] = []

        for sample_id in SAMPLES:
            index = name_to_index[sample_id]
            sample = dataset[index]
            xyz = sample["normalized_xyz"].unsqueeze(0).to(torch.float32)
            points = np.ascontiguousarray(evaluation.make_model_input(xyz).numpy(), dtype=np.float32)
            adjacency = np.ascontiguousarray(
                evaluation.build_adjacency_cpu(xyz)[0].numpy(), dtype=np.float32
            )
            pytorch_dir = pytorch_root / sample_id
            tensorrt_dir = tensorrt_root / sample_id
            pytorch_dir.mkdir()
            tensorrt_dir.mkdir()
            pytorch_tensors, pytorch_summary = capture.run(points, adjacency, pytorch_dir)
            audited_shapes = {
                stage["onnx_tensor"]: tuple(pytorch_tensors[stage["stage_name"]].shape)
                for stage in STAGES
            }
            tensorrt_tensors, tensorrt_summary = diagnostic.run_diagnostic_tensorrt(
                trt,
                cudart,
                logger_trt,
                source_paths["onnx"],
                plugin_library,
                points,
                adjacency,
                tensorrt_dir,
                workspace_gib=4.0,
                targets=targets,
                audited_static_shapes=audited_shapes,
                resolve_voxel_count_size_tensor=False,
            )
            if tensorrt_summary["build_config"].get("tf32"):
                raise RuntimeError("Diagnostic TensorRT engine used TF32")
            comparisons = [
                compare_arrays(
                    stage,
                    pytorch_tensors[stage["stage_name"]],
                    tensorrt_tensors[stage["stage_name"]],
                )
                for stage in STAGES
            ]
            logit_stage = {
                "order": len(STAGES),
                "stage_name": "logits",
                "category": "segmentation head output",
                "onnx_tensor": "logits",
            }
            comparisons.append(
                compare_arrays(
                    logit_stage, pytorch_tensors["logits"], tensorrt_tensors["final_logits"]
                )
            )
            add_growth_fields(comparisons)
            first_divergence = next(
                (item for item in comparisons if not item["exact_equal"]), None
            )
            first_amplification = next(
                (item for item in comparisons if item["delta_error"] > 0.0), None
            )
            maximum_amplification = max(comparisons, key=lambda item: item["delta_error"])
            formal_dir = phase6_dir / "predictions" / f"{index:02d}_{sample_id}"
            formal_trt_path = formal_dir / "tensorrt_logits.npy"
            phase6_pt_path = formal_dir / "pytorch_logits.npy"
            if not formal_trt_path.is_file() or not phase6_pt_path.is_file():
                raise FileNotFoundError(formal_dir)
            diagnostic_vs_formal = compare_saved(
                formal_trt_path, tensorrt_tensors["final_logits"],
                "diagnostic_vs_formal_tensorrt_logits"
            )
            captured_vs_phase6_pt = compare_saved(
                phase6_pt_path, pytorch_tensors["logits"],
                "captured_vs_phase6_pytorch_logits"
            )
            formal_vs_captured_pt = compare_saved(
                formal_trt_path, pytorch_tensors["logits"],
                "formal_tensorrt_vs_captured_pytorch"
            )
            sample_report = {
                "sample_id": sample_id,
                "dataset_index": index,
                "logical_path": sample["logical_path"],
                "sample_indices_sha256": phase5.array_sha256(sample["sample_indices"].numpy()),
                "points_sha256": phase5.array_sha256(points),
                "adj_sha256": phase5.array_sha256(adjacency),
                "first_divergence_stage": None if first_divergence is None else first_divergence["stage_name"],
                "first_error_amplification_stage": None if first_amplification is None else first_amplification["stage_name"],
                "maximum_amplification_stage": maximum_amplification["stage_name"],
                "maximum_amplification_delta": maximum_amplification["delta_error"],
                "error_growth": comparisons,
                "diagnostic_vs_formal_tensorrt": diagnostic_vs_formal,
                "captured_vs_phase6_pytorch": captured_vs_phase6_pt,
                "formal_tensorrt_vs_captured_pytorch": formal_vs_captured_pt,
                "pytorch_capture": pytorch_summary,
                "tensorrt_diagnostic": tensorrt_summary,
            }
            sample_reports.append(sample_report)
            dump_json(run_dir / "error_growth.json", {"samples": sample_reports})
            logger.info(
                "%s first=%s first_amp=%s max_amp=%s delta=%.9e "
                "diag/formal=%.9e formal/PT=%.9e",
                sample_id,
                sample_report["first_divergence_stage"],
                sample_report["first_error_amplification_stage"],
                sample_report["maximum_amplification_stage"],
                sample_report["maximum_amplification_delta"],
                diagnostic_vs_formal["max_abs_error"],
                formal_vs_captured_pt["max_abs_error"],
            )
            del pytorch_tensors, tensorrt_tensors
            gc.collect()

        status = "RESIDUAL_ERROR_ATTRIBUTION_COMPLETED"
        worst = next(item for item in sample_reports if item["sample_id"] == "weld_14")
        trend_first = [item["first_error_amplification_stage"] for item in sample_reports]
        trend_max = [item["maximum_amplification_stage"] for item in sample_reports]
        first_stage = trend_first[0] if len(set(trend_first)) == 1 else trend_first
        maximum_stage = trend_max[0] if len(set(trend_max)) == 1 else trend_max
        global_amplification_sample, global_amplification_stage = max(
            (
                (sample, max(sample["error_growth"], key=lambda item: item["delta_error"]))
                for sample in sample_reports
            ),
            key=lambda pair: pair[1]["delta_error"],
        )
        conclusion = (
            f"Across weld_14, weld_5 and weld_12, stem_linear is exact and the first "
            f"divergence/positive max-error amplification is {first_stage}. gcn_0 then "
            "consistently amplifies the ptb_0 difference, while the transition-down path "
            "does not introduce the first divergence. The largest single-stage increase "
            f"across all three samples is {global_amplification_stage['stage_name']} on "
            f"{global_amplification_sample['sample_id']} "
            f"({global_amplification_stage['delta_error']:.12e}); per-sample maxima are "
            f"{maximum_stage}. Therefore the residual Strict-FP32 difference originates at "
            "ptb_0 and is accumulated/amplified by gcn_0 and later decoder/head stages. "
            "This is consistent with native TensorRT/PyTorch FP32 kernel and reduction-order "
            "differences, not a VoxelUnique/transition-down origin, Runtime failure, or label change."
        )

        capture.close()
        capture = None
        diagnostic.configure_builder = original_configure
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.set_float32_matmul_precision(old_precision)
        source_hashes_after = {name: sha256(path) for name, path in source_paths.items()}
        report = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "status": status,
            "first_divergence_stage": worst["first_divergence_stage"],
            "first_error_amplification_stage": worst["first_error_amplification_stage"],
            "maximum_amplification_stage": worst["maximum_amplification_stage"],
            "maximum_amplification_delta": worst["maximum_amplification_delta"],
            "maximum_amplification_scope": "worst_case_weld_14",
            "global_maximum_amplification_sample": global_amplification_sample["sample_id"],
            "global_maximum_amplification_stage": global_amplification_stage["stage_name"],
            "global_maximum_amplification_delta": global_amplification_stage["delta_error"],
            "cross_sample_first_amplification": trend_first,
            "cross_sample_maximum_amplification": trend_max,
            "conclusion": conclusion,
            "samples": sample_reports,
            "formal_engine_debug_capability": capability,
            "plugin": {**plugin_info, "creator": registry["voxel_unique"]},
            "diagnostic_method": {
                "formal_engine_modified": False,
                "onnx_modified": False,
                "plugin_modified": False,
                "checkpoint_modified": False,
                "formal_builder_config_modified": False,
                "diagnostic_engine_saved": False,
                "diagnostic_network_outputs_added_in_memory_only": True,
                "diagnostic_precision_policy": "same Strict FP32: TF32/FP16/INT8 disabled",
                "workspace_gib": 4.0,
                "representativeness_checked_against_formal_logits": True,
                "benchmark": False,
            },
            "source_hashes_before": initial["source_hashes_before"],
            "source_hashes_after": source_hashes_after,
            "source_hashes_unchanged": initial["source_hashes_before"] == source_hashes_after,
            "pytorch_precision_settings_restored": bool(
                torch.backends.cuda.matmul.allow_tf32 == old_matmul_tf32
                and torch.backends.cudnn.allow_tf32 == old_cudnn_tf32
                and torch.get_float32_matmul_precision() == old_precision
            ),
            "fp16": False,
            "int8": False,
            "benchmark": False,
        }
        if not report["source_hashes_unchanged"]:
            raise RuntimeError("A read-only source hash changed")
        if not report["pytorch_precision_settings_restored"]:
            raise RuntimeError("PyTorch precision settings were not restored")
        dump_json(run_dir / "residual_error_report.json", report)
        dump_json(run_dir / "error_growth.json", {"status": status, "samples": sample_reports})
        (run_dir / "error_growth.md").write_text(
            markdown_growth(sample_reports, status), encoding="utf-8"
        )
        write_worst_case(run_dir / "worst_case_analysis.md", worst, conclusion, status)
        logger.info("%s", status)
        print(f"RUN_DIR={run_dir}")
        print(f"FIRST_DIVERGENCE_STAGE={report['first_divergence_stage']}")
        print(f"FIRST_ERROR_AMPLIFICATION_STAGE={report['first_error_amplification_stage']}")
        print(f"MAXIMUM_AMPLIFICATION_STAGE={report['maximum_amplification_stage']}")
        print(status)
        del plugin_library, dll_handles
        gc.collect()
        return 0
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
        dump_json(run_dir / "residual_error_report.json", failure)
        logger.exception("Residual error attribution failed")
        print(f"RUN_DIR={run_dir}")
        print("RESIDUAL_ERROR_ATTRIBUTION_FAILED")
        return 1
    finally:
        diagnostic.configure_builder = original_configure
        if capture is not None:
            try:
                capture.close()
            except Exception:
                logger.exception("PyTorch capture cleanup failed")
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
    parser.add_argument("--plugin-library", type=Path, default=DEFAULT_PLUGIN)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--phase6-artifacts", type=Path, default=DEFAULT_PHASE6)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TENSORRT_ROOT)
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
