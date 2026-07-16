"""Locate the first TensorRT/PyTorch numerical divergence inside ptb_0.

The validated ONNX, formal FP32 engine, VoxelUnique plugin and checkpoint are
strictly read-only.  TensorRT values are obtained from a one-shot, in-memory
diagnostic engine whose parsed ITensors are exposed as extra outputs.  The
diagnostic engine is not persisted and no benchmark or numerical fix is run.
"""

from __future__ import annotations

import argparse
import gc
import json
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
import locate_gcn_res_tensorrt_first_divergence as phase5b  # noqa: E402
import run_gcn_res_tensorrt_fp32_inference as phase5  # noqa: E402


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
DEFAULT_FORMAL_TRT_LOGITS = phase5b.DEFAULT_FORMAL_TRT_LOGITS


PTB0_TARGETS = [
    {
        "order": 0,
        "logical_name": "stem_linear_features",
        "onnx_tensor": "/model/linear_1/Add_output_0",
        "category": "pre-ptb0 control / stem Linear",
    },
    {
        "order": 1,
        "logical_name": "distance_squared",
        "onnx_tensor": "/model/ptb_0/ReduceSum_output_0",
        "category": "pairwise squared distance / ReduceSum",
    },
    {
        "order": 2,
        "logical_name": "neighbor_distances",
        "onnx_tensor": "/model/ptb_0/Sqrt_output_0",
        "category": "pairwise Euclidean distance / Sqrt",
    },
    {
        "order": 3,
        "logical_name": "topk_values",
        "onnx_tensor": "/model/ptb_0/TopK_output_0",
        "category": "TopK values",
    },
    {
        "order": 4,
        "logical_name": "topk_indices",
        "onnx_tensor": "/model/ptb_0/TopK_output_1",
        "category": "TopK neighbor indices",
    },
    {
        "order": 5,
        "logical_name": "neighbors_xyz",
        "onnx_tensor": "/model/ptb_0/Reshape_1_output_0",
        "category": "GatherElements neighbor XYZ",
    },
    {
        "order": 6,
        "logical_name": "ptb0_linear1_features",
        "onnx_tensor": "/model/ptb_0/linear_1/Add_output_0",
        "category": "ptb0 Linear 1",
    },
    {
        "order": 7,
        "logical_name": "neighbors_features",
        "onnx_tensor": "/model/ptb_0/Reshape_2_output_0",
        "category": "GatherElements neighbor features",
    },
    {
        "order": 8,
        "logical_name": "relative_position_delta",
        "onnx_tensor": "/model/ptb_0/gva/delta_mult/Sub_output_0",
        "category": "relative XYZ",
    },
    {
        "order": 9,
        "logical_name": "delta_mult_encoding",
        "onnx_tensor": "/model/ptb_0/gva/delta_mult/linear_2/Add_output_0",
        "category": "relative position multiplicative encoding",
    },
    {
        "order": 10,
        "logical_name": "delta_bias_encoding",
        "onnx_tensor": "/model/ptb_0/gva/delta_bias/linear_2/Add_output_0",
        "category": "relative position bias encoding",
    },
    {
        "order": 11,
        "logical_name": "q_linear",
        "onnx_tensor": "/model/ptb_0/gva/q/Add_output_0",
        "category": "GVA Q Linear",
    },
    {
        "order": 12,
        "logical_name": "k_linear",
        "onnx_tensor": "/model/ptb_0/gva/k/Add_output_0",
        "category": "GVA K Linear",
    },
    {
        "order": 13,
        "logical_name": "v_linear",
        "onnx_tensor": "/model/ptb_0/gva/v/Add_output_0",
        "category": "GVA V Linear",
    },
    {
        "order": 14,
        "logical_name": "q_minus_k",
        "onnx_tensor": "/model/ptb_0/gva/Sub_output_0",
        "category": "GVA Q-K",
    },
    {
        "order": 15,
        "logical_name": "attention_vector",
        "onnx_tensor": "/model/ptb_0/gva/Add_output_0",
        "category": "delta_mult*(Q-K)+delta_bias",
    },
    {
        "order": 16,
        "logical_name": "attention_logits",
        "onnx_tensor": "/model/ptb_0/gva/conv_weights/Conv_output_0",
        "category": "grouped Conv attention logits",
    },
    {
        "order": 17,
        "logical_name": "attention_softmax",
        "onnx_tensor": "/model/ptb_0/gva/softmax_1d/Softmax_output_0",
        "category": "attention Softmax",
    },
    {
        "order": 18,
        "logical_name": "attention_aggregation",
        "onnx_tensor": "/model/ptb_0/gva/ReduceSum_output_0",
        "category": "weighted value ReduceSum",
    },
    {
        "order": 19,
        "logical_name": "attention_bn",
        "onnx_tensor": "/model/ptb_0/gva/bn/BatchNormalization_output_0",
        "category": "attention aggregation BatchNormalization",
    },
    {
        "order": 20,
        "logical_name": "attention_relu",
        "onnx_tensor": "/model/ptb_0/gva/Relu_output_0",
        "category": "attention aggregation ReLU",
    },
    {
        "order": 21,
        "logical_name": "gva_output",
        "onnx_tensor": "/model/ptb_0/gva/linear/Add_output_0",
        "category": "GVA output Linear",
    },
    {
        "order": 22,
        "logical_name": "residual_branch_before_add",
        "onnx_tensor": "/model/ptb_0/linear_2/Add_output_0",
        "category": "ptb0 Linear 2 / residual branch",
    },
    {
        "order": 23,
        "logical_name": "ptb0_output",
        "onnx_tensor": "/model/ptb_0/Add_output_0",
        "category": "ptb0 residual Add output",
    },
]


PTB0_SHAPES = {
    "/model/linear_1/Add_output_0": (1, 2048, 48),
    "/model/ptb_0/ReduceSum_output_0": (1, 2048, 2048),
    "/model/ptb_0/Sqrt_output_0": (1, 2048, 2048),
    "/model/ptb_0/TopK_output_0": (1, 2048, 16),
    "/model/ptb_0/TopK_output_1": (1, 2048, 16),
    "/model/ptb_0/Reshape_1_output_0": (1, 2048, 16, 3),
    "/model/ptb_0/linear_1/Add_output_0": (1, 2048, 48),
    "/model/ptb_0/Reshape_2_output_0": (1, 2048, 16, 48),
    "/model/ptb_0/gva/delta_mult/Sub_output_0": (1, 2048, 16, 3),
    "/model/ptb_0/gva/delta_mult/linear_2/Add_output_0": (1, 2048, 16, 48),
    "/model/ptb_0/gva/delta_bias/linear_2/Add_output_0": (1, 2048, 16, 48),
    "/model/ptb_0/gva/q/Add_output_0": (1, 2048, 1, 48),
    "/model/ptb_0/gva/k/Add_output_0": (1, 2048, 16, 48),
    "/model/ptb_0/gva/v/Add_output_0": (1, 2048, 16, 48),
    "/model/ptb_0/gva/Sub_output_0": (1, 2048, 16, 48),
    "/model/ptb_0/gva/Add_output_0": (1, 2048, 16, 48),
    "/model/ptb_0/gva/conv_weights/Conv_output_0": (1, 2, 2048, 16),
    "/model/ptb_0/gva/softmax_1d/Softmax_output_0": (1, 2, 2048, 16),
    "/model/ptb_0/gva/ReduceSum_output_0": (1, 2048, 48),
    "/model/ptb_0/gva/bn/BatchNormalization_output_0": (1, 48, 2048),
    "/model/ptb_0/gva/Relu_output_0": (1, 2048, 48),
    "/model/ptb_0/gva/linear/Add_output_0": (1, 2048, 48),
    "/model/ptb_0/linear_2/Add_output_0": (1, 2048, 48),
    "/model/ptb_0/Add_output_0": (1, 2048, 48),
}


def capture_ptb0_pytorch(
    points: np.ndarray,
    adjacency: np.ndarray,
    checkpoint_path: Path,
    dump_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    import deployment.gcn_res_onnx_model as model_module
    from deployment.gcn_res_onnx_model import GCNResStandardOps
    from deployment.gcn_res_onnx_wrapper import GCNResOnnxWrapper

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = GCNResStandardOps(SimpleNamespace(num_class=2), in_dim=4)
    strict_result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    wrapper = GCNResOnnxWrapper(model).to("cuda:0").eval()
    captured: dict[str, torch.Tensor] = {}
    hooks: list[Any] = []

    def save_tensor(name: str, tensor: torch.Tensor) -> None:
        if name not in captured:
            captured[name] = tensor.detach().clone()

    def output_hook(name: str, tuple_index: int | None = None) -> Any:
        def hook(module: Any, inputs: Any, output: Any) -> None:
            del module, inputs
            save_tensor(name, output if tuple_index is None else output[tuple_index])

        return hook

    def pre_hook(name: str, transform: Any | None = None) -> Any:
        def hook(module: Any, inputs: Any) -> None:
            del module
            tensor = inputs[0]
            if transform is not None:
                tensor = transform(tensor)
            save_tensor(name, tensor)

        return hook

    hooks.extend(
        [
            model.linear_1.register_forward_hook(output_hook("stem_linear_features")),
            model.ptb_0.linear_1.register_forward_hook(
                output_hook("ptb0_linear1_features")
            ),
            model.ptb_0.gva.delta_mult.linear_1.register_forward_pre_hook(
                pre_hook("relative_position_delta")
            ),
            model.ptb_0.gva.delta_mult.register_forward_hook(
                output_hook("delta_mult_encoding")
            ),
            model.ptb_0.gva.delta_bias.register_forward_hook(
                output_hook("delta_bias_encoding")
            ),
            model.ptb_0.gva.q.register_forward_hook(output_hook("q_linear")),
            model.ptb_0.gva.k.register_forward_hook(output_hook("k_linear")),
            model.ptb_0.gva.v.register_forward_hook(output_hook("v_linear")),
            model.ptb_0.gva.conv_weights.register_forward_pre_hook(
                pre_hook(
                    "attention_vector",
                    lambda tensor: tensor.permute(0, 2, 3, 1),
                )
            ),
            model.ptb_0.gva.conv_weights.register_forward_hook(
                output_hook("attention_logits")
            ),
            model.ptb_0.gva.softmax_1d.register_forward_hook(
                output_hook("attention_softmax")
            ),
            model.ptb_0.gva.bn.register_forward_pre_hook(
                pre_hook(
                    "attention_aggregation",
                    lambda tensor: tensor.permute(0, 2, 1),
                )
            ),
            model.ptb_0.gva.bn.register_forward_hook(output_hook("attention_bn")),
            model.ptb_0.gva.linear.register_forward_pre_hook(
                pre_hook("attention_relu")
            ),
            model.ptb_0.gva.linear.register_forward_hook(output_hook("gva_output")),
            model.ptb_0.linear_2.register_forward_hook(
                output_hook("residual_branch_before_add")
            ),
            model.ptb_0.register_forward_hook(output_hook("ptb0_output", 1)),
        ]
    )

    original_distance = model_module.standard_pairwise_euclidean_distance
    original_index_points = model_module.index_points
    original_topk = torch.topk

    def capturing_distance(xyz1: torch.Tensor, xyz2: torch.Tensor) -> torch.Tensor:
        coordinate_delta = xyz1.unsqueeze(2) - xyz2.unsqueeze(1)
        squared_distance = torch.sum(coordinate_delta * coordinate_delta, dim=-1)
        distances = torch.sqrt(squared_distance)
        if xyz1.shape[1] == 2048 and "neighbor_distances" not in captured:
            save_tensor("distance_squared", squared_distance)
            save_tensor("neighbor_distances", distances)
        return distances

    def capturing_topk(input_tensor: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        result = original_topk(input_tensor, *args, **kwargs)
        k = kwargs.get("k", args[0] if args else None)
        largest = kwargs.get("largest", True)
        if (
            input_tensor.ndim == 3
            and input_tensor.shape[1:] == (2048, 2048)
            and int(k) == 16
            and not largest
            and "topk_indices" not in captured
        ):
            save_tensor("topk_values", result.values)
            save_tensor("topk_indices", result.indices)
        return result

    def capturing_index_points(input_points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        output = original_index_points(input_points, indices)
        if input_points.shape[1] == 2048 and indices.shape[-1] == 16:
            if input_points.shape[-1] == 3 and "neighbors_xyz" not in captured:
                save_tensor("neighbors_xyz", output)
            elif input_points.shape[-1] == 48 and "neighbors_features" not in captured:
                save_tensor("neighbors_features", output)
        return output

    model_module.standard_pairwise_euclidean_distance = capturing_distance
    model_module.index_points = capturing_index_points
    torch.topk = capturing_topk
    points_tensor = torch.from_numpy(points).to("cuda:0")
    adjacency_tensor = torch.from_numpy(adjacency).to("cuda:0")
    try:
        with torch.inference_mode():
            final_logits = wrapper(points_tensor, adjacency_tensor)
        torch.cuda.synchronize(0)
    finally:
        model_module.standard_pairwise_euclidean_distance = original_distance
        model_module.index_points = original_index_points
        torch.topk = original_topk
        for hook in hooks:
            hook.remove()

    if "q_linear" in captured and "k_linear" in captured:
        save_tensor("q_minus_k", captured["q_linear"] - captured["k_linear"])

    expected_names = {target["logical_name"] for target in PTB0_TARGETS}
    missing = sorted(expected_names - captured.keys())
    if missing:
        raise RuntimeError(f"PyTorch ptb0 capture missing tensors: {missing}")

    tensors = {
        name: np.array(tensor.detach().cpu().numpy(), copy=True, order="C")
        for name, tensor in captured.items()
        if name in expected_names
    }
    for name, array in tensors.items():
        np.save(dump_dir / f"{name}.npy", array, allow_pickle=False)
    summary = {
        "strict_load_result": str(strict_result),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "captured_names": [target["logical_name"] for target in PTB0_TARGETS],
        "tensors": {name: phase5.array_stats(array) for name, array in tensors.items()},
        "forward_hooks_removed": True,
        "distance_monkeypatch_restored": (
            model_module.standard_pairwise_euclidean_distance is original_distance
        ),
        "index_points_monkeypatch_restored": model_module.index_points is original_index_points,
        "torch_topk_monkeypatch_restored": torch.topk is original_topk,
        "source_files_modified": False,
    }
    del final_logits, adjacency_tensor, points_tensor, wrapper, model, checkpoint, captured
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    return tensors, summary


def analyze_topk_semantics(
    points: np.ndarray,
    tensorrt_tensors: dict[str, np.ndarray],
    pytorch_tensors: dict[str, np.ndarray],
) -> dict[str, Any]:
    trt_indices = tensorrt_tensors["topk_indices"]
    torch_indices = pytorch_tensors["topk_indices"]
    mismatch = trt_indices != torch_indices
    batch_indices = np.arange(points.shape[0])[:, None, None]
    xyz = points[..., :3]
    trt_selected_xyz = xyz[batch_indices, trt_indices]
    torch_selected_xyz = xyz[batch_indices, torch_indices]
    selected_difference = np.abs(trt_selected_xyz - torch_selected_xyz)
    trt_values = tensorrt_tensors["topk_values"]
    torch_values = pytorch_tensors["topk_values"]
    value_difference = np.abs(trt_values - torch_values)
    mismatch_count = int(mismatch.sum())
    if mismatch_count:
        mismatch_xyz_equal = np.all(
            trt_selected_xyz[mismatch] == torch_selected_xyz[mismatch], axis=1
        )
        mismatch_value_difference = value_difference[mismatch]
    else:
        mismatch_xyz_equal = np.empty((0,), dtype=np.bool_)
        mismatch_value_difference = np.empty((0,), dtype=np.float32)
    return {
        "index_mismatch_count": mismatch_count,
        "total_neighbor_slots": int(mismatch.size),
        "index_agreement": float((~mismatch).mean()),
        "queries_with_any_index_mismatch": int(np.any(mismatch, axis=2).sum()),
        "query_count": int(mismatch.shape[0] * mismatch.shape[1]),
        "selected_xyz_exact_equal": bool(np.array_equal(trt_selected_xyz, torch_selected_xyz)),
        "selected_xyz_max_abs_error": float(selected_difference.max()),
        "mismatched_slots_with_equal_xyz": int(mismatch_xyz_equal.sum()),
        "mismatched_slots_total": mismatch_count,
        "mismatched_topk_value_max_abs_error": (
            float(mismatch_value_difference.max()) if mismatch_count else 0.0
        ),
        "mismatched_topk_value_mean_abs_error": (
            float(mismatch_value_difference.mean()) if mismatch_count else 0.0
        ),
        "unique_xyz_count": int(np.unique(xyz.reshape(-1, 3), axis=0).shape[0]),
        "total_points": int(xyz.shape[0] * xyz.shape[1]),
        "interpretation": (
            "TopK index IDs differ only among duplicate-coordinate/tied neighbors; "
            "the gathered neighbor XYZ tensor is bitwise identical."
        ),
    }


def write_report(
    path: Path,
    comparisons: list[dict[str, Any]],
    first_exact: dict[str, Any] | None,
    first_tolerance: dict[str, Any] | None,
    first_float_tolerance: dict[str, Any] | None,
    first_discrete: dict[str, Any] | None,
    topk_semantics: dict[str, Any],
    diagnostic_vs_formal: dict[str, Any],
    status: str,
) -> None:
    rows = "\n".join(
        f"| {item['order']} | {item['logical_name']} | `{item['tensorrt_shape']}` | "
        f"`{item['tensorrt_dtype']}` | {item.get('max_abs_error')} | "
        f"{item.get('mean_abs_error')} | {item.get('cosine_similarity')} | "
        f"{item['exact_equal']} | {item['allclose_rtol_1e-5_atol_1e-6']} | "
        f"`{item.get('first_mismatch_index')}` |"
        for item in comparisons
    )
    mapping = {item["logical_name"]: item for item in comparisons}
    topk_indices = mapping["topk_indices"]
    text = f"""# ptb_0 TensorRT / PyTorch internal parity

## Read-only boundary

- Formal ONNX, formal FP32 Engine, VoxelUnique Plugin and checkpoint were hash-checked before and after the run and were not modified.
- TensorRT values came from a one-shot in-memory diagnostic FP32 Engine with selected parsed ITensors exposed as outputs. The diagnostic Engine was not saved.
- No FP16, INT8, benchmark, graph rewrite, Plugin change or numerical fix was performed.

## Diagnostic representativeness

- Diagnostic vs formal TensorRT final logits max abs: `{diagnostic_vs_formal.get('max_abs_error')}`
- Diagnostic vs formal label agreement: `{diagnostic_vs_formal.get('label_agreement')}`
- Exposing intermediate outputs can alter TensorRT fusion/tactic selection, so this run localizes the first boundary in the diagnostic graph; it does not claim bitwise identity with the formal Engine.

## Ordered internal comparison

| Order | Tensor | Shape | Dtype | Max abs | Mean abs | Cosine | Exact | Allclose | First mismatch index |
|---:|---|---|---|---:|---:|---:|---|---|---|
{rows}

## First internal divergence

- First exact divergence: `{None if first_exact is None else first_exact['logical_name']}`
- First engineering-tolerance divergence: `{None if first_tolerance is None else first_tolerance['logical_name']}`
- First discrete-index divergence: `{None if first_discrete is None else first_discrete['logical_name']}`
- First FP32 tensor outside rtol=1e-5, atol=1e-6: `{None if first_float_tolerance is None else first_float_tolerance['logical_name']}`
- First mismatch index: `{None if first_exact is None else first_exact.get('first_mismatch_index')}`
- TensorRT value: `{None if first_exact is None else first_exact.get('first_mismatch_tensorrt_value')}`
- PyTorch value: `{None if first_exact is None else first_exact.get('first_mismatch_pytorch_value')}`
- TopK neighbor indices exact: `{topk_indices.get('exact_equal')}`

## TopK tie semantics

- Index mismatches: `{topk_semantics['index_mismatch_count']}/{topk_semantics['total_neighbor_slots']}` across `{topk_semantics['queries_with_any_index_mismatch']}` queries.
- Unique XYZ points: `{topk_semantics['unique_xyz_count']}/{topk_semantics['total_points']}`.
- Every mismatched index selects the same XYZ: `{topk_semantics['selected_xyz_exact_equal']}`; gathered XYZ max abs `{topk_semantics['selected_xyz_max_abs_error']}`.
- Mismatched-slot TopK value max abs: `{topk_semantics['mismatched_topk_value_max_abs_error']}`.
- Interpretation: {topk_semantics['interpretation']}

The first non-bitwise arithmetic difference is the squared-distance `ReduceSum`. It remains within tolerance. The TopK index IDs differ because tied duplicate points are ordered differently, but the gathered XYZ is identical. The first FP32 feature tensor outside tolerance is `ptb0_linear1_features`, whose input (`stem_linear_features`) is bitwise identical.

This is a localization result only. No fix was attempted.

`{status}`
"""
    path.write_text(text, encoding="utf-8")


def main(args: argparse.Namespace) -> int:
    engine_path = args.engine.resolve()
    onnx_path = args.onnx.resolve()
    input_path = args.input.resolve()
    checkpoint_path = args.checkpoint.resolve()
    plugin_path = args.plugin_library.resolve()
    formal_logits_path = args.formal_tensorrt_logits.resolve()
    for path in (
        engine_path,
        onnx_path,
        input_path,
        checkpoint_path,
        plugin_path,
        formal_logits_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if phase4.sha256(engine_path) != phase5.EXPECTED_ENGINE_SHA256:
        raise RuntimeError("Formal Engine hash mismatch")
    if phase4.sha256(onnx_path) != phase5.EXPECTED_ONNX_SHA256:
        raise RuntimeError("Formal derived ONNX hash mismatch")

    source_hashes_before = {
        "engine": phase4.sha256(engine_path),
        "onnx": phase4.sha256(onnx_path),
        "checkpoint": phase4.sha256(checkpoint_path),
        "plugin": phase4.sha256(plugin_path),
    }
    run_dir = (
        args.output_root.resolve()
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_ptb0_parity"
    )
    pytorch_dump_dir = run_dir / "pytorch_ptb0_dump"
    tensorrt_dump_dir = run_dir / "tensorrt_ptb0_dump"
    pytorch_dump_dir.mkdir(parents=True, exist_ok=False)
    tensorrt_dump_dir.mkdir(parents=True, exist_ok=False)
    failure_payload: dict[str, Any] = {
        "status": "PTB0_INTERNAL_PARITY_FAILED",
        "run_dir": str(run_dir),
        "source_hashes_before": source_hashes_before,
        "fp16": False,
        "int8": False,
        "benchmark": False,
        "fix_attempted": False,
    }
    phase5b.dump_json(run_dir / "ptb0_tensor_compare.json", failure_payload)

    try:
        dll_handles = phase5.configure_dll_search(
            args.tensorrt_root.resolve(), args.cuda_root.resolve(), plugin_path
        )
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        logger = trt.Logger(trt.Logger.INFO)
        if not trt.init_libnvinfer_plugins(logger, ""):
            raise RuntimeError("trt.init_libnvinfer_plugins returned false")
        plugin_library, plugin_info = phase4.load_plugin_library(plugin_path)
        if not plugin_info["registration_function_returned"]:
            raise RuntimeError("VoxelUnique Plugin registration failed")

        points, adjacency, _, input_metadata = phase5.load_inputs(input_path)
        capability = phase5b.query_formal_engine_debug_capability(
            trt, logger, engine_path, targets=PTB0_TARGETS
        )
        phase5b.dump_json(run_dir / "formal_engine_debug_capability.json", capability)

        pytorch_tensors, pytorch_summary = capture_ptb0_pytorch(
            points, adjacency, checkpoint_path, pytorch_dump_dir
        )
        phase5b.dump_json(run_dir / "pytorch_ptb0_summary.json", pytorch_summary)

        tensorrt_tensors, diagnostic_summary = phase5b.run_diagnostic_tensorrt(
            trt,
            cudart,
            logger,
            onnx_path,
            plugin_library,
            points,
            adjacency,
            tensorrt_dump_dir,
            args.workspace_gib,
            targets=PTB0_TARGETS,
            audited_static_shapes=PTB0_SHAPES,
            resolve_voxel_count_size_tensor=False,
        )
        phase5b.dump_json(run_dir / "tensorrt_ptb0_summary.json", diagnostic_summary)

        comparisons: list[dict[str, Any]] = []
        for target in PTB0_TARGETS:
            name = target["logical_name"]
            result = phase5b.compare_arrays(
                name, tensorrt_tensors[name], pytorch_tensors[name]
            )
            result.update(
                {
                    "order": target["order"],
                    "category": target["category"],
                    "onnx_tensor": target["onnx_tensor"],
                }
            )
            comparisons.append(result)

        # Order 0 is the already-proven stem control.  The internal search begins at order 1.
        internal = [item for item in comparisons if item["order"] >= 1]
        first_exact = next((item for item in internal if not item["exact_equal"]), None)
        first_tolerance = next(
            (
                item
                for item in internal
                if not item["allclose_rtol_1e-5_atol_1e-6"]
            ),
            None,
        )
        first_float_tolerance = next(
            (
                item
                for item in internal
                if item["tensorrt_dtype"] == "float32"
                and not item["allclose_rtol_1e-5_atol_1e-6"]
            ),
            None,
        )
        first_discrete = next(
            (
                item
                for item in internal
                if item["tensorrt_dtype"].startswith("int")
                and not item["exact_equal"]
            ),
            None,
        )
        topk_semantics = analyze_topk_semantics(
            points, tensorrt_tensors, pytorch_tensors
        )
        diagnostic_vs_formal = phase5b.compare_final_logits(
            tensorrt_tensors["final_logits"], formal_logits_path
        )
        formal_logits = np.load(formal_logits_path, allow_pickle=False)
        diagnostic_vs_formal["label_agreement"] = float(
            (
                np.argmax(tensorrt_tensors["final_logits"], axis=-1)
                == np.argmax(formal_logits, axis=-1)
            ).mean()
        )
        status = (
            "FIRST_PTB0_INTERNAL_DIVERGENCE_FOUND"
            if first_exact is not None
            else "PTB0_INTERNAL_DIVERGENCE_NOT_FOUND"
        )
        source_hashes_after = {
            "engine": phase4.sha256(engine_path),
            "onnx": phase4.sha256(onnx_path),
            "checkpoint": phase4.sha256(checkpoint_path),
            "plugin": phase4.sha256(plugin_path),
        }
        payload = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "status": status,
            "run_dir": str(run_dir),
            "input_metadata": input_metadata,
            "formal_engine_debug_capability": capability,
            "pytorch_capture": pytorch_summary,
            "tensorrt_diagnostic": diagnostic_summary,
            "comparisons": comparisons,
            "first_exact_internal_divergence": first_exact,
            "first_tolerance_internal_divergence": first_tolerance,
            "first_float_tolerance_internal_divergence": first_float_tolerance,
            "first_discrete_internal_divergence": first_discrete,
            "topk_semantics": topk_semantics,
            "diagnostic_vs_formal_tensorrt_final_logits": diagnostic_vs_formal,
            "source_hashes_before": source_hashes_before,
            "source_hashes_after": source_hashes_after,
            "formal_sources_modified": source_hashes_before != source_hashes_after,
            "diagnostic_engine_saved": False,
            "fp16": False,
            "int8": False,
            "benchmark": False,
            "fix_attempted": False,
        }
        if payload["formal_sources_modified"]:
            raise RuntimeError("A formal source hash changed during read-only diagnosis")
        phase5b.dump_json(run_dir / "ptb0_tensor_compare.json", payload)
        write_report(
            run_dir / "first_ptb0_divergence.md",
            comparisons,
            first_exact,
            first_tolerance,
            first_float_tolerance,
            first_discrete,
            topk_semantics,
            diagnostic_vs_formal,
            status,
        )
        print(f"RUN_DIR={run_dir}")
        print(
            "FIRST_EXACT_PTB0_DIVERGENCE="
            f"{None if first_exact is None else first_exact['logical_name']}"
        )
        print(
            "FIRST_TOLERANCE_PTB0_DIVERGENCE="
            f"{None if first_tolerance is None else first_tolerance['logical_name']}"
        )
        print(status)
        del dll_handles
        return 0 if first_exact is not None else 2
    except Exception:
        failure_payload.update(
            {
                "traceback": traceback.format_exc(),
                "source_hashes_after": {
                    "engine": phase4.sha256(engine_path),
                    "onnx": phase4.sha256(onnx_path),
                    "checkpoint": phase4.sha256(checkpoint_path),
                    "plugin": phase4.sha256(plugin_path),
                },
            }
        )
        failure_payload["formal_sources_modified"] = (
            failure_payload["source_hashes_before"]
            != failure_payload["source_hashes_after"]
        )
        phase5b.dump_json(run_dir / "ptb0_tensor_compare.json", failure_payload)
        print(f"RUN_DIR={run_dir}")
        print(failure_payload["traceback"], file=sys.stderr)
        print("PTB0_INTERNAL_PARITY_FAILED")
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=phase5.DEFAULT_ENGINE)
    parser.add_argument("--onnx", type=Path, default=phase5.DEFAULT_ONNX)
    parser.add_argument("--input", type=Path, default=phase5.DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=phase5.DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--plugin-library", type=Path, default=phase5.DEFAULT_PLUGIN_LIBRARY
    )
    parser.add_argument(
        "--formal-tensorrt-logits", type=Path, default=DEFAULT_FORMAL_TRT_LOGITS
    )
    parser.add_argument("--tensorrt-root", type=Path, default=phase5.DEFAULT_TENSORRT_ROOT)
    parser.add_argument("--cuda-root", type=Path, default=phase5.DEFAULT_CUDA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
