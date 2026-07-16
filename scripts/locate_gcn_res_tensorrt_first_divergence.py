"""Locate the first TensorRT/PyTorch divergence in the first TDB/attention prefix.

The validated ONNX, plugin DLL and formal TensorRT plan are read-only.  Because
the formal plan contains no debug tensors, this script parses the same ONNX and
builds an in-memory-only FP32 diagnostic engine with selected tensors exposed as
additional diagnostic outputs in the parsed Network Definition.  The ONNX and
formal plan are not rewritten, the diagnostic engine is never saved, and no
performance benchmark is performed.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import sys
import time
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
import run_gcn_res_tensorrt_fp32_inference as phase5  # noqa: E402


DEFAULT_ENGINE = phase5.DEFAULT_ENGINE
DEFAULT_ONNX = phase5.DEFAULT_ONNX
DEFAULT_INPUT = phase5.DEFAULT_INPUT
DEFAULT_CHECKPOINT = phase5.DEFAULT_CHECKPOINT
DEFAULT_PLUGIN_LIBRARY = phase5.DEFAULT_PLUGIN_LIBRARY
DEFAULT_TENSORRT_ROOT = phase5.DEFAULT_TENSORRT_ROOT
DEFAULT_CUDA_ROOT = phase5.DEFAULT_CUDA_ROOT
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
DEFAULT_FORMAL_TRT_LOGITS = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260716_212305_673121_fp32_inference"
    / "tensorrt_logits.npy"
)


TARGETS = [
    {
        "order": 0,
        "logical_name": "initial_linear_features",
        "onnx_tensor": "/model/linear_1/Add_output_0",
        "category": "encoder stem Linear",
    },
    {
        "order": 1,
        "logical_name": "ptb0_features",
        "onnx_tensor": "/model/ptb_0/Add_output_0",
        "category": "PointTransformer block 0 output",
    },
    {
        "order": 2,
        "logical_name": "gcn0_features",
        "onnx_tensor": "/model/gcn_0/linear/Add_output_0",
        "category": "GCN block 0 output / TDB1 input",
    },
    {
        "order": 3,
        "logical_name": "tdb1_linear_features",
        "onnx_tensor": "/model/tdb_1/linear/Add_output_0",
        "category": "TransitionDown 1 Linear",
    },
    {
        "order": 4,
        "logical_name": "tdb1_norm_features",
        "onnx_tensor": "/model/tdb_1/norm/BatchNormalization_output_0",
        "category": "TransitionDown 1 BatchNormalization",
    },
    {
        "order": 5,
        "logical_name": "tdb1_relu_features",
        "onnx_tensor": "/model/tdb_1/relu/Relu_output_0",
        "category": "TransitionDown 1 ReLU / voxel feature source",
    },
    {
        "order": 6,
        "logical_name": "scatter_source_points",
        "onnx_tensor": "/model/tdb_1/Reshape_5_output_0",
        "category": "ScatterElements point source",
    },
    {
        "order": 7,
        "logical_name": "scatter_source_features",
        "onnx_tensor": "/model/tdb_1/Reshape_6_output_0",
        "category": "ScatterElements feature source",
    },
    {
        "order": 8,
        "logical_name": "voxel_count",
        "onnx_tensor": "/model/tdb_1/VoxelUnique_voxel_count_output_0",
        "category": "VoxelUnique",
    },
    {
        "order": 9,
        "logical_name": "unique_values",
        "onnx_tensor": "/model/tdb_1/Unique_output_0",
        "category": "VoxelUnique",
    },
    {
        "order": 10,
        "logical_name": "inverse_indices",
        "onnx_tensor": "/model/tdb_1/Unique_output_2",
        "category": "VoxelUnique",
    },
    {
        "order": 11,
        "logical_name": "unique_batch_ids",
        "onnx_tensor": "/model/tdb_1/ScatterElements_output_0",
        "category": "ScatterElements amin",
    },
    {
        "order": 12,
        "logical_name": "voxel_point_counts",
        "onnx_tensor": "/model/tdb_1/ScatterElements_1_output_0",
        "category": "ScatterElements add",
    },
    {
        "order": 13,
        "logical_name": "summed_points",
        "onnx_tensor": "/model/tdb_1/ScatterElements_2_output_0",
        "category": "ScatterElements add",
    },
    {
        "order": 14,
        "logical_name": "pooled_features_all",
        "onnx_tensor": "/model/tdb_1/ScatterElements_3_output_0",
        "category": "ScatterElements max",
    },
    {
        "order": 15,
        "logical_name": "voxel_count_per_batch",
        "onnx_tensor": "/model/tdb_1/ScatterElements_4_output_0",
        "category": "ScatterElements add",
    },
    {
        "order": 16,
        "logical_name": "pooled_points",
        "onnx_tensor": "/model/tdb_1/Reshape_10_output_0",
        "category": "TransitionDown pooled XYZ / Unsqueeze",
    },
    {
        "order": 17,
        "logical_name": "pooled_features",
        "onnx_tensor": "/model/tdb_1/Reshape_11_output_0",
        "category": "TransitionDown pooled features / Unsqueeze",
    },
    {
        "order": 18,
        "logical_name": "ptb1_features",
        "onnx_tensor": "/model/ptb_1/Add_output_0",
        "category": "first PointTransformer block output",
    },
]


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_to_numpy(tensor: Any) -> np.ndarray:
    import torch

    with torch._C._DisableTorchDispatch():
        return np.ascontiguousarray(tensor.detach().cpu().numpy())


def run_pytorch_capture(
    points: np.ndarray,
    adjacency: np.ndarray,
    checkpoint_path: Path,
    dump_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    import deployment.gcn_res_onnx_model as model_module
    from deployment.gcn_res_onnx_model import GCNResStandardOps
    from deployment.gcn_res_onnx_wrapper import GCNResOnnxWrapper
    from deployment.onnx_voxel_pool import standard_voxel_pool_with_metadata

    class ScatterCaptureMode(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.current_stage: int | None = None
            self.records: list[dict[str, Any]] = []

        def __torch_dispatch__(
            self,
            func: Any,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            kwargs = kwargs or {}
            output = func(*args, **kwargs)
            name = str(func)
            if self.current_stage == 1 and (
                "scatter_add" in name or "scatter_reduce" in name
            ):
                reduce_name = kwargs.get("reduce")
                if reduce_name is None and "scatter_reduce" in name and len(args) > 4:
                    reduce_name = args[4]
                with torch._C._DisableTorchDispatch():
                    cloned = output.detach().clone()
                self.records.append(
                    {
                        "operator": name,
                        "reduce": str(reduce_name) if reduce_name is not None else None,
                        "shape": list(output.shape),
                        "dtype": str(output.dtype),
                        "tensor": cloned,
                    }
                )
            return output

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    model = GCNResStandardOps(SimpleNamespace(num_class=2), in_dim=4)
    strict_result = model.load_state_dict(state_dict, strict=True)
    wrapper = GCNResOnnxWrapper(model).to("cuda:0").eval()
    capture_mode = ScatterCaptureMode()
    stage_results: dict[int, Any] = {}
    stage_counter = 0
    ptb1_output: dict[str, Any] = {}
    module_outputs: dict[str, Any] = {}
    original_pool = model_module.standard_voxel_pool

    def capture_module_output(name: str, tuple_index: int | None = None) -> Any:
        def hook(module: Any, inputs: Any, output: Any) -> None:
            del module, inputs
            selected = output if tuple_index is None else output[tuple_index]
            with torch._C._DisableTorchDispatch():
                module_outputs[name] = selected.detach().clone()

        return hook

    def capturing_pool(
        xyz: torch.Tensor, features: torch.Tensor, voxel_size: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal stage_counter
        stage_counter += 1
        capture_mode.current_stage = stage_counter
        try:
            result = standard_voxel_pool_with_metadata(xyz, features, voxel_size)
        finally:
            capture_mode.current_stage = None
        if stage_counter == 1:
            stage_results[1] = result
            with torch._C._DisableTorchDispatch():
                module_outputs["scatter_source_points"] = xyz.reshape(-1, 3).detach().clone()
                module_outputs["scatter_source_features"] = (
                    features.reshape(-1, features.shape[-1]).detach().clone()
                )
        return result.pooled_points, result.pooled_features

    def ptb1_hook(module: Any, inputs: Any, output: Any) -> None:
        del module, inputs
        with torch._C._DisableTorchDispatch():
            ptb1_output["xyz"] = output[0].detach().clone()
            ptb1_output["features"] = output[1].detach().clone()

    hooks = [
        model.linear_1.register_forward_hook(
            capture_module_output("initial_linear_features")
        ),
        model.ptb_0.register_forward_hook(capture_module_output("ptb0_features", 1)),
        model.gcn_0.register_forward_hook(capture_module_output("gcn0_features")),
        model.tdb_1.linear.register_forward_hook(
            capture_module_output("tdb1_linear_features")
        ),
        model.tdb_1.norm.register_forward_hook(
            capture_module_output("tdb1_norm_features")
        ),
        model.tdb_1.relu.register_forward_hook(
            capture_module_output("tdb1_relu_features")
        ),
        model.ptb_1.register_forward_hook(ptb1_hook),
    ]
    model_module.standard_voxel_pool = capturing_pool
    points_tensor = torch.from_numpy(points).to("cuda:0")
    adjacency_tensor = torch.from_numpy(adjacency).to("cuda:0")
    try:
        with torch.inference_mode(), capture_mode:
            final_logits_tensor = wrapper(points_tensor, adjacency_tensor)
        torch.cuda.synchronize(0)
    finally:
        model_module.standard_voxel_pool = original_pool
        for hook in hooks:
            hook.remove()

    required_module_outputs = {
        "initial_linear_features",
        "ptb0_features",
        "gcn0_features",
        "tdb1_linear_features",
        "tdb1_norm_features",
        "tdb1_relu_features",
        "scatter_source_points",
        "scatter_source_features",
    }
    if (
        1 not in stage_results
        or "features" not in ptb1_output
        or not required_module_outputs.issubset(module_outputs)
    ):
        raise RuntimeError("PyTorch stage-1 hooks did not capture required tensors")
    result = stage_results[1]
    scatter_add = [item for item in capture_mode.records if "scatter_add" in item["operator"]]
    scatter_amin = [
        item
        for item in capture_mode.records
        if "scatter_reduce" in item["operator"] and "amin" in str(item["reduce"])
    ]
    scatter_amax = [
        item
        for item in capture_mode.records
        if "scatter_reduce" in item["operator"] and "amax" in str(item["reduce"])
    ]
    if len(scatter_add) != 3 or len(scatter_amin) != 1 or len(scatter_amax) != 1:
        raise RuntimeError(
            "Unexpected stage-1 scatter capture sequence: "
            f"add={len(scatter_add)}, amin={len(scatter_amin)}, amax={len(scatter_amax)}"
        )

    total_voxels = int(result.unique_local_keys.shape[0])
    tensors = {
        **{
            name: tensor_to_numpy(module_outputs[name]).astype(np.float32, copy=False)
            for name in sorted(required_module_outputs)
        },
        "voxel_count": np.asarray(total_voxels, dtype=np.int32),
        # B=1, so collision-free global keys and local keys are identical.
        "unique_values": tensor_to_numpy(result.unique_local_keys).astype(np.int64, copy=False),
        "inverse_indices": tensor_to_numpy(result.point_to_voxel.reshape(-1)).astype(np.int64, copy=False),
        "unique_batch_ids": tensor_to_numpy(scatter_amin[0]["tensor"]).astype(np.int64, copy=False),
        "voxel_point_counts": tensor_to_numpy(scatter_add[0]["tensor"]).astype(np.int64, copy=False),
        "summed_points": tensor_to_numpy(scatter_add[1]["tensor"]).astype(np.float32, copy=False),
        "pooled_features_all": tensor_to_numpy(scatter_amax[0]["tensor"]).astype(np.float32, copy=False),
        "voxel_count_per_batch": tensor_to_numpy(scatter_add[2]["tensor"]).astype(np.int64, copy=False),
        "pooled_points": tensor_to_numpy(result.pooled_points).astype(np.float32, copy=False),
        "pooled_features": tensor_to_numpy(result.pooled_features).astype(np.float32, copy=False),
        "ptb1_features": tensor_to_numpy(ptb1_output["features"]).astype(np.float32, copy=False),
        "final_logits": tensor_to_numpy(final_logits_tensor).astype(np.float32, copy=False),
    }
    for name, array in tensors.items():
        np.save(dump_dir / f"{name}.npy", array, allow_pickle=False)
    summary = {
        "strict_load_result": str(strict_result),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "stage_calls": stage_counter,
        "batch_is_one_global_keys_equal_local_keys": True,
        "captured_scatter_sequence": [
            {key: value for key, value in item.items() if key != "tensor"}
            for item in capture_mode.records
        ],
        "tensors": {name: phase5.array_stats(array) for name, array in tensors.items()},
        "source_files_modified": False,
        "forward_hooks_removed": True,
        "runtime_monkeypatch_restored": model_module.standard_voxel_pool is original_pool,
    }
    del final_logits_tensor, adjacency_tensor, points_tensor, wrapper, model, checkpoint, state_dict
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    return tensors, summary


def capsule_pointer(address: Any) -> int:
    try:
        return int(address)
    except (TypeError, ValueError):
        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        get_pointer.restype = ctypes.c_void_p
        pointer = get_pointer(address, None)
        if not pointer:
            raise RuntimeError("PyCapsule_GetPointer returned null")
        return int(pointer)


def make_debug_listener_class(trt: Any, cudart: Any) -> type:
    dtype_map = {
        trt.float32: np.dtype(np.float32),
        trt.int32: np.dtype(np.int32),
        trt.int64: np.dtype(np.int64),
        trt.bool: np.dtype(np.bool_),
    }

    class DebugListener(trt.IDebugListener):
        def __init__(self) -> None:
            trt.IDebugListener.__init__(self)
            self.tensors: dict[str, np.ndarray] = {}
            self.events: list[dict[str, Any]] = []
            self.errors: list[str] = []

        def process_debug_tensor(
            self,
            addr: Any,
            location: Any,
            dtype: Any,
            shape: Any,
            name: str,
            stream: Any,
        ) -> bool:
            try:
                dimensions = tuple(int(item) for item in shape)
                numpy_dtype = dtype_map.get(dtype)
                if numpy_dtype is None:
                    raise TypeError(f"Unsupported debug tensor dtype: {dtype}")
                array = np.empty(dimensions, dtype=numpy_dtype)
                pointer = capsule_pointer(addr)
                location_name = phase4.enum_name(location)
                if array.nbytes:
                    if location_name == "DEVICE":
                        phase5.cuda_call(
                            cudart,
                            f"debug D2H {name}",
                            cudart.cudaMemcpyAsync(
                                int(array.ctypes.data),
                                pointer,
                                int(array.nbytes),
                                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                                stream,
                            ),
                        )
                        phase5.cuda_call(
                            cudart,
                            f"debug stream synchronize {name}",
                            cudart.cudaStreamSynchronize(stream),
                        )
                    elif location_name == "HOST":
                        ctypes.memmove(int(array.ctypes.data), pointer, int(array.nbytes))
                    else:
                        raise RuntimeError(f"Unknown debug tensor location: {location}")
                self.tensors[name] = np.ascontiguousarray(array)
                self.events.append(
                    {
                        "name": name,
                        "location": location_name,
                        "dtype": str(numpy_dtype),
                        "shape": list(dimensions),
                        "nbytes": int(array.nbytes),
                        "stream_handle_hex": hex(int(stream)),
                    }
                )
                return True
            except Exception:
                self.errors.append(traceback.format_exc())
                return False

    return DebugListener


def parsed_tensor_map(network: Any) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    for layer_index in range(network.num_layers):
        layer = network.get_layer(layer_index)
        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)
            if tensor is not None:
                tensors[tensor.name] = tensor
    return tensors


def configure_builder(trt: Any, config: Any, workspace_gib: float) -> dict[str, Any]:
    workspace_bytes = int(workspace_gib * 1024**3)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    for name in ("FP16", "INT8", "SPARSE_WEIGHTS", "REFIT", "VERSION_COMPATIBLE", "WEIGHT_STREAMING"):
        if hasattr(trt.BuilderFlag, name):
            config.clear_flag(getattr(trt.BuilderFlag, name))
    return {
        "precision": "FP32",
        "workspace_bytes": workspace_bytes,
        "workspace_gib": workspace_gib,
        "fp16": bool(config.get_flag(trt.BuilderFlag.FP16)) if hasattr(trt.BuilderFlag, "FP16") else False,
        "int8": bool(config.get_flag(trt.BuilderFlag.INT8)) if hasattr(trt.BuilderFlag, "INT8") else False,
        "engine_persisted": False,
        "benchmark": False,
    }


def query_formal_engine_debug_capability(
    trt: Any,
    logger: Any,
    engine_path: Path,
    targets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected_targets = TARGETS if targets is None else targets
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("Formal engine deserialization failed during debug capability audit")
    records = [
        {
            "logical_name": target["logical_name"],
            "onnx_tensor": target["onnx_tensor"],
            "is_debug_tensor": bool(engine.is_debug_tensor(target["onnx_tensor"])),
        }
        for target in selected_targets
    ]
    result = {
        "formal_engine": str(engine_path),
        "formal_engine_sha256": phase4.sha256(engine_path),
        "num_io_tensors": int(engine.num_io_tensors),
        "io_tensors": [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)],
        "targets": records,
        "debug_target_count": sum(item["is_debug_tensor"] for item in records),
        "can_dump_from_formal_engine": all(item["is_debug_tensor"] for item in records),
        "inspector_limitation": "Engine Inspector exposes structure/metadata, not intermediate values.",
    }
    del engine, runtime
    gc.collect()
    return result


def run_diagnostic_tensorrt(
    trt: Any,
    cudart: Any,
    logger: Any,
    onnx_path: Path,
    plugin_library: Any,
    points: np.ndarray,
    adjacency: np.ndarray,
    dump_dir: Path,
    workspace_gib: float,
    targets: list[dict[str, Any]] | None = None,
    audited_static_shapes: dict[str, tuple[int, ...]] | None = None,
    resolve_voxel_count_size_tensor: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected_targets = TARGETS if targets is None else targets
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    if builder is None or network is None or parser is None or config is None:
        raise RuntimeError("Diagnostic Builder/Network/Parser/Config creation failed")
    parser_success = bool(parser.parse_from_file(str(onnx_path)))
    parser_errors = phase4.parser_errors(parser)
    if not parser_success or parser_errors:
        raise RuntimeError(f"Diagnostic parser failed: {parser_errors[:1]}")
    tensors = parsed_tensor_map(network)
    mark_results = []
    for target in selected_targets:
        tensor = tensors.get(target["onnx_tensor"])
        if tensor is None:
            raise KeyError(f"Parsed network tensor missing: {target['onnx_tensor']}")
        return_value = network.mark_output(tensor)
        output_names_after = {
            network.get_output(index).name for index in range(network.num_outputs)
        }
        marked = target["onnx_tensor"] in output_names_after
        mark_results.append(
            {
                **target,
                "network_shape": [int(item) for item in tensor.shape],
                "network_dtype": phase4.enum_name(tensor.dtype),
                "mark_output_python_return": repr(return_value),
                "output_present_after_call": marked,
                "network_output_mode": True,
            }
        )
    if not all(item["output_present_after_call"] for item in mark_results):
        raise RuntimeError("One or more network.mark_output calls failed")
    build_config = configure_builder(trt, config, workspace_gib)
    build_count_before = int(plugin_library.getVoxelUniqueBuildCreationCount())
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_elapsed = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError("Diagnostic build_serialized_network returned None")
    engine_bytes = bytes(serialized)
    build_count_after = int(plugin_library.getVoxelUniqueBuildCreationCount())
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("Diagnostic engine deserialization failed")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Diagnostic execution context creation failed")

    io_records = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        io_records.append(
            {
                "index": index,
                "name": name,
                "mode": phase4.enum_name(engine.get_tensor_mode(name)),
                "dtype": phase4.enum_name(engine.get_tensor_dtype(name)),
                "shape": phase4.dims_list(engine.get_tensor_shape(name)),
                "location": phase4.enum_name(engine.get_tensor_location(name)),
            }
        )
    expected_core = phase4.EXPECTED_IO
    actual_core = {
        item["name"]: {
            "mode": item["mode"],
            "dtype": item["dtype"],
            "shape": item["shape"],
        }
        for item in io_records
        if item["name"] in expected_core
    }
    if actual_core != expected_core:
        raise RuntimeError(f"Diagnostic core I/O mismatch: {actual_core}")
    diagnostic_output_names = {target["onnx_tensor"] for target in selected_targets}
    actual_output_names = {
        item["name"] for item in io_records if item["mode"] == "OUTPUT"
    }
    if not diagnostic_output_names.issubset(actual_output_names):
        raise RuntimeError(
            f"Diagnostic outputs missing: {sorted(diagnostic_output_names - actual_output_names)}"
        )

    dtype_map = {
        "FLOAT": np.dtype(np.float32),
        "INT32": np.dtype(np.int32),
        "INT64": np.dtype(np.int64),
        "BOOL": np.dtype(np.bool_),
    }
    host_inputs = {"points": points, "adj": adjacency}
    pointers: dict[str, int] = {}
    host_pointers: dict[str, int] = {}
    output_capacity_bytes: dict[str, int] = {}
    stream = None
    try:
        phase5.cuda_call(cudart, "cudaSetDevice(0)", cudart.cudaSetDevice(0))
        stream = phase5.cuda_call(cudart, "cudaStreamCreate", cudart.cudaStreamCreate())[0]
        for name, array in host_inputs.items():
            pointers[name] = int(
                phase5.cuda_call(cudart, f"cudaMalloc({name})", cudart.cudaMalloc(array.nbytes))[0]
            )
        for item in io_records:
            if item["mode"] != "OUTPUT":
                continue
            name = item["name"]
            maximum_bytes = int(context.get_max_output_size(name))
            if maximum_bytes <= 0:
                raise RuntimeError(f"get_max_output_size returned {maximum_bytes} for {name}")
            output_capacity_bytes[name] = maximum_bytes
            if item["location"] == "DEVICE":
                pointers[name] = int(
                    phase5.cuda_call(
                        cudart, f"cudaMalloc(output {name})", cudart.cudaMalloc(maximum_bytes)
                    )[0]
                )
            elif item["location"] == "HOST":
                host_pointers[name] = int(
                    phase5.cuda_call(
                        cudart,
                        f"cudaHostAlloc(output {name})",
                        cudart.cudaHostAlloc(
                            maximum_bytes, cudart.cudaHostAllocDefault
                        ),
                    )[0]
                )
            else:
                raise RuntimeError(f"Unsupported output location {item['location']} for {name}")
        for name, array in host_inputs.items():
            phase5.cuda_call(
                cudart,
                f"cudaMemcpyAsync H2D {name}",
                cudart.cudaMemcpyAsync(
                    pointers[name],
                    int(array.ctypes.data),
                    int(array.nbytes),
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                ),
            )
        for item in io_records:
            name = item["name"]
            address = host_pointers.get(name, pointers.get(name))
            if address is None or not context.set_tensor_address(name, address):
                raise RuntimeError(f"Diagnostic set_tensor_address failed for {name}")
        inference_started = time.perf_counter()
        if not context.execute_async_v3(stream_handle=int(stream)):
            raise RuntimeError("Diagnostic execute_async_v3 returned false")
        phase5.cuda_call(cudart, "cudaStreamSynchronize", cudart.cudaStreamSynchronize(stream))

        def copy_output(item: dict[str, Any], actual_shape: tuple[int, ...]) -> np.ndarray:
            name = item["name"]
            numpy_dtype = dtype_map.get(item["dtype"])
            if numpy_dtype is None:
                raise TypeError(f"Unsupported diagnostic output dtype {item['dtype']} for {name}")
            array = np.empty(actual_shape, dtype=numpy_dtype)
            if array.nbytes > output_capacity_bytes[name]:
                raise RuntimeError(
                    f"Actual output exceeds capacity for {name}: {array.nbytes} > "
                    f"{output_capacity_bytes[name]}"
                )
            if array.nbytes:
                if item["location"] == "DEVICE":
                    phase5.cuda_call(
                        cudart,
                        f"cudaMemcpyAsync D2H {name}",
                        cudart.cudaMemcpyAsync(
                            int(array.ctypes.data),
                            pointers[name],
                            int(array.nbytes),
                            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                            stream,
                        ),
                    )
                else:
                    ctypes.memmove(
                        int(array.ctypes.data), host_pointers[name], int(array.nbytes)
                    )
            # ndarray.copy preserves rank-0 scalar shape; np.ascontiguousarray
            # promotes a scalar to [1] and would create a false shape mismatch.
            return array.copy(order="C")

        output_by_name = {
            item["name"]: item for item in io_records if item["mode"] == "OUTPUT"
        }
        count_name = "/model/tdb_1/VoxelUnique_voxel_count_output_0"
        voxel_count_array = None
        runtime_m = None
        if resolve_voxel_count_size_tensor:
            if count_name not in output_by_name:
                raise RuntimeError("Voxel-count shape resolution requested without count output")
            voxel_count_array = copy_output(output_by_name[count_name], ())
            phase5.cuda_call(
                cudart,
                "cudaStreamSynchronize after voxel_count copy",
                cudart.cudaStreamSynchronize(stream),
            )
            runtime_m = int(voxel_count_array.item())
            if runtime_m <= 0 or runtime_m > 2048:
                raise RuntimeError(f"Invalid runtime voxel_count M={runtime_m}")
        audited_runtime_shapes = {
            "/model/linear_1/Add_output_0": (1, 2048, 48),
            "/model/ptb_0/Add_output_0": (1, 2048, 48),
            "/model/gcn_0/linear/Add_output_0": (1, 2048, 48),
            "/model/tdb_1/linear/Add_output_0": (1, 2048, 96),
            "/model/tdb_1/norm/BatchNormalization_output_0": (1, 96, 2048),
            "/model/tdb_1/relu/Relu_output_0": (1, 2048, 96),
            "/model/tdb_1/Reshape_5_output_0": (2048, 3),
            "/model/tdb_1/Reshape_6_output_0": (2048, 96),
            count_name: (),
            "/model/tdb_1/Unique_output_0": (runtime_m,),
            "/model/tdb_1/Unique_output_2": (2048,),
            "/model/tdb_1/ScatterElements_output_0": (runtime_m,),
            "/model/tdb_1/ScatterElements_1_output_0": (runtime_m,),
            "/model/tdb_1/ScatterElements_2_output_0": (runtime_m, 3),
            "/model/tdb_1/ScatterElements_3_output_0": (runtime_m, 96),
            "/model/tdb_1/ScatterElements_4_output_0": (1,),
            "/model/tdb_1/Reshape_10_output_0": (1, runtime_m, 3),
            "/model/tdb_1/Reshape_11_output_0": (1, runtime_m, 96),
            "/model/ptb_1/Add_output_0": (1, runtime_m, 96),
            "logits": (1, 2048, 2),
        }
        if audited_static_shapes:
            audited_runtime_shapes.update(audited_static_shapes)
        output_arrays: dict[str, np.ndarray] = {}
        for item in io_records:
            if item["mode"] != "OUTPUT":
                continue
            name = item["name"]
            reported_shape = tuple(int(value) for value in context.get_tensor_shape(name))
            if name not in audited_runtime_shapes:
                raise KeyError(f"No audited runtime shape for diagnostic output {name}")
            audited_shape = audited_runtime_shapes[name]
            if any(value is None for value in audited_shape):
                raise RuntimeError(
                    f"Unresolved audited shape for {name}: {audited_shape}; "
                    "supply audited_static_shapes or enable DDS size-tensor resolution"
                )
            if not any(value < 0 for value in reported_shape) and reported_shape != audited_shape:
                raise RuntimeError(
                    f"Context/audited output shape disagreement for {name}: "
                    f"{reported_shape} != {audited_shape}"
                )
            output_arrays[name] = (
                voxel_count_array
                if resolve_voxel_count_size_tensor and name == count_name
                else copy_output(item, audited_shape)
            )
            item["context_shape_after_enqueue"] = list(reported_shape)
            item["resolved_shape_from_voxel_count"] = list(audited_shape)
            item["shape_resolution"] = (
                "context"
                if not any(value < 0 for value in reported_shape)
                else (
                    "voxel_count_size_tensor"
                    if resolve_voxel_count_size_tensor
                    else "audited_static_contract"
                )
            )
        phase5.cuda_call(
            cudart, "cudaStreamSynchronize after output copies", cudart.cudaStreamSynchronize(stream)
        )
        inference_elapsed = time.perf_counter() - inference_started
    finally:
        for name, pointer in reversed(list(pointers.items())):
            phase5.cuda_call(cudart, f"cudaFree({name})", cudart.cudaFree(pointer))
        for name, pointer in reversed(list(host_pointers.items())):
            phase5.cuda_call(cudart, f"cudaFreeHost({name})", cudart.cudaFreeHost(pointer))
        if stream is not None:
            phase5.cuda_call(cudart, "cudaStreamDestroy", cudart.cudaStreamDestroy(stream))

    logical_tensors = {
        target["logical_name"]: output_arrays[target["onnx_tensor"]]
        for target in selected_targets
    }
    logical_tensors["final_logits"] = output_arrays["logits"]
    for name, array in logical_tensors.items():
        np.save(dump_dir / f"{name}.npy", array, allow_pickle=False)
    summary = {
        "parser_success": parser_success,
        "parser_errors": parser_errors,
        "marked_debug_tensors": mark_results,
        "build_config": build_config,
        "build_elapsed_seconds_not_a_benchmark": build_elapsed,
        "diagnostic_engine_size_bytes": len(engine_bytes),
        "diagnostic_engine_sha256_in_memory_only": bytes_sha256(engine_bytes),
        "diagnostic_engine_saved": False,
        "voxel_unique_build_instance_delta": build_count_after - build_count_before,
        "engine_io": io_records,
        "diagnostic_output_capacity_bytes": output_capacity_bytes,
        "diagnostic_output_strategy": "Parsed network ITensors marked as additional outputs",
        "runtime_voxel_count_m": runtime_m,
        "dynamic_shape_resolution": (
            "TensorRT 11.1 left DDS output context shapes unresolved (-1) without an "
            "IOutputAllocator; actual M was read from the plugin voxel_count size tensor "
            "and applied using the audited tensor contracts."
            if resolve_voxel_count_size_tensor
            else "All selected diagnostic tensors have audited fixed runtime shapes."
        ),
        "single_diagnostic_inference_elapsed_seconds_not_a_benchmark": inference_elapsed,
        "inference_count": 1,
        "fp16": False,
        "int8": False,
        "benchmark": False,
        "tensors": {name: phase5.array_stats(array) for name, array in logical_tensors.items()},
    }
    return logical_tensors, summary


def compare_arrays(name: str, trt_array: np.ndarray, torch_array: np.ndarray) -> dict[str, Any]:
    shape_match = trt_array.shape == torch_array.shape
    dtype_match = trt_array.dtype == torch_array.dtype
    result: dict[str, Any] = {
        "logical_name": name,
        "tensorrt_shape": list(trt_array.shape),
        "pytorch_shape": list(torch_array.shape),
        "shape_match": shape_match,
        "tensorrt_dtype": str(trt_array.dtype),
        "pytorch_dtype": str(torch_array.dtype),
        "dtype_match": dtype_match,
        "tensorrt_finite": bool(np.isfinite(trt_array).all()),
        "pytorch_finite": bool(np.isfinite(torch_array).all()),
    }
    if not shape_match:
        result.update(
            {
                "exact_equal": False,
                "allclose_rtol_1e-5_atol_1e-6": False,
                "first_mismatch_index": None,
                "failure": "shape_mismatch",
            }
        )
        return result
    trt64 = trt_array.astype(np.float64)
    torch64 = torch_array.astype(np.float64)
    difference = trt64 - torch64
    absolute = np.abs(difference)
    exact = bool(np.array_equal(trt_array, torch_array))
    close = bool(np.allclose(trt_array, torch_array, rtol=1.0e-5, atol=1.0e-6))
    mismatch = np.argwhere(trt_array != torch_array)
    first_index = mismatch[0].tolist() if mismatch.size else None
    first_trt = trt_array[tuple(first_index)].item() if first_index is not None else None
    first_torch = torch_array[tuple(first_index)].item() if first_index is not None else None
    trt_flat = trt64.reshape(-1)
    torch_flat = torch64.reshape(-1)
    denominator = float(np.linalg.norm(trt_flat) * np.linalg.norm(torch_flat))
    cosine = float(np.dot(trt_flat, torch_flat) / denominator) if denominator else (1.0 if exact else None)
    result.update(
        {
            "exact_equal": exact,
            "allclose_rtol_1e-5_atol_1e-6": close,
            "max_abs_error": float(absolute.max()) if absolute.size else 0.0,
            "mean_abs_error": float(absolute.mean()) if absolute.size else 0.0,
            "cosine_similarity": cosine,
            "first_mismatch_index": first_index,
            "first_mismatch_tensorrt_value": first_trt,
            "first_mismatch_pytorch_value": first_torch,
        }
    )
    return result


def compare_final_logits(diagnostic: np.ndarray, formal_path: Path) -> dict[str, Any]:
    formal = np.load(formal_path, allow_pickle=False)
    return compare_arrays("diagnostic_vs_formal_tensorrt_final_logits", diagnostic, formal)


def write_first_divergence_report(
    run_dir: Path,
    capability: dict[str, Any],
    comparisons: list[dict[str, Any]],
    first_exact: dict[str, Any] | None,
    first_tolerance: dict[str, Any] | None,
    diagnostic_vs_formal: dict[str, Any],
    status: str,
) -> None:
    comparison_by_name = {item["logical_name"]: item for item in comparisons}
    initial_linear = comparison_by_name.get("initial_linear_features", {})
    ptb0 = comparison_by_name.get("ptb0_features", {})
    feature_source = comparison_by_name.get("scatter_source_features", {})
    voxel_count = comparison_by_name.get("voxel_count", {})
    unique_values = comparison_by_name.get("unique_values", {})
    inverse_indices = comparison_by_name.get("inverse_indices", {})
    voxel_counts = comparison_by_name.get("voxel_point_counts", {})
    summed_points = comparison_by_name.get("summed_points", {})
    pooled_features = comparison_by_name.get("pooled_features_all", {})
    rows = "\n".join(
        f"| {item['logical_name']} | `{item['tensorrt_shape']}` | `{item['tensorrt_dtype']}` | "
        f"{item.get('max_abs_error')} | {item.get('mean_abs_error')} | "
        f"{item.get('cosine_similarity')} | {item['exact_equal']} | "
        f"{item['allclose_rtol_1e-5_atol_1e-6']} | `{item.get('first_mismatch_index')}` |"
        for item in comparisons
    )
    report = f"""# TensorRT / PyTorch first intermediate divergence

## Read-only boundary

- Formal ONNX, Plugin DLL and validated FP32 plan were hash-checked and not modified.
- The formal plan exposes only points/adj/logits and contains {capability['debug_target_count']} marked debug tensors.
- Engine Inspector supplies structure, not values. Direct `mark_debug()` was tested first but TensorRT 11.1 rejected the DDS consumer fusion during diagnostic build. The successful fallback exposes selected parsed-network tensors as additional outputs in a one-shot, in-memory-only FP32 diagnostic engine.
- The diagnostic engine was not saved. No FP16, INT8, benchmark, kernel optimization or graph rewrite was performed.

## Diagnostic engine representativeness

- Diagnostic vs formal TensorRT final logits max abs: `{diagnostic_vs_formal.get('max_abs_error')}`
- Diagnostic vs formal TensorRT final logits exact equal: `{diagnostic_vs_formal.get('exact_equal')}`
- Diagnostic vs formal TensorRT final logits allclose (rtol=1e-5, atol=1e-6): `{diagnostic_vs_formal.get('allclose_rtol_1e-5_atol_1e-6')}`
- Label agreement: `{diagnostic_vs_formal.get('label_agreement')}`

## Ordered comparison

| Tensor | Shape | Dtype | Max abs | Mean abs | Cosine | Exact | Allclose | First mismatch index |
|---|---|---|---:|---:|---:|---|---|---|
{rows}

## First divergence

- First exact divergence: `{None if first_exact is None else first_exact['logical_name']}`
- First engineering-tolerance divergence: `{None if first_tolerance is None else first_tolerance['logical_name']}`
- First mismatch index: `{None if first_exact is None else first_exact.get('first_mismatch_index')}`
- TensorRT value: `{None if first_exact is None else first_exact.get('first_mismatch_tensorrt_value')}`
- PyTorch value: `{None if first_exact is None else first_exact.get('first_mismatch_pytorch_value')}`

## Localization conclusion

- Encoder stem `linear_1` exact: `{initial_linear.get('exact_equal')}`; max abs `{initial_linear.get('max_abs_error')}`.
- First PointTransformer block (`ptb_0`) allclose: `{ptb0.get('allclose_rtol_1e-5_atol_1e-6')}`; max abs `{ptb0.get('max_abs_error')}`. This is the first captured numerical divergence and is upstream of `tdb_1`, VoxelUnique, ScatterElements and pooling.
- VoxelUnique `voxel_count`, `unique_values`, and `inverse_indices` exact: `{voxel_count.get('exact_equal')}`, `{unique_values.get('exact_equal')}`, `{inverse_indices.get('exact_equal')}`. The custom Unique plugin is not the first divergence for this sample.
- Scatter integer voxel counts exact: `{voxel_counts.get('exact_equal')}`. Point-coordinate sum stays within tolerance (`max_abs={summed_points.get('max_abs_error')}`).
- The feature source entering max Scatter already differs (`max_abs={feature_source.get('max_abs_error')}`), so the pooled-feature difference (`max_abs={pooled_features.get('max_abs_error')}`) is propagated from upstream and is not evidence that Scatter max itself first introduced the error.

This is a localization result only. No fix was attempted.

`{status}`
"""
    (run_dir / "first_divergence.md").write_text(report, encoding="utf-8")


def main(args: argparse.Namespace) -> int:
    engine_path = args.engine.resolve()
    onnx_path = args.onnx.resolve()
    input_path = args.input.resolve()
    checkpoint_path = args.checkpoint.resolve()
    plugin_path = args.plugin_library.resolve()
    formal_logits = args.formal_tensorrt_logits.resolve()
    for path in (engine_path, onnx_path, input_path, checkpoint_path, plugin_path, formal_logits):
        if not path.is_file():
            raise FileNotFoundError(path)
    if phase4.sha256(engine_path) != phase5.EXPECTED_ENGINE_SHA256:
        raise RuntimeError("Formal engine hash mismatch")
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
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_intermediate_parity"
    )
    pytorch_dump_dir = run_dir / "pytorch_tensor_dump"
    tensorrt_dump_dir = run_dir / "tensorrt_tensor_dump"
    pytorch_dump_dir.mkdir(parents=True, exist_ok=False)
    tensorrt_dump_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "status": "TENSORRT_INTERMEDIATE_PARITY_FAILED",
        "run_dir": str(run_dir),
        "source_hashes_before": source_hashes_before,
        "formal_sources_modified": False,
        "fp16": False,
        "int8": False,
        "benchmark": False,
        "fix_attempted": False,
    }
    dump_json(run_dir / "tensor_compare_report.json", summary)
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
            raise RuntimeError("VoxelUnique plugin registration failed")
        points, adjacency, _, input_metadata = phase5.load_inputs(input_path)
        capability = query_formal_engine_debug_capability(trt, logger, engine_path)
        dump_json(run_dir / "formal_engine_debug_capability.json", capability)

        pytorch_tensors, pytorch_summary = run_pytorch_capture(
            points, adjacency, checkpoint_path, pytorch_dump_dir
        )
        dump_json(run_dir / "pytorch_capture_summary.json", pytorch_summary)
        tensorrt_tensors, diagnostic_summary = run_diagnostic_tensorrt(
            trt,
            cudart,
            logger,
            onnx_path,
            plugin_library,
            points,
            adjacency,
            tensorrt_dump_dir,
            args.workspace_gib,
        )
        dump_json(run_dir / "diagnostic_engine_summary.json", diagnostic_summary)

        comparisons = []
        for target in TARGETS:
            logical_name = target["logical_name"]
            result = compare_arrays(
                logical_name, tensorrt_tensors[logical_name], pytorch_tensors[logical_name]
            )
            result["order"] = target["order"]
            result["category"] = target["category"]
            result["onnx_tensor"] = target["onnx_tensor"]
            comparisons.append(result)
        first_exact = next((item for item in comparisons if not item["exact_equal"]), None)
        first_tolerance = next(
            (item for item in comparisons if not item["allclose_rtol_1e-5_atol_1e-6"]),
            None,
        )
        diagnostic_vs_formal = compare_final_logits(
            tensorrt_tensors["final_logits"], formal_logits
        )
        diagnostic_labels = np.argmax(tensorrt_tensors["final_logits"], axis=-1)
        formal_labels = np.argmax(np.load(formal_logits, allow_pickle=False), axis=-1)
        diagnostic_vs_formal["label_agreement"] = float(
            (diagnostic_labels == formal_labels).mean()
        )
        status = (
            "FIRST_TENSORRT_PYTORCH_DIVERGENCE_FOUND"
            if first_exact is not None
            else "FIRST_DIVERGENCE_NOT_IN_CAPTURED_PREFIX"
        )
        source_hashes_after = {
            "engine": phase4.sha256(engine_path),
            "onnx": phase4.sha256(onnx_path),
            "checkpoint": phase4.sha256(checkpoint_path),
            "plugin": phase4.sha256(plugin_path),
        }
        summary = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "status": status,
            "run_dir": str(run_dir),
            "input_metadata": input_metadata,
            "input_points": phase5.array_stats(points),
            "input_adj": phase5.array_stats(adjacency),
            "formal_engine_debug_capability": capability,
            "debug_dump_strategy": (
                "In-memory-only diagnostic FP32 engine built from unchanged ONNX; "
                "selected parsed ITensors exposed as additional outputs; no plan persisted."
            ),
            "pytorch_capture": pytorch_summary,
            "tensorrt_diagnostic": diagnostic_summary,
            "comparisons": comparisons,
            "first_exact_divergence": first_exact,
            "first_tolerance_divergence": first_tolerance,
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
        if summary["formal_sources_modified"]:
            raise RuntimeError("Formal ONNX/Engine/checkpoint/plugin hash changed")
        dump_json(run_dir / "tensor_compare_report.json", summary)
        write_first_divergence_report(
            run_dir,
            capability,
            comparisons,
            first_exact,
            first_tolerance,
            diagnostic_vs_formal,
            status,
        )
        print(f"RUN_DIR={run_dir}")
        print(f"FIRST_EXACT_DIVERGENCE={None if first_exact is None else first_exact['logical_name']}")
        print(
            "FIRST_TOLERANCE_DIVERGENCE="
            f"{None if first_tolerance is None else first_tolerance['logical_name']}"
        )
        print(status)
        return 0 if first_exact is not None else 2
    except Exception as error:
        summary.update(
            {
                "status": "TENSORRT_INTERMEDIATE_PARITY_FAILED",
                "first_error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "source_hashes_after": {
                    "engine": phase4.sha256(engine_path),
                    "onnx": phase4.sha256(onnx_path),
                    "checkpoint": phase4.sha256(checkpoint_path),
                    "plugin": phase4.sha256(plugin_path),
                },
            }
        )
        summary["formal_sources_modified"] = (
            summary["source_hashes_before"] != summary["source_hashes_after"]
        )
        dump_json(run_dir / "tensor_compare_report.json", summary)
        (run_dir / "first_divergence.md").write_text(
            "# TensorRT intermediate parity failed\n\n"
            f"```text\n{summary['traceback']}\n```\n\n"
            "`TENSORRT_INTERMEDIATE_PARITY_FAILED`\n",
            encoding="utf-8",
        )
        print(traceback.format_exc(), file=sys.stderr)
        print(f"RUN_DIR={run_dir}")
        print("TENSORRT_INTERMEDIATE_PARITY_FAILED")
        return 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--plugin-library", type=Path, default=DEFAULT_PLUGIN_LIBRARY)
    parser.add_argument("--formal-tensorrt-logits", type=Path, default=DEFAULT_FORMAL_TRT_LOGITS)
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TENSORRT_ROOT)
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
