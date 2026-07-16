"""Run one fixed-shape TensorRT FP32 inference and compare with PyTorch CUDA.

The script uses cuda-python for explicit device allocation, copies and stream
management.  It performs exactly one TensorRT inference; it is not a benchmark.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
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
import compare_tensorrt_pytorch_logits as parity  # noqa: E402


DEFAULT_ENGINE = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260716_190125_699274_dds_reshape_rewrite"
    / "gcn_res_dds_reshape_fp32_b1_n2048.plan"
)
DEFAULT_ONNX = DEFAULT_ENGINE.parent / "dds_reshape_rewritten.onnx"
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_onnx"
    / "20260715_onnx_after_cdist_fp32_opset18"
    / "export_input.npz"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "testParameters" / "GCN_res" / "best_model.pth"
DEFAULT_PLUGIN_LIBRARY = phase4.DEFAULT_PLUGIN_LIBRARY
DEFAULT_TENSORRT_ROOT = phase4.DEFAULT_TENSORRT_ROOT
DEFAULT_CUDA_ROOT = phase4.DEFAULT_CUDA_ROOT
EXPECTED_ENGINE_SHA256 = "7a856d1aa50628360d4acd5ee384fcd5042a8087a3112a361ee3c47db9e7326b"
EXPECTED_ONNX_SHA256 = "f71a585ded3f348a193e37395a124ab8a425ff72fc09ae92114d41872c460b98"
EXPECTED_IO = phase4.EXPECTED_IO


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def array_stats(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
        "c_contiguous": bool(array.flags.c_contiguous),
        "finite": bool(np.isfinite(array).all()),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "array_sha256": array_sha256(array),
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def configure_dll_search(
    tensorrt_root: Path, cuda_root: Path, plugin_path: Path
) -> list[Any]:
    handles = []
    for directory in (tensorrt_root / "bin", cuda_root / "bin", plugin_path.parent):
        if hasattr(os, "add_dll_directory"):
            handles.append(os.add_dll_directory(str(directory)))
    os.environ["TENSORRT_ROOT"] = str(tensorrt_root)
    os.environ["CUDA_PATH"] = str(cuda_root)
    os.environ["PATH"] = os.pathsep.join(
        [str(tensorrt_root / "bin"), str(cuda_root / "bin"), os.environ.get("PATH", "")]
    )
    return handles


def cuda_error_details(cudart: Any, error: Any) -> dict[str, Any]:
    try:
        _, name = cudart.cudaGetErrorName(error)
        _, description = cudart.cudaGetErrorString(error)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        if isinstance(description, bytes):
            description = description.decode("utf-8", errors="replace")
    except Exception:
        name = str(error)
        description = "cudaGetErrorName/cudaGetErrorString unavailable"
    return {"code": int(error), "name": str(name), "description": str(description)}


def cuda_call(cudart: Any, operation: str, result: tuple[Any, ...]) -> tuple[Any, ...]:
    error = result[0]
    if int(error) != int(cudart.cudaError_t.cudaSuccess):
        detail = cuda_error_details(cudart, error)
        raise RuntimeError(f"CUDA {operation} failed: {detail}")
    return result[1:]


def collect_environment(
    trt: Any,
    cudart: Any,
    engine: Path,
    onnx_path: Path,
    input_path: Path,
    checkpoint: Path,
    plugin: Path,
    tensorrt_root: Path,
    cuda_root: Path,
) -> dict[str, Any]:
    import torch

    runtime_version = cuda_call(cudart, "cudaRuntimeGetVersion", cudart.cudaRuntimeGetVersion())[0]
    driver_version = cuda_call(cudart, "cudaDriverGetVersion", cudart.cudaDriverGetVersion())[0]
    properties = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
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
        "cuda_toolkit_root": str(cuda_root),
        "tensorrt_root": str(tensorrt_root),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "gpu": None
        if properties is None
        else {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(properties.total_memory),
        },
        "onnxruntime_version": package_version("onnxruntime-gpu"),
        "torch_geometric_version": package_version("torch-geometric"),
        "torch_cluster_version": package_version("torch-cluster"),
        "torch_scatter_version": package_version("torch-scatter"),
        "torch_sparse_version": package_version("torch-sparse"),
        "pip_check": phase4.command_output([sys.executable, "-m", "pip", "check"]),
        "engine": {"path": str(engine), "size_bytes": engine.stat().st_size, "sha256": phase4.sha256(engine)},
        "onnx": {"path": str(onnx_path), "size_bytes": onnx_path.stat().st_size, "sha256": phase4.sha256(onnx_path)},
        "input": {"path": str(input_path), "size_bytes": input_path.stat().st_size, "sha256": phase4.sha256(input_path)},
        "checkpoint": {"path": str(checkpoint), "size_bytes": checkpoint.stat().st_size, "sha256": phase4.sha256(checkpoint)},
        "plugin": {"path": str(plugin), "size_bytes": plugin.stat().st_size, "sha256": phase4.sha256(plugin)},
    }


def make_error_recorder_class(trt: Any) -> type:
    class ErrorRecorder(trt.IErrorRecorder):
        def __init__(self) -> None:
            trt.IErrorRecorder.__init__(self)
            self.errors: list[dict[str, Any]] = []

        @property
        def num_errors(self) -> int:
            return len(self.errors)

        def get_error_code(self, index: int) -> Any:
            return self.errors[index]["raw_code"]

        def get_error_desc(self, index: int) -> str:
            return self.errors[index]["description"]

        def has_overflowed(self) -> bool:
            return False

        def clear(self) -> None:
            self.errors.clear()

        def report_error(self, code: Any, description: str) -> bool:
            self.errors.append(
                {
                    "raw_code": code,
                    "code": int(code),
                    "code_name": phase4.enum_name(code),
                    "description": str(description),
                }
            )
            return True

        def serializable(self) -> list[dict[str, Any]]:
            return [
                {key: value for key, value in item.items() if key != "raw_code"}
                for item in self.errors
            ]

    return ErrorRecorder


def load_inputs(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        if "points" not in archive or "adj" not in archive:
            raise KeyError("export_input.npz must contain points and adj")
        points = np.ascontiguousarray(archive["points"], dtype=np.float32)
        adjacency = np.ascontiguousarray(archive["adj"], dtype=np.float32)
        labels = (
            np.ascontiguousarray(archive["ground_truth_labels"], dtype=np.int64)
            if "ground_truth_labels" in archive
            else None
        )
        metadata = {
            "keys": archive.files,
            "source_npz": str(archive["source_npz"]) if "source_npz" in archive else None,
            "sample_indices_sha256": (
                array_sha256(archive["sample_indices"])
                if "sample_indices" in archive
                else None
            ),
        }
    if points.shape != (1, 2048, 4) or adjacency.shape != (1, 2048, 2048):
        raise ValueError(f"Fixed input shape mismatch: points={points.shape}, adj={adjacency.shape}")
    if not np.isfinite(points).all() or not np.isfinite(adjacency).all():
        raise FloatingPointError("Input contains NaN/Inf")
    return points, adjacency, labels, metadata


def run_pytorch_baseline(
    points: np.ndarray, adjacency: np.ndarray, checkpoint_path: Path, output_path: Path
) -> dict[str, Any]:
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
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = GCNResStandardOps(SimpleNamespace(num_class=2), in_dim=4)
    strict_result = model.load_state_dict(state_dict, strict=True)
    wrapper = GCNResOnnxWrapper(model).to("cuda:0").eval()
    points_tensor = torch.from_numpy(points).to("cuda:0")
    adjacency_tensor = torch.from_numpy(adjacency).to("cuda:0")
    torch.cuda.synchronize(0)
    started = time.perf_counter()
    with torch.inference_mode():
        logits_tensor = wrapper(points_tensor, adjacency_tensor)
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    logits = np.ascontiguousarray(logits_tensor.detach().cpu().numpy(), dtype=np.float32)
    if logits.shape != (1, 2048, 2) or not np.isfinite(logits).all():
        raise RuntimeError(f"Invalid PyTorch logits: shape={logits.shape}")
    np.save(output_path, logits, allow_pickle=False)
    result = {
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "strict_load_result": str(strict_result),
        "device": "cuda:0",
        "single_forward_elapsed_seconds_not_a_benchmark": elapsed,
        "logits": array_stats(logits),
        "output_path": str(output_path),
        "output_file_sha256": phase4.sha256(output_path),
    }
    del logits_tensor, adjacency_tensor, points_tensor, wrapper, model, checkpoint, state_dict
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    return result


def engine_io_records(trt: Any, engine: Any, context: Any) -> list[dict[str, Any]]:
    records = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        records.append(
            {
                "index": index,
                "name": name,
                "mode": phase4.enum_name(engine.get_tensor_mode(name)),
                "dtype": phase4.enum_name(engine.get_tensor_dtype(name)),
                "engine_shape": phase4.dims_list(engine.get_tensor_shape(name)),
                "context_shape": phase4.dims_list(context.get_tensor_shape(name)),
                "location": phase4.enum_name(engine.get_tensor_location(name)),
            }
        )
    actual = {
        record["name"]: {
            "mode": record["mode"],
            "dtype": record["dtype"],
            "shape": record["engine_shape"],
        }
        for record in records
    }
    if actual != EXPECTED_IO:
        raise RuntimeError(f"Engine I/O mismatch: expected={EXPECTED_IO}, actual={actual}")
    if any(record["context_shape"] != record["engine_shape"] for record in records):
        raise RuntimeError("Execution context I/O shape differs from fixed engine shape")
    return records


def run_tensorrt(
    trt: Any,
    cudart: Any,
    engine_path: Path,
    plugin_path: Path,
    points: np.ndarray,
    adjacency: np.ndarray,
    output_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logger = trt.Logger(trt.Logger.INFO)
    if not trt.init_libnvinfer_plugins(logger, ""):
        raise RuntimeError("trt.init_libnvinfer_plugins returned false")
    plugin_library, plugin_info = phase4.load_plugin_library(plugin_path)
    if not plugin_info["registration_function_returned"]:
        raise RuntimeError("VoxelUnique plugin registration failed")
    registry = phase4.collect_registry(trt, trt.get_plugin_registry())
    if not registry["voxel_unique_creator_found"]:
        raise RuntimeError("VoxelUnique Creator not found")

    ErrorRecorder = make_error_recorder_class(trt)
    recorder = ErrorRecorder()
    runtime = trt.Runtime(logger)
    runtime.error_recorder = recorder
    engine_bytes = engine_path.read_bytes()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("deserialize_cuda_engine returned None")
    engine.error_recorder = recorder
    runtime_instances = int(plugin_library.getVoxelUniqueRuntimeCreationCount())
    if runtime_instances != 4:
        raise RuntimeError(f"Expected 4 VoxelUnique runtime instances, got {runtime_instances}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("create_execution_context returned None")
    context.error_recorder = recorder
    io_records = engine_io_records(trt, engine, context)

    logits = np.empty((1, 2048, 2), dtype=np.float32, order="C")
    host_arrays = {"points": points, "adj": adjacency, "logits": logits}
    device_pointers: dict[str, int] = {}
    stream = None
    cuda_operations: list[dict[str, Any]] = []
    try:
        cuda_call(cudart, "cudaSetDevice(0)", cudart.cudaSetDevice(0))
        stream = cuda_call(cudart, "cudaStreamCreate", cudart.cudaStreamCreate())[0]
        for name, array in host_arrays.items():
            pointer = int(cuda_call(cudart, f"cudaMalloc({name})", cudart.cudaMalloc(array.nbytes))[0])
            device_pointers[name] = pointer
            cuda_operations.append(
                {
                    "operation": "cudaMalloc",
                    "tensor": name,
                    "bytes": int(array.nbytes),
                    "device_pointer_hex": hex(pointer),
                }
            )
        for name in ("points", "adj"):
            array = host_arrays[name]
            cuda_call(
                cudart,
                f"cudaMemcpyAsync H2D {name}",
                cudart.cudaMemcpyAsync(
                    device_pointers[name],
                    int(array.ctypes.data),
                    int(array.nbytes),
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                ),
            )
            cuda_operations.append(
                {"operation": "cudaMemcpyAsync", "direction": "H2D", "tensor": name, "bytes": int(array.nbytes)}
            )
        for name in ("points", "adj", "logits"):
            if not context.set_tensor_address(name, device_pointers[name]):
                raise RuntimeError(f"set_tensor_address returned false for {name}")

        # Exactly one enqueueV3-equivalent execution via TensorRT Python's
        # execute_async_v3 binding.  No warmup/repeat/benchmark loop exists.
        started = time.perf_counter()
        execution_ok = bool(context.execute_async_v3(stream_handle=int(stream)))
        if not execution_ok:
            raise RuntimeError("execute_async_v3 (enqueueV3) returned false")
        cuda_call(
            cudart,
            "cudaMemcpyAsync D2H logits",
            cudart.cudaMemcpyAsync(
                int(logits.ctypes.data),
                device_pointers["logits"],
                int(logits.nbytes),
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            ),
        )
        cuda_operations.append(
            {"operation": "cudaMemcpyAsync", "direction": "D2H", "tensor": "logits", "bytes": int(logits.nbytes)}
        )
        cuda_call(cudart, "cudaStreamSynchronize", cudart.cudaStreamSynchronize(stream))
        elapsed = time.perf_counter() - started
    finally:
        for name, pointer in reversed(list(device_pointers.items())):
            cuda_call(cudart, f"cudaFree({name})", cudart.cudaFree(pointer))
        if stream is not None:
            cuda_call(cudart, "cudaStreamDestroy", cudart.cudaStreamDestroy(stream))

    if logits.shape != (1, 2048, 2) or not np.isfinite(logits).all():
        raise RuntimeError(f"Invalid TensorRT logits: shape={logits.shape}, finite={np.isfinite(logits).all()}")
    np.save(output_path, logits, allow_pickle=False)
    result = {
        "runtime_deserialize_success": True,
        "execution_context_created": True,
        "engine_io": io_records,
        "voxel_unique_runtime_instances": runtime_instances,
        "cuda_stream": {
            "api": "cuda-python cuda.bindings.runtime.cudaStreamCreate",
            "stream_handle_hex": hex(int(stream)),
            "destroyed_after_inference": True,
        },
        "buffers": {
            name: {
                "bytes": int(array.nbytes),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "device_pointer_hex_diagnostic": hex(device_pointers[name]),
                "freed_after_inference": True,
            }
            for name, array in host_arrays.items()
        },
        "cuda_operations": cuda_operations,
        "enqueue_api": "IExecutionContext.execute_async_v3 / enqueueV3",
        "enqueue_count": 1,
        "single_inference_and_copy_elapsed_seconds_not_a_benchmark": elapsed,
        "logits": array_stats(logits),
        "output_path": str(output_path),
        "output_file_sha256": phase4.sha256(output_path),
        "tensorrt_error_recorder": {
            "num_errors": recorder.num_errors,
            "has_overflowed": recorder.has_overflowed(),
            "errors": recorder.serializable(),
        },
        "inference_success": True,
        "fp16_used": False,
        "int8_used": False,
        "benchmark_attempted": False,
    }
    return result, recorder.serializable()


def main(args: argparse.Namespace) -> int:
    engine = args.engine.resolve()
    onnx_path = args.onnx.resolve()
    input_path = args.input.resolve()
    checkpoint = args.checkpoint.resolve()
    plugin = args.plugin_library.resolve()
    for path in (engine, onnx_path, input_path, checkpoint, plugin):
        if not path.is_file():
            raise FileNotFoundError(path)
    if phase4.sha256(engine) != EXPECTED_ENGINE_SHA256:
        raise RuntimeError("Engine SHA-256 differs from the structurally validated Phase 4 plan")
    if phase4.sha256(onnx_path) != EXPECTED_ONNX_SHA256:
        raise RuntimeError("Derived ONNX SHA-256 differs from Phase 4")

    run_dir = (
        args.output_root.resolve()
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_fp32_inference"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    summary_path = run_dir / "inference_summary.json"
    summary: dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "TENSORRT_FP32_INFERENCE_RUNTIME_FAILED",
        "run_dir": str(run_dir),
        "engine": str(engine),
        "onnx": str(onnx_path),
        "input": str(input_path),
        "checkpoint": str(checkpoint),
        "runtime_inference_attempted": False,
        "runtime_inference_success": False,
        "parity_attempted": False,
        "execution_context_created": False,
        "fp16_used": False,
        "int8_used": False,
        "benchmark_attempted": False,
        "cuda_errors": [],
        "tensorrt_error_recorder": [],
    }
    dump_json(summary_path, summary)
    source_hashes_before = {
        "engine": phase4.sha256(engine),
        "onnx": phase4.sha256(onnx_path),
        "checkpoint": phase4.sha256(checkpoint),
        "plugin": phase4.sha256(plugin),
    }
    try:
        dll_handles = configure_dll_search(
            args.tensorrt_root.resolve(), args.cuda_root.resolve(), plugin
        )
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        environment = collect_environment(
            trt,
            cudart,
            engine,
            onnx_path,
            input_path,
            checkpoint,
            plugin,
            args.tensorrt_root.resolve(),
            args.cuda_root.resolve(),
        )
        dump_json(run_dir / "runtime_environment.json", environment)
        points, adjacency, labels, input_metadata = load_inputs(input_path)
        summary["input_metadata"] = input_metadata
        summary["inputs"] = {"points": array_stats(points), "adj": array_stats(adjacency)}
        summary["ground_truth_available"] = labels is not None
        dump_json(summary_path, summary)

        pytorch_path = run_dir / "pytorch_logits.npy"
        summary["pytorch_baseline"] = run_pytorch_baseline(
            points, adjacency, checkpoint, pytorch_path
        )
        dump_json(summary_path, summary)

        tensorrt_path = run_dir / "tensorrt_logits.npy"
        summary["runtime_inference_attempted"] = True
        dump_json(summary_path, summary)
        tensorrt_result, recorder_errors = run_tensorrt(
            trt, cudart, engine, plugin, points, adjacency, tensorrt_path
        )
        summary["tensorrt_runtime"] = tensorrt_result
        summary["runtime_inference_success"] = True
        summary["execution_context_created"] = True
        summary["tensorrt_error_recorder"] = recorder_errors
        dump_json(summary_path, summary)

        summary["parity_attempted"] = True
        parity_report = parity.compare(
            tensorrt_path,
            pytorch_path,
            input_path if labels is not None else None,
            args.max_abs_threshold,
            args.cosine_threshold,
        )
        dump_json(run_dir / "parity_report.json", parity_report)
        summary["parity"] = parity_report
        summary["status"] = parity_report["status"]
        summary["source_hashes_before"] = source_hashes_before
        summary["source_hashes_after"] = {
            "engine": phase4.sha256(engine),
            "onnx": phase4.sha256(onnx_path),
            "checkpoint": phase4.sha256(checkpoint),
            "plugin": phase4.sha256(plugin),
        }
        summary["sources_unchanged"] = (
            summary["source_hashes_before"] == summary["source_hashes_after"]
        )
        summary["fp16_used"] = False
        summary["int8_used"] = False
        summary["benchmark_attempted"] = False
        dump_json(summary_path, summary)
        comparison = parity_report["numerical_comparison"]
        agreement = parity_report["classification_agreement"]
        print(f"RUN_DIR={run_dir}")
        print(f"MAX_ABSOLUTE_ERROR={comparison['max_absolute_error']:.9e}")
        print(f"MEAN_ABSOLUTE_ERROR={comparison['mean_absolute_error']:.9e}")
        print(f"RMSE={comparison['rmse']:.9e}")
        print(f"COSINE_SIMILARITY={comparison['cosine_similarity']:.12f}")
        print(f"LABEL_AGREEMENT={agreement['agreement']:.12f}")
        print(summary["status"])
        return 0 if parity_report["acceptance"]["passed"] else 2
    except Exception as error:
        summary["status"] = "TENSORRT_FP32_INFERENCE_RUNTIME_FAILED"
        summary["first_error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        summary["source_hashes_before"] = source_hashes_before
        summary["source_hashes_after"] = {
            "engine": phase4.sha256(engine),
            "onnx": phase4.sha256(onnx_path),
            "checkpoint": phase4.sha256(checkpoint),
            "plugin": phase4.sha256(plugin),
        }
        summary["sources_unchanged"] = (
            summary["source_hashes_before"] == summary["source_hashes_after"]
        )
        dump_json(summary_path, summary)
        (run_dir / "runtime_error.txt").write_text(summary["traceback"], encoding="utf-8")
        print(traceback.format_exc(), file=sys.stderr)
        print(f"RUN_DIR={run_dir}")
        print("TENSORRT_FP32_INFERENCE_RUNTIME_FAILED")
        return 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--plugin-library", type=Path, default=DEFAULT_PLUGIN_LIBRARY)
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TENSORRT_ROOT)
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-abs-threshold", type=float, default=1.0e-4)
    parser.add_argument("--cosine-threshold", type=float, default=0.9999)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
