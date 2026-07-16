"""Build and validate a TensorRT GCN_res engine with TF32 explicitly disabled.

The source ONNX, VoxelUnique plugin and checkpoint are read-only.  The only
builder-policy change relative to the validated FP32 engine is clearing the
TensorRT TF32 flag.  One PyTorch CUDA reference and one TensorRT inference are
executed; no warmup, benchmark, FP16, INT8 or graph modification is performed.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
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
import locate_gcn_res_tensorrt_first_divergence as phase5b  # noqa: E402
import run_gcn_res_tensorrt_fp32_inference as phase5  # noqa: E402


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
PLAN_NAME = "strict_fp32.plan"


def builder_config_payload(trt: Any, config: Any, workspace_gib: float) -> dict[str, Any]:
    def flag_state(name: str) -> bool:
        return bool(config.get_flag(getattr(trt.BuilderFlag, name))) if hasattr(
            trt.BuilderFlag, name
        ) else False

    return {
        "precision_policy": "strict_fp32_tensor_operations",
        "workspace_gib": workspace_gib,
        "workspace_bytes": int(workspace_gib * 1024**3),
        "tf32_enabled": flag_state("TF32"),
        "fp16_enabled": flag_state("FP16"),
        "int8_enabled": flag_state("INT8"),
        "builder_flag_api_availability": {
            name: hasattr(trt.BuilderFlag, name) for name in ("TF32", "FP16", "INT8")
        },
        "tf32_explicitly_cleared": True,
        "fp16_explicitly_cleared": True,
        "int8_explicitly_cleared": True,
        "fixed_shapes": {
            "points": [1, 2048, 4],
            "adj": [1, 2048, 2048],
            "logits": [1, 2048, 2],
        },
        "optimization_profile_created": False,
        "fp16": False,
        "int8": False,
        "benchmark": False,
    }


def inspect_strict_engine(
    trt: Any, engine: Any, run_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    io_records, inspector_payload, voxel_count = phase4.inspect_engine(
        trt, engine, run_dir
    )
    phase4.validate_io(io_records)
    if voxel_count != 4:
        raise RuntimeError(f"Inspector found {voxel_count} VoxelUnique instances, expected 4")
    layers = (
        inspector_payload.get("Layers", [])
        if isinstance(inspector_payload, dict)
        else inspector_payload
    )
    gemm_layers = [
        {"index": index, **layer}
        for index, layer in enumerate(layers)
        if str(layer.get("LayerType", "")).lower() == "gemm"
    ]
    tf32_gemm_layers = [
        layer
        for layer in gemm_layers
        if "tf32" in str(layer.get("TacticName", "")).lower()
    ]
    linear1_layers = [
        layer
        for layer in gemm_layers
        if "/model/ptb_0/linear_1/MatMul" in str(layer.get("Metadata", ""))
    ]
    if len(linear1_layers) != 1:
        raise RuntimeError(
            f"Expected one ptb_0.linear_1 GEMM in Inspector, found {len(linear1_layers)}"
        )
    audit = {
        "inspector_layer_count": len(layers),
        "gemm_layer_count": len(gemm_layers),
        "tf32_gemm_layer_count": len(tf32_gemm_layers),
        "tf32_gemm_layers": tf32_gemm_layers,
        "all_gemm_tactics": [
            {
                "index": layer["index"],
                "name": layer.get("Name"),
                "tactic_name": layer.get("TacticName"),
                "metadata": layer.get("Metadata"),
                "input_datatypes": [
                    item.get("Datatype") for item in layer.get("Inputs", [])
                ],
                "output_datatypes": [
                    item.get("Datatype") for item in layer.get("Outputs", [])
                ],
            }
            for layer in gemm_layers
        ],
        "ptb0_linear1": linear1_layers[0],
        "ptb0_linear1_tactic": linear1_layers[0].get("TacticName"),
        "ptb0_linear1_tactic_contains_tf32": (
            "tf32" in str(linear1_layers[0].get("TacticName", "")).lower()
        ),
        "voxel_unique_instances": voxel_count,
        "strict_fp32_inspector_passed": not tf32_gemm_layers,
    }
    phase5b.dump_json(run_dir / "strict_fp32_gemm_audit.json", audit)
    if tf32_gemm_layers:
        raise RuntimeError(
            "Strict-FP32 Inspector found TF32 GEMM tactics: "
            + ", ".join(str(layer.get("Name")) for layer in tf32_gemm_layers[:5])
        )
    return io_records, audit


def execute_once(
    trt: Any,
    cudart: Any,
    engine: Any,
    recorder: Any,
    points: np.ndarray,
    adjacency: np.ndarray,
    output_path: Path,
) -> dict[str, Any]:
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Strict-FP32 create_execution_context returned None")
    context.error_recorder = recorder
    io_records = phase5.engine_io_records(trt, engine, context)
    logits = np.empty((1, 2048, 2), dtype=np.float32, order="C")
    arrays = {"points": points, "adj": adjacency, "logits": logits}
    pointers: dict[str, int] = {}
    stream = None
    try:
        phase5.cuda_call(cudart, "cudaSetDevice(0)", cudart.cudaSetDevice(0))
        stream = phase5.cuda_call(
            cudart, "cudaStreamCreate", cudart.cudaStreamCreate()
        )[0]
        for name, array in arrays.items():
            pointers[name] = int(
                phase5.cuda_call(
                    cudart, f"cudaMalloc({name})", cudart.cudaMalloc(array.nbytes)
                )[0]
            )
        for name in ("points", "adj"):
            array = arrays[name]
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
        for name in ("points", "adj", "logits"):
            if not context.set_tensor_address(name, pointers[name]):
                raise RuntimeError(f"set_tensor_address failed for {name}")
        started = time.perf_counter()
        if not context.execute_async_v3(stream_handle=int(stream)):
            raise RuntimeError("Strict-FP32 execute_async_v3 returned false")
        phase5.cuda_call(
            cudart,
            "cudaMemcpyAsync D2H logits",
            cudart.cudaMemcpyAsync(
                int(logits.ctypes.data),
                pointers["logits"],
                int(logits.nbytes),
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            ),
        )
        phase5.cuda_call(
            cudart, "cudaStreamSynchronize", cudart.cudaStreamSynchronize(stream)
        )
        elapsed = time.perf_counter() - started
    finally:
        for name, pointer in reversed(list(pointers.items())):
            phase5.cuda_call(cudart, f"cudaFree({name})", cudart.cudaFree(pointer))
        if stream is not None:
            phase5.cuda_call(
                cudart, "cudaStreamDestroy", cudart.cudaStreamDestroy(stream)
            )
    if logits.shape != (1, 2048, 2) or not np.isfinite(logits).all():
        raise RuntimeError("Strict-FP32 TensorRT logits are invalid")
    np.save(output_path, logits, allow_pickle=False)
    return {
        "execution_context_created": True,
        "engine_io": io_records,
        "enqueue_api": "IExecutionContext.execute_async_v3 / enqueueV3",
        "enqueue_count": 1,
        "single_inference_and_copy_elapsed_seconds_not_a_benchmark": elapsed,
        "buffers": {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "bytes": int(array.nbytes),
                "freed_after_inference": True,
            }
            for name, array in arrays.items()
        },
        "cuda_stream_created": True,
        "cuda_stream_destroyed": True,
        "logits": phase5.array_stats(logits),
        "output_path": str(output_path),
        "output_sha256": phase4.sha256(output_path),
        "tensorrt_error_recorder": {
            "num_errors": recorder.num_errors,
            "has_overflowed": recorder.has_overflowed(),
            "errors": recorder.serializable(),
        },
        "outputs_finite": True,
        "fp16_used": False,
        "int8_used": False,
        "benchmark_attempted": False,
    }


def write_report(path: Path, summary: dict[str, Any], parity_result: dict[str, Any]) -> None:
    metrics = parity_result["numerical_comparison"]
    agreement = parity_result["classification_agreement"]
    inspector = summary["inspector_audit"]
    report = f"""# TensorRT Strict FP32 build and parity

## Build policy

- Source ONNX: `{summary['onnx_path']}`
- Source ONNX SHA-256: `{summary['onnx_sha256']}`
- TensorRT TF32: explicitly disabled
- TensorRT FP16: disabled
- TensorRT INT8: disabled
- Workspace: `{summary['builder_config']['workspace_gib']}` GiB
- Graph, Plugin and checkpoint were not modified.

## Engine

- Engine: `{summary['engine_path']}`
- SHA-256: `{summary['engine_sha256']}`
- Parser errors: `{summary['parser_error_count']}`
- VoxelUnique build/runtime/inspector instances: `{summary['voxel_unique_build_instances']}` / `{summary['voxel_unique_runtime_instances']}` / `{inspector['voxel_unique_instances']}`
- GEMM layers: `{inspector['gemm_layer_count']}`
- GEMM tactics containing `tf32`: `{inspector['tf32_gemm_layer_count']}`
- `ptb_0.linear_1` tactic: `{inspector['ptb0_linear1_tactic']}`

## Runtime parity

| Metric | Strict TensorRT FP32 vs PyTorch CUDA FP32 |
|---|---:|
| max absolute error | `{metrics['max_absolute_error']:.12e}` |
| mean absolute error | `{metrics['mean_absolute_error']:.12e}` |
| RMSE | `{metrics['rmse']:.12e}` |
| cosine similarity | `{metrics['cosine_similarity']:.15f}` |
| label agreement | `{agreement['matching_points']}/{agreement['total_points']} = {agreement['agreement']:.12f}` |
| outputs finite | `{metrics['outputs_finite']}` |

Acceptance requires finite outputs, max abs `<1e-4`, cosine `>0.9999`, and label agreement `>=99.99%`.

No FP16, INT8, warmup, benchmark, graph rewrite, Plugin optimization or C++ deployment was performed.

`{summary['status']}`
"""
    path.write_text(report, encoding="utf-8")


def main(args: argparse.Namespace) -> int:
    onnx_path = args.onnx.resolve()
    plugin_path = args.plugin_library.resolve()
    checkpoint_path = args.checkpoint.resolve()
    input_path = args.input.resolve()
    for path in (onnx_path, plugin_path, checkpoint_path, input_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if phase4.sha256(onnx_path) != phase5.EXPECTED_ONNX_SHA256:
        raise RuntimeError("Source ONNX hash mismatch")
    source_hashes_before = {
        "onnx": phase4.sha256(onnx_path),
        "plugin": phase4.sha256(plugin_path),
        "checkpoint": phase4.sha256(checkpoint_path),
    }
    run_dir = (
        args.output_root.resolve()
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_strict_fp32"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    engine_path = run_dir / PLAN_NAME
    pytorch_logits_path = run_dir / "pytorch_cuda_fp32_logits.npy"
    tensorrt_logits_path = run_dir / "tensorrt_strict_fp32_logits.npy"
    initial = {
        "status": "TENSORRT_STRICT_FP32_BUILD_OR_PARITY_FAILED",
        "run_dir": str(run_dir),
        "onnx_path": str(onnx_path),
        "source_hashes_before": source_hashes_before,
        "tf32_enabled": False,
        "fp16_enabled": False,
        "int8_enabled": False,
        "benchmark": False,
    }
    phase5b.dump_json(run_dir / "build_summary.json", initial)

    try:
        dll_handles = phase5.configure_dll_search(
            args.tensorrt_root.resolve(), args.cuda_root.resolve(), plugin_path
        )
        import tensorrt as trt
        import torch
        from cuda.bindings import runtime as cudart

        logger = trt.Logger(trt.Logger.INFO)
        if not trt.init_libnvinfer_plugins(logger, ""):
            raise RuntimeError("trt.init_libnvinfer_plugins returned false")
        plugin_library, plugin_info = phase4.load_plugin_library(plugin_path)
        if not plugin_info["registration_function_returned"]:
            raise RuntimeError("VoxelUnique Plugin registration failed")
        registry = phase4.collect_registry(trt, trt.get_plugin_registry())
        if not registry["voxel_unique_creator_found"]:
            raise RuntimeError("VoxelUnique Creator not found")

        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        config = builder.create_builder_config()
        if builder is None or network is None or parser is None or config is None:
            raise RuntimeError("Builder/Network/Parser/Config creation failed")
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, int(args.workspace_gib * 1024**3)
        )
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        for flag_name in (
            "TF32",
            "FP16",
            "INT8",
            "SPARSE_WEIGHTS",
            "REFIT",
            "VERSION_COMPATIBLE",
            "WEIGHT_STREAMING",
        ):
            if hasattr(trt.BuilderFlag, flag_name):
                config.clear_flag(getattr(trt.BuilderFlag, flag_name))
        build_config = builder_config_payload(trt, config, args.workspace_gib)
        if (
            build_config["tf32_enabled"]
            or build_config["fp16_enabled"]
            or build_config["int8_enabled"]
        ):
            raise RuntimeError(f"Strict precision flags were not cleared: {build_config}")
        phase5b.dump_json(run_dir / "builder_config.json", build_config)
        phase5b.dump_json(run_dir / "build_config.json", build_config)

        parser_success = bool(parser.parse_from_file(str(onnx_path)))
        parser_errors = phase4.parser_errors(parser)
        if not parser_success or parser_errors:
            raise RuntimeError(f"Strict-FP32 Parser failed: {parser_errors[:1]}")
        build_instances = int(plugin_library.getVoxelUniqueBuildCreationCount())
        if build_instances != 4:
            raise RuntimeError(f"Expected 4 VoxelUnique build instances, got {build_instances}")
        started = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        build_elapsed = time.perf_counter() - started
        if serialized is None:
            raise RuntimeError("Strict-FP32 build_serialized_network returned None")
        engine_bytes = bytes(serialized)
        temporary = run_dir / (PLAN_NAME + ".tmp")
        temporary.write_bytes(engine_bytes)
        temporary.replace(engine_path)

        ErrorRecorder = phase5.make_error_recorder_class(trt)
        recorder = ErrorRecorder()
        runtime = trt.Runtime(logger)
        runtime.error_recorder = recorder
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError("Strict-FP32 Engine deserialization failed")
        engine.error_recorder = recorder
        runtime_instances = int(plugin_library.getVoxelUniqueRuntimeCreationCount())
        if runtime_instances != 4:
            raise RuntimeError(
                f"Expected 4 VoxelUnique runtime instances, got {runtime_instances}"
            )
        _, inspector_audit = inspect_strict_engine(trt, engine, run_dir)

        points, adjacency, labels, input_metadata = phase5.load_inputs(input_path)
        old_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        old_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
        old_precision = torch.get_float32_matmul_precision()
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            pytorch_summary = phase5.run_pytorch_baseline(
                points, adjacency, checkpoint_path, pytorch_logits_path
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            torch.set_float32_matmul_precision(old_precision)
        pytorch_summary["strict_fp32_runtime_policy"] = {
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "settings_restored": (
                bool(torch.backends.cuda.matmul.allow_tf32) == old_matmul_tf32
                and bool(torch.backends.cudnn.allow_tf32) == old_cudnn_tf32
                and torch.get_float32_matmul_precision() == old_precision
            ),
        }
        runtime_summary = execute_once(
            trt,
            cudart,
            engine,
            recorder,
            points,
            adjacency,
            tensorrt_logits_path,
        )
        if recorder.num_errors:
            raise RuntimeError(f"TensorRT ErrorRecorder contains errors: {recorder.serializable()}")

        parity_result = parity.compare(
            tensorrt_logits_path,
            pytorch_logits_path,
            input_path if labels is not None else None,
            max_abs_threshold=1.0e-4,
            cosine_threshold=0.9999,
        )
        label_agreement = parity_result["classification_agreement"]["agreement"]
        strict_pass = bool(
            parity_result["acceptance"]["passed"] and label_agreement >= 0.9999
        )
        status = (
            "TENSORRT_STRICT_FP32_PARITY_PASSED"
            if strict_pass
            else "TENSORRT_STRICT_FP32_PARITY_FAILED"
        )
        parity_result["status"] = status
        parity_result["strict_fp32_acceptance"] = {
            "max_abs_error_less_than": 1.0e-4,
            "cosine_similarity_greater_than": 0.9999,
            "label_agreement_at_least": 0.9999,
            "outputs_finite": True,
            "passed": strict_pass,
        }
        phase5b.dump_json(run_dir / "parity_report.json", parity_result)
        phase5b.dump_json(run_dir / "runtime_summary.json", runtime_summary)

        source_hashes_after = {
            "onnx": phase4.sha256(onnx_path),
            "plugin": phase4.sha256(plugin_path),
            "checkpoint": phase4.sha256(checkpoint_path),
        }
        summary = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "status": status,
            "run_dir": str(run_dir),
            "onnx_path": str(onnx_path),
            "onnx_sha256": phase4.sha256(onnx_path),
            "engine_path": str(engine_path),
            "engine_sha256": phase4.sha256(engine_path),
            "engine_size_bytes": engine_path.stat().st_size,
            "builder_config": build_config,
            "parser_success": parser_success,
            "parser_error_count": len(parser_errors),
            "parser_errors": parser_errors,
            "build_elapsed_seconds_not_a_benchmark": build_elapsed,
            "voxel_unique_build_instances": build_instances,
            "voxel_unique_runtime_instances": runtime_instances,
            "deserialize_success": True,
            "inspector_audit": inspector_audit,
            "input_metadata": input_metadata,
            "pytorch": pytorch_summary,
            "tensorrt_runtime": runtime_summary,
            "parity": parity_result,
            "source_hashes_before": source_hashes_before,
            "source_hashes_after": source_hashes_after,
            "formal_sources_modified": source_hashes_before != source_hashes_after,
            "tf32_enabled": False,
            "fp16_enabled": False,
            "int8_enabled": False,
            "benchmark": False,
            "graph_modified": False,
            "plugin_modified": False,
            "checkpoint_modified": False,
        }
        if summary["formal_sources_modified"]:
            raise RuntimeError("A read-only source hash changed")
        phase5b.dump_json(run_dir / "build_summary.json", summary)
        write_report(run_dir / "strict_fp32_parity_report.md", summary, parity_result)
        metrics = parity_result["numerical_comparison"]
        print(f"RUN_DIR={run_dir}")
        print(f"ENGINE_SHA256={summary['engine_sha256']}")
        print(f"LINEAR1_TACTIC={inspector_audit['ptb0_linear1_tactic']}")
        print(f"MAX_ABSOLUTE_ERROR={metrics['max_absolute_error']:.9e}")
        print(f"MEAN_ABSOLUTE_ERROR={metrics['mean_absolute_error']:.9e}")
        print(f"RMSE={metrics['rmse']:.9e}")
        print(f"COSINE_SIMILARITY={metrics['cosine_similarity']:.12f}")
        print(f"LABEL_AGREEMENT={label_agreement:.12f}")
        print(status)
        del engine, runtime, recorder, plugin_library, serialized, engine_bytes, dll_handles
        gc.collect()
        return 0 if strict_pass else 2
    except Exception:
        initial.update(
            {
                "traceback": traceback.format_exc(),
                "source_hashes_after": {
                    "onnx": phase4.sha256(onnx_path),
                    "plugin": phase4.sha256(plugin_path),
                    "checkpoint": phase4.sha256(checkpoint_path),
                },
            }
        )
        initial["formal_sources_modified"] = (
            initial["source_hashes_before"] != initial["source_hashes_after"]
        )
        phase5b.dump_json(run_dir / "build_summary.json", initial)
        print(f"RUN_DIR={run_dir}")
        print(initial["traceback"], file=sys.stderr)
        print("TENSORRT_STRICT_FP32_PARITY_FAILED")
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=phase5.DEFAULT_ONNX)
    parser.add_argument(
        "--plugin-library", type=Path, default=phase5.DEFAULT_PLUGIN_LIBRARY
    )
    parser.add_argument("--checkpoint", type=Path, default=phase5.DEFAULT_CHECKPOINT)
    parser.add_argument("--input", type=Path, default=phase5.DEFAULT_INPUT)
    parser.add_argument("--tensorrt-root", type=Path, default=phase5.DEFAULT_TENSORRT_ROOT)
    parser.add_argument("--cuda-root", type=Path, default=phase5.DEFAULT_CUDA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
