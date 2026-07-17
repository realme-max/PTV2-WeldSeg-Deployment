"""Benchmark TensorRT strict-FP32 pure enqueue and end-to-end latency."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
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
import run_gcn_res_tensorrt_fp32_inference as phase5  # noqa: E402
import smoke_test_gcn_res_tensorrt_engine as phase7a  # noqa: E402


WARMUP_ITERATIONS = 100
BENCHMARK_ITERATIONS = 1000
DEFAULT_ENGINE = phase7a.DEFAULT_ENGINE
DEFAULT_ONNX = phase7a.DEFAULT_ONNX
DEFAULT_PLUGIN_LIBRARY = phase7a.DEFAULT_PLUGIN_LIBRARY
DEFAULT_CHECKPOINT = phase7a.DEFAULT_CHECKPOINT
DEFAULT_PHASE6_RESULTS = phase7a.DEFAULT_PHASE6_RESULTS
DEFAULT_TENSORRT_ROOT = phase7a.DEFAULT_TENSORRT_ROOT
DEFAULT_CUDA_ROOT = phase7a.DEFAULT_CUDA_ROOT


def dump_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def latency_statistics(samples_ms: list[float]) -> dict[str, Any]:
    values = np.asarray(samples_ms, dtype=np.float64)
    if values.shape != (BENCHMARK_ITERATIONS,) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid latency vector: shape={values.shape}")
    return {
        "unit": "ms",
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "raw_samples_ms": [float(value) for value in values],
    }


def memory_snapshot(cudart: Any, label: str) -> dict[str, Any]:
    free_bytes, total_bytes = phase5.cuda_call(
        cudart, "cudaMemGetInfo", cudart.cudaMemGetInfo()
    )
    return {
        "label": label,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "global_used_bytes": int(total_bytes - free_bytes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--plugin-library", type=Path, default=DEFAULT_PLUGIN_LIBRARY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--phase6-results", type=Path, default=DEFAULT_PHASE6_RESULTS)
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TENSORRT_ROOT)
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA_ROOT)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    parser.add_argument("--iterations", type=int, default=BENCHMARK_ITERATIONS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "tensorrt_latency.json"
    failure: dict[str, Any] = {
        "status": "TENSORRT_LATENCY_BENCHMARK_FAILED",
        "component": "tensorrt_latency",
    }
    dump_json(output_path, failure)
    dll_handles: list[Any] = []
    device_pointers: dict[str, int] = {}
    stream: Any = None
    start_event: Any = None
    stop_event: Any = None
    recorder: Any = None
    plugin_library: Any = None
    memory_snapshots: list[dict[str, Any]] = []
    try:
        if args.warmup != WARMUP_ITERATIONS or args.iterations != BENCHMARK_ITERATIONS:
            raise ValueError("Formal Phase 7B requires warmup=100 and iterations=1000")
        engine_path = args.engine.resolve()
        onnx_path = args.onnx.resolve()
        plugin_path = args.plugin_library.resolve()
        checkpoint_path = args.checkpoint.resolve()
        protected = {
            "engine": (engine_path, phase7a.EXPECTED_ENGINE_SHA256),
            "onnx": (onnx_path, phase7a.EXPECTED_ONNX_SHA256),
            "plugin_binary": (plugin_path, phase7a.EXPECTED_PLUGIN_SHA256),
            "checkpoint": (checkpoint_path, phase7a.EXPECTED_CHECKPOINT_SHA256),
        }
        hashes_before = {
            name: phase7a.assert_source_hash(path, expected, name)
            for name, (path, expected) in protected.items()
        }
        plugin_sources_before = phase7a.source_manifest()
        points, adjacency, input_metadata = phase7a.read_fixed_input(
            args.phase6_results.resolve()
        )

        dll_handles = phase5.configure_dll_search(
            args.tensorrt_root.resolve(), args.cuda_root.resolve(), plugin_path
        )
        import tensorrt as trt
        import torch
        from cuda.bindings import runtime as cudart

        phase5.cuda_call(cudart, "cudaSetDevice(0)", cudart.cudaSetDevice(0))
        memory_snapshots.append(memory_snapshot(cudart, "process_baseline_after_cudaSetDevice"))
        logger = trt.Logger(trt.Logger.WARNING)
        if not trt.init_libnvinfer_plugins(logger, ""):
            raise RuntimeError("trt.init_libnvinfer_plugins returned false")
        plugin_library, plugin_info = phase4.load_plugin_library(plugin_path)
        registry = phase4.collect_registry(trt, trt.get_plugin_registry())
        if not plugin_info["registration_function_returned"]:
            raise RuntimeError("VoxelUnique plugin registration failed")
        if not registry["voxel_unique_creator_found"]:
            raise RuntimeError("VoxelUnique Plugin Creator was not found")
        ErrorRecorder = phase5.make_error_recorder_class(trt)
        recorder = ErrorRecorder()
        runtime = trt.Runtime(logger)
        runtime.error_recorder = recorder
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError("deserialize_cuda_engine returned None")
        engine.error_recorder = recorder
        memory_snapshots.append(memory_snapshot(cudart, "after_engine_deserialize"))
        if int(plugin_library.getVoxelUniqueRuntimeCreationCount()) != 4:
            raise RuntimeError("Expected four VoxelUnique runtime plugin instances")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("create_execution_context returned None")
        context.error_recorder = recorder
        memory_snapshots.append(memory_snapshot(cudart, "after_execution_context"))
        io_records = phase5.engine_io_records(trt, engine, context)

        logits = np.empty((1, 2048, 2), dtype=np.float32, order="C")
        buffer_sizes = {
            "points": int(points.nbytes),
            "adj": int(adjacency.nbytes),
            "logits": int(logits.nbytes),
        }
        stream = phase5.cuda_call(cudart, "cudaStreamCreate", cudart.cudaStreamCreate())[0]
        start_event = phase5.cuda_call(
            cudart, "cudaEventCreate(start)", cudart.cudaEventCreate()
        )[0]
        stop_event = phase5.cuda_call(
            cudart, "cudaEventCreate(stop)", cudart.cudaEventCreate()
        )[0]
        for name, byte_count in buffer_sizes.items():
            pointer = int(
                phase5.cuda_call(
                    cudart, f"cudaMalloc({name})", cudart.cudaMalloc(byte_count)
                )[0]
            )
            device_pointers[name] = pointer
            if not context.set_tensor_address(name, pointer):
                raise RuntimeError(f"set_tensor_address failed for {name}")
        memory_snapshots.append(memory_snapshot(cudart, "after_device_buffers"))

        # Seed resident device inputs before measuring pure enqueue latency.
        for name, array in (("points", points), ("adj", adjacency)):
            phase5.cuda_call(
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
        phase5.cuda_call(cudart, "cudaStreamSynchronize", cudart.cudaStreamSynchronize(stream))

        for _ in range(WARMUP_ITERATIONS):
            if not context.execute_async_v3(stream_handle=int(stream)):
                raise RuntimeError("execute_async_v3 failed during pure warmup")
        phase5.cuda_call(cudart, "pure warmup synchronize", cudart.cudaStreamSynchronize(stream))
        memory_snapshots.append(memory_snapshot(cudart, "after_pure_warmup"))

        pure_samples_ms: list[float] = []
        for _ in range(BENCHMARK_ITERATIONS):
            phase5.cuda_call(
                cudart, "cudaEventRecord(start)", cudart.cudaEventRecord(start_event, stream)
            )
            if not context.execute_async_v3(stream_handle=int(stream)):
                raise RuntimeError("execute_async_v3 failed during pure benchmark")
            phase5.cuda_call(
                cudart, "cudaEventRecord(stop)", cudart.cudaEventRecord(stop_event, stream)
            )
            phase5.cuda_call(
                cudart, "cudaEventSynchronize(stop)", cudart.cudaEventSynchronize(stop_event)
            )
            elapsed_ms = phase5.cuda_call(
                cudart,
                "cudaEventElapsedTime",
                cudart.cudaEventElapsedTime(start_event, stop_event),
            )[0]
            pure_samples_ms.append(float(elapsed_ms))
        memory_snapshots.append(memory_snapshot(cudart, "after_pure_benchmark"))

        def end_to_end_once() -> None:
            for tensor_name, host_array in (("points", points), ("adj", adjacency)):
                phase5.cuda_call(
                    cudart,
                    f"cudaMemcpyAsync H2D {tensor_name}",
                    cudart.cudaMemcpyAsync(
                        device_pointers[tensor_name],
                        int(host_array.ctypes.data),
                        int(host_array.nbytes),
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        stream,
                    ),
                )
            if not context.execute_async_v3(stream_handle=int(stream)):
                raise RuntimeError("execute_async_v3 failed during end-to-end path")
            phase5.cuda_call(
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
            phase5.cuda_call(
                cudart, "end-to-end synchronize", cudart.cudaStreamSynchronize(stream)
            )

        for _ in range(WARMUP_ITERATIONS):
            end_to_end_once()
        memory_snapshots.append(memory_snapshot(cudart, "after_end_to_end_warmup"))
        end_to_end_samples_ms: list[float] = []
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter()
            end_to_end_once()
            end_to_end_samples_ms.append((time.perf_counter() - start) * 1000.0)
        memory_snapshots.append(memory_snapshot(cudart, "after_end_to_end_benchmark"))

        if recorder.num_errors:
            raise RuntimeError(f"TensorRT errors: {recorder.serializable()}")
        if not np.isfinite(logits).all():
            raise FloatingPointError("TensorRT logits contain NaN/Inf")
        hashes_after = {name: phase7a.sha256(path) for name, (path, _expected) in protected.items()}
        plugin_sources_after = phase7a.source_manifest()
        if hashes_before != hashes_after or plugin_sources_before != plugin_sources_after:
            raise RuntimeError("A protected Phase 7B source changed during benchmark")

        baseline_used = memory_snapshots[0]["global_used_bytes"]
        for snapshot in memory_snapshots:
            snapshot["incremental_used_from_process_baseline_bytes"] = max(
                0, int(snapshot["global_used_bytes"] - baseline_used)
            )
        observed_peak_incremental = max(
            item["incremental_used_from_process_baseline_bytes"] for item in memory_snapshots
        )
        tensorrt_latency = {
            "status": "TENSORRT_STRICT_FP32_LATENCY_BENCHMARK_COMPLETED",
            "timestamp": datetime.now().astimezone().isoformat(),
            "warmup_iterations_per_mode": WARMUP_ITERATIONS,
            "benchmark_iterations_per_mode": BENCHMARK_ITERATIONS,
            "pure_inference": {
                "scope": "execute_async_v3/enqueueV3 only; inputs already resident on device; no H2D/D2H",
                "timer": "CUDA events on the TensorRT execution stream",
                "latency": latency_statistics(pure_samples_ms),
            },
            "end_to_end": {
                "scope": "pageable-host H2D(points+adj) + enqueueV3 + pageable-host D2H(logits) + stream synchronize",
                "timer": "time.perf_counter wall clock",
                "latency": latency_statistics(end_to_end_samples_ms),
            },
            "input": input_metadata,
            "output_shape": list(logits.shape),
            "output_dtype": str(logits.dtype),
            "output_finite": True,
            "output_sha256": phase5.array_sha256(logits),
            "engine_io": io_records,
            "error_recorder_errors": int(recorder.num_errors),
            "error_recorder": recorder.serializable(),
            "buffer_bytes": buffer_sizes,
            "plugin_runtime_instances": int(plugin_library.getVoxelUniqueRuntimeCreationCount()),
            "source_integrity": {
                name: {
                    "path": str(protected[name][0]),
                    "sha256_before": hashes_before[name],
                    "sha256_after": hashes_after[name],
                    "unchanged": hashes_before[name] == hashes_after[name],
                }
                for name in protected
            },
            "plugin_sources_unchanged": plugin_sources_before == plugin_sources_after,
            "strict_fp32": {
                "tf32_enabled": False,
                "fp16_enabled": False,
                "int8_enabled": False,
            },
        }
        dump_json(output_path, tensorrt_latency)

        pytorch_result_path = run_dir / "pytorch_latency.json"
        if not pytorch_result_path.is_file():
            raise FileNotFoundError(pytorch_result_path)
        pytorch_result = json.loads(pytorch_result_path.read_text(encoding="utf-8"))
        if pytorch_result.get("status") != "PYTORCH_CUDA_LATENCY_BENCHMARK_COMPLETED":
            raise RuntimeError("PyTorch benchmark did not complete successfully")
        memory_summary = {
            "pytorch": pytorch_result["memory"],
            "tensorrt": {
                "method": "cudaMemGetInfo lifecycle snapshots in an isolated TensorRT process",
                "engine_size_bytes": int(engine_path.stat().st_size),
                "runtime_snapshots": memory_snapshots,
                "observed_peak_incremental_bytes_from_process_baseline": int(
                    observed_peak_incremental
                ),
                "caveat": (
                    "cudaMemGetInfo is global free-memory sampling. The reported peak is the "
                    "largest observed lifecycle snapshot delta, not a kernel-internal allocator peak."
                ),
            },
        }
        dump_json(run_dir / "memory_summary.json", memory_summary)

        runtime_api = int(
            phase5.cuda_call(
                cudart, "cudaRuntimeGetVersion", cudart.cudaRuntimeGetVersion()
            )[0]
        )
        driver_api = int(
            phase5.cuda_call(
                cudart, "cudaDriverGetVersion", cudart.cudaDriverGetVersion()
            )[0]
        )
        environment = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_driver": phase4.command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,name,compute_cap",
                    "--format=csv,noheader",
                ]
            ),
            "cuda_runtime_api_version": runtime_api,
            "cuda_driver_api_version": driver_api,
            "cuda_toolkit_root": str(args.cuda_root.resolve()),
            "tensorrt_version": trt.__version__,
            "tensorrt_root": str(args.tensorrt_root.resolve()),
            "pytorch_version": torch.__version__,
            "pytorch_cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "engine_path": str(engine_path),
            "engine_sha256": hashes_before["engine"],
            "onnx_path": str(onnx_path),
            "onnx_sha256": hashes_before["onnx"],
            "plugin_path": str(plugin_path),
            "plugin_sha256": hashes_before["plugin_binary"],
            "checkpoint_sha256": hashes_before["checkpoint"],
            "pip_check": phase4.command_output([sys.executable, "-m", "pip", "check"]),
            "benchmark_contract": {
                "sample_id": "weld_65",
                "batch": 1,
                "num_points": 2048,
                "warmup_iterations": WARMUP_ITERATIONS,
                "benchmark_iterations": BENCHMARK_ITERATIONS,
                "fp32": True,
                "tf32": False,
                "fp16": False,
                "int8": False,
                "accuracy_regression": False,
            },
        }
        dump_json(run_dir / "environment.json", environment)

        pytorch_mean = float(pytorch_result["latency"]["mean"])
        pure_mean = float(tensorrt_latency["pure_inference"]["latency"]["mean"])
        e2e_mean = float(tensorrt_latency["end_to_end"]["latency"]["mean"])
        summary = {
            "status": "TENSORRT_LATENCY_BENCHMARK_COMPLETED",
            "pytorch_mean_ms": pytorch_mean,
            "tensorrt_pure_mean_ms": pure_mean,
            "tensorrt_end_to_end_mean_ms": e2e_mean,
            "pure_inference_speedup_vs_pytorch": pytorch_mean / pure_mean,
            "end_to_end_speedup_vs_pytorch": pytorch_mean / e2e_mean,
        }
        dump_json(run_dir / "benchmark_summary.json", summary)
        print(f"TENSORRT_LATENCY_JSON={output_path}")
        print(json.dumps(summary, ensure_ascii=False))
        print("TENSORRT_LATENCY_BENCHMARK_COMPLETED")
        return 0
    except Exception as exc:
        failure.update(
            {
                "exception_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "error_recorder": (
                    recorder.serializable() if recorder is not None else []
                ),
            }
        )
        dump_json(output_path, failure)
        print(traceback.format_exc(), file=sys.stderr)
        print("TENSORRT_LATENCY_BENCHMARK_FAILED")
        return 1
    finally:
        try:
            if "cudart" in locals():
                for name, pointer in reversed(list(device_pointers.items())):
                    phase5.cuda_call(
                        cudart, f"cudaFree({name})", cudart.cudaFree(pointer)
                    )
                if start_event is not None:
                    phase5.cuda_call(
                        cudart, "cudaEventDestroy(start)", cudart.cudaEventDestroy(start_event)
                    )
                if stop_event is not None:
                    phase5.cuda_call(
                        cudart, "cudaEventDestroy(stop)", cudart.cudaEventDestroy(stop_event)
                    )
                if stream is not None:
                    phase5.cuda_call(
                        cudart, "cudaStreamDestroy", cudart.cudaStreamDestroy(stream)
                    )
        except Exception:
            pass
        del plugin_library
        gc.collect()
        for handle in dll_handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
