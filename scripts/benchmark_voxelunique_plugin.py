"""Benchmark the isolated VoxelUnique TensorRT plugin engine with CUDA events.

The helper deserializes the existing single-plugin correctness engine. It does
not build an engine and does not execute the full PTV2 network.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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


MAX_N = 2048
WARMUP_ITERATIONS = 100
BENCHMARK_ITERATIONS = 1000
DEFAULT_ENGINE = (
    PROJECT_ROOT
    / "artifacts"
    / "tensorrt_plugin_prototype"
    / "20260715_203305_357432_correctness"
    / "voxel_unique_correctness.plan"
)
DEFAULT_PLUGIN_LIBRARY = phase4.DEFAULT_PLUGIN_LIBRARY
DEFAULT_TENSORRT_ROOT = phase4.DEFAULT_TENSORRT_ROOT
DEFAULT_CUDA_ROOT = phase4.DEFAULT_CUDA_ROOT
EXPECTED_ENGINE_SHA256 = "e7939a4ba0f4ddf40c9efd3eb4b9188d6c1582acefc88c22f105a1a230a25b75"
EXPECTED_PLUGIN_SHA256 = "60c48c400cc926f6aa183de62298a203916b1add6393a3483a332624da39d0ab"


def latency_statistics(samples_ms: list[float]) -> dict[str, Any]:
    values = np.asarray(samples_ms, dtype=np.float64)
    if values.shape != (BENCHMARK_ITERATIONS,) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid timing samples: {values.shape}")
    return {
        "unit": "ms",
        "count": int(values.size),
        "avg_ms": float(values.mean()),
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


def run_cases(
    cases: dict[str, np.ndarray],
    engine_path: Path = DEFAULT_ENGINE,
    plugin_path: Path = DEFAULT_PLUGIN_LIBRARY,
    tensorrt_root: Path = DEFAULT_TENSORRT_ROOT,
    cuda_root: Path = DEFAULT_CUDA_ROOT,
) -> dict[str, Any]:
    engine_path = engine_path.resolve()
    plugin_path = plugin_path.resolve()
    if phase4.sha256(engine_path) != EXPECTED_ENGINE_SHA256:
        raise RuntimeError("Isolated correctness Engine SHA-256 mismatch")
    if phase4.sha256(plugin_path) != EXPECTED_PLUGIN_SHA256:
        raise RuntimeError("VoxelUnique Plugin DLL SHA-256 mismatch")
    normalized_cases: dict[str, np.ndarray] = {}
    for name, values in cases.items():
        array = np.ascontiguousarray(values, dtype=np.int64).reshape(-1)
        if array.shape != (MAX_N,):
            raise ValueError(f"{name}: expected [{MAX_N}], got {array.shape}")
        normalized_cases[name] = array

    dll_handles = phase5.configure_dll_search(
        tensorrt_root.resolve(), cuda_root.resolve(), plugin_path
    )
    device_pointers: dict[str, int] = {}
    stream: Any = None
    events: list[Any] = []
    try:
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        class FixedOutputAllocator(trt.IOutputAllocator):
            def __init__(self, pointer: int, capacity_bytes: int) -> None:
                trt.IOutputAllocator.__init__(self)
                self.pointer = int(pointer)
                self.capacity_bytes = int(capacity_bytes)
                self.reset()

            def reset(self) -> None:
                self.reallocation_calls = 0
                self.shape_notifications = 0
                self.max_requested_bytes = 0
                self.latest_shape: list[int] | None = None

            def _allocate(self, size: int) -> int:
                self.reallocation_calls += 1
                self.max_requested_bytes = max(self.max_requested_bytes, int(size))
                if int(size) > self.capacity_bytes:
                    return 0
                return self.pointer

            def reallocate_output(
                self, tensor_name: str, memory: int, size: int, alignment: int
            ) -> int:
                del tensor_name, memory, alignment
                return self._allocate(size)

            def reallocate_output_async(
                self,
                tensor_name: str,
                memory: int,
                size: int,
                alignment: int,
                stream_handle: int,
            ) -> int:
                del tensor_name, memory, alignment, stream_handle
                return self._allocate(size)

            def notify_shape(self, tensor_name: str, shape: Any) -> None:
                del tensor_name
                self.shape_notifications += 1
                self.latest_shape = [int(value) for value in shape]

        phase5.cuda_call(cudart, "cudaSetDevice(0)", cudart.cudaSetDevice(0))
        logger = trt.Logger(trt.Logger.WARNING)
        if not trt.init_libnvinfer_plugins(logger, ""):
            raise RuntimeError("trt.init_libnvinfer_plugins returned false")
        plugin_library, plugin_info = phase4.load_plugin_library(plugin_path)
        registry = phase4.collect_registry(trt, trt.get_plugin_registry())
        if not plugin_info["registration_function_returned"]:
            raise RuntimeError("VoxelUnique plugin registration failed")
        if not registry["voxel_unique_creator_found"]:
            raise RuntimeError("VoxelUnique Plugin Creator was not found")
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError("deserialize_cuda_engine returned None")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("create_execution_context returned None")
        expected_io = {
            "voxel_key": ("INPUT", "INT64"),
            "voxel_count": ("OUTPUT", "INT32"),
            "unique_values": ("OUTPUT", "INT64"),
            "inverse_indices": ("OUTPUT", "INT64"),
        }
        actual_io: list[dict[str, Any]] = []
        for index in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(index)
            mode = phase4.enum_name(engine.get_tensor_mode(tensor_name))
            dtype = phase4.enum_name(engine.get_tensor_dtype(tensor_name))
            actual_io.append(
                {
                    "index": index,
                    "name": tensor_name,
                    "mode": mode,
                    "dtype": dtype,
                    "engine_shape": phase4.dims_list(engine.get_tensor_shape(tensor_name)),
                }
            )
            if expected_io.get(tensor_name) != (mode, dtype):
                raise RuntimeError(f"Unexpected isolated Engine I/O: {tensor_name} {mode} {dtype}")

        host_count = np.empty((1,), dtype=np.int32)
        host_values = np.empty((MAX_N,), dtype=np.int64)
        host_inverse = np.empty((MAX_N,), dtype=np.int64)
        buffer_bytes = {
            "voxel_key": MAX_N * np.dtype(np.int64).itemsize,
            "voxel_count": np.dtype(np.int32).itemsize,
            "unique_values": MAX_N * np.dtype(np.int64).itemsize,
            "inverse_indices": MAX_N * np.dtype(np.int64).itemsize,
        }
        for name, size in buffer_bytes.items():
            device_pointers[name] = int(
                phase5.cuda_call(
                    cudart, f"cudaMalloc({name})", cudart.cudaMalloc(size)
                )[0]
            )
            if not context.set_tensor_address(name, device_pointers[name]):
                raise RuntimeError(f"set_tensor_address failed for {name}")
        allocator = FixedOutputAllocator(
            device_pointers["unique_values"], buffer_bytes["unique_values"]
        )
        if not context.set_output_allocator("unique_values", allocator):
            raise RuntimeError("set_output_allocator(unique_values) failed")
        if not context.set_input_shape("voxel_key", (MAX_N,)):
            raise RuntimeError("set_input_shape(voxel_key=[2048]) failed")
        stream = phase5.cuda_call(cudart, "cudaStreamCreate", cudart.cudaStreamCreate())[0]
        for index in range(4):
            events.append(
                phase5.cuda_call(
                    cudart, f"cudaEventCreate({index})", cudart.cudaEventCreate()
                )[0]
            )

        results: dict[str, Any] = {}
        for case_name, keys in normalized_cases.items():
            allocator.reset()
            phase5.cuda_call(
                cudart,
                f"initial H2D {case_name}",
                cudart.cudaMemcpyAsync(
                    device_pointers["voxel_key"],
                    int(keys.ctypes.data),
                    int(keys.nbytes),
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                ),
            )
            for _ in range(WARMUP_ITERATIONS):
                if not context.execute_async_v3(stream_handle=int(stream)):
                    raise RuntimeError(f"Warmup enqueue failed: {case_name}")
            phase5.cuda_call(
                cudart, f"warmup synchronize {case_name}", cudart.cudaStreamSynchronize(stream)
            )

            h2d_ms: list[float] = []
            kernel_ms: list[float] = []
            d2h_ms: list[float] = []
            memory_copy_ms: list[float] = []
            total_ms: list[float] = []
            for _ in range(BENCHMARK_ITERATIONS):
                phase5.cuda_call(
                    cudart, "record total start", cudart.cudaEventRecord(events[0], stream)
                )
                phase5.cuda_call(
                    cudart,
                    "timed H2D keys",
                    cudart.cudaMemcpyAsync(
                        device_pointers["voxel_key"],
                        int(keys.ctypes.data),
                        int(keys.nbytes),
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        stream,
                    ),
                )
                phase5.cuda_call(
                    cudart, "record kernel start", cudart.cudaEventRecord(events[1], stream)
                )
                if not context.execute_async_v3(stream_handle=int(stream)):
                    raise RuntimeError(f"Timed enqueue failed: {case_name}")
                phase5.cuda_call(
                    cudart, "record kernel stop", cudart.cudaEventRecord(events[2], stream)
                )
                for output_name, host_array in (
                    ("voxel_count", host_count),
                    ("unique_values", host_values),
                    ("inverse_indices", host_inverse),
                ):
                    phase5.cuda_call(
                        cudart,
                        f"timed D2H {output_name}",
                        cudart.cudaMemcpyAsync(
                            int(host_array.ctypes.data),
                            device_pointers[output_name],
                            int(host_array.nbytes),
                            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                            stream,
                        ),
                    )
                phase5.cuda_call(
                    cudart, "record total stop", cudart.cudaEventRecord(events[3], stream)
                )
                phase5.cuda_call(
                    cudart, "timed iteration synchronize", cudart.cudaEventSynchronize(events[3])
                )
                h2d = float(
                    phase5.cuda_call(
                        cudart,
                        "elapsed H2D",
                        cudart.cudaEventElapsedTime(events[0], events[1]),
                    )[0]
                )
                kernel = float(
                    phase5.cuda_call(
                        cudart,
                        "elapsed kernel",
                        cudart.cudaEventElapsedTime(events[1], events[2]),
                    )[0]
                )
                d2h = float(
                    phase5.cuda_call(
                        cudart,
                        "elapsed D2H",
                        cudart.cudaEventElapsedTime(events[2], events[3]),
                    )[0]
                )
                total = float(
                    phase5.cuda_call(
                        cudart,
                        "elapsed total",
                        cudart.cudaEventElapsedTime(events[0], events[3]),
                    )[0]
                )
                h2d_ms.append(h2d)
                kernel_ms.append(kernel)
                d2h_ms.append(d2h)
                memory_copy_ms.append(h2d + d2h)
                total_ms.append(total)

            plugin_count = int(host_count[0])
            if plugin_count < 1 or plugin_count > MAX_N:
                raise RuntimeError(f"Invalid unique count {plugin_count}: {case_name}")
            reference_values, reference_inverse = np.unique(keys, return_inverse=True)
            values_match = bool(
                np.array_equal(host_values[:plugin_count], reference_values.astype(np.int64))
            )
            inverse_match = bool(
                np.array_equal(host_inverse, reference_inverse.astype(np.int64))
            )
            count_match = plugin_count == int(reference_values.size)
            shape_match = allocator.latest_shape == [plugin_count]
            if not (values_match and inverse_match and count_match and shape_match):
                raise RuntimeError(f"Plugin correctness failed after benchmark: {case_name}")
            results[case_name] = {
                "input_size": MAX_N,
                "unique_count": plugin_count,
                "input_sha256": phase5.array_sha256(keys),
                "warmup_iterations": WARMUP_ITERATIONS,
                "benchmark_iterations": BENCHMARK_ITERATIONS,
                "kernel_execution": latency_statistics(kernel_ms),
                "memory_copy_combined": latency_statistics(memory_copy_ms),
                "h2d": latency_statistics(h2d_ms),
                "d2h": latency_statistics(d2h_ms),
                "total": latency_statistics(total_ms),
                "memory_bytes": {
                    "input": int(keys.nbytes),
                    "output_count": int(host_count.nbytes),
                    "output_values_capacity": int(host_values.nbytes),
                    "output_inverse": int(host_inverse.nbytes),
                    "device_total": int(sum(buffer_bytes.values())),
                    "bytes_copied_per_iteration": int(
                        keys.nbytes + host_count.nbytes + host_values.nbytes + host_inverse.nbytes
                    ),
                },
                "output_allocator": {
                    "reallocation_calls": allocator.reallocation_calls,
                    "shape_notifications": allocator.shape_notifications,
                    "max_requested_bytes": allocator.max_requested_bytes,
                    "latest_shape": allocator.latest_shape,
                    "capacity_bytes": allocator.capacity_bytes,
                },
                "correctness": {
                    "count_match": count_match,
                    "values_match": values_match,
                    "inverse_match": inverse_match,
                    "shape_match": shape_match,
                    "sorted_true": True,
                    "passed": True,
                },
            }
        return {
            "status": "VOXELUNIQUE_ISOLATED_BASELINE_COMPLETED",
            "engine": {
                "path": str(engine_path),
                "sha256": phase4.sha256(engine_path),
                "scope": "single VoxelUnique plugin only; no PTV2 layers",
                "io": actual_io,
            },
            "plugin": {
                "path": str(plugin_path),
                "sha256": phase4.sha256(plugin_path),
                "creator": registry["voxel_unique"],
                "runtime_creation_count": int(
                    plugin_library.getVoxelUniqueRuntimeCreationCount()
                ),
            },
            "timer": "CUDA events on one stream",
            "host_memory": "pageable NumPy arrays",
            "cases": results,
        }
    finally:
        try:
            if "cudart" in locals():
                for event in events:
                    phase5.cuda_call(cudart, "cudaEventDestroy", cudart.cudaEventDestroy(event))
                for name, pointer in reversed(list(device_pointers.items())):
                    phase5.cuda_call(
                        cudart, f"cudaFree({name})", cudart.cudaFree(pointer)
                    )
                if stream is not None:
                    phase5.cuda_call(
                        cudart, "cudaStreamDestroy", cudart.cudaStreamDestroy(stream)
                    )
        except Exception:
            pass
        for handle in dll_handles:
            handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--plugin-library", type=Path, default=DEFAULT_PLUGIN_LIBRARY)
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TENSORRT_ROOT)
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with np.load(args.keys_npz, allow_pickle=False) as archive:
        cases = {name: archive[name] for name in archive.files}
    result = run_cases(
        cases,
        args.engine,
        args.plugin_library,
        args.tensorrt_root,
        args.cuda_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"OUTPUT={args.output.resolve()}")
    print("VOXELUNIQUE_ISOLATED_BASELINE_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
