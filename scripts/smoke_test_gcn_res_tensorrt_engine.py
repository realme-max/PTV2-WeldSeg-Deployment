"""Prepare and smoke-test the formal GCN_res strict-FP32 TensorRT engine.

Phase 7A deliberately performs exactly one TensorRT enqueue.  It does not
rebuild the engine, compare accuracy/parity, warm up, repeat, benchmark, or
measure latency/memory.
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
import evaluate_gcn_res_checkpoint as evaluation  # noqa: E402
import run_gcn_res_tensorrt_fp32_inference as phase5  # noqa: E402


NUM_POINTS = 2048
SAMPLE_ID = "weld_65"

DEFAULT_ENGINE = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260716_224643_531592_strict_fp32"
    / "strict_fp32.plan"
)
DEFAULT_BUILD_DIR = DEFAULT_ENGINE.parent
DEFAULT_ONNX = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260716_190125_699274_dds_reshape_rewrite"
    / "dds_reshape_rewritten.onnx"
)
DEFAULT_PLUGIN_LIBRARY = phase4.DEFAULT_PLUGIN_LIBRARY
DEFAULT_CHECKPOINT = phase5.DEFAULT_CHECKPOINT
DEFAULT_TENSORRT_ROOT = phase4.DEFAULT_TENSORRT_ROOT
DEFAULT_CUDA_ROOT = phase4.DEFAULT_CUDA_ROOT
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
DEFAULT_PHASE6_RESULTS = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260717_110500_836041_strict_fp32_multisample"
    / "per_sample_results.json"
)

EXPECTED_ENGINE_SHA256 = "b76563580089bbc0684e7a8e1edda6028c2d89fd91397c22b4efbd89ecd7fc2c"
EXPECTED_ONNX_SHA256 = "f71a585ded3f348a193e37395a124ab8a425ff72fc09ae92114d41872c460b98"
EXPECTED_PLUGIN_SHA256 = "60c48c400cc926f6aa183de62298a203916b1add6393a3483a332624da39d0ab"
EXPECTED_CHECKPOINT_SHA256 = "311bddf3607d76e6b7ded450b8419bf6ae98f34f50578608b3e6a1c1c3e58d21"
EXPECTED_IO = phase4.EXPECTED_IO

PLUGIN_SOURCE_FILES = (
    PROJECT_ROOT / "deployment" / "tensorrt_voxel_unique_plugin" / "CMakeLists.txt",
    PROJECT_ROOT
    / "deployment"
    / "tensorrt_voxel_unique_plugin"
    / "VoxelUniquePluginLibrary.cpp",
    PROJECT_ROOT
    / "tests"
    / "tensorrt_voxel_unique_correctness"
    / "VoxelUniqueCorrectnessPlugin.h",
    PROJECT_ROOT
    / "tests"
    / "tensorrt_voxel_unique_correctness"
    / "VoxelUniqueCorrectnessPlugin.cu",
)


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


def source_manifest() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in PLUGIN_SOURCE_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"Plugin source file is missing: {path}")
        result[str(path.resolve())] = {
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        }
    return result


def assert_source_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return actual


def make_run_directory(output_root: Path, run_id: str | None) -> Path:
    name = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_phase7a_engine_prepare"
    if any(token in name for token in ("/", "\\", "..")):
        raise ValueError(f"Unsafe run id: {name!r}")
    run_dir = output_root.resolve() / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def read_fixed_input(phase6_results: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    records = json.loads(phase6_results.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError(f"Expected a list in {phase6_results}")
    reference = next((item for item in records if item.get("sample_id") == SAMPLE_ID), None)
    if reference is None:
        raise KeyError(f"{SAMPLE_ID} is absent from {phase6_results}")

    dataset = evaluation.FixedWeldEvaluationDataset("test")
    sample = dataset[0]
    if sample["sample_name"] != SAMPLE_ID:
        raise RuntimeError(
            f"Fixed test sample changed: index 0 is {sample['sample_name']!r}, expected {SAMPLE_ID!r}"
        )
    xyz = sample["normalized_xyz"].unsqueeze(0)
    points_tensor = evaluation.make_model_input(xyz)
    adjacency_tensor, _unused_construction_seconds = evaluation.build_adjacency_cpu(xyz)
    points = np.ascontiguousarray(points_tensor.numpy(), dtype=np.float32)
    adjacency = np.ascontiguousarray(adjacency_tensor.numpy(), dtype=np.float32)
    sample_indices = np.ascontiguousarray(sample["sample_indices"].numpy(), dtype=np.int64)

    actual_hashes = {
        "points_sha256": phase5.array_sha256(points),
        "adj_sha256": phase5.array_sha256(adjacency),
        "sample_indices_sha256": phase5.array_sha256(sample_indices),
    }
    expected_hashes = {
        key: reference[key]
        for key in ("points_sha256", "adj_sha256", "sample_indices_sha256")
    }
    if actual_hashes != expected_hashes:
        raise RuntimeError(
            "Fixed input no longer matches Phase 6: "
            + json.dumps({"expected": expected_hashes, "actual": actual_hashes})
        )
    if points.shape != (1, NUM_POINTS, 4) or adjacency.shape != (
        1,
        NUM_POINTS,
        NUM_POINTS,
    ):
        raise ValueError(f"Input shape mismatch: points={points.shape}, adj={adjacency.shape}")
    if not np.isfinite(points).all() or not np.isfinite(adjacency).all():
        raise FloatingPointError("Fixed input contains NaN/Inf")
    return points, adjacency, {
        "sample_id": SAMPLE_ID,
        "split": "test",
        "split_index": 0,
        "logical_path": sample["logical_path"],
        "seed": evaluation.SEED,
        "num_points": NUM_POINTS,
        "k_neighbors": evaluation.K_NEIGHBORS,
        **actual_hashes,
        "hashes_match_phase6": True,
        "phase6_reference": str(phase6_results.resolve()),
    }


def collect_profiles(engine: Any) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    input_names = [
        engine.get_tensor_name(index)
        for index in range(engine.num_io_tensors)
        if phase4.enum_name(engine.get_tensor_mode(engine.get_tensor_name(index))) == "INPUT"
    ]
    for profile_index in range(int(engine.num_optimization_profiles)):
        inputs: dict[str, Any] = {}
        for name in input_names:
            try:
                minimum, optimum, maximum = engine.get_tensor_profile_shape(name, profile_index)
                inputs[name] = {
                    "min": phase4.dims_list(minimum),
                    "opt": phase4.dims_list(optimum),
                    "max": phase4.dims_list(maximum),
                }
            except Exception as exc:  # Static engines can expose no profile triplet.
                inputs[name] = {
                    "fixed_engine_shape": phase4.dims_list(engine.get_tensor_shape(name)),
                    "profile_shape_query": f"unavailable: {type(exc).__name__}: {exc}",
                }
        profiles.append({"index": profile_index, "inputs": inputs})
    return profiles


def inspect_engine(trt: Any, engine: Any, context: Any) -> dict[str, Any]:
    io_records = phase5.engine_io_records(trt, engine, context)
    actual = {
        item["name"]: {
            "mode": item["mode"],
            "dtype": item["dtype"],
            "shape": item["engine_shape"],
        }
        for item in io_records
    }
    if actual != EXPECTED_IO:
        raise RuntimeError(
            "Engine I/O mismatch: "
            + json.dumps({"expected": EXPECTED_IO, "actual": actual})
        )
    inspector = engine.create_engine_inspector()
    if inspector is None:
        raise RuntimeError("create_engine_inspector returned None")
    payload = json.loads(
        inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    )
    layers = payload.get("Layers", [])
    plugin_layers = [
        layer
        for layer in layers
        if layer.get("LayerType") == "PluginV3" and layer.get("PluginType") == "VoxelUnique"
    ]
    expected_names = [f"/model/tdb_{index}/Unique" for index in range(1, 5)]
    actual_names = [layer.get("Name") for layer in plugin_layers]
    if actual_names != expected_names:
        raise RuntimeError(
            f"VoxelUnique PluginV3 layers mismatch: {actual_names} != {expected_names}"
        )
    condensed_plugins = [
        {
            "name": layer.get("Name"),
            "layer_type": layer.get("LayerType"),
            "plugin_interface": "IPluginV3",
            "plugin_type": layer.get("PluginType"),
            "plugin_version": layer.get("PluginVersion"),
            "inputs": layer.get("Inputs", []),
            "outputs": layer.get("Outputs", []),
            "tactic_value": layer.get("TacticValue"),
        }
        for layer in plugin_layers
    ]
    normalized_io = [
        {
            "index": item["index"],
            "name": item["name"],
            "mode": item["mode"],
            "dtype": item["dtype"],
            "shape": item["engine_shape"],
            "context_shape": item["context_shape"],
            "location": item["location"],
        }
        for item in io_records
    ]
    return {
        "layer_count": len(layers),
        "input_tensors": [item for item in normalized_io if item["mode"] == "INPUT"],
        "output_tensors": [item for item in normalized_io if item["mode"] == "OUTPUT"],
        "num_optimization_profiles": int(engine.num_optimization_profiles),
        "optimization_profiles": collect_profiles(engine),
        "plugin_layer_count": len(plugin_layers),
        "plugin_layers": condensed_plugins,
        "inspector_engine_metadata": payload.get("Engine Metadata"),
    }


def make_phase7a_report(
    run_dir: Path,
    metadata: dict[str, Any],
    smoke: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    input_rows = "\n".join(
        f"- `{item['name']}`: {item['dtype']} `{item['shape']}` ({item['mode']})"
        for item in metadata["engine_inspector"]["input_tensors"]
        + metadata["engine_inspector"]["output_tensors"]
    )
    report = f"""# TensorRT Phase 7A Engine Benchmark Preparation

## Status

```text
ENGINE_BENCHMARK_PREPARATION_COMPLETED
```

This phase only prepared the benchmark runtime boundary and executed exactly one
smoke inference. It did not perform latency, throughput, memory, parity, or
accuracy measurements and did not rebuild or modify the engine.

## Formal engine

- Path: `{metadata['engine_path']}`
- SHA-256: `{metadata['engine_sha256']}`
- TensorRT: `{metadata['tensorrt_version']}`
- Precision: strict FP32 (TF32/FP16/INT8 all disabled)
- Inspector layers: `{metadata['engine_inspector']['layer_count']}`
- Optimization profiles: `{metadata['engine_inspector']['num_optimization_profiles']}`
- VoxelUnique PluginV3 layers: `{metadata['engine_inspector']['plugin_layer_count']}`

## Tensor contract

{input_rows}

## Fixed smoke input

- Sample: `{smoke['input']['sample_id']}` (`test` split index 0)
- points SHA-256: `{smoke['input']['points_sha256']}`
- adj SHA-256: `{smoke['input']['adj_sha256']}`
- Phase 6 hashes matched: `{smoke['input']['hashes_match_phase6']}`

## One-shot runtime smoke result

- Deserialize: `{smoke['deserialize']}`
- Context creation: `{smoke['context_creation']}`
- enqueueV3: `{smoke['enqueue']}`
- Output: `{smoke['output_dtype']} {smoke['output_shape']}`
- Output finite: `{smoke['output_finite']}`
- ErrorRecorder errors: `{smoke['error_recorder_errors']}`
- Logits min/max/mean/std: `{smoke['logits_stats']['min']}`, `{smoke['logits_stats']['max']}`, `{smoke['logits_stats']['mean']}`, `{smoke['logits_stats']['std']}`

## Safety boundary

- Engine unchanged: `{environment['source_integrity']['engine']['unchanged_during_run']}`
- ONNX unchanged: `{environment['source_integrity']['onnx']['unchanged_during_run']}`
- Plugin DLL unchanged: `{environment['source_integrity']['plugin_binary']['unchanged_during_run']}`
- Plugin sources unchanged: `{environment['source_integrity']['plugin_sources_unchanged_during_run']}`
- Checkpoint unchanged: `{environment['source_integrity']['checkpoint']['unchanged_during_run']}`

The next authorized Phase 7 step can use this script and fixed input contract as
the benchmark entry point. No benchmark was run in Phase 7A.
"""
    (run_dir / "phase7a_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--plugin-library", type=Path, default=DEFAULT_PLUGIN_LIBRARY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--phase6-results", type=Path, default=DEFAULT_PHASE6_RESULTS)
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TENSORRT_ROOT)
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = make_run_directory(args.output_root, args.run_id)
    failure: dict[str, Any] = {
        "status": "ENGINE_BENCHMARK_PREPARATION_FAILED",
        "run_directory": str(run_dir),
    }
    dump_json(run_dir / "smoke_test_result.json", failure)

    dll_handles: list[Any] = []
    device_pointers: dict[str, int] = {}
    stream: Any = None
    recorder: Any = None
    plugin_library: Any = None
    try:
        engine_path = args.engine.resolve()
        onnx_path = args.onnx.resolve()
        plugin_path = args.plugin_library.resolve()
        checkpoint_path = args.checkpoint.resolve()
        phase6_results = args.phase6_results.resolve()
        source_paths = {
            "engine": (engine_path, EXPECTED_ENGINE_SHA256),
            "onnx": (onnx_path, EXPECTED_ONNX_SHA256),
            "plugin_binary": (plugin_path, EXPECTED_PLUGIN_SHA256),
            "checkpoint": (checkpoint_path, EXPECTED_CHECKPOINT_SHA256),
        }
        hashes_before = {
            name: assert_source_hash(path, expected, name)
            for name, (path, expected) in source_paths.items()
        }
        plugin_sources_before = source_manifest()

        dll_handles = phase5.configure_dll_search(
            args.tensorrt_root.resolve(), args.cuda_root.resolve(), plugin_path
        )
        import tensorrt as trt
        import torch
        from cuda.bindings import runtime as cudart

        phase5.cuda_call(cudart, "cudaSetDevice(0)", cudart.cudaSetDevice(0))
        runtime_api_version = int(
            phase5.cuda_call(
                cudart, "cudaRuntimeGetVersion", cudart.cudaRuntimeGetVersion()
            )[0]
        )
        driver_api_version = int(
            phase5.cuda_call(
                cudart, "cudaDriverGetVersion", cudart.cudaDriverGetVersion()
            )[0]
        )
        if not trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.INFO), ""):
            raise RuntimeError("trt.init_libnvinfer_plugins returned false")
        plugin_library, plugin_info = phase4.load_plugin_library(plugin_path)
        registry = phase4.collect_registry(trt, trt.get_plugin_registry())
        if not plugin_info["registration_function_returned"]:
            raise RuntimeError("VoxelUnique plugin registration failed")
        if not registry["voxel_unique_creator_found"]:
            raise RuntimeError("VoxelUnique Plugin Creator was not found")

        points, adjacency, input_metadata = read_fixed_input(phase6_results)

        logger = trt.Logger(trt.Logger.INFO)
        ErrorRecorder = phase5.make_error_recorder_class(trt)
        recorder = ErrorRecorder()
        runtime = trt.Runtime(logger)
        runtime.error_recorder = recorder
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError("deserialize_cuda_engine returned None")
        engine.error_recorder = recorder
        if int(plugin_library.getVoxelUniqueRuntimeCreationCount()) != 4:
            raise RuntimeError(
                "Expected four VoxelUnique runtime plugin instances after deserialization"
            )
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("create_execution_context returned None")
        context.error_recorder = recorder
        inspector_summary = inspect_engine(trt, engine, context)

        build_config = json.loads(
            (engine_path.parent / "builder_config.json").read_text(encoding="utf-8")
        )
        gemm_audit = json.loads(
            (engine_path.parent / "strict_fp32_gemm_audit.json").read_text(encoding="utf-8")
        )
        if any(
            (
                build_config.get("tf32_enabled"),
                build_config.get("fp16_enabled"),
                build_config.get("int8_enabled"),
            )
        ):
            raise RuntimeError("The formal engine build configuration is not strict FP32")
        if gemm_audit.get("tf32_gemm_layer_count") != 0:
            raise RuntimeError("Engine Inspector GEMM audit contains TF32 tactics")

        engine_metadata = {
            "engine_path": str(engine_path),
            "engine_sha256": hashes_before["engine"],
            "engine_size_bytes": int(engine_path.stat().st_size),
            "tensorrt_version": trt.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_runtime_api_version": runtime_api_version,
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "precision_mode": "FP32",
            "tf32_enabled": False,
            "fp16_enabled": False,
            "int8_enabled": False,
            "strict_fp32_build_config_path": str(
                (engine_path.parent / "builder_config.json").resolve()
            ),
            "strict_fp32_gemm_audit_path": str(
                (engine_path.parent / "strict_fp32_gemm_audit.json").resolve()
            ),
            "tf32_gemm_layer_count": int(gemm_audit["tf32_gemm_layer_count"]),
            "engine_inspector": inspector_summary,
            "plugin_creator": registry["voxel_unique"],
            "plugin_runtime_instance_count": int(
                plugin_library.getVoxelUniqueRuntimeCreationCount()
            ),
        }
        dump_json(run_dir / "engine_metadata.json", engine_metadata)

        logits = np.empty((1, NUM_POINTS, 2), dtype=np.float32, order="C")
        buffer_sizes = {
            "points": int(points.nbytes),
            "adj": int(adjacency.nbytes),
            "logits": int(logits.nbytes),
        }
        stream = phase5.cuda_call(
            cudart, "cudaStreamCreate", cudart.cudaStreamCreate()
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

        # Phase 7A's only enqueue. There is no warmup, repetition or timer.
        if not context.execute_async_v3(stream_handle=int(stream)):
            raise RuntimeError("execute_async_v3/enqueueV3 returned false")
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
            cudart, "cudaStreamSynchronize", cudart.cudaStreamSynchronize(stream)
        )
        if recorder.num_errors:
            raise RuntimeError(
                f"TensorRT ErrorRecorder contains errors: {recorder.serializable()}"
            )
        if not np.isfinite(logits).all():
            raise FloatingPointError("TensorRT logits contain NaN/Inf")

        for name, pointer in reversed(list(device_pointers.items())):
            phase5.cuda_call(cudart, f"cudaFree({name})", cudart.cudaFree(pointer))
        device_pointers.clear()
        phase5.cuda_call(
            cudart, "cudaStreamDestroy", cudart.cudaStreamDestroy(stream)
        )
        stream = None

        hashes_after = {name: sha256(path) for name, (path, _expected) in source_paths.items()}
        plugin_sources_after = source_manifest()
        source_integrity = {
            name: {
                "path": str(source_paths[name][0]),
                "sha256_before": hashes_before[name],
                "sha256_after": hashes_after[name],
                "unchanged_during_run": hashes_before[name] == hashes_after[name],
            }
            for name in source_paths
        }
        plugin_sources_unchanged = plugin_sources_before == plugin_sources_after
        if not all(item["unchanged_during_run"] for item in source_integrity.values()):
            raise RuntimeError("A protected binary/model source changed during Phase 7A")
        if not plugin_sources_unchanged:
            raise RuntimeError("VoxelUnique plugin source changed during Phase 7A")

        gpu_properties = torch.cuda.get_device_properties(0)
        environment = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "tensorrt_version": trt.__version__,
            "cuda_toolkit_root": str(args.cuda_root.resolve()),
            "cuda_runtime_api_version": runtime_api_version,
            "cuda_driver_api_version": driver_api_version,
            "cuda_python_version": package_version("cuda-python"),
            "pytorch_version": torch.__version__,
            "pytorch_cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_total_memory_bytes": int(gpu_properties.total_memory),
            "gpu_driver": phase4.command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,name,compute_cap",
                    "--format=csv,noheader",
                ]
            ),
            "pip_check": phase4.command_output([sys.executable, "-m", "pip", "check"]),
            "tensorrt_root": str(args.tensorrt_root.resolve()),
            "source_integrity": {
                **source_integrity,
                "plugin_sources_before": plugin_sources_before,
                "plugin_sources_after": plugin_sources_after,
                "plugin_sources_unchanged_during_run": plugin_sources_unchanged,
            },
            "phase_boundaries": {
                "engine_rebuilt": False,
                "onnx_modified": False,
                "plugin_modified": False,
                "checkpoint_modified": False,
                "deployment_model_modified": False,
                "builder_configuration_modified": False,
                "fp16": False,
                "int8": False,
                "latency_benchmark": False,
                "memory_benchmark": False,
                "accuracy_or_parity_comparison": False,
                "tensorrt_enqueue_count": 1,
            },
        }
        dump_json(run_dir / "environment.json", environment)

        smoke = {
            "status": "ENGINE_BENCHMARK_PREPARATION_COMPLETED",
            "deserialize": "PASS",
            "context_creation": "PASS",
            "enqueue": "PASS",
            "enqueue_api": "IExecutionContext.execute_async_v3 / enqueueV3",
            "enqueue_count": 1,
            "output_finite": True,
            "output_shape": list(logits.shape),
            "output_dtype": str(logits.dtype),
            "error_recorder_errors": int(recorder.num_errors),
            "error_recorder": recorder.serializable(),
            "logits_stats": {
                "min": float(logits.min()),
                "max": float(logits.max()),
                "mean": float(logits.mean()),
                "std": float(logits.std()),
            },
            "buffer_bytes": buffer_sizes,
            "cuda_stream_count": 1,
            "input": input_metadata,
            "parity_or_accuracy_comparison_performed": False,
            "timing_measurement_performed": False,
            "memory_measurement_performed": False,
        }
        dump_json(run_dir / "smoke_test_result.json", smoke)
        make_phase7a_report(run_dir, engine_metadata, smoke, environment)
        print(f"RUN_DIRECTORY={run_dir}")
        print("ENGINE_BENCHMARK_PREPARATION_COMPLETED")
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
        dump_json(run_dir / "smoke_test_result.json", failure)
        (run_dir / "phase7a_report.md").write_text(
            "# TensorRT Phase 7A Engine Benchmark Preparation\n\n"
            "```text\nENGINE_BENCHMARK_PREPARATION_FAILED\n```\n\n"
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        print(traceback.format_exc(), file=sys.stderr)
        print(f"RUN_DIRECTORY={run_dir}")
        print("ENGINE_BENCHMARK_PREPARATION_FAILED")
        return 1
    finally:
        try:
            if "cudart" in locals():
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
        del plugin_library
        gc.collect()
        for handle in dll_handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
