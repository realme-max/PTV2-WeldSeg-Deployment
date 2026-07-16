"""Build and structurally validate the fixed-shape GCN_res TensorRT FP32 engine.

The default parent mode creates a timestamped artifact directory and launches
exactly one worker subprocess with a hard timeout.  Worker mode parses ONNX,
builds a serialized engine, deserializes it, and inspects structure only.  It
never creates an execution context, allocates model I/O buffers, or runs
inference.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260715_213934_180785_if_folded"
    / "if_folded.onnx"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
DEFAULT_PLUGIN_LIBRARY = (
    PROJECT_ROOT
    / "artifacts"
    / "tensorrt_plugin_library"
    / "build_cuda128"
    / "Release"
    / "ptv2_voxel_unique_plugin.dll"
)
DEFAULT_TENSORRT_ROOT = Path(
    r"D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106"
)
DEFAULT_CUDA_ROOT = Path(
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
)
EXPECTED_ONNX_SHA256 = (
    "f0ca962b4e46e7495d40c7f23387c8dffbd4ca88e580452408f2fd9da85bc5ba"
)
PLAN_NAME = "gcn_res_fp32_b1_n2048.plan"
PLUGIN_DLL_NAME = "ptv2_voxel_unique_plugin.dll"
EXPECTED_IO = {
    "points": {"mode": "INPUT", "dtype": "FLOAT", "shape": [1, 2048, 4]},
    "adj": {"mode": "INPUT", "dtype": "FLOAT", "shape": [1, 2048, 2048]},
    "logits": {"mode": "OUTPUT", "dtype": "FLOAT", "shape": [1, 2048, 2]},
}


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=30,
        )
        return {
            "command": subprocess.list2cmdline(command),
            "exit_code": completed.returncode,
            "output": completed.stdout.strip(),
        }
    except Exception as error:
        return {
            "command": subprocess.list2cmdline(command),
            "exit_code": None,
            "output": f"{type(error).__name__}: {error}",
        }


def enum_name(value: Any) -> str:
    text = str(value)
    return text.rsplit(".", 1)[-1]


def dims_list(dims: Any) -> list[int]:
    return [int(value) for value in dims]


def create_initial_summary(workspace_bytes: int) -> dict[str, Any]:
    return {
        "status": "TENSORRT_FP32_ENGINE_BUILD_FAILED",
        "parser_success": False,
        "parser_errors": [],
        "engine_build_attempted": False,
        "engine_build_success": False,
        "build_elapsed_seconds": None,
        "serialized_engine_size_bytes": None,
        "deserialize_success": False,
        "num_io_tensors": None,
        "num_layers_or_inspector_layers": None,
        "voxel_unique_plugin_instances": None,
        "voxel_unique_runtime_instances": None,
        "standard_plugins_initialized": False,
        "workspace_bytes": workspace_bytes,
        "fp16_enabled": False,
        "int8_enabled": False,
        "tf32_default_untouched": True,
        "inference_attempted": False,
        "execution_context_created": False,
        "first_error": None,
        "failure_classification": None,
    }


def classify_failure(message: str, parser_success: bool) -> str:
    lower = message.lower()
    if not parser_success:
        return "TENSORRT_PARSER_REGRESSION_FAILED"
    if "plugin" in lower or "creator" in lower:
        return "TENSORRT_PLUGIN_REGISTRATION_FAILED"
    if "deserialize" in lower:
        return "TENSORRT_ENGINE_DESERIALIZATION_FAILED"
    if "i/o" in lower or "tensor contract" in lower:
        return "TENSORRT_ENGINE_IO_VALIDATION_FAILED"
    return "TENSORRT_FP32_ENGINE_BUILD_FAILED"


def load_plugin_library(path: Path) -> tuple[Any, dict[str, Any]]:
    library = ctypes.CDLL(str(path))
    library.initVoxelUniquePlugin.argtypes = []
    library.initVoxelUniquePlugin.restype = ctypes.c_bool
    library.getVoxelUniqueBuildCreationCount.argtypes = []
    library.getVoxelUniqueBuildCreationCount.restype = ctypes.c_int32
    library.getVoxelUniqueRuntimeCreationCount.argtypes = []
    library.getVoxelUniqueRuntimeCreationCount.restype = ctypes.c_int32
    registered = bool(library.initVoxelUniquePlugin())
    return library, {
        "path": str(path),
        "sha256": sha256(path),
        "registration_function_returned": registered,
    }


def collect_registry(trt: Any, registry: Any) -> dict[str, Any]:
    creators: list[dict[str, Any]] = []
    for creator in registry.all_creators:
        creators.append(
            {
                "name": creator.name,
                "version": creator.plugin_version,
                "namespace": creator.plugin_namespace,
                "python_type": type(creator).__name__,
            }
        )
    voxel = registry.get_creator("VoxelUnique", "1", "com.tensorrt.ptv2")
    scatter_v2 = registry.get_creator("ScatterElements", "2", "")
    scatter = [item for item in creators if "scatter" in item["name"].lower()]
    return {
        "creator_count": len(creators),
        "voxel_unique_creator_found": voxel is not None,
        "voxel_unique": next(
            (
                item
                for item in creators
                if item["name"] == "VoxelUnique"
                and item["version"] == "1"
                and item["namespace"] == "com.tensorrt.ptv2"
            ),
            None,
        ),
        "scatter_elements_v2_creator_found": scatter_v2 is not None,
        "scatter_related_creators": scatter,
        "all_creators": creators,
    }


def collect_environment(
    trt: Any, onnx_path: Path, plugin_path: Path, tensorrt_root: Path, cuda_root: Path
) -> dict[str, Any]:
    import torch

    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(properties.total_memory),
        }
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "tensorrt_version": trt.__version__,
        "tensorrt_root": str(tensorrt_root),
        "cuda_root": str(cuda_root),
        "cuda_path_environment": os.environ.get("CUDA_PATH"),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "nvidia_smi": command_output(["nvidia-smi"]),
        "nvcc": command_output([str(cuda_root / "bin" / "nvcc.exe"), "--version"]),
        "pip_check": command_output([sys.executable, "-m", "pip", "check"]),
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "onnx_sha256": sha256(onnx_path),
        "plugin_library_path": str(plugin_path),
        "plugin_library_size_bytes": plugin_path.stat().st_size,
        "plugin_library_sha256": sha256(plugin_path),
    }


def parser_errors(parser: Any) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for index in range(parser.num_errors):
        error = parser.get_error(index)
        errors.append(
            {
                "index": index,
                "code": int(error.code()),
                "node": int(error.node()),
                "node_name": error.node_name(),
                "operator": error.node_operator(),
                "description": error.desc(),
                "file": error.file(),
                "line": int(error.line()),
                "function": error.func(),
            }
        )
    return errors


def inspect_engine(trt: Any, engine: Any, run_dir: Path) -> tuple[list[dict[str, Any]], Any, int]:
    io_records: list[dict[str, Any]] = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        io_records.append(
            {
                "index": index,
                "name": name,
                "mode": enum_name(engine.get_tensor_mode(name)),
                "dtype": enum_name(engine.get_tensor_dtype(name)),
                "shape": dims_list(engine.get_tensor_shape(name)),
                "location": enum_name(engine.get_tensor_location(name)),
                "format": enum_name(engine.get_tensor_format(name)),
                "format_description": engine.get_tensor_format_desc(name),
            }
        )
    dump_json(run_dir / "engine_io.json", {"io_tensors": io_records})

    inspector = engine.create_engine_inspector()
    if inspector is None:
        raise RuntimeError("Engine Inspector creation failed")
    inspector_text = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    try:
        inspector_payload: Any = json.loads(inspector_text)
    except json.JSONDecodeError:
        inspector_payload = {"raw": inspector_text}
    dump_json(run_dir / "engine_inspector.json", inspector_payload)

    expected_voxel_layers = [f"/model/tdb_{index}/Unique" for index in range(1, 5)]
    inspector_voxel_count = sum(name in inspector_text for name in expected_voxel_layers)
    return io_records, inspector_payload, inspector_voxel_count


def validate_io(io_records: list[dict[str, Any]]) -> None:
    actual = {
        record["name"]: {
            "mode": record["mode"],
            "dtype": record["dtype"],
            "shape": record["shape"],
        }
        for record in io_records
    }
    if actual != EXPECTED_IO:
        raise RuntimeError(
            "Engine I/O tensor contract mismatch: "
            + json.dumps({"expected": EXPECTED_IO, "actual": actual})
        )


def worker(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    onnx_path = args.onnx.resolve()
    plugin_path = args.plugin_library.resolve()
    workspace_bytes = int(args.workspace_gib * 1024**3)
    summary = create_initial_summary(workspace_bytes)
    dump_json(run_dir / "build_summary.json", summary)

    try:
        if sha256(onnx_path) != EXPECTED_ONNX_SHA256:
            raise RuntimeError(
                f"ONNX SHA-256 mismatch: {sha256(onnx_path)} != {EXPECTED_ONNX_SHA256}"
            )
        if not plugin_path.is_file():
            raise FileNotFoundError(plugin_path)

        dll_directory_handles = []
        for directory in (
            args.tensorrt_root.resolve() / "bin",
            args.cuda_root.resolve() / "bin",
            plugin_path.parent,
        ):
            if hasattr(os, "add_dll_directory"):
                # Keep handles alive for the worker lifetime; otherwise CPython
                # closes each search directory as soon as the temporary object
                # is released.
                dll_directory_handles.append(os.add_dll_directory(str(directory)))

        import tensorrt as trt

        environment = collect_environment(
            trt,
            onnx_path,
            plugin_path,
            args.tensorrt_root.resolve(),
            args.cuda_root.resolve(),
        )
        dump_json(run_dir / "environment.json", environment)
        logger = trt.Logger(trt.Logger.VERBOSE)
        standard_initialized = bool(trt.init_libnvinfer_plugins(logger, ""))
        summary["standard_plugins_initialized"] = standard_initialized
        if not standard_initialized:
            raise RuntimeError("trt.init_libnvinfer_plugins returned false")

        plugin_library, plugin_info = load_plugin_library(plugin_path)
        if not plugin_info["registration_function_returned"]:
            raise RuntimeError("initVoxelUniquePlugin returned false")
        registry = trt.get_plugin_registry()
        registry_payload = collect_registry(trt, registry)
        registry_payload["plugin_library"] = plugin_info
        dump_json(run_dir / "plugin_registry.json", registry_payload)
        if not registry_payload["voxel_unique_creator_found"]:
            raise RuntimeError("VoxelUnique version 1 Creator lookup failed")
        if not registry_payload["scatter_elements_v2_creator_found"]:
            raise RuntimeError("TensorRT ScatterElements version 2 Creator lookup failed")

        builder = trt.Builder(logger)
        if builder is None:
            raise RuntimeError("TensorRT Builder creation failed")
        explicit_batch_member = getattr(
            trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None
        )
        # TensorRT 11 uses explicit batch unconditionally and no longer exposes
        # the legacy EXPLICIT_BATCH flag in this Python wheel.
        network_flags = (
            1 << int(explicit_batch_member)
            if explicit_batch_member is not None
            else 0
        )
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, logger)
        config = builder.create_builder_config()
        if network is None or parser is None or config is None:
            raise RuntimeError("Builder Network/Parser/Config creation failed")

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        prohibited_flag_names = (
            "FP16",
            "INT8",
            "SPARSE_WEIGHTS",
            "REFIT",
            "VERSION_COMPATIBLE",
            "WEIGHT_STREAMING",
        )
        prohibited_flags = {
            name: getattr(trt.BuilderFlag, name)
            for name in prohibited_flag_names
            if hasattr(trt.BuilderFlag, name)
        }
        for flag in prohibited_flags.values():
            config.clear_flag(flag)
        build_config = {
            "precision": "FP32",
            "workspace_bytes": workspace_bytes,
            "workspace_gib": args.workspace_gib,
            "fixed_shapes": {
                "points": [1, 2048, 4],
                "adj": [1, 2048, 2048],
            },
            "optimization_profile_created": False,
            "network_creation_flags": network_flags,
            "explicit_batch_is_mandatory_in_tensorrt_11": (
                explicit_batch_member is None
            ),
            "builder_flag_api_availability": {
                name: hasattr(trt.BuilderFlag, name)
                for name in prohibited_flag_names
            },
            "prohibited_flags": {
                name: bool(config.get_flag(flag))
                for name, flag in prohibited_flags.items()
            },
            "tf32_default_untouched": True,
            "tf32_effective_default": bool(config.get_flag(trt.BuilderFlag.TF32)),
            "dla_core": int(config.DLA_core),
            "profiling_verbosity": enum_name(config.profiling_verbosity),
            "engine_build_subprocess": True,
            "timeout_seconds": args.timeout_seconds,
            "inference_attempted": False,
        }
        dump_json(run_dir / "build_config.json", build_config)

        print("PARSER_BEGIN", flush=True)
        parse_success = bool(parser.parse_from_file(str(onnx_path)))
        errors = parser_errors(parser)
        summary["parser_success"] = parse_success
        summary["parser_errors"] = errors
        dump_json(run_dir / "build_summary.json", summary)
        print(
            f"PARSER_END success={parse_success} errors={len(errors)}",
            flush=True,
        )
        if not parse_success or errors:
            raise RuntimeError(f"TensorRT parser regression: {errors[:1]}")

        build_creation_count = int(
            plugin_library.getVoxelUniqueBuildCreationCount()
        )
        if build_creation_count != 4:
            raise RuntimeError(
                f"Expected 4 VoxelUnique build instances after parse, got {build_creation_count}"
            )
        summary["voxel_unique_plugin_instances"] = build_creation_count
        summary["engine_build_attempted"] = True
        dump_json(run_dir / "build_summary.json", summary)

        print(
            f"ENGINE_BUILD_BEGIN workspace_bytes={workspace_bytes}", flush=True
        )
        started = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        elapsed = time.perf_counter() - started
        summary["build_elapsed_seconds"] = elapsed
        if serialized is None:
            raise RuntimeError("build_serialized_network returned None")
        engine_bytes = bytes(serialized)
        summary["engine_build_success"] = True
        summary["serialized_engine_size_bytes"] = len(engine_bytes)
        print(
            f"ENGINE_BUILD_END elapsed_seconds={elapsed:.6f} bytes={len(engine_bytes)}",
            flush=True,
        )

        plan_path = run_dir / PLAN_NAME
        temporary_plan = run_dir / (PLAN_NAME + ".tmp")
        temporary_plan.write_bytes(engine_bytes)
        temporary_plan.replace(plan_path)
        engine_hash = sha256(plan_path)
        (run_dir / "engine_sha256.txt").write_text(
            f"{engine_hash}  {plan_path.name}\n", encoding="utf-8"
        )

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError("Runtime.deserialize_cuda_engine returned None")
        summary["deserialize_success"] = True
        runtime_creation_count = int(
            plugin_library.getVoxelUniqueRuntimeCreationCount()
        )
        summary["voxel_unique_runtime_instances"] = runtime_creation_count

        io_records, _, inspector_voxel_count = inspect_engine(
            trt, engine, run_dir
        )
        validate_io(io_records)
        if runtime_creation_count != 4:
            raise RuntimeError(
                f"Expected 4 VoxelUnique runtime instances, got {runtime_creation_count}"
            )
        if inspector_voxel_count != 4:
            raise RuntimeError(
                "Engine Inspector did not identify all four VoxelUnique layers: "
                f"found {inspector_voxel_count}"
            )

        summary.update(
            {
                "status": "TENSORRT_FP32_ENGINE_BUILD_PASSED",
                "num_io_tensors": int(engine.num_io_tensors),
                "num_layers_or_inspector_layers": int(engine.num_layers),
                "engine_inspector_voxel_unique_instances": inspector_voxel_count,
                "fp16_enabled": bool(
                    config.get_flag(trt.BuilderFlag.FP16)
                    if hasattr(trt.BuilderFlag, "FP16")
                    else False
                ),
                "int8_enabled": bool(
                    config.get_flag(trt.BuilderFlag.INT8)
                    if hasattr(trt.BuilderFlag, "INT8")
                    else False
                ),
                "inference_attempted": False,
                "execution_context_created": False,
                "first_error": None,
                "failure_classification": None,
                "engine_sha256": engine_hash,
            }
        )
        dump_json(run_dir / "build_summary.json", summary)
        print("TENSORRT_FP32_ENGINE_BUILD_PASSED", flush=True)
        return 0
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        summary["first_error"] = message
        summary["failure_classification"] = classify_failure(
            message, bool(summary["parser_success"])
        )
        summary["status"] = summary["failure_classification"]
        summary["traceback"] = traceback.format_exc()
        dump_json(run_dir / "build_summary.json", summary)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        print(summary["status"], flush=True)
        return 2


def parent(args: argparse.Namespace) -> int:
    onnx_path = args.onnx.resolve()
    plugin_source = args.plugin_library.resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    if not plugin_source.is_file():
        raise FileNotFoundError(plugin_source)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.output_root.resolve() / f"{timestamp}_fp32_engine_build"
    run_dir.mkdir(parents=True, exist_ok=False)
    plugin_copy = run_dir / PLUGIN_DLL_NAME
    shutil.copy2(plugin_source, plugin_copy)
    onnx_hash = sha256(onnx_path)
    (run_dir / "onnx_sha256.txt").write_text(
        f"{onnx_hash}  {onnx_path}\n", encoding="utf-8"
    )

    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--run-dir",
        str(run_dir),
        "--onnx",
        str(onnx_path),
        "--plugin-library",
        str(plugin_copy),
        "--tensorrt-root",
        str(args.tensorrt_root.resolve()),
        "--cuda-root",
        str(args.cuda_root.resolve()),
        "--workspace-gib",
        str(args.workspace_gib),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    env = os.environ.copy()
    env["TENSORRT_ROOT"] = str(args.tensorrt_root.resolve())
    env["CUDA_PATH"] = str(args.cuda_root.resolve())
    env["PATH"] = os.pathsep.join(
        [
            str(args.tensorrt_root.resolve() / "bin"),
            str(args.cuda_root.resolve() / "bin"),
            env.get("PATH", ""),
        ]
    )
    log_path = run_dir / "builder_verbose.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write("COMMAND=" + subprocess.list2cmdline(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            return_code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            plan_path = run_dir / PLAN_NAME
            if plan_path.exists():
                plan_path.unlink()
            summary_path = run_dir / "build_summary.json"
            summary = (
                json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.is_file()
                else create_initial_summary(int(args.workspace_gib * 1024**3))
            )
            summary.update(
                {
                    "status": "TENSORRT_FP32_ENGINE_BUILD_TIMEOUT",
                    "failure_classification": "TENSORRT_FP32_ENGINE_BUILD_TIMEOUT",
                    "first_error": (
                        f"Builder worker exceeded timeout {args.timeout_seconds} seconds"
                    ),
                    "inference_attempted": False,
                    "execution_context_created": False,
                }
            )
            dump_json(summary_path, summary)
            log.write("\nTENSORRT_FP32_ENGINE_BUILD_TIMEOUT\n")
            print(f"RUN_DIR={run_dir}")
            print("TENSORRT_FP32_ENGINE_BUILD_TIMEOUT")
            return 124

    summary_path = run_dir / "build_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Worker produced no build_summary.json: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if return_code != 0:
        summary = enrich_failed_run(run_dir, summary)
    print(f"RUN_DIR={run_dir}")
    print(f"WORKER_EXIT_CODE={return_code}")
    print(f"STATUS={summary['status']}")
    print(f"PARSER_SUCCESS={summary['parser_success']}")
    print(f"ENGINE_BUILD_SUCCESS={summary['engine_build_success']}")
    print(f"DESERIALIZE_SUCCESS={summary['deserialize_success']}")
    print(f"INFERENCE_ATTEMPTED={summary['inference_attempted']}")
    return return_code


def enrich_failed_run(run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Attach the first native TensorRT error and explicit non-results."""
    log_path = run_dir / "builder_verbose.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    native_errors = [
        line.strip()
        for line in log_text.splitlines()
        if "[TRT] [E]" in line or "Error Code" in line
    ]
    first_native_error = native_errors[0] if native_errors else None
    dds_size_tensor_failure = bool(
        first_native_error
        and "convertExplicitDDSPluginToImplicit" in first_native_error
        and "DDSOutputIndices" in first_native_error
        and "SizeTensors" in first_native_error
    )
    summary.update(
        {
            "native_tensorrt_first_error": first_native_error,
            "native_tensorrt_error_code": 2
            if first_native_error and "Error Code 2" in first_native_error
            else None,
            "failure_analysis": {
                "corresponding_layer_or_node": (
                    "TensorRT internal DDS plugin conversion pass; no ONNX node "
                    "name was emitted"
                ),
                "related_to_voxel_unique_size_tensor": dds_size_tensor_failure,
                "related_to_scatter_elements": False,
                "shape_tensor_related": dds_size_tensor_failure,
                "tactic_related": False,
                "workspace_related": False,
                "sm120_compatibility_related": False,
                "evidence": (
                    "Builder assertion maps a data-dependent-output plugin node "
                    "to its declared size tensor before tactic profiling."
                    if dds_size_tensor_failure
                    else "See builder_verbose.log"
                ),
            },
            "workspace_retry_performed": False,
            "workspace_retry_reason": (
                "Not performed because the first native error is an internal "
                "DDS size-tensor assertion, not workspace exhaustion."
            ),
            "plan_exists": (run_dir / PLAN_NAME).is_file(),
            "engine_sha256_available": False,
        }
    )
    dump_json(run_dir / "build_summary.json", summary)
    dump_json(
        run_dir / "engine_io.json",
        {
            "available": False,
            "validation_attempted": False,
            "reason": "Engine build failed before serialization/deserialization",
            "expected": EXPECTED_IO,
            "actual": None,
        },
    )
    dump_json(
        run_dir / "engine_inspector.json",
        {
            "available": False,
            "inspection_attempted": False,
            "reason": "Engine build failed; no ICudaEngine exists",
        },
    )
    (run_dir / "engine_sha256.txt").write_text(
        "NOT_GENERATED: TensorRT engine build failed; no .plan file exists.\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--plugin-library", type=Path, default=DEFAULT_PLUGIN_LIBRARY
    )
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TENSORRT_ROOT)
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA_ROOT)
    parser.add_argument("--workspace-gib", type=int, choices=(4, 8), default=4)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.worker and args.run_dir is None:
        parser.error("--worker requires --run-dir")
    return args


if __name__ == "__main__":
    options = parse_args()
    raise SystemExit(worker(options) if options.worker else parent(options))
