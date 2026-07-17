"""Profile the formal GCN_res TensorRT engine with TensorRT IProfiler.

This Phase 7C diagnostic is read-only: it deserializes the existing engine,
warms it up, then collects per-layer timings for exactly 100 resident-input
inferences.  It never rebuilds or modifies the engine, ONNX graph or plugin.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import traceback
from collections import defaultdict
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
PROFILE_ITERATIONS = 100
DEFAULT_ENGINE = phase7a.DEFAULT_ENGINE
DEFAULT_INSPECTOR = DEFAULT_ENGINE.parent / "engine_inspector.json"
DEFAULT_GEMM_AUDIT = DEFAULT_ENGINE.parent / "strict_fp32_gemm_audit.json"
DEFAULT_BUILD_CONFIG = DEFAULT_ENGINE.parent / "builder_config.json"
DEFAULT_ONNX = phase7a.DEFAULT_ONNX
DEFAULT_PLUGIN_LIBRARY = phase7a.DEFAULT_PLUGIN_LIBRARY
DEFAULT_CHECKPOINT = phase7a.DEFAULT_CHECKPOINT
DEFAULT_PHASE6_RESULTS = phase7a.DEFAULT_PHASE6_RESULTS
DEFAULT_TENSORRT_ROOT = phase7a.DEFAULT_TENSORRT_ROOT
DEFAULT_CUDA_ROOT = phase7a.DEFAULT_CUDA_ROOT
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"


def dump_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def make_run_directory(output_root: Path, run_id: str | None) -> Path:
    name = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_phase7c_profiling"
    if any(token in name for token in ("/", "\\", "..")):
        raise ValueError(f"Unsafe run id: {name!r}")
    run_dir = output_root.resolve() / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def tensor_datatypes(layer: dict[str, Any]) -> list[str]:
    values: set[str] = set()
    for key in ("Inputs", "Constants", "Outputs"):
        for tensor in layer.get(key, []) or []:
            value = tensor.get("Datatype")
            if value:
                values.add(str(value))
    return sorted(values)


def match_inspector_layer(
    profile_name: str,
    exact: dict[str, dict[str, Any]],
    layers: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    if profile_name in exact:
        return exact[profile_name], "exact"
    candidates = [
        layer
        for layer in layers
        if layer.get("Name")
        and (
            str(layer["Name"]) in profile_name
            or profile_name in str(layer["Name"])
        )
    ]
    if len(candidates) == 1:
        return candidates[0], "substring"
    return None, "unmatched"


def flags_for_layer(name: str, layer: dict[str, Any] | None) -> list[str]:
    layer_type = "" if layer is None else str(layer.get("LayerType", ""))
    plugin_type = "" if layer is None else str(layer.get("PluginType", ""))
    metadata = "" if layer is None else str(layer.get("Metadata", ""))
    text = " ".join((name, layer_type, plugin_type, metadata)).lower()
    flags: list[str] = []
    if layer_type.lower() == "pluginv3" and plugin_type.lower() == "voxelunique":
        flags.extend(("Plugin", "VoxelUnique", "DynamicShape"))
    elif "plugin" in layer_type.lower():
        flags.append("Plugin")
    if layer_type.lower() == "gemm" or "matmul" in text or "matrixmultiply" in text:
        flags.append("GEMM")
    if "scatter" in text:
        flags.append("Scatter")
    if "gather" in text:
        flags.append("Gather")
    if "reduce" in text:
        flags.append("Reduce")
    if "shuffle" in text or layer_type.lower() == "reshape" or "reshape" in text:
        flags.append("Shuffle")
    dynamic_tokens = (
        "devicetoshapehost",
        "shapehosttodevice",
        "shape_call",
        "nonzero",
        "constantofshape",
        "expand",
        "[size]",
    )
    if any(token in text for token in dynamic_tokens):
        flags.append("DynamicShape")
    if not flags:
        flags.append("Other")
    return list(dict.fromkeys(flags))


def primary_category(flags: list[str]) -> str:
    for category in (
        "VoxelUnique",
        "Plugin",
        "GEMM",
        "Scatter",
        "Gather",
        "Reduce",
        "DynamicShape",
        "Shuffle",
        "Other",
    ):
        if category in flags:
            return category
    return "Other"


def write_top50_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "rank",
        "layer_name",
        "layer_type",
        "primary_category",
        "flags",
        "avg_time_ms",
        "total_time_ms",
        "percentage",
        "reported_calls",
        "tactic",
        "precision",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows[:50], start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "layer_name": row["layer_name"],
                    "layer_type": row["layer_type"],
                    "primary_category": row["primary_category"],
                    "flags": "|".join(row["flags"]),
                    "avg_time_ms": row["avg_time_ms"],
                    "total_time_ms": row["total_time_ms"],
                    "percentage": row["percentage"],
                    "reported_calls": row["reported_calls"],
                    "tactic": row["tactic"],
                    "precision": "|".join(row["precision"]),
                }
            )


def make_summary_markdown(
    run_dir: Path,
    profile: dict[str, Any],
    plugin: dict[str, Any],
    gemm: dict[str, Any],
) -> None:
    top = profile["layers"][:10]
    top_rows = "\n".join(
        f"| {index} | `{row['layer_name']}` | {row['layer_type']} | "
        f"{row['avg_time_ms']:.6f} | {row['percentage']:.3f}% |"
        for index, row in enumerate(top, start=1)
    )
    text = f"""# TensorRT Phase 7C Profiling Summary

```text
TENSORRT_PERFORMANCE_PROFILING_COMPLETED
```

- Profiled iterations: `{profile['profile_iterations']}`
- Average total reported layer time: `{profile['summary']['average_total_layer_time_ms']:.6f} ms`
- Profiled layer count: `{profile['summary']['profiled_layer_count']}`
- VoxelUnique share: `{plugin['summary']['percentage']:.6f}%`
- GEMM share: `{gemm['summary']['percentage']:.6f}%`
- Dynamic-shape flagged share: `{profile['category_aggregates']['overlapping_flags']['DynamicShape']['percentage']:.6f}%`
- Scatter/Gather combined share: `{profile['summary']['scatter_gather_combined_percentage']:.6f}%`
- Classification: `{profile['bottleneck_classification']['code']} - {profile['bottleneck_classification']['label']}`

| Rank | Layer | Type | Avg ms/inference | Share |
|---:|---|---|---:|---:|
{top_rows}

The classification is based on TensorRT IProfiler and Engine Inspector layer
accounting. It is not a replacement for Nsight Compute memory/occupancy metrics.
No optimization or engine rebuild was performed.
"""
    (run_dir / "profiling_summary.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--inspector", type=Path, default=DEFAULT_INSPECTOR)
    parser.add_argument("--gemm-audit", type=Path, default=DEFAULT_GEMM_AUDIT)
    parser.add_argument("--build-config", type=Path, default=DEFAULT_BUILD_CONFIG)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--plugin-library", type=Path, default=DEFAULT_PLUGIN_LIBRARY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--phase6-results", type=Path, default=DEFAULT_PHASE6_RESULTS)
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TENSORRT_ROOT)
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    parser.add_argument("--profile-iterations", type=int, default=PROFILE_ITERATIONS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = make_run_directory(args.output_root, args.run_id)
    failure = {
        "status": "TENSORRT_PERFORMANCE_PROFILING_FAILED",
        "run_directory": str(run_dir),
    }
    dump_json(run_dir / "layer_profile.json", failure)
    dll_handles: list[Any] = []
    device_pointers: dict[str, int] = {}
    stream: Any = None
    recorder: Any = None
    plugin_library: Any = None
    try:
        if args.warmup != WARMUP_ITERATIONS or args.profile_iterations != PROFILE_ITERATIONS:
            raise ValueError("Formal Phase 7C requires warmup=100 and profile_iterations=100")
        engine_path = args.engine.resolve()
        onnx_path = args.onnx.resolve()
        plugin_path = args.plugin_library.resolve()
        checkpoint_path = args.checkpoint.resolve()
        inspector_path = args.inspector.resolve()
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
        inspector_hash_before = phase7a.sha256(inspector_path)
        plugin_sources_before = phase7a.source_manifest()
        points, adjacency, input_metadata = phase7a.read_fixed_input(
            args.phase6_results.resolve()
        )
        inspector_payload = json.loads(inspector_path.read_text(encoding="utf-8"))
        inspector_layers: list[dict[str, Any]] = inspector_payload["Layers"]
        inspector_exact = {
            str(layer["Name"]): layer for layer in inspector_layers if layer.get("Name")
        }
        gemm_audit = json.loads(args.gemm_audit.read_text(encoding="utf-8"))
        build_config = json.loads(args.build_config.read_text(encoding="utf-8"))
        if build_config.get("tf32_enabled") or build_config.get("fp16_enabled") or build_config.get("int8_enabled"):
            raise RuntimeError("Formal builder config is not strict FP32")
        if int(gemm_audit.get("tf32_gemm_layer_count", -1)) != 0:
            raise RuntimeError("Strict engine GEMM audit contains TF32")

        dll_handles = phase5.configure_dll_search(
            args.tensorrt_root.resolve(), args.cuda_root.resolve(), plugin_path
        )
        import tensorrt as trt
        import torch
        from cuda.bindings import runtime as cudart

        class RecordingProfiler(trt.IProfiler):
            def __init__(self) -> None:
                trt.IProfiler.__init__(self)
                self.records: list[tuple[str, float]] = []

            def report_layer_time(self, layer_name: str, ms: float) -> None:
                self.records.append((str(layer_name), float(ms)))

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
        ErrorRecorder = phase5.make_error_recorder_class(trt)
        recorder = ErrorRecorder()
        runtime = trt.Runtime(logger)
        runtime.error_recorder = recorder
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError("deserialize_cuda_engine returned None")
        engine.error_recorder = recorder
        if int(plugin_library.getVoxelUniqueRuntimeCreationCount()) != 4:
            raise RuntimeError("Expected four VoxelUnique runtime instances")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("create_execution_context returned None")
        context.error_recorder = recorder
        io_records = phase5.engine_io_records(trt, engine, context)
        logits = np.empty((1, 2048, 2), dtype=np.float32, order="C")
        buffers = {
            "points": (points, int(points.nbytes)),
            "adj": (adjacency, int(adjacency.nbytes)),
            "logits": (logits, int(logits.nbytes)),
        }
        stream = phase5.cuda_call(cudart, "cudaStreamCreate", cudart.cudaStreamCreate())[0]
        for name, (_array, byte_count) in buffers.items():
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
        phase5.cuda_call(cudart, "input synchronize", cudart.cudaStreamSynchronize(stream))

        # Warmup happens without an attached profiler.
        for _ in range(WARMUP_ITERATIONS):
            if not context.execute_async_v3(stream_handle=int(stream)):
                raise RuntimeError("execute_async_v3 failed during warmup")
        phase5.cuda_call(cudart, "warmup synchronize", cudart.cudaStreamSynchronize(stream))

        profiler = RecordingProfiler()
        context.profiler = profiler
        context.enqueue_emits_profile = True
        for _ in range(PROFILE_ITERATIONS):
            if not context.execute_async_v3(stream_handle=int(stream)):
                raise RuntimeError("execute_async_v3 failed during profiling")
            phase5.cuda_call(
                cudart, "profile iteration synchronize", cudart.cudaStreamSynchronize(stream)
            )
        if not profiler.records:
            raise RuntimeError("TensorRT IProfiler did not report any layer timing")

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
        phase5.cuda_call(cudart, "output synchronize", cudart.cudaStreamSynchronize(stream))
        if recorder.num_errors:
            raise RuntimeError(f"TensorRT errors: {recorder.serializable()}")
        if not np.isfinite(logits).all():
            raise FloatingPointError("Profiled TensorRT output contains NaN/Inf")

        grouped: dict[str, list[float]] = defaultdict(list)
        for layer_name, elapsed_ms in profiler.records:
            grouped[layer_name].append(elapsed_ms)
        rows: list[dict[str, Any]] = []
        total_reported_ms = float(sum(sum(values) for values in grouped.values()))
        if total_reported_ms <= 0:
            raise RuntimeError("Total reported profiling time is not positive")
        for profile_name, samples in grouped.items():
            layer, match_method = match_inspector_layer(
                profile_name, inspector_exact, inspector_layers
            )
            flags = flags_for_layer(profile_name, layer)
            total_ms = float(sum(samples))
            rows.append(
                {
                    "layer_name": profile_name,
                    "layer_type": "UNMATCHED" if layer is None else str(layer.get("LayerType", "UNKNOWN")),
                    "inspector_match": match_method,
                    "inspector_name": None if layer is None else layer.get("Name"),
                    "primary_category": primary_category(flags),
                    "flags": flags,
                    "avg_time_ms": total_ms / PROFILE_ITERATIONS,
                    "avg_time_ms_per_report": total_ms / len(samples),
                    "total_time_ms": total_ms,
                    "percentage": total_ms / total_reported_ms * 100.0,
                    "reported_calls": len(samples),
                    "expected_calls": PROFILE_ITERATIONS,
                    "min_report_ms": float(min(samples)),
                    "max_report_ms": float(max(samples)),
                    "tactic": None if layer is None else layer.get("TacticName", layer.get("TacticValue")),
                    "precision": [] if layer is None else tensor_datatypes(layer),
                    "metadata": None if layer is None else layer.get("Metadata"),
                }
            )
        rows.sort(key=lambda row: row["total_time_ms"], reverse=True)

        primary_totals: dict[str, float] = defaultdict(float)
        flag_totals: dict[str, float] = defaultdict(float)
        for row in rows:
            primary_totals[row["primary_category"]] += row["total_time_ms"]
            for flag in row["flags"]:
                flag_totals[flag] += row["total_time_ms"]

        def aggregate(values: dict[str, float]) -> dict[str, dict[str, float]]:
            return {
                name: {
                    "total_time_ms": float(total),
                    "avg_time_ms_per_inference": float(total / PROFILE_ITERATIONS),
                    "percentage": float(total / total_reported_ms * 100.0),
                }
                for name, total in sorted(values.items(), key=lambda item: item[1], reverse=True)
            }

        primary_aggregates = aggregate(primary_totals)
        flag_aggregates = aggregate(flag_totals)
        for required_flag in (
            "VoxelUnique",
            "Plugin",
            "GEMM",
            "Scatter",
            "Gather",
            "Reduce",
            "Shuffle",
            "DynamicShape",
            "Other",
        ):
            flag_aggregates.setdefault(
                required_flag,
                {"total_time_ms": 0.0, "avg_time_ms_per_inference": 0.0, "percentage": 0.0},
            )

        dynamic_share = flag_aggregates["DynamicShape"]["percentage"]
        gemm_share = flag_aggregates["GEMM"]["percentage"]
        scatter_gather_share = (
            flag_aggregates["Scatter"]["percentage"]
            + flag_aggregates["Gather"]["percentage"]
        )
        tiny_layers = sum(1 for row in rows if row["avg_time_ms"] < 0.05)
        if dynamic_share >= max(gemm_share, scatter_gather_share) and dynamic_share >= 20.0:
            classification = {
                "code": "C",
                "label": "dynamic shape overhead",
                "reason": "Dynamic-shape flagged layers are the largest measured category and exceed 20%.",
            }
        elif gemm_share >= max(dynamic_share, scatter_gather_share) and gemm_share >= 40.0:
            classification = {
                "code": "A",
                "label": "compute bound",
                "reason": "GEMM layers are the largest measured category and exceed 40%.",
            }
        elif scatter_gather_share >= max(dynamic_share, gemm_share) and scatter_gather_share >= 30.0:
            classification = {
                "code": "B",
                "label": "memory-access dominated",
                "reason": "Scatter/Gather layers are the largest measured category and exceed 30%.",
            }
        else:
            classification = {
                "code": "D",
                "label": "kernel launch / fragmented execution overhead",
                "reason": (
                    f"No major category dominates; {tiny_layers}/{len(rows)} profiled layers "
                    "average below 0.05 ms."
                ),
            }
        classification["limitation"] = (
            "IProfiler does not provide achieved bandwidth, occupancy or FLOP utilization; "
            "A/B classification remains an engineering inference without Nsight metrics."
        )

        plugin_rows = [row for row in rows if "VoxelUnique" in row["flags"]]
        expected_plugins = [f"/model/tdb_{index}/Unique" for index in range(1, 5)]
        if sorted(row["layer_name"] for row in plugin_rows) != sorted(expected_plugins):
            raise RuntimeError(
                f"Expected four profiled VoxelUnique layers, got {[row['layer_name'] for row in plugin_rows]}"
            )
        plugin_total = float(sum(row["total_time_ms"] for row in plugin_rows))
        plugin_profile = {
            "status": "VOXELUNIQUE_PLUGIN_PROFILE_COMPLETED",
            "plugin_count": len(plugin_rows),
            "summary": {
                "total_time_ms": plugin_total,
                "average_time_ms_per_inference": plugin_total / PROFILE_ITERATIONS,
                "percentage": plugin_total / total_reported_ms * 100.0,
            },
            "instances": plugin_rows,
        }
        gemm_rows = [row for row in rows if "GEMM" in row["flags"]]
        gemm_total = float(sum(row["total_time_ms"] for row in gemm_rows))
        gemm_profile = {
            "status": "GEMM_PROFILE_COMPLETED",
            "strict_fp32_confirmed": True,
            "gemm_layer_count_in_inspector": int(gemm_audit["gemm_layer_count"]),
            "profiled_gemm_layer_count": len(gemm_rows),
            "tf32_gemm_layer_count": int(gemm_audit["tf32_gemm_layer_count"]),
            "fp16_enabled": False,
            "summary": {
                "total_time_ms": gemm_total,
                "average_time_ms_per_inference": gemm_total / PROFILE_ITERATIONS,
                "percentage": gemm_total / total_reported_ms * 100.0,
            },
            "layers": gemm_rows,
        }

        hashes_after = {name: phase7a.sha256(path) for name, (path, _expected) in protected.items()}
        plugin_sources_after = phase7a.source_manifest()
        if hashes_before != hashes_after:
            raise RuntimeError("A protected artifact changed during profiling")
        if plugin_sources_before != plugin_sources_after:
            raise RuntimeError("Plugin source changed during profiling")
        if inspector_hash_before != phase7a.sha256(inspector_path):
            raise RuntimeError("Engine Inspector JSON changed during profiling")

        profile_payload = {
            "status": "TENSORRT_PERFORMANCE_PROFILING_COMPLETED",
            "timestamp": datetime.now().astimezone().isoformat(),
            "warmup_iterations": WARMUP_ITERATIONS,
            "profile_iterations": PROFILE_ITERATIONS,
            "measurement_scope": "resident-input TensorRT execution; no H2D/D2H in profiled iterations",
            "profiler": "TensorRT IProfiler with enqueue_emits_profile=True",
            "summary": {
                "average_total_layer_time_ms": total_reported_ms / PROFILE_ITERATIONS,
                "total_reported_time_ms": total_reported_ms,
                "profiled_layer_count": len(rows),
                "total_report_callbacks": len(profiler.records),
                "inspector_layer_count": len(inspector_layers),
                "exact_inspector_matches": sum(row["inspector_match"] == "exact" for row in rows),
                "unmatched_profile_layers": sum(row["inspector_match"] == "unmatched" for row in rows),
                "scatter_gather_combined_percentage": scatter_gather_share,
            },
            "input": input_metadata,
            "engine_io": io_records,
            "output_finite": True,
            "output_shape": list(logits.shape),
            "output_dtype": str(logits.dtype),
            "error_recorder_errors": int(recorder.num_errors),
            "category_aggregates": {
                "exclusive_primary": primary_aggregates,
                "overlapping_flags": flag_aggregates,
            },
            "bottleneck_classification": classification,
            "layers": rows,
            "source_integrity": {
                name: {
                    "path": str(protected[name][0]),
                    "sha256_before": hashes_before[name],
                    "sha256_after": hashes_after[name],
                    "unchanged": hashes_before[name] == hashes_after[name],
                }
                for name in protected
            },
            "plugin_sources_unchanged": True,
            "inspector": {
                "path": str(inspector_path),
                "sha256": inspector_hash_before,
                "strict_fp32_build_config": build_config,
            },
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "python_executable": sys.executable,
                "tensorrt_version": trt.__version__,
                "torch_version": torch.__version__,
                "torch_cuda_runtime": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "driver": phase4.command_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version,name,compute_cap",
                        "--format=csv,noheader",
                    ]
                ),
            },
            "boundaries": {
                "engine_rebuilt": False,
                "engine_modified": False,
                "onnx_modified": False,
                "plugin_modified": False,
                "checkpoint_modified": False,
                "builder_config_modified": False,
                "fp16": False,
                "int8": False,
                "optimization_performed": False,
            },
        }
        dump_json(run_dir / "layer_profile.json", profile_payload)
        dump_json(run_dir / "plugin_profile.json", plugin_profile)
        dump_json(run_dir / "gemm_profile.json", gemm_profile)
        write_top50_csv(run_dir / "top50_layers.csv", rows)
        make_summary_markdown(run_dir, profile_payload, plugin_profile, gemm_profile)
        print(f"RUN_DIRECTORY={run_dir}")
        print(f"PROFILED_TOTAL_MS={profile_payload['summary']['average_total_layer_time_ms']:.6f}")
        print(f"VOXELUNIQUE_PERCENT={plugin_profile['summary']['percentage']:.6f}")
        print(f"GEMM_PERCENT={gemm_profile['summary']['percentage']:.6f}")
        print("TENSORRT_PERFORMANCE_PROFILING_COMPLETED")
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
        dump_json(run_dir / "layer_profile.json", failure)
        (run_dir / "profiling_summary.md").write_text(
            "# TensorRT Phase 7C Profiling\n\n"
            "```text\nTENSORRT_PERFORMANCE_PROFILING_FAILED\n```\n\n"
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        print(traceback.format_exc(), file=sys.stderr)
        print(f"RUN_DIRECTORY={run_dir}")
        print("TENSORRT_PERFORMANCE_PROFILING_FAILED")
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
        for handle in dll_handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
