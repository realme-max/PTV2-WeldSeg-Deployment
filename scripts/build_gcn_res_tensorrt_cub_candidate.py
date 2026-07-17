"""Parse and build the isolated Phase 8C VoxelUniqueCub candidate engine."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import build_gcn_res_tensorrt_fp32 as phase4  # noqa: E402
import gcn_res_tensorrt_cub_common as common  # noqa: E402


def flag_enabled(trt: Any, config: Any, name: str) -> bool:
    return bool(config.get_flag(getattr(trt.BuilderFlag, name))) if hasattr(trt.BuilderFlag, name) else False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    onnx_path = args.onnx.resolve()
    plugin_path = args.plugin.resolve()
    plan_path = run_dir / "strict_fp32_voxelunique_cub_candidate.plan"
    run_dir.mkdir(parents=True, exist_ok=True)
    protected_before = common.protected_snapshot()
    summary: dict[str, Any] = {
        "status": "TENSORRT_VOXELUNIQUE_CUB_CANDIDATE_BUILD_FAILED",
        "onnx": {"path": str(onnx_path), "sha256": common.sha256(onnx_path)},
        "plugin": {"path": str(plugin_path), "sha256": common.sha256(plugin_path)},
        "protected_before": protected_before,
    }
    common.dump_json(run_dir / "builder_report.json", summary)
    handles: list[Any] = []
    try:
        handles = common.configure_dll_search(plugin_path)
        import tensorrt as trt
        import torch

        logger = trt.Logger(trt.Logger.INFO)
        standard_ok = bool(trt.init_libnvinfer_plugins(logger, ""))
        if not standard_ok:
            raise RuntimeError("trt.init_libnvinfer_plugins returned false")
        library, plugin_info = common.load_cub_plugin(plugin_path)
        if not plugin_info["registered"]:
            raise RuntimeError("VoxelUniqueCub registration returned false")
        registry = common.registry_audit(trt.get_plugin_registry())
        if registry["experimental_match_count"] != 1 or registry["creator_conflict"]:
            raise RuntimeError("VoxelUniqueCub creator registry audit failed")
        if registry["baseline_custom_creator_present"]:
            raise RuntimeError("Baseline VoxelUnique creator is present in isolated candidate process")

        builder = trt.Builder(logger)
        network = builder.create_network(0)
        onnx_parser = trt.OnnxParser(network, logger)
        config = builder.create_builder_config()
        if any(item is None for item in (builder, network, onnx_parser, config)):
            raise RuntimeError("TensorRT Builder/Network/Parser/Config creation failed")
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gib * 1024**3))
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        for flag_name in ("TF32", "FP16", "INT8", "SPARSE_WEIGHTS", "REFIT", "VERSION_COMPATIBLE", "WEIGHT_STREAMING"):
            if hasattr(trt.BuilderFlag, flag_name):
                config.clear_flag(getattr(trt.BuilderFlag, flag_name))
        build_config = {
            "workspace_gib": args.workspace_gib,
            "workspace_bytes": int(args.workspace_gib * 1024**3),
            "tf32_enabled": flag_enabled(trt, config, "TF32"),
            "fp16_enabled": flag_enabled(trt, config, "FP16"),
            "int8_enabled": flag_enabled(trt, config, "INT8"),
            "fixed_shapes": common.EXPECTED_IO,
            "precision_mode": "strict_fp32",
        }
        if any(build_config[name] for name in ("tf32_enabled", "fp16_enabled", "int8_enabled")):
            raise RuntimeError(f"Strict FP32 flags not cleared: {build_config}")
        common.dump_json(run_dir / "builder_config.json", build_config)

        parse_started = time.perf_counter()
        parser_success = bool(onnx_parser.parse_from_file(str(onnx_path)))
        parse_elapsed = time.perf_counter() - parse_started
        parser_errors = phase4.parser_errors(onnx_parser)
        build_creator_count = int(library.getVoxelUniqueCubBuildCreationCount())
        parser_report = {
            "status": "TENSORRT_VOXELUNIQUE_CUB_PARSER_PASSED" if parser_success and not parser_errors else "TENSORRT_VOXELUNIQUE_CUB_PARSER_FAILED",
            "parser_success": parser_success,
            "parser_error_count": len(parser_errors),
            "parser_errors": parser_errors,
            "elapsed_seconds": parse_elapsed,
            "network_layers": int(network.num_layers),
            "network_inputs": int(network.num_inputs),
            "network_outputs": int(network.num_outputs),
            "standard_plugins_initialized": standard_ok,
            "creator_registry": registry,
            "voxelunique_cub_build_creator_count_after_parse": build_creator_count,
            "baseline_creator_loaded": registry["baseline_custom_creator_present"],
        }
        common.dump_json(run_dir / "parser_report.json", parser_report)
        if not parser_success or parser_errors:
            raise RuntimeError(f"TensorRT parser failed: {parser_errors[:1]}")
        if build_creator_count != 4:
            raise RuntimeError(f"Expected 4 VoxelUniqueCub parser creations, got {build_creator_count}")

        build_started = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        build_elapsed = time.perf_counter() - build_started
        if serialized is None:
            raise RuntimeError("build_serialized_network returned None")
        plan_bytes = bytes(serialized)
        temporary = plan_path.with_suffix(".plan.tmp")
        temporary.write_bytes(plan_bytes)
        temporary.replace(plan_path)

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(plan_bytes)
        if engine is None:
            raise RuntimeError("Candidate engine deserialization returned None")
        runtime_creator_count = int(library.getVoxelUniqueCubRuntimeCreationCount())
        io_records = common.engine_io(trt, engine)
        _, inspector_payload, named_plugin_count = phase4.inspect_engine(trt, engine, run_dir)
        layers = inspector_payload.get("Layers", []) if isinstance(inspector_payload, dict) else []
        plugin_layers = [
            layer for layer in layers
            if "voxeluniquecub" in json.dumps(layer, ensure_ascii=False).lower()
            or str(layer.get("Name", "")) in common.EXPECTED_NODE_NAMES
        ]
        # Name-based count is the stable contract across TensorRT Inspector JSON variants.
        unique_plugin_names = sorted({str(layer.get("Name", "")) for layer in plugin_layers if str(layer.get("Name", "")) in common.EXPECTED_NODE_NAMES})
        tf32_layers = [layer for layer in layers if "tf32" in str(layer.get("TacticName", "")).lower()]
        if named_plugin_count != 4 or len(unique_plugin_names) != 4:
            raise RuntimeError(f"Inspector did not retain exactly four VoxelUniqueCub layers: {unique_plugin_names}")
        if tf32_layers:
            raise RuntimeError("Inspector found TF32 tactics in strict FP32 candidate")
        if runtime_creator_count != 4:
            raise RuntimeError(f"Expected 4 runtime plugin creations, got {runtime_creator_count}")
        protected_after = common.protected_snapshot()
        if protected_before != protected_after:
            raise RuntimeError("A protected artifact changed during candidate build")
        engine_metadata = {
            "path": str(plan_path),
            "sha256": common.sha256(plan_path),
            "size_bytes": plan_path.stat().st_size,
            "tensorrt_version": trt.__version__,
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "io_tensors": io_records,
            "inspector_layer_count": len(layers),
            "voxelunique_cub_plugin_count": len(unique_plugin_names),
            "voxelunique_cub_plugin_names": unique_plugin_names,
            "runtime_creation_count": runtime_creator_count,
            "tf32_tactic_count": len(tf32_layers),
        }
        common.dump_json(run_dir / "engine_metadata.json", engine_metadata)
        summary.update({
            "status": "TENSORRT_VOXELUNIQUE_CUB_CANDIDATE_ENGINE_BUILD_PASSED",
            "parser_report": parser_report,
            "builder_config": build_config,
            "build_elapsed_seconds": build_elapsed,
            "serialized_engine_generated": True,
            "engine": engine_metadata,
            "plugin_info": plugin_info,
            "protected_after": protected_after,
        })
        common.dump_json(run_dir / "builder_report.json", summary)
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "tensorrt": trt.__version__,
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "candidate_onnx_sha256": common.sha256(onnx_path),
            "candidate_plugin_sha256": common.sha256(plugin_path),
            "candidate_engine_sha256": common.sha256(plan_path),
        }
        common.dump_json(run_dir / "environment.json", environment)
        print("TENSORRT_VOXELUNIQUE_CUB_PARSER_PASSED")
        print("TENSORRT_VOXELUNIQUE_CUB_CANDIDATE_ENGINE_BUILD_PASSED")
        print(f"CANDIDATE_ENGINE={plan_path}")
        return 0
    except Exception as exc:
        summary["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        summary["protected_after_failure"] = common.protected_snapshot()
        common.dump_json(run_dir / "builder_report.json", summary)
        print(traceback.format_exc(), file=sys.stderr)
        print("TENSORRT_VOXELUNIQUE_CUB_CANDIDATE_ENGINE_BUILD_FAILED")
        return 1
    finally:
        _ = handles


if __name__ == "__main__":
    raise SystemExit(main())
