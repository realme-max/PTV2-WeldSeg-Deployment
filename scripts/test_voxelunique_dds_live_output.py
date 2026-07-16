"""Keep VoxelUnique size-tensor outputs live and test TensorRT FP32 build.

This diagnostic does not modify the plugin implementation.  It copies the
folded ONNX, adds one Identity consumer and one auxiliary graph output for each
VoxelUnique output 0, runs ONNX checker, then launches one parser/build worker.
No engine deserialization, execution context, inference, parity, or benchmark
is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, helper

import build_gcn_res_tensorrt_fp32 as phase4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260715_213934_180785_if_folded"
    / "if_folded.onnx"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
DEFAULT_PLUGIN_LIBRARY = phase4.DEFAULT_PLUGIN_LIBRARY
DEFAULT_TENSORRT_ROOT = phase4.DEFAULT_TENSORRT_ROOT
DEFAULT_CUDA_ROOT = phase4.DEFAULT_CUDA_ROOT
EXPECTED_INPUT_SHA256 = phase4.EXPECTED_ONNX_SHA256
EXPECTED_PLUGIN_NODES = [f"/model/tdb_{index}/Unique" for index in range(1, 5)]
PLUGIN_SOURCE = (
    PROJECT_ROOT
    / "tests"
    / "tensorrt_voxel_unique_correctness"
    / "VoxelUniqueCorrectnessPlugin.cu"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def tensor_info(tensor: Any) -> dict[str, Any]:
    return {
        "name": tensor.name,
        "dtype": str(tensor.dtype).rsplit(".", 1)[-1],
        "shape": [int(value) for value in tensor.shape],
    }


def rewrite_live_outputs(source: Path, destination: Path) -> dict[str, Any]:
    source_hash_before = phase4.sha256(source)
    plugin_hash_before = phase4.sha256(PLUGIN_SOURCE)
    if source_hash_before != EXPECTED_INPUT_SHA256:
        raise RuntimeError(
            f"Input ONNX hash mismatch: {source_hash_before} != {EXPECTED_INPUT_SHA256}"
        )
    model = onnx.load_model(str(source), load_external_data=False)
    onnx.checker.check_model(model)
    original_graph_outputs = [item.name for item in model.graph.output]
    records: list[dict[str, Any]] = []
    found_nodes: list[str] = []
    for node in list(model.graph.node):
        if node.op_type != "VoxelUnique" or node.domain != "com.tensorrt.ptv2":
            continue
        if node.name not in EXPECTED_PLUGIN_NODES:
            raise RuntimeError(f"Unexpected VoxelUnique node: {node.name}")
        if len(node.output) != 3:
            raise RuntimeError(f"{node.name}: expected 3 outputs")
        found_nodes.append(node.name)
        count_tensor = node.output[0]
        stage = node.name.split("/")[2]
        identity_output = f"/{stage}/VoxelUnique_voxel_count_live_graph_output"
        identity_name = f"{node.name}/KeepVoxelCountLive"
        model.graph.node.append(
            helper.make_node(
                "Identity",
                inputs=[count_tensor],
                outputs=[identity_output],
                name=identity_name,
            )
        )
        model.graph.output.append(
            helper.make_tensor_value_info(identity_output, TensorProto.INT32, [])
        )
        records.append(
            {
                "voxel_unique_node": node.name,
                "plugin_output_index": 0,
                "plugin_output_tensor": count_tensor,
                "dummy_consumer": identity_name,
                "diagnostic_graph_output": identity_output,
                "dtype": "INT32",
                "shape": [],
            }
        )

    if sorted(found_nodes) != sorted(EXPECTED_PLUGIN_NODES):
        raise RuntimeError(
            f"Expected nodes {EXPECTED_PLUGIN_NODES}, found {found_nodes}"
        )
    onnx.save_model(model, str(destination))
    candidate = onnx.load_model(str(destination), load_external_data=False)
    onnx.checker.check_model(candidate)

    consumers: dict[str, list[str]] = {}
    for node in candidate.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node.name)
    graph_outputs = [item.name for item in candidate.graph.output]
    for record in records:
        if consumers.get(record["plugin_output_tensor"]) != [record["dummy_consumer"]]:
            raise RuntimeError(
                f"Unexpected consumers for {record['plugin_output_tensor']}: "
                f"{consumers.get(record['plugin_output_tensor'])}"
            )
        if record["diagnostic_graph_output"] not in graph_outputs:
            raise RuntimeError("Diagnostic graph output was not retained")

    source_hash_after = phase4.sha256(source)
    plugin_hash_after = phase4.sha256(PLUGIN_SOURCE)
    if source_hash_after != source_hash_before or plugin_hash_after != plugin_hash_before:
        raise RuntimeError("Read-only source/input hash changed")
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "input_onnx": str(source),
        "input_sha256_before": source_hash_before,
        "input_sha256_after": source_hash_after,
        "input_onnx_unchanged": True,
        "plugin_source": str(PLUGIN_SOURCE),
        "plugin_sha256_before": plugin_hash_before,
        "plugin_sha256_after": plugin_hash_after,
        "plugin_unchanged": True,
        "output_onnx": str(destination),
        "output_sha256": phase4.sha256(destination),
        "onnx_checker_passed": True,
        "rewrite_scope": "Four VoxelUnique output-0 tensors only",
        "plugin_output_order_changed": False,
        "declare_size_tensor_changed": False,
        "model_computation_path_changed": False,
        "original_graph_outputs": original_graph_outputs,
        "diagnostic_graph_outputs_added": [
            item["diagnostic_graph_output"] for item in records
        ],
        "final_graph_outputs": graph_outputs,
        "identity_nodes_added": len(records),
        "records": records,
        "engine_build_called_during_rewrite": False,
        "inference_called": False,
    }


def worker(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    onnx_path = args.onnx.resolve()
    plugin_path = args.plugin_library.resolve()
    workspace_bytes = 4 * 1024**3
    parser_summary: dict[str, Any] = {
        "parser_success": False,
        "parser_error_count": None,
        "parser_errors": [],
        "inputs": [],
        "outputs": [],
        "standard_plugins_initialized": False,
        "voxel_unique_creator_found": False,
        "scatter_elements_v2_creator_found": False,
        "voxel_unique_plugin_instances": None,
        "engine_build_called": False,
        "inference_called": False,
    }
    build_summary: dict[str, Any] = {
        "status": "TENSORRT_DDS_LIVE_OUTPUT_BUILD_FAILED",
        "hypothesis": (
            "Keeping VoxelUnique output 0 live prevents DDS size-tensor map mismatch"
        ),
        "workspace_bytes": workspace_bytes,
        "fp32_only": True,
        "parser_success": False,
        "engine_build_attempted": False,
        "engine_build_success": False,
        "build_elapsed_seconds": None,
        "serialized_engine_size_bytes": None,
        "serialized_engine_sha256": None,
        "serialized_engine_retained": False,
        "deserialization_attempted": False,
        "execution_context_created": False,
        "inference_attempted": False,
        "first_error": None,
    }
    phase4.dump_json(run_dir / "parser_summary.json", parser_summary)
    phase4.dump_json(run_dir / "build_summary.json", build_summary)
    try:
        dll_handles = []
        for directory in (
            args.tensorrt_root.resolve() / "bin",
            args.cuda_root.resolve() / "bin",
            plugin_path.parent,
        ):
            if hasattr(os, "add_dll_directory"):
                dll_handles.append(os.add_dll_directory(str(directory)))

        import tensorrt as trt

        logger = trt.Logger(trt.Logger.VERBOSE)
        standard_initialized = bool(trt.init_libnvinfer_plugins(logger, ""))
        parser_summary["standard_plugins_initialized"] = standard_initialized
        if not standard_initialized:
            raise RuntimeError("trt.init_libnvinfer_plugins returned false")
        plugin_library, _ = phase4.load_plugin_library(plugin_path)
        registry_payload = phase4.collect_registry(trt, trt.get_plugin_registry())
        parser_summary["voxel_unique_creator_found"] = registry_payload[
            "voxel_unique_creator_found"
        ]
        parser_summary["scatter_elements_v2_creator_found"] = registry_payload[
            "scatter_elements_v2_creator_found"
        ]
        if not parser_summary["voxel_unique_creator_found"]:
            raise RuntimeError("VoxelUnique Creator not found")
        if not parser_summary["scatter_elements_v2_creator_found"]:
            raise RuntimeError("ScatterElements v2 Creator not found")

        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        config = builder.create_builder_config()
        if builder is None or network is None or parser is None or config is None:
            raise RuntimeError("TensorRT Builder/Network/Parser/Config creation failed")
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        for flag_name in (
            "FP16",
            "INT8",
            "SPARSE_WEIGHTS",
            "REFIT",
            "VERSION_COMPATIBLE",
            "WEIGHT_STREAMING",
        ):
            if hasattr(trt.BuilderFlag, flag_name):
                config.clear_flag(getattr(trt.BuilderFlag, flag_name))

        print("PARSER_BEGIN", flush=True)
        parse_success = bool(parser.parse_from_file(str(onnx_path)))
        errors = phase4.parser_errors(parser)
        parser_summary.update(
            {
                "parser_success": parse_success,
                "parser_error_count": len(errors),
                "parser_errors": errors,
                "inputs": [
                    tensor_info(network.get_input(index))
                    for index in range(network.num_inputs)
                ],
                "outputs": [
                    tensor_info(network.get_output(index))
                    for index in range(network.num_outputs)
                ],
                "voxel_unique_plugin_instances": int(
                    plugin_library.getVoxelUniqueBuildCreationCount()
                ),
            }
        )
        phase4.dump_json(run_dir / "parser_summary.json", parser_summary)
        print(
            f"PARSER_END success={parse_success} errors={len(errors)}",
            flush=True,
        )
        if not parse_success or errors:
            raise RuntimeError(f"Parser failed: {errors[:1]}")
        if parser_summary["voxel_unique_plugin_instances"] != 4:
            raise RuntimeError(
                "Expected four VoxelUnique instances, got "
                f"{parser_summary['voxel_unique_plugin_instances']}"
            )

        expected_outputs = {
            "logits": {"dtype": "FLOAT", "shape": [1, 2048, 2]},
            **{
                f"/tdb_{index}/VoxelUnique_voxel_count_live_graph_output": {
                    "dtype": "INT32",
                    "shape": [],
                }
                for index in range(1, 5)
            },
        }
        actual_outputs = {
            item["name"]: {"dtype": item["dtype"], "shape": item["shape"]}
            for item in parser_summary["outputs"]
        }
        if actual_outputs != expected_outputs:
            raise RuntimeError(
                "Parser output contract mismatch: "
                + json.dumps(
                    {"expected": expected_outputs, "actual": actual_outputs}
                )
            )

        build_summary["parser_success"] = True
        build_summary["engine_build_attempted"] = True
        parser_summary["engine_build_called"] = True
        phase4.dump_json(run_dir / "parser_summary.json", parser_summary)
        phase4.dump_json(run_dir / "build_summary.json", build_summary)
        print("ENGINE_BUILD_BEGIN", flush=True)
        started = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        elapsed = time.perf_counter() - started
        build_summary["build_elapsed_seconds"] = elapsed
        if serialized is None:
            raise RuntimeError("build_serialized_network returned None")
        engine_bytes = bytes(serialized)
        build_summary.update(
            {
                "status": "TENSORRT_DDS_LIVE_OUTPUT_BUILD_PASSED",
                "engine_build_success": True,
                "serialized_engine_size_bytes": len(engine_bytes),
                "serialized_engine_sha256": sha256_bytes(engine_bytes),
                "serialized_engine_retained": False,
                "deserialization_attempted": False,
                "execution_context_created": False,
                "inference_attempted": False,
                "first_error": None,
                "hypothesis_result": (
                    "SUPPORTED: retaining all four size-tensor output-0 values "
                    "eliminated the prior DDS conversion assertion"
                ),
            }
        )
        phase4.dump_json(run_dir / "build_summary.json", build_summary)
        print(
            f"ENGINE_BUILD_END elapsed={elapsed:.6f} bytes={len(engine_bytes)}",
            flush=True,
        )
        print("TENSORRT_DDS_LIVE_OUTPUT_BUILD_PASSED", flush=True)
        return 0
    except Exception as error:
        build_summary.update(
            {
                "status": "TENSORRT_DDS_LIVE_OUTPUT_BUILD_FAILED",
                "first_error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        phase4.dump_json(run_dir / "parser_summary.json", parser_summary)
        phase4.dump_json(run_dir / "build_summary.json", build_summary)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        print("TENSORRT_DDS_LIVE_OUTPUT_BUILD_FAILED", flush=True)
        return 2


def parent(args: argparse.Namespace) -> int:
    source = args.onnx.resolve()
    plugin_library = args.plugin_library.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not plugin_library.is_file():
        raise FileNotFoundError(plugin_library)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.output_root.resolve() / f"{timestamp}_dds_live_output_test"
    run_dir.mkdir(parents=True, exist_ok=False)
    candidate = run_dir / "dds_live.onnx"
    rewrite_summary = rewrite_live_outputs(source, candidate)
    phase4.dump_json(run_dir / "rewrite_summary.json", rewrite_summary)

    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--run-dir",
        str(run_dir),
        "--onnx",
        str(candidate),
        "--plugin-library",
        str(plugin_library),
        "--tensorrt-root",
        str(args.tensorrt_root.resolve()),
        "--cuda-root",
        str(args.cuda_root.resolve()),
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
            summary_path = run_dir / "build_summary.json"
            summary = (
                json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.is_file()
                else {}
            )
            summary.update(
                {
                    "status": "TENSORRT_DDS_LIVE_OUTPUT_BUILD_FAILED",
                    "first_error": (
                        f"Builder worker exceeded {args.timeout_seconds} seconds"
                    ),
                    "timeout": True,
                    "execution_context_created": False,
                    "inference_attempted": False,
                }
            )
            phase4.dump_json(summary_path, summary)
            log.write("\nTENSORRT_DDS_LIVE_OUTPUT_BUILD_FAILED\n")
            return_code = 124

    build_summary_path = run_dir / "build_summary.json"
    if not build_summary_path.is_file():
        raise RuntimeError("Worker did not produce build_summary.json")
    build_summary = json.loads(build_summary_path.read_text(encoding="utf-8"))
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    native_errors = [
        line.strip()
        for line in log_text.splitlines()
        if "[TRT] [E]" in line or "Error Code" in line
    ]
    build_summary["native_tensorrt_errors"] = native_errors
    build_summary["prior_dds_assertion_reproduced"] = any(
        "nodeIdxToDDSOutputIndices.count(i) == nodeIdxToSizeTensors.count(i)"
        in line
        for line in native_errors
    )
    if not build_summary.get("engine_build_success"):
        build_summary["hypothesis_result"] = (
            "REJECTED_BY_CONTROLLED_EXPERIMENT: all four size-tensor outputs "
            "were retained as diagnostic graph outputs, but the identical DDS "
            "conversion assertion remained"
            if build_summary["prior_dds_assertion_reproduced"]
            else "INCONCLUSIVE: build failed with a different error"
        )
        build_summary["controlled_change_verified"] = {
            "voxel_count_output_0_live_for_all_four_nodes": True,
            "plugin_implementation_unchanged": True,
            "declare_size_tensor_unchanged": True,
            "plugin_output_order_unchanged": True,
            "original_logits_path_unchanged": True,
        }
    phase4.dump_json(build_summary_path, build_summary)

    print(f"RUN_DIR={run_dir}")
    print(f"WORKER_EXIT_CODE={return_code}")
    print(f"ONNX_CHECKER_PASSED={rewrite_summary['onnx_checker_passed']}")
    print(f"STATUS={build_summary['status']}")
    print(
        "PRIOR_DDS_ASSERTION_REPRODUCED="
        f"{build_summary['prior_dds_assertion_reproduced']}"
    )
    print(f"INFERENCE_ATTEMPTED={build_summary['inference_attempted']}")
    return return_code


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
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.worker and args.run_dir is None:
        parser.error("--worker requires --run-dir")
    return args


if __name__ == "__main__":
    options = parse_args()
    raise SystemExit(worker(options) if options.worker else parent(options))
