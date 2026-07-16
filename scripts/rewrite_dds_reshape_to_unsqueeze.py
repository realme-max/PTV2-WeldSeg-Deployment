"""Rewrite audited DDS Reshape nodes to Unsqueeze and build a structural FP32 engine.

Only nodes classified as A_EXACT_UNSQUEEZE_EQUIVALENT by
audit_dds_reshape_unsqueeze_candidates.py are changed.  The worker performs a
TensorRT parser/build/deserialization/inspection sequence but never creates an
execution context and never runs inference.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import audit_dds_reshape_unsqueeze_candidates as audit  # noqa: E402
import build_gcn_res_tensorrt_fp32 as phase4  # noqa: E402


DEFAULT_ONNX = audit.DEFAULT_ONNX
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
DEFAULT_PLUGIN_LIBRARY = phase4.DEFAULT_PLUGIN_LIBRARY
DEFAULT_TENSORRT_ROOT = phase4.DEFAULT_TENSORRT_ROOT
DEFAULT_CUDA_ROOT = phase4.DEFAULT_CUDA_ROOT
PLAN_NAME = "gcn_res_dds_reshape_fp32_b1_n2048.plan"
REWRITTEN_NAME = "dds_reshape_rewritten.onnx"


def node_stats(model: onnx.ModelProto) -> dict[str, Any]:
    counts = Counter(
        f"{node.domain or 'ai.onnx'}::{node.op_type}" for node in model.graph.node
    )
    return {
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "op_counts": dict(sorted(counts.items())),
    }


def contracts(model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
    values = (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    )
    return {value.name: audit.tensor_contract(value) for value in values}


def consumers(model: onnx.ModelProto, tensor: str) -> list[dict[str, Any]]:
    return [
        {"node_name": node.name, "op_type": node.op_type, "input_slot": slot}
        for node in model.graph.node
        for slot, input_name in enumerate(node.input)
        if input_name == tensor
    ]


def load_and_validate_audit(
    source: Path, audit_json: Path
) -> tuple[dict[str, Any], list[str]]:
    payload = json.loads(audit_json.read_text(encoding="utf-8"))
    if payload.get("status") != "DDS_RESHAPE_UNSQUEEZE_AUDIT_COMPLETED":
        raise RuntimeError(f"Audit status is not complete: {payload.get('status')}")
    if Path(payload["source_onnx"]).resolve() != source.resolve():
        raise RuntimeError("Audit source path does not match rewrite source")
    if payload.get("source_sha256") != phase4.sha256(source):
        raise RuntimeError("Audit source SHA-256 does not match rewrite source")
    candidates = [
        record["node_name"]
        for record in payload["records"]
        if record.get("classification") == audit.A and record.get("rewrite_allowed")
    ]
    if candidates != payload.get("rewrite_candidates"):
        raise RuntimeError("Audit candidate list is internally inconsistent")
    if not candidates:
        raise RuntimeError("Audit contains no authorized rewrite candidates")

    # Re-run the read-only proof on the exact source to prevent a stale or
    # manually edited JSON file from authorizing a graph mutation.
    model = onnx.load_model(str(source), load_external_data=False)
    live_summary, _ = audit.audit_model(model, source)
    if live_summary["rewrite_candidates"] != candidates:
        raise RuntimeError(
            "Live equivalence proof differs from the saved audit: "
            f"{live_summary['rewrite_candidates']} != {candidates}"
        )
    return payload, candidates


def rewrite_model(
    source: Path, destination: Path, candidates: list[str]
) -> dict[str, Any]:
    source_model = onnx.load_model(str(source), load_external_data=False)
    model = onnx.ModelProto()
    model.CopyFrom(source_model)
    source_contracts = contracts(source_model)
    source_nodes = {node.name: node for node in source_model.graph.node}
    candidate_set = set(candidates)
    replacements: list[dict[str, Any]] = []
    existing_initializer_names = {item.name for item in model.graph.initializer}

    for index, node in enumerate(model.graph.node):
        if node.name not in candidate_set:
            continue
        original = source_nodes[node.name]
        if original.op_type != "Reshape" or len(original.input) != 2:
            raise RuntimeError(f"Authorized node contract changed: {node.name}")
        axes_name = f"{node.name}_TensorRT_Unsqueeze_axes"
        if axes_name in existing_initializer_names:
            raise RuntimeError(f"Axes initializer already exists: {axes_name}")
        axes = numpy_helper.from_array(np.asarray([0], dtype=np.int64), name=axes_name)
        model.graph.initializer.append(axes)
        existing_initializer_names.add(axes_name)
        replacement = helper.make_node(
            "Unsqueeze",
            inputs=[original.input[0], axes_name],
            outputs=list(original.output),
            name=original.name,
        )
        del model.graph.node[index]
        model.graph.node.insert(index, replacement)
        output = original.output[0]
        replacements.append(
            {
                "node_index": index,
                "node_name": original.name,
                "old_op_type": "Reshape",
                "new_op_type": "Unsqueeze",
                "old_inputs": list(original.input),
                "new_inputs": [original.input[0], axes_name],
                "outputs": list(original.output),
                "axes_initializer": axes_name,
                "axes_dtype": "INT64",
                "axes_value": [0],
                "output_contract_before": source_contracts.get(output),
                "consumers_before": consumers(source_model, output),
            }
        )

    if {item["node_name"] for item in replacements} != candidate_set:
        missing = candidate_set - {item["node_name"] for item in replacements}
        raise RuntimeError(f"Authorized nodes missing from graph: {sorted(missing)}")
    onnx.save_model(model, str(destination))
    return {"replacements": replacements}


def validate_graph_diff(
    source: Path, candidate: Path, candidates: list[str], rewrite: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = onnx.load_model(str(source), load_external_data=False)
    rewritten = onnx.load_model(str(candidate), load_external_data=False)
    onnx.checker.check_model(rewritten, full_check=True)

    original_nodes = list(original.graph.node)
    rewritten_nodes = list(rewritten.graph.node)
    if len(original_nodes) != len(rewritten_nodes):
        raise RuntimeError("Node count changed during one-for-one rewrite")
    changed: list[str] = []
    for before, after in zip(original_nodes, rewritten_nodes):
        if before.SerializeToString() != after.SerializeToString():
            changed.append(before.name)
    if changed != candidates:
        raise RuntimeError(f"Changed node set/order differs from audit: {changed}")

    original_contracts = contracts(original)
    rewritten_contracts = contracts(rewritten)
    contract_results: list[dict[str, Any]] = []
    for replacement in rewrite["replacements"]:
        output = replacement["outputs"][0]
        before_contract = original_contracts.get(output)
        after_contract = rewritten_contracts.get(output)
        after_consumers = consumers(rewritten, output)
        contract_ok = (
            before_contract == after_contract
            and replacement["consumers_before"] == after_consumers
            and before_contract is not None
            and before_contract["dtype"] == "FLOAT"
            and before_contract["rank"] == 3
            and before_contract["shape"][0] == 1
        )
        contract_results.append(
            {
                "node_name": replacement["node_name"],
                "output": output,
                "before_contract": before_contract,
                "after_contract": after_contract,
                "consumers_before": replacement["consumers_before"],
                "consumers_after": after_consumers,
                "contract_and_consumers_preserved": contract_ok,
            }
        )
    if not all(item["contract_and_consumers_preserved"] for item in contract_results):
        raise RuntimeError("One or more rewritten output contracts/consumers changed")

    original_initializers = {item.name for item in original.graph.initializer}
    rewritten_initializers = {item.name for item in rewritten.graph.initializer}
    added_initializers = sorted(rewritten_initializers - original_initializers)
    expected_initializers = sorted(
        item["axes_initializer"] for item in rewrite["replacements"]
    )
    if added_initializers != expected_initializers:
        raise RuntimeError("Unexpected initializer change")
    if not original_initializers.issubset(rewritten_initializers):
        raise RuntimeError("Original initializer was removed")

    # Shape inference is a validation artifact only; the inferred model is not
    # used for TensorRT parsing and is intentionally not saved as the candidate.
    inferred = shape_inference.infer_shapes(
        rewritten, check_type=True, strict_mode=True, data_prop=True
    )
    inferred_contracts = contracts(inferred)
    inferred_results = []
    for replacement in rewrite["replacements"]:
        output = replacement["outputs"][0]
        expected = original_contracts[output]
        actual = inferred_contracts.get(output)
        passed = (
            actual is not None
            and actual["dtype"] == expected["dtype"]
            and actual["rank"] == 3
            and actual["shape"][0] == 1
            and actual["shape"][-1] == expected["shape"][-1]
        )
        inferred_results.append(
            {
                "node_name": replacement["node_name"],
                "expected_contract": expected,
                "inferred_contract": actual,
                "shape_inference_contract_passed": passed,
            }
        )
    if not all(item["shape_inference_contract_passed"] for item in inferred_results):
        raise RuntimeError("Shape inference contract validation failed")

    source_stats = node_stats(original)
    rewritten_stats = node_stats(rewritten)
    graph_diff = {
        "source_onnx": str(source),
        "source_sha256": phase4.sha256(source),
        "rewritten_onnx": str(candidate),
        "rewritten_sha256": phase4.sha256(candidate),
        "source_stats": source_stats,
        "rewritten_stats": rewritten_stats,
        "changed_nodes": changed,
        "changed_node_count": len(changed),
        "only_audited_nodes_changed": changed == candidates,
        "added_initializers": added_initializers,
        "removed_initializers": sorted(original_initializers - rewritten_initializers),
        "node_count_preserved": source_stats["node_count"] == rewritten_stats["node_count"],
        "graph_inputs_preserved": [
            item.SerializeToString() for item in original.graph.input
        ]
        == [item.SerializeToString() for item in rewritten.graph.input],
        "graph_outputs_preserved": [
            item.SerializeToString() for item in original.graph.output
        ]
        == [item.SerializeToString() for item in rewritten.graph.output],
        "value_info_preserved": [
            item.SerializeToString() for item in original.graph.value_info
        ]
        == [item.SerializeToString() for item in rewritten.graph.value_info],
        "output_contract_and_consumer_checks": contract_results,
        "unused_shape_construction_chains_removed": False,
        "other_nodes_modified": False,
    }
    checker = {
        "onnx_checker_passed": True,
        "onnx_full_check": True,
        "shape_inference_passed": True,
        "shape_inference_results": inferred_results,
        "all_rewritten_contracts_passed": all(
            item["shape_inference_contract_passed"] for item in inferred_results
        ),
    }
    return graph_diff, checker


def initial_build_summary(workspace_bytes: int) -> dict[str, Any]:
    return {
        "status": "TENSORRT_DDS_RESHAPE_REWRITE_ENGINE_BUILD_FAILED",
        "parser_success": False,
        "parser_error_count": None,
        "parser_errors": [],
        "engine_build_attempted": False,
        "engine_build_success": False,
        "build_elapsed_seconds": None,
        "serialized_engine_size_bytes": None,
        "serialized_engine_generated": False,
        "plan_saved": False,
        "plan_path": None,
        "plan_sha256": None,
        "deserialize_success": False,
        "io_validation_passed": False,
        "engine_inspector_voxel_unique_instances": None,
        "voxel_unique_build_instances": None,
        "voxel_unique_runtime_instances": None,
        "workspace_bytes": workspace_bytes,
        "fp16_enabled": False,
        "int8_enabled": False,
        "execution_context_created": False,
        "inference_attempted": False,
        "first_error": None,
        "first_tensorrt_native_error": None,
        "dds_assertion_present": None,
        "traceback": None,
    }


def network_tensor_record(tensor: Any) -> dict[str, Any]:
    return {
        "name": tensor.name,
        "dtype": phase4.enum_name(tensor.dtype),
        "shape": phase4.dims_list(tensor.shape),
    }


def worker(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    onnx_path = args.onnx.resolve()
    plugin_path = args.plugin_library.resolve()
    workspace_bytes = int(args.workspace_gib * 1024**3)
    summary = initial_build_summary(workspace_bytes)
    phase4.dump_json(run_dir / "build_summary.json", summary)
    dll_handles = []
    try:
        if not onnx_path.is_file() or not plugin_path.is_file():
            raise FileNotFoundError(f"ONNX={onnx_path}, plugin={plugin_path}")
        for directory in (
            args.tensorrt_root.resolve() / "bin",
            args.cuda_root.resolve() / "bin",
            plugin_path.parent,
        ):
            if hasattr(os, "add_dll_directory"):
                dll_handles.append(os.add_dll_directory(str(directory)))

        import tensorrt as trt

        phase4.dump_json(
            run_dir / "environment.json",
            phase4.collect_environment(
                trt,
                onnx_path,
                plugin_path,
                args.tensorrt_root.resolve(),
                args.cuda_root.resolve(),
            ),
        )
        logger = trt.Logger(trt.Logger.VERBOSE)
        standard_initialized = bool(trt.init_libnvinfer_plugins(logger, ""))
        if not standard_initialized:
            raise RuntimeError("trt.init_libnvinfer_plugins returned false")
        plugin_library, plugin_info = phase4.load_plugin_library(plugin_path)
        if not plugin_info["registration_function_returned"]:
            raise RuntimeError("initVoxelUniquePlugin returned false")
        registry_payload = phase4.collect_registry(trt, trt.get_plugin_registry())
        registry_payload["plugin_library"] = plugin_info
        phase4.dump_json(run_dir / "plugin_registry.json", registry_payload)
        if not registry_payload["voxel_unique_creator_found"]:
            raise RuntimeError("VoxelUnique Plugin Creator was not registered")
        if not registry_payload["scatter_elements_v2_creator_found"]:
            raise RuntimeError("TensorRT standard ScatterElements plugin is missing")

        builder = trt.Builder(logger)
        explicit = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
        flags = 1 << int(explicit) if explicit is not None else 0
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)
        config = builder.create_builder_config()
        if builder is None or network is None or parser is None or config is None:
            raise RuntimeError("TensorRT Builder/Network/Parser/Config creation failed")
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        for name in (
            "FP16",
            "INT8",
            "SPARSE_WEIGHTS",
            "REFIT",
            "VERSION_COMPATIBLE",
            "WEIGHT_STREAMING",
        ):
            if hasattr(trt.BuilderFlag, name):
                config.clear_flag(getattr(trt.BuilderFlag, name))
        phase4.dump_json(
            run_dir / "build_config.json",
            {
                "precision": "FP32",
                "workspace_gib": args.workspace_gib,
                "workspace_bytes": workspace_bytes,
                "fixed_shapes": {"points": [1, 2048, 4], "adj": [1, 2048, 2048]},
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
                "execution_context_planned": False,
                "inference_planned": False,
            },
        )

        print("PARSER_BEGIN", flush=True)
        parse_success = bool(parser.parse_from_file(str(onnx_path)))
        parser_errors = phase4.parser_errors(parser)
        summary["parser_success"] = parse_success
        summary["parser_error_count"] = len(parser_errors)
        summary["parser_errors"] = parser_errors
        parser_summary = {
            "parser_success": parse_success,
            "parser_error_count": len(parser_errors),
            "parser_errors": parser_errors,
            "num_layers": int(network.num_layers),
            "num_inputs": int(network.num_inputs),
            "num_outputs": int(network.num_outputs),
            "inputs": [
                network_tensor_record(network.get_input(index))
                for index in range(network.num_inputs)
            ],
            "outputs": [
                network_tensor_record(network.get_output(index))
                for index in range(network.num_outputs)
            ],
        }
        phase4.dump_json(run_dir / "parser_summary.json", parser_summary)
        phase4.dump_json(run_dir / "build_summary.json", summary)
        print(f"PARSER_END success={parse_success} errors={len(parser_errors)}", flush=True)
        if not parse_success or parser_errors:
            raise RuntimeError(f"TensorRT parser failed: {parser_errors[:1]}")

        build_instances = int(plugin_library.getVoxelUniqueBuildCreationCount())
        summary["voxel_unique_build_instances"] = build_instances
        if build_instances != 4:
            raise RuntimeError(f"Expected 4 VoxelUnique build instances, got {build_instances}")
        summary["engine_build_attempted"] = True
        phase4.dump_json(run_dir / "build_summary.json", summary)

        print(f"ENGINE_BUILD_BEGIN workspace_bytes={workspace_bytes}", flush=True)
        started = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        elapsed = time.perf_counter() - started
        summary["build_elapsed_seconds"] = elapsed
        if serialized is None:
            raise RuntimeError("build_serialized_network returned None")
        engine_bytes = bytes(serialized)
        summary["engine_build_success"] = True
        summary["serialized_engine_generated"] = True
        summary["serialized_engine_size_bytes"] = len(engine_bytes)
        print(f"ENGINE_BUILD_END elapsed={elapsed:.6f} bytes={len(engine_bytes)}", flush=True)

        plan_path = run_dir / PLAN_NAME
        temporary = plan_path.with_suffix(plan_path.suffix + ".tmp")
        temporary.write_bytes(engine_bytes)
        temporary.replace(plan_path)
        plan_hash = phase4.sha256(plan_path)
        (run_dir / "engine_sha256.txt").write_text(
            f"{plan_hash}  {plan_path.name}\n", encoding="utf-8"
        )
        summary.update(
            {
                "plan_saved": True,
                "plan_path": str(plan_path),
                "plan_sha256": plan_hash,
            }
        )

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError("Runtime.deserialize_cuda_engine returned None")
        summary["deserialize_success"] = True
        runtime_instances = int(plugin_library.getVoxelUniqueRuntimeCreationCount())
        summary["voxel_unique_runtime_instances"] = runtime_instances
        io_records, _, inspector_count = phase4.inspect_engine(trt, engine, run_dir)
        phase4.validate_io(io_records)
        summary["io_validation_passed"] = True
        summary["engine_inspector_voxel_unique_instances"] = inspector_count
        if runtime_instances != 4:
            raise RuntimeError(f"Expected 4 runtime plugin instances, got {runtime_instances}")
        if inspector_count != 4:
            raise RuntimeError(
                f"Engine Inspector found {inspector_count}, not 4, VoxelUnique layers"
            )

        summary.update(
            {
                "status": "TENSORRT_FP32_ENGINE_BUILD_PASSED",
                "num_io_tensors": int(engine.num_io_tensors),
                "num_layers": int(engine.num_layers),
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
                "execution_context_created": False,
                "inference_attempted": False,
                "first_error": None,
                "dds_assertion_present": False,
            }
        )
        phase4.dump_json(run_dir / "build_summary.json", summary)
        print("TENSORRT_DDS_RESHAPE_REWRITE_ENGINE_BUILD_PASSED", flush=True)
        print("TENSORRT_FP32_ENGINE_BUILD_PASSED", flush=True)
        return 0
    except Exception as error:
        summary["first_error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        summary["status"] = "TENSORRT_DDS_RESHAPE_REWRITE_ENGINE_BUILD_FAILED"
        phase4.dump_json(run_dir / "build_summary.json", summary)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        print("TENSORRT_DDS_RESHAPE_REWRITE_ENGINE_BUILD_FAILED", flush=True)
        return 2


def extract_native_failure(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    lines = text.splitlines()
    markers = (
        "Assertion failed",
        "Error Code",
        "INTERNAL_ERROR",
        "UNSUPPORTED_NODE",
        "build_serialized_network returned None",
    )
    native = next((line.strip() for line in lines if any(marker in line for marker in markers)), None)
    dds = (
        "convertExplicitDDSPluginToImplicit" in text
        or "nodeIdxToDDSOutputIndices" in text
        or "nodeIdxToSizeTensors" in text
    )
    return {
        "first_tensorrt_native_error": native,
        "dds_assertion_present": dds,
        "log_size_bytes": log_path.stat().st_size if log_path.is_file() else 0,
    }


def run_worker(args: argparse.Namespace, run_dir: Path, candidate: Path) -> dict[str, Any]:
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
        str(args.plugin_library.resolve()),
        "--tensorrt-root",
        str(args.tensorrt_root.resolve()),
        "--cuda-root",
        str(args.cuda_root.resolve()),
        "--workspace-gib",
        str(args.workspace_gib),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    environment = os.environ.copy()
    environment["TENSORRT_ROOT"] = str(args.tensorrt_root.resolve())
    environment["CUDA_PATH"] = str(args.cuda_root.resolve())
    environment["PATH"] = os.pathsep.join(
        [
            str(args.tensorrt_root.resolve() / "bin"),
            str(args.cuda_root.resolve() / "bin"),
            environment.get("PATH", ""),
        ]
    )
    log_path = run_dir / "builder_verbose.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write("COMMAND=" + subprocess.list2cmdline(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=environment,
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
                else initial_build_summary(int(args.workspace_gib * 1024**3))
            )
            summary.update(
                {
                    "status": "TENSORRT_DDS_RESHAPE_REWRITE_ENGINE_BUILD_FAILED",
                    "first_error": f"Builder exceeded {args.timeout_seconds}s outer timeout",
                    "timeout": True,
                    "execution_context_created": False,
                    "inference_attempted": False,
                }
            )
            phase4.dump_json(summary_path, summary)
            log.write("\nBUILDER_OUTER_TIMEOUT\n")
            return_code = 124
    summary_path = run_dir / "build_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    native = extract_native_failure(log_path)
    summary.update(native)
    summary["worker_exit_code"] = return_code
    if return_code == 0 and summary.get("status") == "TENSORRT_FP32_ENGINE_BUILD_PASSED":
        summary["rewrite_build_status"] = "TENSORRT_DDS_RESHAPE_REWRITE_ENGINE_BUILD_PASSED"
    else:
        summary["status"] = "TENSORRT_DDS_RESHAPE_REWRITE_ENGINE_BUILD_FAILED"
        summary["rewrite_build_status"] = summary["status"]
    phase4.dump_json(summary_path, summary)
    return summary


def write_report(
    run_dir: Path,
    audit_payload: dict[str, Any],
    rewrite_summary: dict[str, Any],
    graph_diff: dict[str, Any],
    checker: dict[str, Any],
    build: dict[str, Any],
) -> None:
    replacement_rows = "\n".join(
        f"| `{item['node_name']}` | `{item['old_op_type']}` | `{item['new_op_type']}` | "
        f"`{item['output_contract_before']['shape']}` |"
        for item in rewrite_summary["replacements"]
    )
    report = f"""# DDS Reshape → Unsqueeze controlled rewrite and TensorRT build

## Sources

- Formal input: `{rewrite_summary['source_onnx']}`
- Formal SHA-256 before/after: `{rewrite_summary['source_sha256_before']}` / `{rewrite_summary['source_sha256_after']}`
- Formal source unchanged: `{rewrite_summary['source_onnx_unchanged']}`
- Equivalence audit: `{rewrite_summary['audit_json']}`
- Audit candidates: `{audit_payload['rewrite_candidate_count']}`
- Derived ONNX: `{rewrite_summary['rewritten_onnx']}`
- Derived SHA-256: `{rewrite_summary['rewritten_sha256']}`

## Controlled changes

| Node | Before | After | Preserved output shape metadata |
|---|---|---|---|
{replacement_rows}

- Only audited A nodes changed: `{graph_diff['only_audited_nodes_changed']}`
- Node count preserved: `{graph_diff['node_count_preserved']}`
- Graph inputs/outputs/value_info preserved: `{graph_diff['graph_inputs_preserved']}` / `{graph_diff['graph_outputs_preserved']}` / `{graph_diff['value_info_preserved']}`
- Unused shape chains removed: `{graph_diff['unused_shape_construction_chains_removed']}`
- ONNX checker: `{checker['onnx_checker_passed']}`
- Shape inference contracts: `{checker['all_rewritten_contracts_passed']}`

## TensorRT parser and FP32 builder

- Parser success: `{build.get('parser_success')}`
- Parser errors: `{build.get('parser_error_count')}`
- Engine build success: `{build.get('engine_build_success')}`
- Builder elapsed seconds: `{build.get('build_elapsed_seconds')}`
- Serialized engine generated: `{build.get('serialized_engine_generated')}`
- First TensorRT native error: `{build.get('first_tensorrt_native_error')}`
- DDS assertion present: `{build.get('dds_assertion_present')}`
- Plan saved: `{build.get('plan_saved')}`
- Plan SHA-256: `{build.get('plan_sha256')}`
- Deserialization: `{build.get('deserialize_success')}`
- I/O validation: `{build.get('io_validation_passed')}`
- Build/runtime/Inspector VoxelUnique instances: `{build.get('voxel_unique_build_instances')}` / `{build.get('voxel_unique_runtime_instances')}` / `{build.get('engine_inspector_voxel_unique_instances')}`
- Execution context created: `{build.get('execution_context_created')}`
- Inference attempted: `{build.get('inference_attempted')}`
- FP16 / INT8: `{build.get('fp16_enabled')}` / `{build.get('int8_enabled')}`

## Status

`{build.get('rewrite_build_status')}`

{('`TENSORRT_FP32_ENGINE_BUILD_PASSED`' if build.get('status') == 'TENSORRT_FP32_ENGINE_BUILD_PASSED' else '')}
"""
    (run_dir / "graph_diff_report.md").write_text(report, encoding="utf-8")


def parent(args: argparse.Namespace) -> int:
    source = args.onnx.resolve()
    audit_json = args.audit_json.resolve()
    if not source.is_file() or not audit_json.is_file():
        raise FileNotFoundError(f"source={source}, audit={audit_json}")
    if not args.plugin_library.resolve().is_file():
        raise FileNotFoundError(args.plugin_library.resolve())
    source_hash_before = phase4.sha256(source)
    if source_hash_before != audit.EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Formal if_folded.onnx SHA-256 changed")
    audit_payload, candidates = load_and_validate_audit(source, audit_json)

    run_dir = (
        args.output_root.resolve()
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_dds_reshape_rewrite"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    candidate = run_dir / REWRITTEN_NAME
    rewrite = rewrite_model(source, candidate, candidates)
    graph_diff, checker = validate_graph_diff(source, candidate, candidates, rewrite)
    source_hash_after = phase4.sha256(source)
    rewrite_summary = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "source_onnx": str(source),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_onnx_unchanged": source_hash_before == source_hash_after,
        "audit_json": str(audit_json),
        "audit_json_sha256": phase4.sha256(audit_json),
        "authorized_classification": audit.A,
        "authorized_candidate_count": len(candidates),
        "authorized_candidates": candidates,
        "rewritten_onnx": str(candidate),
        "rewritten_sha256": phase4.sha256(candidate),
        "replacements": rewrite["replacements"],
        "unused_shape_chains_retained": True,
        "other_nodes_modified": False,
    }
    if not rewrite_summary["source_onnx_unchanged"]:
        raise RuntimeError("Formal input changed during rewrite")
    phase4.dump_json(run_dir / "rewrite_summary.json", rewrite_summary)
    phase4.dump_json(run_dir / "onnx_check_result.json", checker)
    phase4.dump_json(run_dir / "graph_diff.json", graph_diff)

    print(f"RUN_DIR={run_dir}", flush=True)
    print(f"REWRITTEN_ONNX={candidate}", flush=True)
    print(f"REPLACED_NODES={len(candidates)}", flush=True)
    print("ONNX_CHECK_AND_SHAPE_INFERENCE_PASSED", flush=True)
    build = run_worker(args, run_dir, candidate)
    write_report(run_dir, audit_payload, rewrite_summary, graph_diff, checker, build)
    print(f"PARSER_ERRORS={build.get('parser_error_count')}")
    print(f"BUILD_ELAPSED_SECONDS={build.get('build_elapsed_seconds')}")
    print(f"DDS_ASSERTION_PRESENT={build.get('dds_assertion_present')}")
    print(build["rewrite_build_status"])
    if build.get("status") == "TENSORRT_FP32_ENGINE_BUILD_PASSED":
        print("TENSORRT_FP32_ENGINE_BUILD_PASSED")
        return 0
    return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--audit-json", type=Path, required=not "--worker" in sys.argv)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plugin-library", type=Path, default=DEFAULT_PLUGIN_LIBRARY)
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TENSORRT_ROOT)
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA_ROOT)
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.worker and args.run_dir is None:
        parser.error("--run-dir is required in worker mode")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(worker(arguments) if arguments.worker else parent(arguments))
