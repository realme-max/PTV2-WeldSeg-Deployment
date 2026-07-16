"""Isolate TensorRT VoxelUnique DDS instance-count and cascade behavior.

This diagnostic creates two standalone ONNX graphs without editing the formal
GCN_res ONNX or the VoxelUnique plugin implementation:

1. one VoxelUnique node with a fixed INT64[2048] input (the tdb_1 contract);
2. only if (1) builds, four independent VoxelUnique nodes, each with its own
   fixed INT64[2048] input.  No output of one plugin feeds another plugin.

Each graph is parsed and passed to the FP32 TensorRT builder.  Serialized
engine bytes are hashed in memory and deliberately not retained.  The script
never deserializes an engine, creates an execution context, or runs inference.
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
from collections import deque
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
PLUGIN_SOURCE = (
    PROJECT_ROOT
    / "tests"
    / "tensorrt_voxel_unique_correctness"
    / "VoxelUniqueCorrectnessPlugin.cu"
)
EXPECTED_PLUGIN_NODES = [f"/model/tdb_{index}/Unique" for index in range(1, 5)]
PLUGIN_DOMAIN = "com.tensorrt.ptv2"
PLUGIN_OP = "VoxelUnique"
PLUGIN_VERSION = "1"
FIXED_N = 2048


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def dim_value(dim: Any) -> int | str | None:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    if dim.HasField("dim_param"):
        return str(dim.dim_param)
    return None


def value_contracts(model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    for value in values:
        tensor_type = value.type.tensor_type
        records[value.name] = {
            "dtype": TensorProto.DataType.Name(tensor_type.elem_type),
            "shape": [dim_value(dim) for dim in tensor_type.shape.dim],
        }
    for initializer in model.graph.initializer:
        records.setdefault(
            initializer.name,
            {
                "dtype": TensorProto.DataType.Name(initializer.data_type),
                "shape": [int(value) for value in initializer.dims],
            },
        )
    return records


def get_plugin_nodes(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    nodes = [
        node
        for node in model.graph.node
        if node.op_type == PLUGIN_OP and node.domain == PLUGIN_DOMAIN
    ]
    if [node.name for node in nodes] != EXPECTED_PLUGIN_NODES:
        raise RuntimeError(
            f"Unexpected VoxelUnique nodes: {[node.name for node in nodes]}"
        )
    for node in nodes:
        if len(node.input) != 1 or len(node.output) != 3:
            raise RuntimeError(f"Unexpected VoxelUnique I/O contract: {node.name}")
    return nodes


def make_plugin_node(name: str, input_name: str, output_names: list[str]) -> onnx.NodeProto:
    return helper.make_node(
        PLUGIN_OP,
        inputs=[input_name],
        outputs=output_names,
        name=name,
        domain=PLUGIN_DOMAIN,
        plugin_version=PLUGIN_VERSION,
        plugin_namespace=PLUGIN_DOMAIN,
    )


def create_diagnostic_model(instance_count: int, path: Path) -> dict[str, Any]:
    if instance_count not in (1, 4):
        raise ValueError(instance_count)
    inputs = []
    outputs = []
    nodes = []
    instance_records = []
    for index in range(1, instance_count + 1):
        input_name = f"keys_{index}"
        count_name = f"voxel_count_{index}"
        values_name = f"unique_values_{index}"
        inverse_name = f"inverse_indices_{index}"
        node_name = f"/diagnostic/tdb_{index}/Unique"
        inputs.append(helper.make_tensor_value_info(input_name, TensorProto.INT64, [FIXED_N]))
        outputs.extend(
            [
                helper.make_tensor_value_info(count_name, TensorProto.INT32, []),
                helper.make_tensor_value_info(values_name, TensorProto.INT64, [f"M_{index}"]),
                helper.make_tensor_value_info(inverse_name, TensorProto.INT64, [FIXED_N]),
            ]
        )
        nodes.append(
            make_plugin_node(
                node_name,
                input_name,
                [count_name, values_name, inverse_name],
            )
        )
        instance_records.append(
            {
                "instance_index": index,
                "node_name": node_name,
                "input": {"name": input_name, "dtype": "INT64", "shape": [FIXED_N]},
                "outputs": [
                    {"index": 0, "name": count_name, "dtype": "INT32", "shape": []},
                    {"index": 1, "name": values_name, "dtype": "INT64", "shape": [f"M_{index}"]},
                    {"index": 2, "name": inverse_name, "dtype": "INT64", "shape": [FIXED_N]},
                ],
                "depends_on_previous_dds": False,
            }
        )
    graph = helper.make_graph(
        nodes,
        f"VoxelUniqueDDSIsolation{instance_count}",
        inputs,
        outputs,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid(PLUGIN_DOMAIN, 1),
        ],
        producer_name="PTV2-WeldSeg-DDS-isolation",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save_model(model, str(path))
    onnx.checker.check_model(onnx.load_model(str(path), load_external_data=False))
    return {
        "path": str(path),
        "sha256": phase4.sha256(path),
        "onnx_checker_passed": True,
        "instance_count": instance_count,
        "instances": instance_records,
        "all_plugin_inputs_fixed_n": True,
        "dds_cascade_present": False,
        "formal_model_computation_reproduced": False,
        "purpose": (
            "Isolate DDS plugin conversion mechanics; this graph is not a model "
            "equivalence candidate and is never used for inference."
        ),
    }


def create_direct_cascade_model(depth: int, path: Path) -> dict[str, Any]:
    """Create a minimal direct DDS cascade for locating the first boundary.

    The first plugin receives fixed INT64[2048].  Every later plugin consumes
    the preceding plugin's unique_values[M] output directly.  All outputs stay
    graph outputs so this experiment is not confounded by output liveness.
    """
    if depth < 2:
        raise ValueError("A DDS cascade requires at least two plugin instances")
    graph_inputs = [
        helper.make_tensor_value_info("keys_1", TensorProto.INT64, [FIXED_N])
    ]
    graph_outputs = []
    nodes = []
    instance_records = []
    input_name = "keys_1"
    input_shape: list[int | str] = [FIXED_N]
    for index in range(1, depth + 1):
        count_name = f"cascade_voxel_count_{index}"
        values_name = f"cascade_unique_values_{index}"
        inverse_name = f"cascade_inverse_indices_{index}"
        runtime_m = f"cascade_M_{index}"
        node_name = f"/diagnostic/cascade/tdb_{index}/Unique"
        nodes.append(
            make_plugin_node(
                node_name,
                input_name,
                [count_name, values_name, inverse_name],
            )
        )
        graph_outputs.extend(
            [
                helper.make_tensor_value_info(count_name, TensorProto.INT32, []),
                helper.make_tensor_value_info(values_name, TensorProto.INT64, [runtime_m]),
                helper.make_tensor_value_info(inverse_name, TensorProto.INT64, input_shape),
            ]
        )
        instance_records.append(
            {
                "instance_index": index,
                "node_name": node_name,
                "input": {"name": input_name, "dtype": "INT64", "shape": input_shape},
                "depends_on_previous_dds": index > 1,
                "previous_dds_output": None if index == 1 else f"cascade_unique_values_{index - 1}",
                "outputs": {
                    "size_tensor": count_name,
                    "dds_values": {"name": values_name, "shape": [runtime_m]},
                    "inverse_indices": {"name": inverse_name, "shape": input_shape},
                },
            }
        )
        input_name = values_name
        input_shape = [runtime_m]
    graph = helper.make_graph(
        nodes,
        f"VoxelUniqueDirectDDSCascade{depth}",
        graph_inputs,
        graph_outputs,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid(PLUGIN_DOMAIN, 1),
        ],
        producer_name="PTV2-WeldSeg-DDS-cascade-isolation",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save_model(model, str(path))
    onnx.checker.check_model(onnx.load_model(str(path), load_external_data=False))
    return {
        "path": str(path),
        "sha256": phase4.sha256(path),
        "onnx_checker_passed": True,
        "instance_count": depth,
        "instances": instance_records,
        "dds_cascade_present": True,
        "first_dds_dependent_instance": 2,
        "all_outputs_retained_as_graph_outputs": True,
        "formal_model_computation_reproduced": False,
        "purpose": (
            "Locate the first TensorRT DDS-on-DDS conversion boundary; this graph "
            "is not a model equivalence candidate and is never used for inference."
        ),
    }


def shortest_upstream_plugin_path(
    tensor_name: str,
    producer: dict[str, onnx.NodeProto],
    current_node: str,
) -> dict[str, Any] | None:
    queue: deque[tuple[str, list[dict[str, Any]]]] = deque([(tensor_name, [])])
    visited: set[str] = set()
    while queue:
        tensor, path = queue.popleft()
        if tensor in visited:
            continue
        visited.add(tensor)
        node = producer.get(tensor)
        if node is None:
            continue
        step = {
            "tensor": tensor,
            "producer_node": node.name,
            "producer_op_type": node.op_type,
            "producer_domain": node.domain,
        }
        new_path = path + [step]
        if (
            node.name != current_node
            and node.op_type == PLUGIN_OP
            and node.domain == PLUGIN_DOMAIN
        ):
            try:
                output_index = list(node.output).index(tensor)
            except ValueError:
                output_index = None
            return {
                "upstream_plugin": node.name,
                "upstream_plugin_output_tensor": tensor,
                "upstream_plugin_output_index": output_index,
                "path_from_current_input_to_upstream_plugin": new_path,
                "edge_count": len(new_path),
            }
        for input_name in node.input:
            if input_name:
                queue.append((input_name, new_path))
    return None


def upstream_tensor_path(
    destination_tensor: str,
    source_tensor: str,
    producer: dict[str, onnx.NodeProto],
) -> list[dict[str, Any]] | None:
    """Return a shortest backward dependency path to one exact source tensor."""
    queue: deque[tuple[str, list[dict[str, Any]]]] = deque(
        [(destination_tensor, [])]
    )
    visited: set[str] = set()
    while queue:
        tensor, path = queue.popleft()
        if tensor in visited:
            continue
        visited.add(tensor)
        if tensor == source_tensor:
            return path
        node = producer.get(tensor)
        if node is None:
            continue
        step = {
            "tensor": tensor,
            "producer_node": node.name,
            "producer_op_type": node.op_type,
            "producer_domain": node.domain,
        }
        for input_name in node.input:
            if input_name:
                queue.append((input_name, path + [step]))
    return None


def trace_formal_chain(model: onnx.ModelProto) -> dict[str, Any]:
    contracts = value_contracts(model)
    nodes = get_plugin_nodes(model)
    producer = {
        output: node for node in model.graph.node for output in node.output if output
    }
    records = []
    for index, node in enumerate(nodes, start=1):
        input_name = node.input[0]
        contract = contracts.get(input_name, {"dtype": None, "shape": None})
        upstream = shortest_upstream_plugin_path(input_name, producer, node.name)
        runtime_m_dependency = None
        if index > 1:
            previous_values = nodes[index - 2].output[1]
            runtime_m_path = upstream_tensor_path(input_name, previous_values, producer)
            if runtime_m_path is not None:
                runtime_m_dependency = {
                    "previous_plugin": nodes[index - 2].name,
                    "previous_dds_values_output_index": 1,
                    "previous_dds_values_tensor": previous_values,
                    "path_from_current_input_to_previous_dds_values": runtime_m_path,
                    "edge_count": len(runtime_m_path),
                    "contains_shape_operator": any(
                        step["producer_op_type"] == "Shape" for step in runtime_m_path
                    ),
                }
        is_static = contract.get("shape") == [FIXED_N]
        if is_static and upstream is None:
            source = "STATIC_N"
        elif runtime_m_dependency is not None:
            source = "PREVIOUS_DDS_M"
        elif upstream is not None:
            source = "PREVIOUS_VOXELUNIQUE_DATA_DEPENDENCY"
        else:
            source = "DYNAMIC_OTHER"
        records.append(
            {
                "instance_index": index,
                "node_name": node.name,
                "input_tensor": input_name,
                "input_contract": contract,
                "input_shape_source": source,
                "depends_on_previous_voxelunique": upstream is not None,
                "nearest_upstream_voxelunique": upstream,
                "runtime_m_shape_dependency": runtime_m_dependency,
                "dds_output_contracts": [
                    {
                        "output_index": output_index,
                        "tensor": output_name,
                        "contract": contracts.get(output_name),
                        "role": ("size_tensor" if output_index == 0 else ("dds_values" if output_index == 1 else "inverse_indices")),
                    }
                    for output_index, output_name in enumerate(node.output)
                ],
            }
        )
    first_chained = next(
        (record["node_name"] for record in records if record["depends_on_previous_voxelunique"]),
        None,
    )
    return {
        "formal_onnx": str(DEFAULT_ONNX),
        "formal_onnx_sha256": phase4.sha256(DEFAULT_ONNX),
        "instances": records,
        "first_structurally_chained_instance": first_chained,
        "all_later_instances_depend_on_previous_dds": all(
            record["depends_on_previous_voxelunique"] for record in records[1:]
        ),
        "builder_log_identifies_exact_failing_instance": False,
        "exact_failure_instance_note": (
            "TensorRT's convertExplicitDDSPluginToImplicit assertion does not emit a "
            "layer/node index. Static dependency identifies the first possible cascade "
            "boundary, not the exact internal failing instance."
        ),
    }


def tensor_info(tensor: Any) -> dict[str, Any]:
    return {
        "name": tensor.name,
        "dtype": str(tensor.dtype).rsplit(".", 1)[-1],
        "shape": [int(value) for value in tensor.shape],
    }


def worker(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    onnx_path = args.onnx.resolve()
    prefix = args.result_prefix
    expected_instances = args.expected_instances
    parser_path = run_dir / f"{prefix}_parser_summary.json"
    build_path = run_dir / f"{prefix}_build_summary.json"
    parser_summary: dict[str, Any] = {
        "parser_success": False,
        "parser_error_count": None,
        "parser_errors": [],
        "inputs": [],
        "outputs": [],
        "standard_plugins_initialized": False,
        "voxel_unique_creator_found": False,
        "voxel_unique_plugin_instances": None,
        "engine_build_called": False,
        "inference_called": False,
    }
    build_summary: dict[str, Any] = {
        "status": f"{prefix.upper()}_BUILD_FAILED",
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
        "fp32_only": True,
        "workspace_bytes": 4 * 1024**3,
        "first_error": None,
    }
    phase4.dump_json(parser_path, parser_summary)
    phase4.dump_json(build_path, build_summary)
    try:
        dll_handles = []
        for directory in (
            args.tensorrt_root.resolve() / "bin",
            args.cuda_root.resolve() / "bin",
            args.plugin_library.resolve().parent,
        ):
            if hasattr(os, "add_dll_directory"):
                dll_handles.append(os.add_dll_directory(str(directory)))

        import tensorrt as trt

        logger = trt.Logger(trt.Logger.VERBOSE)
        parser_summary["standard_plugins_initialized"] = bool(
            trt.init_libnvinfer_plugins(logger, "")
        )
        if not parser_summary["standard_plugins_initialized"]:
            raise RuntimeError("trt.init_libnvinfer_plugins returned false")
        plugin_library, _ = phase4.load_plugin_library(args.plugin_library.resolve())
        registry = phase4.collect_registry(trt, trt.get_plugin_registry())
        parser_summary["voxel_unique_creator_found"] = registry[
            "voxel_unique_creator_found"
        ]
        if not parser_summary["voxel_unique_creator_found"]:
            raise RuntimeError("VoxelUnique Creator not found")

        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        config = builder.create_builder_config()
        if builder is None or network is None or parser is None or config is None:
            raise RuntimeError("TensorRT Builder/Network/Parser/Config creation failed")
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024**3)
        for flag_name in ("FP16", "INT8", "SPARSE_WEIGHTS", "REFIT"):
            if hasattr(trt.BuilderFlag, flag_name):
                config.clear_flag(getattr(trt.BuilderFlag, flag_name))

        print(f"{prefix}:PARSER_BEGIN", flush=True)
        parse_success = bool(parser.parse_from_file(str(onnx_path)))
        errors = phase4.parser_errors(parser)
        instance_count = int(plugin_library.getVoxelUniqueBuildCreationCount())
        parser_summary.update(
            {
                "parser_success": parse_success,
                "parser_error_count": len(errors),
                "parser_errors": errors,
                "inputs": [tensor_info(network.get_input(i)) for i in range(network.num_inputs)],
                "outputs": [tensor_info(network.get_output(i)) for i in range(network.num_outputs)],
                "voxel_unique_plugin_instances": instance_count,
            }
        )
        phase4.dump_json(parser_path, parser_summary)
        print(f"{prefix}:PARSER_END success={parse_success} errors={len(errors)} instances={instance_count}", flush=True)
        if not parse_success or errors:
            raise RuntimeError(f"Parser failed: {errors[:1]}")
        if instance_count != expected_instances:
            raise RuntimeError(
                f"Expected {expected_instances} VoxelUnique instances, got {instance_count}"
            )

        parser_summary["engine_build_called"] = True
        build_summary["parser_success"] = True
        build_summary["engine_build_attempted"] = True
        phase4.dump_json(parser_path, parser_summary)
        phase4.dump_json(build_path, build_summary)
        print(f"{prefix}:ENGINE_BUILD_BEGIN", flush=True)
        started = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        elapsed = time.perf_counter() - started
        build_summary["build_elapsed_seconds"] = elapsed
        if serialized is None:
            raise RuntimeError("build_serialized_network returned None")
        engine_bytes = bytes(serialized)
        build_summary.update(
            {
                "status": f"{prefix.upper()}_BUILD_PASSED",
                "engine_build_success": True,
                "serialized_engine_size_bytes": len(engine_bytes),
                "serialized_engine_sha256": sha256_bytes(engine_bytes),
                "serialized_engine_retained": False,
            }
        )
        phase4.dump_json(build_path, build_summary)
        print(f"{prefix}:ENGINE_BUILD_END elapsed={elapsed:.6f} bytes={len(engine_bytes)}", flush=True)
        print(f"{prefix.upper()}_BUILD_PASSED", flush=True)
        return 0
    except Exception as error:
        build_summary.update(
            {
                "status": f"{prefix.upper()}_BUILD_FAILED",
                "first_error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        phase4.dump_json(parser_path, parser_summary)
        phase4.dump_json(build_path, build_summary)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        print(f"{prefix.upper()}_BUILD_FAILED", flush=True)
        return 2


def run_worker(
    args: argparse.Namespace,
    run_dir: Path,
    onnx_path: Path,
    prefix: str,
    expected_instances: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--run-dir",
        str(run_dir),
        "--onnx",
        str(onnx_path),
        "--result-prefix",
        prefix,
        "--expected-instances",
        str(expected_instances),
        "--plugin-library",
        str(args.plugin_library.resolve()),
        "--tensorrt-root",
        str(args.tensorrt_root.resolve()),
        "--cuda-root",
        str(args.cuda_root.resolve()),
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
    log_path = run_dir / f"{prefix}_builder_verbose.log"
    started = time.perf_counter()
    timed_out = False
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
            timed_out = True
            process.terminate()
            try:
                return_code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait(timeout=15)
            log.write(f"\nWORKER_TIMEOUT_SECONDS={args.timeout_seconds}\n")
    return {
        "command": subprocess.list2cmdline(command),
        "return_code": return_code,
        "timed_out": timed_out,
        "wall_elapsed_seconds": time.perf_counter() - started,
        "log": str(log_path),
    }


def write_report(run_dir: Path, conclusion: dict[str, Any]) -> None:
    single = conclusion["experiment_1_single_instance"]
    independent = conclusion.get("experiment_2_four_independent_instances")
    chain = conclusion["formal_chain_dependency"]
    report = f"""# TensorRT VoxelUnique DDS cascade isolation

## Scope

- Formal ONNX and plugin source were read-only and hash-checked.
- No plugin implementation or `declareSizeTensor` change was made.
- Diagnostic ONNX graphs are isolated mechanics tests, not formal rewrites.
- FP32 parser/builder only; no engine retention, deserialization, execution context, inference, FP16, or benchmark.

## Experiment 1: single fixed-N VoxelUnique

- Parser: `{single.get('parser_success')}`
- Builder: `{single.get('engine_build_success')}`
- Status: `{single.get('status')}`
- First error: `{single.get('first_error')}`

## Experiment 2: four independent fixed-N VoxelUnique instances

{('Not run because experiment 1 failed.' if independent is None else f'''- Parser: `{independent.get('parser_success')}`
- Builder: `{independent.get('engine_build_success')}`
- Status: `{independent.get('status')}`
- First error: `{independent.get('first_error')}`''')}

## Formal GCN_res chain

- First structurally chained instance: `{chain['first_structurally_chained_instance']}`
- All tdb_2..tdb_4 inputs depend on an upstream VoxelUnique: `{chain['all_later_instances_depend_on_previous_dds']}`

## Answers

1. Multiple plugin instances alone: `{conclusion['answers']['multiple_instances_alone']}`
2. Later plugin input depends on prior runtime M: `{conclusion['answers']['later_instances_depend_on_previous_runtime_m']}`
3. Builder failure instance: `{conclusion['answers']['builder_failure_instance']}`

## Conclusion

`{conclusion['status']}`
"""
    (run_dir / "report.md").write_text(report, encoding="utf-8")


def parent(args: argparse.Namespace) -> int:
    source = args.onnx.resolve()
    plugin_path = args.plugin_library.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not plugin_path.is_file():
        raise FileNotFoundError(plugin_path)
    source_hash_before = phase4.sha256(source)
    plugin_hash_before = phase4.sha256(PLUGIN_SOURCE)
    if source_hash_before != phase4.EXPECTED_ONNX_SHA256:
        raise RuntimeError(
            f"Formal ONNX hash mismatch: {source_hash_before} != {phase4.EXPECTED_ONNX_SHA256}"
        )

    run_dir = (
        args.output_root.resolve()
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_single_voxelunique_test"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    formal_model = onnx.load_model(str(source), load_external_data=False)
    onnx.checker.check_model(formal_model)
    chain = trace_formal_chain(formal_model)
    chain["formal_onnx"] = str(source)
    phase4.dump_json(run_dir / "voxelunique_chain_dependency.json", chain)

    single_onnx = run_dir / "single_voxelunique.onnx"
    single_generation = create_diagnostic_model(1, single_onnx)
    phase4.dump_json(run_dir / "single_graph_summary.json", single_generation)
    single_process = run_worker(args, run_dir, single_onnx, "single_voxelunique", 1)
    phase4.dump_json(run_dir / "single_worker_process.json", single_process)
    single_build = json.loads(
        (run_dir / "single_voxelunique_build_summary.json").read_text(encoding="utf-8")
    )
    single_parser = json.loads(
        (run_dir / "single_voxelunique_parser_summary.json").read_text(encoding="utf-8")
    )
    single_result = {**single_build, "parser_success": single_parser["parser_success"]}

    independent_result = None
    independent_process = None
    if single_build.get("engine_build_success"):
        independent_onnx = run_dir / "four_independent_voxelunique.onnx"
        independent_generation = create_diagnostic_model(4, independent_onnx)
        phase4.dump_json(
            run_dir / "four_independent_graph_summary.json", independent_generation
        )
        independent_process = run_worker(
            args,
            run_dir,
            independent_onnx,
            "four_independent_voxelunique",
            4,
        )
        phase4.dump_json(
            run_dir / "four_independent_worker_process.json", independent_process
        )
        independent_build = json.loads(
            (run_dir / "four_independent_voxelunique_build_summary.json").read_text(
                encoding="utf-8"
            )
        )
        independent_parser = json.loads(
            (run_dir / "four_independent_voxelunique_parser_summary.json").read_text(
                encoding="utf-8"
            )
        )
        independent_result = {
            **independent_build,
            "parser_success": independent_parser["parser_success"],
        }

    if not single_build.get("engine_build_success"):
        multiple_answer = "NOT_TESTED: single instance itself failed"
        failure_answer = "Single diagnostic VoxelUnique; see single builder log"
        status = "SINGLE_VOXELUNIQUE_BUILD_FAILED"
    elif not independent_result or not independent_result.get("engine_build_success"):
        multiple_answer = "SUPPORTED_AS_CAUSE: single passed but four independent instances failed"
        failure_answer = (
            "The independent-four graph failed, but TensorRT did not emit an exact instance index"
        )
        status = "FOUR_INDEPENDENT_VOXELUNIQUE_BUILD_FAILED"
    else:
        multiple_answer = "EXCLUDED: four independent fixed-N instances build successfully"
        failure_answer = (
            f"Formal builder log has no instance index; the first structural DDS cascade boundary is "
            f"{chain['first_structurally_chained_instance']}"
        )
        status = "DDS_CASCADE_ISOLATION_COMPLETED"

    conclusion = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": status,
        "formal_onnx": str(source),
        "formal_onnx_sha256_before": source_hash_before,
        "formal_onnx_sha256_after": phase4.sha256(source),
        "formal_onnx_unchanged": phase4.sha256(source) == source_hash_before,
        "plugin_source": str(PLUGIN_SOURCE),
        "plugin_source_sha256_before": plugin_hash_before,
        "plugin_source_sha256_after": phase4.sha256(PLUGIN_SOURCE),
        "plugin_source_unchanged": phase4.sha256(PLUGIN_SOURCE) == plugin_hash_before,
        "plugin_library": str(plugin_path),
        "plugin_library_sha256": phase4.sha256(plugin_path),
        "experiment_1_single_instance": single_result,
        "experiment_2_four_independent_instances": independent_result,
        "formal_chain_dependency": chain,
        "answers": {
            "multiple_instances_alone": multiple_answer,
            "later_instances_depend_on_previous_runtime_m": (
                "YES" if chain["all_later_instances_depend_on_previous_dds"] else "NO/INCOMPLETE"
            ),
            "builder_failure_instance": failure_answer,
        },
        "engine_files_retained": False,
        "deserialization_attempted": False,
        "execution_context_created": False,
        "inference_attempted": False,
        "fp16_attempted": False,
        "benchmark_attempted": False,
    }
    if not conclusion["formal_onnx_unchanged"] or not conclusion["plugin_source_unchanged"]:
        raise RuntimeError("Read-only formal source hash changed")
    phase4.dump_json(run_dir / "isolation_summary.json", conclusion)
    write_report(run_dir, conclusion)
    print(f"RUN_DIR={run_dir}")
    print(f"SINGLE_STATUS={single_result['status']}")
    if independent_result is not None:
        print(f"FOUR_INDEPENDENT_STATUS={independent_result['status']}")
    print(f"FIRST_STRUCTURAL_CASCADE={chain['first_structurally_chained_instance']}")
    print(status)
    return 0 if single_build.get("engine_build_success") else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plugin-library", type=Path, default=DEFAULT_PLUGIN_LIBRARY)
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TENSORRT_ROOT)
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA_ROOT)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--result-prefix", default="single_voxelunique")
    parser.add_argument("--expected-instances", type=int, default=1)
    args = parser.parse_args()
    if args.worker and args.run_dir is None:
        parser.error("--run-dir is required with --worker")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(worker(arguments) if arguments.worker else parent(arguments))
