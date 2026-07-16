"""Incrementally restore formal TDB consumers around VoxelUnique DDS outputs.

The script extracts read-only dependency-closed subgraphs from the formal
if-folded GCN_res ONNX.  It never edits that ONNX or the plugin source.  Each
candidate is checked, parsed, and passed to the FP32 builder in a separate
subprocess.  Serialized engine bytes are not retained and no inference is run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto
from onnx.utils import extract_model

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
WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "test_voxelunique_dds_cascade_isolation.py"
PLUGIN_SOURCE = (
    PROJECT_ROOT
    / "tests"
    / "tensorrt_voxel_unique_correctness"
    / "VoxelUniqueCorrectnessPlugin.cu"
)
PLUGIN_DOMAIN = "com.tensorrt.ptv2"
PLUGIN_OP = "VoxelUnique"
FORMAL_INPUTS = ["points", "adj"]
DIAGNOSTIC_OUTPUT_SHAPE_OVERRIDES: dict[str, tuple[int, list[int | None]]] = {
    # These formal intermediates have tensor dtype but no shape field in
    # value_info.  ONNX allows that for intermediates, whereas an extracted
    # graph output must carry a shape.  The overrides only annotate diagnostic
    # graph outputs; they do not alter nodes, edges, attributes, or dataflow.
    "/model/tdb_1/ScatterElements_3_output_0": (TensorProto.FLOAT, [None, 96]),
    "/model/tdb_1/GatherND_output_0": (TensorProto.FLOAT, [None, 3]),
    "/model/tdb_1/GatherND_1_output_0": (TensorProto.FLOAT, [None, 96]),
    "/model/tdb_2/ScatterElements_3_output_0": (TensorProto.FLOAT, [None, 192]),
    "/model/tdb_2/GatherND_output_0": (TensorProto.FLOAT, [None, 3]),
    "/model/tdb_2/GatherND_1_output_0": (TensorProto.FLOAT, [None, 192]),
}


def value_contracts(model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    for value in values:
        tensor_type = value.type.tensor_type
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                shape.append(str(dim.dim_param))
            else:
                shape.append(None)
        records[value.name] = {
            "dtype": TensorProto.DataType.Name(tensor_type.elem_type),
            "shape": shape,
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


def get_nodes(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    result = {node.name: node for node in model.graph.node}
    if len(result) != len(model.graph.node):
        raise RuntimeError("Formal graph contains duplicate/empty node names")
    return result


def get_plugin_nodes(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    nodes = [
        node
        for node in model.graph.node
        if node.op_type == PLUGIN_OP and node.domain == PLUGIN_DOMAIN
    ]
    expected = [f"/model/tdb_{index}/Unique" for index in range(1, 5)]
    if [node.name for node in nodes] != expected:
        raise RuntimeError(f"Unexpected VoxelUnique nodes: {[node.name for node in nodes]}")
    return nodes


def experiment_definitions(plugin_nodes: list[onnx.NodeProto]) -> list[dict[str, Any]]:
    plugin_outputs = {node.name: list(node.output) for node in plugin_nodes}
    tdb1 = plugin_outputs["/model/tdb_1/Unique"]
    tdb2 = plugin_outputs["/model/tdb_2/Unique"]

    experiments = [
        {
            "id": "A_tdb1_scatterelements_1",
            "description": "Restore the dependency-closed real tdb_1 ScatterElements_1 count aggregation.",
            "stage": "tdb_1",
            "boundary_node": "/model/tdb_1/ScatterElements_1",
            "outputs": tdb1 + ["/model/tdb_1/ScatterElements_1_output_0"],
            "expected_plugin_instances": 1,
        },
        {
            "id": "B_tdb1_shape_gather_unsqueeze",
            "description": "Keep A and additionally retain real Shape/Gather/Unsqueeze consumer outputs.",
            "stage": "tdb_1",
            "boundary_node": "/model/tdb_1/Unsqueeze_7",
            "outputs": tdb1
            + [
                "/model/tdb_1/ScatterElements_1_output_0",
                "/model/tdb_1/Shape_output_0",
                "/model/tdb_1/Gather_2_output_0",
                "/model/tdb_1/Unsqueeze_7_output_0",
            ],
            "expected_plugin_instances": 1,
        },
        {
            "id": "C_tdb1_expand_constantofshape",
            "description": "Keep B and additionally retain the real index Expand and pooled-XYZ allocation.",
            "stage": "tdb_1",
            "boundary_node": "/model/tdb_1/ConstantOfShape_5",
            "outputs": tdb1
            + [
                "/model/tdb_1/ScatterElements_1_output_0",
                "/model/tdb_1/Shape_output_0",
                "/model/tdb_1/Gather_2_output_0",
                "/model/tdb_1/Unsqueeze_7_output_0",
                "/model/tdb_1/Expand_1_output_0",
                "/model/tdb_1/ConstantOfShape_5_output_0",
            ],
            "expected_plugin_instances": 1,
        },
        {
            "id": "D1_tdb1_scatterelements_2",
            "description": "Add real FLOAT ScatterElements_2 summed-XYZ aggregation after C.",
            "stage": "tdb_1",
            "boundary_node": "/model/tdb_1/ScatterElements_2",
            "outputs": tdb1
            + [
                "/model/tdb_1/ScatterElements_1_output_0",
                "/model/tdb_1/Shape_output_0",
                "/model/tdb_1/Gather_2_output_0",
                "/model/tdb_1/Unsqueeze_7_output_0",
                "/model/tdb_1/Expand_1_output_0",
                "/model/tdb_1/ConstantOfShape_5_output_0",
                "/model/tdb_1/ScatterElements_2_output_0",
            ],
            "expected_plugin_instances": 1,
        },
        {
            "id": "D2_tdb1_mean_xyz",
            "description": "Add count cast/unsqueeze and Div_2 mean-XYZ output after ScatterElements_2.",
            "stage": "tdb_1",
            "boundary_node": "/model/tdb_1/Div_2",
            "outputs": tdb1
            + [
                "/model/tdb_1/ScatterElements_1_output_0",
                "/model/tdb_1/Shape_output_0",
                "/model/tdb_1/Gather_2_output_0",
                "/model/tdb_1/Unsqueeze_7_output_0",
                "/model/tdb_1/Expand_1_output_0",
                "/model/tdb_1/ConstantOfShape_5_output_0",
                "/model/tdb_1/ScatterElements_2_output_0",
                "/model/tdb_1/Div_2_output_0",
            ],
            "expected_plugin_instances": 1,
        },
        {
            "id": "D3_tdb1_scatterelements_3",
            "description": "Add real FLOAT ScatterElements_3 max-feature aggregation.",
            "stage": "tdb_1",
            "boundary_node": "/model/tdb_1/ScatterElements_3",
            "outputs": tdb1
            + [
                "/model/tdb_1/ScatterElements_1_output_0",
                "/model/tdb_1/Div_2_output_0",
                "/model/tdb_1/ScatterElements_3_output_0",
            ],
            "expected_plugin_instances": 1,
        },
        {
            "id": "D4_tdb1_scatterelements_4",
            "description": "Add real INT64 ScatterElements_4 per-batch voxel-count aggregation.",
            "stage": "tdb_1",
            "boundary_node": "/model/tdb_1/ScatterElements_4",
            "outputs": tdb1
            + [
                "/model/tdb_1/ScatterElements_1_output_0",
                "/model/tdb_1/Div_2_output_0",
                "/model/tdb_1/ScatterElements_3_output_0",
                "/model/tdb_1/ScatterElements_4_output_0",
            ],
            "expected_plugin_instances": 1,
        },
        {
            "id": "D5_tdb1_gathernd_crop",
            "description": "Add min-count crop masks, NonZero, and both GatherND pooled tensors.",
            "stage": "tdb_1",
            "boundary_node": "/model/tdb_1/GatherND_1",
            "outputs": tdb1
            + [
                "/model/tdb_1/ScatterElements_1_output_0",
                "/model/tdb_1/Div_2_output_0",
                "/model/tdb_1/ScatterElements_3_output_0",
                "/model/tdb_1/ScatterElements_4_output_0",
                "/model/tdb_1/GatherND_output_0",
                "/model/tdb_1/GatherND_1_output_0",
            ],
            "expected_plugin_instances": 1,
        },
        {
            "id": "D5a_tdb1_pooled_xyz_shape_tensor",
            "description": "After D5, build the real [1,min_voxel_count,3] shape tensor without Reshape_10.",
            "stage": "tdb_1",
            "boundary_node": "/model/tdb_1/Concat_5",
            "outputs": tdb1
            + [
                "/model/tdb_1/ScatterElements_1_output_0",
                "/model/tdb_1/Div_2_output_0",
                "/model/tdb_1/ScatterElements_3_output_0",
                "/model/tdb_1/ScatterElements_4_output_0",
                "/model/tdb_1/GatherND_output_0",
                "/model/tdb_1/GatherND_1_output_0",
                "/model/tdb_1/Unsqueeze_11_output_0",
                "/model/tdb_1/Concat_5_output_0",
            ],
            "expected_plugin_instances": 1,
        },
        {
            "id": "D6a_tdb1_pooled_xyz_reshape",
            "description": "After D5, restore only the final pooled XYZ Reshape_10 output.",
            "stage": "tdb_1",
            "boundary_node": "/model/tdb_1/Reshape_10",
            "outputs": tdb1
            + [
                "/model/tdb_1/ScatterElements_1_output_0",
                "/model/tdb_1/Div_2_output_0",
                "/model/tdb_1/ScatterElements_3_output_0",
                "/model/tdb_1/ScatterElements_4_output_0",
                "/model/tdb_1/GatherND_output_0",
                "/model/tdb_1/GatherND_1_output_0",
                "/model/tdb_1/Reshape_10_output_0",
            ],
            "expected_plugin_instances": 1,
        },
        {
            "id": "D6b_tdb1_pooled_feature_reshape",
            "description": "After D6a, additionally restore final pooled feature Reshape_11.",
            "stage": "tdb_1",
            "boundary_node": "/model/tdb_1/Reshape_11",
            "outputs": tdb1
            + [
                "/model/tdb_1/ScatterElements_1_output_0",
                "/model/tdb_1/Div_2_output_0",
                "/model/tdb_1/ScatterElements_3_output_0",
                "/model/tdb_1/ScatterElements_4_output_0",
                "/model/tdb_1/GatherND_output_0",
                "/model/tdb_1/GatherND_1_output_0",
                "/model/tdb_1/Reshape_10_output_0",
                "/model/tdb_1/Reshape_11_output_0",
            ],
            "expected_plugin_instances": 1,
        },
        {
            "id": "E_tdb2_voxelunique_boundary",
            "description": "After complete tdb_1, add formal tdb_2 VoxelUnique and keep all its outputs live.",
            "stage": "tdb_2",
            "boundary_node": "/model/tdb_2/Unique",
            "outputs": tdb1
            + ["/model/tdb_1/Reshape_10_output_0", "/model/tdb_1/Reshape_11_output_0"]
            + tdb2,
            "expected_plugin_instances": 2,
        },
        {
            "id": "F_tdb2_scatterelements_1",
            "description": "Restore the dependency-closed real tdb_2 ScatterElements_1 count aggregation.",
            "stage": "tdb_2",
            "boundary_node": "/model/tdb_2/ScatterElements_1",
            "outputs": tdb1
            + ["/model/tdb_1/Reshape_10_output_0", "/model/tdb_1/Reshape_11_output_0"]
            + tdb2
            + ["/model/tdb_2/ScatterElements_1_output_0"],
            "expected_plugin_instances": 2,
        },
        {
            "id": "G_tdb2_shape_gather_unsqueeze",
            "description": "Keep F and retain the real tdb_2 Shape/Gather/Unsqueeze outputs.",
            "stage": "tdb_2",
            "boundary_node": "/model/tdb_2/Unsqueeze_11",
            "outputs": tdb1
            + ["/model/tdb_1/Reshape_10_output_0", "/model/tdb_1/Reshape_11_output_0"]
            + tdb2
            + [
                "/model/tdb_2/ScatterElements_1_output_0",
                "/model/tdb_2/Shape_3_output_0",
                "/model/tdb_2/Gather_4_output_0",
                "/model/tdb_2/Unsqueeze_11_output_0",
            ],
            "expected_plugin_instances": 2,
        },
        {
            "id": "H_tdb2_expand_constantofshape",
            "description": "Keep G and retain the real tdb_2 Expand and ConstantOfShape pooling branches.",
            "stage": "tdb_2",
            "boundary_node": "/model/tdb_2/ConstantOfShape_6",
            "outputs": tdb1
            + ["/model/tdb_1/Reshape_10_output_0", "/model/tdb_1/Reshape_11_output_0"]
            + tdb2
            + [
                "/model/tdb_2/ScatterElements_1_output_0",
                "/model/tdb_2/Shape_3_output_0",
                "/model/tdb_2/Gather_4_output_0",
                "/model/tdb_2/Unsqueeze_11_output_0",
                "/model/tdb_2/Expand_1_output_0",
                "/model/tdb_2/ConstantOfShape_6_output_0",
            ],
            "expected_plugin_instances": 2,
        },
        {
            "id": "I_tdb2_complete_pooling",
            "description": "Restore complete tdb_2 pooled XYZ and pooled feature outputs.",
            "stage": "tdb_2",
            "boundary_node": "/model/tdb_2/Reshape_14",
            "outputs": tdb1
            + ["/model/tdb_1/Reshape_10_output_0", "/model/tdb_1/Reshape_11_output_0"]
            + tdb2
            + ["/model/tdb_2/Reshape_13_output_0", "/model/tdb_2/Reshape_14_output_0"],
            "expected_plugin_instances": 2,
        },
    ]
    for experiment in experiments:
        experiment["outputs"] = list(dict.fromkeys(experiment["outputs"]))
    return experiments


def upstream_node_names(
    output_names: list[str], producer: dict[str, onnx.NodeProto]
) -> set[str]:
    result: set[str] = set()
    queue: deque[str] = deque(output_names)
    seen_tensors: set[str] = set()
    while queue:
        tensor = queue.popleft()
        if tensor in seen_tensors:
            continue
        seen_tensors.add(tensor)
        node = producer.get(tensor)
        if node is None:
            continue
        result.add(node.name)
        queue.extend(input_name for input_name in node.input if input_name)
    return result


def downstream_from_plugin(
    model: onnx.ModelProto,
    included_nodes: set[str],
    plugin_names: set[str],
) -> set[str]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    nodes_by_name = get_nodes(model)
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    queue: deque[str] = deque()
    for plugin_name in plugin_names:
        queue.extend(nodes_by_name[plugin_name].output)
    result: set[str] = set()
    seen_tensors: set[str] = set()
    while queue:
        tensor = queue.popleft()
        if tensor in seen_tensors:
            continue
        seen_tensors.add(tensor)
        for node in consumers.get(tensor, []):
            if node.name not in included_nodes:
                continue
            if node.name not in result:
                result.add(node.name)
                queue.extend(node.output)
    return result


def node_record(
    node: onnx.NodeProto,
    contracts: dict[str, dict[str, Any]],
    plugin_output_names: set[str],
) -> dict[str, Any]:
    return {
        "name": node.name,
        "op_type": node.op_type,
        "domain": node.domain,
        "inputs": [
            {
                "name": input_name,
                "contract": contracts.get(input_name),
                "is_voxelunique_output": input_name in plugin_output_names,
            }
            for input_name in node.input
        ],
        "outputs": [
            {"name": output_name, "contract": contracts.get(output_name)}
            for output_name in node.output
        ],
        "contains_direct_dds_output_input": any(
            input_name in plugin_output_names for input_name in node.input
        ),
    }


def create_extracted_graph(
    source: Path,
    destination: Path,
    output_names: list[str],
) -> None:
    extract_model(
        str(source),
        str(destination),
        FORMAL_INPUTS,
        output_names,
        check_model=False,
        infer_shapes=False,
    )
    candidate = onnx.load_model(str(destination), load_external_data=False)
    annotations_added = 0
    for output in candidate.graph.output:
        tensor_type = output.type.tensor_type
        if tensor_type.HasField("shape"):
            continue
        override = DIAGNOSTIC_OUTPUT_SHAPE_OVERRIDES.get(output.name)
        if override is None:
            raise RuntimeError(
                f"Extracted graph output lacks shape and has no diagnostic annotation: {output.name}"
            )
        dtype, dimensions = override
        tensor_type.elem_type = dtype
        tensor_type.shape.SetInParent()
        for index, dimension in enumerate(dimensions):
            dim = tensor_type.shape.dim.add()
            if dimension is None:
                dim.dim_param = f"diagnostic_dynamic_{index}"
            else:
                dim.dim_value = dimension
        annotations_added += 1
    onnx.save_model(candidate, str(destination))
    onnx.checker.check_model(candidate)


def run_worker(
    args: argparse.Namespace,
    run_dir: Path,
    experiment: dict[str, Any],
    onnx_path: Path,
) -> dict[str, Any]:
    prefix = experiment["id"].lower()
    command = [
        sys.executable,
        "-u",
        str(WORKER_SCRIPT),
        "--worker",
        "--run-dir",
        str(run_dir),
        "--onnx",
        str(onnx_path),
        "--result-prefix",
        prefix,
        "--expected-instances",
        str(experiment["expected_plugin_instances"]),
        "--plugin-library",
        str(args.plugin_library.resolve()),
        "--tensorrt-root",
        str(args.tensorrt_root.resolve()),
        "--cuda-root",
        str(args.cuda_root.resolve()),
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
    log_path = run_dir / f"{prefix}_builder_verbose.log"
    started = time.perf_counter()
    timed_out = False
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
            timed_out = True
            process.terminate()
            try:
                return_code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait(timeout=15)
            log.write(f"\nWORKER_TIMEOUT_SECONDS={args.timeout_seconds}\n")
    parser_summary = json.loads(
        (run_dir / f"{prefix}_parser_summary.json").read_text(encoding="utf-8")
    )
    build_summary = json.loads(
        (run_dir / f"{prefix}_build_summary.json").read_text(encoding="utf-8")
    )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    important_log_lines = [
        line
        for line in log_text.splitlines()
        if any(
            marker in line
            for marker in (
                "convertExplicitDDSPluginToImplicit",
                "nodeIdxToDDSOutputIndices",
                "Error Code",
                "Assertion",
            )
        )
    ]
    return {
        "command": subprocess.list2cmdline(command),
        "return_code": return_code,
        "timed_out": timed_out,
        "wall_elapsed_seconds": time.perf_counter() - started,
        "log_path": str(log_path),
        "parser_summary_path": str(run_dir / f"{prefix}_parser_summary.json"),
        "build_summary_path": str(run_dir / f"{prefix}_build_summary.json"),
        "parser_success": parser_summary.get("parser_success"),
        "parser_error_count": parser_summary.get("parser_error_count"),
        "plugin_instances": parser_summary.get("voxel_unique_plugin_instances"),
        "builder_success": build_summary.get("engine_build_success"),
        "build_elapsed_seconds": build_summary.get("build_elapsed_seconds"),
        "first_error": build_summary.get("first_error"),
        "important_builder_log_lines": important_log_lines,
    }


def write_report(
    run_dir: Path,
    matrix: list[dict[str, Any]],
    failure: dict[str, Any],
    source_hash: str,
    plugin_hash: str,
) -> None:
    table_rows = []
    for item in matrix:
        table_rows.append(
            "| {id} | {stage} | {checker} | {parser} | {builder} | {boundary} |".format(
                id=item["id"],
                stage=item["stage"],
                checker="PASS" if item["onnx_checker_passed"] else "FAIL",
                parser="PASS" if item.get("parser_success") else "FAIL",
                builder="PASS" if item.get("builder_success") else "FAIL",
                boundary=item["boundary_node"],
            )
        )
    failure_text = (
        "No failure boundary was found in the executed scope."
        if not failure.get("found")
        else (
            f"The first failed experiment is `{failure['experiment_id']}` at candidate "
            f"boundary `{failure['boundary_node']}` (`{failure['boundary_op_type']}`)."
        )
    )
    report = f"""# TensorRT incremental TDB DDS consumer restore

## Safety

- Formal ONNX SHA-256 before/after: `{source_hash}`
- Plugin source SHA-256 before/after: `{plugin_hash}`
- No source/plugin/declareSizeTensor modification.
- FP32 parser and builder only; no retained engine, deserialization, inference,
  FP16, or benchmark.

## Experiment matrix

| Experiment | Stage | ONNX checker | Parser | FP32 builder | Candidate boundary |
|---|---|---|---|---|---|
{chr(10).join(table_rows)}

## Failure boundary

{failure_text}

The candidates are dependency-closed extracts from the formal graph. Because
`ScatterElements_1` requires real Shape/Gather/ConstantOfShape/Slice inputs,
experiment A necessarily includes those ancestors; B and C retain additional
parallel consumer branches rather than replacing A's prerequisites.

`{'DDS_FIRST_FAILURE_BOUNDARY_FOUND' if failure.get('found') else 'DDS_FIRST_FAILURE_BOUNDARY_NOT_FOUND'}`
"""
    (run_dir / "report.md").write_text(report, encoding="utf-8")


def main(args: argparse.Namespace) -> int:
    source = args.onnx.resolve()
    plugin_library = args.plugin_library.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not plugin_library.is_file():
        raise FileNotFoundError(plugin_library)
    source_hash_before = phase4.sha256(source)
    plugin_hash_before = phase4.sha256(PLUGIN_SOURCE)
    if source_hash_before != phase4.EXPECTED_ONNX_SHA256:
        raise RuntimeError(
            f"Formal ONNX hash mismatch: {source_hash_before} != {phase4.EXPECTED_ONNX_SHA256}"
        )

    run_dir = (
        args.output_root.resolve()
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_incremental_tdb_restore"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    model = onnx.load_model(str(source), load_external_data=False)
    onnx.checker.check_model(model)
    contracts = value_contracts(model)
    nodes_by_name = get_nodes(model)
    plugin_nodes = get_plugin_nodes(model)
    plugin_outputs = {output for node in plugin_nodes for output in node.output}
    producer = {
        output: node for node in model.graph.node for output in node.output if output
    }
    experiments = experiment_definitions(plugin_nodes)

    matrix: list[dict[str, Any]] = []
    dependency_payload: dict[str, Any] = {
        "formal_onnx": str(source),
        "formal_onnx_sha256": source_hash_before,
        "plugin_nodes": [
            {
                "name": node.name,
                "input": [
                    {"name": name, "contract": contracts.get(name)} for name in node.input
                ],
                "outputs": [
                    {
                        "index": index,
                        "name": name,
                        "contract": contracts.get(name),
                        "role": "size_tensor" if index == 0 else ("dds_values" if index == 1 else "inverse_indices"),
                    }
                    for index, name in enumerate(node.output)
                ],
            }
            for node in plugin_nodes
        ],
        "experiments": [],
    }
    previous_downstream_nodes: set[str] = set()
    failure: dict[str, Any] = {"found": False}

    for order, experiment in enumerate(experiments, start=1):
        experiment_dir = run_dir / experiment["id"]
        experiment_dir.mkdir(parents=False, exist_ok=False)
        candidate = experiment_dir / "candidate.onnx"
        create_extracted_graph(source, candidate, experiment["outputs"])
        candidate_model = onnx.load_model(str(candidate), load_external_data=False)
        onnx.checker.check_model(candidate_model)
        candidate_nodes = {node.name for node in candidate_model.graph.node}
        expected_ancestors = upstream_node_names(experiment["outputs"], producer)
        if candidate_nodes != expected_ancestors:
            raise RuntimeError(
                f"Extractor dependency mismatch for {experiment['id']}: "
                f"missing={sorted(expected_ancestors-candidate_nodes)[:5]} "
                f"extra={sorted(candidate_nodes-expected_ancestors)[:5]}"
            )
        included_plugin_names = {
            node.name
            for node in candidate_model.graph.node
            if node.op_type == PLUGIN_OP and node.domain == PLUGIN_DOMAIN
        }
        downstream_nodes = downstream_from_plugin(
            model, candidate_nodes, included_plugin_names
        )
        added_downstream = downstream_nodes - previous_downstream_nodes
        boundary_node = nodes_by_name[experiment["boundary_node"]]
        dependency_record = {
            "order": order,
            "id": experiment["id"],
            "description": experiment["description"],
            "target_outputs": [
                {"name": name, "contract": contracts.get(name)}
                for name in experiment["outputs"]
            ],
            "candidate_onnx": str(candidate),
            "candidate_sha256": phase4.sha256(candidate),
            "included_node_count": len(candidate_nodes),
            "included_voxelunique_nodes": sorted(included_plugin_names),
            "plugin_dependent_downstream_node_count": len(downstream_nodes),
            "added_plugin_dependent_nodes": [
                node_record(nodes_by_name[name], contracts, plugin_outputs)
                for name in sorted(
                    added_downstream,
                    key=lambda name: list(model.graph.node).index(nodes_by_name[name]),
                )
            ],
            "boundary": node_record(boundary_node, contracts, plugin_outputs),
        }
        dependency_payload["experiments"].append(dependency_record)
        phase4.dump_json(experiment_dir / "graph_summary.json", dependency_record)

        worker_result = run_worker(args, experiment_dir, experiment, candidate)
        row = {
            "order": order,
            "id": experiment["id"],
            "stage": experiment["stage"],
            "description": experiment["description"],
            "boundary_node": experiment["boundary_node"],
            "boundary_op_type": boundary_node.op_type,
            "candidate_onnx": str(candidate),
            "candidate_sha256": phase4.sha256(candidate),
            "onnx_checker_passed": True,
            "included_node_count": len(candidate_nodes),
            "added_plugin_dependent_node_count": len(added_downstream),
            **worker_result,
        }
        matrix.append(row)
        phase4.dump_json(run_dir / "experiment_matrix.json", {"experiments": matrix})
        if not worker_result["parser_success"] or not worker_result["builder_success"]:
            failure = {
                "found": True,
                "experiment_id": experiment["id"],
                "stage": experiment["stage"],
                "boundary_node": experiment["boundary_node"],
                "boundary_op_type": boundary_node.op_type,
                "boundary_inputs": node_record(
                    boundary_node, contracts, plugin_outputs
                )["inputs"],
                "boundary_outputs": node_record(
                    boundary_node, contracts, plugin_outputs
                )["outputs"],
                "contains_direct_dds_output": node_record(
                    boundary_node, contracts, plugin_outputs
                )["contains_direct_dds_output_input"],
                "contains_transitive_dds_dependency": (
                    experiment["boundary_node"] in downstream_nodes
                ),
                "failure_isolated_to_single_new_plugin_dependent_node": (
                    len(added_downstream) == 1
                ),
                "parser_success": worker_result["parser_success"],
                "builder_success": worker_result["builder_success"],
                "first_error": worker_result["first_error"],
                "important_builder_log_lines": worker_result[
                    "important_builder_log_lines"
                ],
                "added_plugin_dependent_nodes": dependency_record[
                    "added_plugin_dependent_nodes"
                ],
                "previous_experiment": matrix[-2]["id"] if len(matrix) > 1 else "KNOWN_SINGLE_VOXELUNIQUE_REFERENCE",
                "stop_condition_applied": True,
            }
            break
        previous_downstream_nodes = downstream_nodes

    phase4.dump_json(run_dir / "node_dependency.json", dependency_payload)
    phase4.dump_json(run_dir / "failure_boundary.json", failure)
    source_hash_after = phase4.sha256(source)
    plugin_hash_after = phase4.sha256(PLUGIN_SOURCE)
    safety = {
        "formal_onnx_sha256_before": source_hash_before,
        "formal_onnx_sha256_after": source_hash_after,
        "formal_onnx_unchanged": source_hash_after == source_hash_before,
        "plugin_source_sha256_before": plugin_hash_before,
        "plugin_source_sha256_after": plugin_hash_after,
        "plugin_source_unchanged": plugin_hash_after == plugin_hash_before,
        "engine_files_retained": False,
        "deserialization_attempted": False,
        "execution_context_created": False,
        "inference_attempted": False,
        "fp16_attempted": False,
        "benchmark_attempted": False,
    }
    phase4.dump_json(run_dir / "safety_summary.json", safety)
    if not safety["formal_onnx_unchanged"] or not safety["plugin_source_unchanged"]:
        raise RuntimeError("Read-only source hash changed")
    write_report(run_dir, matrix, failure, source_hash_before, plugin_hash_before)
    print(f"RUN_DIR={run_dir}")
    for row in matrix:
        print(
            f"{row['id']}: checker=PASS parser={row['parser_success']} "
            f"builder={row['builder_success']}"
        )
    if failure.get("found"):
        print(f"FIRST_FAILURE_EXPERIMENT={failure['experiment_id']}")
        print(f"FIRST_FAILURE_NODE={failure['boundary_node']}")
        print(f"FIRST_FAILURE_OP={failure['boundary_op_type']}")
        print("DDS_FIRST_FAILURE_BOUNDARY_FOUND")
        return 0
    print("DDS_FIRST_FAILURE_BOUNDARY_NOT_FOUND")
    return 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plugin-library", type=Path, default=DEFAULT_PLUGIN_LIBRARY)
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TENSORRT_ROOT)
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA_ROOT)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
