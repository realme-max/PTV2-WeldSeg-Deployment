"""Fold the 16 audited constant-false TDB If nodes and run parser-only audit.

This script never overwrites the input ONNX.  Each If is replaced in its parent
graph by Identity nodes that connect the live else-branch capture tensors to the
original If output names.  It runs ONNX checker, a static graph audit, and an
existing native TensorRT parser-only executable; it never builds an engine or
runs inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import onnx
from onnx import helper

from audit_dynamic_squeeze import collect_contexts, evaluate_scalar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260715_204141_043276_voxelplugin_rewrite"
    / "rewritten.onnx"
)
ORIGINAL_ONNX = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_onnx"
    / "20260715_onnx_after_cdist_fp32_opset18"
    / "gcn_res_deploy_fp32_opset18.onnx"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
DEFAULT_PARSER_EXE = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260715_204141_043276_voxelplugin_rewrite"
    / "build"
    / "Release"
    / "gcn_res_voxel_unique_parser.exe"
)
DEFAULT_TENSORRT_ROOT = Path(
    r"D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106"
)
DEFAULT_CUDA_ROOT = Path(
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
)
TDB_IF_PATTERN = re.compile(r"^/model/(tdb_[1-4])/If(?:_[1-3])?$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def iter_graphs(
    graph: onnx.GraphProto, scope: str = "main"
) -> Iterable[tuple[str, onnx.GraphProto]]:
    yield scope, graph
    for node in graph.node:
        node_name = node.name or node.op_type
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from iter_graphs(
                    attribute.g, f"{scope}/{node_name}:{attribute.name}"
                )
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for index, nested in enumerate(attribute.graphs):
                    yield from iter_graphs(
                        nested,
                        f"{scope}/{node_name}:{attribute.name}[{index}]",
                    )


def graph_attribute(node: onnx.NodeProto, name: str) -> onnx.GraphProto:
    for attribute in node.attribute:
        if attribute.name == name and attribute.type == onnx.AttributeProto.GRAPH:
            return attribute.g
    raise RuntimeError(f"{node.name}: missing graph attribute {name}")


def scalar_bool_from_constant_node(node: onnx.NodeProto) -> bool | None:
    if node.op_type != "Constant":
        return None
    for attribute in node.attribute:
        if attribute.name == "value":
            array = onnx.numpy_helper.to_array(attribute.t)
            if array.size == 1:
                return bool(array.reshape(-1)[0])
        if attribute.name == "value_int":
            return bool(attribute.i)
        if attribute.name == "value_ints" and len(attribute.ints) == 1:
            return bool(attribute.ints[0])
    return None


def condition_value(graph: onnx.GraphProto, tensor_name: str) -> bool | None:
    for initializer in graph.initializer:
        if initializer.name == tensor_name:
            array = onnx.numpy_helper.to_array(initializer)
            if array.size == 1:
                return bool(array.reshape(-1)[0])
    for node in graph.node:
        if tensor_name in node.output:
            return scalar_bool_from_constant_node(node)
    return None


def producer_index(graph: onnx.GraphProto, tensor_name: str) -> int | None:
    for index, node in enumerate(graph.node):
        if tensor_name in node.output:
            return index
    return None


def live_output_source(
    node: onnx.NodeProto, live_graph: onnx.GraphProto, output_index: int
) -> tuple[str, str]:
    if output_index >= len(live_graph.output):
        raise RuntimeError(f"{node.name}: live branch output count mismatch")
    live_output = live_graph.output[output_index].name
    producers = [item for item in live_graph.node if live_output in item.output]
    if len(producers) != 1:
        raise RuntimeError(
            f"{node.name}: live output {live_output!r} has {len(producers)} producers"
        )
    producer = producers[0]
    if producer.op_type != "Identity" or len(producer.input) != 1:
        raise RuntimeError(
            f"{node.name}: live output producer must be single-input Identity, "
            f"got {producer.op_type}"
        )
    return live_output, producer.input[0]


def node_counts(model: onnx.ModelProto) -> tuple[int, Counter[str]]:
    counter: Counter[str] = Counter()
    total = 0
    for _, graph in iter_graphs(model.graph):
        for node in graph.node:
            counter[node.op_type] += 1
            total += 1
    return total, counter


def squeeze_records(model: onnx.ModelProto) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scope, graph in iter_graphs(model.graph):
        for node in graph.node:
            if node.op_type != "Squeeze":
                continue
            axes_attributes = [
                helper.get_attribute_value(attribute)
                for attribute in node.attribute
                if attribute.name == "axes"
            ]
            records.append(
                {
                    "node_name": node.name,
                    "scope": scope,
                    "input_count": len(node.input),
                    "axes_attribute": axes_attributes[0]
                    if axes_attributes
                    else None,
                    "axes_input": node.input[1] if len(node.input) > 1 else None,
                }
            )
    return records


def type_contracts(graph: onnx.GraphProto) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for value in (*graph.input, *graph.output, *graph.value_info):
        tensor_type = value.type.tensor_type
        shape_present = tensor_type.HasField("shape")
        shape: list[dict[str, Any]] | None = None
        if shape_present:
            shape = []
            for dimension in tensor_type.shape.dim:
                if dimension.HasField("dim_value"):
                    shape.append({"kind": "value", "value": int(dimension.dim_value)})
                elif dimension.HasField("dim_param"):
                    shape.append({"kind": "param", "value": dimension.dim_param})
                else:
                    shape.append({"kind": "unknown", "value": None})
        contracts[value.name] = {
            "dtype_code": int(tensor_type.elem_type),
            "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type),
            "shape_present": shape_present,
            "shape": shape,
        }
    return contracts


def consumer_connections(
    graph: onnx.GraphProto, tensor_names: Iterable[str]
) -> dict[str, list[dict[str, Any]]]:
    names = set(tensor_names)
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for node in graph.node:
        for slot, input_name in enumerate(node.input):
            if input_name in names:
                result[input_name].append(
                    {
                        "node_name": node.name,
                        "op_type": node.op_type,
                        "input_slot": slot,
                    }
                )
    for items in result.values():
        items.sort(key=lambda item: (item["node_name"], item["input_slot"]))
    return result


def fold_tdb_ifs(model: onnx.ModelProto) -> list[dict[str, Any]]:
    graph = model.graph
    root_context = collect_contexts(graph)[0]
    original_nodes = list(graph.node)
    producer_indices = {
        output: index
        for index, node in enumerate(original_nodes)
        for output in node.output
    }
    replacements: list[onnx.NodeProto] = []
    records: list[dict[str, Any]] = []
    matched_names: list[str] = []

    for node_index, node in enumerate(original_nodes):
        match = TDB_IF_PATTERN.fullmatch(node.name) if node.op_type == "If" else None
        if not match:
            replacements.append(node)
            continue

        matched_names.append(node.name)
        value = evaluate_scalar(root_context, node.input[0])
        if value is not False:
            raise RuntimeError(
                f"{node.name}: expected statically false condition, got {value!r}"
            )
        live_graph = graph_attribute(node, "else_branch")
        dead_graph = graph_attribute(node, "then_branch")
        if len(node.output) != len(live_graph.output):
            raise RuntimeError(
                f"{node.name}: If/live output count mismatch "
                f"{len(node.output)} != {len(live_graph.output)}"
            )

        mappings: list[dict[str, Any]] = []
        for output_index, if_output in enumerate(node.output):
            branch_output, source = live_output_source(
                node, live_graph, output_index
            )
            source_index = producer_indices.get(source)
            source_is_input = any(item.name == source for item in graph.input)
            source_is_initializer = any(
                item.name == source for item in graph.initializer
            )
            source_available_before_if = (
                source_is_input
                or source_is_initializer
                or (source_index is not None and source_index < node_index)
            )
            if not source_available_before_if:
                raise RuntimeError(
                    f"{node.name}: source {source!r} is not available before If"
                )
            replacement = helper.make_node(
                "Identity",
                inputs=[source],
                outputs=[if_output],
                name=f"{node.name}/folded_else_identity_{output_index}",
            )
            replacements.append(replacement)
            consumers = [
                {
                    "node_name": consumer.name,
                    "op_type": consumer.op_type,
                    "input_slots": [
                        slot
                        for slot, name in enumerate(consumer.input)
                        if name == if_output
                    ],
                }
                for consumer in original_nodes
                if if_output in consumer.input
            ]
            mappings.append(
                {
                    "output_index": output_index,
                    "if_output_preserved": if_output,
                    "live_branch_output": branch_output,
                    "live_source": source,
                    "replacement_node": replacement.name,
                    "source_producer_index": source_index,
                    "if_node_index": node_index,
                    "source_available_before_if": source_available_before_if,
                    "consumers_unchanged": consumers,
                }
            )

        records.append(
            {
                "node_name": node.name,
                "tdb_stage": match.group(1),
                "condition_input": node.input[0],
                "condition_value": value,
                "live_branch": "else_branch",
                "dead_branch": "then_branch",
                "if_output_count": len(node.output),
                "replacement_identity_count": len(mappings),
                "dead_branch_node_count": len(dead_graph.node),
                "dead_branch_squeeze_count": sum(
                    item.op_type == "Squeeze" for item in dead_graph.node
                ),
                "live_branch_node_count": len(live_graph.node),
                "mappings": mappings,
            }
        )

    if len(records) != 16:
        raise RuntimeError(
            f"Expected 16 audited TDB If nodes, found {len(records)}: {matched_names}"
        )
    del graph.node[:]
    graph.node.extend(replacements)
    return records


def run_parser_only(
    parser_exe: Path,
    model_path: Path,
    summary_path: Path,
    log_path: Path,
    tensorrt_root: Path,
    cuda_root: Path,
) -> tuple[int, dict[str, Any]]:
    if not parser_exe.is_file():
        raise FileNotFoundError(f"Parser-only executable not found: {parser_exe}")
    env = os.environ.copy()
    path_parts = [str(tensorrt_root / "bin"), str(cuda_root / "bin")]
    env["PATH"] = os.pathsep.join(path_parts + [env.get("PATH", "")])
    command = [str(parser_exe), str(model_path), str(summary_path)]
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(
        "COMMAND=" + subprocess.list2cmdline(command) + "\n\n" + completed.stdout,
        encoding="utf-8",
    )
    if not summary_path.is_file():
        raise RuntimeError(
            "Parser process did not produce parser_summary.json; "
            f"exit_code={completed.returncode}; see {log_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return completed.returncode, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--original-onnx", type=Path, default=ORIGINAL_ONNX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--parser-exe", type=Path, default=DEFAULT_PARSER_EXE)
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TENSORRT_ROOT)
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA_ROOT)
    args = parser.parse_args()

    source = args.input.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash_before = sha256(source)
    original = args.original_onnx.resolve()
    original_hash_before = sha256(original) if original.is_file() else None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.output_root.resolve() / f"{timestamp}_if_folded"
    run_dir.mkdir(parents=True, exist_ok=False)
    output = run_dir / "if_folded.onnx"

    model = onnx.load_model(str(source), load_external_data=False)
    onnx.checker.check_model(model)
    before_total, before_ops = node_counts(model)
    before_graph_count = sum(1 for _ in iter_graphs(model.graph))
    before_squeezes = squeeze_records(model)
    before_if_count = before_ops["If"]
    target_output_names = [
        output_name
        for node in model.graph.node
        if node.op_type == "If" and TDB_IF_PATTERN.fullmatch(node.name)
        for output_name in node.output
    ]
    before_contracts_all = type_contracts(model.graph)
    before_target_contracts = {
        name: before_contracts_all.get(name) for name in target_output_names
    }
    before_consumers = consumer_connections(model.graph, target_output_names)
    before_graph_io = {
        "inputs": [item.name for item in model.graph.input],
        "outputs": [item.name for item in model.graph.output],
    }

    rewrite_records = fold_tdb_ifs(model)
    onnx.save_model(model, str(output))

    check_result: dict[str, Any] = {
        "onnx_path": str(output),
        "checker_passed": False,
        "error": None,
    }
    try:
        checked_model = onnx.load_model(str(output), load_external_data=False)
        onnx.checker.check_model(checked_model)
        check_result["checker_passed"] = True
    except Exception as error:
        check_result["error"] = f"{type(error).__name__}: {error}"
        dump_json(run_dir / "onnx_check_result.json", check_result)
        raise
    dump_json(run_dir / "onnx_check_result.json", check_result)

    after_total, after_ops = node_counts(checked_model)
    after_graph_count = sum(1 for _ in iter_graphs(checked_model.graph))
    after_squeezes = squeeze_records(checked_model)
    after_contracts_all = type_contracts(checked_model.graph)
    after_target_contracts = {
        name: after_contracts_all.get(name) for name in target_output_names
    }
    after_consumers = consumer_connections(checked_model.graph, target_output_names)
    after_graph_io = {
        "inputs": [item.name for item in checked_model.graph.input],
        "outputs": [item.name for item in checked_model.graph.output],
    }
    tensor_contracts_exactly_preserved = (
        before_target_contracts == after_target_contracts
        and all(value is not None for value in before_target_contracts.values())
    )
    consumer_connections_exactly_preserved = before_consumers == after_consumers
    graph_io_names_preserved = before_graph_io == after_graph_io
    after_if_nodes = [
        {"scope": scope, "name": node.name}
        for scope, graph in iter_graphs(checked_model.graph)
        for node in graph.node
        if node.op_type == "If"
    ]
    folded_identity_names = {
        mapping["replacement_node"]
        for record in rewrite_records
        for mapping in record["mappings"]
    }
    found_folded_identities = {
        node.name
        for node in checked_model.graph.node
        if node.op_type == "Identity" and node.name in folded_identity_names
    }
    graph_outputs = {item.name for item in checked_model.graph.output}
    top_level_producers = {
        output_name: node.name
        for node in checked_model.graph.node
        for output_name in node.output
    }
    preserved_outputs = [
        mapping["if_output_preserved"]
        for record in rewrite_records
        for mapping in record["mappings"]
    ]
    all_preserved_outputs_produced = all(
        name in top_level_producers or name in graph_outputs
        for name in preserved_outputs
    )

    graph_audit_passed = (
        before_if_count == 16
        and not after_if_nodes
        and len(rewrite_records) == 16
        and len(found_folded_identities) == 32
        and len(preserved_outputs) == 32
        and all_preserved_outputs_produced
        and tensor_contracts_exactly_preserved
        and consumer_connections_exactly_preserved
        and graph_io_names_preserved
        and all(
            mapping["source_available_before_if"]
            for record in rewrite_records
            for mapping in record["mappings"]
        )
    )
    if not graph_audit_passed:
        raise RuntimeError("Post-rewrite graph audit failed")

    parser_summary_path = run_dir / "parser_summary.json"
    parser_log_path = run_dir / "parser_verbose.log"
    parser_exit_code, parser_summary = run_parser_only(
        args.parser_exe.resolve(),
        output,
        parser_summary_path,
        parser_log_path,
        args.tensorrt_root.resolve(),
        args.cuda_root.resolve(),
    )
    if parser_summary.get("engine_build_called") is not False:
        raise RuntimeError("Parser summary indicates an engine build call")
    if parser_summary.get("inference_called") is not False:
        raise RuntimeError("Parser summary indicates an inference call")

    generated_engines = sorted(
        str(path.relative_to(run_dir)).replace("\\", "/")
        for pattern in ("*.engine", "*.plan")
        for path in run_dir.rglob(pattern)
    )
    if generated_engines:
        raise RuntimeError(f"Unexpected engine artifacts: {generated_engines}")

    source_hash_after = sha256(source)
    if source_hash_after != source_hash_before:
        raise RuntimeError("Input ONNX changed during rewrite")
    output_hash = sha256(output)
    original_hash_after = sha256(original) if original.is_file() else None
    if original_hash_after != original_hash_before:
        raise RuntimeError("Original deployment ONNX changed during rewrite")
    first_error = (
        parser_summary.get("errors", [None])[0]
        if parser_summary.get("errors")
        else None
    )
    parser_success = bool(parser_summary.get("parser_success"))

    op_diff = {
        op_type: {
            "before": before_ops.get(op_type, 0),
            "after": after_ops.get(op_type, 0),
            "delta": after_ops.get(op_type, 0) - before_ops.get(op_type, 0),
        }
        for op_type in sorted(set(before_ops) | set(after_ops))
        if before_ops.get(op_type, 0) != after_ops.get(op_type, 0)
    }
    summary = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "IF_FOLD_REWRITE_AND_PARSER_AUDIT_COMPLETED",
        "input_onnx": str(source),
        "input_alias_from_request": "voxelplugin_rewritten.onnx",
        "input_sha256_before": source_hash_before,
        "input_sha256_after": source_hash_after,
        "input_unchanged": source_hash_before == source_hash_after,
        "original_deployment_onnx": str(original),
        "original_deployment_sha256_before": original_hash_before,
        "original_deployment_sha256_after": original_hash_after,
        "original_deployment_onnx_unchanged": original_hash_before
        == original_hash_after,
        "output_onnx": str(output),
        "output_sha256": output_hash,
        "rewrite": {
            "target_if_count": 16,
            "folded_if_count": len(rewrite_records),
            "condition_value": False,
            "live_branch": "else_branch",
            "replacement_identity_count": len(folded_identity_names),
            "original_output_names_preserved": True,
            "tensor_contracts_exactly_preserved": tensor_contracts_exactly_preserved,
            "consumer_connections_exactly_preserved": consumer_connections_exactly_preserved,
            "graph_io_names_preserved": graph_io_names_preserved,
            "before_target_contracts": before_target_contracts,
            "after_target_contracts": after_target_contracts,
            "before_consumers": before_consumers,
            "after_consumers": after_consumers,
            "records": rewrite_records,
        },
        "onnx_checker_passed": check_result["checker_passed"],
        "graph_audit": {
            "passed": graph_audit_passed,
            "graph_count_before": before_graph_count,
            "graph_count_after": after_graph_count,
            "node_count_before_including_subgraphs": before_total,
            "node_count_after_including_subgraphs": after_total,
            "if_count_before": before_if_count,
            "if_count_after": len(after_if_nodes),
            "squeeze_count_before": len(before_squeezes),
            "squeeze_count_after": len(after_squeezes),
            "dead_branch_squeeze_nodes_removed": len(before_squeezes)
            - len(after_squeezes),
            "folded_identity_count": len(found_folded_identities),
            "preserved_output_tensor_count": len(preserved_outputs),
            "all_preserved_outputs_produced": all_preserved_outputs_produced,
            "op_count_diff": op_diff,
        },
        "parser_audit": {
            "parser_only": True,
            "parser_executable": str(args.parser_exe.resolve()),
            "process_exit_code": parser_exit_code,
            "parser_success": parser_success,
            "parser_error_count": parser_summary.get("parser_error_count"),
            "first_blocking_node": first_error.get("node_name")
            if first_error
            else None,
            "first_blocking_operator": first_error.get("op_type")
            if first_error
            else None,
            "first_blocking_error": first_error.get("description")
            if first_error
            else None,
            "network_layer_count": parser_summary.get("network_layer_count"),
            "network_inputs": parser_summary.get("network_inputs", []),
            "network_outputs": parser_summary.get("network_outputs", []),
            "voxel_unique_plugin_instances_created": parser_summary.get(
                "plugin_creator_build_calls"
            ),
            "standard_plugins_initialized": parser_summary.get(
                "standard_plugins_initialized"
            ),
            "engine_build_called": False,
            "inference_called": False,
        },
        "engine_artifacts": generated_engines,
    }
    dump_json(run_dir / "rewrite_summary.json", summary)

    report = f"""# GCN_res constant If folding graph diff

## Scope

- Input: `{source}`
- Input SHA-256: `{source_hash_before}`
- Output: `{output}`
- Output SHA-256: `{output_hash}`
- Input unchanged: `{source_hash_before == source_hash_after}`
- Engine build: `false`
- Inference: `false`

## Rewrite

- Constant-false If nodes folded: `{len(rewrite_records)}/16`
- Live path: `else_branch`
- Replacement: parent-graph `Identity(live_source -> original If output)`
- Original If output names preserved: `true`
- Dtype/shape metadata exactly preserved: `{tensor_contracts_exactly_preserved}`
- Consumer node/input slots exactly preserved: `{consumer_connections_exactly_preserved}`
- Graph input/output names preserved: `{graph_io_names_preserved}`
- Replacement Identity nodes: `{len(found_folded_identities)}`
- Downstream consumers remain connected to the same tensor names: `true`

Removing each If removes both embedded branch GraphProto objects. The live branch
only forwarded parent tensors through Identity nodes, so equivalent parent-graph
Identity nodes were inserted. The dead then branches and their Squeeze nodes are
therefore absent from the candidate graph.

## Static graph diff

- Graphs including subgraphs: `{before_graph_count}` -> `{after_graph_count}`
- Nodes including subgraphs: `{before_total}` -> `{after_total}`
- If: `{before_if_count}` -> `{len(after_if_nodes)}`
- Squeeze: `{len(before_squeezes)}` -> `{len(after_squeezes)}`
- Dead-branch Squeeze removed: `{len(before_squeezes) - len(after_squeezes)}`
- ONNX checker: `{'PASS' if check_result['checker_passed'] else 'FAIL'}`
- Post-rewrite graph audit: `{'PASS' if graph_audit_passed else 'FAIL'}`

Changed op counts:

```json
{json.dumps(op_diff, ensure_ascii=False, indent=2)}
```

## TensorRT parser-only audit

- Standard plugins initialized: `{parser_summary.get('standard_plugins_initialized')}`
- VoxelUnique creator instances: `{parser_summary.get('plugin_creator_build_calls')}`
- Parser success: `{parser_success}`
- Parser errors: `{parser_summary.get('parser_error_count')}`
- First blocking node: `{first_error.get('node_name') if first_error else None}`
- First blocking operator: `{first_error.get('op_type') if first_error else None}`
- First blocking error: `{first_error.get('description') if first_error else None}`
- Engine build called: `false`
- Inference called: `false`

The parser result is an audit result only. No engine, inference, FP16, or
benchmark operation was performed.

## Status

IF_FOLD_REWRITE_AND_PARSER_AUDIT_COMPLETED
"""
    (run_dir / "graph_diff_report.md").write_text(report, encoding="utf-8")

    print(f"RUN_DIR={run_dir}")
    print(f"INPUT_ONNX_UNCHANGED={str(source_hash_before == source_hash_after).lower()}")
    print(f"IF_FOLDED_COUNT={len(rewrite_records)}")
    print(f"IF_COUNT_AFTER={len(after_if_nodes)}")
    print(f"DEAD_BRANCH_SQUEEZE_REMOVED={len(before_squeezes) - len(after_squeezes)}")
    print(f"ONNX_CHECKER_PASSED={check_result['checker_passed']}")
    print(f"GRAPH_AUDIT_PASSED={graph_audit_passed}")
    print(f"TENSOR_CONTRACTS_EXACTLY_PRESERVED={tensor_contracts_exactly_preserved}")
    print(f"CONSUMER_CONNECTIONS_EXACTLY_PRESERVED={consumer_connections_exactly_preserved}")
    print(f"TENSORRT_PARSER_SUCCESS={parser_success}")
    if first_error:
        print(f"FIRST_BLOCKING_NODE={first_error.get('node_name')}")
        print(f"FIRST_BLOCKING_OPERATOR={first_error.get('op_type')}")
    print("ENGINE_BUILD_CALLED=false")
    print("INFERENCE_CALLED=false")
    print("IF_FOLD_REWRITE_AND_PARSER_AUDIT_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
