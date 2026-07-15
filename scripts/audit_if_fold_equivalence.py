"""Read-only tensor-contract audit for folding the 16 constant-false TDB If nodes."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import onnx

from audit_constant_if_branches import branch_graphs, graph_context_for
from audit_dynamic_squeeze import (
    DEFAULT_ONNX,
    ORIGINAL_ONNX,
    GraphContext,
    collect_contexts,
    describe_node,
    evaluate_scalar,
    resolve_initializer,
    resolve_producer,
    sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
TDB_PATTERN = re.compile(r"/model/(tdb_[1-4])/If(?:_|$)")


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def dimension_value(dimension: onnx.TensorShapeProto.Dimension) -> int | str:
    if dimension.HasField("dim_value"):
        return int(dimension.dim_value)
    if dimension.HasField("dim_param"):
        return dimension.dim_param
    return "unknown"


def contract_from_value_info(value: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value.type.tensor_type
    shape_present = tensor_type.HasField("shape")
    shape = (
        [dimension_value(item) for item in tensor_type.shape.dim]
        if shape_present
        else None
    )
    return {
        "name": value.name,
        "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type),
        "shape_present": shape_present,
        "shape": shape,
        "rank": len(shape) if shape is not None else None,
    }


def tensor_contract(context: GraphContext, name: str) -> dict[str, Any]:
    current: GraphContext | None = context
    while current is not None:
        for value in (*current.graph.input, *current.graph.output, *current.graph.value_info):
            if value.name == name:
                result = contract_from_value_info(value)
                result["declared_in_scope"] = current.scope
                return result
        initializer = resolve_initializer(current, name)
        if initializer and initializer[0] is current:
            tensor = initializer[1]
            return {
                "name": name,
                "dtype": onnx.TensorProto.DataType.Name(tensor.data_type),
                "shape_present": True,
                "shape": list(tensor.dims),
                "rank": len(tensor.dims),
                "declared_in_scope": current.scope,
            }
        current = current.parent
    return {
        "name": name,
        "dtype": "UNKNOWN",
        "shape_present": False,
        "shape": None,
        "rank": None,
        "declared_in_scope": None,
    }


def shapes_compatible(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if not left["shape_present"] or not right["shape_present"]:
        return {
            "compatible": True,
            "exact_metadata_match": left["shape_present"] == right["shape_present"]
            and left["shape"] == right["shape"],
            "reason": "unknown rank imposes no conflicting shape constraint",
        }
    if left["rank"] != right["rank"]:
        return {
            "compatible": False,
            "exact_metadata_match": False,
            "reason": f"rank mismatch: {left['rank']} != {right['rank']}",
        }
    conflicts: list[dict[str, Any]] = []
    for index, (left_dim, right_dim) in enumerate(zip(left["shape"], right["shape"])):
        if (
            isinstance(left_dim, int)
            and isinstance(right_dim, int)
            and left_dim != right_dim
        ):
            conflicts.append(
                {"axis": index, "left": left_dim, "right": right_dim}
            )
    return {
        "compatible": not conflicts,
        "exact_metadata_match": left["shape"] == right["shape"],
        "reason": "static dimensions compatible" if not conflicts else "static dimension conflict",
        "conflicts": conflicts,
    }


def producer_available_in_parent(context: GraphContext, name: str) -> dict[str, Any]:
    producer = resolve_producer(context, name)
    initializer = resolve_initializer(context, name)
    graph_input = any(value.name == name for value in context.graph.input)
    contract = tensor_contract(context, name)
    available = (
        producer is not None
        or initializer is not None
        or graph_input
        or contract["dtype"] != "UNKNOWN"
    )
    return {
        "available": available,
        "kind": (
            "node_output"
            if producer is not None
            else "initializer"
            if initializer is not None
            else "graph_input"
            if graph_input
            else "declared_tensor"
            if contract["dtype"] != "UNKNOWN"
            else "unknown"
        ),
        "producer": describe_node(*producer) if producer is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--original-onnx", type=Path, default=ORIGINAL_ONNX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    onnx_path = args.onnx.resolve()
    original_path = args.original_onnx.resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    rewritten_hash_before = sha256(onnx_path)
    original_hash_before = sha256(original_path) if original_path.is_file() else None
    model = onnx.load_model(str(onnx_path), load_external_data=False)
    onnx.checker.check_model(model)
    contexts = collect_contexts(model.graph)

    candidates: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for context in contexts:
        for node in context.graph.node:
            if node.op_type != "If" or not TDB_PATTERN.search(node.name):
                continue
            condition_value = evaluate_scalar(context, node.input[0])
            branches = branch_graphs(node)
            else_context = graph_context_for(
                contexts, context, node, "else_branch"
            )
            live_outputs = list(else_context.graph.output)
            output_count_match = len(node.output) == len(live_outputs)
            candidate_mappings: list[dict[str, Any]] = []
            if output_count_match:
                for index, (if_output_name, live_output_info) in enumerate(
                    zip(node.output, live_outputs)
                ):
                    live_output_name = live_output_info.name
                    live_producer = resolve_producer(else_context, live_output_name)
                    live_identity = bool(
                        live_producer
                        and live_producer[0] is else_context
                        and live_producer[1].op_type == "Identity"
                        and len(live_producer[1].input) == 1
                    )
                    source_name = (
                        live_producer[1].input[0] if live_identity else live_output_name
                    )
                    if_contract = tensor_contract(context, if_output_name)
                    live_contract = contract_from_value_info(live_output_info)
                    live_contract["declared_in_scope"] = else_context.scope
                    source_contract = tensor_contract(context, source_name)
                    dtype_match = (
                        if_contract["dtype"]
                        == live_contract["dtype"]
                        == source_contract["dtype"]
                        and if_contract["dtype"] != "UNKNOWN"
                    )
                    if_live_shape = shapes_compatible(if_contract, live_contract)
                    live_source_shape = shapes_compatible(
                        live_contract, source_contract
                    )
                    if_source_shape = shapes_compatible(
                        if_contract, source_contract
                    )
                    source_availability = producer_available_in_parent(
                        context, source_name
                    )
                    consumers = context.consumers.get(if_output_name, [])
                    consumer_records = [
                        {
                            "consumer_node": consumer.name,
                            "consumer_op_type": consumer.op_type,
                            "input_slot": slot,
                            "old_tensor": if_output_name,
                            "replacement_tensor": source_name,
                            "slot_matches_old_tensor": consumer.input[slot]
                            == if_output_name,
                        }
                        for consumer, slot in consumers
                    ]
                    parent_graph_output_indices = [
                        output_index
                        for output_index, graph_output in enumerate(context.graph.output)
                        if graph_output.name == if_output_name
                    ]
                    consumer_connections_preservable = (
                        live_identity
                        and source_availability["available"]
                        and all(
                            item["slot_matches_old_tensor"]
                            for item in consumer_records
                        )
                        and not parent_graph_output_indices
                    )
                    mapping_passed = (
                        dtype_match
                        and if_live_shape["compatible"]
                        and live_source_shape["compatible"]
                        and if_source_shape["compatible"]
                        and consumer_connections_preservable
                    )
                    mapping = {
                        "if_node": node.name,
                        "output_index": index,
                        "if_output": if_contract,
                        "live_branch_output": live_contract,
                        "live_output_producer": describe_node(*live_producer)
                        if live_producer
                        else None,
                        "live_output_is_single_input_identity": live_identity,
                        "replacement_source": source_contract,
                        "replacement_source_availability": source_availability,
                        "dtype_match": dtype_match,
                        "shape_checks": {
                            "if_vs_live": if_live_shape,
                            "live_vs_source": live_source_shape,
                            "if_vs_source": if_source_shape,
                            "all_compatible": if_live_shape["compatible"]
                            and live_source_shape["compatible"]
                            and if_source_shape["compatible"],
                        },
                        "consumers": consumer_records,
                        "consumer_count": len(consumer_records),
                        "parent_graph_output_indices": parent_graph_output_indices,
                        "consumer_connections_preservable": consumer_connections_preservable,
                        "mapping_passed": mapping_passed,
                    }
                    candidate_mappings.append(mapping)
                    mappings.append(mapping)
            condition_is_constant_false = condition_value is False
            candidate_passed = (
                condition_is_constant_false
                and output_count_match
                and len(candidate_mappings) == len(node.output)
                and all(item["mapping_passed"] for item in candidate_mappings)
            )
            candidates.append(
                {
                    "node_name": node.name,
                    "node_index_in_scope": context.node_indices.get(node.name, -1),
                    "parent_graph_scope": context.scope,
                    "tdb_stage": TDB_PATTERN.search(node.name).group(1),
                    "condition_input": node.input[0],
                    "condition_value": condition_value,
                    "condition_is_constant_false": condition_is_constant_false,
                    "live_branch": "else_branch",
                    "dead_branch": "then_branch",
                    "if_outputs": [
                        tensor_contract(context, name) for name in node.output
                    ],
                    "live_branch_output_tensors": [
                        contract_from_value_info(item) for item in live_outputs
                    ],
                    "output_count_match": output_count_match,
                    "output_count_before": len(node.output),
                    "output_count_after_candidate": len(live_outputs),
                    "tensor_mapping_count": len(candidate_mappings),
                    "tensor_mappings": candidate_mappings,
                    "candidate_passed": candidate_passed,
                    "rewrite_performed": False,
                }
            )

    if len(candidates) != 16:
        raise RuntimeError(f"Expected 16 TDB If candidates, found {len(candidates)}")
    all_candidates_passed = all(item["candidate_passed"] for item in candidates)
    all_mappings_passed = all(item["mapping_passed"] for item in mappings)
    output_count_match_all = all(item["output_count_match"] for item in candidates)
    dtype_match_all = all(item["dtype_match"] for item in mappings)
    shape_compatible_all = all(
        item["shape_checks"]["all_compatible"] for item in mappings
    )
    consumers_preservable_all = all(
        item["consumer_connections_preservable"] for item in mappings
    )
    unknown_if_output_shape_count = sum(
        not item["if_output"]["shape_present"] for item in mappings
    )
    exact_if_source_shape_count = sum(
        item["shape_checks"]["if_vs_source"]["exact_metadata_match"]
        for item in mappings
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.output_root.resolve() / f"{timestamp}_if_fold_audit"
    run_dir.mkdir(parents=True, exist_ok=False)
    common = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "onnx_path": str(onnx_path),
        "onnx_sha256": rewritten_hash_before,
        "onnx_checker_passed": True,
        "read_only": True,
        "engine_build_called": False,
        "inference_called": False,
    }
    candidate_payload = {
        **common,
        "candidate_count": len(candidates),
        "all_candidates_passed": all_candidates_passed,
        "checks": {
            "all_conditions_constant_false": all(
                item["condition_is_constant_false"] for item in candidates
            ),
            "output_count_match_all": output_count_match_all,
            "dtype_match_all": dtype_match_all,
            "shape_contract_compatible_all": shape_compatible_all,
            "consumer_connections_preservable_all": consumers_preservable_all,
            "all_tensor_mappings_passed": all_mappings_passed,
        },
        "candidates": candidates,
    }
    mapping_payload = {
        **common,
        "mapping_count": len(mappings),
        "all_mappings_passed": all_mappings_passed,
        "unknown_if_output_shape_count": unknown_if_output_shape_count,
        "exact_if_source_shape_metadata_match_count": exact_if_source_shape_count,
        "shape_policy": {
            "exact_static_dimension_conflicts_fail": True,
            "unknown_rank_is_compatible_with_more_specific_live_shape": True,
            "symbolic_dimension_names_need_not_match": True,
        },
        "mappings": mappings,
    }
    dump_json(run_dir / "if_fold_candidates.json", candidate_payload)
    dump_json(run_dir / "tensor_mapping.json", mapping_payload)

    rewritten_hash_after = sha256(onnx_path)
    original_hash_after = sha256(original_path) if original_path.is_file() else None
    if (
        rewritten_hash_after != rewritten_hash_before
        or original_hash_after != original_hash_before
    ):
        raise RuntimeError("ONNX hash changed during read-only fold audit")
    integrity = {
        "rewritten_sha256_before": rewritten_hash_before,
        "rewritten_sha256_after": rewritten_hash_after,
        "rewritten_unchanged": True,
        "original_sha256_before": original_hash_before,
        "original_sha256_after": original_hash_after,
        "original_unchanged": True,
    }
    candidate_payload["source_integrity"] = integrity
    mapping_payload["source_integrity"] = integrity
    dump_json(run_dir / "if_fold_candidates.json", candidate_payload)
    dump_json(run_dir / "tensor_mapping.json", mapping_payload)

    rows = "\n".join(
        f"| `{item['node_name']}` | {item['output_count_before']} | "
        f"{item['output_count_match']} | "
        f"{all(mapping['dtype_match'] for mapping in item['tensor_mappings'])} | "
        f"{all(mapping['shape_checks']['all_compatible'] for mapping in item['tensor_mappings'])} | "
        f"{all(mapping['consumer_connections_preservable'] for mapping in item['tensor_mappings'])} | "
        f"{item['candidate_passed']} |"
        for item in candidates
    )
    conclusion = (
        "IF_FOLD_EQUIVALENCE_CONFIRMED"
        if all_candidates_passed
        else "IF_FOLD_EQUIVALENCE_NOT_CONFIRMED"
    )
    report = f"""# GCN_res If folding equivalence audit

## Scope and constraints

- Input ONNX: `{onnx_path}`
- SHA-256: `{rewritten_hash_before}`
- Candidate If nodes: {len(candidates)}
- Tensor mappings: {len(mappings)}
- Read-only: true
- Engine build: false
- Inference: false
- Rewritten and original ONNX unchanged: PASS

## Equivalence policy

Every candidate must have a statically false condition, the same output count,
matching dtype, compatible shape contracts, and a live `else_branch` Identity
whose captured source tensor already exists in the parent graph. Every existing
consumer input slot must be rewritable from the If output to that source.

Shape metadata is reported separately from semantic compatibility. An absent
If-output shape means unknown rank, not a scalar; it introduces no conflicting
constraint when the live output provides a more specific shape. Any static rank
or dimension conflict fails the candidate.

## Results

- Output counts match: {output_count_match_all}
- Dtypes match: {dtype_match_all}
- Shape contracts compatible: {shape_compatible_all}
- Consumer connections preservable: {consumers_preservable_all}
- Unknown-rank If output mappings: {unknown_if_output_shape_count}
- Exact If/source shape metadata matches: {exact_if_source_shape_count}/{len(mappings)}

| If node | Outputs | Count | Dtype | Shape | Consumers | Passed |
|---|---:|---|---|---|---|---|
{rows}

## Candidate mapping

For each output, `tensor_mapping.json` records:

```text
If output
    -> live else_branch graph output
    -> Identity input captured from the parent graph
    -> original consumer node/input slots
```

No mapping was applied. This confirms static graph-contract equivalence only;
an authorized rewrite must still be followed by ONNX checker, parser audit, and
numerical parity against the unchanged deployment reference.

## Conclusion

{conclusion}
"""
    (run_dir / "equivalence_report.md").write_text(report, encoding="utf-8")
    print(f"RUN_DIR={run_dir}")
    print(f"IF_FOLD_CANDIDATE_COUNT={len(candidates)}")
    print(f"TENSOR_MAPPING_COUNT={len(mappings)}")
    print(f"OUTPUT_COUNT_MATCH_ALL={output_count_match_all}")
    print(f"DTYPE_MATCH_ALL={dtype_match_all}")
    print(f"SHAPE_CONTRACT_COMPATIBLE_ALL={shape_compatible_all}")
    print(f"CONSUMER_CONNECTIONS_PRESERVABLE_ALL={consumers_preservable_all}")
    print("REWRITTEN_ONNX_UNCHANGED=true")
    print("ORIGINAL_ONNX_UNCHANGED=true")
    print(conclusion)
    return 0 if all_candidates_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
