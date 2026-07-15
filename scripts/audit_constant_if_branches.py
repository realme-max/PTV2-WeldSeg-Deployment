"""Read-only recursive audit of constant and dynamic ONNX If conditions."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import onnx

from audit_dynamic_squeeze import (
    DEFAULT_ONNX,
    ORIGINAL_ONNX,
    GraphContext,
    collect_contexts,
    describe_node,
    evaluate_scalar,
    jsonable,
    resolve_producer,
    resolve_tensor_info,
    sha256,
    trace_upstream,
    value_info,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
TDB_PATTERN = re.compile(r"/model/(tdb_[1-4])/If(?:_|$)")


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def branch_graphs(node: onnx.NodeProto) -> dict[str, onnx.GraphProto]:
    return {
        attribute.name: attribute.g
        for attribute in node.attribute
        if attribute.type == onnx.AttributeProto.GRAPH
        and attribute.name in ("then_branch", "else_branch")
    }


def graph_context_for(
    contexts: list[GraphContext],
    parent_context: GraphContext,
    parent_node: onnx.NodeProto,
    branch: str,
) -> GraphContext:
    matches = [
        context
        for context in contexts
        if context.parent is parent_context
        and context.parent_node is not None
        and context.parent_node.name == parent_node.name
        and context.parent_attribute == branch
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {parent_node.name}:{branch} context, found {len(matches)}"
        )
    return matches[0]


def branch_description(context: GraphContext) -> dict[str, Any]:
    output_records: list[dict[str, Any]] = []
    for index, output in enumerate(context.graph.output):
        producer = resolve_producer(context, output.name)
        output_records.append(
            {
                "index": index,
                "tensor": value_info(output),
                "producer": describe_node(*producer) if producer else None,
            }
        )
    return {
        "scope": context.scope,
        "node_count": len(context.graph.node),
        "nodes": [describe_node(context, node) for node in context.graph.node],
        "graph_inputs": [value_info(item) for item in context.graph.input],
        "graph_outputs": output_records,
    }


def folding_strategy(live_branch: dict[str, Any]) -> dict[str, Any]:
    output_producers = [item["producer"] for item in live_branch["graph_outputs"]]
    all_identity = bool(output_producers) and all(
        producer is not None and producer["op_type"] == "Identity"
        for producer in output_producers
    )
    if all_identity and live_branch["node_count"] == len(output_producers):
        strategy = "REWIRE_OR_IDENTITY_REPLACEMENT_CANDIDATE"
    else:
        strategy = "INLINE_LIVE_BRANCH_SUBGRAPH_CANDIDATE"
    return {
        "strategy": strategy,
        "live_branch_node_count": live_branch["node_count"],
        "live_branch_output_count": len(live_branch["graph_outputs"]),
        "all_live_outputs_from_identity": all_identity,
        "requires_tensor_name_collision_check": live_branch["node_count"] > 0,
        "requires_shape_and_dtype_equivalence_test": True,
        "rewrite_performed": False,
    }


def classify_condition(value: Any) -> tuple[str, str | None, str | None]:
    if value is True:
        return "A_CONSTANT_TRUE", "then_branch", "else_branch"
    if value is False:
        return "B_CONSTANT_FALSE", "else_branch", "then_branch"
    return "C_RUNTIME_DYNAMIC", None, None


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
    opsets = {item.domain or "ai.onnx": int(item.version) for item in model.opset_import}

    if_records: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    dead_candidates: list[dict[str, Any]] = []
    for context in contexts:
        for node in context.graph.node:
            if node.op_type != "If":
                continue
            branches = branch_graphs(node)
            if set(branches) != {"then_branch", "else_branch"}:
                raise RuntimeError(f"Malformed If branches at {node.name}: {branches}")
            condition_name = node.input[0]
            condition_value = evaluate_scalar(context, condition_name)
            classification, live_branch_name, dead_branch_name = classify_condition(
                condition_value
            )
            stage_match = TDB_PATTERN.search(node.name)
            stage = stage_match.group(1) if stage_match else None
            then_context = graph_context_for(contexts, context, node, "then_branch")
            else_context = graph_context_for(contexts, context, node, "else_branch")
            branch_details = {
                "then_branch": branch_description(then_context),
                "else_branch": branch_description(else_context),
            }
            record = describe_node(context, node)
            record.update(
                {
                    "opset": opsets.get(node.domain or "ai.onnx"),
                    "tdb_stage": stage,
                    "condition_input": condition_name,
                    "condition_tensor": resolve_tensor_info(context, condition_name),
                    "condition_static_value": jsonable(condition_value),
                    "classification": classification,
                    "live_branch": live_branch_name,
                    "dead_branch": dead_branch_name,
                    "branches": branch_details,
                }
            )
            if_records.append(record)
            trace = {
                "node_name": node.name,
                "parent_graph_scope": context.scope,
                "tdb_stage": stage,
                "condition_input": condition_name,
                "condition_static_value": jsonable(condition_value),
                "classification": classification,
                "condition_source_chain": trace_upstream(context, condition_name),
            }
            trace_records.append(trace)
            if live_branch_name and dead_branch_name:
                live = branch_details[live_branch_name]
                dead = branch_details[dead_branch_name]
                if len(live["graph_outputs"]) != len(node.output):
                    raise RuntimeError(f"Live branch output mismatch at {node.name}")
                candidate = {
                    "node_name": node.name,
                    "node_index_in_scope": record["node_index_in_scope"],
                    "parent_graph_scope": context.scope,
                    "tdb_stage": stage,
                    "condition_input": condition_name,
                    "condition_static_value": jsonable(condition_value),
                    "classification": classification,
                    "live_branch": live_branch_name,
                    "dead_branch": dead_branch_name,
                    "if_outputs": [
                        resolve_tensor_info(context, name) for name in node.output
                    ],
                    "live_branch_outputs": live["graph_outputs"],
                    "dead_branch_node_count": dead["node_count"],
                    "dead_branch_nodes": dead["nodes"],
                    "folding_assessment": folding_strategy(live),
                    "condition_proof_reference": node.name,
                }
                dead_candidates.append(candidate)

    if not if_records:
        raise RuntimeError("No If nodes found")
    classification_counts: dict[str, int] = {}
    for item in if_records:
        key = item["classification"]
        classification_counts[key] = classification_counts.get(key, 0) + 1
    tdb_records = [item for item in if_records if item["tdb_stage"] is not None]
    stage_summary: dict[str, dict[str, int]] = {}
    for stage in ("tdb_1", "tdb_2", "tdb_3", "tdb_4"):
        stage_items = [item for item in tdb_records if item["tdb_stage"] == stage]
        stage_summary[stage] = {
            "if_count": len(stage_items),
            "constant_true": sum(
                item["classification"] == "A_CONSTANT_TRUE" for item in stage_items
            ),
            "constant_false": sum(
                item["classification"] == "B_CONSTANT_FALSE" for item in stage_items
            ),
            "runtime_dynamic": sum(
                item["classification"] == "C_RUNTIME_DYNAMIC" for item in stage_items
            ),
        }
    all_tdb_if_constant = bool(tdb_records) and all(
        item["classification"] != "C_RUNTIME_DYNAMIC" for item in tdb_records
    )
    all_if_constant = all(
        item["classification"] != "C_RUNTIME_DYNAMIC" for item in if_records
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.output_root.resolve() / f"{timestamp}_if_audit"
    run_dir.mkdir(parents=True, exist_ok=False)
    common = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "onnx_path": str(onnx_path),
        "onnx_sha256": rewritten_hash_before,
        "onnx_checker_passed": True,
    }
    if_payload = {
        **common,
        "graph_scope_count": len(contexts),
        "if_count": len(if_records),
        "classification_counts": classification_counts,
        "tdb_if_count": len(tdb_records),
        "all_if_constant": all_if_constant,
        "all_tdb_if_constant": all_tdb_if_constant,
        "tdb_stage_summary": stage_summary,
        "nodes": if_records,
    }
    trace_payload = {
        **common,
        "trace_count": len(trace_records),
        "traces": trace_records,
    }
    folding_decision = {
        "all_tdb_if_constant": all_tdb_if_constant,
        "all_if_constant": all_if_constant,
        "if_folding_candidate_count": len(dead_candidates),
        "assessment": (
            "TDB_IF_FOLDING_CANDIDATE_REQUIRES_EQUIVALENCE_AUDIT"
            if all_tdb_if_constant
            else "TDB_IF_FOLDING_NOT_GLOBALLY_PROVEN"
        ),
        "rewrite_performed": False,
        "required_before_rewrite": [
            "verify condition values with an independent evaluator",
            "verify live branch output dtype and shape against If outputs",
            "verify all affected downstream tensors and final logits",
            "preserve tensor names or update every consumer deterministically",
        ],
    }
    dead_payload = {
        **common,
        "summary": folding_decision,
        "candidates": dead_candidates,
    }
    dump_json(run_dir / "if_nodes.json", if_payload)
    dump_json(run_dir / "condition_trace.json", trace_payload)
    dump_json(run_dir / "dead_branch_candidates.json", dead_payload)

    rewritten_hash_after = sha256(onnx_path)
    original_hash_after = sha256(original_path) if original_path.is_file() else None
    if (
        rewritten_hash_after != rewritten_hash_before
        or original_hash_after != original_hash_before
    ):
        raise RuntimeError("ONNX hash changed during read-only If audit")
    integrity = {
        "rewritten_sha256_before": rewritten_hash_before,
        "rewritten_sha256_after": rewritten_hash_after,
        "rewritten_unchanged": True,
        "original_sha256_before": original_hash_before,
        "original_sha256_after": original_hash_after,
        "original_unchanged": True,
    }
    if_payload["source_integrity"] = integrity
    trace_payload["source_integrity"] = integrity
    dead_payload["source_integrity"] = integrity
    dump_json(run_dir / "if_nodes.json", if_payload)
    dump_json(run_dir / "condition_trace.json", trace_payload)
    dump_json(run_dir / "dead_branch_candidates.json", dead_payload)

    tdb_lines = "\n".join(
        f"| {stage} | {values['if_count']} | {values['constant_true']} | "
        f"{values['constant_false']} | {values['runtime_dynamic']} |"
        for stage, values in stage_summary.items()
    )
    candidate_lines = "\n".join(
        f"| `{item['node_name']}` | {item['condition_static_value']} | "
        f"{item['live_branch']} | {item['dead_branch']} | "
        f"{item['folding_assessment']['strategy']} |"
        for item in dead_candidates
    )
    report = f"""# GCN_res constant If branch audit

## Scope and integrity

- Rewritten ONNX: `{onnx_path}`
- SHA-256: `{rewritten_hash_before}`
- Recursive graph scopes: {len(contexts)}
- If nodes: {len(if_records)}
- TDB If nodes: {len(tdb_records)}
- Classification counts: `{classification_counts}`
- Rewritten ONNX unchanged: PASS
- Original ONNX unchanged: PASS

This was a read-only audit. No branch was removed or folded, and no ONNX was
re-exported or rewritten. No Engine build or inference was executed.

## Classification

- A: condition statically evaluates to true; `else_branch` is dead.
- B: condition statically evaluates to false; `then_branch` is dead.
- C: condition remains runtime dynamic; neither branch is proven dead.

## TDB summary

| Stage | If count | Constant true | Constant false | Runtime dynamic |
|---|---:|---:|---:|---:|
{tdb_lines}

All TDB If conditions static: **{all_tdb_if_constant}**.

## Dead branch candidates

| If node | Condition | Live branch | Dead branch | Candidate strategy |
|---|---|---|---|---|
{candidate_lines}

## If folding assessment

- Assessment: `{folding_decision['assessment']}`
- Candidate count: {len(dead_candidates)}
- Rewrite performed: false

Static condition evaluation proves branch reachability, but it does not by
itself authorize graph mutation. The live branch output mapping, dtype, shape,
downstream tensors, and final logits must be independently aligned before a
derived ONNX rewrite is allowed.

## Conclusion

CONSTANT_IF_BRANCH_AUDIT_COMPLETED
"""
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"RUN_DIR={run_dir}")
    print(f"IF_NODE_COUNT={len(if_records)}")
    print(f"TDB_IF_NODE_COUNT={len(tdb_records)}")
    print(f"CLASSIFICATION_COUNTS={classification_counts}")
    print(f"ALL_TDB_IF_CONSTANT={all_tdb_if_constant}")
    print(f"IF_FOLDING_ASSESSMENT={folding_decision['assessment']}")
    print("REWRITTEN_ONNX_UNCHANGED=true")
    print("ORIGINAL_ONNX_UNCHANGED=true")
    print("CONSTANT_IF_BRANCH_AUDIT_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
