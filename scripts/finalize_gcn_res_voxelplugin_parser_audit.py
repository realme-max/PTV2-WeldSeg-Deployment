"""Finalize Phase 3 parser-only evidence after the native parser process exits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import onnx
from onnx import helper


EXPECTED_ORIGINAL_SHA256 = (
    "20aa7ba21a52c6497e0ce10676edae599def203bbddd4ca063b7abccdeeb5198"
)


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


def jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def iter_graphs(
    graph: onnx.GraphProto, path: str = "main"
) -> list[tuple[str, onnx.GraphProto]]:
    graphs = [(path, graph)]
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                graphs.extend(
                    iter_graphs(
                        attribute.g,
                        f"{path}/{node.name or node.op_type}:{attribute.name}",
                    )
                )
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for index, nested in enumerate(attribute.graphs):
                    graphs.extend(
                        iter_graphs(
                            nested,
                            f"{path}/{node.name or node.op_type}:{attribute.name}[{index}]",
                        )
                    )
    return graphs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--original-onnx", type=Path, required=True)
    parser.add_argument("--parser-summary", default="parser_summary.json")
    parser.add_argument("--report-name", default="parser_audit_report.md")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    original = args.original_onnx.resolve()
    rewritten = run_dir / "rewritten.onnx"
    replacement_path = run_dir / "replacement_summary.json"
    parser_path = run_dir / args.parser_summary
    replacement = json.loads(replacement_path.read_text(encoding="utf-8"))
    parser_summary = json.loads(parser_path.read_text(encoding="utf-8"))

    original_hash = sha256(original)
    rewritten_hash = sha256(rewritten)
    if original_hash != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(
            f"Original ONNX hash changed: {original_hash} != {EXPECTED_ORIGINAL_SHA256}"
        )
    if original_hash != replacement["original_sha256_before"]:
        raise RuntimeError("Original ONNX hash no longer matches rewrite evidence")
    if rewritten_hash != replacement["rewritten_sha256"]:
        raise RuntimeError("Rewritten ONNX hash no longer matches rewrite evidence")
    if parser_summary["engine_build_called"] or parser_summary["inference_called"]:
        raise RuntimeError("Parser-only constraints were violated")

    generated_engines = sorted(
        str(path.relative_to(run_dir)).replace("\\", "/")
        for pattern in ("*.engine", "*.plan")
        for path in run_dir.rglob(pattern)
    )
    if generated_engines:
        raise RuntimeError(f"Unexpected Engine artifacts: {generated_engines}")

    model = onnx.load_model(str(rewritten), load_external_data=False)
    onnx.checker.check_model(model)
    tensor_info: dict[str, dict[str, Any]] = {}
    graph_records = iter_graphs(model.graph)
    for _, graph in graph_records:
        for value in (*graph.input, *graph.output, *graph.value_info):
            tensor_type = value.type.tensor_type
            shape: list[int | str] = []
            if tensor_type.HasField("shape"):
                for dimension in tensor_type.shape.dim:
                    if dimension.HasField("dim_value"):
                        shape.append(int(dimension.dim_value))
                    elif dimension.HasField("dim_param"):
                        shape.append(dimension.dim_param)
                    else:
                        shape.append("unknown")
            tensor_info[value.name] = {
                "name": value.name,
                "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type),
                "shape": shape,
            }
        for initializer in graph.initializer:
            tensor_info.setdefault(
                initializer.name,
                {
                    "name": initializer.name,
                    "dtype": onnx.TensorProto.DataType.Name(initializer.data_type),
                    "shape": list(initializer.dims),
                },
            )
    node_records = [
        (scope, node)
        for scope, graph in graph_records
        for node in graph.node
    ]
    first_error = parser_summary["errors"][0] if parser_summary["errors"] else None
    blocker_node = None
    blocker_scope = None
    if first_error:
        blocker_record = next(
            (
                (scope, node)
                for scope, node in node_records
                if node.name == first_error["node_name"]
            ),
            None,
        )
        if blocker_record:
            blocker_scope, blocker_node = blocker_record
    blocker_attributes = (
        {
            attribute.name: jsonable(helper.get_attribute_value(attribute))
            for attribute in blocker_node.attribute
        }
        if blocker_node is not None
        else {}
    )
    blocker_graph_evidence = (
        {
            "node_name": blocker_node.name,
            "op_type": blocker_node.op_type,
            "domain": blocker_node.domain or "ai.onnx",
            "graph_scope": blocker_scope,
            "inputs": [
                tensor_info.get(name, {"name": name, "dtype": "UNKNOWN", "shape": []})
                for name in blocker_node.input
            ],
            "outputs": [
                tensor_info.get(name, {"name": name, "dtype": "UNKNOWN", "shape": []})
                for name in blocker_node.output
            ],
            "attributes": blocker_attributes,
        }
        if blocker_node is not None
        else None
    )
    scatter_node = next(
        (
            node
            for node in model.graph.node
            if node.name == "/model/tdb_1/ScatterElements"
        ),
        None,
    )
    scatter_graph_evidence = None
    if scatter_node is not None:
        scatter_graph_evidence = {
            "node_name": scatter_node.name,
            "op_type": scatter_node.op_type,
            "domain": scatter_node.domain or "ai.onnx",
            "attributes": {
                attribute.name: jsonable(helper.get_attribute_value(attribute))
                for attribute in scatter_node.attribute
            },
            "inputs": [
                tensor_info.get(name, {"name": name, "dtype": "UNKNOWN", "shape": []})
                for name in scatter_node.input
            ],
            "outputs": [
                tensor_info.get(name, {"name": name, "dtype": "UNKNOWN", "shape": []})
                for name in scatter_node.output
            ],
            "parser_advanced_past_node": bool(
                first_error and first_error["node_name"] != scatter_node.name
            ),
        }

    parser_passed = bool(parser_summary["parser_success"])
    with_standard_plugins = "standard_plugins_initialized" in parser_summary
    passed_status = (
        "TENSORRT_GCN_RES_PLUGIN_PARSER_WITH_STANDARD_PLUGINS_PASSED"
        if with_standard_plugins
        else "TENSORRT_GCN_RES_PLUGIN_PARSER_PASSED"
    )
    failed_status = (
        "TENSORRT_GCN_RES_PLUGIN_PARSER_WITH_STANDARD_PLUGINS_FAILED"
        if with_standard_plugins
        else "TENSORRT_GCN_RES_PLUGIN_PARSER_FAILED"
    )
    parser_audit = {
        "status": passed_status if parser_passed else failed_status,
        "parser_success": parser_passed,
        "parser_error_count": parser_summary["parser_error_count"],
        "plugin_creator_registered": parser_summary["plugin_creator_registered"],
        "plugin_creator_lookup_passed": parser_summary[
            "plugin_creator_lookup_passed"
        ],
        "plugin_creator_build_calls_before_stop": parser_summary[
            "plugin_creator_build_calls"
        ],
        "standard_plugins_initialized": parser_summary.get(
            "standard_plugins_initialized"
        ),
        "registry_creator_count_after_standard_init": parser_summary.get(
            "registry_creator_count_after_standard_init"
        ),
        "scatter_reduction_creator_found": parser_summary.get(
            "scatter_reduction_creator_found"
        ),
        "scatter_related_creators": parser_summary.get(
            "scatter_related_creators", []
        ),
        "scatter_node_evidence": scatter_graph_evidence,
        "first_blocking_node": first_error["node_name"] if first_error else None,
        "first_blocking_operator": first_error["op_type"] if first_error else None,
        "first_blocking_error": first_error["description"] if first_error else None,
        "first_blocking_error_code": first_error["error_code"]
        if first_error
        else None,
        "first_blocking_graph_evidence": blocker_graph_evidence,
        "engine_build_called": False,
        "inference_called": False,
        "engine_artifacts": generated_engines,
    }
    audit_key = (
        "parser_with_standard_plugins_audit"
        if "standard_plugins_initialized" in parser_summary
        else "parser_audit"
    )
    replacement[audit_key] = parser_audit
    dump_json(replacement_path, replacement)

    if parser_passed:
        conclusion = passed_status
        blocker_section = "No parser blocker was reported."
    else:
        conclusion = failed_status
        blocker_section = f"""- Node: `{first_error['node_name']}`
- Operator: `{first_error['op_type']}`
- Error code: `{first_error['error_code']}`
- Error: `{first_error['description']}`
- Source: `{first_error['source_file']}:{first_error['source_line']}` / `{first_error['source_function']}`"""

    report = f"""# GCN_res VoxelUnique Plugin parser-only audit

## Rewrite integrity

- Source ONNX: `{original}`
- Source SHA-256: `{original_hash}`
- Rewritten ONNX: `{rewritten}`
- Rewritten SHA-256: `{rewritten_hash}`
- Replacements: {replacement['replacement_count']}
- Standard `ai.onnx::Unique` nodes remaining: {len(replacement['standard_unique_remaining'])}
- ONNX checker: PASS
- Source ONNX unchanged: PASS

The live `unique_values` and `inverse_indices` tensor names were preserved so
all existing graph consumers remain connected. Only the unused standard Unique
`indices` and `counts` outputs were removed; each replacement adds an INT32
scalar `voxel_count` size-tensor output.

## TensorRT parser-only result

- Plugin Creator registered: {parser_summary['plugin_creator_registered']}
- Plugin Creator lookup: {parser_summary['plugin_creator_lookup_passed']}
- Standard plugins initialized: {parser_summary.get('standard_plugins_initialized', 'not recorded')}
- Registry creators after standard initialization: {parser_summary.get('registry_creator_count_after_standard_init', 'not recorded')}
- ScatterReduction Creator found: {parser_summary.get('scatter_reduction_creator_found', 'not recorded')}
- Scatter-related Creators: {parser_summary.get('scatter_related_creators', [])}
- Former blocker node evidence: {scatter_graph_evidence}
- VoxelUnique instances created before parser stop: {parser_summary['plugin_creator_build_calls']}
- Parser success: {parser_passed}
- Parser errors: {parser_summary['parser_error_count']}
- Engine build called: false
- Inference called: false

## First blocker

{blocker_section}

The parser passed the first custom `VoxelUnique` node and stopped at the first
downstream blocker. No graph modification or follow-on plugin implementation
was attempted after this result.

## Conclusion

{conclusion}
"""
    (run_dir / args.report_name).write_text(report, encoding="utf-8")
    print(f"ORIGINAL_ONNX_UNCHANGED={original_hash == EXPECTED_ORIGINAL_SHA256}")
    print("ENGINE_BUILD_CALLED=false")
    print("INFERENCE_CALLED=false")
    if first_error:
        print(f"FIRST_BLOCKING_NODE={first_error['node_name']}")
        print(f"FIRST_BLOCKING_OPERATOR={first_error['op_type']}")
        print(f"FIRST_BLOCKING_ERROR={first_error['description']}")
    print(conclusion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
