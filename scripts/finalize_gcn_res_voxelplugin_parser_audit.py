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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--original-onnx", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    original = args.original_onnx.resolve()
    rewritten = run_dir / "rewritten.onnx"
    replacement_path = run_dir / "replacement_summary.json"
    parser_path = run_dir / "parser_summary.json"
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
    first_error = parser_summary["errors"][0] if parser_summary["errors"] else None
    blocker_node = None
    if first_error:
        blocker_node = next(
            (node for node in model.graph.node if node.name == first_error["node_name"]),
            None,
        )
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
            "inputs": list(blocker_node.input),
            "outputs": list(blocker_node.output),
            "attributes": blocker_attributes,
        }
        if blocker_node is not None
        else None
    )

    parser_passed = bool(parser_summary["parser_success"])
    parser_audit = {
        "status": (
            "TENSORRT_GCN_RES_PLUGIN_PARSER_PASSED"
            if parser_passed
            else "TENSORRT_GCN_RES_PLUGIN_PARSER_FAILED"
        ),
        "parser_success": parser_passed,
        "parser_error_count": parser_summary["parser_error_count"],
        "plugin_creator_registered": parser_summary["plugin_creator_registered"],
        "plugin_creator_lookup_passed": parser_summary[
            "plugin_creator_lookup_passed"
        ],
        "plugin_creator_build_calls_before_stop": parser_summary[
            "plugin_creator_build_calls"
        ],
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
    replacement["parser_audit"] = parser_audit
    dump_json(replacement_path, replacement)

    if parser_passed:
        conclusion = "TENSORRT_GCN_RES_PLUGIN_PARSER_PASSED"
        blocker_section = "No parser blocker was reported."
    else:
        conclusion = "TENSORRT_GCN_RES_PLUGIN_PARSER_FAILED"
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
    (run_dir / "parser_audit_report.md").write_text(report, encoding="utf-8")
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
