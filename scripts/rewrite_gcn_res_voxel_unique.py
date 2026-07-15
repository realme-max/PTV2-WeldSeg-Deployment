"""Rewrite the four GCN_res ONNX Unique nodes to the VoxelUnique plugin op.

The source model is opened read-only and is never overwritten.  The rewrite
preserves the live values/inverse tensor names so every existing consumer stays
connected, adds an explicit INT32 scalar size-tensor output, and drops only the
unused indices/counts outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, helper


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_onnx"
    / "20260715_onnx_after_cdist_fp32_opset18"
    / "gcn_res_deploy_fp32_opset18.onnx"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
PLUGIN_DOMAIN = "com.tensorrt.ptv2"
PLUGIN_OP = "VoxelUnique"
PLUGIN_VERSION = "1"
TARGETS = tuple(f"/model/tdb_{stage}/Unique" for stage in range(1, 5))


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


def attributes(node: onnx.NodeProto) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attribute in node.attribute:
        result[attribute.name] = helper.get_attribute_value(attribute)
    return result


def consumer_map(model: onnx.ModelProto) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for index, node in enumerate(model.graph.node):
        for input_slot, name in enumerate(node.input):
            if name:
                result.setdefault(name, []).append(
                    {
                        "node_index": index,
                        "node_name": node.name,
                        "op_type": node.op_type,
                        "input_slot": input_slot,
                    }
                )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    source = args.input.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.run_dir:
        run_dir = args.run_dir.resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = args.output_root.resolve() / f"{timestamp}_voxelplugin_rewrite"
    run_dir.mkdir(parents=True, exist_ok=False)
    rewritten_path = run_dir / "rewritten.onnx"
    if rewritten_path == source:
        raise RuntimeError("Refusing to overwrite the source ONNX")

    original_hash_before = sha256(source)
    model = onnx.load_model(str(source), load_external_data=False)
    original_inputs = [item.name for item in model.graph.input]
    original_outputs = [item.name for item in model.graph.output]
    original_node_count = len(model.graph.node)
    original_consumers = consumer_map(model)
    graph_output_names = {item.name for item in model.graph.output}
    target_by_name = {node.name: node for node in model.graph.node if node.name in TARGETS}
    if set(target_by_name) != set(TARGETS):
        missing = sorted(set(TARGETS) - set(target_by_name))
        raise RuntimeError(f"Missing target Unique nodes: {missing}")

    removed_value_info_names: set[str] = set()
    added_value_infos: list[onnx.ValueInfoProto] = []
    replacements: list[dict[str, Any]] = []
    for node_index, node in enumerate(model.graph.node):
        if node.name not in TARGETS:
            continue
        old_attributes = attributes(node)
        old_outputs = list(node.output)
        if node.op_type != "Unique" or node.domain not in ("", "ai.onnx"):
            raise RuntimeError(
                f"Target is not ai.onnx::Unique: {node.name} "
                f"domain={node.domain!r} op={node.op_type!r}"
            )
        if len(node.input) != 1 or len(old_outputs) != 4:
            raise RuntimeError(f"Unexpected Unique signature at {node.name}")
        if old_attributes.get("sorted", 1) != 1 or "axis" in old_attributes:
            raise RuntimeError(f"Unsupported Unique attributes at {node.name}: {old_attributes}")

        values_name, indices_name, inverse_name, counts_name = old_outputs
        indices_consumers = original_consumers.get(indices_name, [])
        counts_consumers = original_consumers.get(counts_name, [])
        if indices_consumers or counts_consumers:
            raise RuntimeError(
                f"Cannot drop used outputs at {node.name}: "
                f"indices={indices_consumers}, counts={counts_consumers}"
            )
        if indices_name in graph_output_names or counts_name in graph_output_names:
            raise RuntimeError(f"Cannot drop graph outputs at {node.name}")

        stage = node.name.split("/")[2]
        count_name = f"/model/{stage}/VoxelUnique_voxel_count_output_0"
        node.op_type = PLUGIN_OP
        node.domain = PLUGIN_DOMAIN
        node.ClearField("attribute")
        node.attribute.extend(
            [
                helper.make_attribute("plugin_version", PLUGIN_VERSION),
                helper.make_attribute("plugin_namespace", PLUGIN_DOMAIN),
            ]
        )
        node.ClearField("output")
        node.output.extend([count_name, values_name, inverse_name])
        removed_value_info_names.update((indices_name, counts_name))
        added_value_infos.append(
            helper.make_tensor_value_info(count_name, TensorProto.INT32, [])
        )
        replacements.append(
            {
                "node_index": node_index,
                "node_name": node.name,
                "input": node.input[0],
                "before": {
                    "domain": "ai.onnx",
                    "op_type": "Unique",
                    "attributes": old_attributes,
                    "outputs": {
                        "values": values_name,
                        "indices": indices_name,
                        "inverse_indices": inverse_name,
                        "counts": counts_name,
                    },
                },
                "after": {
                    "domain": PLUGIN_DOMAIN,
                    "op_type": PLUGIN_OP,
                    "plugin_version": PLUGIN_VERSION,
                    "plugin_namespace": PLUGIN_DOMAIN,
                    "outputs": {
                        "voxel_count": count_name,
                        "unique_values": values_name,
                        "inverse_indices": inverse_name,
                    },
                },
                "preserved_consumers": {
                    "unique_values": original_consumers.get(values_name, []),
                    "inverse_indices": original_consumers.get(inverse_name, []),
                },
                "removed_unused_outputs": {
                    "indices": indices_name,
                    "counts": counts_name,
                },
            }
        )

    retained_value_infos = [
        item
        for item in model.graph.value_info
        if item.name not in removed_value_info_names
    ]
    model.graph.ClearField("value_info")
    model.graph.value_info.extend(retained_value_infos)
    model.graph.value_info.extend(added_value_infos)

    domain_imports = [item for item in model.opset_import if item.domain == PLUGIN_DOMAIN]
    if domain_imports:
        if len(domain_imports) != 1:
            raise RuntimeError(f"Duplicate opset imports for {PLUGIN_DOMAIN}")
        domain_imports[0].version = 1
    else:
        model.opset_import.append(helper.make_opsetid(PLUGIN_DOMAIN, 1))

    remaining_unique = [
        node.name
        for node in model.graph.node
        if node.op_type == "Unique" and node.domain in ("", "ai.onnx")
    ]
    plugin_nodes = [
        node.name
        for node in model.graph.node
        if node.op_type == PLUGIN_OP and node.domain == PLUGIN_DOMAIN
    ]
    if remaining_unique:
        raise RuntimeError(f"Standard Unique nodes remain: {remaining_unique}")
    if plugin_nodes != list(TARGETS):
        raise RuntimeError(f"Unexpected plugin nodes/order: {plugin_nodes}")
    if len(model.graph.node) != original_node_count:
        raise RuntimeError("Node count changed during in-place node replacement")
    if [item.name for item in model.graph.input] != original_inputs:
        raise RuntimeError("Graph inputs changed")
    if [item.name for item in model.graph.output] != original_outputs:
        raise RuntimeError("Graph outputs changed")

    onnx.checker.check_model(model)
    onnx.save_model(model, str(rewritten_path))
    reloaded = onnx.load_model(str(rewritten_path), load_external_data=False)
    onnx.checker.check_model(reloaded)
    original_hash_after = sha256(source)
    if original_hash_after != original_hash_before:
        raise RuntimeError("Source ONNX changed during rewrite")
    rewritten_hash = sha256(rewritten_path)

    (run_dir / "original_sha256.txt").write_text(
        f"{original_hash_before}  {source}\n", encoding="utf-8"
    )
    (run_dir / "rewritten_sha256.txt").write_text(
        f"{rewritten_hash}  {rewritten_path}\n", encoding="utf-8"
    )
    summary = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "source_onnx": str(source),
        "rewritten_onnx": str(rewritten_path),
        "source_size_bytes": source.stat().st_size,
        "rewritten_size_bytes": rewritten_path.stat().st_size,
        "original_sha256_before": original_hash_before,
        "original_sha256_after": original_hash_after,
        "source_unchanged": original_hash_before == original_hash_after,
        "rewritten_sha256": rewritten_hash,
        "node_count_before": original_node_count,
        "node_count_after": len(reloaded.graph.node),
        "replacement_count": len(replacements),
        "standard_unique_remaining": remaining_unique,
        "custom_voxel_unique_nodes": plugin_nodes,
        "removed_value_info": sorted(removed_value_info_names),
        "added_size_tensor_value_info": [item.name for item in added_value_infos],
        "graph_inputs_unchanged": [item.name for item in reloaded.graph.input]
        == original_inputs,
        "graph_outputs_unchanged": [item.name for item in reloaded.graph.output]
        == original_outputs,
        "onnx_checker_passed": True,
        "replacements": replacements,
    }
    dump_json(run_dir / "replacement_summary.json", summary)
    print(f"RUN_DIR={run_dir}")
    print(f"ORIGINAL_SHA256={original_hash_before}")
    print(f"REWRITTEN_SHA256={rewritten_hash}")
    print(f"REPLACEMENT_COUNT={len(replacements)}")
    print("GCN_RES_VOXEL_UNIQUE_REWRITE_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
