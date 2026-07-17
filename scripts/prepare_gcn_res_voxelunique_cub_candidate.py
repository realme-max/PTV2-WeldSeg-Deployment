"""Create and audit the Phase 8C derived VoxelUniqueCub ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import onnx
from onnx import helper

import gcn_res_tensorrt_cub_common as common


def json_safe_attribute_value(value: Any) -> Any:
    """Represent protobuf-valued ONNX attributes deterministically for hashing."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe_attribute_value(item) for item in value]
    serializer = getattr(value, "SerializeToString", None)
    if callable(serializer):
        payload = serializer()
        return {
            "protobuf_type": value.__class__.__name__,
            "serialized_size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return repr(value)


def attribute_payload(node: Any) -> list[dict[str, Any]]:
    result = []
    for attribute in node.attribute:
        value = json_safe_attribute_value(helper.get_attribute_value(attribute))
        result.append({"name": attribute.name, "value": value, "type": int(attribute.type)})
    return result


def value_info(value: Any) -> dict[str, Any]:
    tensor = value.type.tensor_type
    shape = []
    for dimension in tensor.shape.dim:
        shape.append(
            int(dimension.dim_value) if dimension.HasField("dim_value")
            else dimension.dim_param or None
        )
    return {"name": value.name, "dtype": int(tensor.elem_type), "shape": shape}


def node_signature(node: Any, ignore_plugin_identity: bool = False) -> dict[str, Any]:
    attributes = attribute_payload(node)
    if ignore_plugin_identity and node.name in common.EXPECTED_NODE_NAMES:
        attributes = [item for item in attributes if item["name"] != "plugin_namespace"]
        domain = "<PLUGIN_DOMAIN>"
        op_type = "<PLUGIN_OP>"
    else:
        domain = node.domain
        op_type = node.op_type
    return {
        "name": node.name,
        "domain": domain,
        "op_type": op_type,
        "inputs": list(node.input),
        "outputs": list(node.output),
        "attributes": attributes,
    }


def digest_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def graph_audit(model: Any) -> dict[str, Any]:
    plugin_nodes = [node for node in model.graph.node if node.name in common.EXPECTED_NODE_NAMES]
    non_plugin = [node_signature(node) for node in model.graph.node if node.name not in common.EXPECTED_NODE_NAMES]
    initializer_records = [
        {
            "name": item.name,
            "dtype": int(item.data_type),
            "dims": list(item.dims),
            "sha256": hashlib.sha256(item.SerializeToString()).hexdigest(),
        }
        for item in model.graph.initializer
    ]
    return {
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "graph_inputs": [value_info(item) for item in model.graph.input],
        "graph_outputs": [value_info(item) for item in model.graph.output],
        "opset_imports": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "plugin_nodes": [node_signature(node) for node in plugin_nodes],
        "non_plugin_node_structure_sha256": digest_json(non_plugin),
        "initializer_structure_and_content_sha256": digest_json(initializer_records),
        "topology_with_plugin_identity_normalized_sha256": digest_json(
            [node_signature(node, ignore_plugin_identity=True) for node in model.graph.node]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=common.FORMAL_ONNX)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    audit_path = args.audit.resolve()
    protected_before = common.protected_snapshot()
    if common.sha256(source) != common.PROTECTED_HASHES["formal_onnx"]:
        raise RuntimeError("Source is not the frozen formal TensorRT-ready ONNX")

    model = onnx.load(str(source), load_external_data=False)
    before = graph_audit(model)
    targets = [node for node in model.graph.node if node.name in common.EXPECTED_NODE_NAMES]
    if len(targets) != 4 or {node.name for node in targets} != set(common.EXPECTED_NODE_NAMES):
        raise RuntimeError("The four expected VoxelUnique nodes were not found exactly once")
    mappings = []
    for node in targets:
        if node.op_type != "VoxelUnique" or node.domain != "com.tensorrt.ptv2":
            raise RuntimeError(f"Unexpected source identity for {node.name}")
        old_attributes = attribute_payload(node)
        namespace_attributes = [item for item in node.attribute if item.name == "plugin_namespace"]
        if len(namespace_attributes) != 1:
            raise RuntimeError(f"Expected one plugin_namespace attribute on {node.name}")
        node.op_type = common.PLUGIN_NAME
        node.domain = common.PLUGIN_NAMESPACE
        namespace_attributes[0].s = common.PLUGIN_NAMESPACE.encode("utf-8")
        mappings.append(
            {
                "node_name": node.name,
                "old": {"domain": "com.tensorrt.ptv2", "op_type": "VoxelUnique", "attributes": old_attributes},
                "new": {"domain": node.domain, "op_type": node.op_type, "attributes": attribute_payload(node)},
                "inputs_unchanged": list(node.input),
                "outputs_unchanged": list(node.output),
            }
        )
    if not any(item.domain == common.PLUGIN_NAMESPACE for item in model.opset_import):
        model.opset_import.append(helper.make_opsetid(common.PLUGIN_NAMESPACE, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    output.write_bytes(model.SerializeToString())
    reloaded = onnx.load(str(output), load_external_data=False)
    onnx.checker.check_model(reloaded)
    after = graph_audit(reloaded)

    invariants = {
        "node_count_unchanged": before["node_count"] == after["node_count"],
        "initializer_count_unchanged": before["initializer_count"] == after["initializer_count"],
        "graph_inputs_unchanged": before["graph_inputs"] == after["graph_inputs"],
        "graph_outputs_unchanged": before["graph_outputs"] == after["graph_outputs"],
        "non_plugin_nodes_unchanged": before["non_plugin_node_structure_sha256"] == after["non_plugin_node_structure_sha256"],
        "initializers_unchanged": before["initializer_structure_and_content_sha256"] == after["initializer_structure_and_content_sha256"],
        "topology_and_contract_unchanged": before["topology_with_plugin_identity_normalized_sha256"] == after["topology_with_plugin_identity_normalized_sha256"],
        "four_plugin_nodes_replaced": len(after["plugin_nodes"]) == 4 and all(
            item["op_type"] == common.PLUGIN_NAME and item["domain"] == common.PLUGIN_NAMESPACE
            for item in after["plugin_nodes"]
        ),
    }
    if not all(invariants.values()):
        raise RuntimeError(f"Derived ONNX invariants failed: {invariants}")
    protected_after = common.protected_snapshot()
    audit = {
        "status": "PHASE8C_DERIVED_ONNX_AUDIT_PASSED",
        "source": {"path": str(source), "sha256": common.sha256(source)},
        "derived": {"path": str(output), "sha256": common.sha256(output), "size_bytes": output.stat().st_size},
        "before": before,
        "after": after,
        "plugin_node_mappings": mappings,
        "identity_metadata_changes": {
            "plugin_namespace_attribute_updated": True,
            "experimental_domain_opset_added": True,
            "reason": "TensorRT ONNX Parser resolves custom Plugin Creator using plugin_namespace; these are parser identity metadata, not mathematical attributes.",
            "plugin_version_unchanged": True,
        },
        "invariants": invariants,
        "onnx_checker": "PASS",
        "protected_before": protected_before,
        "protected_after": protected_after,
        "protected_unchanged": protected_before == protected_after,
    }
    common.dump_json(audit_path, audit)
    print(f"DERIVED_ONNX={output}")
    print(f"DERIVED_ONNX_SHA256={common.sha256(output)}")
    print("PHASE8C_DERIVED_ONNX_AUDIT_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
