"""Read-only recursive semantic audit of Squeeze nodes in rewritten GCN_res ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import onnx
from onnx import helper, numpy_helper


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = (
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
TARGET_NAME = "/model/tdb_1/Squeeze"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, onnx.TensorProto):
        return jsonable(numpy_helper.to_array(value))
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return value


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def value_info(value: onnx.ValueInfoProto) -> dict[str, Any]:
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
    return {
        "name": value.name,
        "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type),
        "shape": shape,
    }


@dataclass
class GraphContext:
    graph: onnx.GraphProto
    scope: str
    parent: "GraphContext | None" = None
    parent_node: onnx.NodeProto | None = None
    parent_attribute: str | None = None
    producers: dict[str, onnx.NodeProto] = field(default_factory=dict)
    consumers: dict[str, list[tuple[onnx.NodeProto, int]]] = field(
        default_factory=dict
    )
    tensors: dict[str, dict[str, Any]] = field(default_factory=dict)
    initializers: dict[str, onnx.TensorProto] = field(default_factory=dict)
    node_indices: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.node_indices = {
            node.name or f"{node.op_type}#{index}": index
            for index, node in enumerate(self.graph.node)
        }
        for node in self.graph.node:
            for output in node.output:
                if output:
                    self.producers[output] = node
            for slot, input_name in enumerate(node.input):
                if input_name:
                    self.consumers.setdefault(input_name, []).append((node, slot))
        for value in (*self.graph.input, *self.graph.output, *self.graph.value_info):
            self.tensors[value.name] = value_info(value)
        self.initializers = {item.name: item for item in self.graph.initializer}
        for item in self.graph.initializer:
            self.tensors.setdefault(
                item.name,
                {
                    "name": item.name,
                    "dtype": onnx.TensorProto.DataType.Name(item.data_type),
                    "shape": list(item.dims),
                },
            )


def collect_contexts(
    graph: onnx.GraphProto,
    scope: str = "main",
    parent: GraphContext | None = None,
    parent_node: onnx.NodeProto | None = None,
    parent_attribute: str | None = None,
) -> list[GraphContext]:
    context = GraphContext(
        graph=graph,
        scope=scope,
        parent=parent,
        parent_node=parent_node,
        parent_attribute=parent_attribute,
    )
    contexts = [context]
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                nested_scope = f"{scope}{node.name or '/' + node.op_type}:{attribute.name}"
                contexts.extend(
                    collect_contexts(
                        attribute.g,
                        nested_scope,
                        context,
                        node,
                        attribute.name,
                    )
                )
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for index, nested in enumerate(attribute.graphs):
                    nested_scope = (
                        f"{scope}{node.name or '/' + node.op_type}:"
                        f"{attribute.name}[{index}]"
                    )
                    contexts.extend(
                        collect_contexts(
                            nested,
                            nested_scope,
                            context,
                            node,
                            f"{attribute.name}[{index}]",
                        )
                    )
    return contexts


def resolve_tensor_info(context: GraphContext, name: str) -> dict[str, Any]:
    current: GraphContext | None = context
    while current is not None:
        if name in current.tensors:
            return current.tensors[name]
        current = current.parent
    return {"name": name, "dtype": "UNKNOWN", "shape": []}


def resolve_producer(
    context: GraphContext, name: str
) -> tuple[GraphContext, onnx.NodeProto] | None:
    current: GraphContext | None = context
    while current is not None:
        if name in current.producers:
            return current, current.producers[name]
        current = current.parent
    return None


def resolve_initializer(
    context: GraphContext, name: str
) -> tuple[GraphContext, onnx.TensorProto] | None:
    current: GraphContext | None = context
    while current is not None:
        if name in current.initializers:
            return current, current.initializers[name]
        current = current.parent
    return None


def node_attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {
        attribute.name: jsonable(helper.get_attribute_value(attribute))
        for attribute in node.attribute
        if attribute.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
    }


def describe_node(context: GraphContext, node: onnx.NodeProto) -> dict[str, Any]:
    return {
        "node_name": node.name,
        "node_index_in_scope": context.node_indices.get(node.name, -1),
        "op_type": node.op_type,
        "domain": node.domain or "ai.onnx",
        "parent_graph_scope": context.scope,
        "input_names": list(node.input),
        "inputs": [resolve_tensor_info(context, name) for name in node.input if name],
        "output_names": list(node.output),
        "outputs": [resolve_tensor_info(context, name) for name in node.output if name],
        "attributes": node_attributes(node),
    }


def constant_tensor_value(context: GraphContext, name: str) -> dict[str, Any] | None:
    initializer = resolve_initializer(context, name)
    if initializer:
        source_context, tensor = initializer
        return {
            "kind": "initializer",
            "scope": source_context.scope,
            "name": name,
            "value": jsonable(numpy_helper.to_array(tensor)),
        }
    producer = resolve_producer(context, name)
    if producer and producer[1].op_type == "Constant":
        source_context, node = producer
        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        for key in (
            "value",
            "value_int",
            "value_ints",
            "value_float",
            "value_floats",
        ):
            if key not in attributes:
                continue
            value = attributes[key]
            if isinstance(value, onnx.TensorProto):
                value = numpy_helper.to_array(value)
            return {
                "kind": "Constant",
                "scope": source_context.scope,
                "node_name": node.name,
                "value": jsonable(value),
            }
    return None


def trace_upstream(
    context: GraphContext,
    tensor_name: str,
    depth: int = 0,
    seen: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    seen = set() if seen is None else seen
    key = (context.scope, tensor_name)
    trace: dict[str, Any] = {"tensor": resolve_tensor_info(context, tensor_name)}
    if key in seen:
        trace["source_kind"] = "cycle"
        return trace
    seen.add(key)
    constant = constant_tensor_value(context, tensor_name)
    if constant:
        trace["source_kind"] = "constant"
        trace["constant"] = constant
        return trace
    producer = resolve_producer(context, tensor_name)
    if producer is None:
        trace["source_kind"] = "graph_input_or_captured_value"
        return trace
    producer_context, producer_node = producer
    trace["source_kind"] = "node"
    trace["producer"] = describe_node(producer_context, producer_node)
    if depth < 8:
        trace["upstream_inputs"] = [
            trace_upstream(producer_context, name, depth + 1, seen.copy())
            for name in producer_node.input
            if name
        ]
    else:
        trace["truncated_at_depth"] = depth
    return trace


def evaluate_scalar(
    context: GraphContext,
    tensor_name: str,
    depth: int = 0,
    seen: set[tuple[str, str]] | None = None,
) -> Any:
    if depth > 12:
        return None
    seen = set() if seen is None else seen
    key = (context.scope, tensor_name)
    if key in seen:
        return None
    seen.add(key)
    constant = constant_tensor_value(context, tensor_name)
    if constant:
        value = constant["value"]
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value
    producer = resolve_producer(context, tensor_name)
    if producer is None:
        return None
    producer_context, node = producer
    values = [
        evaluate_scalar(producer_context, name, depth + 1, seen.copy())
        for name in node.input
        if name
    ]
    if node.op_type in ("Identity", "Cast") and values:
        return values[0]
    if node.op_type == "Shape" and node.input:
        shape = resolve_tensor_info(producer_context, node.input[0])["shape"]
        return list(shape) if shape else None
    if node.op_type == "Size" and node.input:
        input_value = values[0] if values else None
        if isinstance(input_value, list):
            return len(input_value)
        shape = resolve_tensor_info(producer_context, node.input[0])["shape"]
        if shape and all(isinstance(item, int) for item in shape):
            return math.prod(shape)
        return None
    if node.op_type == "Equal" and len(values) == 2:
        if values[0] is not None and values[1] is not None:
            return values[0] == values[1]
    return None


def axes_analysis(context: GraphContext, node: onnx.NodeProto) -> dict[str, Any]:
    attributes = node_attributes(node)
    if "axes" in attributes:
        return {
            "classification": "A_STATIC_ATTRIBUTE_AXES",
            "axes": attributes["axes"],
            "source": "node attribute",
        }
    axes_name = node.input[1] if len(node.input) > 1 else ""
    if axes_name:
        constant = constant_tensor_value(context, axes_name)
        if constant:
            return {
                "classification": "B_CONSTANT_AXES_INPUT",
                "axes_input": axes_name,
                "axes": constant["value"],
                "source": constant,
            }
        return {
            "classification": "C_DYNAMIC_TENSOR_AXES",
            "axes_input": axes_name,
            "source_chain": trace_upstream(context, axes_name),
        }
    if context.parent_node is not None:
        return {
            "classification": "D_CONTROL_FLOW_SUBGRAPH_RELATED",
            "axes_input": None,
            "axes_attribute": None,
            "semantics": "axes omitted: squeeze every singleton input dimension",
            "parent_control_flow_node": context.parent_node.name,
            "parent_control_flow_op": context.parent_node.op_type,
            "branch": context.parent_attribute,
        }
    return {
        "classification": "D_IMPLICIT_AXES_OMITTED",
        "axes_input": None,
        "axes_attribute": None,
        "semantics": "axes omitted: squeeze every singleton input dimension",
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
    onnx_hash_before = sha256(onnx_path)
    original_hash_before = sha256(original_path) if original_path.is_file() else None
    model = onnx.load_model(str(onnx_path), load_external_data=False)
    onnx.checker.check_model(model)
    opsets = {item.domain or "ai.onnx": int(item.version) for item in model.opset_import}
    contexts = collect_contexts(model.graph)

    squeeze_records: list[dict[str, Any]] = []
    target_records: list[tuple[GraphContext, onnx.NodeProto, dict[str, Any]]] = []
    for context in contexts:
        for node in context.graph.node:
            if node.op_type != "Squeeze":
                continue
            record = describe_node(context, node)
            record["opset"] = opsets.get(node.domain or "ai.onnx")
            record["axes_analysis"] = axes_analysis(context, node)
            squeeze_records.append(record)
            if node.name == TARGET_NAME:
                target_records.append((context, node, record))
    if len(target_records) != 1:
        raise RuntimeError(
            f"Expected exactly one target {TARGET_NAME}, found {len(target_records)}"
        )
    target_context, target_node, target_record = target_records[0]

    data_input = target_node.input[0]
    producer_record = resolve_producer(target_context, data_input)
    direct_producer = (
        describe_node(*producer_record) if producer_record is not None else None
    )
    output_dependencies: list[dict[str, Any]] = []
    chain: list[dict[str, Any]] = []
    if direct_producer:
        chain.append(
            {
                "from": direct_producer["node_name"],
                "tensor": data_input,
                "to": target_node.name,
                "edge": "captured_data_input"
                if producer_record[0] is not target_context
                else "data_input",
            }
        )
    for output_name in target_node.output:
        local_consumers = target_context.consumers.get(output_name, [])
        for consumer, slot in local_consumers:
            item = {
                "kind": "node_consumer",
                "tensor": output_name,
                "consumer": describe_node(target_context, consumer),
                "input_slot": slot,
            }
            output_dependencies.append(item)
            chain.append(
                {
                    "from": target_node.name,
                    "tensor": output_name,
                    "to": consumer.name,
                    "edge": "node_consumer",
                }
            )
        graph_output_names = [item.name for item in target_context.graph.output]
        if output_name in graph_output_names and target_context.parent is not None:
            output_index = graph_output_names.index(output_name)
            parent_node = target_context.parent_node
            parent_output = (
                parent_node.output[output_index]
                if parent_node is not None and output_index < len(parent_node.output)
                else None
            )
            parent_consumers = (
                target_context.parent.consumers.get(parent_output, [])
                if parent_output
                else []
            )
            bridge = {
                "kind": "control_flow_graph_output",
                "branch": target_context.parent_attribute,
                "branch_output_index": output_index,
                "branch_output_tensor": output_name,
                "parent_node": describe_node(target_context.parent, parent_node)
                if parent_node is not None
                else None,
                "parent_output_tensor": parent_output,
                "parent_output_consumers": [
                    {
                        "consumer": describe_node(target_context.parent, consumer),
                        "input_slot": slot,
                    }
                    for consumer, slot in parent_consumers
                ],
            }
            output_dependencies.append(bridge)
            chain.append(
                {
                    "from": target_node.name,
                    "tensor": output_name,
                    "to": parent_node.name if parent_node is not None else None,
                    "edge": f"{target_context.parent_attribute}_graph_output",
                }
            )
            for consumer, _ in parent_consumers:
                chain.append(
                    {
                        "from": parent_node.name if parent_node is not None else None,
                        "tensor": parent_output,
                        "to": consumer.name,
                        "edge": "parent_if_output_consumer",
                    }
                )

    control_flow: dict[str, Any] | None = None
    condition_value = None
    if target_context.parent_node is not None and target_context.parent is not None:
        parent_node = target_context.parent_node
        condition_name = parent_node.input[0] if parent_node.input else None
        condition_value = (
            evaluate_scalar(target_context.parent, condition_name)
            if condition_name
            else None
        )
        control_flow = {
            "parent_node": describe_node(target_context.parent, parent_node),
            "branch": target_context.parent_attribute,
            "condition_tensor": condition_name,
            "condition_static_evaluation": condition_value,
            "condition_source_chain": trace_upstream(
                target_context.parent, condition_name
            )
            if condition_name
            else None,
        }

    input_info = resolve_tensor_info(target_context, data_input)
    input_rank = len(input_info["shape"])
    if (
        target_record["axes_analysis"]["classification"]
        == "D_CONTROL_FLOW_SUBGRAPH_RELATED"
        and input_rank == 1
    ):
        if condition_value is False and target_context.parent_attribute == "then_branch":
            feasibility = {
                "classification": "MINIMAL_REWRITE_CANDIDATE_WITH_STATIC_DEAD_BRANCH_PROOF",
                "candidate_axes": [0],
                "reason": (
                    "The axes are omitted, the input rank is one, and the audited "
                    "If condition statically evaluates to false, so this then_branch "
                    "is not selected in the exported graph. A rewrite still requires "
                    "a separate equivalence test before authorization."
                ),
                "rewrite_performed": False,
            }
        else:
            feasibility = {
                "classification": "CONDITIONAL_MINIMAL_REWRITE_CANDIDATE",
                "candidate_axes": [0],
                "reason": (
                    "For rank-1 input, explicit axes=[0] is equivalent to omitted axes "
                    "only when the selected branch guarantees dimension 0 equals one. "
                    "That invariant is not proven by shape alone."
                ),
                "rewrite_performed": False,
            }
    else:
        feasibility = {
            "classification": "NOT_A_MINIMAL_REWRITE_CANDIDATE_FROM_CURRENT_EVIDENCE",
            "candidate_axes": None,
            "rewrite_performed": False,
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.output_root.resolve() / f"{timestamp}_squeeze_audit"
    run_dir.mkdir(parents=True, exist_ok=False)
    classifications: dict[str, int] = {}
    for record in squeeze_records:
        category = record["axes_analysis"]["classification"]
        classifications[category] = classifications.get(category, 0) + 1
    squeeze_payload = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "onnx_path": str(onnx_path),
        "onnx_sha256": onnx_hash_before,
        "onnx_checker_passed": True,
        "opset_imports": opsets,
        "graph_scope_count": len(contexts),
        "squeeze_count": len(squeeze_records),
        "classification_counts": classifications,
        "nodes": squeeze_records,
    }
    dependency_payload = {
        "target_node": target_record,
        "input_producer": direct_producer,
        "input_source_chain": trace_upstream(target_context, data_input),
        "output_dependencies": output_dependencies,
        "control_flow": control_flow,
        "complete_dependency_chain": chain,
        "minimal_rewrite_assessment": feasibility,
        "source_integrity": {
            "rewritten_onnx_sha256_before": onnx_hash_before,
            "original_onnx_sha256_before": original_hash_before,
        },
    }
    dump_json(run_dir / "squeeze_nodes.json", squeeze_payload)
    dump_json(run_dir / "squeeze_dependency.json", dependency_payload)

    onnx_hash_after = sha256(onnx_path)
    original_hash_after = sha256(original_path) if original_path.is_file() else None
    if onnx_hash_after != onnx_hash_before or original_hash_after != original_hash_before:
        raise RuntimeError("ONNX hash changed during read-only audit")
    dependency_payload["source_integrity"].update(
        {
            "rewritten_onnx_sha256_after": onnx_hash_after,
            "rewritten_onnx_unchanged": onnx_hash_after == onnx_hash_before,
            "original_onnx_sha256_after": original_hash_after,
            "original_onnx_unchanged": original_hash_after == original_hash_before,
        }
    )
    dump_json(run_dir / "squeeze_dependency.json", dependency_payload)

    target_axes = target_record["axes_analysis"]
    report = f"""# TensorRT dynamic Squeeze semantic audit

## Scope and integrity

- Rewritten ONNX: `{onnx_path}`
- SHA-256: `{onnx_hash_before}`
- ONNX checker: PASS
- Recursive graph scopes: {len(contexts)}
- Squeeze nodes: {len(squeeze_records)}
- Classification counts: `{classifications}`
- Rewritten ONNX unchanged: PASS
- Original ONNX unchanged: PASS

This audit was read-only. It did not re-export or rewrite a graph, build an
Engine, or run inference.

## Blocking node

- Name: `{target_node.name}`
- Scope: `{target_context.scope}`
- Opset: {target_record['opset']}
- Input: `{data_input}` / `{input_info}`
- Output: `{list(target_node.output)}`
- Axes classification: `{target_axes['classification']}`
- Axes attribute: absent
- Axes input: absent
- Effective ONNX semantics: squeeze every singleton input dimension

This is not a dynamic axes tensor. It is an omitted-axes Squeeze located in an
`If` control-flow subgraph; TensorRT cannot infer a fixed output rank from the
dynamic input dimension while parsing both branches.

## Dependency

- Direct producer: `{direct_producer}`
- Output/control-flow dependencies: `{output_dependencies}`
- Parent control flow: `{control_flow}`
- Complete edge chain: `{chain}`

## Minimal rewrite assessment

- Classification: `{feasibility['classification']}`
- Candidate explicit axes: `{feasibility.get('candidate_axes')}`
- Reason: {feasibility.get('reason', 'Current evidence is insufficient.')}
- Rewrite performed: false

The result identifies a narrowly scoped rewrite candidate, but it does not
authorize a rewrite. A dedicated equivalence audit must prove the branch and
shape invariant before modifying a derived ONNX graph.

## Conclusion

DYNAMIC_SQUEEZE_AUDIT_COMPLETED
"""
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"RUN_DIR={run_dir}")
    print(f"SQUEEZE_NODE_COUNT={len(squeeze_records)}")
    print(f"TARGET_AXES_CLASSIFICATION={target_axes['classification']}")
    print(f"MINIMAL_REWRITE_ASSESSMENT={feasibility['classification']}")
    print("REWRITTEN_ONNX_UNCHANGED=true")
    print("ORIGINAL_ONNX_UNCHANGED=true")
    print("DYNAMIC_SQUEEZE_AUDIT_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
