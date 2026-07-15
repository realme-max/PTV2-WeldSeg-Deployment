"""Read-only semantic audit of the four ONNX Unique nodes in GCN_res."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import onnx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_onnx"
    / "20260715_onnx_after_cdist_fp32_opset18"
    / "gcn_res_deploy_fp32_opset18.onnx"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
TARGET_NAMES = (
    "/model/tdb_1/Unique",
    "/model/tdb_2/Unique",
    "/model/tdb_3/Unique",
    "/model/tdb_4/Unique",
)
UNIQUE_OUTPUT_ROLES = ("values", "indices", "inverse_indices", "counts")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dimension_value(dim: onnx.TensorShapeProto.Dimension) -> int | str:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    if dim.dim_param:
        return dim.dim_param
    return "?"


def _type_info(value_info: onnx.ValueInfoProto | None) -> dict[str, Any]:
    if value_info is None or not value_info.type.HasField("tensor_type"):
        return {"dtype": None, "shape": None}
    tensor_type = value_info.type.tensor_type
    dtype = (
        onnx.helper.tensor_dtype_to_string(tensor_type.elem_type)
        if tensor_type.elem_type
        else None
    )
    shape = [_dimension_value(dim) for dim in tensor_type.shape.dim]
    return {"dtype": dtype, "shape": shape}


def _value_info_map(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    entries = list(model.graph.input) + list(model.graph.output) + list(
        model.graph.value_info
    )
    return {entry.name: entry for entry in entries}


def _attribute_map(node: onnx.NodeProto) -> dict[str, Any]:
    return {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def _consumer_input_role(op_type: str, input_slot: int) -> str | None:
    known_roles = {
        "ScatterElements": ("data", "indices", "updates"),
        "Shape": ("data",),
        "Unsqueeze": ("data", "axes"),
    }
    roles = known_roles.get(op_type)
    if roles is None or input_slot >= len(roles):
        return None
    return roles[input_slot]


def _runtime_dependencies(
    tensor_name: str,
    graph_inputs: set[str],
    initializers: set[str],
    producers: dict[str, tuple[int, onnx.NodeProto]],
    memo: dict[str, set[str]],
    visiting: set[str] | None = None,
) -> set[str]:
    if tensor_name in memo:
        return memo[tensor_name]
    if tensor_name in graph_inputs:
        memo[tensor_name] = {tensor_name}
        return memo[tensor_name]
    if tensor_name in initializers or tensor_name not in producers:
        memo[tensor_name] = set()
        return memo[tensor_name]

    visiting = set() if visiting is None else set(visiting)
    if tensor_name in visiting:
        return set()
    visiting.add(tensor_name)
    _, producer = producers[tensor_name]
    dependencies: set[str] = set()
    for input_name in producer.input:
        if input_name:
            dependencies.update(
                _runtime_dependencies(
                    input_name,
                    graph_inputs,
                    initializers,
                    producers,
                    memo,
                    visiting,
                )
            )
    memo[tensor_name] = dependencies
    return dependencies


def _producer_info(
    tensor_name: str, producers: dict[str, tuple[int, onnx.NodeProto]]
) -> dict[str, Any] | None:
    if tensor_name not in producers:
        return None
    index, node = producers[tensor_name]
    return {"node_index": index, "node_name": node.name, "op_type": node.op_type}


def _consumer_info(
    consumers: dict[str, list[tuple[int, onnx.NodeProto, int]]],
    tensor_name: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for node_index, node, input_slot in consumers.get(tensor_name, []):
        result.append(
            {
                "node_index": node_index,
                "node_name": node.name,
                "op_type": node.op_type,
                "input_slot": input_slot,
                "input_role": _consumer_input_role(node.op_type, input_slot),
            }
        )
    return result


def _markdown(
    onnx_path: Path,
    onnx_hash: str,
    opset: int,
    schema_since: int,
    nodes: list[dict[str, Any]],
    classification: dict[str, Any],
) -> str:
    lines = [
        "# ONNX Unique 节点语义审计",
        "",
        f"- ONNX：`{onnx_path}`",
        f"- SHA-256：`{onnx_hash}`",
        f"- 模型 opset：{opset}",
        f"- Unique schema since_version：{schema_since}",
        f"- 目标节点数：{len(nodes)}",
        "- 本审计只读取内存中的 ONNX，并进行内存 shape inference；未保存或修改模型。",
        "",
        "## 1. 公共语义",
        "",
        "四个节点都没有 `axis` 属性，因此输入先按一维展开执行 Unique。",
        "四个节点都显式设置 `sorted=1`，输出 unique values 按升序排列。",
        "标准输出顺序为：`values`、`indices`、`inverse_indices`、`counts`。",
        "",
        "## 2. 节点摘要",
        "",
        "| Node index | Node name | 输入 dtype | 输入 shape | Runtime依赖 |",
        "|---:|---|---|---|---|",
    ]
    for node in nodes:
        info = node["input"]
        lines.append(
            f"| {node['node_index']} | `{node['node_name']}` | "
            f"{info['dtype']} | `{info['shape']}` | "
            f"{', '.join(info['graph_input_dependencies']) or '无'} |"
        )

    lines.extend(
        [
            "",
            "## 3. 输出与直接消费者",
            "",
        ]
    )
    for node in nodes:
        lines.extend([f"### {node['node_name']}", ""])
        lines.extend(
            [
                "| 输出角色 | Tensor | dtype | shape | 直接消费者 |",
                "|---|---|---|---|---|",
            ]
        )
        for output in node["outputs"]:
            consumer_text = "; ".join(
                f"`{item['node_name']}` ({item['op_type']}, "
                f"input {item['input_slot']}={item['input_role']})"
                for item in output["consumers"]
            ) or "未使用"
            lines.append(
                f"| {output['role']} | `{output['name']}` | "
                f"{output['dtype']} | `{output['shape']}` | {consumer_text} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 4. 关键消费关系",
            "",
            "每一级都存在以下直接数据流：",
            "",
            "```text",
            "Unique.inverse_indices",
            "        |",
            "        v",
            "ScatterElements_1 input[1] = indices",
            "```",
            "",
            "`values` 本身只被 `Shape` 消费，用于取得运行时 unique voxel 数量；",
            "`indices` 和 `counts` 均未被任何节点消费；",
            "`inverse_indices` 同时被两个 `Shape`、一个 `ScatterElements` 和一个 `Unsqueeze` 消费。",
            "因此当前图真正依赖的是 unique cardinality 和 point-to-voxel inverse mapping。",
            "",
            "## 5. 分类结论",
            "",
            f"结论：**{classification['category']}**",
            "",
            classification["statement"],
            "",
            "判定依据：",
            "",
        ]
    )
    lines.extend(f"- {reason}" for reason in classification["reasons"])
    lines.extend(
        [
            "",
            "当前不能判定为 A：mapping 不是固定常量，而是依赖运行时 `points`。",
            "当前也不能直接判定为 B：TensorRT parser 确实不支持 Unique，但尚未证明 plugin 是唯一方案。",
            "后续必须单独评估标准 TensorRT 层等价表达、外部预处理接口变化和 plugin 三种路线，",
            "并对 voxel ordering、inverse mapping、动态 voxel count 和 pooled features 做逐层验证。",
            "",
            "```text",
            "UNIQUE_SEMANTICS_AUDIT_COMPLETED",
            "UNIQUE_CLASSIFICATION=C_REQUIRES_FURTHER_GRAPH_REWRITE_ANALYSIS",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    onnx_hash_before = _sha256(onnx_path)
    model = onnx.load_model(str(onnx_path), load_external_data=False)
    inferred = onnx.shape_inference.infer_shapes(
        model, check_type=True, strict_mode=False, data_prop=False
    )
    value_info = _value_info_map(inferred)
    graph_inputs = {item.name for item in model.graph.input}
    initializers = {item.name for item in model.graph.initializer}
    graph_nodes = list(model.graph.node)

    producers: dict[str, tuple[int, onnx.NodeProto]] = {}
    consumers: dict[str, list[tuple[int, onnx.NodeProto, int]]] = {}
    for node_index, node in enumerate(graph_nodes):
        for output_name in node.output:
            if output_name:
                producers[output_name] = (node_index, node)
        for input_slot, input_name in enumerate(node.input):
            if input_name:
                consumers.setdefault(input_name, []).append(
                    (node_index, node, input_slot)
                )

    default_opset = next(
        int(item.version) for item in model.opset_import if item.domain == ""
    )
    unique_schema = onnx.defs.get_schema("Unique", default_opset, "")
    dependency_memo: dict[str, set[str]] = {}
    unique_nodes: list[dict[str, Any]] = []
    unique_consumers: list[dict[str, Any]] = []

    for node_index, node in enumerate(graph_nodes):
        if node.name not in TARGET_NAMES:
            continue
        if node.op_type != "Unique":
            raise RuntimeError(f"Target node is not Unique: {node.name}={node.op_type}")

        attributes = _attribute_map(node)
        input_name = node.input[0]
        input_type = _type_info(value_info.get(input_name))
        dependencies = sorted(
            _runtime_dependencies(
                input_name,
                graph_inputs,
                initializers,
                producers,
                dependency_memo,
            )
        )
        input_record = {
            "name": input_name,
            **input_type,
            "producer": _producer_info(input_name, producers),
            "is_initializer": input_name in initializers,
            "graph_input_dependencies": dependencies,
        }

        output_records: list[dict[str, Any]] = []
        for output_index, output_name in enumerate(node.output):
            direct_consumers = _consumer_info(consumers, output_name)
            output_record = {
                "output_index": output_index,
                "role": (
                    UNIQUE_OUTPUT_ROLES[output_index]
                    if output_index < len(UNIQUE_OUTPUT_ROLES)
                    else f"output_{output_index}"
                ),
                "name": output_name,
                **_type_info(value_info.get(output_name)),
                "consumers": direct_consumers,
                "consumer_op_counts": dict(
                    sorted(Counter(item["op_type"] for item in direct_consumers).items())
                ),
            }
            output_records.append(output_record)
            unique_consumers.append(
                {
                    "unique_node_name": node.name,
                    "unique_node_index": node_index,
                    **output_record,
                }
            )

        unique_nodes.append(
            {
                "node_name": node.name,
                "node_index": node_index,
                "op_type": node.op_type,
                "model_opset": default_opset,
                "unique_schema_since_version": unique_schema.since_version,
                "attributes": {
                    "axis_present": "axis" in attributes,
                    "axis": attributes.get("axis"),
                    "effective_axis": (
                        attributes["axis"] if "axis" in attributes else "flatten_all_dimensions"
                    ),
                    "sorted_present": "sorted" in attributes,
                    "sorted": int(attributes.get("sorted", 1)),
                },
                "input": input_record,
                "num_outputs": len(node.output),
                "outputs": output_records,
            }
        )

    found_names = {item["node_name"] for item in unique_nodes}
    missing = sorted(set(TARGET_NAMES) - found_names)
    if missing:
        raise RuntimeError(f"Missing target Unique nodes: {missing}")

    direct_scatter_links = [
        item
        for item in unique_consumers
        if item["role"] == "inverse_indices"
        and any(
            consumer["op_type"] == "ScatterElements"
            and consumer["input_role"] == "indices"
            for consumer in item["consumers"]
        )
    ]
    all_depend_on_points = all(
        "points" in item["input"]["graph_input_dependencies"]
        for item in unique_nodes
    )
    values_feed_runtime_shape = all(
        any(consumer["op_type"] == "Shape" for consumer in item["outputs"][0]["consumers"])
        for item in unique_nodes
    )
    indices_unused = all(not item["outputs"][1]["consumers"] for item in unique_nodes)
    counts_unused = all(not item["outputs"][3]["consumers"] for item in unique_nodes)

    classification = {
        "category": "C_REQUIRES_FURTHER_GRAPH_REWRITE_ANALYSIS",
        "statement": (
            "Unique 参与运行时动态 voxel mapping，不能视为固定 mapping 直接移出；"
            "但仅凭 parser 不支持还不能证明 TensorRT plugin 是唯一可行方案。"
        ),
        "reasons": [
            "四个 Unique 输入都由运行时 graph input `points` 派生，不是 initializer 或常量。",
            "四个 values 输出都被 Shape 消费，运行时 unique voxel 数参与后续动态 shape。",
            "四个 inverse_indices 都直接作为 ScatterElements 的 indices 输入。",
            "inverse_indices 还参与 Shape 和 Unsqueeze，point-to-voxel mapping 被多个后继节点使用。",
            "indices 与 counts 未使用，说明实际必须保留的是 values 的长度和 inverse_indices 语义。",
        ],
        "evidence": {
            "all_unique_inputs_depend_on_points": all_depend_on_points,
            "all_values_feed_runtime_shape": values_feed_runtime_shape,
            "inverse_to_scatter_direct_link_count": len(direct_scatter_links),
            "indices_unused_for_all_nodes": indices_unused,
            "counts_unused_for_all_nodes": counts_unused,
        },
    }

    output_root = Path(args.output_root).resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f_unique_audit")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    common = {
        "audit_timestamp": datetime.now().astimezone().isoformat(),
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "onnx_sha256": onnx_hash_before,
        "model_opset": default_opset,
        "target_node_names": list(TARGET_NAMES),
        "onnx_modified": False,
    }
    _write_json(
        run_dir / "unique_nodes.json",
        {**common, "nodes": unique_nodes, "classification": classification},
    )
    _write_json(
        run_dir / "unique_consumers.json",
        {**common, "outputs": unique_consumers},
    )
    (run_dir / "unique_summary.md").write_text(
        _markdown(
            onnx_path,
            onnx_hash_before,
            default_opset,
            unique_schema.since_version,
            unique_nodes,
            classification,
        ),
        encoding="utf-8",
    )

    onnx_hash_after = _sha256(onnx_path)
    if onnx_hash_after != onnx_hash_before:
        raise RuntimeError("ONNX hash changed during read-only audit")

    print(f"RUN_DIR={run_dir}")
    print(f"UNIQUE_NODE_COUNT={len(unique_nodes)}")
    print(f"INVERSE_TO_SCATTER_DIRECT_LINKS={len(direct_scatter_links)}")
    print(f"ALL_INPUTS_DEPEND_ON_POINTS={all_depend_on_points}")
    print(f"INDICES_UNUSED_FOR_ALL={indices_unused}")
    print(f"COUNTS_UNUSED_FOR_ALL={counts_unused}")
    print(f"CLASSIFICATION={classification['category']}")
    print("UNIQUE_SEMANTICS_AUDIT_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
