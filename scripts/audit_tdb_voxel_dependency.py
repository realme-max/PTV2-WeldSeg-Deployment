"""Static downstream dependency audit for GCN_res Transition Down Blocks.

The audit treats each Unique.inverse_indices tensor as a taint source and walks
the existing ONNX graph in topological order. It does not save or mutate ONNX,
modify model code, invoke TensorRT, or execute inference.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
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
STAGES = ("tdb_1", "tdb_2", "tdb_3", "tdb_4")
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


def _dimension(dim: onnx.TensorShapeProto.Dimension) -> int | str:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    if dim.dim_param:
        return dim.dim_param
    return "?"


def _value_info_map(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    values = list(model.graph.input) + list(model.graph.output) + list(
        model.graph.value_info
    )
    return {item.name: item for item in values}


def _type_info(
    name: str, value_info: dict[str, onnx.ValueInfoProto]
) -> dict[str, Any]:
    item = value_info.get(name)
    if item is None or not item.type.HasField("tensor_type"):
        return {"dtype": None, "shape": None, "has_dynamic_dimension": None}
    tensor_type = item.type.tensor_type
    shape = [_dimension(dim) for dim in tensor_type.shape.dim]
    return {
        "dtype": (
            onnx.helper.tensor_dtype_to_string(tensor_type.elem_type)
            if tensor_type.elem_type
            else None
        ),
        "shape": shape,
        "has_dynamic_dimension": any(not isinstance(dim, int) for dim in shape),
    }


def _build_graph_maps(
    nodes: list[onnx.NodeProto],
) -> tuple[
    dict[str, tuple[int, onnx.NodeProto]],
    dict[str, list[tuple[int, onnx.NodeProto, int]]],
]:
    producers: dict[str, tuple[int, onnx.NodeProto]] = {}
    consumers: dict[str, list[tuple[int, onnx.NodeProto, int]]] = {}
    for node_index, node in enumerate(nodes):
        for output_name in node.output:
            if output_name:
                producers[output_name] = (node_index, node)
        for input_slot, input_name in enumerate(node.input):
            if input_name:
                consumers.setdefault(input_name, []).append(
                    (node_index, node, input_slot)
                )
    return producers, consumers


def _find_unique(
    nodes: list[onnx.NodeProto], stage: str
) -> tuple[int, onnx.NodeProto]:
    expected_name = f"/model/{stage}/Unique"
    matches = [
        (index, node)
        for index, node in enumerate(nodes)
        if node.name == expected_name and node.op_type == "Unique"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {expected_name}, found {len(matches)}")
    return matches[0]


def _taint_subgraph(
    nodes: list[onnx.NodeProto], seed_tensor: str, seed_node_index: int
) -> dict[str, Any]:
    tainted_tensors = {seed_tensor}
    dependent_nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for node_index in range(seed_node_index + 1, len(nodes)):
        node = nodes[node_index]
        tainted_inputs = [name for name in node.input if name in tainted_tensors]
        if not tainted_inputs:
            continue
        outputs = [name for name in node.output if name]
        dependent_nodes.append(
            {
                "node_index": node_index,
                "node_name": node.name,
                "op_type": node.op_type,
                "tainted_inputs": tainted_inputs,
                "outputs_marked_dependent": outputs,
            }
        )
        for input_name in tainted_inputs:
            edges.append(
                {
                    "from_kind": "tensor",
                    "from": input_name,
                    "to_kind": "node",
                    "to": node.name or f"node_{node_index}",
                    "to_node_index": node_index,
                    "to_op_type": node.op_type,
                }
            )
        for output_name in outputs:
            edges.append(
                {
                    "from_kind": "node",
                    "from": node.name or f"node_{node_index}",
                    "from_node_index": node_index,
                    "from_op_type": node.op_type,
                    "to_kind": "tensor",
                    "to": output_name,
                }
            )
        tainted_tensors.update(outputs)

    return {
        "seed_tensor": seed_tensor,
        "dependent_nodes": dependent_nodes,
        "dependent_tensors": sorted(tainted_tensors),
        "dependency_edges": edges,
    }


def _shortest_path(
    seed_tensor: str,
    target_tensor: str,
    consumers: dict[str, list[tuple[int, onnx.NodeProto, int]]],
) -> list[dict[str, Any]]:
    queue: deque[tuple[str, str]] = deque([("tensor", seed_tensor)])
    previous: dict[tuple[str, str], tuple[str, str] | None] = {
        ("tensor", seed_tensor): None
    }
    metadata: dict[tuple[str, str], dict[str, Any]] = {
        ("tensor", seed_tensor): {"kind": "tensor", "name": seed_tensor}
    }

    target_key = ("tensor", target_tensor)
    while queue:
        kind, name = queue.popleft()
        key = (kind, name)
        if key == target_key:
            break
        if kind != "tensor":
            continue
        for node_index, node, input_slot in consumers.get(name, []):
            node_key = ("node", f"{node_index}:{node.name}")
            if node_key not in previous:
                previous[node_key] = key
                metadata[node_key] = {
                    "kind": "node",
                    "node_index": node_index,
                    "node_name": node.name,
                    "op_type": node.op_type,
                    "entered_via_input_slot": input_slot,
                    "entered_via_tensor": name,
                }
            for output_name in node.output:
                if not output_name:
                    continue
                output_key = ("tensor", output_name)
                if output_key in previous:
                    continue
                previous[output_key] = node_key
                metadata[output_key] = {"kind": "tensor", "name": output_name}
                queue.append(output_key)

    if target_key not in previous:
        return []
    keys: list[tuple[str, str]] = []
    cursor: tuple[str, str] | None = target_key
    while cursor is not None:
        keys.append(cursor)
        cursor = previous[cursor]
    return [metadata[key] for key in reversed(keys)]


def _module_kind(node_name: str, op_type: str, current_stage: str) -> str:
    if node_name.startswith(f"/model/{current_stage}/"):
        return "current_tdb_voxel_pool"
    if any(node_name.startswith(f"/model/{stage}/") for stage in STAGES):
        return "downstream_tdb"
    if "/ptb_" in node_name or "/gva/" in node_name:
        return "point_transformer_attention_or_geometry"
    if "/tub_" in node_name:
        return "transition_up_interpolation_or_skip"
    if "/mlp/" in node_name:
        return "segmentation_head"
    if op_type in {"TopK", "Gather", "GatherND", "GatherElements"}:
        return "geometry_or_indexing"
    return "other"


def _first_scope_exits(
    stage: str,
    dependency: dict[str, Any],
    producers: dict[str, tuple[int, onnx.NodeProto]],
) -> list[dict[str, Any]]:
    prefix = f"/model/{stage}/"
    exits: list[dict[str, Any]] = []
    for node in dependency["dependent_nodes"]:
        if node["node_name"].startswith(prefix):
            continue
        crossing_inputs: list[str] = []
        for tensor_name in node["tainted_inputs"]:
            producer = producers.get(tensor_name)
            if producer is not None and producer[1].name.startswith(prefix):
                crossing_inputs.append(tensor_name)
        if crossing_inputs:
            exits.append(
                {
                    "node_index": node["node_index"],
                    "node_name": node["node_name"],
                    "op_type": node["op_type"],
                    "crossing_tensors": crossing_inputs,
                }
            )
    return exits


def _path_text(path: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in path:
        if item["kind"] == "tensor":
            parts.append(f"tensor:{item['name']}")
        else:
            parts.append(
                f"node:{item['node_name']}[{item['op_type']}]"
            )
    return " → ".join(parts)


def _report(
    onnx_path: Path,
    onnx_hash: str,
    graph_outputs: list[str],
    stages: list[dict[str, Any]],
    external_inputs: list[dict[str, Any]],
) -> str:
    lines = [
        "# Transition Down Block voxel dependency audit",
        "",
        f"- ONNX：`{onnx_path}`",
        f"- SHA-256：`{onnx_hash}`",
        f"- Network outputs：{', '.join(f'`{name}`' for name in graph_outputs)}",
        "- 分析方式：从每级 `Unique.inverse_indices` 开始，对主图执行保守静态 taint 传播。",
        "- 完整受影响节点、张量和边记录在 `tdb_dependency.json`。",
        "- 未修改或保存 ONNX，未运行 TensorRT、模型 forward 或 inference。",
        "",
        "## 1. 总结",
        "",
        "| Stage | Unique node | 受影响节点数 | 到达 logits | 影响 attention/geometry | 分类 |",
        "|---|---|---:|---|---|---|",
    ]
    for item in stages:
        lines.append(
            f"| {item['stage']} | `{item['unique']['node_name']}` | "
            f"{item['reachability']['dependent_node_count']} | "
            f"{item['reachability']['reaches_network_output']} | "
            f"{item['classification']['influences_attention_or_geometry']} | "
            f"{item['classification']['category']} |"
        )

    lines.extend(
        [
            "",
            "统一结论：**B_AFFECTS_ATTENTION_OR_GEOMETRY**。",
            "",
            "`inverse_indices` 首先控制 voxel count、mean XYZ 和 max pooled features 的聚合，",
            "随后 pooled XYZ/features 进入后续 PointTransformer attention、KNN geometry、",
            "Transition Up 插值和 segmentation head，最终影响 `logits`。它不只是局部临时计数。",
            "",
            "## 2. 每级完整链路摘要",
            "",
        ]
    )
    for item in stages:
        lines.extend(
            [
                f"### {item['stage']}",
                "",
                f"- Seed：`{item['unique']['inverse_indices_name']}`",
                f"- 输入点数 shape：`{item['unique']['input_shape']}`",
                f"- 直接 ScatterElements：`{item['unique']['direct_scatter_node']}`",
                f"- 受影响节点数：{item['reachability']['dependent_node_count']}",
                f"- 受影响张量数：{item['reachability']['dependent_tensor_count']}",
                f"- 到达 network output：{item['reachability']['reached_network_outputs']}",
                f"- 当前 TDB 外第一批消费者：{item['reachability']['first_scope_exits']}",
                "",
                "到 `logits` 的最短静态依赖路径：",
                "",
                "```text",
                item["reachability"]["shortest_path_to_logits_text"],
                "```",
                "",
                "模块影响计数：",
                "",
                "```json",
                json.dumps(
                    item["reachability"]["module_kind_counts"],
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## 3. 如果将 voxel mapping 外移，最小逻辑输入",
            "",
            "仅提供四个 mapping 还不够：当前图还通过 `Shape(Unique.values)` 获取每级动态 voxel 数。",
            "最小逻辑信息应包含每级 inverse mapping 与 voxel count，共八个输入候选：",
            "",
            "| 建议输入 | dtype | shape | 动态性 | 语义 |",
            "|---|---|---|---|---|",
        ]
    )
    for item in external_inputs:
        lines.append(
            f"| `{item['name']}` | {item['dtype']} | `{item['shape']}` | "
            f"{item['dynamic']} | {item['semantics']} |"
        )
    lines.extend(
        [
            "",
            "约束：",
            "",
            "- mapping ID 必须与 `sorted=1` 的 unique key 顺序一致，连续覆盖 `[0, M_i-1]`。",
            "- 第一级 mapping 长度固定为 2048；后三级长度分别等于前一级动态 voxel count。",
            "- 第二至四级 mapping 依赖前一级 mean pooled XYZ，因此外部预处理必须顺序复现整个几何 hierarchy。",
            "- feature max pooling 仍依赖网络内部 learned features，不能在纯几何预处理阶段提前计算。",
            "- 当前 B=1，不需要 batch offsets；扩展到 B>1 时还需审计 batch offsets 和最小 voxel count crop。",
            "- 这些是静态分析得到的最小逻辑信息，不代表 TensorRT 已支持用 shape tensor 构造对应动态输出。",
            "",
            "## 4. 可拆分性判断",
            "",
            "Unique 的 mapping 生成是纯几何、依赖 `points`，理论上可候选外移；",
            "但四级 mapping 存在顺序依赖，而且 mapping 同时决定 pooled XYZ 的顺序、动态点数和 feature aggregation。",
            "因此不能只增加四个未经约束的 `voxel_mapping_i` 就视为完成拆分。",
            "下一步若评估外移方案，必须验证八个逻辑输入、逐级 pooled XYZ/features 和最终 logits。",
            "",
            "```text",
            "TDB_VOXEL_DEPENDENCY_AUDIT_COMPLETED",
            "TDB_UNIQUE_ROLE_CLASSIFICATION=B_AFFECTS_ATTENTION_OR_GEOMETRY",
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
    hash_before = _sha256(onnx_path)

    model = onnx.load_model(str(onnx_path), load_external_data=False)
    inferred = onnx.shape_inference.infer_shapes(
        model, check_type=True, strict_mode=False, data_prop=False
    )
    nodes = list(model.graph.node)
    value_info = _value_info_map(inferred)
    producers, consumers = _build_graph_maps(nodes)
    graph_outputs = [item.name for item in model.graph.output]
    if "logits" not in graph_outputs:
        raise RuntimeError(f"Expected logits graph output, got {graph_outputs}")

    stage_results: list[dict[str, Any]] = []
    proposed_inputs: list[dict[str, Any]] = []
    for stage_number, stage in enumerate(STAGES, start=1):
        unique_index, unique_node = _find_unique(nodes, stage)
        if len(unique_node.output) < 3:
            raise RuntimeError(f"Unique node lacks inverse_indices: {unique_node.name}")
        inverse_name = unique_node.output[2]
        unique_input_name = unique_node.input[0]
        input_info = _type_info(unique_input_name, value_info)
        inverse_info = _type_info(inverse_name, value_info)
        direct_consumers = consumers.get(inverse_name, [])
        scatter_consumers = [
            (index, node, input_slot)
            for index, node, input_slot in direct_consumers
            if node.op_type == "ScatterElements" and input_slot == 1
        ]
        if len(scatter_consumers) != 1:
            raise RuntimeError(
                f"Expected one inverse->ScatterElements link for {stage}, "
                f"found {len(scatter_consumers)}"
            )

        dependency = _taint_subgraph(nodes, inverse_name, unique_index)
        reached_outputs = sorted(
            set(dependency["dependent_tensors"]).intersection(graph_outputs)
        )
        shortest = _shortest_path(inverse_name, "logits", consumers)
        if not shortest:
            raise RuntimeError(f"No dependency path from {inverse_name} to logits")

        module_counts = Counter(
            _module_kind(item["node_name"], item["op_type"], stage)
            for item in dependency["dependent_nodes"]
        )
        attention_examples = [
            {
                "node_index": item["node_index"],
                "node_name": item["node_name"],
                "op_type": item["op_type"],
            }
            for item in dependency["dependent_nodes"]
            if _module_kind(item["node_name"], item["op_type"], stage)
            == "point_transformer_attention_or_geometry"
        ][:20]
        geometry_examples = [
            {
                "node_index": item["node_index"],
                "node_name": item["node_name"],
                "op_type": item["op_type"],
            }
            for item in dependency["dependent_nodes"]
            if _module_kind(item["node_name"], item["op_type"], stage)
            in {
                "point_transformer_attention_or_geometry",
                "transition_up_interpolation_or_skip",
                "geometry_or_indexing",
            }
        ][:30]
        influences_attention_or_geometry = bool(attention_examples or geometry_examples)

        stage_result = {
            "stage": stage,
            "unique": {
                "node_index": unique_index,
                "node_name": unique_node.name,
                "unique_input_name": unique_input_name,
                "input_dtype": input_info["dtype"],
                "input_shape": input_info["shape"],
                "inverse_indices_name": inverse_name,
                "inverse_indices_dtype": inverse_info["dtype"],
                "inverse_indices_shape": inverse_info["shape"],
                "direct_scatter_node": scatter_consumers[0][1].name,
                "direct_scatter_node_index": scatter_consumers[0][0],
                "direct_scatter_input_slot": scatter_consumers[0][2],
            },
            "reachability": {
                "dependent_node_count": len(dependency["dependent_nodes"]),
                "dependent_tensor_count": len(dependency["dependent_tensors"]),
                "reaches_network_output": bool(reached_outputs),
                "reached_network_outputs": reached_outputs,
                "first_scope_exits": _first_scope_exits(
                    stage, dependency, producers
                ),
                "module_kind_counts": dict(sorted(module_counts.items())),
                "attention_examples": attention_examples,
                "geometry_examples": geometry_examples,
                "shortest_path_to_logits": shortest,
                "shortest_path_to_logits_text": _path_text(shortest),
                "full_dependency_subgraph": dependency,
            },
            "classification": {
                "category": "B_AFFECTS_ATTENTION_OR_GEOMETRY",
                "influences_feature_pooling": True,
                "influences_attention_or_geometry": influences_attention_or_geometry,
                "reaches_logits": "logits" in reached_outputs,
                "cannot_be_treated_as_local_only": True,
            },
        }
        stage_results.append(stage_result)

        input_shape = input_info["shape"]
        proposed_inputs.extend(
            [
                {
                    "stage": stage,
                    "name": f"voxel_inverse_indices_{stage_number}",
                    "dtype": "int64",
                    "shape": input_shape,
                    "dynamic": input_info["has_dynamic_dimension"],
                    "semantics": (
                        "每个 stage 输入点到 sorted unique voxel ID 的映射；"
                        "ID 连续且顺序必须与原图完全一致"
                    ),
                    "replaces_unique_output_role": "inverse_indices",
                },
                {
                    "stage": stage,
                    "name": f"voxel_count_{stage_number}",
                    "dtype": "int64",
                    "shape": [1],
                    "dynamic": "runtime_value",
                    "semantics": (
                        "该级 materialized unique voxel 数 M_i；"
                        "替代 Shape(Unique.values) 提供的运行时长度"
                    ),
                    "replaces_unique_output_role": "values_shape",
                },
            ]
        )

    if not all(
        item["classification"]["influences_attention_or_geometry"]
        and item["classification"]["reaches_logits"]
        for item in stage_results
    ):
        raise RuntimeError("Not every TDB dependency reaches attention/geometry and logits")

    payload = {
        "audit_timestamp": datetime.now().astimezone().isoformat(),
        "analysis_method": (
            "Conservative static taint propagation: if any node input depends on "
            "Unique.inverse_indices, all node outputs are marked dependent."
        ),
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "onnx_sha256": hash_before,
        "graph_inputs": [item.name for item in model.graph.input],
        "graph_outputs": graph_outputs,
        "stages": stage_results,
        "externalization_analysis": {
            "current_inputs": [item.name for item in model.graph.input],
            "proposed_minimum_logical_inputs": proposed_inputs,
            "proposed_input_count": len(proposed_inputs),
            "mapping_only_is_insufficient": True,
            "why_mapping_only_is_insufficient": (
                "The graph also consumes Shape(Unique.values) as the runtime voxel "
                "count used to size aggregation targets."
            ),
            "inter_stage_dependency": (
                "voxel_inverse_indices_2..4 depend on mean pooled XYZ produced by "
                "the preceding stage, so preprocessing must reproduce the hierarchy "
                "sequentially."
            ),
            "learned_feature_constraint": (
                "Feature max pooling uses learned stage features and cannot be fully "
                "precomputed by geometry-only preprocessing."
            ),
        },
        "overall_classification": {
            "category": "B_AFFECTS_ATTENTION_OR_GEOMETRY",
            "reason": (
                "All four inverse mappings alter pooled XYZ/features, propagate into "
                "PointTransformer attention or geometry, and reach logits."
            ),
        },
        "onnx_modified": False,
        "model_execution_performed": False,
        "tensorrt_execution_performed": False,
    }

    output_root = Path(args.output_root).resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f_tdb_dependency_audit")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "tdb_dependency.json", payload)
    (run_dir / "tdb_dependency_report.md").write_text(
        _report(onnx_path, hash_before, graph_outputs, stage_results, proposed_inputs),
        encoding="utf-8",
    )

    if _sha256(onnx_path) != hash_before:
        raise RuntimeError("ONNX hash changed during read-only audit")

    print(f"RUN_DIR={run_dir}")
    for item in stage_results:
        print(
            f"{item['stage']} dependent_nodes="
            f"{item['reachability']['dependent_node_count']} "
            f"reaches_logits={item['classification']['reaches_logits']} "
            "classification="
            f"{item['classification']['category']}"
        )
    print(f"PROPOSED_LOGICAL_INPUT_COUNT={len(proposed_inputs)}")
    print("TDB_VOXEL_DEPENDENCY_AUDIT_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
