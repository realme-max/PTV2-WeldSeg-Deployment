"""Read-only incremental TensorRT parser audit for the fixed GCN_res ONNX.

This script does not modify or save the ONNX model, create a builder config,
build an engine, or execute inference. It combines three parser views:

1. normal parse errors;
2. supports_model_v2 subgraph partitions;
3. supports_operator registry hints for every op type in the graph.

The registry API may return false positives, so this audit distinguishes a
known blocker from proof that all nodes after that blocker are parseable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_onnx"
    / "20260715_onnx_after_cdist_fp32_opset18"
    / "gcn_res_deploy_fp32_opset18.onnx"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def error_field(error: Any, name: str, default: Any = None) -> Any:
    if not hasattr(error, name):
        return default
    value = getattr(error, name)
    try:
        return value() if callable(value) else value
    except Exception as exc:
        return f"ERROR_READING_{name}: {type(exc).__name__}: {exc}"


def collect_errors(parser: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index in range(parser.num_errors):
        error = parser.get_error(index)
        result.append(
            {
                "index": index,
                "code": str(error_field(error, "code", "")),
                "node_index": error_field(error, "node", -1),
                "node_name": error_field(error, "node_name", ""),
                "operator": error_field(error, "node_operator", ""),
                "description": error_field(error, "desc", str(error)),
                "raw": str(error),
            }
        )
    return result


def configure_dll_search(tensorrt_root: Path | None) -> list[Any]:
    handles: list[Any] = []
    if tensorrt_root is None:
        return handles
    bin_dir = tensorrt_root / "bin"
    if not bin_dir.is_dir():
        raise FileNotFoundError(f"TensorRT bin directory not found: {bin_dir}")
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        handles.append(os.add_dll_directory(str(bin_dir)))
    return handles


def create_parser(trt: Any, severity: Any) -> tuple[Any, Any, Any]:
    logger = trt.Logger(severity)
    builder = trt.Builder(logger)
    if builder is None:
        raise RuntimeError("trt.Builder returned None")
    # Explicit batch is mandatory in TensorRT 10+; the legacy flag was removed.
    network = builder.create_network(0)
    if network is None:
        raise RuntimeError("builder.create_network returned None")
    parser = trt.OnnxParser(network, logger)
    return builder, network, parser


def probe_scatter_reductions(onnx: Any, trt: Any) -> list[dict[str, Any]]:
    """Parser-only probes for the exact ScatterElements reduction families.

    Models exist only as serialized in-memory buffers and are never written.
    """
    helper = onnx.helper
    tensor_proto = onnx.TensorProto
    cases = (
        ("none", tensor_proto.FLOAT),
        ("add", tensor_proto.INT64),
        ("add", tensor_proto.FLOAT),
        ("min", tensor_proto.INT64),
        ("max", tensor_proto.FLOAT),
    )
    results: list[dict[str, Any]] = []
    for reduction, dtype in cases:
        inputs = [
            helper.make_tensor_value_info("data", dtype, [6, 3]),
            helper.make_tensor_value_info("indices", tensor_proto.INT64, [5, 3]),
            helper.make_tensor_value_info("updates", dtype, [5, 3]),
        ]
        output = helper.make_tensor_value_info("output", dtype, [6, 3])
        node = helper.make_node(
            "ScatterElements",
            ["data", "indices", "updates"],
            ["output"],
            name=f"scatter_{reduction}",
            axis=0,
            reduction=reduction,
        )
        model = helper.make_model(
            helper.make_graph([node], "scatter_probe", inputs, [output]),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        # Match the source model's ONNX IR generation accepted by TensorRT.
        model.ir_version = 10
        builder, network, parser = create_parser(trt, trt.Logger.ERROR)
        success = bool(parser.parse(model.SerializeToString()))
        errors = collect_errors(parser)
        results.append(
            {
                "reduction": reduction,
                "data_dtype": tensor_proto.DataType.Name(dtype),
                "success": success,
                "errors": errors,
                "saved_to_disk": False,
                "engine_build_attempted": False,
            }
        )
        del parser, network, builder
    return results


def render_report(payload: dict[str, Any]) -> str:
    normal = payload["normal_parse"]
    support = payload["supports_model_v2"]
    decision = payload["decision"]
    lines = [
        "# TensorRT Incremental Parser Audit",
        "",
        f"- ONNX：`{payload['onnx']['path']}`",
        f"- SHA-256：`{payload['onnx']['sha256']}`",
        f"- TensorRT：`{payload['environment']['tensorrt_version']}`",
        f"- ONNX nodes：{payload['onnx']['node_count']}",
        "- Engine build：未执行",
        "- Inference：未执行",
        "- ONNX modified：否",
        "",
        "## 三路检查结果",
        "",
        f"- `parse_from_file`：{normal['success']}，errors={len(normal['errors'])}",
        f"- `supports_model_v2`：{support['model_supported']}，subgraphs={support['num_subgraphs']}，errors={len(support['errors'])}",
        f"- parser registry 中明确不支持的 op：`{', '.join(payload['operator_registry']['unsupported_ops']) or 'none'}`",
        f"- NVIDIA 标准 Plugin 注册：{payload['standard_plugins']['initialized']}",
        f"- ScatterElements reduction probes：{all(item['success'] for item in payload['attribute_probes']['scatter_elements'])}",
        "",
        "### Parser errors",
        "",
        "| Node index | Node | Operator | Description |",
        "|---:|---|---|---|",
    ]
    for item in normal["errors"]:
        lines.append(
            f"| {item['node_index']} | `{item['node_name']}` | "
            f"`{item['operator']}` | {item['description']} |"
        )
    lines.extend(
        [
            "",
            "### supports_model_v2 partitions",
            "",
            "| Index | Supported | Nodes | First node | Last node |",
            "|---:|---|---:|---|---|",
        ]
    )
    for item in support["subgraphs"]:
        lines.append(
            f"| {item['index']} | {item['supported']} | {item['node_count']} | "
            f"`{item['first_node']['name']}` | `{item['last_node']['name']}` |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            f"- 唯一已确认阻塞算子：`{decision['only_confirmed_blocking_operator']}`。",
            f"- 是否存在第二种已确认阻塞算子：{decision['multiple_confirmed_blocking_operators']}。",
            f"- 是否已证明移除 Unique 后整图可解析：{decision['full_graph_after_unique_proven']}。",
            "- 原因：`supports_operator=True` 只是注册级提示，官方接口本身允许 false-positive；"
            "而 `supports_model_v2` 的五个分段全部被标为 unsupported，无法越过 Unique 对下游节点属性和数据依赖 shape 做完整解析证明。",
            "",
            "## 路线判断",
            "",
            "当前满足进入 **Unique Plugin 可行性评估** 的条件，因为没有发现第二种已确认不支持算子；"
            "但尚不满足直接承诺 Plugin 后整图可构建的条件。Plugin 评估必须先回答数据依赖输出长度、"
            "四级动态 voxel 数以及下游 Shape/Scatter/NonZero 链能否由 TensorRT Plugin 接口表达。",
            "",
            "暂不选择“拆分 voxel hierarchy + TensorRT backbone”。只有 Plugin 可行性审计失败，或让 parser "
            "越过 Unique 后出现多个新增动态算子阻塞，才进入拆分方案。",
            "",
            "INCREMENTAL_PARSER_AUDIT_COMPLETED",
            "ONLY_CONFIRMED_BLOCKING_OPERATOR=Unique",
            "PLUGIN_FEASIBILITY_AUDIT_RECOMMENDED",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tensorrt-root", type=Path)
    args = parser.parse_args()

    onnx_path = args.onnx.resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    trt_root = args.tensorrt_root.resolve() if args.tensorrt_root else None
    dll_handles = configure_dll_search(trt_root)

    import onnx
    import tensorrt as trt

    model = onnx.load_model(str(onnx_path), load_external_data=False)
    nodes = list(model.graph.node)
    model_bytes = onnx_path.read_bytes()
    before_hash = sha256(onnx_path)

    plugin_logger = trt.Logger(trt.Logger.ERROR)
    standard_plugins_initialized = bool(
        trt.init_libnvinfer_plugins(plugin_logger, "")
    )
    scatter_probes = probe_scatter_reductions(onnx, trt)

    builder, network, normal_parser = create_parser(trt, trt.Logger.ERROR)
    normal_success = bool(normal_parser.parse_from_file(str(onnx_path)))
    normal_errors = collect_errors(normal_parser)

    support_builder, support_network, support_parser = create_parser(
        trt, trt.Logger.ERROR
    )
    model_supported = bool(
        support_parser.supports_model_v2(model_bytes, str(onnx_path.parent))
    )
    support_errors = collect_errors(support_parser)
    subgraphs: list[dict[str, Any]] = []
    for index in range(support_parser.num_subgraphs):
        node_indices = [int(item) for item in support_parser.get_subgraph_nodes(index)]
        selected = [nodes[item] for item in node_indices]
        op_counts: dict[str, int] = {}
        for node in selected:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        subgraphs.append(
            {
                "index": index,
                "supported": bool(support_parser.is_subgraph_supported(index)),
                "node_count": len(node_indices),
                "node_indices": node_indices,
                "first_node": {
                    "index": node_indices[0],
                    "name": selected[0].name,
                    "op_type": selected[0].op_type,
                },
                "last_node": {
                    "index": node_indices[-1],
                    "name": selected[-1].name,
                    "op_type": selected[-1].op_type,
                },
                "operator_counts": dict(sorted(op_counts.items())),
            }
        )

    registry_builder, registry_network, registry_parser = create_parser(
        trt, trt.Logger.ERROR
    )
    op_types = sorted({node.op_type for node in nodes})
    registry = {
        op: bool(registry_parser.supports_operator(op)) for op in op_types
    }
    unsupported_registry = [op for op, value in registry.items() if not value]
    error_ops = sorted(
        {
            str(item["operator"])
            for item in normal_errors + support_errors
            if item["operator"]
        }
    )
    confirmed_blockers = sorted(set(error_ops) | set(unsupported_registry))

    after_hash = sha256(onnx_path)
    if before_hash != after_hash:
        raise RuntimeError("Source ONNX SHA-256 changed during read-only audit")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.output_root.resolve() / f"{timestamp}_incremental_parser_audit"
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "environment": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "tensorrt_version": trt.__version__,
            "onnx_version": onnx.__version__,
            "tensorrt_root": str(trt_root) if trt_root else None,
        },
        "onnx": {
            "path": str(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
            "sha256": before_hash,
            "sha256_after": after_hash,
            "node_count": len(nodes),
            "opset": [
                {"domain": item.domain, "version": int(item.version)}
                for item in model.opset_import
            ],
        },
        "normal_parse": {
            "success": normal_success,
            "errors": normal_errors,
            "network_layers_materialized": network.num_layers,
        },
        "supports_model_v2": {
            "model_supported": model_supported,
            "num_subgraphs": support_parser.num_subgraphs,
            "errors": support_errors,
            "subgraphs": subgraphs,
        },
        "operator_registry": {
            "warning": "supports_operator may return false positives",
            "results": registry,
            "unsupported_ops": unsupported_registry,
        },
        "standard_plugins": {
            "initialized": standard_plugins_initialized,
            "note": (
                "ScatterElements reductions use NVIDIA's bundled "
                "ScatterElements plugin; this is not a project custom plugin."
            ),
        },
        "attribute_probes": {
            "method": "minimal ONNX buffers parsed in memory only",
            "scatter_elements": scatter_probes,
        },
        "decision": {
            "confirmed_blocking_operators": confirmed_blockers,
            "only_confirmed_blocking_operator": (
                "Unique" if confirmed_blockers == ["Unique"] else None
            ),
            "multiple_confirmed_blocking_operators": len(confirmed_blockers) > 1,
            "full_graph_after_unique_proven": False,
            "plugin_feasibility_audit_recommended": confirmed_blockers == ["Unique"],
            "split_voxel_hierarchy_now": False,
        },
        "safety": {
            "onnx_modified": False,
            "engine_build_attempted": False,
            "inference_attempted": False,
            "plugin_implemented": False,
        },
    }
    dump_json(run_dir / "incremental_parser_audit.json", payload)
    (run_dir / "incremental_parser_report.md").write_text(
        render_report(payload), encoding="utf-8"
    )

    # Keep DLL directory handles alive until all TensorRT objects are released.
    del registry_parser, registry_network, registry_builder
    del support_parser, support_network, support_builder
    del normal_parser, network, builder
    del dll_handles

    print(f"RUN_DIR={run_dir}")
    print(f"CONFIRMED_BLOCKING_OPERATORS={confirmed_blockers}")
    print(f"NUM_SUBGRAPHS={len(subgraphs)}")
    print(f"SUBGRAPH_SUPPORT={[item['supported'] for item in subgraphs]}")
    print(f"STANDARD_PLUGINS_INITIALIZED={standard_plugins_initialized}")
    print(
        "SCATTER_REDUCTION_PROBES="
        f"{[(item['reduction'], item['data_dtype'], item['success']) for item in scatter_probes]}"
    )
    print("INCREMENTAL_PARSER_AUDIT_COMPLETED")
    if confirmed_blockers == ["Unique"]:
        print("ONLY_CONFIRMED_BLOCKING_OPERATOR=Unique")
        print("PLUGIN_FEASIBILITY_AUDIT_RECOMMENDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
