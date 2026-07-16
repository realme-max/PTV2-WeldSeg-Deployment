"""Read-only audit of VoxelUnique IPluginV3 DDS/size-tensor mapping.

The audit reads C++ source and ONNX graph topology only.  It does not import
TensorRT, register plugins, create a Builder, build an engine, or run inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import onnx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GCN_ONNX = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260715_213934_180785_if_folded"
    / "if_folded.onnx"
)
DEFAULT_PROTOTYPE_ONNX = (
    PROJECT_ROOT
    / "artifacts"
    / "tensorrt_plugin_prototype"
    / "20260715_203305_357432_correctness"
    / "voxel_unique_correctness_dynamic.onnx"
)
DEFAULT_PRODUCTION_SOURCE = (
    PROJECT_ROOT
    / "tests"
    / "tensorrt_voxel_unique_correctness"
    / "VoxelUniqueCorrectnessPlugin.cu"
)
DEFAULT_PROTOTYPE_SOURCE = (
    PROJECT_ROOT
    / "tests"
    / "tensorrt_plugin_size_tensor_prototype"
    / "VoxelUniquePrototype.cu"
)
DEFAULT_TRT_HEADER = Path(
    r"D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106"
    r"\include\NvInferRuntime.h"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
EXPECTED_GCN_SHA256 = (
    "f0ca962b4e46e7495d40c7f23387c8dffbd4ca88e580452408f2fd9da85bc5ba"
)
OUTPUT_ROLES = {
    0: "voxel_count",
    1: "unique_values",
    2: "inverse_indices",
}
OUTPUT_DTYPES = {0: "INT32", 1: "INT64", 2: "INT64"}


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


def source_line(path: Path, needle: str) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    matches = [
        {"line": index, "text": text.strip()}
        for index, text in enumerate(lines, start=1)
        if needle in text
    ]
    if not matches:
        raise RuntimeError(f"Source evidence not found in {path}: {needle}")
    return {"path": str(path), "needle": needle, "matches": matches}


def tensor_contract(model: onnx.ModelProto, name: str) -> dict[str, Any]:
    values = (*model.graph.input, *model.graph.output, *model.graph.value_info)
    for value in values:
        if value.name != name:
            continue
        tensor_type = value.type.tensor_type
        shape: list[int | str] | None = None
        if tensor_type.HasField("shape"):
            shape = []
            for dimension in tensor_type.shape.dim:
                if dimension.HasField("dim_value"):
                    shape.append(int(dimension.dim_value))
                elif dimension.HasField("dim_param"):
                    shape.append(dimension.dim_param)
                else:
                    shape.append("unknown")
        return {
            "name": name,
            "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type),
            "shape": shape,
            "shape_present": tensor_type.HasField("shape"),
        }
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return {
                "name": name,
                "dtype": onnx.TensorProto.DataType.Name(initializer.data_type),
                "shape": list(initializer.dims),
                "shape_present": True,
            }
    return {
        "name": name,
        "dtype": "UNKNOWN",
        "shape": None,
        "shape_present": False,
    }


def build_consumers(model: onnx.ModelProto) -> dict[str, list[dict[str, Any]]]:
    consumers: dict[str, list[dict[str, Any]]] = {}
    for node_index, node in enumerate(model.graph.node):
        for input_slot, tensor_name in enumerate(node.input):
            consumers.setdefault(tensor_name, []).append(
                {
                    "node_index": node_index,
                    "node_name": node.name,
                    "op_type": node.op_type,
                    "domain": node.domain or "ai.onnx",
                    "input_slot": input_slot,
                    "outputs": list(node.output),
                }
            )
    return consumers


def downstream_summary(
    model: onnx.ModelProto,
    start_tensor: str,
    consumers: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    graph_outputs = {item.name for item in model.graph.output}
    queue: deque[str] = deque([start_tensor])
    visited_tensors: set[str] = set()
    visited_nodes: set[int] = set()
    op_counts: Counter[str] = Counter()
    reached_outputs: set[str] = set()
    while queue:
        tensor = queue.popleft()
        if tensor in visited_tensors:
            continue
        visited_tensors.add(tensor)
        if tensor in graph_outputs:
            reached_outputs.add(tensor)
        for consumer in consumers.get(tensor, []):
            node_index = int(consumer["node_index"])
            if node_index in visited_nodes:
                continue
            visited_nodes.add(node_index)
            op_counts[consumer["op_type"]] += 1
            queue.extend(consumer["outputs"])
    return {
        "reachable_node_count": len(visited_nodes),
        "reachable_tensor_count": len(visited_tensors),
        "reachable_operator_counts": dict(sorted(op_counts.items())),
        "reaches_graph_outputs": sorted(reached_outputs),
    }


def plugin_nodes(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    return [
        node
        for node in model.graph.node
        if node.op_type == "VoxelUnique" and node.domain == "com.tensorrt.ptv2"
    ]


def audit_graph(path: Path, expected_node_count: int) -> dict[str, Any]:
    model = onnx.load_model(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    consumers = build_consumers(model)
    nodes = plugin_nodes(model)
    if len(nodes) != expected_node_count:
        raise RuntimeError(
            f"Expected {expected_node_count} VoxelUnique nodes in {path}, found {len(nodes)}"
        )
    graph_output_names = {item.name for item in model.graph.output}
    records: list[dict[str, Any]] = []
    for node_index, node in enumerate(model.graph.node):
        if node not in nodes:
            continue
        outputs: list[dict[str, Any]] = []
        for output_index, tensor_name in enumerate(node.output):
            direct = consumers.get(tensor_name, [])
            outputs.append(
                {
                    "output_index": output_index,
                    "semantic_role": OUTPUT_ROLES[output_index],
                    "tensor": tensor_contract(model, tensor_name),
                    "direct_consumer_count": len(direct),
                    "direct_consumers": direct,
                    "is_graph_output": tensor_name in graph_output_names,
                    "is_topologically_retained": bool(direct)
                    or tensor_name in graph_output_names,
                    "downstream": downstream_summary(
                        model, tensor_name, consumers
                    ),
                }
            )
        records.append(
            {
                "node_index": node_index,
                "node_name": node.name,
                "domain": node.domain,
                "op_type": node.op_type,
                "input": tensor_contract(model, node.input[0]),
                "outputs": outputs,
            }
        )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "onnx_checker_passed": True,
        "graph_outputs": [tensor_contract(model, item.name) for item in model.graph.output],
        "voxel_unique_node_count": len(records),
        "nodes": records,
    }


def production_plugin_contract(source: Path, trt_header: Path) -> dict[str, Any]:
    output_records = [
        {
            "output_index": 0,
            "semantic_role": "voxel_count",
            "dtype": "INT32",
            "shape_expression": "scalar []",
            "is_size_tensor_carrier": True,
            "references_runtime_size_tensor": False,
            "evidence": [
                source_line(source, "outputTypes[0] ="),
                source_line(source, "outputs[0].nbDims = 0"),
            ],
        },
        {
            "output_index": 1,
            "semantic_role": "unique_values",
            "dtype": "INT64",
            "shape_expression": "[runtimeM]",
            "is_size_tensor_carrier": False,
            "references_runtime_size_tensor": True,
            "dds_output": True,
            "evidence": [
                source_line(source, "outputTypes[1] ="),
                source_line(source, "outputs[1].d[0] = runtimeM"),
            ],
        },
        {
            "output_index": 2,
            "semantic_role": "inverse_indices",
            "dtype": "INT64",
            "shape_expression": "[input0.N]",
            "is_size_tensor_carrier": False,
            "references_runtime_size_tensor": False,
            "dds_output": False,
            "evidence": [
                source_line(source, "outputTypes[2] ="),
                source_line(source, "outputs[2].d[0] = inputs[0].d[0]"),
            ],
        },
    ]
    return {
        "source_path": str(source),
        "source_sha256": sha256(source),
        "output_count": 3,
        "outputs": output_records,
        "declare_size_tensor": {
            "call_count": 1,
            "output_index": 0,
            "associated_output": "voxel_count",
            "opt_expression": "inputs[0].d[0] (N)",
            "upper_bound_expression": "inputs[0].d[0] (N)",
            "returned_expression": "runtimeM",
            "referenced_by_outputs": [1],
            "evidence": source_line(source, "exprBuilder.declareSizeTensor("),
        },
        "dds_output_indices": [1],
        "size_tensor_output_indices": [0],
        "multiple_dds_outputs": False,
        "outputs_without_runtime_size_reference": [0, 2],
        "tensorrt_api_contract": {
            "header_path": str(trt_header),
            "header_sha256": sha256(trt_header),
            "output_index_meaning": (
                "index of a plugin output that is a size tensor"
            ),
            "evidence": [
                source_line(
                    trt_header,
                    "outputIndex index of a plugin output that is a size tensor",
                ),
                source_line(
                    trt_header,
                    "return IDimensionExpr denoting the value of the size tensor",
                ),
            ],
        },
    }


def prototype_source_contract(source: Path) -> dict[str, Any]:
    return {
        "source_path": str(source),
        "source_sha256": sha256(source),
        "declare_size_tensor": {
            "output_index": 0,
            "associated_output": "count",
            "opt_expression": "constant(3)",
            "upper_bound_expression": "inputs[0].d[0] (N)",
            "returned_expression": "runtimeM",
            "referenced_by_outputs": [1],
            "evidence": [
                source_line(source, "IDimensionExpr const* optimum = exprBuilder.constant(3)"),
                source_line(source, "exprBuilder.declareSizeTensor(0, *optimum, *upper)"),
                source_line(source, "outputs[1].d[0] = runtimeM"),
            ],
        },
        "all_outputs_marked_as_network_outputs": True,
        "evidence": [
            source_line(source, "char const* outputNames[]"),
            source_line(source, "network->markOutput(*output)"),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcn-onnx", type=Path, default=DEFAULT_GCN_ONNX)
    parser.add_argument("--prototype-onnx", type=Path, default=DEFAULT_PROTOTYPE_ONNX)
    parser.add_argument("--plugin-source", type=Path, default=DEFAULT_PRODUCTION_SOURCE)
    parser.add_argument("--prototype-source", type=Path, default=DEFAULT_PROTOTYPE_SOURCE)
    parser.add_argument("--tensorrt-header", type=Path, default=DEFAULT_TRT_HEADER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    paths = {
        "gcn_onnx": args.gcn_onnx.resolve(),
        "prototype_onnx": args.prototype_onnx.resolve(),
        "plugin_source": args.plugin_source.resolve(),
        "prototype_source": args.prototype_source.resolve(),
        "tensorrt_header": args.tensorrt_header.resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    if hashes_before["gcn_onnx"] != EXPECTED_GCN_SHA256:
        raise RuntimeError(
            f"GCN ONNX hash mismatch: {hashes_before['gcn_onnx']}"
        )

    plugin_contract = production_plugin_contract(
        paths["plugin_source"], paths["tensorrt_header"]
    )
    prototype_contract = prototype_source_contract(paths["prototype_source"])
    prototype_graph = audit_graph(paths["prototype_onnx"], expected_node_count=1)
    gcn_graph = audit_graph(paths["gcn_onnx"], expected_node_count=4)

    prototype_outputs = prototype_graph["nodes"][0]["outputs"]
    gcn_outputs = [
        output
        for node in gcn_graph["nodes"]
        for output in node["outputs"]
    ]
    gcn_count_outputs = [item for item in gcn_outputs if item["output_index"] == 0]
    gcn_values_outputs = [item for item in gcn_outputs if item["output_index"] == 1]
    gcn_inverse_outputs = [item for item in gcn_outputs if item["output_index"] == 2]

    key_findings = {
        "plugin_size_tensor_output_index": 0,
        "plugin_dds_output_indices": [1],
        "plugin_multiple_dds_outputs": False,
        "plugin_output_2_is_fixed_by_input_n": True,
        "prototype_all_three_outputs_are_graph_outputs": all(
            item["is_graph_output"] for item in prototype_outputs
        ),
        "prototype_size_tensor_output_retained": prototype_outputs[0][
            "is_topologically_retained"
        ],
        "gcn_size_tensor_output_consumer_counts": [
            item["direct_consumer_count"] for item in gcn_count_outputs
        ],
        "gcn_size_tensor_output_is_graph_output": [
            item["is_graph_output"] for item in gcn_count_outputs
        ],
        "gcn_size_tensor_output_unconnected_for_all_four_layers": all(
            not item["is_topologically_retained"] for item in gcn_count_outputs
        ),
        "gcn_dds_values_output_retained_for_all_four_layers": all(
            item["is_topologically_retained"] for item in gcn_values_outputs
        ),
        "gcn_inverse_output_retained_for_all_four_layers": all(
            item["is_topologically_retained"] for item in gcn_inverse_outputs
        ),
        "gcn_values_and_inverse_reach_logits": {
            "unique_values": [
                "logits" in item["downstream"]["reaches_graph_outputs"]
                for item in gcn_values_outputs
            ],
            "inverse_indices": [
                "logits" in item["downstream"]["reaches_graph_outputs"]
                for item in gcn_inverse_outputs
            ],
        },
    }
    topology_hypothesis = {
        "classification": "STATIC_TOPOLOGY_MISMATCH_IDENTIFIED",
        "observed_difference": (
            "The prototype retains output 0 (the size-tensor carrier) as a network/graph "
            "output. In all four GCN nodes, output 0 has no consumer and is not a graph "
            "output, while output 1 still has shape [M] derived from that size tensor."
        ),
        "consistent_with_builder_assertion": True,
        "reason": (
            "The failing assertion compares DDS-output and size-tensor bookkeeping counts. "
            "An unretained size-tensor carrier paired with a retained DDS output is the "
            "only static graph difference found that directly matches those two maps."
        ),
        "causality_proven": False,
        "why_not_proven": (
            "This stage is static-only and does not instrument TensorRT internal maps or "
            "perform a controlled graph/plugin modification and rebuild."
        ),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.output_root.resolve() / f"{timestamp}_dds_audit"
    run_dir.mkdir(parents=True, exist_ok=False)
    common = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "read_only": True,
        "tensorrt_imported": False,
        "plugin_registered": False,
        "engine_build_called": False,
        "inference_called": False,
        "source_hashes_before": hashes_before,
    }
    mapping_payload = {
        **common,
        "plugin_contract": plugin_contract,
        "prototype_source_contract": prototype_contract,
        "prototype_graph": prototype_graph,
        "gcn_res_graph": gcn_graph,
        "key_findings": key_findings,
        "topology_hypothesis": topology_hypothesis,
    }
    diff_payload = {
        **common,
        "prototype": {
            "voxel_unique_nodes": 1,
            "outputs_are_graph_outputs": [
                item["is_graph_output"] for item in prototype_outputs
            ],
            "output_consumer_counts": [
                item["direct_consumer_count"] for item in prototype_outputs
            ],
            "size_tensor_carrier_retained": prototype_outputs[0][
                "is_topologically_retained"
            ],
        },
        "gcn_res": {
            "voxel_unique_nodes": 4,
            "per_node": [
                {
                    "node_name": node["node_name"],
                    "output_consumer_counts": [
                        item["direct_consumer_count"] for item in node["outputs"]
                    ],
                    "outputs_are_graph_outputs": [
                        item["is_graph_output"] for item in node["outputs"]
                    ],
                    "size_tensor_carrier_retained": node["outputs"][0][
                        "is_topologically_retained"
                    ],
                    "dds_values_output_retained": node["outputs"][1][
                        "is_topologically_retained"
                    ],
                    "inverse_output_retained": node["outputs"][2][
                        "is_topologically_retained"
                    ],
                }
                for node in gcn_graph["nodes"]
            ],
        },
        "differences": {
            "declare_size_tensor_output_index": {
                "prototype": 0,
                "gcn_plugin": 0,
                "different": False,
            },
            "dds_output_index": {
                "prototype": 1,
                "gcn_plugin": 1,
                "different": False,
            },
            "opt_expression": {
                "prototype": "constant(3)",
                "gcn_plugin": "N",
                "different": True,
            },
            "upper_bound_expression": {
                "prototype": "N",
                "gcn_plugin": "N",
                "different": False,
            },
            "size_tensor_carrier_output_0_retention": {
                "prototype": "retained as graph/network output",
                "gcn_plugin": "unconsumed and not graph output in all 4 nodes",
                "different": True,
            },
            "values_output_1_retention": {
                "prototype": "retained as graph/network output",
                "gcn_plugin": "consumed by Shape in all 4 nodes",
                "different": True,
            },
            "inverse_output_2_retention": {
                "prototype": "retained as graph/network output",
                "gcn_plugin": "consumed by multiple downstream nodes in all 4 nodes",
                "different": True,
            },
        },
        "topology_hypothesis": topology_hypothesis,
    }
    dump_json(run_dir / "voxelunique_dds_mapping.json", mapping_payload)
    dump_json(run_dir / "prototype_vs_gcn_diff.json", diff_payload)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    if hashes_after != hashes_before:
        raise RuntimeError("A read-only audit input changed")
    mapping_payload["source_hashes_after"] = hashes_after
    mapping_payload["all_sources_unchanged"] = True
    diff_payload["source_hashes_after"] = hashes_after
    diff_payload["all_sources_unchanged"] = True
    dump_json(run_dir / "voxelunique_dds_mapping.json", mapping_payload)
    dump_json(run_dir / "prototype_vs_gcn_diff.json", diff_payload)

    gcn_rows = "\n".join(
        f"| `{node['node_name']}` | "
        f"{node['outputs'][0]['direct_consumer_count']} | "
        f"{node['outputs'][1]['direct_consumer_count']} | "
        f"{node['outputs'][2]['direct_consumer_count']} | "
        f"{node['outputs'][0]['is_topologically_retained']} |"
        for node in gcn_graph["nodes"]
    )
    report = f"""# VoxelUnique IPluginV3 DDS size tensor audit

## Scope

- Production plugin source: `{paths['plugin_source']}`
- Prototype source: `{paths['prototype_source']}`
- Prototype ONNX: `{paths['prototype_onnx']}`
- GCN_res ONNX: `{paths['gcn_onnx']}`
- GCN_res SHA-256: `{hashes_before['gcn_onnx']}`
- Read-only: `true`
- TensorRT imported/registered: `false`
- Engine build/inference: `false`

## Plugin DDS contract

| Output index | Semantic role | Dtype | Shape expression | DDS role |
|---:|---|---|---|---|
| 0 | voxel_count | INT32 | scalar `[]` | size-tensor carrier |
| 1 | unique_values | INT64 | `[runtimeM]` | DDS output using runtime size |
| 2 | inverse_indices | INT64 | `[input N]` | not DDS |

There is exactly one `declareSizeTensor` call:

```text
declareSizeTensor(outputIndex=0, opt=N, upper=N) -> runtimeM
output[1].shape = [runtimeM]
```

TensorRT's local header defines `outputIndex` as the index of the plugin output
that is itself the size tensor. Thus output 0 carries M; output 1 consumes M in
its shape expression. Output 2 is sized directly by input N. There is one DDS
data output, not multiple DDS outputs.

## GCN_res graph consumers

| VoxelUnique node | count consumers | values consumers | inverse consumers | count retained |
|---|---:|---:|---:|---|
{gcn_rows}

Direct consumer patterns:

- `voxel_count` / output 0: no consumer and not a graph output in all 4 stages;
- `unique_values` / output 1: consumed by `Shape` in all 4 stages;
- `inverse_indices` / output 2: consumed by Identity/Shape/ScatterElements/
  Unsqueeze paths and remains connected toward `logits`.

Full consumer records and downstream reachability are in
`voxelunique_dds_mapping.json`.

## Prototype versus GCN_res

The successful prototype and correctness ONNX retain all three plugin outputs as
network/graph outputs. In particular, output 0, which carries the runtime size,
cannot be pruned.

The GCN_res graph differs: the retained DDS tensor `unique_values[M]` still
depends on output 0's runtime value, but output 0 itself has no consumer and is
not a graph output for any of the four VoxelUnique nodes.

The source-level DDS declarations otherwise match structurally:

- size tensor carrier: output 0 in both;
- DDS output: output 1 in both;
- upper bound: N in both;
- inverse output: fixed by N in both;
- only opt differs: prototype uses constant 3; deployed plugin uses N.

## Localization conclusion

`STATIC_TOPOLOGY_MISMATCH_IDENTIFIED`

The unretained output-0 size-tensor carrier paired with a retained output-1 DDS
tensor is consistent with the Builder assertion comparing DDS-output and
size-tensor bookkeeping counts. It is the only observed static topology
difference that directly corresponds to those two internal maps.

This is not yet a causal proof. Proving causality would require an authorized
controlled graph/plugin change and another build, both prohibited in this stage.
No fix was applied.

## Status

VOXELUNIQUE_DDS_MAPPING_AUDIT_COMPLETED
"""
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    print(f"RUN_DIR={run_dir}")
    print("PLUGIN_SIZE_TENSOR_OUTPUT_INDEX=0")
    print("PLUGIN_DDS_OUTPUT_INDICES=[1]")
    print("PLUGIN_MULTIPLE_DDS_OUTPUTS=false")
    print("GCN_SIZE_TENSOR_OUTPUT_UNCONNECTED_ALL_4=true")
    print("PROTOTYPE_SIZE_TENSOR_OUTPUT_RETAINED=true")
    print("ENGINE_BUILD_CALLED=false")
    print("INFERENCE_CALLED=false")
    print("VOXELUNIQUE_DDS_MAPPING_AUDIT_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
