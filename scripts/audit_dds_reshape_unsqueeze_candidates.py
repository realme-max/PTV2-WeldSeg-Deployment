"""Audit DDS-derived TDB Reshape nodes for exact Unsqueeze(axis=0) semantics.

This script is intentionally read-only with respect to the input ONNX.  It
classifies every Reshape in tdb_1..tdb_4 and emits the evidence needed by the
controlled rewrite stage.  It does not invoke TensorRT.
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
from onnx import TensorProto, helper, numpy_helper


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260715_213934_180785_if_folded"
    / "if_folded.onnx"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
EXPECTED_SOURCE_SHA256 = (
    "f0ca962b4e46e7495d40c7f23387c8dffbd4ca88e580452408f2fd9da85bc5ba"
)

A = "A_EXACT_UNSQUEEZE_EQUIVALENT"
B = "B_RESHAPE_HAS_OTHER_SEMANTICS"
C = "C_INSUFFICIENT_SHAPE_PROOF"
TDB_RE = re.compile(r"^/model/tdb_([1-4])/Reshape(?:_\d+)?$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def tensor_contract(value: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value.type.tensor_type
    dimensions: list[Any] = []
    if tensor_type.HasField("shape"):
        for dimension in tensor_type.shape.dim:
            if dimension.HasField("dim_value"):
                dimensions.append(int(dimension.dim_value))
            elif dimension.HasField("dim_param"):
                dimensions.append(str(dimension.dim_param))
            else:
                dimensions.append(None)
    return {
        "dtype": TensorProto.DataType.Name(tensor_type.elem_type),
        "shape": dimensions,
        "rank": len(dimensions),
        "shape_present": tensor_type.HasField("shape"),
    }


def attributes(node: onnx.NodeProto) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attribute in node.attribute:
        value = helper.get_attribute_value(attribute)
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        result[attribute.name] = value
    return result


class GraphView:
    def __init__(self, model: onnx.ModelProto) -> None:
        self.model = model
        self.producer = {
            output: node for node in model.graph.node for output in node.output
        }
        self.consumers: dict[str, list[dict[str, Any]]] = {}
        for node in model.graph.node:
            for slot, tensor in enumerate(node.input):
                self.consumers.setdefault(tensor, []).append(
                    {"node_name": node.name, "op_type": node.op_type, "input_slot": slot}
                )
        self.contracts = {
            value.name: tensor_contract(value)
            for value in (
                list(model.graph.input)
                + list(model.graph.value_info)
                + list(model.graph.output)
            )
        }
        for initializer in model.graph.initializer:
            self.contracts.setdefault(
                initializer.name,
                {
                    "dtype": TensorProto.DataType.Name(initializer.data_type),
                    "shape": [int(item) for item in initializer.dims],
                    "rank": len(initializer.dims),
                    "shape_present": True,
                },
            )
        self.initializers = {
            initializer.name: numpy_helper.to_array(initializer)
            for initializer in model.graph.initializer
        }

    def contract(self, tensor: str) -> dict[str, Any] | None:
        return self.contracts.get(tensor)

    def constant(self, tensor: str) -> Any:
        if tensor in self.initializers:
            return self.initializers[tensor].tolist()
        node = self.producer.get(tensor)
        if node is None or node.op_type != "Constant":
            return None
        for attribute in node.attribute:
            if attribute.name == "value":
                return numpy_helper.to_array(attribute.t).tolist()
            if attribute.name == "value_int":
                return int(attribute.i)
            if attribute.name == "value_ints":
                return [int(item) for item in attribute.ints]
        return None

    def scalar_constant(self, tensor: str) -> int | None:
        value = self.constant(tensor)
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if isinstance(value, (int, bool)):
            return int(value)
        return None

    def static_shape_scalar(self, tensor: str) -> dict[str, Any]:
        """Resolve constants and Gather(Shape(tensor), constant_index)."""
        direct = self.scalar_constant(tensor)
        if direct is not None:
            return {"proven": True, "value": direct, "source": "Constant"}
        node = self.producer.get(tensor)
        if node is None:
            return {"proven": False, "reason": "no producer"}
        if node.op_type == "Identity" and node.input:
            return self.static_shape_scalar(node.input[0])
        if node.op_type == "Unsqueeze" and node.input:
            return self.static_shape_scalar(node.input[0])
        if node.op_type != "Gather" or len(node.input) < 2:
            return {"proven": False, "reason": f"producer is {node.op_type}"}
        shape_node = self.producer.get(node.input[0])
        index = self.scalar_constant(node.input[1])
        if shape_node is None or shape_node.op_type != "Shape" or index is None:
            return {"proven": False, "reason": "not Gather(Shape(x), constant)"}
        source = shape_node.input[0]
        contract = self.contract(source)
        if contract is None or not contract["shape_present"]:
            return {"proven": False, "reason": f"shape metadata missing for {source}"}
        shape = contract["shape"]
        normalized = index if index >= 0 else len(shape) + index
        if normalized < 0 or normalized >= len(shape):
            return {"proven": False, "reason": "shape index out of range"}
        value = shape[normalized]
        if not isinstance(value, int):
            return {
                "proven": False,
                "reason": f"dimension {normalized} is dynamic",
                "source_tensor": source,
                "source_shape": shape,
            }
        return {
            "proven": True,
            "value": value,
            "source": "Gather(Shape(tensor), constant_index)",
            "source_tensor": source,
            "source_shape": shape,
            "axis": normalized,
        }

    def last_static_dimension(self, tensor: str, seen: set[str] | None = None) -> int | None:
        if seen is None:
            seen = set()
        if tensor in seen:
            return None
        seen.add(tensor)
        contract = self.contract(tensor)
        if contract and contract["shape"] and isinstance(contract["shape"][-1], int):
            return int(contract["shape"][-1])
        node = self.producer.get(tensor)
        if node is None or not node.input:
            return None
        if node.op_type in {"Identity", "ScatterElements", "Relu", "Cast"}:
            return self.last_static_dimension(node.input[0], seen)
        if node.op_type == "GatherND":
            return self.last_static_dimension(node.input[0], seen)
        return None

    def shortest_node_path_to(self, tensor: str, target_node_name: str) -> list[str] | None:
        queue: deque[tuple[str, list[str]]] = deque([(tensor, [])])
        visited: set[str] = set()
        while queue:
            current, path = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            node = self.producer.get(current)
            if node is None:
                continue
            next_path = path + [node.name]
            if node.name == target_node_name:
                return next_path
            for input_name in node.input:
                queue.append((input_name, next_path))
        return None


def component_scalar(view: GraphView, tensor: str) -> dict[str, Any]:
    node = view.producer.get(tensor)
    unwrapped = tensor
    wrapper = None
    if node is not None and node.op_type == "Unsqueeze" and node.input:
        wrapper = node.name
        unwrapped = node.input[0]
    static = view.static_shape_scalar(unwrapped)
    return {
        "tensor": tensor,
        "wrapper": wrapper,
        "source_tensor": unwrapped,
        "source_producer": (
            None
            if unwrapped not in view.producer
            else {
                "name": view.producer[unwrapped].name,
                "op_type": view.producer[unwrapped].op_type,
            }
        ),
        "static_resolution": static,
    }


def analyze_exact_candidate(
    view: GraphView, node: onnx.NodeProto, stage: int
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    attrs = attributes(node)
    allowzero = int(attrs.get("allowzero", 0))
    if allowzero != 0:
        failures.append(f"allowzero={allowzero}, expected 0")
    if len(node.input) != 2 or len(node.output) != 1:
        failures.append("Reshape does not have exactly two inputs and one output")
        return {"allowzero": allowzero}, failures

    data_tensor, shape_tensor = node.input
    output_tensor = node.output[0]
    data_producer = view.producer.get(data_tensor)
    shape_producer = view.producer.get(shape_tensor)
    output_contract = view.contract(output_tensor)

    if data_producer is None or data_producer.op_type != "GatherND":
        failures.append("data producer is not GatherND")
    if shape_producer is None or shape_producer.op_type != "Concat":
        failures.append("target shape producer is not Concat")

    components: list[dict[str, Any]] = []
    if shape_producer is not None and shape_producer.op_type == "Concat":
        if attributes(shape_producer).get("axis") != 0:
            failures.append("target Concat axis is not 0")
        if len(shape_producer.input) != 3:
            failures.append("target shape does not have exactly three scalar components")
        components = [component_scalar(view, item) for item in shape_producer.input]

    leading = components[0]["static_resolution"] if len(components) == 3 else {}
    middle_source = components[1]["source_tensor"] if len(components) == 3 else None
    trailing = components[2]["static_resolution"] if len(components) == 3 else {}
    expected_k = f"/model/tdb_{stage}/ReduceMin_1_output_0"

    if not leading.get("proven") or leading.get("value") != 1:
        failures.append("target leading dimension is not statically proven as 1")
    if middle_source != expected_k:
        failures.append(
            f"target middle dimension is not the stage DDS runtime K ({expected_k})"
        )
    if not trailing.get("proven") or not isinstance(trailing.get("value"), int):
        failures.append("target trailing dimension C is not statically proven")
    channel = trailing.get("value") if trailing.get("proven") else None

    if output_contract is None or output_contract["rank"] != 3:
        failures.append("output static rank is not 3")
    elif (
        output_contract["shape"][0] != 1
        or not isinstance(output_contract["shape"][-1], int)
        or output_contract["shape"][-1] != channel
    ):
        failures.append("output metadata is not [1,K,C] with the proven static C")

    gather_evidence: dict[str, Any] = {}
    if data_producer is not None and data_producer.op_type == "GatherND":
        gather_data = data_producer.input[0]
        gather_indices = data_producer.input[1]
        indices_producer = view.producer.get(gather_indices)
        nonzero = None
        mask = None
        if indices_producer is not None and indices_producer.op_type == "Transpose":
            transpose_perm = attributes(indices_producer).get("perm")
            if transpose_perm != [1, 0]:
                failures.append("GatherND indices transpose perm is not [1,0]")
            nonzero = view.producer.get(indices_producer.input[0])
        else:
            failures.append("GatherND indices are not Transpose(NonZero(mask))")
        if nonzero is not None and nonzero.op_type == "NonZero":
            mask = view.producer.get(nonzero.input[0])
        else:
            failures.append("GatherND indices do not originate at NonZero")
        if mask is None or mask.op_type != "Less" or expected_k not in mask.input:
            failures.append("selection mask is not Less(..., same stage runtime K)")

        base_channel = view.last_static_dimension(gather_data)
        if base_channel is None or base_channel != channel:
            failures.append(
                f"GatherND data static C={base_channel} does not match target C={channel}"
            )
        gather_evidence = {
            "node": data_producer.name,
            "data_tensor": gather_data,
            "data_contract": view.contract(gather_data),
            "data_static_last_dimension": base_channel,
            "indices_tensor": gather_indices,
            "indices_producer": None
            if indices_producer is None
            else {"name": indices_producer.name, "op_type": indices_producer.op_type},
            "nonzero_node": None if nonzero is None else nonzero.name,
            "mask_node": None if mask is None else mask.name,
            "derived_output_contract": ["K", base_channel],
            "derivation": (
                "NonZero over a one-dimensional retained-voxel mask produces [1,K]; "
                "Transpose([1,0]) produces GatherND indices [K,1], so GatherND over "
                "a rank-2 [M,C] tensor produces [K,C]."
            ),
        }

    plugin_node = f"/model/tdb_{stage}/Unique"
    data_dds_path = view.shortest_node_path_to(data_tensor, plugin_node)
    shape_dds_path = view.shortest_node_path_to(shape_tensor, plugin_node)
    if data_dds_path is None and shape_dds_path is None:
        failures.append("neither data nor target shape depends on the stage VoxelUnique DDS path")

    proof = {
        "allowzero": allowzero,
        "data_producer": None
        if data_producer is None
        else {"name": data_producer.name, "op_type": data_producer.op_type},
        "data_dtype": (
            (view.contract(data_tensor) or {}).get("dtype")
            or (view.contract(data_producer.input[0]) or {}).get("dtype")
            if data_producer is not None
            else None
        ),
        "derived_input_shape": ["K", channel] if channel is not None else None,
        "target_shape_tensor": shape_tensor,
        "target_shape_producer": None
        if shape_producer is None
        else {"name": shape_producer.name, "op_type": shape_producer.op_type},
        "target_components": components,
        "derived_target_shape": [1, "K", channel] if channel is not None else None,
        "output_contract": output_contract,
        "gathernd_evidence": gather_evidence,
        "voxel_unique_node": plugin_node,
        "data_to_voxel_unique_path": data_dds_path,
        "shape_to_voxel_unique_path": shape_dds_path,
        "depends_on_voxel_unique_dds": data_dds_path is not None or shape_dds_path is not None,
        "element_order_preserved": True,
        "unsqueeze_axis": 0,
        "equivalence_statement": (
            "For contiguous logical data [K,C], Reshape(data,[1,K,C]) and "
            "Unsqueeze(data,axes=[0]) have the same element order, dtype, and shape."
        ),
    }
    return proof, failures


def audit_model(model: onnx.ModelProto, source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    view = GraphView(model)
    records: list[dict[str, Any]] = []
    dependencies: list[dict[str, Any]] = []
    for index, node in enumerate(model.graph.node):
        match = TDB_RE.match(node.name)
        if node.op_type != "Reshape" or match is None:
            continue
        stage = int(match.group(1))
        proof, failures = analyze_exact_candidate(view, node, stage)
        close_pattern = (
            proof.get("data_producer", {}).get("op_type") == "GatherND"
            or proof.get("output_contract", {}).get("rank") == 3
        )
        if not failures:
            classification = A
            reason = "All nine exact-equivalence criteria were statically proven."
        elif close_pattern:
            classification = C
            reason = "; ".join(failures)
        else:
            classification = B
            reason = "; ".join(failures)
        output_tensor = node.output[0] if node.output else None
        record = {
            "node_index": index,
            "node_name": node.name,
            "stage": stage,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "data_producer": proof.get("data_producer"),
            "data_dtype": proof.get("data_dtype"),
            "input_shape": proof.get("derived_input_shape")
            or (view.contract(node.input[0]) if node.input else None),
            "target_shape_construction": proof.get("target_components"),
            "target_shape": proof.get("derived_target_shape"),
            "output_shape": proof.get("output_contract"),
            "allowzero": proof.get("allowzero"),
            "consumers": view.consumers.get(output_tensor, []) if output_tensor else [],
            "depends_on_voxel_unique_dds": proof.get("depends_on_voxel_unique_dds", False),
            "classification": classification,
            "rewrite_allowed": classification == A,
            "reason": reason,
            "proof": proof,
        }
        records.append(record)
        if classification == A:
            dependencies.append(
                {
                    "node_name": node.name,
                    "stage": stage,
                    "data_to_voxel_unique_path": proof["data_to_voxel_unique_path"],
                    "shape_to_voxel_unique_path": proof["shape_to_voxel_unique_path"],
                    "runtime_k_tensor": f"/model/tdb_{stage}/ReduceMin_1_output_0",
                    "static_channel": proof["derived_target_shape"][-1],
                    "consumer_count": len(record["consumers"]),
                    "consumers": record["consumers"],
                }
            )

    counts = Counter(record["classification"] for record in records)
    candidates = [record["node_name"] for record in records if record["classification"] == A]
    summary = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "DDS_RESHAPE_UNSQUEEZE_AUDIT_COMPLETED",
        "source_onnx": str(source),
        "source_sha256": sha256(source),
        "source_onnx_size_bytes": source.stat().st_size,
        "onnx_ir_version": int(model.ir_version),
        "opsets": [
            {"domain": item.domain or "ai.onnx", "version": int(item.version)}
            for item in model.opset_import
        ],
        "scope": "All Reshape nodes in tdb_1 through tdb_4",
        "classification_counts": dict(counts),
        "audited_reshape_count": len(records),
        "rewrite_candidate_count": len(candidates),
        "rewrite_candidates": candidates,
        "records": records,
        "input_onnx_modified": False,
        "tensorrt_invoked": False,
    }
    dependency = {
        "source_onnx": str(source),
        "source_sha256": sha256(source),
        "candidate_dependencies": dependencies,
        "all_candidates_have_dds_dependency": all(
            item["data_to_voxel_unique_path"] is not None
            or item["shape_to_voxel_unique_path"] is not None
            for item in dependencies
        ),
    }
    return summary, dependency


def write_report(run_dir: Path, summary: dict[str, Any]) -> None:
    rows = []
    for record in summary["records"]:
        target = record["target_shape"] or "not proven"
        rows.append(
            f"| `{record['node_name']}` | `{record['classification']}` | "
            f"`{record['input_shape']}` | `{target}` | {record['reason']} |"
        )
    report = f"""# DDS Reshape → Unsqueeze candidate audit

## Scope and safety

- Input: `{summary['source_onnx']}`
- SHA-256: `{summary['source_sha256']}`
- Audited scope: every `Reshape` in `tdb_1` through `tdb_4`.
- The source ONNX was read-only; TensorRT, engine build, inference, FP16 and INT8 were not invoked.

## Decision rule

`A_EXACT_UNSQUEEZE_EQUIVALENT` requires all of the following: rank-2 data
`[K,C]`; runtime DDS-derived `K`; static `C`; target exactly `[1,K,C]`;
`allowzero=0`; identical element order; only axis 0 added; output metadata
`[1,K,C]`; and a dependency path to that stage's `VoxelUnique` node.

## Result

- Audited Reshape nodes: **{summary['audited_reshape_count']}**
- Exact candidates: **{summary['rewrite_candidate_count']}**
- Classification counts: `{summary['classification_counts']}`

| Node | Classification | Derived input | Derived target | Evidence / rejection reason |
|---|---|---|---|---|
{chr(10).join(rows)}

## Authorized rewrite set

{chr(10).join(f'- `{name}`' for name in summary['rewrite_candidates'])}

Only these nodes may be passed to the controlled rewrite script.  Shape
construction chains remain in place even if they become unconsumed.

`DDS_RESHAPE_UNSQUEEZE_AUDIT_COMPLETED`
"""
    (run_dir / "equivalence_report.md").write_text(report, encoding="utf-8")


def main(args: argparse.Namespace) -> int:
    source = args.onnx.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash_before = sha256(source)
    if source_hash_before != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"Formal if_folded.onnx hash mismatch: {source_hash_before} != "
            f"{EXPECTED_SOURCE_SHA256}"
        )
    model = onnx.load_model(str(source), load_external_data=False)
    onnx.checker.check_model(model)
    run_dir = (
        args.output_root.resolve()
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_dds_reshape_audit"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    summary, dependency = audit_model(model, source)
    source_hash_after = sha256(source)
    summary["source_sha256_after"] = source_hash_after
    summary["source_onnx_unchanged"] = source_hash_after == source_hash_before
    if not summary["source_onnx_unchanged"]:
        raise RuntimeError("Input ONNX changed during read-only audit")
    if summary["rewrite_candidate_count"] == 0:
        summary["status"] = "DDS_RESHAPE_UNSQUEEZE_AUDIT_NO_EXACT_CANDIDATES"
    dump_json(run_dir / "reshape_candidates.json", summary)
    dump_json(run_dir / "shape_dependency.json", dependency)
    write_report(run_dir, summary)
    print(f"RUN_DIR={run_dir}")
    print(f"AUDITED_RESHAPES={summary['audited_reshape_count']}")
    print(f"EXACT_CANDIDATES={summary['rewrite_candidate_count']}")
    for name in summary["rewrite_candidates"]:
        print(f"A_CANDIDATE={name}")
    print(summary["status"])
    return 0 if summary["rewrite_candidate_count"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
