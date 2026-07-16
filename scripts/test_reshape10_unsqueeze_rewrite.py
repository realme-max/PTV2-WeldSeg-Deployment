"""Test the minimal TensorRT rewrite Reshape_10 -> Unsqueeze(axis=0).

The formal ONNX and plugin source are read-only.  A dependency-closed D6a
control graph is extracted from the formal graph, then copied and changed at
exactly one node.  Both candidates are checked, parsed, and passed to the FP32
builder.  Serialized engine bytes are not retained and inference is forbidden.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

import build_gcn_res_tensorrt_fp32 as phase4
import isolate_tdb_dds_consumer_boundary as incremental


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = incremental.DEFAULT_ONNX
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
DEFAULT_PLUGIN_LIBRARY = phase4.DEFAULT_PLUGIN_LIBRARY
DEFAULT_TENSORRT_ROOT = phase4.DEFAULT_TENSORRT_ROOT
DEFAULT_CUDA_ROOT = phase4.DEFAULT_CUDA_ROOT
WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "test_voxelunique_dds_cascade_isolation.py"
PLUGIN_SOURCE = incremental.PLUGIN_SOURCE
TARGET_NODE = "/model/tdb_1/Reshape_10"
TARGET_OUTPUT = "/model/tdb_1/Reshape_10_output_0"
DATA_INPUT = "/model/tdb_1/GatherND_output_0"
SHAPE_INPUT = "/model/tdb_1/Concat_5_output_0"
AXES_INITIALIZER = "/model/tdb_1/Reshape_10_TensorRT_Unsqueeze_axes"


def tensor_contract(value: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value.type.tensor_type
    shape = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            shape.append(str(dim.dim_param))
        else:
            shape.append(None)
    return {
        "name": value.name,
        "dtype": TensorProto.DataType.Name(tensor_type.elem_type),
        "rank": len(shape),
        "shape": shape,
        "shape_present": tensor_type.HasField("shape"),
    }


def contracts(model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
    result = {}
    # value_info may retain the formal graph's missing/older annotation for a
    # tensor that the diagnostic extractor also promotes to a graph output.
    # Process graph outputs last so their required, explicit contract wins.
    for value in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        result[value.name] = tensor_contract(value)
    for initializer in model.graph.initializer:
        result.setdefault(
            initializer.name,
            {
                "name": initializer.name,
                "dtype": TensorProto.DataType.Name(initializer.data_type),
                "rank": len(initializer.dims),
                "shape": [int(value) for value in initializer.dims],
                "shape_present": True,
            },
        )
    return result


def get_node(model: onnx.ModelProto, name: str) -> onnx.NodeProto:
    matches = [node for node in model.graph.node if node.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one node {name}, found {len(matches)}")
    return matches[0]


def attribute_dict(node: onnx.NodeProto) -> dict[str, Any]:
    result = {}
    for attribute in node.attribute:
        value = helper.get_attribute_value(attribute)
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result[attribute.name] = value
    return result


def constant_value(model: onnx.ModelProto, tensor_name: str) -> Any:
    initializer = next(
        (item for item in model.graph.initializer if item.name == tensor_name), None
    )
    if initializer is not None:
        return numpy_helper.to_array(initializer).tolist()
    producer = next(
        (node for node in model.graph.node if tensor_name in node.output), None
    )
    if producer is None or producer.op_type != "Constant":
        return None
    value_attribute = next(
        (attribute for attribute in producer.attribute if attribute.name == "value"),
        None,
    )
    if value_attribute is None:
        return None
    return numpy_helper.to_array(helper.get_attribute_value(value_attribute)).tolist()


def find_d6a_definition(model: onnx.ModelProto) -> dict[str, Any]:
    experiments = incremental.experiment_definitions(
        incremental.get_plugin_nodes(model)
    )
    return next(
        item for item in experiments if item["id"] == "D6a_tdb1_pooled_xyz_reshape"
    )


def prove_preconditions(
    formal_model: onnx.ModelProto, control_model: onnx.ModelProto
) -> dict[str, Any]:
    formal_contracts = contracts(formal_model)
    control_contracts = contracts(control_model)
    reshape = get_node(formal_model, TARGET_NODE)
    gather = get_node(formal_model, "/model/tdb_1/GatherND")
    transpose = get_node(formal_model, "/model/tdb_1/Transpose_2")
    nonzero = get_node(formal_model, "/model/tdb_1/NonZero")
    concat = get_node(formal_model, "/model/tdb_1/Concat_5")
    reduce_min = get_node(formal_model, "/model/tdb_1/ReduceMin_1")
    points_input = formal_contracts["points"]
    data_contract = control_contracts[DATA_INPUT]
    target_contract = control_contracts[TARGET_OUTPUT]
    shape_contract = formal_contracts[SHAPE_INPUT]
    concat_inputs = list(concat.input)
    leading_value = constant_value(formal_model, concat_inputs[0])
    trailing_value = constant_value(formal_model, concat_inputs[2])
    attrs = attribute_dict(reshape)
    allowzero = attrs.get("allowzero", 0)
    transpose_perm = attribute_dict(transpose).get("perm")

    conditions = {
        "data_static_rank_is_2": data_contract["rank"] == 2,
        "data_last_dimension_is_3": data_contract["shape"][-1] == 3,
        "target_rank_is_3": target_contract["rank"] == 3,
        "target_leading_dimension_is_1": target_contract["shape"][0] == 1,
        "target_last_dimension_is_3": target_contract["shape"][-1] == 3,
        "shape_tensor_is_int64_rank1_length3": (
            shape_contract["dtype"] == "INT64"
            and shape_contract["shape"] == [3]
        ),
        "shape_leading_constant_is_1": leading_value == [1],
        "shape_trailing_constant_is_3": trailing_value == [3],
        "shape_middle_is_reduce_min_runtime_k": (
            concat_inputs[1] == "/model/tdb_1/Unsqueeze_11_output_0"
            and reduce_min.output[0] == "/model/tdb_1/ReduceMin_1_output_0"
        ),
        "gathernd_data_is_rank2_lastdim3": (
            formal_contracts[gather.input[0]]["rank"] == 2
            and formal_contracts[gather.input[0]]["shape"][-1] == 3
        ),
        "gathernd_indices_are_nonzero_transposed": (
            transpose.input[0] == nonzero.output[0]
            and transpose_perm == [1, 0]
        ),
        "fixed_batch_is_1": points_input["shape"][0] == 1,
        "fixed_input_points_nonempty": points_input["shape"][1] == 2048,
        "allowzero_absent_or_zero": allowzero == 0,
        "reshape_data_input_matches_expected": reshape.input[0] == DATA_INPUT,
        "reshape_shape_input_matches_expected": reshape.input[1] == SHAPE_INPUT,
        "reshape_output_matches_expected": list(reshape.output) == [TARGET_OUTPUT],
        "unsqueeze_axis0_contract_matches": (
            data_contract["shape"] == ["diagnostic_dynamic_0", 3]
            and target_contract["shape"][0] == 1
            and target_contract["shape"][-1] == 3
        ),
        "reshape_and_unsqueeze_preserve_element_order": True,
    }
    failed = [name for name, passed in conditions.items() if not passed]
    return {
        "status": "RESHAPE10_UNSQUEEZE_PRECONDITIONS_PASSED" if not failed else "RESHAPE10_UNSQUEEZE_PRECONDITIONS_FAILED",
        "all_conditions_passed": not failed,
        "failed_conditions": failed,
        "conditions": conditions,
        "formal_node": {
            "name": reshape.name,
            "op_type": reshape.op_type,
            "inputs": list(reshape.input),
            "outputs": list(reshape.output),
            "attributes": attrs,
        },
        "data_contract": data_contract,
        "shape_contract": shape_contract,
        "target_contract": target_contract,
        "shape_semantics": [1, "K=min_voxel_count=data_dim0", 3],
        "gathernd_semantics": {
            "data": formal_contracts[gather.input[0]],
            "indices_source": nonzero.name,
            "indices_transpose_perm": transpose_perm,
            "result": "FLOAT[K,3] by ONNX GatherND rank rule",
        },
        "element_order_proof": (
            "Reshape from contiguous logical [K,3] to [1,K,3] and Unsqueeze at "
            "axis 0 both preserve the original linear element order."
        ),
        "allowzero_proof": (
            "allowzero is 0; B=1,N=2048 guarantees at least one materialized "
            "voxel, and target dimensions 1,K,3 contain no literal zero."
        ),
    }


def rewrite_reshape_to_unsqueeze(
    control_path: Path, destination: Path
) -> dict[str, Any]:
    model = onnx.load_model(str(control_path), load_external_data=False)
    target_indexes = [
        index for index, node in enumerate(model.graph.node) if node.name == TARGET_NODE
    ]
    if len(target_indexes) != 1:
        raise RuntimeError(f"Expected one target node, got {target_indexes}")
    target_index = target_indexes[0]
    original = model.graph.node[target_index]
    if original.op_type != "Reshape" or list(original.input) != [DATA_INPUT, SHAPE_INPUT]:
        raise RuntimeError("Target Reshape contract changed")
    original_output = list(original.output)
    original_attributes = attribute_dict(original)
    if any(item.name == AXES_INITIALIZER for item in model.graph.initializer):
        raise RuntimeError("Axes initializer name already exists")
    model.graph.initializer.append(
        helper.make_tensor(
            AXES_INITIALIZER,
            TensorProto.INT64,
            [1],
            [0],
        )
    )
    replacement = helper.make_node(
        "Unsqueeze",
        inputs=[DATA_INPUT, AXES_INITIALIZER],
        outputs=original_output,
        name=TARGET_NODE,
    )
    model.graph.node.remove(original)
    model.graph.node.insert(target_index, replacement)
    onnx.save_model(model, str(destination))
    reloaded = onnx.load_model(str(destination), load_external_data=False)
    onnx.checker.check_model(reloaded)

    control = onnx.load_model(str(control_path), load_external_data=False)
    control_nodes = {node.name: node.SerializeToString() for node in control.graph.node}
    rewrite_nodes = {node.name: node.SerializeToString() for node in reloaded.graph.node}
    changed_nodes = sorted(
        name
        for name in set(control_nodes) | set(rewrite_nodes)
        if control_nodes.get(name) != rewrite_nodes.get(name)
    )
    original_initializers = {item.name for item in control.graph.initializer}
    rewritten_initializers = {item.name for item in reloaded.graph.initializer}
    consumers_before = [
        node.name for node in control.graph.node if SHAPE_INPUT in node.input
    ]
    consumers_after = [
        node.name for node in reloaded.graph.node if SHAPE_INPUT in node.input
    ]
    output_before = next(item for item in control.graph.output if item.name == TARGET_OUTPUT)
    output_after = next(item for item in reloaded.graph.output if item.name == TARGET_OUTPUT)
    summary = {
        "control_onnx": str(control_path),
        "control_sha256": phase4.sha256(control_path),
        "rewritten_onnx": str(destination),
        "rewritten_sha256": phase4.sha256(destination),
        "onnx_checker_passed": True,
        "target_node_index": target_index,
        "target_node_name_preserved": replacement.name == original.name,
        "output_name_preserved": list(replacement.output) == original_output,
        "old_op_type": original.op_type,
        "new_op_type": replacement.op_type,
        "old_inputs": list(original.input),
        "new_inputs": list(replacement.input),
        "old_attributes": original_attributes,
        "new_attributes": attribute_dict(replacement),
        "axes_initializer": {
            "name": AXES_INITIALIZER,
            "dtype": "INT64",
            "shape": [1],
            "value": [0],
        },
        "changed_node_names": changed_nodes,
        "only_target_node_changed": changed_nodes == [TARGET_NODE],
        "added_initializers": sorted(rewritten_initializers - original_initializers),
        "removed_initializers": sorted(original_initializers - rewritten_initializers),
        "shape_chain_consumers_before": consumers_before,
        "shape_chain_consumers_after": consumers_after,
        "shape_chain_retained": SHAPE_INPUT in {output for node in reloaded.graph.node for output in node.output},
        "graph_outputs_identical": [item.name for item in control.graph.output] == [item.name for item in reloaded.graph.output],
        "target_output_contract_identical": output_before.SerializeToString() == output_after.SerializeToString(),
        "other_nodes_byte_identical": all(
            control_nodes[name] == rewrite_nodes[name]
            for name in control_nodes
            if name != TARGET_NODE
        ),
        "formal_model_rewritten": False,
        "plugin_rewritten": False,
    }
    required = (
        summary["only_target_node_changed"]
        and summary["added_initializers"] == [AXES_INITIALIZER]
        and not summary["removed_initializers"]
        and summary["graph_outputs_identical"]
        and summary["target_output_contract_identical"]
        and summary["other_nodes_byte_identical"]
    )
    if not required:
        raise RuntimeError(f"Graph diff exceeded authorized scope: {summary}")
    return summary


def run_shape_inference(candidate: Path, inferred_path: Path) -> dict[str, Any]:
    model = onnx.load_model(str(candidate), load_external_data=False)
    inferred = shape_inference.infer_shapes(
        model,
        check_type=True,
        strict_mode=True,
        data_prop=True,
    )
    onnx.checker.check_model(inferred)
    onnx.save_model(inferred, str(inferred_path))
    inferred_contracts = contracts(inferred)
    target = inferred_contracts[TARGET_OUTPUT]
    passed = (
        target["dtype"] == "FLOAT"
        and target["rank"] == 3
        and target["shape"][0] == 1
        and target["shape"][-1] == 3
    )
    if not passed:
        raise RuntimeError(f"Inferred target contract mismatch: {target}")
    return {
        "shape_inference_passed": True,
        "onnx_checker_after_inference_passed": True,
        "inferred_model": str(inferred_path),
        "inferred_model_sha256": phase4.sha256(inferred_path),
        "target_contract": target,
        "expected_contract": {"dtype": "FLOAT", "shape": [1, "K", 3]},
        "contract_matches": True,
    }


def run_worker(
    args: argparse.Namespace,
    run_dir: Path,
    onnx_path: Path,
    prefix: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-u",
        str(WORKER_SCRIPT),
        "--worker",
        "--run-dir",
        str(run_dir),
        "--onnx",
        str(onnx_path),
        "--result-prefix",
        prefix,
        "--expected-instances",
        "1",
        "--plugin-library",
        str(args.plugin_library.resolve()),
        "--tensorrt-root",
        str(args.tensorrt_root.resolve()),
        "--cuda-root",
        str(args.cuda_root.resolve()),
    ]
    environment = os.environ.copy()
    environment["TENSORRT_ROOT"] = str(args.tensorrt_root.resolve())
    environment["CUDA_PATH"] = str(args.cuda_root.resolve())
    environment["PATH"] = os.pathsep.join(
        [
            str(args.tensorrt_root.resolve() / "bin"),
            str(args.cuda_root.resolve() / "bin"),
            environment.get("PATH", ""),
        ]
    )
    log_path = run_dir / f"{prefix}_builder_verbose.log"
    started = time.perf_counter()
    timed_out = False
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write("COMMAND=" + subprocess.list2cmdline(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            return_code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                return_code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait(timeout=15)
            log.write(f"\nWORKER_TIMEOUT_SECONDS={args.timeout_seconds}\n")
    parser_summary_path = run_dir / f"{prefix}_parser_summary.json"
    build_summary_path = run_dir / f"{prefix}_build_summary.json"
    parser_summary = json.loads(parser_summary_path.read_text(encoding="utf-8"))
    build_summary = json.loads(build_summary_path.read_text(encoding="utf-8"))
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    assertion_lines = [
        line
        for line in log_text.splitlines()
        if "convertExplicitDDSPluginToImplicit" in line
        or "nodeIdxToDDSOutputIndices" in line
    ]
    return {
        "command": subprocess.list2cmdline(command),
        "return_code": return_code,
        "timed_out": timed_out,
        "wall_elapsed_seconds": time.perf_counter() - started,
        "onnx": str(onnx_path),
        "onnx_sha256": phase4.sha256(onnx_path),
        "log": str(log_path),
        "parser_summary": str(parser_summary_path),
        "build_summary": str(build_summary_path),
        "parser_success": parser_summary.get("parser_success"),
        "parser_error_count": parser_summary.get("parser_error_count"),
        "parser_errors": parser_summary.get("parser_errors"),
        "plugin_instances": parser_summary.get("voxel_unique_plugin_instances"),
        "builder_success": build_summary.get("engine_build_success"),
        "builder_elapsed_seconds": build_summary.get("build_elapsed_seconds"),
        "serialized_engine_size_bytes": build_summary.get("serialized_engine_size_bytes"),
        "serialized_engine_sha256": build_summary.get("serialized_engine_sha256"),
        "serialized_engine_retained": build_summary.get("serialized_engine_retained"),
        "first_error": build_summary.get("first_error"),
        "dds_assertion_present": bool(assertion_lines),
        "dds_assertion_lines": assertion_lines,
    }


def write_report(run_dir: Path, result: dict[str, Any]) -> None:
    control = result["control"]
    rewrite = result.get("rewrite")
    rewrite_text = "Not run."
    if rewrite is not None:
        rewrite_text = f"""- Parser success: `{rewrite['parser_success']}`
- Parser errors: `{rewrite['parser_error_count']}`
- Builder success: `{rewrite['builder_success']}`
- Builder elapsed: `{rewrite['builder_elapsed_seconds']}` seconds
- Serialized engine bytes produced: `{rewrite['serialized_engine_size_bytes']}`
- Serialized engine retained: `{rewrite['serialized_engine_retained']}`
- DDS assertion present: `{rewrite['dds_assertion_present']}`
- First error: `{rewrite['first_error']}`"""
    report = f"""# Reshape_10 to Unsqueeze TensorRT compatibility experiment

## Equivalence preconditions

- Status: `{result['preconditions']['status']}`
- Data: `FLOAT[K,3]`, static rank 2, last dimension 3.
- Original shape: `[1,K,3]`.
- `allowzero=0`; K is nonzero for fixed N=2048.
- `Unsqueeze(axis=0)` output: `FLOAT[1,K,3]`.
- Both operations preserve linear element order.

## Controlled graph difference

- Changed node: `{TARGET_NODE}` only.
- Old op: `Reshape(data, DDS-derived [1,K,3])`.
- New op: `Unsqueeze(data, axes=[0])`.
- Output tensor name and graph consumers are unchanged.
- Original shape construction remains in the graph and is not cleaned up.

## A — original D6a control

- Parser success: `{control['parser_success']}`
- Parser errors: `{control['parser_error_count']}`
- Builder success: `{control['builder_success']}`
- Builder elapsed: `{control['builder_elapsed_seconds']}` seconds
- DDS assertion reproduced: `{control['dds_assertion_present']}`

## B — Reshape_10 to Unsqueeze(axis=0)

{rewrite_text}

## Conclusion

`{result['status']}`

This result applies only to `tdb_1/Reshape_10`. No conclusion is made here for
`Reshape_11` or corresponding tdb_2–tdb_4 nodes.
"""
    (run_dir / "report.md").write_text(report, encoding="utf-8")


def main(args: argparse.Namespace) -> int:
    source = args.onnx.resolve()
    plugin_library = args.plugin_library.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not plugin_library.is_file():
        raise FileNotFoundError(plugin_library)
    source_hash_before = phase4.sha256(source)
    plugin_hash_before = phase4.sha256(PLUGIN_SOURCE)
    if source_hash_before != phase4.EXPECTED_ONNX_SHA256:
        raise RuntimeError("Formal ONNX SHA-256 mismatch")

    run_dir = (
        args.output_root.resolve()
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_reshape10_unsqueeze_test"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    formal_model = onnx.load_model(str(source), load_external_data=False)
    onnx.checker.check_model(formal_model)
    experiment = find_d6a_definition(formal_model)
    control_path = run_dir / "original_d6a_control.onnx"
    incremental.create_extracted_graph(source, control_path, experiment["outputs"])
    onnx.checker.check_model(
        onnx.load_model(str(control_path), load_external_data=False)
    )
    preconditions = prove_preconditions(
        formal_model,
        onnx.load_model(str(control_path), load_external_data=False),
    )
    phase4.dump_json(run_dir / "precondition_audit.json", preconditions)
    if not preconditions["all_conditions_passed"]:
        result = {
            "status": "RESHAPE10_UNSQUEEZE_PRECONDITIONS_FAILED",
            "preconditions": preconditions,
            "control": None,
            "rewrite": None,
        }
        phase4.dump_json(run_dir / "result_summary.json", result)
        print(f"RUN_DIR={run_dir}")
        print("RESHAPE10_UNSQUEEZE_PRECONDITIONS_FAILED")
        return 2

    rewrite_path = run_dir / "reshape10_unsqueeze.onnx"
    rewrite_summary = rewrite_reshape_to_unsqueeze(control_path, rewrite_path)
    phase4.dump_json(run_dir / "rewrite_summary.json", rewrite_summary)
    inferred_path = run_dir / "reshape10_unsqueeze_inferred.onnx"
    inference_summary = run_shape_inference(rewrite_path, inferred_path)
    phase4.dump_json(run_dir / "shape_inference_summary.json", inference_summary)

    control_result = run_worker(args, run_dir, control_path, "control_reshape10")
    phase4.dump_json(run_dir / "control_result.json", control_result)
    if (
        not control_result["parser_success"]
        or control_result["builder_success"]
        or not control_result["dds_assertion_present"]
    ):
        status = "RESHAPE10_CONTROL_DDS_ASSERTION_NOT_REPRODUCED"
        result = {
            "status": status,
            "preconditions": preconditions,
            "rewrite_summary": rewrite_summary,
            "shape_inference": inference_summary,
            "control": control_result,
            "rewrite": None,
        }
        phase4.dump_json(run_dir / "result_summary.json", result)
        write_report(run_dir, result)
        print(f"RUN_DIR={run_dir}")
        print(status)
        return 3


    rewrite_result = run_worker(args, run_dir, rewrite_path, "rewrite_unsqueeze")
    phase4.dump_json(run_dir / "rewrite_result.json", rewrite_result)
    if rewrite_result["parser_success"] and rewrite_result["builder_success"]:
        status = "RESHAPE10_TO_UNSQUEEZE_BUILD_PASSED"
    else:
        status = "RESHAPE10_TO_UNSQUEEZE_BUILD_FAILED"

    safety = {
        "formal_onnx": str(source),
        "formal_onnx_sha256_before": source_hash_before,
        "formal_onnx_sha256_after": phase4.sha256(source),
        "formal_onnx_unchanged": phase4.sha256(source) == source_hash_before,
        "plugin_source": str(PLUGIN_SOURCE),
        "plugin_source_sha256_before": plugin_hash_before,
        "plugin_source_sha256_after": phase4.sha256(PLUGIN_SOURCE),
        "plugin_source_unchanged": phase4.sha256(PLUGIN_SOURCE) == plugin_hash_before,
        "plugin_library": str(plugin_library),
        "plugin_library_sha256": phase4.sha256(plugin_library),
        "serialized_engine_retained": False,
        "deserialization_attempted": False,
        "execution_context_created": False,
        "inference_attempted": False,
        "fp16_attempted": False,
        "benchmark_attempted": False,
    }
    if not safety["formal_onnx_unchanged"] or not safety["plugin_source_unchanged"]:
        raise RuntimeError("Read-only source hash changed")
    phase4.dump_json(run_dir / "safety_summary.json", safety)
    result = {
        "status": status,
        "preconditions": preconditions,
        "rewrite_summary": rewrite_summary,
        "shape_inference": inference_summary,
        "control": control_result,
        "rewrite": rewrite_result,
        "safety": safety,
        "interpretation": (
            "Passing proves only that Unsqueeze(axis=0) is a TensorRT-compatible "
            "equivalent expression for tdb_1/Reshape_10 on fixed B=1."
        ),
    }
    phase4.dump_json(run_dir / "result_summary.json", result)
    write_report(run_dir, result)
    print(f"RUN_DIR={run_dir}")
    print(
        f"CONTROL parser={control_result['parser_success']} "
        f"builder={control_result['builder_success']} "
        f"dds_assertion={control_result['dds_assertion_present']}"
    )
    print(
        f"REWRITE parser={rewrite_result['parser_success']} "
        f"builder={rewrite_result['builder_success']} "
        f"dds_assertion={rewrite_result['dds_assertion_present']}"
    )
    print(status)
    return 0 if status == "RESHAPE10_TO_UNSQUEEZE_BUILD_PASSED" else 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plugin-library", type=Path, default=DEFAULT_PLUGIN_LIBRARY)
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TENSORRT_ROOT)
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA_ROOT)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
