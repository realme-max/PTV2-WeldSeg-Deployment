"""Generate an isolated custom-domain ONNX model for TensorRT parser testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


PLUGIN_DOMAIN = "com.tensorrt.ptv2"
PLUGIN_NAME = "VoxelUnique"
PLUGIN_VERSION = "1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    keys = helper.make_tensor_value_info("keys", TensorProto.INT64, [4])
    voxel_count = helper.make_tensor_value_info(
        "voxel_count", TensorProto.INT32, []
    )
    unique_values = helper.make_tensor_value_info(
        "unique_values", TensorProto.INT64, [None]
    )
    inverse_indices = helper.make_tensor_value_info(
        "inverse_indices", TensorProto.INT64, [4]
    )

    node = helper.make_node(
        PLUGIN_NAME,
        inputs=["keys"],
        outputs=["voxel_count", "unique_values", "inverse_indices"],
        name="VoxelUniquePrototypeNode",
        domain=PLUGIN_DOMAIN,
        plugin_version=PLUGIN_VERSION,
        plugin_namespace=PLUGIN_DOMAIN,
    )
    graph = helper.make_graph(
        [node],
        "VoxelUniqueCustomPluginGraph",
        [keys],
        [voxel_count, unique_values, inverse_indices],
    )
    model = helper.make_model(
        graph,
        producer_name="PTV2-WeldSeg-Deployment",
        producer_version="phase2b",
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid(PLUGIN_DOMAIN, 1),
        ],
    )
    # TensorRT 11.1 accepts the same IR generation used by the deployment ONNX.
    model.ir_version = 10
    model.metadata_props.add(
        key="purpose", value="isolated TensorRT custom plugin parser audit"
    )
    model.metadata_props.add(key="plugin_name", value=PLUGIN_NAME)
    model.metadata_props.add(key="plugin_version", value=PLUGIN_VERSION)
    model.metadata_props.add(key="plugin_namespace", value=PLUGIN_DOMAIN)

    onnx.checker.check_model(model)
    output.write_bytes(model.SerializeToString())
    loaded = onnx.load_model(str(output), load_external_data=False)
    onnx.checker.check_model(loaded)

    input_path = (
        args.input_npz.resolve()
        if args.input_npz
        else output.with_name("custom_input.npz")
    )
    np.savez(
        input_path,
        keys=np.asarray([3, 1, 3, 2], dtype=np.int64),
        expected_count=np.asarray(3, dtype=np.int32),
        expected_values=np.asarray([1, 2, 3], dtype=np.int64),
        expected_inverse=np.asarray([2, 0, 2, 1], dtype=np.int64),
    )

    print(f"CUSTOM_ONNX={output}")
    print(f"CUSTOM_INPUT={input_path}")
    print(f"DOMAIN={PLUGIN_DOMAIN}")
    print(f"OP_TYPE={PLUGIN_NAME}")
    print("CUSTOM_ONNX_GENERATION_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

