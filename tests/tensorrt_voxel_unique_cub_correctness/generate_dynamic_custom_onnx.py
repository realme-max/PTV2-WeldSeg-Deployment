"""Generate an isolated dynamic-N ONNX for the experimental CUB plugin."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


DOMAIN = "com.tensorrt.ptv2.experimental"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    graph = helper.make_graph(
        [
            helper.make_node(
                "VoxelUniqueCub",
                ["voxel_key"],
                ["voxel_count", "unique_values", "inverse_indices"],
                name="VoxelUniqueCubCorrectnessNode",
                domain=DOMAIN,
                plugin_version="1",
                plugin_namespace=DOMAIN,
            )
        ],
        "VoxelUniqueCubCorrectnessGraph",
        [helper.make_tensor_value_info("voxel_key", TensorProto.INT64, ["N"])],
        [
            helper.make_tensor_value_info("voxel_count", TensorProto.INT32, []),
            helper.make_tensor_value_info("unique_values", TensorProto.INT64, ["M"]),
            helper.make_tensor_value_info("inverse_indices", TensorProto.INT64, ["N"]),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="PTV2-WeldSeg-Deployment",
        producer_version="phase8b",
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid(DOMAIN, 1)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    output.write_bytes(model.SerializeToString())
    onnx.checker.check_model(onnx.load_model(str(output), load_external_data=False))
    print(f"DYNAMIC_CUSTOM_ONNX={output}")
    print("VOXELUNIQUE_CUB_DIAGNOSTIC_ONNX_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
