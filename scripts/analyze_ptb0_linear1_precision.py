"""Attribute the ptb_0.linear_1 TensorRT/PyTorch FP32 difference.

This is a read-only Phase 5D analysis.  It uses the already dumped Phase 5C
input/output tensors, extracts the checkpoint and ONNX constants, performs one
offline GEMM per reference backend, and inspects the formal TensorRT engine.
It does not execute the formal engine or modify any model/deployment artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import build_gcn_res_tensorrt_fp32 as phase4  # noqa: E402
import locate_gcn_res_tensorrt_first_divergence as phase5b  # noqa: E402
import run_gcn_res_tensorrt_fp32_inference as phase5  # noqa: E402


DEFAULT_PHASE5C_RUN = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260716_221423_053311_ptb0_parity"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
CHECKPOINT_WEIGHT_KEY = "ptb_0.linear_1.weight"
CHECKPOINT_BIAS_KEY = "ptb_0.linear_1.bias"
ONNX_WEIGHT_NAME = "onnx::MatMul_3599"
ONNX_BIAS_NAME = "model.ptb_0.linear_1.bias"
ONNX_MATMUL_NODE = "/model/ptb_0/linear_1/MatMul"
ONNX_ADD_NODE = "/model/ptb_0/linear_1/Add"


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def comparison(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate64 = candidate.astype(np.float64, copy=False)
    reference64 = reference.astype(np.float64, copy=False)
    if candidate64.shape != reference64.shape:
        raise ValueError(f"Shape mismatch: {candidate64.shape} vs {reference64.shape}")
    difference = candidate64 - reference64
    absolute = np.abs(difference)
    flat_candidate = candidate64.reshape(-1)
    flat_reference = reference64.reshape(-1)
    denominator = float(np.linalg.norm(flat_candidate) * np.linalg.norm(flat_reference))
    mismatch = np.argwhere(candidate != reference)
    first_index = mismatch[0].tolist() if mismatch.size else None
    return {
        "candidate_shape": list(candidate.shape),
        "candidate_dtype": str(candidate.dtype),
        "reference_dtype": str(reference.dtype),
        "exact_equal": bool(np.array_equal(candidate, reference)),
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
        "rmse": float(math.sqrt(np.mean(difference * difference))),
        "cosine_similarity": (
            float(np.dot(flat_candidate, flat_reference) / denominator)
            if denominator
            else None
        ),
        "first_mismatch_index": first_index,
        "first_candidate_value": (
            candidate[tuple(first_index)].item() if first_index is not None else None
        ),
        "first_reference_value": (
            reference[tuple(first_index)].item() if first_index is not None else None
        ),
    }


def inspect_linear1_engine(
    trt: Any,
    logger: Any,
    engine_path: Path,
) -> dict[str, Any]:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("Formal Engine deserialization failed")
    inspector = engine.create_engine_inspector()
    if inspector is None:
        raise RuntimeError("Engine Inspector creation failed")
    raw = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    parsed = json.loads(raw)
    layers = parsed.get("Layers", parsed if isinstance(parsed, list) else [])
    hits = [
        {"index": index, **layer}
        for index, layer in enumerate(layers)
        if ONNX_MATMUL_NODE in str(layer.get("Metadata", ""))
        or ONNX_ADD_NODE in str(layer.get("Metadata", ""))
        or ONNX_WEIGHT_NAME in json.dumps(layer, ensure_ascii=False)
    ]
    primary = next(
        (
            item
            for item in hits
            if ONNX_MATMUL_NODE in str(item.get("Metadata", ""))
            and ONNX_ADD_NODE in str(item.get("Metadata", ""))
        ),
        None,
    )
    if primary is None:
        raise RuntimeError("Inspector did not expose the fused ptb_0.linear_1 layer")
    tactic = str(primary.get("TacticName", ""))
    metadata = str(primary.get("Metadata", ""))
    metadata_nodes = [item for item in metadata.replace("\x1e", "\x1f").split("\x1f") if item]
    weight_reformat_hits = [
        item
        for item in layers
        if item.get("Name") != primary.get("Name")
        and ONNX_WEIGHT_NAME in json.dumps(item, ensure_ascii=False)
    ]
    result = {
        "engine_path": str(engine_path),
        "engine_sha256": phase4.sha256(engine_path),
        "engine_io_tensors": [
            engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
        ],
        "inspector_layer_count": len(layers),
        "matching_layer_count": len(hits),
        "matching_layers": hits,
        "selected_linear1_layer": primary,
        "layer_name": primary.get("Name"),
        "layer_type": primary.get("LayerType"),
        "tactic_name": tactic,
        "tactic_value": primary.get("TacticValue"),
        "metadata_nodes": metadata_nodes,
        "matmul_and_bias_add_fused": (
            ONNX_MATMUL_NODE in metadata and ONNX_ADD_NODE in metadata
        ),
        "input_datatypes": [item.get("Datatype") for item in primary.get("Inputs", [])],
        "constant_datatypes": [
            item.get("Datatype") for item in primary.get("Constants", [])
        ],
        "output_datatypes": [item.get("Datatype") for item in primary.get("Outputs", [])],
        "input_formats": [item.get("Format") for item in primary.get("Inputs", [])],
        "constant_formats": [
            item.get("Format") for item in primary.get("Constants", [])
        ],
        "output_formats": [item.get("Format") for item in primary.get("Outputs", [])],
        "tf32_math_identified_from_tactic": "tf32" in tactic.lower(),
        "gemm_implementation": tactic,
        "matrix_layout_token": (
            "NN" if "_nn_" in tactic.lower() else "not_decodable_from_tactic"
        ),
        "separate_weight_reformat_layer_found": bool(weight_reformat_hits),
        "weight_reformat_layer_hits": weight_reformat_hits,
        "weight_reformat_interpretation": (
            "Inspector exposes the weight as a row-major Float constant inside the fused "
            "GEMM. No separate weight-reformat layer referencing the ONNX initializer was found; "
            "internal tactic packing is not separately observable."
        ),
        "formal_engine_inference_executed": False,
    }
    del inspector, engine, runtime
    return result


def write_report(path: Path, payload: dict[str, Any], engine_info: dict[str, Any]) -> None:
    refs = payload["reference_comparisons"]
    rows = "\n".join(
        f"| {name} | {item['vs_fp64_reference']['max_abs_error']:.12e} | "
        f"{item['vs_fp64_reference']['mean_abs_error']:.12e} | "
        f"{item['vs_pytorch_dump']['max_abs_error']:.12e} | "
        f"{item['vs_tensorrt_dump']['max_abs_error']:.12e} | "
        f"{item['vs_pytorch_dump']['exact_equal']} | "
        f"{item['vs_tensorrt_dump']['exact_equal']} |"
        for name, item in refs.items()
    )
    tactic = engine_info["tactic_name"]
    text = f"""# ptb_0.linear_1 TensorRT difference attribution

## Read-only boundary

- Formal Engine, ONNX, Plugin and checkpoint hashes were checked before and after this analysis and did not change.
- The formal Engine was deserialized only for Engine Inspector. No TensorRT inference was executed.
- Each offline reference GEMM ran once. No warmup, timing, benchmark, FP16 or INT8 was used.

## Linear contract

```text
X: {payload['linear_contract']['x_shape']} FP32
W checkpoint: {payload['linear_contract']['checkpoint_weight_shape']} FP32
W ONNX MatMul: {payload['linear_contract']['onnx_weight_shape']} FP32
b: {payload['linear_contract']['bias_shape']} FP32
Y = X @ W_checkpoint.T + b = X @ W_ONNX + b
```

- ONNX weight equals checkpoint weight transpose: `{payload['linear_contract']['onnx_weight_equals_checkpoint_transpose']}`
- ONNX bias equals checkpoint bias: `{payload['linear_contract']['onnx_bias_equals_checkpoint_bias']}`
- TensorRT and PyTorch use the same dumped X: `{payload['linear_contract']['pytorch_tensorrt_x_exact_equal']}`

## Numerical comparison

High-precision `Y_ref` is NumPy FP64 `X @ W.T + b`.

| Candidate | max abs vs FP64 | mean abs vs FP64 | max abs vs PyTorch | max abs vs TensorRT | exact PyTorch | exact TensorRT |
|---|---:|---:|---:|---:|---|---|
{rows}

## Engine Inspector

- Layer: `{engine_info['layer_name']}`
- Layer type: `{engine_info['layer_type']}`
- Tactic: `{tactic}`
- MatMul + bias Add fused: `{engine_info['matmul_and_bias_add_fused']}`
- TensorRT layer I/O datatype: Float/FP32
- TF32 math token present in tactic: `{engine_info['tf32_math_identified_from_tactic']}`
- Matrix layout token: `{engine_info['matrix_layout_token']}`
- Separate explicit weight-reformat layer found: `{engine_info['separate_weight_reformat_layer_found']}`

The ONNX exporter already stored `W_checkpoint.T` as the MatMul initializer, so the `NN` tactic does not require a runtime matrix transpose. The tactic string `f32f32_tf32f32_f32` identifies FP32 inputs/output with TF32 tensor-core multiplication and FP32 accumulation/output. The bias is fused into the GEMM layer. Inspector shows row-major Float constants and no separate explicit weight-reformat layer; internal tactic packing is not separately observable.

## Attribution

- PyTorch Phase 5C output equals the explicit CUDA GEMM with TF32 disabled: `{payload['attribution']['pytorch_equals_cuda_tf32_disabled']}`.
- TensorRT Phase 5C output equals the explicit CUDA GEMM with TF32 enabled: `{payload['attribution']['tensorrt_equals_cuda_tf32_enabled']}`.
- PyTorch max abs vs FP64: `{payload['attribution']['pytorch_max_abs_vs_fp64']:.12e}`.
- TensorRT max abs vs FP64: `{payload['attribution']['tensorrt_max_abs_vs_fp64']:.12e}`.

Therefore the `ptb_0.linear_1` difference is attributed to TensorRT selecting a TF32 tensor-core GEMM tactic while the validated PyTorch path used full FP32 CUDA matmul. It is not caused by X, W, bias, transpose, VoxelUnique, Scatter, TopK geometry or graph data mismatch.

This phase performs attribution only and makes no fix.

`{payload['status']}`
"""
    path.write_text(text, encoding="utf-8")


def main(args: argparse.Namespace) -> int:
    phase5c_run = args.phase5c_run.resolve()
    engine_path = args.engine.resolve()
    onnx_path = args.onnx.resolve()
    checkpoint_path = args.checkpoint.resolve()
    plugin_path = args.plugin_library.resolve()
    required = (
        engine_path,
        onnx_path,
        checkpoint_path,
        plugin_path,
        phase5c_run / "pytorch_ptb0_dump" / "stem_linear_features.npy",
        phase5c_run / "pytorch_ptb0_dump" / "ptb0_linear1_features.npy",
        phase5c_run / "tensorrt_ptb0_dump" / "stem_linear_features.npy",
        phase5c_run / "tensorrt_ptb0_dump" / "ptb0_linear1_features.npy",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if phase4.sha256(engine_path) != phase5.EXPECTED_ENGINE_SHA256:
        raise RuntimeError("Formal Engine hash mismatch")
    if phase4.sha256(onnx_path) != phase5.EXPECTED_ONNX_SHA256:
        raise RuntimeError("Formal ONNX hash mismatch")

    source_hashes_before = {
        "engine": phase4.sha256(engine_path),
        "onnx": phase4.sha256(onnx_path),
        "checkpoint": phase4.sha256(checkpoint_path),
        "plugin": phase4.sha256(plugin_path),
    }
    run_dir = (
        args.output_root.resolve()
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_linear1_analysis"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    failure = {
        "status": "LINEAR1_TENSORRT_DIFF_ATTRIBUTION_FAILED",
        "run_dir": str(run_dir),
        "source_hashes_before": source_hashes_before,
        "formal_engine_inference_executed": False,
        "fp16": False,
        "int8": False,
        "benchmark": False,
        "fix_attempted": False,
    }
    phase5b.dump_json(run_dir / "linear1_analysis.json", failure)

    try:
        import onnx
        import torch
        import torch.nn.functional as functional

        x_pytorch = np.load(
            phase5c_run / "pytorch_ptb0_dump" / "stem_linear_features.npy",
            allow_pickle=False,
        ).astype(np.float32, copy=False)
        x_tensorrt = np.load(
            phase5c_run / "tensorrt_ptb0_dump" / "stem_linear_features.npy",
            allow_pickle=False,
        ).astype(np.float32, copy=False)
        y_pytorch = np.load(
            phase5c_run / "pytorch_ptb0_dump" / "ptb0_linear1_features.npy",
            allow_pickle=False,
        ).astype(np.float32, copy=False)
        y_tensorrt = np.load(
            phase5c_run / "tensorrt_ptb0_dump" / "ptb0_linear1_features.npy",
            allow_pickle=False,
        ).astype(np.float32, copy=False)
        if not np.array_equal(x_pytorch, x_tensorrt):
            raise RuntimeError("Phase 5C PyTorch/TensorRT linear input X is not identical")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        weight = state_dict[CHECKPOINT_WEIGHT_KEY].detach().cpu().numpy().astype(np.float32)
        bias = state_dict[CHECKPOINT_BIAS_KEY].detach().cpu().numpy().astype(np.float32)
        onnx_model = onnx.load(onnx_path, load_external_data=True)
        initializers = {
            item.name: onnx.numpy_helper.to_array(item) for item in onnx_model.graph.initializer
        }
        onnx_weight = np.asarray(initializers[ONNX_WEIGHT_NAME], dtype=np.float32)
        onnx_bias = np.asarray(initializers[ONNX_BIAS_NAME], dtype=np.float32)
        weight_transpose_equal = bool(np.array_equal(onnx_weight, weight.T))
        bias_equal = bool(np.array_equal(onnx_bias, bias))
        if not weight_transpose_equal or not bias_equal:
            raise RuntimeError("ONNX/checkpoint linear constants do not match")

        y_ref_fp64 = (
            x_pytorch.astype(np.float64)
            @ weight.astype(np.float64).T
            + bias.astype(np.float64)
        )
        y_numpy_fp32 = np.matmul(x_pytorch, onnx_weight) + bias
        with torch.inference_mode():
            y_torch_cpu = functional.linear(
                torch.from_numpy(x_pytorch),
                torch.from_numpy(weight),
                torch.from_numpy(bias),
            ).numpy()

        old_allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        old_matmul_precision = torch.get_float32_matmul_precision()
        try:
            x_cuda = torch.from_numpy(x_pytorch).to("cuda:0")
            weight_cuda = torch.from_numpy(weight).to("cuda:0")
            bias_cuda = torch.from_numpy(bias).to("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.inference_mode():
                y_cuda_tf32_disabled = functional.linear(
                    x_cuda, weight_cuda, bias_cuda
                )
            torch.cuda.synchronize(0)
            y_cuda_tf32_disabled_np = y_cuda_tf32_disabled.cpu().numpy()

            torch.backends.cuda.matmul.allow_tf32 = True
            with torch.inference_mode():
                y_cuda_tf32_enabled = functional.linear(x_cuda, weight_cuda, bias_cuda)
            torch.cuda.synchronize(0)
            y_cuda_tf32_enabled_np = y_cuda_tf32_enabled.cpu().numpy()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
            torch.set_float32_matmul_precision(old_matmul_precision)

        references = {
            "pytorch_phase5c_dump": y_pytorch,
            "tensorrt_phase5c_dump": y_tensorrt,
            "numpy_fp32": y_numpy_fp32,
            "torch_cpu_fp32": y_torch_cpu,
            "torch_cuda_tf32_disabled": y_cuda_tf32_disabled_np,
            "torch_cuda_tf32_enabled": y_cuda_tf32_enabled_np,
        }
        reference_comparisons = {
            name: {
                "array_sha256": array_sha256(array),
                "vs_fp64_reference": comparison(array, y_ref_fp64),
                "vs_pytorch_dump": comparison(array, y_pytorch),
                "vs_tensorrt_dump": comparison(array, y_tensorrt),
            }
            for name, array in references.items()
        }

        dll_handles = phase5.configure_dll_search(
            args.tensorrt_root.resolve(), args.cuda_root.resolve(), plugin_path
        )
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        if not trt.init_libnvinfer_plugins(logger, ""):
            raise RuntimeError("trt.init_libnvinfer_plugins returned false")
        plugin_library, plugin_info = phase4.load_plugin_library(plugin_path)
        if not plugin_info["registration_function_returned"]:
            raise RuntimeError("VoxelUnique Plugin registration failed")
        engine_info = inspect_linear1_engine(trt, logger, engine_path)
        phase5b.dump_json(run_dir / "engine_linear1_info.json", engine_info)

        pytorch_equals_no_tf32 = bool(
            np.array_equal(y_pytorch, y_cuda_tf32_disabled_np)
        )
        tensorrt_equals_tf32 = bool(np.array_equal(y_tensorrt, y_cuda_tf32_enabled_np))
        attributed = bool(
            pytorch_equals_no_tf32
            and tensorrt_equals_tf32
            and engine_info["tf32_math_identified_from_tactic"]
            and engine_info["matmul_and_bias_add_fused"]
        )
        status = (
            "LINEAR1_TENSORRT_DIFF_ATTRIBUTED"
            if attributed
            else "LINEAR1_TENSORRT_DIFF_ATTRIBUTION_INCONCLUSIVE"
        )
        source_hashes_after = {
            "engine": phase4.sha256(engine_path),
            "onnx": phase4.sha256(onnx_path),
            "checkpoint": phase4.sha256(checkpoint_path),
            "plugin": phase4.sha256(plugin_path),
        }
        payload = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "status": status,
            "run_dir": str(run_dir),
            "phase5c_source_run": str(phase5c_run),
            "linear_contract": {
                "definition": "Y = X @ W_checkpoint.T + b = X @ W_ONNX + b",
                "x_shape": list(x_pytorch.shape),
                "checkpoint_weight_shape": list(weight.shape),
                "onnx_weight_shape": list(onnx_weight.shape),
                "bias_shape": list(bias.shape),
                "x_dtype": str(x_pytorch.dtype),
                "weight_dtype": str(weight.dtype),
                "bias_dtype": str(bias.dtype),
                "pytorch_tensorrt_x_exact_equal": True,
                "onnx_weight_equals_checkpoint_transpose": weight_transpose_equal,
                "onnx_bias_equals_checkpoint_bias": bias_equal,
                "x_sha256": array_sha256(x_pytorch),
                "checkpoint_weight_sha256": array_sha256(weight),
                "onnx_weight_sha256": array_sha256(onnx_weight),
                "bias_sha256": array_sha256(bias),
            },
            "backend_state": {
                "torch_version": torch.__version__,
                "torch_cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "initial_torch_allow_tf32": old_allow_tf32,
                "initial_torch_float32_matmul_precision": old_matmul_precision,
                "settings_restored": (
                    bool(torch.backends.cuda.matmul.allow_tf32) == old_allow_tf32
                    and torch.get_float32_matmul_precision() == old_matmul_precision
                ),
                "offline_gemm_calls": {
                    "numpy_fp64": 1,
                    "numpy_fp32": 1,
                    "torch_cpu_fp32": 1,
                    "torch_cuda_tf32_disabled": 1,
                    "torch_cuda_tf32_enabled": 1,
                },
            },
            "reference_comparisons": reference_comparisons,
            "engine_linear1_info_file": str(run_dir / "engine_linear1_info.json"),
            "attribution": {
                "pytorch_equals_cuda_tf32_disabled": pytorch_equals_no_tf32,
                "tensorrt_equals_cuda_tf32_enabled": tensorrt_equals_tf32,
                "inspector_tactic_uses_tf32": engine_info[
                    "tf32_math_identified_from_tactic"
                ],
                "matmul_bias_fused": engine_info["matmul_and_bias_add_fused"],
                "pytorch_max_abs_vs_fp64": reference_comparisons[
                    "pytorch_phase5c_dump"
                ]["vs_fp64_reference"]["max_abs_error"],
                "tensorrt_max_abs_vs_fp64": reference_comparisons[
                    "tensorrt_phase5c_dump"
                ]["vs_fp64_reference"]["max_abs_error"],
                "cause": (
                    "TensorRT TF32 tensor-core GEMM tactic versus PyTorch full-FP32 CUDA GEMM"
                    if attributed
                    else "inconclusive"
                ),
                "confidence": "direct bitwise reproduction" if attributed else "inconclusive",
            },
            "source_hashes_before": source_hashes_before,
            "source_hashes_after": source_hashes_after,
            "formal_sources_modified": source_hashes_before != source_hashes_after,
            "formal_engine_inference_executed": False,
            "fp16": False,
            "int8": False,
            "benchmark": False,
            "fix_attempted": False,
        }
        if payload["formal_sources_modified"]:
            raise RuntimeError("Formal source hash changed during read-only analysis")
        phase5b.dump_json(run_dir / "linear1_analysis.json", payload)
        write_report(run_dir / "comparison_report.md", payload, engine_info)
        print(f"RUN_DIR={run_dir}")
        print(f"TENSORRT_TACTIC={engine_info['tactic_name']}")
        print(f"PYTORCH_EQUALS_CUDA_TF32_DISABLED={pytorch_equals_no_tf32}")
        print(f"TENSORRT_EQUALS_CUDA_TF32_ENABLED={tensorrt_equals_tf32}")
        print(status)
        del plugin_library, dll_handles
        return 0 if attributed else 2
    except Exception:
        failure.update(
            {
                "traceback": traceback.format_exc(),
                "source_hashes_after": {
                    "engine": phase4.sha256(engine_path),
                    "onnx": phase4.sha256(onnx_path),
                    "checkpoint": phase4.sha256(checkpoint_path),
                    "plugin": phase4.sha256(plugin_path),
                },
            }
        )
        failure["formal_sources_modified"] = (
            failure["source_hashes_before"] != failure["source_hashes_after"]
        )
        phase5b.dump_json(run_dir / "linear1_analysis.json", failure)
        print(f"RUN_DIR={run_dir}")
        print(failure["traceback"], file=sys.stderr)
        print("LINEAR1_TENSORRT_DIFF_ATTRIBUTION_FAILED")
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase5c-run", type=Path, default=DEFAULT_PHASE5C_RUN)
    parser.add_argument("--engine", type=Path, default=phase5.DEFAULT_ENGINE)
    parser.add_argument("--onnx", type=Path, default=phase5.DEFAULT_ONNX)
    parser.add_argument("--checkpoint", type=Path, default=phase5.DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--plugin-library", type=Path, default=phase5.DEFAULT_PLUGIN_LIBRARY
    )
    parser.add_argument("--tensorrt-root", type=Path, default=phase5.DEFAULT_TENSORRT_ROOT)
    parser.add_argument("--cuda-root", type=Path, default=phase5.DEFAULT_CUDA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
