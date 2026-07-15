"""Read-only TensorRT ONNX parser audit for the fixed GCN_res deployment graph.

This script never creates a builder configuration, builds an engine, or runs
inference. A child process is used so native TensorRT VERBOSE output can be
captured completely in parser_verbose.log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
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
FOCUS_OPERATORS = (
    "Unique",
    "NonZero",
    "ScatterElements",
    "GatherND",
    "GatherElements",
    "TopK",
    "ReduceMin",
    "ReduceMax",
    "Shape",
    "Expand",
)


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape_to_list(shape: Any) -> list[int | str]:
    values: list[int | str] = []
    for dim in shape:
        try:
            values.append(int(dim))
        except (TypeError, ValueError):
            values.append(str(dim))
    return values


def _tensor_info(tensor: Any) -> dict[str, Any]:
    return {
        "name": tensor.name,
        "shape": _shape_to_list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def _classify_parser_error(error: dict[str, Any]) -> str:
    code = str(error.get("error_code", "")).lower()
    operator = str(error.get("operator", "")).lower()
    description = str(error.get("description", "")).lower()
    combined = " ".join((code, operator, description))

    if "plugin" in combined:
        return "PLUGIN_REQUIRED"
    if "data-dependent" in combined or "data dependent" in combined:
        return "DATA_DEPENDENT_SHAPE"
    if "shape tensor" in combined:
        return "SHAPE_TENSOR_ERROR"
    if "dynamic shape" in combined or "dynamic dimensions" in combined:
        return "DYNAMIC_SHAPE_ERROR"
    if "attribute" in combined and (
        "unsupported" in combined or "not supported" in combined
    ):
        return "UNSUPPORTED_ATTRIBUTE"
    if any(
        marker in combined
        for marker in (
            "unsupported operator",
            "unsupported op",
            "no importer registered",
            "not supported",
            "unsupported_node",
        )
    ):
        return "UNSUPPORTED_OPERATOR"
    return "UNKNOWN"


def _error_value(error: Any, name: str, default: Any = None) -> Any:
    if not hasattr(error, name):
        return default
    value = getattr(error, name)
    try:
        return value() if callable(value) else value
    except Exception as exc:  # Preserve API extraction failures in the audit.
        return f"ERROR_READING_{name}: {type(exc).__name__}: {exc}"


def _collect_parser_errors(parser: Any) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for index in range(parser.num_errors):
        error = parser.get_error(index)
        item = {
            "index": index,
            "error_code": str(_error_value(error, "code", "")),
            "node_index": _error_value(error, "node", -1),
            "node_name": _error_value(error, "node_name", ""),
            "operator": _error_value(error, "node_operator", ""),
            "description": _error_value(error, "desc", str(error)),
            "source_file": _error_value(error, "file", ""),
            "source_function": _error_value(error, "func", ""),
            "source_line": _error_value(error, "line", -1),
            "raw": str(error),
        }
        item["classification"] = _classify_parser_error(item)
        errors.append(item)
    return errors


def _onnx_inventory(onnx_path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load_model(str(onnx_path), load_external_data=False)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return {
        "ir_version": int(model.ir_version),
        "opset_imports": [
            {"domain": item.domain, "version": int(item.version)}
            for item in model.opset_import
        ],
        "num_onnx_nodes": len(model.graph.node),
        "operator_counts": dict(sorted(counts.items())),
        "focus_operator_counts": {
            name: counts.get(name, 0) for name in FOCUS_OPERATORS
        },
    }


def _nvcc_version() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "return_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except FileNotFoundError as exc:
        return {"return_code": None, "stdout": "", "stderr": str(exc)}


def _worker(args: argparse.Namespace) -> int:
    onnx_path = Path(args.onnx).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    import onnx
    import tensorrt as trt
    import torch

    environment = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "tensorrt_version": trt.__version__,
        "onnx_version": onnx.__version__,
        "pytorch_version": torch.__version__,
        "pytorch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_toolkit": _nvcc_version(),
        "gpu_name": torch.cuda.get_device_name(args.device),
        "compute_capability": list(torch.cuda.get_device_capability(args.device)),
        "gpu_index": args.device,
        "onnx_path": str(onnx_path),
        "onnx_file_size_bytes": onnx_path.stat().st_size,
        "onnx_sha256": _sha256(onnx_path),
        "tensorrt_root": args.tensorrt_root,
        "cuda_root": args.cuda_root,
    }
    _json_dump(run_dir / "environment.json", environment)

    inventory = _onnx_inventory(onnx_path)
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    if builder is None:
        raise RuntimeError("trt.Builder returned None")

    explicit_batch_flag = getattr(
        trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None
    )
    if explicit_batch_flag is None:
        # TensorRT 10+ removed EXPLICIT_BATCH because explicit batch is mandatory.
        network_flags = 0
        explicit_batch_api = "implicit_explicit_batch_default_in_tensorrt_11"
    else:
        network_flags = 1 << int(explicit_batch_flag)
        explicit_batch_api = "EXPLICIT_BATCH_flag"

    network = builder.create_network(network_flags)
    if network is None:
        raise RuntimeError("builder.create_network returned None")
    parser = trt.OnnxParser(network, logger)

    print(f"ONNX_PATH={onnx_path}", flush=True)
    print(f"TENSORRT_VERSION={trt.__version__}", flush=True)
    print(f"NETWORK_FLAGS={network_flags}", flush=True)
    print(f"EXPLICIT_BATCH_API={explicit_batch_api}", flush=True)
    print("PARSER_PARSE_FROM_FILE_BEGIN", flush=True)
    parser_success = bool(parser.parse_from_file(str(onnx_path)))
    print(f"PARSER_PARSE_FROM_FILE_END success={parser_success}", flush=True)

    errors = _collect_parser_errors(parser)
    inputs = [_tensor_info(network.get_input(i)) for i in range(network.num_inputs)]
    outputs = [
        _tensor_info(network.get_output(i)) for i in range(network.num_outputs)
    ]

    expected_inputs = {
        "points": [1, 2048, 4],
        "adj": [1, 2048, 2048],
    }
    expected_outputs = {"logits": [1, 2048, 2]}
    input_map = {item["name"]: item["shape"] for item in inputs}
    output_map = {item["name"]: item["shape"] for item in outputs}
    io_matches_expected = (
        input_map == expected_inputs and output_map == expected_outputs
    )

    unsupported_ops = sorted(
        {
            item["operator"]
            for item in errors
            if item["operator"]
            and item["classification"]
            in {"UNSUPPORTED_OPERATOR", "PLUGIN_REQUIRED"}
        }
    )
    summary = {
        "parser_success": parser_success,
        "num_parser_errors": len(errors),
        "num_layers": network.num_layers,
        "num_inputs": network.num_inputs,
        "num_outputs": network.num_outputs,
        "inputs": inputs,
        "outputs": outputs,
        "io_matches_expected": io_matches_expected,
        "unsupported_ops": unsupported_ops,
        "error_classification_counts": {
            name: sum(item["classification"] == name for item in errors)
            for name in (
                "UNSUPPORTED_OPERATOR",
                "UNSUPPORTED_ATTRIBUTE",
                "DYNAMIC_SHAPE_ERROR",
                "SHAPE_TENSOR_ERROR",
                "DATA_DEPENDENT_SHAPE",
                "PLUGIN_REQUIRED",
                "UNKNOWN",
            )
        },
        "network_creation_flags": network_flags,
        "explicit_batch_api": explicit_batch_api,
        "onnx_inventory": inventory,
        "engine_build_attempted": False,
        "inference_attempted": False,
    }
    _json_dump(run_dir / "parser_errors.json", errors)
    _json_dump(run_dir / "parser_summary.json", summary)

    for item in inputs:
        print(
            f"INPUT name={item['name']} shape={item['shape']} dtype={item['dtype']}",
            flush=True,
        )
    for item in outputs:
        print(
            f"OUTPUT name={item['name']} shape={item['shape']} dtype={item['dtype']}",
            flush=True,
        )
    for item in errors:
        print(
            "PARSER_ERROR "
            f"index={item['index']} code={item['error_code']} "
            f"node={item['node_name']} operator={item['operator']} "
            f"classification={item['classification']} "
            f"description={item['description']}",
            flush=True,
        )

    del parser
    del network
    del builder
    return 0 if parser_success else 2


def _run_trtexec_help(
    trtexec_path: Path, run_dir: Path, child_env: dict[str, str]
) -> dict[str, Any]:
    result = subprocess.run(
        [str(trtexec_path), "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=child_env,
    )
    text = result.stdout.decode("utf-8", errors="replace")
    (run_dir / "trtexec_help.log").write_text(text, encoding="utf-8")
    lower = text.lower()
    parser_only_available = any(
        marker in lower for marker in ("--parseonly", "--parseronly", "--parse-only")
    )
    return {
        "help_return_code": result.returncode,
        "parser_only_available": parser_only_available,
        "skip_inference_available": "--skipinference" in lower,
        "onnx_invocation_attempted": False,
        "reason_onnx_not_invoked": (
            "No true parser-only trtexec option is exposed. "
            "--skipInference exits only after an engine has been built."
        ),
    }


def _main(args: argparse.Namespace) -> int:
    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    output_root = Path(args.output_root).resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f_parser_audit")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    child_env = os.environ.copy()
    path_prefixes: list[str] = []
    trtexec_path: Path | None = None
    if args.tensorrt_root:
        trt_bin = Path(args.tensorrt_root).resolve() / "bin"
        path_prefixes.append(str(trt_bin))
        candidate = trt_bin / "trtexec.exe"
        if candidate.is_file():
            trtexec_path = candidate
    if args.cuda_root:
        path_prefixes.append(str(Path(args.cuda_root).resolve() / "bin"))
    child_env["PATH"] = os.pathsep.join(path_prefixes + [child_env.get("PATH", "")])
    child_env["PYTHONUNBUFFERED"] = "1"

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--onnx",
        str(onnx_path),
        "--run-dir",
        str(run_dir),
        "--device",
        str(args.device),
        "--tensorrt-root",
        args.tensorrt_root,
        "--cuda-root",
        args.cuda_root,
    ]
    log_path = run_dir / "parser_verbose.log"
    with log_path.open("wb") as log_stream:
        result = subprocess.run(
            command,
            check=False,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            env=child_env,
        )

    trtexec_audit: dict[str, Any]
    if trtexec_path is not None:
        trtexec_audit = _run_trtexec_help(trtexec_path, run_dir, child_env)
    else:
        trtexec_audit = {
            "help_return_code": None,
            "parser_only_available": False,
            "skip_inference_available": False,
            "onnx_invocation_attempted": False,
            "reason_onnx_not_invoked": "trtexec.exe was not found under TensorRT root",
        }

    summary_path = run_dir / "parser_summary.json"
    if not summary_path.is_file():
        print(f"RUN_DIR={run_dir}")
        print(f"PARSER_WORKER_EXIT_CODE={result.returncode}")
        print(f"PARSER_VERBOSE_LOG={log_path}")
        print("TENSORRT_ONNX_PARSER_FAILED")
        return result.returncode or 3

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["trtexec_audit"] = trtexec_audit
    _json_dump(summary_path, summary)

    print(f"RUN_DIR={run_dir}")
    print(f"PARSER_WORKER_EXIT_CODE={result.returncode}")
    print(f"PARSER_VERBOSE_LOG={log_path}")
    for item in summary["inputs"]:
        print(f"INPUT {item['name']} shape={item['shape']} dtype={item['dtype']}")
    for item in summary["outputs"]:
        print(f"OUTPUT {item['name']} shape={item['shape']} dtype={item['dtype']}")
    print(f"NUM_LAYERS={summary['num_layers']}")
    print(f"NUM_PARSER_ERRORS={summary['num_parser_errors']}")
    print(f"IO_MATCHES_EXPECTED={summary['io_matches_expected']}")
    print(
        "TRTEXEC_TRUE_PARSER_ONLY_AVAILABLE="
        f"{trtexec_audit['parser_only_available']}"
    )
    if summary["parser_success"]:
        print("TENSORRT_ONNX_PARSER_PASSED")
        return 0
    print("TENSORRT_ONNX_PARSER_FAILED")
    return 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tensorrt-root", default=os.environ.get("TENSORRT_ROOT", ""))
    parser.add_argument("--cuda-root", default=os.environ.get("CUDA_PATH", ""))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker and not args.run_dir:
        parser.error("--worker requires --run-dir")
    return args


if __name__ == "__main__":
    parsed_args = _parse_args()
    try:
        exit_code = _worker(parsed_args) if parsed_args.worker else _main(parsed_args)
    except Exception as exc:
        print(f"FATAL_ERROR={type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    raise SystemExit(exit_code)
