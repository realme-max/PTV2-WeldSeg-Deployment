"""Fail-closed production entry point for the promoted GCN_res TensorRT package."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for item in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import gcn_res_tensorrt_phase8d_common as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--adj", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification-mode", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_artifact(package_root: Path, manifest: dict, key: str) -> Path:
    value = manifest.get("artifacts", {}).get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest.artifacts.{key} is missing")
    path = (package_root / value).resolve()
    try:
        path.relative_to(package_root.resolve())
    except ValueError as exc:
        raise ValueError(f"manifest artifact escapes package: {key}={value}") from exc
    return path


def main() -> int:
    args = parse_args()
    summary_path = args.output.with_suffix(args.output.suffix + ".json")
    failure = {"runtime_status": "FAILED", "inference_executed": False}
    common.dump_json(summary_path, failure)
    runner = None
    dll_handles = []
    try:
        manifest_path = args.manifest.resolve()
        manifest = common.load_json(manifest_path)
        allowed = {"production_baseline"}
        if args.qualification_mode:
            allowed.add("qualification_pending")
        if manifest.get("status") not in allowed:
            raise RuntimeError(f"Manifest status is not runnable: {manifest.get('status')!r}")
        package_root = manifest_path.parent.parent
        engine = resolve_artifact(package_root, manifest, "engine")
        plugin = resolve_artifact(package_root, manifest, "plugin")
        onnx = resolve_artifact(package_root, manifest, "onnx")
        for label, path, expected in (
            ("engine", engine, manifest.get("engine_sha256")),
            ("plugin", plugin, manifest.get("plugin_sha256")),
            ("onnx", onnx, manifest.get("onnx_sha256")),
        ):
            if not isinstance(expected, str):
                raise ValueError(f"Missing {label}_sha256")
            common.assert_hash(path, expected, label)

        # TensorRT's Python package imports nvinfer_11.dll immediately on Windows.
        # Add only the declared SDK/CUDA/package-plugin directories before that import;
        # the custom creator itself is still loaded explicitly by TensorRTRunner below.
        dll_handles = common.cub_common.configure_dll_search(plugin)
        import torch
        import tensorrt as trt

        compatibility = manifest.get("compatibility", {})
        required_trt = str(compatibility.get("tensorrt", ""))
        if trt.__version__ != required_trt:
            raise RuntimeError(f"TensorRT version mismatch: {trt.__version__} != {required_trt}")
        actual_cc = ".".join(str(item) for item in torch.cuda.get_device_capability(0))
        required_cc = str(compatibility.get("compute_capability", ""))
        if actual_cc != required_cc:
            raise RuntimeError(f"GPU compute capability mismatch: {actual_cc} != {required_cc}")

        points = np.load(args.points.resolve(), allow_pickle=False)
        adj = np.load(args.adj.resolve(), allow_pickle=False)
        expected_points = tuple(manifest["input_contract"]["points"])
        expected_adj = tuple(manifest["input_contract"]["adj"])
        if points.dtype != np.float32:
            raise TypeError(f"points dtype mismatch: {points.dtype} != float32")
        if adj.dtype != np.float32:
            raise TypeError(f"adj dtype mismatch: {adj.dtype} != float32")
        if points.shape != expected_points:
            raise ValueError(f"points shape mismatch: {points.shape} != {expected_points}")
        if adj.shape != expected_adj:
            raise ValueError(f"adj shape mismatch: {adj.shape} != {expected_adj}")
        if not np.isfinite(points).all() or not np.isfinite(adj).all():
            raise FloatingPointError("Input contains NaN/Inf")
        points = np.ascontiguousarray(points)
        adj = np.ascontiguousarray(adj)

        runner = common.TensorRTRunner(engine, plugin, "candidate")
        logits = runner.infer(points, adj)
        expected_output = tuple(manifest["output_contract"]["logits"])
        if logits.shape != expected_output:
            raise RuntimeError(f"logits shape mismatch: {logits.shape} != {expected_output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, logits, allow_pickle=False)
        summary = {
            "deployment_id": manifest["deployment_id"],
            "runtime_status": "PASS",
            "inference_executed": True,
            "engine_sha256": common.sha256(engine),
            "plugin_sha256": common.sha256(plugin),
            "points_sha256": common.array_sha256(points),
            "adj_sha256": common.array_sha256(adj),
            "output_path": str(args.output.resolve()),
            "output_shape": list(logits.shape),
            "output_dtype": str(logits.dtype),
            "output_sha256": common.array_sha256(logits),
            "predicted_labels_sha256": common.array_sha256(np.argmax(logits, axis=-1).astype(np.int64)),
            "output_finite": bool(np.isfinite(logits).all()),
            "error_recorder_errors": int(runner.recorder.num_errors),
            "runtime_plugin_instances": int(runner.runtime_plugin_instances),
        }
        common.dump_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False))
        print("GCN_RES_TENSORRT_PRODUCTION_INFERENCE_PASSED")
        return 0
    except Exception as exc:
        failure.update({"exception_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        common.dump_json(summary_path, failure)
        print(f"PRODUCTION_INFERENCE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if runner is not None:
            runner.close()
        for handle in dll_handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
