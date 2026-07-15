"""Profile one warmup and one measured GCN_res ONNX Runtime CUDA inference."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import onnxruntime as ort


DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_onnx"
    / "20260715_onnx_after_cdist_fp32_opset18"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def analyze_profile(profile_path: Path) -> dict[str, Any]:
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise RuntimeError(f"Unexpected ORT profile format: {type(events)!r}")

    nodes: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict) or "dur" not in event:
            continue
        args = event.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        op_type = args.get("op_name") or args.get("op_type")
        provider = args.get("provider") or args.get("execution_provider")
        category = str(event.get("cat", ""))
        name = str(event.get("name", ""))
        if category != "Node" and op_type is None and not name.endswith("_kernel_time"):
            continue
        duration_us = float(event.get("dur", 0.0))
        nodes.append(
            {
                "name": name,
                "op_type": str(op_type or "UNKNOWN"),
                "provider": str(provider or "UNKNOWN"),
                "duration_us": duration_us,
                "duration_ms": duration_us / 1000.0,
                "timestamp_us": float(event.get("ts", 0.0)),
            }
        )

    nodes.sort(key=lambda item: item["duration_us"], reverse=True)
    focus_tokens = (
        "cdist",
        "gather",
        "scatter",
        "reduce",
        "expand",
        "matmul",
        "shape",
    )
    focus = [
        node
        for node in nodes
        if any(
            token in f"{node['name']} {node['op_type']}".lower()
            for token in focus_tokens
        )
    ]
    by_op: dict[str, dict[str, float | int]] = {}
    for node in nodes:
        record = by_op.setdefault(
            node["op_type"], {"count": 0, "total_duration_ms": 0.0, "max_duration_ms": 0.0}
        )
        record["count"] = int(record["count"]) + 1
        record["total_duration_ms"] = float(record["total_duration_ms"]) + node["duration_ms"]
        record["max_duration_ms"] = max(float(record["max_duration_ms"]), node["duration_ms"])

    chronological = sorted(nodes, key=lambda item: item["timestamp_us"])
    return {
        "event_count": len(events),
        "node_event_count": len(nodes),
        "top_20_nodes": nodes[:20],
        "focus_nodes_top_50": focus[:50],
        "op_type_summary": dict(
            sorted(
                by_op.items(),
                key=lambda item: float(item[1]["total_duration_ms"]),
                reverse=True,
            )
        ),
        "last_completed_node": chronological[-1] if chronological else None,
    }


class PhaseTimeout(RuntimeError):
    """Raised after cooperatively terminating an ORT run that exceeded its limit."""


def run_with_timeout(
    session: ort.InferenceSession,
    feed: dict[str, np.ndarray],
    timeout_seconds: float,
) -> np.ndarray:
    run_options = ort.RunOptions()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(session.run, ["logits"], feed, run_options)
    try:
        return np.asarray(future.result(timeout=timeout_seconds)[0], dtype=np.float32)
    except concurrent.futures.TimeoutError as exc:
        run_options.terminate = True
        try:
            future.result(timeout=60.0)
        except BaseException:
            pass
        raise PhaseTimeout(
            f"ORT inference exceeded {timeout_seconds:.3f} seconds and was terminated"
        ) from exc
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    model_path = source_dir / "gcn_res_deploy_fp32_opset18.onnx"
    input_path = source_dir / "export_input.npz"
    for path in (model_path, input_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(input_path) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        adjacency = np.asarray(data["adj"], dtype=np.float32)
    if points.shape != (1, 2048, 4) or adjacency.shape != (1, 2048, 2048):
        raise RuntimeError(f"Unexpected input shapes: {points.shape}, {adjacency.shape}")

    summary: dict[str, Any] = {
        "status": "RUNNING",
        "started_at": datetime.now().astimezone().isoformat(),
        "source_model": str(model_path),
        "source_input": str(input_path),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "onnxruntime_version": ort.__version__,
        "requested_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "phase_timeout_seconds": args.phase_timeout_seconds,
        "warmup_elapsed_seconds": None,
        "inference_elapsed_seconds": None,
        "active_phase": "session_creation",
    }
    save_json(output_dir / "profile_run_summary.json", summary)

    session: ort.InferenceSession | None = None
    profile_path: Path | None = None
    exit_code = 1
    phase_started = time.perf_counter()
    try:
        ort.preload_dlls()
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(output_dir / "ort_cuda_profile")
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        providers = session.get_providers()
        summary["session_providers"] = providers
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"Refusing CPU fallback: {providers}")
        print(f"SESSION_PROVIDERS={providers}", flush=True)

        feed = {"points": points, "adj": adjacency}
        summary["active_phase"] = "warmup"
        phase_started = time.perf_counter()
        run_with_timeout(session, feed, args.phase_timeout_seconds)
        summary["warmup_elapsed_seconds"] = time.perf_counter() - phase_started
        print(f"WARMUP_ELAPSED_SECONDS={summary['warmup_elapsed_seconds']:.9f}", flush=True)

        summary["active_phase"] = "measured_inference"
        phase_started = time.perf_counter()
        output = run_with_timeout(session, feed, args.phase_timeout_seconds)
        summary["inference_elapsed_seconds"] = time.perf_counter() - phase_started
        summary["output_shape"] = list(output.shape)
        summary["output_finite"] = bool(np.isfinite(output).all())
        summary["status"] = "PROFILE_RUN_COMPLETED"
        exit_code = 0
        print(f"INFERENCE_ELAPSED_SECONDS={summary['inference_elapsed_seconds']:.9f}", flush=True)
    except KeyboardInterrupt:
        summary["status"] = "PROFILE_RUN_INTERRUPTED"
        summary["interrupted_phase"] = summary.get("active_phase")
        summary["interrupted_phase_elapsed_seconds"] = time.perf_counter() - phase_started
        summary["error"] = "KeyboardInterrupt"
        exit_code = 130
    except PhaseTimeout as exc:
        summary["status"] = "PROFILE_RUN_TIMEOUT"
        summary["timeout_phase"] = summary.get("active_phase")
        summary["timeout_phase_elapsed_seconds"] = time.perf_counter() - phase_started
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        exit_code = 124
    except BaseException as exc:
        summary["status"] = "PROFILE_RUN_FAILED"
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        summary["traceback"] = traceback.format_exc()
        exit_code = 1
    finally:
        if session is not None:
            try:
                profile_path = Path(session.end_profiling()).resolve()
                summary["profile_json"] = str(profile_path)
                print(f"PROFILE_JSON={profile_path}", flush=True)
            except BaseException as exc:
                summary["profile_finalize_error"] = f"{type(exc).__name__}: {exc}"
        summary["finished_at"] = datetime.now().astimezone().isoformat()
        if profile_path is not None and profile_path.is_file():
            try:
                analysis = analyze_profile(profile_path)
                summary["profile_analysis"] = analysis
                save_json(output_dir / "profile_analysis.json", analysis)
            except BaseException as exc:
                summary["profile_analysis_error"] = f"{type(exc).__name__}: {exc}"
        save_json(output_dir / "profile_run_summary.json", summary)

    print(f"PROFILE_RUN_STATUS={summary['status']}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
