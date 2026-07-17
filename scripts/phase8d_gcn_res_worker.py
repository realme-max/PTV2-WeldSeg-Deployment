"""Fresh-process worker for Phase 8D cold start, benchmark, regression and soak tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for item in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import gcn_res_tensorrt_phase8d_common as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("cold-start", "benchmark", "regression", "determinism", "soak"))
    parser.add_argument("--runtime", choices=("candidate", "baseline", "pytorch"), default="candidate")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-order", default="")
    parser.add_argument("--sample-id", default="weld_65")
    parser.add_argument("--samples", default="weld_65,weld_5,weld_14")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--soak-iterations", type=int, default=5000)
    parser.add_argument("--reference-dir", type=Path)
    return parser.parse_args()


def runner_for(args: argparse.Namespace) -> Any:
    if args.runtime == "pytorch":
        return common.PyTorchRunner()
    if args.engine is None or args.plugin is None:
        raise ValueError("TensorRT worker requires --engine and --plugin")
    return common.TensorRTRunner(args.engine, args.plugin, args.runtime)


def ordered_records(args: argparse.Namespace, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    by_name = {item["sample_id"]: item for item in manifest["samples"]}
    requested = [item for item in args.sample_order.split(",") if item]
    names = requested or [item["sample_id"] for item in manifest["samples"]]
    if len(names) != len(set(names)) or set(names) != set(by_name):
        raise ValueError("--sample-order must contain every frozen test sample exactly once")
    return [by_name[name] for name in names]


def cold_start(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    record = next(item for item in manifest["samples"] if item["sample_id"] == args.sample_id)
    points, adj, _ = common.load_frozen_sample(args.input_dir, record)
    total_started = time.perf_counter()
    runner = None
    try:
        runner = runner_for(args)
        first_started = time.perf_counter()
        logits = runner.infer(points, adj)
        first_ms = (time.perf_counter() - first_started) * 1000.0
        return {
            "status": "PASS",
            "pid": os.getpid(),
            "sample_id": args.sample_id,
            "plugin_load_ms": float(runner.plugin_load_ms),
            "engine_deserialize_ms": float(runner.deserialize_ms),
            "context_creation_ms": float(runner.context_creation_ms),
            "first_inference_ms": first_ms,
            "total_startup_ms": (time.perf_counter() - total_started) * 1000.0,
            "output_sha256": common.array_sha256(logits),
            "predicted_labels_sha256": common.array_sha256(np.argmax(logits, axis=-1).astype(np.int64)),
            "output_finite": bool(np.isfinite(logits).all()),
            "error_recorder_errors": int(runner.recorder.num_errors),
            "runtime_plugin_instances": int(runner.runtime_plugin_instances),
            "io": runner.io,
        }
    finally:
        if runner is not None:
            runner.close()


def benchmark(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    records = ordered_records(args, manifest)
    runner = runner_for(args)
    per_sample: list[dict[str, Any]] = []
    all_pure: list[float] = []
    all_e2e: list[float] = []
    try:
        for record in records:
            points, adj, _ = common.load_frozen_sample(args.input_dir, record)
            measured = runner.benchmark(points, adj, args.warmup, args.iterations)
            all_pure.extend(measured["pure_samples_ms"])
            all_e2e.extend(measured["e2e_samples_ms"])
            per_sample.append({
                "sample_id": record["sample_id"],
                "points_sha256": record["points_sha256"],
                "adj_sha256": record["adj_sha256"],
                "pure": common.latency_statistics(measured["pure_samples_ms"]),
                "e2e": common.latency_statistics(measured["e2e_samples_ms"]),
                "pure_samples_ms": measured["pure_samples_ms"],
                "e2e_samples_ms": measured["e2e_samples_ms"],
                "output_sha256": measured["output_sha256"],
            })
        return {
            "status": "PASS",
            "runtime": args.runtime,
            "pid": os.getpid(),
            "warmup_per_sample": args.warmup,
            "measurements_per_sample": args.iterations,
            "sample_order": [item["sample_id"] for item in records],
            "per_sample": per_sample,
            "aggregate": {
                "pure": common.latency_statistics(all_pure),
                "e2e": common.latency_statistics(all_e2e),
                "sample_macro_pure_mean_ms": float(np.mean([item["pure"]["mean"] for item in per_sample])),
                "sample_macro_e2e_mean_ms": float(np.mean([item["e2e"]["mean"] for item in per_sample])),
            },
        }
    finally:
        runner.close()


def regression(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    records = [item for item in manifest["samples"]]
    output_dir = args.output_dir or args.output.parent / f"{args.runtime}_logits"
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = runner_for(args)
    results: list[dict[str, Any]] = []
    try:
        for record in records:
            points, adj, labels = common.load_frozen_sample(args.input_dir, record)
            logits = runner.infer(points, adj)
            prediction = np.argmax(logits, axis=-1).astype(np.int64)
            logits_path = output_dir / f"{record['sample_id']}_logits.npy"
            labels_path = output_dir / f"{record['sample_id']}_prediction.npy"
            np.save(logits_path, logits, allow_pickle=False)
            np.save(labels_path, prediction, allow_pickle=False)
            results.append({
                "sample_id": record["sample_id"],
                "logits_path": str(logits_path.resolve()),
                "prediction_path": str(labels_path.resolve()),
                "logits_sha256": common.array_sha256(logits),
                "prediction_sha256": common.array_sha256(prediction),
                "finite": bool(np.isfinite(logits).all()),
                "metrics": common.segmentation_metrics(labels, prediction),
            })
        return {"status": "PASS", "runtime": args.runtime, "total_samples": len(results), "samples": results}
    finally:
        runner.close()


def determinism(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    by_name = {item["sample_id"]: item for item in manifest["samples"]}
    names = [item for item in args.samples.split(",") if item]
    runner = runner_for(args)
    samples: list[dict[str, Any]] = []
    try:
        for name in names:
            points, adj, _ = common.load_frozen_sample(args.input_dir, by_name[name])
            outputs: list[np.ndarray] = []
            records: list[dict[str, Any]] = []
            for iteration in range(args.iterations):
                logits = runner.infer(points, adj)
                outputs.append(logits)
                records.append({
                    "iteration": iteration,
                    "logits_sha256": common.array_sha256(logits),
                    "labels_sha256": common.array_sha256(np.argmax(logits, axis=-1).astype(np.int64)),
                    "min": float(logits.min()), "max": float(logits.max()), "mean": float(logits.mean()),
                    "finite": bool(np.isfinite(logits).all()),
                })
            reference = outputs[0]
            max_repeat_error = max(float(np.max(np.abs(item.astype(np.float64) - reference.astype(np.float64)))) for item in outputs)
            samples.append({
                "sample_id": name,
                "iterations": args.iterations,
                "bitwise_logits_stable": len({item["logits_sha256"] for item in records}) == 1,
                "labels_stable": len({item["labels_sha256"] for item in records}) == 1,
                "max_repeat_abs_error": max_repeat_error,
                "runs": records,
            })
        all_logits = all(item["bitwise_logits_stable"] for item in samples)
        all_labels = all(item["labels_stable"] for item in samples)
        return {
            "status": "DETERMINISTIC_LOGITS_CONFIRMED" if all_logits else ("DETERMINISTIC_LABELS_ONLY" if all_labels else "DETERMINISM_FAILED"),
            "runtime": args.runtime,
            "samples": samples,
        }
    finally:
        runner.close()


def soak(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    if args.reference_dir is None:
        raise ValueError("soak requires --reference-dir")
    runner = runner_for(args)
    samples = manifest["samples"]
    snapshots: list[dict[str, Any]] = []
    latencies: list[float] = []
    try:
        import run_gcn_res_tensorrt_fp32_inference as phase5

        start_event = phase5.cuda_call(runner.cudart, "cudaEventCreate(start)", runner.cudart.cudaEventCreate())[0]
        stop_event = phase5.cuda_call(runner.cudart, "cudaEventCreate(stop)", runner.cudart.cudaEventCreate())[0]
        for iteration in range(args.soak_iterations):
            record = samples[iteration % len(samples)]
            points, adj, _ = common.load_frozen_sample(args.input_dir, record)
            expected = np.load(args.reference_dir / f"{record['sample_id']}_prediction.npy", allow_pickle=False)
            runner.copy_inputs(points, adj); runner.synchronize()
            phase5.cuda_call(runner.cudart, "record start", runner.cudart.cudaEventRecord(start_event, runner.stream))
            runner.enqueue()
            phase5.cuda_call(runner.cudart, "record stop", runner.cudart.cudaEventRecord(stop_event, runner.stream))
            runner.copy_output(); runner.synchronize()
            latency = float(phase5.cuda_call(runner.cudart, "elapsed", runner.cudart.cudaEventElapsedTime(start_event, stop_event))[0])
            latencies.append(latency)
            logits = np.ascontiguousarray(runner.host_logits.copy(), dtype=np.float32)
            prediction = np.argmax(logits, axis=-1).astype(np.int64)
            if not np.isfinite(logits).all() or not np.array_equal(prediction, expected):
                raise RuntimeError(f"Soak output regression at iteration {iteration}, sample {record['sample_id']}")
            if (iteration + 1) % 100 == 0:
                snapshots.append({
                    "iteration": iteration + 1,
                    "sample_id": record["sample_id"],
                    "memory": runner.memory_info(),
                    "rolling_latency_mean_ms": float(np.mean(latencies[-100:])),
                    "error_recorder_errors": int(runner.recorder.num_errors),
                    "telemetry": common.gpu_telemetry(),
                })
        phase5.cuda_call(runner.cudart, "destroy start", runner.cudart.cudaEventDestroy(start_event))
        phase5.cuda_call(runner.cudart, "destroy stop", runner.cudart.cudaEventDestroy(stop_event))
        free_values = [item["memory"]["free_bytes"] for item in snapshots]
        rolling = [item["rolling_latency_mean_ms"] for item in snapshots]
        memory_slope = float(np.polyfit(np.arange(len(free_values)), np.asarray(free_values, dtype=np.float64), 1)[0]) if len(free_values) > 1 else 0.0
        latency_ratio = float(np.mean(rolling[-10:]) / np.mean(rolling[:10])) if len(rolling) >= 20 else 1.0
        return {
            "status": "PASS",
            "iterations": args.soak_iterations,
            "successful_enqueues": args.soak_iterations,
            "latency": common.latency_statistics(latencies),
            "snapshots": snapshots,
            "free_memory_linear_slope_bytes_per_snapshot": memory_slope,
            "last10_vs_first10_rolling_latency_ratio": latency_ratio,
            "monotonic_memory_growth_detected": bool(all(a > b for a, b in zip(free_values, free_values[1:])) and free_values[-1] < free_values[0]),
            "obvious_latency_degradation_detected": bool(latency_ratio > 1.25),
            "error_recorder_errors": int(runner.recorder.num_errors),
        }
    finally:
        runner.close()


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.input_manifest = args.input_manifest.resolve()
    args.output = args.output.resolve()
    manifest = common.load_json(args.input_manifest)
    failure: dict[str, Any] = {"status": "FAILED", "mode": args.mode, "runtime": args.runtime}
    common.dump_json(args.output, failure)
    try:
        if args.mode == "cold-start":
            result = cold_start(args, manifest)
        elif args.mode == "benchmark":
            result = benchmark(args, manifest)
        elif args.mode == "regression":
            result = regression(args, manifest)
        elif args.mode == "determinism":
            result = determinism(args, manifest)
        elif args.mode == "soak":
            result = soak(args, manifest)
        else:
            raise AssertionError(args.mode)
        result.update({"mode": args.mode, "completed_at": common.now_iso()})
        common.dump_json(args.output, result)
        print(json.dumps({"status": result["status"], "output": str(args.output)}, ensure_ascii=False))
        return 0
    except Exception as exc:
        failure.update({"exception_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        common.dump_json(args.output, failure)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
