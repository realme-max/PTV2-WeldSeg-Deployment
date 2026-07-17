"""Benchmark strict-FP32 PyTorch CUDA forward latency on fixed weld_65 input."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_gcn_res_tensorrt_fp32_inference as phase5  # noqa: E402
import smoke_test_gcn_res_tensorrt_engine as phase7a  # noqa: E402


WARMUP_ITERATIONS = 100
BENCHMARK_ITERATIONS = 1000
DEFAULT_CHECKPOINT = phase7a.DEFAULT_CHECKPOINT
DEFAULT_PHASE6_RESULTS = phase7a.DEFAULT_PHASE6_RESULTS
EXPECTED_CHECKPOINT_SHA256 = phase7a.EXPECTED_CHECKPOINT_SHA256


def dump_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def latency_statistics(samples_ms: list[float]) -> dict[str, Any]:
    values = np.asarray(samples_ms, dtype=np.float64)
    if values.shape != (BENCHMARK_ITERATIONS,) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid latency vector: shape={values.shape}")
    return {
        "unit": "ms",
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "raw_samples_ms": [float(value) for value in values],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--phase6-results", type=Path, default=DEFAULT_PHASE6_RESULTS)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    parser.add_argument("--iterations", type=int, default=BENCHMARK_ITERATIONS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "pytorch_latency.json"
    failure: dict[str, Any] = {
        "status": "TENSORRT_LATENCY_BENCHMARK_FAILED",
        "component": "pytorch_cuda_latency",
    }
    dump_json(output_path, failure)
    old_settings: dict[str, Any] = {}
    try:
        if args.warmup != WARMUP_ITERATIONS or args.iterations != BENCHMARK_ITERATIONS:
            raise ValueError("Formal Phase 7B requires warmup=100 and iterations=1000")
        checkpoint = args.checkpoint.resolve()
        checkpoint_hash_before = phase7a.assert_source_hash(
            checkpoint, EXPECTED_CHECKPOINT_SHA256, "checkpoint"
        )
        points, adjacency, input_metadata = phase7a.read_fixed_input(
            args.phase6_results.resolve()
        )

        import torch
        from deployment.gcn_res_onnx_model import GCNResStandardOps
        from deployment.gcn_res_onnx_wrapper import GCNResOnnxWrapper

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        device = torch.device("cuda:0")
        old_settings = {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise KeyError("checkpoint.model_state_dict missing")
        state_dict = payload["model_state_dict"]
        if tuple(state_dict["linear_1.weight"].shape) != (48, 4):
            raise RuntimeError("linear_1.weight is not (48,4)")
        if tuple(state_dict["mlp.weight"].shape) != (2, 48):
            raise RuntimeError("mlp.weight is not (2,48)")
        non_finite = [
            name
            for name, tensor in state_dict.items()
            if torch.is_tensor(tensor) and not torch.isfinite(tensor).all()
        ]
        if non_finite:
            raise FloatingPointError(f"Checkpoint contains non-finite tensors: {non_finite}")
        model = GCNResStandardOps(SimpleNamespace(num_class=2), in_dim=4)
        strict_result = model.load_state_dict(state_dict, strict=True)
        wrapper = GCNResOnnxWrapper(model).to(device).eval()
        points_gpu = torch.from_numpy(points).to(device)
        adjacency_gpu = torch.from_numpy(adjacency).to(device)
        torch.cuda.synchronize(device)

        memory_after_setup = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        }
        logits = None
        with torch.inference_mode():
            for _ in range(WARMUP_ITERATIONS):
                logits = wrapper(points_gpu, adjacency_gpu)
            torch.cuda.synchronize(device)

            memory_after_warmup = {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            }
            torch.cuda.reset_peak_memory_stats(device)
            samples_ms: list[float] = []
            for _ in range(BENCHMARK_ITERATIONS):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                logits = wrapper(points_gpu, adjacency_gpu)
                torch.cuda.synchronize(device)
                samples_ms.append((time.perf_counter() - start) * 1000.0)

        if logits is None or tuple(logits.shape) != (1, 2048, 2):
            raise RuntimeError(f"Unexpected logits: {None if logits is None else logits.shape}")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("PyTorch logits contain NaN/Inf")
        memory_after_benchmark = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "device_total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        }
        output = {
            "status": "PYTORCH_CUDA_LATENCY_BENCHMARK_COMPLETED",
            "timestamp": datetime.now().astimezone().isoformat(),
            "framework": "PyTorch CUDA deployment",
            "measurement_scope": "model forward only; inputs remain resident on cuda:0",
            "timer": "time.perf_counter with torch.cuda.synchronize before and after each forward",
            "warmup_iterations": WARMUP_ITERATIONS,
            "benchmark_iterations": BENCHMARK_ITERATIONS,
            "latency": latency_statistics(samples_ms),
            "input": input_metadata,
            "input_residency": "cuda:0 before warmup and throughout benchmark",
            "output_shape": list(logits.shape),
            "output_dtype": str(logits.dtype),
            "output_finite": True,
            "output_sha256": phase5.array_sha256(
                np.ascontiguousarray(logits.detach().cpu().numpy(), dtype=np.float32)
            ),
            "checkpoint": {
                "path": str(checkpoint),
                "sha256_before": checkpoint_hash_before,
                "sha256_after": phase7a.sha256(checkpoint),
                "strict_load_result": str(strict_result),
                "epoch": int(payload.get("epoch", -1)),
            },
            "strict_fp32": {
                "matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "float32_matmul_precision": "highest",
                "fp16": False,
                "int8": False,
            },
            "memory": {
                "allocator": "torch.cuda caching allocator",
                "after_setup": memory_after_setup,
                "after_warmup": memory_after_warmup,
                "benchmark": memory_after_benchmark,
                "peak_reset_after_warmup": True,
            },
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "python_executable": sys.executable,
                "torch_version": torch.__version__,
                "torch_cuda_runtime": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_name": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
            },
            "boundaries": {
                "engine_or_onnx_used": False,
                "accuracy_regression": False,
                "fp16": False,
                "int8": False,
            },
        }
        if output["checkpoint"]["sha256_before"] != output["checkpoint"]["sha256_after"]:
            raise RuntimeError("Checkpoint changed during PyTorch benchmark")
        dump_json(output_path, output)
        print(f"PYTORCH_LATENCY_JSON={output_path}")
        print("PYTORCH_CUDA_LATENCY_BENCHMARK_COMPLETED")
        del logits, adjacency_gpu, points_gpu, wrapper, model, state_dict, payload
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        return 0
    except Exception as exc:
        failure.update(
            {
                "exception_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        dump_json(output_path, failure)
        print(traceback.format_exc(), file=sys.stderr)
        print("TENSORRT_LATENCY_BENCHMARK_FAILED")
        return 1
    finally:
        if old_settings and "torch" in locals():
            torch.backends.cuda.matmul.allow_tf32 = old_settings["matmul_allow_tf32"]
            torch.backends.cudnn.allow_tf32 = old_settings["cudnn_allow_tf32"]
            torch.set_float32_matmul_precision(old_settings["float32_matmul_precision"])


if __name__ == "__main__":
    raise SystemExit(main())
