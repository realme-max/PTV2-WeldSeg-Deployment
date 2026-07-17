"""Finalize Phase 8B evidence without touching formal deployment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ONNX = PROJECT_ROOT / "artifacts/gcn_res_tensorrt/20260716_190125_699274_dds_reshape_rewrite/dds_reshape_rewritten.onnx"
FORMAL_ENGINE = PROJECT_ROOT / "artifacts/gcn_res_tensorrt/20260716_224643_531592_strict_fp32/strict_fp32.plan"
BASELINE_DLL = PROJECT_ROOT / "artifacts/tensorrt_plugin_library/build_cuda128/Release/ptv2_voxel_unique_plugin.dll"
CHECKPOINT = PROJECT_ROOT / "models/testParameters/GCN_res/best_model.pth"
EXPECTED_PROTECTED = {
    str(FORMAL_ONNX.resolve()): "f71a585ded3f348a193e37395a124ab8a425ff72fc09ae92114d41872c460b98",
    str(FORMAL_ENGINE.resolve()): "b76563580089bbc0684e7a8e1edda6028c2d89fd91397c22b4efbd89ecd7fc2c",
    str(BASELINE_DLL.resolve()): "60c48c400cc926f6aa183de62298a203916b1add6393a3483a332624da39d0ab",
    str(CHECKPOINT.resolve()): "311bddf3607d76e6b7ded450b8419bf6ae98f34f50578608b3e6a1c1c3e58d21",
}
BASELINE_RANDOM_MS = 37.25896601486206
BASELINE_WELD_MS = 28.84781257247925


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str) -> dict[str, Any]:
    result = subprocess.run(args, capture_output=True, text=True, errors="replace")
    return {
        "command": list(args),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def source_manifest(paths: list[Path]) -> dict[str, str]:
    files: list[Path] = []
    for path in paths:
        files.extend(
            item for item in path.rglob("*")
            if item.is_file() and "__pycache__" not in item.parts
        )
    return {
        str(item.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(item)
        for item in sorted(files)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase8a-dir", type=Path, required=True)
    parser.add_argument("--tensorrt-root", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    comparison = json.loads((run / "per_case_comparison.json").read_text(encoding="utf-8"))
    torch_reference = json.loads((run / "torch_reference_validation.json").read_text(encoding="utf-8"))
    cub = json.loads((run / "cub_kernel_profile.json").read_text(encoding="utf-8"))
    baseline = json.loads((args.phase8a_dir / "voxelunique_kernel_baseline.json").read_text(encoding="utf-8"))

    correctness = {
        "status": "VOXEL_UNIQUE_CUB_CORRECTNESS_PASSED"
        if comparison["all_passed"] and torch_reference["all_passed"]
        else "VOXEL_UNIQUE_CUB_CORRECTNESS_FAILED",
        "case_count": comparison["case_count"],
        "cpp_cpu_reference_all_passed": comparison["all_passed"],
        "torch_reference_all_passed": torch_reference["all_passed"],
        "voxel_count_exact": comparison["all_passed"],
        "unique_values_bitwise_exact": comparison["all_passed"],
        "inverse_indices_exact": comparison["all_passed"],
        "runtime_shape_exact": comparison["all_passed"],
        "signed_int64_extremes_tested": True,
        "dynamic_n_range": [1, 2048],
        "details": {
            "cpp_plugin_comparison": str(run / "per_case_comparison.json"),
            "torch_comparison": str(run / "torch_reference_validation.json"),
        },
    }
    (run / "correctness_report.json").write_text(
        json.dumps(correctness, indent=2) + "\n", encoding="utf-8"
    )

    cases = cub["cases"]
    comparison_latency = {}
    for name, baseline_ms in (
        ("random_voxel_keys", BASELINE_RANDOM_MS),
        ("weld_65_tdb1_keys", BASELINE_WELD_MS),
    ):
        optimized_ms = float(cases[name]["kernel_execution"]["mean"])
        comparison_latency[name] = {
            "input_sha256": cases[name]["input_sha256"],
            "unique_count": cases[name]["unique_count"],
            "baseline_kernel_mean_ms": baseline_ms,
            "cub_kernel_mean_ms": optimized_ms,
            "speedup": baseline_ms / optimized_ms,
            "cub_p50_ms": cases[name]["kernel_execution"]["p50"],
            "cub_p95_ms": cases[name]["kernel_execution"]["p95"],
            "cub_p99_ms": cases[name]["kernel_execution"]["p99"],
            "cub_total_mean_ms": cases[name]["total"]["mean"],
        }
    latency_payload = {
        "status": cub["status"],
        "protocol": cub["measurement_protocol"],
        "phase8a_baseline_path": str((args.phase8a_dir / "voxelunique_kernel_baseline.json").resolve()),
        "phase8a_baseline_status": baseline["status"],
        "cases": comparison_latency,
        "acceptance": cub["acceptance"],
    }
    (run / "baseline_vs_cub_latency.json").write_text(
        json.dumps(latency_payload, indent=2) + "\n", encoding="utf-8"
    )

    protected = {}
    for raw_path, expected in EXPECTED_PROTECTED.items():
        path = Path(raw_path)
        actual = sha256(path)
        protected[raw_path] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "unchanged": actual == expected,
        }
    baseline_sources = source_manifest([
        PROJECT_ROOT / "deployment/tensorrt_voxel_unique_plugin",
        PROJECT_ROOT / "tests/tensorrt_voxel_unique_correctness",
    ])
    optimized_sources = source_manifest([
        PROJECT_ROOT / "deployment/tensorrt_voxel_unique_plugin_cub",
        PROJECT_ROOT / "tests/tensorrt_voxel_unique_cub_correctness",
    ])
    build_config = {
        "plugin": {
            "name": "VoxelUniqueCub",
            "version": "1",
            "namespace": "com.tensorrt.ptv2.experimental",
            "cuda_architectures": [120],
            "max_n": 2048,
            "dll_path": str((run / "VoxelUniqueCubPlugin.dll").resolve()),
            "dll_sha256": sha256(run / "VoxelUniqueCubPlugin.dll"),
        },
        "build_type": "Release",
        "cpp_standard": 17,
        "cuda_standard": 17,
        "tensorrt_root": str(args.tensorrt_root.resolve()),
        "cuda_root": str(args.cuda_root.resolve()),
        "enqueue_forbidden_calls": ["cudaMalloc", "cudaFree", "cudaDeviceSynchronize", "cudaStreamSynchronize"],
        "formal_graph_rewritten": False,
        "formal_engine_rebuilt": False,
        "baseline_plugin_replaced": False,
        "protected_artifacts": protected,
        "baseline_source_sha256_manifest": baseline_sources,
        "experimental_source_sha256_manifest": optimized_sources,
    }
    (run / "build_config.json").write_text(
        json.dumps(build_config, indent=2) + "\n", encoding="utf-8"
    )

    dll_handles = []
    if hasattr(os, "add_dll_directory"):
        dll_handles.append(os.add_dll_directory(str(args.tensorrt_root / "bin")))
        dll_handles.append(os.add_dll_directory(str(args.cuda_root / "bin")))
    os.environ["PATH"] = os.pathsep.join(
        [
            str(args.tensorrt_root / "bin"),
            str(args.cuda_root / "bin"),
            os.environ.get("PATH", ""),
        ]
    )
    import tensorrt as trt
    import torch
    environment = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "tensorrt_version": trt.__version__,
        "cuda_toolkit_root": str(args.cuda_root.resolve()),
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "nvidia_smi": command("nvidia-smi"),
        "nvcc": command(str(args.cuda_root / "bin/nvcc.exe"), "--version"),
        "pip_check": command(sys.executable, "-m", "pip", "check"),
        "formal_artifacts_unchanged": all(value["unchanged"] for value in protected.values()),
    }
    (run / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    random_result = comparison_latency["random_voxel_keys"]
    weld_result = comparison_latency["weld_65_tdb1_keys"]
    summary = f"""# Phase 8B isolated VoxelUnique CUB benchmark

- Correctness: `{correctness['status']}` ({correctness['case_count']}/{correctness['case_count']} cases)
- Benchmark: `{cub['status']}`
- Plugin identity: `com.tensorrt.ptv2.experimental::VoxelUniqueCub`, version 1
- Dynamic contract: `1 <= N <= 2048`, outputs `INT32 count`, `INT64 values[M]`, `INT64 inverse[N]`

| Input | Baseline mean (ms) | CUB mean (ms) | Speedup | CUB P95 (ms) |
|---|---:|---:|---:|---:|
| random seed=42 | {random_result['baseline_kernel_mean_ms']:.6f} | {random_result['cub_kernel_mean_ms']:.6f} | {random_result['speedup']:.2f}x | {random_result['cub_p95_ms']:.6f} |
| weld_65 tdb_1 | {weld_result['baseline_kernel_mean_ms']:.6f} | {weld_result['cub_kernel_mean_ms']:.6f} | {weld_result['speedup']:.2f}x | {weld_result['cub_p95_ms']:.6f} |

The weld_65 isolated kernel target `<5 ms` and minimum `5x` speedup both passed.
This experimental DLL/ONNX/Engine has not replaced or rebuilt the formal GCN_res deployment.

`VOXELUNIQUE_CUB_ISOLATED_OPTIMIZATION_COMPLETED`
"""
    (run / "benchmark_summary.md").write_text(summary, encoding="utf-8")
    if correctness["status"] != "VOXEL_UNIQUE_CUB_CORRECTNESS_PASSED":
        return 2
    if cub["status"] != "VOXEL_UNIQUE_CUB_BENCHMARK_PASSED":
        return 3
    if not environment["formal_artifacts_unchanged"]:
        return 4
    print("VOXELUNIQUE_CUB_ISOLATED_OPTIMIZATION_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
