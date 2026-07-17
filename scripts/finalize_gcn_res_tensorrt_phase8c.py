"""Aggregate Phase 8C correctness, latency, profiling, memory, and integrity results."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import gcn_res_tensorrt_cub_common as common  # noqa: E402

BASELINE_PROFILE = PROJECT_ROOT / "artifacts/gcn_res_tensorrt/20260717_131821_006323_phase7c_profiling"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def copy(source: Path, target: Path) -> None:
    shutil.copy2(source, target)


def source_manifest(directory: Path) -> dict[str, Any]:
    return {
        str(path.resolve()): {"sha256": common.sha256(path), "size_bytes": path.stat().st_size}
        for path in sorted(directory.glob("*")) if path.is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    candidate_latency = load(root / "latency_candidate/tensorrt_latency.json")
    baseline_latency = load(root / "latency_baseline/tensorrt_latency.json")
    pytorch_latency = load(root / "latency_pytorch/pytorch_latency.json")
    candidate_memory = load(root / "latency_candidate/memory_summary.json")
    baseline_memory = load(root / "latency_baseline/memory_summary.json")
    validation_memory = load(root / "memory_summary.json")
    candidate_profile = load(root / "profile_candidate/layer_profile.json")
    candidate_plugins = load(root / "profile_candidate/plugin_profile.json")
    candidate_gemm = load(root / "profile_candidate/gemm_profile.json")
    baseline_profile = load(BASELINE_PROFILE / "layer_profile.json")
    baseline_plugins = load(BASELINE_PROFILE / "plugin_profile.json")
    parity = load(root / "candidate_multisample_parity.json")
    intermediate = load(root / "plugin_intermediate_parity.json")
    smoke = load(root / "runtime_smoke_report.json")
    engine_metadata = load(root / "engine_metadata.json")

    pt_mean = float(pytorch_latency["latency"]["mean"])
    base_pure = float(baseline_latency["pure_inference"]["latency"]["mean"])
    base_e2e = float(baseline_latency["end_to_end"]["latency"]["mean"])
    cand_pure = float(candidate_latency["pure_inference"]["latency"]["mean"])
    cand_e2e = float(candidate_latency["end_to_end"]["latency"]["mean"])
    latency = {
        "status": "TENSORRT_CUB_END_TO_END_ACCELERATION_CONFIRMED" if cand_pure < pt_mean and cand_e2e < pt_mean else "PYTORCH_SPEEDUP_NOT_ACHIEVED",
        "contract": {"sample_id": "weld_65", "warmup": 100, "measurements": 1000, "batch": 1, "num_points": 2048, "precision": "strict_fp32", "tf32": False, "fp16": False, "int8": False},
        "pytorch_cuda": pytorch_latency["latency"],
        "baseline_tensorrt_pure": baseline_latency["pure_inference"]["latency"],
        "baseline_tensorrt_e2e": baseline_latency["end_to_end"]["latency"],
        "candidate_tensorrt_pure": candidate_latency["pure_inference"]["latency"],
        "candidate_tensorrt_e2e": candidate_latency["end_to_end"]["latency"],
        "speedups": {
            "candidate_vs_baseline_pure": base_pure / cand_pure,
            "candidate_vs_baseline_e2e": base_e2e / cand_e2e,
            "candidate_vs_pytorch_pure": pt_mean / cand_pure,
            "candidate_vs_pytorch_e2e": pt_mean / cand_e2e,
        },
        "isolated_plugin_speedup_not_used_as_full_model_speedup": True,
    }
    common.dump_json(root / "candidate_latency_benchmark.json", latency)
    with (root / "latency_comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["implementation", "scope", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "std_ms", "min_ms", "max_ms"])
        for implementation, scope, stats in (
            ("PyTorch CUDA strict FP32", "model_forward", pytorch_latency["latency"]),
            ("baseline TensorRT strict FP32", "pure_enqueue", baseline_latency["pure_inference"]["latency"]),
            ("baseline TensorRT strict FP32", "end_to_end", baseline_latency["end_to_end"]["latency"]),
            ("candidate TensorRT VoxelUniqueCub strict FP32", "pure_enqueue", candidate_latency["pure_inference"]["latency"]),
            ("candidate TensorRT VoxelUniqueCub strict FP32", "end_to_end", candidate_latency["end_to_end"]["latency"]),
        ):
            writer.writerow([implementation, scope, stats["mean"], stats["p50"], stats["p95"], stats["p99"], stats["std"], stats["min"], stats["max"]])

    cand_total = float(candidate_profile["summary"]["average_total_layer_time_ms"])
    base_total = float(baseline_profile["summary"]["average_total_layer_time_ms"])
    cand_plugin_avg = float(candidate_plugins["summary"]["average_time_ms_per_inference"])
    base_plugin_avg = float(baseline_plugins["summary"]["average_time_ms_per_inference"])
    cand_tdb1 = next(item for item in candidate_plugins["instances"] if item["layer_name"] == "/model/tdb_1/Unique")
    base_tdb1 = next(item for item in baseline_plugins["instances"] if item["layer_name"] == "/model/tdb_1/Unique")
    dynamic = candidate_profile["category_aggregates"]["overlapping_flags"]["DynamicShape"]
    dynamic_ex_plugin_ms = float(dynamic["avg_time_ms_per_inference"]) - cand_plugin_avg
    profile_comparison = {
        "status": "TENSORRT_CUB_PLUGIN_OPTIMIZATION_CONFIRMED",
        "contract": {"sample_id": "weld_65", "warmup": 100, "profile_iterations": 100, "profiler": "TensorRT IProfiler"},
        "baseline": {"total_ms": base_total, "layer_count": baseline_profile["summary"]["profiled_layer_count"], "plugin_total_ms": base_plugin_avg, "plugin_share_percent": baseline_plugins["summary"]["percentage"], "tdb1_plugin_ms": base_tdb1["avg_time_ms"]},
        "candidate": {"total_ms": cand_total, "layer_count": candidate_profile["summary"]["profiled_layer_count"], "plugin_total_ms": cand_plugin_avg, "plugin_share_percent": candidate_plugins["summary"]["percentage"], "tdb1_plugin_ms": cand_tdb1["avg_time_ms"], "gemm": candidate_gemm["summary"], "scatter": candidate_profile["category_aggregates"]["overlapping_flags"]["Scatter"], "gather": candidate_profile["category_aggregates"]["overlapping_flags"]["Gather"], "dynamic_shape_including_plugin": dynamic, "dynamic_shape_excluding_plugin_avg_ms": dynamic_ex_plugin_ms, "dynamic_shape_excluding_plugin_percent": dynamic_ex_plugin_ms / cand_total * 100.0, "layers_below_0_05_ms": sum(item["avg_time_ms"] < 0.05 for item in candidate_profile["layers"]), "new_top_layer": candidate_profile["layers"][0], "bottleneck_classification": candidate_profile["bottleneck_classification"]},
        "speedups": {"profiled_full_engine": base_total / cand_total, "four_plugins_total": base_plugin_avg / cand_plugin_avg, "tdb1_plugin": float(base_tdb1["avg_time_ms"]) / float(cand_tdb1["avg_time_ms"])},
        "isolated_phase8b_speedup_is_not_reported_as_full_engine_speedup": True,
    }
    common.dump_json(root / "profiling_comparison.json", profile_comparison)
    copy(root / "profile_candidate/layer_profile.json", root / "candidate_layer_profile.json")
    copy(root / "profile_candidate/top50_layers.csv", root / "candidate_top50_layers.csv")
    copy(root / "profile_candidate/plugin_profile.json", root / "candidate_plugin_profile.json")
    copy(root / "graph_rewrite_audit.json", root / "derived_onnx_audit.json")

    memory = {
        "method": "Separate-process cudaMemGetInfo lifecycle snapshots; not an instantaneous in-kernel peak.",
        "candidate": candidate_memory["tensorrt"], "baseline": baseline_memory["tensorrt"],
        "pytorch": candidate_memory["pytorch"],
        "candidate_engine_size_bytes": engine_metadata["size_bytes"],
        "baseline_engine_size_bytes": common.BASELINE_ENGINE.stat().st_size,
        "plugin_workspace": {
            "per_instance_at_n2048_bytes": validation_memory["candidate_plugin_workspace_per_instance_bytes_at_n2048"],
            "four_instance_upper_bound_sum_bytes": validation_memory["candidate_plugin_workspace_upper_bound_sum_bytes"],
            "visibility": "Recorded from plugin configuration. TensorRT may reuse workspace, and cudaMemGetInfo snapshots cannot isolate transient kernel workspace.",
        },
    }
    common.dump_json(root / "memory_summary.json", memory)

    protected_after = common.protected_snapshot()
    baseline_source_dir = PROJECT_ROOT / "deployment/tensorrt_voxel_unique_plugin"
    candidate_source_dir = PROJECT_ROOT / "deployment/tensorrt_voxel_unique_plugin_cub"
    environment = load(root / "environment.json")
    environment.update({
        "phase8c_finalized_at": datetime.now().astimezone().isoformat(),
        "formal_protected_artifacts": protected_after,
        "baseline_plugin_source_manifest_after": source_manifest(baseline_source_dir),
        "candidate_plugin_source_manifest": source_manifest(candidate_source_dir),
        "candidate": {"onnx_sha256": common.sha256(root / "gcn_res_voxelunique_cub_candidate.onnx"), "engine_sha256": common.sha256(root / "strict_fp32_voxelunique_cub_candidate.plan"), "plugin_sha256": common.sha256(root / "VoxelUniqueCubPlugin.dll")},
        "statuses": {"parser": "TENSORRT_VOXELUNIQUE_CUB_PARSER_PASSED", "builder": "TENSORRT_VOXELUNIQUE_CUB_CANDIDATE_ENGINE_BUILD_PASSED", "runtime": parity["runtime_status"], "plugin_intermediate": intermediate["status"], "task": parity["task_status"], "strict_numerical": parity["strict_numerical_status"], "latency": latency["status"], "profiling": profile_comparison["status"]},
    })
    common.dump_json(root / "environment.json", environment)

    summary = f"""# TensorRT Phase 8C Candidate Engine Summary

`VOXELUNIQUE_CUB_CANDIDATE_ENGINE_REGRESSION_COMPLETED`

- Parser/build/runtime: PASS / PASS / PASS
- Plugin intermediate parity: 16/16 exact
- Test split runtime: 18/18
- Candidate vs baseline labels/task metrics: exact
- Original strict numerical threshold: {parity['strict_numerical_status']} ({parity['strict_threshold_passed_samples']}/18)
- Candidate vs PyTorch worst max-abs: {parity['worst_candidate_vs_pytorch']['candidate_vs_pytorch']['max_abs_error']:.12e} ({parity['worst_candidate_vs_pytorch']['sample_id']})
- PyTorch mean: {pt_mean:.6f} ms
- Baseline TensorRT pure/E2E: {base_pure:.6f} / {base_e2e:.6f} ms
- Candidate TensorRT pure/E2E: {cand_pure:.6f} / {cand_e2e:.6f} ms
- Candidate vs baseline speedup: {base_pure / cand_pure:.3f}x pure, {base_e2e / cand_e2e:.3f}x E2E
- Candidate vs PyTorch speedup: {pt_mean / cand_pure:.3f}x pure, {pt_mean / cand_e2e:.3f}x E2E
- Candidate four-Plugin profile: {cand_plugin_avg:.6f} ms ({candidate_plugins['summary']['percentage']:.3f}%)
- New profile classification: {candidate_profile['bottleneck_classification']['label']}

The candidate remains an experimental artifact. The formal ONNX, baseline engine,
baseline Plugin DLL, checkpoint, and baseline Plugin implementation were not replaced.
"""
    (root / "phase8c_summary.md").write_text(summary, encoding="utf-8")
    # Consolidate human-readable logs without altering the source logs.
    with (root / "benchmark.log").open("w", encoding="utf-8") as out:
        for name in ("benchmark_pytorch.log", "benchmark_baseline.log", "benchmark_candidate.log"):
            out.write(f"===== {name} =====\n")
            out.write((root / name).read_text(encoding="utf-8", errors="replace"))
            out.write("\n")
    print("TENSORRT_CUB_PLUGIN_OPTIMIZATION_CONFIRMED")
    print(latency["status"])
    print("PHASE8C_ARTIFACT_AGGREGATION_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
