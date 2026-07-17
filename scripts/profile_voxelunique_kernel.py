"""Generate and run the Phase 8A baseline VoxelUnique CUDA profile."""

from __future__ import annotations

import argparse
import json
import platform
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

import benchmark_voxelunique_plugin as isolated  # noqa: E402
import build_gcn_res_tensorrt_fp32 as phase4  # noqa: E402
import run_gcn_res_tensorrt_fp32_inference as phase5  # noqa: E402
import smoke_test_gcn_res_tensorrt_engine as phase7a  # noqa: E402


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_tensorrt"
DEFAULT_CORRECTNESS = (
    PROJECT_ROOT
    / "artifacts"
    / "tensorrt_plugin_prototype"
    / "20260715_203305_357432_correctness"
    / "comparison.json"
)
DEFAULT_PHASE7C = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_tensorrt"
    / "20260717_131821_006323_phase7c_profiling"
    / "plugin_profile.json"
)


def dump_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def make_run_directory(output_root: Path, run_id: str | None) -> Path:
    name = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_phase8a_voxelunique"
    if any(token in name for token in ("/", "\\", "..")):
        raise ValueError(f"Unsafe run id: {name!r}")
    run_dir = output_root.resolve() / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def real_weld_65_tdb1_keys() -> tuple[np.ndarray, dict[str, Any]]:
    points, _adjacency, input_metadata = phase7a.read_fixed_input(
        phase7a.DEFAULT_PHASE6_RESULTS.resolve()
    )
    xyz = np.ascontiguousarray(points[0, :, :3], dtype=np.float32)
    voxel_size = np.float32(0.06)
    start = np.min(xyz, axis=0).astype(np.float32, copy=False)
    end = np.max(xyz, axis=0).astype(np.float32, copy=False)
    coordinates = np.floor((xyz - start) / voxel_size).astype(np.int64)
    extents = np.floor((end - start) / voxel_size).astype(np.int64) + 1
    strides = np.asarray([1, extents[0], extents[0] * extents[1]], dtype=np.int64)
    keys = np.ascontiguousarray(np.sum(coordinates * strides, axis=1), dtype=np.int64)
    if keys.shape != (2048,):
        raise RuntimeError(f"Unexpected real key shape: {keys.shape}")
    return keys, {
        "source_sample": input_metadata,
        "voxel_size": [0.06, 0.06, 0.06],
        "start": start.tolist(),
        "end": end.tolist(),
        "extents": extents.tolist(),
        "key_formula": "x + extent_x*y + extent_x*extent_y*z",
        "unique_count_cpu": int(np.unique(keys).size),
        "key_sha256": phase5.array_sha256(keys),
    }


def validate_historical_correctness(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    names = {case["name"] for case in cases}
    required_prefixes = ("random_n4_", "random_n8_", "random_n32_", "random_n2048_")
    required_names = {"all_same", "all_unique", "sorted", "reversed", "repeated_groups", "int64_extremes"}
    missing_prefixes = [prefix for prefix in required_prefixes if not any(name.startswith(prefix) for name in names)]
    missing_names = sorted(required_names - names)
    if not payload.get("all_passed") or missing_prefixes or missing_names:
        raise RuntimeError(
            f"Historical correctness evidence incomplete: prefixes={missing_prefixes}, names={missing_names}"
        )
    if not all(case.get("passed") for case in cases):
        raise RuntimeError("Historical correctness contains a failed case")
    return {
        "status": "CURRENT_PLUGIN_CORRECTNESS_EVIDENCE_PASSED",
        "source_path": str(path.resolve()),
        "source_sha256": phase4.sha256(path),
        "case_count": len(cases),
        "all_passed": True,
        "covered_random_sizes": [4, 8, 32, 2048],
        "covered_boundaries": sorted(required_names),
        "sorted_true": True,
        "all_values_inverse_count_shape_match": True,
        "note": "This validates the current baseline Plugin. No optimized Plugin exists in Phase 8A analysis-only scope.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--correctness", type=Path, default=DEFAULT_CORRECTNESS)
    parser.add_argument("--phase7c", type=Path, default=DEFAULT_PHASE7C)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = make_run_directory(args.output_root, args.run_id)
    failure = {
        "status": "VOXELUNIQUE_OPTIMIZATION_FAILED",
        "scope": "baseline_analysis",
    }
    dump_json(run_dir / "baseline_profile.json", failure)
    try:
        protected = {
            "formal_engine": (phase7a.DEFAULT_ENGINE.resolve(), phase7a.EXPECTED_ENGINE_SHA256),
            "formal_onnx": (phase7a.DEFAULT_ONNX.resolve(), phase7a.EXPECTED_ONNX_SHA256),
            "plugin_binary": (phase7a.DEFAULT_PLUGIN_LIBRARY.resolve(), phase7a.EXPECTED_PLUGIN_SHA256),
            "checkpoint": (phase7a.DEFAULT_CHECKPOINT.resolve(), phase7a.EXPECTED_CHECKPOINT_SHA256),
            "isolated_engine": (isolated.DEFAULT_ENGINE.resolve(), isolated.EXPECTED_ENGINE_SHA256),
        }
        hashes_before = {
            name: phase7a.assert_source_hash(path, expected, name)
            for name, (path, expected) in protected.items()
        }
        plugin_sources_before = phase7a.source_manifest()

        random_generator = np.random.default_rng(42)
        random_keys = np.ascontiguousarray(
            random_generator.integers(0, 512, size=2048, dtype=np.int64)
        )
        real_keys, real_metadata = real_weld_65_tdb1_keys()
        np.savez_compressed(
            run_dir / "benchmark_inputs.npz",
            random_voxel_keys=random_keys,
            weld_65_tdb1_keys=real_keys,
        )
        baseline = isolated.run_cases(
            {
                "random_voxel_keys": random_keys,
                "weld_65_tdb1_keys": real_keys,
            }
        )
        baseline.update(
            {
                "phase": "8A A1 isolated kernel baseline",
                "random_input": {
                    "seed": 42,
                    "distribution": "uniform integer voxel keys in [0,512)",
                    "unique_count_cpu": int(np.unique(random_keys).size),
                    "key_sha256": phase5.array_sha256(random_keys),
                },
                "real_input": real_metadata,
                "formal_engine_not_executed": True,
                "full_ptv2_bypassed": True,
            }
        )
        dump_json(run_dir / "baseline_profile.json", baseline)
        dump_json(run_dir / "voxelunique_kernel_baseline.json", baseline)

        correctness = validate_historical_correctness(args.correctness.resolve())
        correctness["profile_cases"] = {
            name: case["correctness"] for name, case in baseline["cases"].items()
        }
        dump_json(run_dir / "correctness_report.json", correctness)

        phase7c = json.loads(args.phase7c.read_text(encoding="utf-8"))
        tdb1 = next(
            item for item in phase7c["instances"] if item["layer_name"] == "/model/tdb_1/Unique"
        )
        comparison = {
            "status": "BASELINE_ONLY_NO_OPTIMIZED_IMPLEMENTATION",
            "performance_target_ms": 5.0,
            "phase7c_full_engine_tdb1_ms": float(tdb1["avg_time_ms"]),
            "isolated_real_weld_65_kernel_ms": baseline["cases"]["weld_65_tdb1_keys"]["kernel_execution"]["avg_ms"],
            "isolated_random_kernel_ms": baseline["cases"]["random_voxel_keys"]["kernel_execution"]["avg_ms"],
            "optimized_kernel_ms": None,
            "target_evaluated": False,
            "target_met": None,
            "reason": "Phase 8A was explicitly limited to baseline analysis before rewriting CUDA.",
        }
        dump_json(run_dir / "latency_comparison.json", comparison)
        dump_json(
            run_dir / "optimized_profile.json",
            {
                "status": "NOT_RUN",
                "optimized_implementation_exists": False,
                "reason": "Baseline analysis completed; CUDA rewrite was intentionally not started.",
            },
        )

        hashes_after = {name: phase7a.sha256(path) for name, (path, _expected) in protected.items()}
        plugin_sources_after = phase7a.source_manifest()
        if hashes_before != hashes_after or plugin_sources_before != plugin_sources_after:
            raise RuntimeError("A protected source changed during Phase 8A baseline analysis")
        environment = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "source_integrity": {
                name: {
                    "path": str(protected[name][0]),
                    "sha256_before": hashes_before[name],
                    "sha256_after": hashes_after[name],
                    "unchanged": hashes_before[name] == hashes_after[name],
                }
                for name in protected
            },
            "plugin_sources_unchanged": True,
            "boundaries": {
                "cuda_source_modified": False,
                "plugin_recompiled": False,
                "formal_engine_modified": False,
                "formal_engine_executed": False,
                "onnx_modified": False,
                "checkpoint_modified": False,
                "optimized_profile_executed": False,
                "fp16": False,
                "int8": False,
                "cpp_deployment": False,
            },
        }
        dump_json(run_dir / "environment.json", environment)
        dump_json(
            run_dir / "phase8a_summary.json",
            {
                "status": "VOXELUNIQUE_ANALYSIS_COMPLETED",
                "run_directory": str(run_dir),
                "baseline_profile": str(run_dir / "baseline_profile.json"),
                "optimization_performed": False,
                "regression_validation_performed": False,
            },
        )
        print(f"RUN_DIRECTORY={run_dir}")
        print("VOXELUNIQUE_ANALYSIS_COMPLETED")
        return 0
    except Exception as exc:
        failure.update(
            {
                "exception_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        dump_json(run_dir / "baseline_profile.json", failure)
        print(traceback.format_exc(), file=sys.stderr)
        print(f"RUN_DIRECTORY={run_dir}")
        print("VOXELUNIQUE_OPTIMIZATION_FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
