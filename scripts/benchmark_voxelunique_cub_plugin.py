"""Benchmark the isolated experimental VoxelUniqueCub TensorRT plugin.

This reuses the Phase 8A CUDA-event harness so baseline and CUB measurements
have identical H2D/plugin/D2H boundaries, warmup count, and sample count.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.benchmark_voxelunique_plugin as baseline  # noqa: E402


PLUGIN_NAME = "VoxelUniqueCub"
PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = "com.tensorrt.ptv2.experimental"


def load_cub_plugin_library(path: Path) -> tuple[Any, dict[str, Any]]:
    library = ctypes.CDLL(str(path))
    library.initVoxelUniqueCubPlugin.argtypes = []
    library.initVoxelUniqueCubPlugin.restype = ctypes.c_bool
    library.getVoxelUniqueCubBuildCreationCount.argtypes = []
    library.getVoxelUniqueCubBuildCreationCount.restype = ctypes.c_int32
    library.getVoxelUniqueCubRuntimeCreationCount.argtypes = []
    library.getVoxelUniqueCubRuntimeCreationCount.restype = ctypes.c_int32
    # The Phase 8A harness reads the baseline counter symbol after inference.
    # Alias only the Python CDLL attribute; no DLL export or plugin identity is changed.
    library.getVoxelUniqueRuntimeCreationCount = (
        library.getVoxelUniqueCubRuntimeCreationCount
    )
    registered = bool(library.initVoxelUniqueCubPlugin())
    return library, {
        "path": str(path),
        "sha256": baseline.phase4.sha256(path),
        "registration_function_returned": registered,
    }


def collect_cub_registry(trt: Any, registry: Any) -> dict[str, Any]:
    creators = [
        {
            "name": creator.name,
            "version": creator.plugin_version,
            "namespace": creator.plugin_namespace,
            "python_type": type(creator).__name__,
        }
        for creator in registry.all_creators
    ]
    creator = registry.get_creator(PLUGIN_NAME, PLUGIN_VERSION, PLUGIN_NAMESPACE)
    selected = next(
        (
            item
            for item in creators
            if item["name"] == PLUGIN_NAME
            and item["version"] == PLUGIN_VERSION
            and item["namespace"] == PLUGIN_NAMESPACE
        ),
        None,
    )
    return {
        "creator_count": len(creators),
        # Compatibility keys consumed by the unchanged Phase 8A harness.
        "voxel_unique_creator_found": creator is not None,
        "voxel_unique": selected,
        "experimental_creator": selected,
        "all_creators": creators,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--plugin-library", type=Path, required=True)
    parser.add_argument(
        "--tensorrt-root", type=Path, default=baseline.DEFAULT_TENSORRT_ROOT
    )
    parser.add_argument("--cuda-root", type=Path, default=baseline.DEFAULT_CUDA_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = args.engine.resolve()
    plugin = args.plugin_library.resolve()
    # The shared harness protects its inputs using fixed expected hashes. For
    # this new isolated pair, freeze the current hashes before deserialization.
    baseline.EXPECTED_ENGINE_SHA256 = baseline.phase4.sha256(engine)
    baseline.EXPECTED_PLUGIN_SHA256 = baseline.phase4.sha256(plugin)
    baseline.phase4.load_plugin_library = load_cub_plugin_library
    baseline.phase4.collect_registry = collect_cub_registry

    with np.load(args.keys_npz, allow_pickle=False) as archive:
        cases = {name: archive[name] for name in archive.files}
    result = baseline.run_cases(
        cases,
        engine,
        plugin,
        args.tensorrt_root,
        args.cuda_root,
    )
    weld_mean = float(result["cases"]["weld_65_tdb1_keys"]["kernel_execution"]["mean"])
    baseline_weld_mean = 28.84781257247925
    speedup = baseline_weld_mean / weld_mean
    benchmark_passed = weld_mean < 5.0 and speedup >= 5.0
    result["status"] = (
        "VOXEL_UNIQUE_CUB_BENCHMARK_PASSED"
        if benchmark_passed
        else "VOXEL_UNIQUE_CUB_BENCHMARK_TARGET_NOT_MET"
    )
    result["acceptance"] = {
        "weld_65_kernel_mean_ms": weld_mean,
        "target_kernel_mean_ms_less_than": 5.0,
        "baseline_kernel_mean_ms": baseline_weld_mean,
        "speedup": speedup,
        "minimum_speedup": 5.0,
        "passed": benchmark_passed,
    }
    result["plugin"]["identity"] = {
        "name": PLUGIN_NAME,
        "version": PLUGIN_VERSION,
        "namespace": PLUGIN_NAMESPACE,
    }
    result["measurement_protocol"] = {
        "phase8a_harness_reused": True,
        "warmup_iterations": baseline.WARMUP_ITERATIONS,
        "benchmark_iterations": baseline.BENCHMARK_ITERATIONS,
        "timer": "CUDA events on one stream",
        "boundaries": ["H2D", "plugin enqueueV3", "D2H", "total"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"OUTPUT={args.output.resolve()}")
    print(result["status"])
    return 0 if benchmark_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
