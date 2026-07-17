"""Run the Phase 7B benchmark implementation against the Phase 8C CUB candidate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import benchmark_gcn_res_tensorrt_latency as benchmark  # noqa: E402
import gcn_res_tensorrt_cub_common as common  # noqa: E402


EXPERIMENTAL_SOURCES = tuple((PROJECT_ROOT / "deployment/tensorrt_voxel_unique_plugin_cub").glob("*"))


def candidate_loader(path: Path) -> tuple[Any, dict[str, Any]]:
    library, info = common.load_cub_plugin(path)
    library.getVoxelUniqueRuntimeCreationCount = library.getVoxelUniqueCubRuntimeCreationCount
    library.getVoxelUniqueBuildCreationCount = library.getVoxelUniqueCubBuildCreationCount
    return library, {
        "registration_function_returned": info["registered"],
        "library_path": info["path"],
        "library_sha256": info["sha256"],
        "plugin_name": common.PLUGIN_NAME,
        "plugin_version": common.PLUGIN_VERSION,
        "plugin_namespace": common.PLUGIN_NAMESPACE,
    }


def candidate_registry(trt: Any, registry: Any) -> dict[str, Any]:
    del trt
    audit = common.registry_audit(registry)
    creator = audit["experimental_matches"][0] if audit["experimental_matches"] else None
    return {
        "voxel_unique_creator_found": creator is not None,
        "voxel_unique": creator,
        "creator_count": audit["creator_count"],
        "candidate_registry": audit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase8c-dir", type=Path, required=True)
    args = parser.parse_args()
    phase8c = args.phase8c_dir.resolve()
    engine = phase8c / "strict_fp32_voxelunique_cub_candidate.plan"
    onnx = phase8c / "gcn_res_voxelunique_cub_candidate.onnx"
    plugin = phase8c / "VoxelUniqueCubPlugin.dll"
    benchmark.phase7a.EXPECTED_ENGINE_SHA256 = common.sha256(engine)
    benchmark.phase7a.EXPECTED_ONNX_SHA256 = common.sha256(onnx)
    benchmark.phase7a.EXPECTED_PLUGIN_SHA256 = common.sha256(plugin)
    benchmark.phase7a.EXPECTED_CHECKPOINT_SHA256 = common.PROTECTED_HASHES["checkpoint"]
    benchmark.phase7a.PLUGIN_SOURCE_FILES = EXPERIMENTAL_SOURCES
    benchmark.phase4.load_plugin_library = candidate_loader
    benchmark.phase4.collect_registry = candidate_registry
    sys.argv = [
        sys.argv[0], "--run-dir", str(args.run_dir.resolve()),
        "--engine", str(engine), "--onnx", str(onnx),
        "--plugin-library", str(plugin), "--checkpoint", str(common.CHECKPOINT),
        "--phase6-results", str(common.PHASE6_DIR / "per_sample_results.json"),
        "--tensorrt-root", str(common.TENSORRT_ROOT), "--cuda-root", str(common.CUDA_ROOT),
        "--warmup", "100", "--iterations", "1000",
    ]
    return benchmark.main()


if __name__ == "__main__":
    raise SystemExit(main())
