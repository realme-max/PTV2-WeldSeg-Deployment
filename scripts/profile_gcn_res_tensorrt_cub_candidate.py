"""Run the Phase 7C IProfiler path against the Phase 8C CUB candidate engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import gcn_res_tensorrt_cub_common as common  # noqa: E402
import profile_gcn_res_tensorrt_engine as profile  # noqa: E402


EXPERIMENTAL_SOURCES = tuple((PROJECT_ROOT / "deployment/tensorrt_voxel_unique_plugin_cub").glob("*"))


def candidate_loader(path: Path) -> tuple[Any, dict[str, Any]]:
    library, info = common.load_cub_plugin(path)
    library.getVoxelUniqueRuntimeCreationCount = library.getVoxelUniqueCubRuntimeCreationCount
    library.getVoxelUniqueBuildCreationCount = library.getVoxelUniqueCubBuildCreationCount
    return library, {
        "registration_function_returned": info["registered"],
        "library_path": info["path"], "library_sha256": info["sha256"],
        "plugin_name": common.PLUGIN_NAME, "plugin_version": common.PLUGIN_VERSION,
        "plugin_namespace": common.PLUGIN_NAMESPACE,
    }


def candidate_registry(trt: Any, registry: Any) -> dict[str, Any]:
    del trt
    audit = common.registry_audit(registry)
    creator = audit["experimental_matches"][0] if audit["experimental_matches"] else None
    return {"voxel_unique_creator_found": creator is not None, "voxel_unique": creator, "candidate_registry": audit}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase8c-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.phase8c_dir.resolve()
    engine = root / "strict_fp32_voxelunique_cub_candidate.plan"
    onnx = root / "gcn_res_voxelunique_cub_candidate.onnx"
    plugin = root / "VoxelUniqueCubPlugin.dll"
    inspector = root / "engine_inspector.json"
    inspector_payload = json.loads(inspector.read_text(encoding="utf-8"))
    layers = inspector_payload.get("Layers", [])
    gemm_layers = [layer for layer in layers if str(layer.get("LayerType", "")).lower() == "gemm"]
    gemm_audit = {
        "gemm_layer_count": len(gemm_layers),
        "tf32_gemm_layer_count": sum("tf32" in str(layer.get("TacticName", "")).lower() for layer in gemm_layers),
        "fp16_gemm_layer_count": sum("fp16" in str(layer.get("TacticName", "")).lower() for layer in gemm_layers),
        "source": "Phase 8C candidate engine Inspector",
    }
    common.dump_json(root / "candidate_strict_fp32_gemm_audit.json", gemm_audit)
    profile.phase7a.EXPECTED_ENGINE_SHA256 = common.sha256(engine)
    profile.phase7a.EXPECTED_ONNX_SHA256 = common.sha256(onnx)
    profile.phase7a.EXPECTED_PLUGIN_SHA256 = common.sha256(plugin)
    profile.phase7a.EXPECTED_CHECKPOINT_SHA256 = common.PROTECTED_HASHES["checkpoint"]
    profile.phase7a.PLUGIN_SOURCE_FILES = EXPERIMENTAL_SOURCES
    profile.phase4.load_plugin_library = candidate_loader
    profile.phase4.collect_registry = candidate_registry
    original_flags = profile.flags_for_layer

    def cub_flags(name: str, layer: dict[str, Any] | None) -> list[str]:
        if layer is not None and str(layer.get("LayerType", "")).lower() == "pluginv3" and str(layer.get("PluginType", "")).lower() == "voxeluniquecub":
            return ["Plugin", "VoxelUnique", "DynamicShape"]
        return original_flags(name, layer)

    profile.flags_for_layer = cub_flags
    sys.argv = [
        sys.argv[0], "--engine", str(engine), "--inspector", str(inspector),
        "--gemm-audit", str(root / "candidate_strict_fp32_gemm_audit.json"),
        "--build-config", str(root / "builder_config.json"), "--onnx", str(onnx),
        "--plugin-library", str(plugin), "--checkpoint", str(common.CHECKPOINT),
        "--phase6-results", str(common.PHASE6_DIR / "per_sample_results.json"),
        "--tensorrt-root", str(common.TENSORRT_ROOT), "--cuda-root", str(common.CUDA_ROOT),
        "--output-root", str(root), "--run-id", "profile_candidate",
    ]
    return profile.main()


if __name__ == "__main__":
    raise SystemExit(main())
