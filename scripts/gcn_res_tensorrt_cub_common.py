"""Shared helpers for the isolated Phase 8C TensorRT candidate."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_ONNX = PROJECT_ROOT / "artifacts/gcn_res_tensorrt/20260716_190125_699274_dds_reshape_rewrite/dds_reshape_rewritten.onnx"
BASELINE_ENGINE = PROJECT_ROOT / "artifacts/gcn_res_tensorrt/20260716_224643_531592_strict_fp32/strict_fp32.plan"
BASELINE_PLUGIN = PROJECT_ROOT / "artifacts/tensorrt_plugin_library/build_cuda128/Release/ptv2_voxel_unique_plugin.dll"
CHECKPOINT = PROJECT_ROOT / "models/testParameters/GCN_res/best_model.pth"
PHASE8B_DIR = PROJECT_ROOT / "artifacts/gcn_res_tensorrt/20260717_151544_915303_phase8b_voxelunique_cub"
EXPERIMENTAL_PLUGIN = PHASE8B_DIR / "VoxelUniqueCubPlugin.dll"
PHASE6_DIR = PROJECT_ROOT / "artifacts/gcn_res_tensorrt/20260717_110500_836041_strict_fp32_multisample"
TENSORRT_ROOT = Path(r"D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106")
CUDA_ROOT = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")

PLUGIN_NAME = "VoxelUniqueCub"
PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = "com.tensorrt.ptv2.experimental"
EXPECTED_NODE_NAMES = [f"/model/tdb_{index}/Unique" for index in range(1, 5)]
EXPECTED_IO = {
    "points": {"mode": "INPUT", "dtype": "FLOAT", "shape": [1, 2048, 4]},
    "adj": {"mode": "INPUT", "dtype": "FLOAT", "shape": [1, 2048, 2048]},
    "logits": {"mode": "OUTPUT", "dtype": "FLOAT", "shape": [1, 2048, 2]},
}
PROTECTED_HASHES = {
    "formal_onnx": "f71a585ded3f348a193e37395a124ab8a425ff72fc09ae92114d41872c460b98",
    "baseline_engine": "b76563580089bbc0684e7a8e1edda6028c2d89fd91397c22b4efbd89ecd7fc2c",
    "baseline_plugin": "60c48c400cc926f6aa183de62298a203916b1add6393a3483a332624da39d0ab",
    "checkpoint": "311bddf3607d76e6b7ded450b8419bf6ae98f34f50578608b3e6a1c1c3e58d21",
}
PROTECTED_PATHS = {
    "formal_onnx": FORMAL_ONNX,
    "baseline_engine": BASELINE_ENGINE,
    "baseline_plugin": BASELINE_PLUGIN,
    "checkpoint": CHECKPOINT,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def protected_snapshot() -> dict[str, Any]:
    result = {}
    for name, path in PROTECTED_PATHS.items():
        actual = sha256(path)
        expected = PROTECTED_HASHES[name]
        if actual != expected:
            raise RuntimeError(f"Protected {name} hash mismatch: {actual} != {expected}")
        result[name] = {
            "path": str(path.resolve()),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    return result


def configure_dll_search(plugin_path: Path) -> list[Any]:
    handles = []
    for directory in (TENSORRT_ROOT / "bin", CUDA_ROOT / "bin", plugin_path.parent):
        if hasattr(os, "add_dll_directory"):
            handles.append(os.add_dll_directory(str(directory)))
    os.environ["PATH"] = os.pathsep.join(
        [str(TENSORRT_ROOT / "bin"), str(CUDA_ROOT / "bin"), os.environ.get("PATH", "")]
    )
    return handles


def load_cub_plugin(path: Path) -> tuple[Any, dict[str, Any]]:
    library = ctypes.CDLL(str(path.resolve()))
    library.initVoxelUniqueCubPlugin.argtypes = []
    library.initVoxelUniqueCubPlugin.restype = ctypes.c_bool
    library.getVoxelUniqueCubBuildCreationCount.argtypes = []
    library.getVoxelUniqueCubBuildCreationCount.restype = ctypes.c_int32
    library.getVoxelUniqueCubRuntimeCreationCount.argtypes = []
    library.getVoxelUniqueCubRuntimeCreationCount.restype = ctypes.c_int32
    registered = bool(library.initVoxelUniqueCubPlugin())
    return library, {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "registered": registered,
        "name": PLUGIN_NAME,
        "version": PLUGIN_VERSION,
        "namespace": PLUGIN_NAMESPACE,
    }


def registry_audit(registry: Any) -> dict[str, Any]:
    creators = [
        {
            "name": creator.name,
            "version": creator.plugin_version,
            "namespace": creator.plugin_namespace,
            "python_type": type(creator).__name__,
        }
        for creator in registry.all_creators
    ]
    matches = [
        item for item in creators
        if item["name"] == PLUGIN_NAME
        and item["version"] == PLUGIN_VERSION
        and item["namespace"] == PLUGIN_NAMESPACE
    ]
    baseline_matches = [
        item for item in creators
        if item["name"] == "VoxelUnique"
        and item["version"] == "1"
        and item["namespace"] == "com.tensorrt.ptv2"
    ]
    return {
        "creator_count": len(creators),
        "experimental_matches": matches,
        "experimental_match_count": len(matches),
        "baseline_custom_creator_present": bool(baseline_matches),
        "baseline_matches": baseline_matches,
        "creator_conflict": len(matches) != 1,
        "all_creators": creators,
    }


def enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value).replace("DataType.", "").replace("TensorIOMode.", "")


def dims_list(dims: Any) -> list[int]:
    return [int(value) for value in dims]


def engine_io(trt: Any, engine: Any) -> list[dict[str, Any]]:
    records = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        records.append(
            {
                "index": index,
                "name": name,
                "mode": enum_name(engine.get_tensor_mode(name)),
                "dtype": enum_name(engine.get_tensor_dtype(name)),
                "shape": dims_list(engine.get_tensor_shape(name)),
            }
        )
    actual = {
        row["name"]: {key: row[key] for key in ("mode", "dtype", "shape")}
        for row in records
    }
    if actual != EXPECTED_IO:
        raise RuntimeError(f"Candidate I/O mismatch: {actual}")
    return records
