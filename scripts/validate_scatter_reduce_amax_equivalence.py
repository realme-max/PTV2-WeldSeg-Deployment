"""Audit include_self equivalence for voxel feature amax pooling.

The old reduction ignores a zero-initialized target.  The candidate includes a
negative-infinity-initialized target.  This script performs no ONNX export.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch


DATA_ROOT = PROJECT_ROOT / "data" / "weld"
POINT_ROOT = DATA_ROOT / "000001"
PREDICTIONS_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_evaluation"
    / "20260714_160831_945091_historical_checkpoint"
    / "predictions"
)
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "scatter_reduce_amax_equivalence"
VOXEL_SPECS = ((0.06, 96), (0.13, 192), (0.325, 384), (0.8125, 512))
FIXED_NPZ_NAMES = (
    "val_00_weld_7",
    "val_01_weld_61",
    "val_02_weld_49",
    "test_00_weld_65",
    "test_01_weld_30",
    "test_02_weld_28",
)
RTOL = 1e-5
ATOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def make_run_dir(requested: Path | None) -> Path:
    if requested is not None:
        run_dir = requested.resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir
    run_dir = ARTIFACTS_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_audit"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"scatter_reduce_amax.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "validation.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    normalized = xyz.astype(np.float32, copy=True)
    normalized -= normalized.mean(axis=0, keepdims=True)
    radius = np.sqrt(np.sum(normalized**2, axis=1)).max()
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"Invalid normalization radius: {radius}")
    normalized /= radius
    return normalized


def make_features(
    xyz: np.ndarray,
    labels: np.ndarray | None,
    feature_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if labels is None:
        label_column = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    else:
        label_column = labels.astype(np.float32, copy=False).reshape(-1, 1)
    order = np.linspace(-1.0, 1.0, xyz.shape[0], dtype=np.float32).reshape(-1, 1)
    base = np.concatenate((xyz.astype(np.float32, copy=False), label_column, order), axis=1)
    repeats = (feature_dim + base.shape[1] - 1) // base.shape[1]
    tiled = np.tile(base, (1, repeats))[:, :feature_dim]
    channel_offset = np.linspace(-2.0, 2.0, feature_dim, dtype=np.float32)
    feature_array = tiled + channel_offset.reshape(1, -1)
    points = torch.as_tensor(xyz, dtype=torch.float32, device=device).unsqueeze(0)
    features = torch.as_tensor(
        feature_array, dtype=torch.float32, device=device
    ).unsqueeze(0)
    return points, features


def feature_amax_old(
    inverse: torch.Tensor, flat_features: torch.Tensor, total_voxels: int
) -> torch.Tensor:
    feature_dim = flat_features.shape[1]
    index = inverse.unsqueeze(1).expand(-1, feature_dim)
    return torch.zeros(
        (total_voxels, feature_dim),
        device=flat_features.device,
        dtype=flat_features.dtype,
    ).scatter_reduce(
        0, index, flat_features, reduce="amax", include_self=False
    )


def feature_amax_candidate(
    inverse: torch.Tensor, flat_features: torch.Tensor, total_voxels: int
) -> torch.Tensor:
    feature_dim = flat_features.shape[1]
    index = inverse.unsqueeze(1).expand(-1, feature_dim)
    return torch.full(
        (total_voxels, feature_dim),
        float("-inf"),
        device=flat_features.device,
        dtype=flat_features.dtype,
    ).scatter_reduce(
        0, index, flat_features, reduce="amax", include_self=True
    )


def pool_variant(
    points: torch.Tensor,
    features: torch.Tensor,
    voxel_size: float,
    *,
    candidate: bool,
) -> dict[str, torch.Tensor]:
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] == 0:
        raise ValueError(f"Expected non-empty points [B,N,3], got {tuple(points.shape)}")
    if features.ndim != 3 or features.shape[:2] != points.shape[:2]:
        raise ValueError(f"Invalid features shape: {tuple(features.shape)}")
    if features.dtype != torch.float32 or not bool(torch.isfinite(features).all().item()):
        raise ValueError("The audited deployment domain requires finite float32 features")

    batch_size, points_per_batch, _ = points.shape
    size = torch.full((1, 1, 3), voxel_size, device=points.device, dtype=points.dtype)
    start = torch.amin(points, dim=1)
    end = torch.amax(points, dim=1)
    coordinates = torch.floor((points - start.unsqueeze(1)) / size).to(torch.int64)
    extents = torch.floor((end - start) / size.reshape(1, 3)).to(torch.int64) + 1
    strides = torch.stack(
        (
            torch.ones_like(extents[:, 0]),
            extents[:, 0],
            extents[:, 0] * extents[:, 1],
        ),
        dim=1,
    )
    local_keys = torch.sum(coordinates * strides.unsqueeze(1), dim=2)
    capacities = torch.prod(extents, dim=1)
    offsets = torch.cumsum(capacities, dim=0) - capacities
    global_keys = local_keys + offsets.unsqueeze(1)
    unique_global_keys, inverse = torch.unique(
        global_keys.reshape(-1), sorted=True, return_inverse=True
    )
    total_voxels = unique_global_keys.shape[0]
    point_batch_ids = (
        torch.arange(batch_size, device=points.device, dtype=torch.int64)
        .reshape(batch_size, 1)
        .expand(batch_size, points_per_batch)
        .reshape(-1)
    )
    unique_batch_ids = torch.full(
        (total_voxels,),
        torch.iinfo(torch.int64).max,
        device=points.device,
        dtype=torch.int64,
    ).scatter_reduce(
        0, inverse, point_batch_ids, reduce="amin", include_self=True
    )
    counts = torch.zeros(
        total_voxels, device=points.device, dtype=torch.int64
    ).scatter_add(0, inverse, torch.ones_like(inverse))
    if not bool(torch.all(counts > 0).item()):
        raise AssertionError("A materialized unique voxel has no points")

    flat_features = features.reshape(-1, features.shape[2])
    if candidate:
        pooled_features = feature_amax_candidate(inverse, flat_features, total_voxels)
    else:
        pooled_features = feature_amax_old(inverse, flat_features, total_voxels)

    unique_local_keys = unique_global_keys - offsets.index_select(0, unique_batch_ids)
    voxel_count_per_batch = torch.zeros(
        batch_size, device=points.device, dtype=torch.int64
    ).scatter_add(0, unique_batch_ids, torch.ones_like(unique_batch_ids))
    batch_prefix = torch.cumsum(voxel_count_per_batch, dim=0) - voxel_count_per_batch
    local_unique_indices = (
        torch.arange(total_voxels, device=points.device, dtype=torch.int64)
        - batch_prefix.index_select(0, unique_batch_ids)
    )
    point_to_voxel = (
        inverse - batch_prefix.index_select(0, point_batch_ids)
    ).reshape(batch_size, points_per_batch)
    minimum = torch.amin(voxel_count_per_batch)
    retained = local_unique_indices < minimum
    retained_keys = unique_local_keys[retained]
    retained_batch_ids = unique_batch_ids[retained]
    retained_extents = extents.index_select(0, retained_batch_ids)
    retained_coordinates = torch.stack(
        (
            torch.remainder(retained_keys, retained_extents[:, 0]),
            torch.remainder(
                torch.div(retained_keys, retained_extents[:, 0], rounding_mode="floor"),
                retained_extents[:, 1],
            ),
            torch.div(
                retained_keys,
                retained_extents[:, 0] * retained_extents[:, 1],
                rounding_mode="floor",
            ),
        ),
        dim=1,
    ).reshape(batch_size, minimum, 3)
    return {
        "start": start,
        "extents": extents,
        "unique_global_keys": unique_global_keys,
        "unique_local_keys": unique_local_keys,
        "unique_batch_ids": unique_batch_ids,
        "counts": counts,
        "voxel_count_per_batch": voxel_count_per_batch,
        "point_to_voxel": point_to_voxel,
        "retained_coordinates": retained_coordinates,
        "pooled_features": pooled_features[retained].reshape(
            batch_size, minimum, features.shape[2]
        ),
    }


def compare_pool(
    name: str,
    points: torch.Tensor,
    features: torch.Tensor,
    voxel_size: float,
) -> dict[str, Any]:
    old = pool_variant(points, features, voxel_size, candidate=False)
    new = pool_variant(points, features, voxel_size, candidate=True)
    discrete_fields = (
        "start",
        "extents",
        "unique_global_keys",
        "unique_local_keys",
        "unique_batch_ids",
        "counts",
        "voxel_count_per_batch",
        "point_to_voxel",
        "retained_coordinates",
    )
    discrete_checks = {
        field: bool(torch.equal(old[field], new[field])) for field in discrete_fields
    }
    feature_error = float(
        (old["pooled_features"] - new["pooled_features"]).abs().max().item()
    )
    feature_allclose = bool(
        torch.allclose(
            old["pooled_features"], new["pooled_features"], rtol=RTOL, atol=ATOL
        )
    )
    feature_exact = bool(torch.equal(old["pooled_features"], new["pooled_features"]))
    if not all(discrete_checks.values()) or not feature_allclose:
        raise AssertionError(
            f"{name} size={voxel_size} failed: discrete={discrete_checks}, "
            f"feature_allclose={feature_allclose}, max_error={feature_error}"
        )
    return {
        "name": name,
        "voxel_size": voxel_size,
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "voxel_count_per_batch": old["voxel_count_per_batch"].cpu().tolist(),
        "materialized_empty_voxels": int((old["counts"] == 0).sum().item()),
        "discrete_checks": discrete_checks,
        "pooled_feature_allclose": feature_allclose,
        "pooled_feature_exact": feature_exact,
        "pooled_feature_max_abs_error": feature_error,
    }


def artificial_cases(device: torch.device) -> list[tuple[str, torch.Tensor, torch.Tensor, float]]:
    cases: list[tuple[str, torch.Tensor, torch.Tensor, float]] = []

    def add(name: str, xyz: list[list[float]], feature_values: list[list[float]], size: float = 1.0) -> None:
        cases.append(
            (
                name,
                torch.tensor([xyz], dtype=torch.float32, device=device),
                torch.tensor([feature_values], dtype=torch.float32, device=device),
                size,
            )
        )

    add(
        "one_voxel_multiple_features",
        [[0, 0, 0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]],
        [[1, -2, 3], [4, -5, 2], [0, -1, 8]],
    )
    add("one_voxel_single_feature", [[-2, 1, 3]], [[-7, 0.5, 9]])
    add(
        "extreme_finite_features",
        [[0, 0, 0], [0.1, 0, 0]],
        [[-1.0e30, 1.0e30, -3.4e20], [1.0e30, -1.0e30, 3.4e20]],
    )
    add(
        "all_negative_features",
        [[-3, -2, -1], [-2.9, -1.9, -0.9], [-2.8, -1.8, -0.8]],
        [[-1, -2, -3], [-4, -0.5, -9], [-2, -7, -1]],
    )
    multi_points = torch.tensor(
        [
            [[-1, 0, 0], [-0.9, 0, 0], [1, 0, 0], [1.1, 0, 0]],
            [[10, -2, 1], [10.1, -2, 1], [12, -1, 2], [12.1, -1, 2]],
        ],
        dtype=torch.float32,
        device=device,
    )
    multi_features = torch.tensor(
        [
            [[-4, 1], [-3, 2], [-2, 3], [-1, 4]],
            [[8, -8], [7, -7], [6, -6], [5, -5]],
        ],
        dtype=torch.float32,
        device=device,
    )
    cases.append(("multi_batch", multi_points, multi_features, 0.5))
    return cases


def new_summary(sample_count: int) -> dict[str, Any]:
    return {
        "sample_count": sample_count,
        "comparison_count": 0,
        "feature_dims": [],
        "max_pooled_feature_abs_error": 0.0,
        "max_materialized_empty_voxels": 0,
        "all_discrete_checks_exact": True,
        "all_pooled_features_allclose": True,
        "all_pooled_features_exact": True,
    }


def update_summary(summary: dict[str, Any], result: dict[str, Any]) -> None:
    summary["comparison_count"] += 1
    summary["feature_dims"].append(result["feature_shape"][2])
    summary["max_pooled_feature_abs_error"] = max(
        summary["max_pooled_feature_abs_error"], result["pooled_feature_max_abs_error"]
    )
    summary["max_materialized_empty_voxels"] = max(
        summary["max_materialized_empty_voxels"], result["materialized_empty_voxels"]
    )
    summary["all_discrete_checks_exact"] = summary["all_discrete_checks_exact"] and all(
        result["discrete_checks"].values()
    )
    summary["all_pooled_features_allclose"] = (
        summary["all_pooled_features_allclose"] and result["pooled_feature_allclose"]
    )
    summary["all_pooled_features_exact"] = (
        summary["all_pooled_features_exact"] and result["pooled_feature_exact"]
    )


def main() -> int:
    args = parse_args()
    run_dir = make_run_dir(args.run_dir)
    logger = make_logger(run_dir)
    result_path = run_dir / "scatter_reduce_amax_equivalence.json"
    payload: dict[str, Any] = {
        "status": "started",
        "device": args.device,
        "old": "zeros(FP32).scatter_reduce(amax, include_self=False)",
        "candidate": "full(-inf, FP32).scatter_reduce(amax, include_self=True)",
        "voxel_specs": [
            {"voxel_size": size, "feature_dim": dim} for size, dim in VOXEL_SPECS
        ],
        "rtol": RTOL,
        "atol": ATOL,
    }
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        logger.info(
            "PROJECT_ROOT=%s python=%s torch=%s device=%s GPU=%s",
            PROJECT_ROOT,
            sys.executable,
            torch.__version__,
            device,
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        )

        artificial_results = [
            compare_pool(name, points, features, size)
            for name, points, features, size in artificial_cases(device)
        ]
        payload["artificial_results"] = artificial_results
        logger.info("Artificial feature pooling cases passed: %d", len(artificial_results))

        weld_files = sorted(POINT_ROOT.glob("weld_*.txt"))
        if len(weld_files) != 90:
            raise AssertionError(f"Expected 90 weld files, found {len(weld_files)}")
        weld_summary = new_summary(len(weld_files))
        for path in weld_files:
            data = np.loadtxt(path, dtype=np.float32)
            if data.ndim != 2 or data.shape[1] != 4:
                raise ValueError(f"Expected four columns in {path}, got {data.shape}")
            xyz = normalize_xyz(data[:, :3])
            for voxel_size, feature_dim in VOXEL_SPECS:
                points, features = make_features(
                    xyz, data[:, 3], feature_dim, device
                )
                update_summary(
                    weld_summary,
                    compare_pool(path.stem, points, features, voxel_size),
                )
        weld_summary["feature_dims"] = sorted(set(weld_summary["feature_dims"]))
        payload["all_90_weld_files"] = weld_summary
        logger.info("All 90 weld files passed: %s", weld_summary)

        fixed_summary = new_summary(len(FIXED_NPZ_NAMES))
        for name in FIXED_NPZ_NAMES:
            path = PREDICTIONS_ROOT / f"{name}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path) as data:
                xyz = np.asarray(data["normalized_xyz"], dtype=np.float32)
                label_key = (
                    "ground_truth_labels"
                    if "ground_truth_labels" in data.files
                    else "ground_truth"
                )
                labels = np.asarray(data[label_key], dtype=np.float32)
            for voxel_size, feature_dim in VOXEL_SPECS:
                points, features = make_features(xyz, labels, feature_dim, device)
                update_summary(
                    fixed_summary,
                    compare_pool(name, points, features, voxel_size),
                )
        fixed_summary["feature_dims"] = sorted(set(fixed_summary["feature_dims"]))
        payload["fixed_6_npz"] = fixed_summary
        logger.info("Fixed 6 NPZ passed: %s", fixed_summary)

        if weld_summary["max_materialized_empty_voxels"] != 0:
            raise AssertionError("Actual weld data materialized an empty voxel")
        if fixed_summary["max_materialized_empty_voxels"] != 0:
            raise AssertionError("Fixed NPZ data materialized an empty voxel")
        payload["mathematical_domain"] = {
            "src_dtype": "torch.float32",
            "src_shape": "[B*N,F]",
            "index_shape": "[B*N,F] expanded from inverse_global",
            "target_shape": "[number_of_unique_voxels,F]",
            "all_materialized_voxels_nonempty": True,
            "all_audited_features_finite": True,
        }
        payload["status"] = "SCATTER_REDUCE_AMAX_EQUIVALENCE_PASSED"
        payload["replacement_authorized_by_audit"] = True
        payload["onnx_export_run"] = False
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("SCATTER_REDUCE_AMAX_EQUIVALENCE_PASSED")
        print(f"ARTIFACT_DIR={run_dir}")
        print("SCATTER_REDUCE_AMAX_EQUIVALENCE_PASSED")
        return 0
    except Exception as exc:
        payload.update(
            {
                "status": "SCATTER_REDUCE_AMAX_EQUIVALENCE_FAILED",
                "replacement_authorized_by_audit": False,
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "traceback": traceback.format_exc(),
                "onnx_export_run": False,
            }
        )
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.exception("SCATTER_REDUCE_AMAX_EQUIVALENCE_FAILED")
        print("SCATTER_REDUCE_AMAX_EQUIVALENCE_FAILED")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
