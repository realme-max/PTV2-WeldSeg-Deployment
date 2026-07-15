"""Audit include_self equivalence for the voxel-to-batch amin reduction.

The candidate uses the maximum int64 value as the type-preserving +infinity
identity.  No ONNX export is performed by this script.
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
SPLIT_ROOT = DATA_ROOT / "train_test_split"
PREDICTIONS_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "gcn_res_evaluation"
    / "20260714_160831_945091_historical_checkpoint"
    / "predictions"
)
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "scatter_reduce_amin_equivalence"
VOXEL_SIZES = (0.06, 0.13, 0.325, 0.8125)
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
INT64_AMIN_IDENTITY = torch.iinfo(torch.int64).max


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
    logger = logging.getLogger(f"scatter_reduce_amin.{run_dir.name}")
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


def old_amin(index: torch.Tensor, source: torch.Tensor, target_size: int) -> torch.Tensor:
    return torch.zeros(
        target_size, device=source.device, dtype=torch.int64
    ).scatter_reduce(0, index, source, reduce="amin", include_self=False)


def candidate_amin(
    index: torch.Tensor, source: torch.Tensor, target_size: int
) -> torch.Tensor:
    # torch.inf is not representable in the int64 dtype required by batch IDs.
    # int64 max is the exact amin identity on source values in [0, B-1].
    return torch.full(
        (target_size,),
        INT64_AMIN_IDENTITY,
        device=source.device,
        dtype=torch.int64,
    ).scatter_reduce(0, index, source, reduce="amin", include_self=True)


def reduce_case(
    name: str,
    index_values: list[int],
    source_values: list[int],
    target_size: int,
    device: torch.device,
) -> dict[str, Any]:
    index = torch.tensor(index_values, dtype=torch.int64, device=device)
    source = torch.tensor(source_values, dtype=torch.int64, device=device)
    old = old_amin(index, source, target_size)
    new = candidate_amin(index, source, target_size)
    counts = torch.zeros(target_size, dtype=torch.int64, device=device).scatter_add(
        0, index, torch.ones_like(index)
    )
    occupied = counts > 0
    empty = ~occupied
    occupied_equal = bool(torch.equal(old[occupied], new[occupied]))
    if not occupied_equal:
        raise AssertionError(f"{name}: occupied targets differ")
    return {
        "name": name,
        "index": index.cpu().tolist(),
        "source": source.cpu().tolist(),
        "target_size": target_size,
        "old": old.cpu().tolist(),
        "candidate": new.cpu().tolist(),
        "counts": counts.cpu().tolist(),
        "occupied_targets_equal": occupied_equal,
        "empty_target_count": int(empty.sum().item()),
        "raw_outputs_equal": bool(torch.equal(old, new)),
        "empty_targets_retain_old_zero": bool(
            empty.any() and torch.all(old[empty] == 0).item()
        ),
        "empty_targets_retain_candidate_identity": bool(
            empty.any() and torch.all(new[empty] == INT64_AMIN_IDENTITY).item()
        ),
    }


def pool_variant(
    points: torch.Tensor,
    features: torch.Tensor,
    voxel_size: torch.Tensor,
    *,
    candidate: bool,
) -> dict[str, torch.Tensor]:
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] == 0:
        raise ValueError(f"Expected non-empty points [B,N,3], got {tuple(points.shape)}")
    if features.ndim != 3 or features.shape[:2] != points.shape[:2]:
        raise ValueError(f"Invalid features shape: {tuple(features.shape)}")
    batch_size, points_per_batch, _ = points.shape
    voxel_size = voxel_size.to(points.device, points.dtype).reshape(1, 1, 3)
    start = torch.amin(points, dim=1)
    end = torch.amax(points, dim=1)
    coordinates = torch.floor((points - start.unsqueeze(1)) / voxel_size).to(torch.int64)
    extents = (
        torch.floor((end - start) / voxel_size.reshape(1, 3)).to(torch.int64) + 1
    )
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
    if candidate:
        unique_batch_ids = candidate_amin(inverse, point_batch_ids, total_voxels)
    else:
        unique_batch_ids = old_amin(inverse, point_batch_ids, total_voxels)

    counts = torch.zeros(
        total_voxels, device=points.device, dtype=torch.int64
    ).scatter_add(0, inverse, torch.ones_like(inverse))
    if not bool(torch.all(counts > 0).item()):
        raise AssertionError("A materialized unique voxel has no source points")
    if not bool(torch.all(unique_batch_ids != INT64_AMIN_IDENTITY).item()):
        raise AssertionError("A materialized unique voxel retained the amin identity")

    unique_local_keys = unique_global_keys - offsets.index_select(0, unique_batch_ids)
    summed_points = torch.zeros(
        (total_voxels, 3), device=points.device, dtype=points.dtype
    ).scatter_add(
        0,
        inverse.unsqueeze(1).expand(-1, 3),
        points.reshape(-1, 3),
    )
    mean_points = summed_points / counts.to(points.dtype).unsqueeze(1)
    feature_dim = features.shape[2]
    pooled_features = torch.zeros(
        (total_voxels, feature_dim), device=features.device, dtype=features.dtype
    ).scatter_reduce(
        0,
        inverse.unsqueeze(1).expand(-1, feature_dim),
        features.reshape(-1, feature_dim),
        reduce="amax",
        include_self=False,
    )
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
                torch.div(
                    retained_keys, retained_extents[:, 0], rounding_mode="floor"
                ),
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
        "end": end,
        "extents": extents,
        "unique_global_keys": unique_global_keys,
        "unique_local_keys": unique_local_keys,
        "unique_batch_ids": unique_batch_ids,
        "counts": counts,
        "voxel_count_per_batch": voxel_count_per_batch,
        "point_to_voxel": point_to_voxel,
        "retained_coordinates": retained_coordinates,
        "pooled_points": mean_points[retained].reshape(batch_size, minimum, 3),
        "pooled_features": pooled_features[retained].reshape(
            batch_size, minimum, feature_dim
        ),
    }


def compare_pool(
    name: str,
    points: torch.Tensor,
    features: torch.Tensor,
    voxel_size: float,
) -> dict[str, Any]:
    size = torch.full((3,), voxel_size, device=points.device, dtype=points.dtype)
    old = pool_variant(points, features, size, candidate=False)
    new = pool_variant(points, features, size, candidate=True)
    exact_fields = (
        "start",
        "end",
        "extents",
        "unique_global_keys",
        "unique_local_keys",
        "unique_batch_ids",
        "counts",
        "voxel_count_per_batch",
        "point_to_voxel",
        "retained_coordinates",
    )
    exact_checks = {field: bool(torch.equal(old[field], new[field])) for field in exact_fields}
    pooled_feature_error = float(
        (old["pooled_features"] - new["pooled_features"]).abs().max().item()
    )
    pooled_point_error = float(
        (old["pooled_points"] - new["pooled_points"]).abs().max().item()
    )
    pooled_features_equal = bool(torch.equal(old["pooled_features"], new["pooled_features"]))
    pooled_points_close = bool(
        torch.allclose(old["pooled_points"], new["pooled_points"], rtol=RTOL, atol=ATOL)
    )
    if not all(exact_checks.values()) or not pooled_features_equal or not pooled_points_close:
        raise AssertionError(
            f"{name} size={voxel_size} mismatch: exact={exact_checks}, "
            f"feature_equal={pooled_features_equal}, point_close={pooled_points_close}"
        )
    return {
        "name": name,
        "batch_size": int(points.shape[0]),
        "points_per_batch": int(points.shape[1]),
        "voxel_size": voxel_size,
        "voxel_count_per_batch": old["voxel_count_per_batch"].cpu().tolist(),
        "materialized_empty_voxels": int((old["counts"] == 0).sum().item()),
        "exact_checks": exact_checks,
        "pooled_features_exact": pooled_features_equal,
        "pooled_features_max_abs_error": pooled_feature_error,
        "pooled_points_max_abs_error": pooled_point_error,
    }


def normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    normalized = xyz.astype(np.float32, copy=True)
    normalized -= normalized.mean(axis=0, keepdims=True)
    radius = np.sqrt(np.sum(normalized**2, axis=1)).max()
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"Invalid radius: {radius}")
    normalized /= radius
    return normalized


def make_features(
    xyz: np.ndarray, labels: np.ndarray | None, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    points = torch.as_tensor(xyz, dtype=torch.float32, device=device).unsqueeze(0)
    if labels is None:
        label_column = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    else:
        label_column = labels.astype(np.float32, copy=False).reshape(-1, 1)
    order = np.linspace(0.0, 1.0, xyz.shape[0], dtype=np.float32).reshape(-1, 1)
    feature_array = np.concatenate((xyz, label_column, order), axis=1)
    features = torch.as_tensor(
        feature_array, dtype=torch.float32, device=device
    ).unsqueeze(0)
    return points, features


def artificial_pool_cases(device: torch.device) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    def sample(values: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
        xyz = np.asarray(values, dtype=np.float32)
        return make_features(xyz, None, device)

    cases = [
        ("one_voxel_multiple_points", *sample([[0, 0, 0], [0.1, 0.2, 0.1], [0.2, 0.1, 0.2]])),
        ("one_voxel_one_point", *sample([[1.25, -2.5, 4.0]])),
        ("extreme_coordinates", *sample([[-1000, -500, -250], [1000, 500, 250], [0, 0, 0], [999.9, -499.9, 249.9]])),
        ("negative_coordinates", *sample([[-3, -2, -1], [-2.9, -1.2, -0.1], [-0.2, -0.1, -0.05], [0.1, 0.2, 0.3]])),
    ]
    batch_points = torch.tensor(
        [
            [[-2.0, 0.0, 0.0], [-1.9, 0.0, 0.0], [-0.5, 0.2, 0.0], [0.5, 0.0, 0.0]],
            [[10.0, -3.0, 1.0], [10.1, -3.0, 1.0], [11.5, -2.0, 1.0], [12.5, -1.0, 2.0]],
        ],
        dtype=torch.float32,
        device=device,
    )
    batch_features = torch.cat(
        (batch_points, torch.arange(8, device=device, dtype=torch.float32).reshape(2, 4, 1)),
        dim=2,
    )
    cases.append(("multi_batch", batch_points, batch_features))
    return cases


def update_summary(summary: dict[str, Any], result: dict[str, Any]) -> None:
    summary["comparison_count"] += 1
    summary["max_pooled_feature_abs_error"] = max(
        summary["max_pooled_feature_abs_error"], result["pooled_features_max_abs_error"]
    )
    summary["max_pooled_point_abs_error"] = max(
        summary["max_pooled_point_abs_error"], result["pooled_points_max_abs_error"]
    )
    summary["max_materialized_empty_voxels"] = max(
        summary["max_materialized_empty_voxels"], result["materialized_empty_voxels"]
    )


def new_summary(sample_count: int) -> dict[str, Any]:
    return {
        "sample_count": sample_count,
        "comparison_count": 0,
        "max_pooled_feature_abs_error": 0.0,
        "max_pooled_point_abs_error": 0.0,
        "max_materialized_empty_voxels": 0,
        "all_discrete_checks_exact": True,
    }


def main() -> int:
    args = parse_args()
    run_dir = make_run_dir(args.run_dir)
    logger = make_logger(run_dir)
    result_path = run_dir / "scatter_reduce_amin_equivalence.json"
    payload: dict[str, Any] = {
        "status": "started",
        "device": args.device,
        "old": "zeros(int64).scatter_reduce(amin, include_self=False)",
        "candidate": "full(int64_max).scatter_reduce(amin, include_self=True)",
        "int64_amin_identity": INT64_AMIN_IDENTITY,
        "voxel_sizes": list(VOXEL_SIZES),
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

        payload["direct_reduce_cases"] = [
            reduce_case("one_voxel_multiple_points", [0, 0, 0], [0, 0, 0], 1, device),
            reduce_case("one_voxel_one_point", [0], [0], 1, device),
            reduce_case("empty_target_slots", [0, 2], [0, 1], 4, device),
            reduce_case(
                "extreme_valid_source",
                [0, 0],
                [INT64_AMIN_IDENTITY - 2, INT64_AMIN_IDENTITY - 1],
                1,
                device,
            ),
        ]
        empty_case = payload["direct_reduce_cases"][2]
        if empty_case["raw_outputs_equal"]:
            raise AssertionError("Empty artificial target unexpectedly produced equal raw outputs")
        payload["empty_voxel_conclusion"] = (
            "Raw outputs differ for manually allocated empty target slots, but such slots are "
            "unreachable because target_size is len(unique_global_keys) and every unique ID "
            "appears in inverse_global at least once."
        )

        artificial_results: list[dict[str, Any]] = []
        for name, points, features in artificial_pool_cases(device):
            for voxel_size in VOXEL_SIZES:
                artificial_results.append(
                    compare_pool(name, points, features, voxel_size)
                )
        payload["artificial_pool_results"] = artificial_results
        logger.info("Artificial pooling comparisons passed: %d", len(artificial_results))

        all_files = sorted(POINT_ROOT.glob("weld_*.txt"))
        if len(all_files) != 90:
            raise AssertionError(f"Expected 90 weld files, found {len(all_files)}")
        weld_summary = new_summary(len(all_files))
        for path in all_files:
            data = np.loadtxt(path, dtype=np.float32)
            if data.ndim != 2 or data.shape[1] != 4:
                raise ValueError(f"Expected four columns in {path}, got {data.shape}")
            xyz = normalize_xyz(data[:, :3])
            points, features = make_features(xyz, data[:, 3], device)
            for voxel_size in VOXEL_SIZES:
                update_summary(
                    weld_summary,
                    compare_pool(path.stem, points, features, voxel_size),
                )
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
            points, features = make_features(xyz, labels, device)
            for voxel_size in VOXEL_SIZES:
                update_summary(
                    fixed_summary,
                    compare_pool(name, points, features, voxel_size),
                )
        payload["fixed_6_npz"] = fixed_summary
        logger.info("Fixed 6 NPZ passed: %s", fixed_summary)

        if weld_summary["max_materialized_empty_voxels"] != 0:
            raise AssertionError("Actual weld data materialized an empty voxel")
        if fixed_summary["max_materialized_empty_voxels"] != 0:
            raise AssertionError("Fixed NPZ data materialized an empty voxel")

        payload["status"] = "SCATTER_REDUCE_AMIN_EQUIVALENCE_PASSED"
        payload["replacement_authorized_by_audit"] = True
        payload["onnx_export_run"] = False
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("SCATTER_REDUCE_AMIN_EQUIVALENCE_PASSED")
        print(f"ARTIFACT_DIR={run_dir}")
        print("SCATTER_REDUCE_AMIN_EQUIVALENCE_PASSED")
        return 0
    except Exception as exc:
        payload.update(
            {
                "status": "SCATTER_REDUCE_AMIN_EQUIVALENCE_FAILED",
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
        logger.exception("SCATTER_REDUCE_AMIN_EQUIVALENCE_FAILED")
        print("SCATTER_REDUCE_AMIN_EQUIVALENCE_FAILED")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
