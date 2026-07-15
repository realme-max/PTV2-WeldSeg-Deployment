"""Validate standard-operator voxel pooling against torch_cluster::grid."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch_cluster import grid_cluster

from deployment.onnx_voxel_pool import standard_voxel_pool_with_metadata


RTOL = 1e-5
ATOL = 1e-6
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "gcn_res_standard_ops"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def make_run_dir(requested: Path | None) -> Path:
    if requested is not None:
        path = requested.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = ARTIFACTS_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_equivalence"
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def reference_pool(
    points: torch.Tensor, features: torch.Tensor, voxel_size: torch.Tensor
) -> dict[str, Any]:
    pooled_points_per_batch: list[torch.Tensor] = []
    pooled_features_per_batch: list[torch.Tensor] = []
    counts_per_batch: list[torch.Tensor] = []
    coordinates_per_batch: list[torch.Tensor] = []
    inverse_per_batch: list[torch.Tensor] = []
    unique_keys_per_batch: list[torch.Tensor] = []
    starts: list[torch.Tensor] = []
    ends: list[torch.Tensor] = []
    extents_per_batch: list[torch.Tensor] = []

    for batch_index in range(points.shape[0]):
        xyz = points[batch_index]
        feat = features[batch_index]
        keys = grid_cluster(xyz, voxel_size)
        unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        voxel_count = unique_keys.shape[0]
        pooled_features = torch.zeros(
            (voxel_count, feat.shape[1]), device=feat.device, dtype=feat.dtype
        ).scatter_reduce(
            0,
            inverse[:, None].expand(-1, feat.shape[1]),
            feat,
            reduce="amax",
            include_self=False,
        )
        summed_xyz = torch.zeros(
            (voxel_count, 3), device=xyz.device, dtype=xyz.dtype
        ).scatter_add(0, inverse[:, None].expand(-1, 3), xyz)
        counts = torch.bincount(inverse, minlength=voxel_count)
        pooled_xyz = summed_xyz / counts.to(xyz.dtype).unsqueeze(1)

        start = torch.amin(xyz, dim=0)
        end = torch.amax(xyz, dim=0)
        extents = torch.trunc((end - start) / voxel_size).to(torch.int64) + 1
        coord_x = torch.remainder(unique_keys, extents[0])
        coord_y = torch.remainder(
            torch.div(unique_keys, extents[0], rounding_mode="floor"), extents[1]
        )
        coord_z = torch.div(
            unique_keys, extents[0] * extents[1], rounding_mode="floor"
        )

        pooled_points_per_batch.append(pooled_xyz)
        pooled_features_per_batch.append(pooled_features)
        counts_per_batch.append(counts)
        coordinates_per_batch.append(torch.stack((coord_x, coord_y, coord_z), dim=1))
        inverse_per_batch.append(inverse)
        unique_keys_per_batch.append(unique_keys)
        starts.append(start)
        ends.append(end)
        extents_per_batch.append(extents)

    minimum = min(item.shape[0] for item in pooled_points_per_batch)
    return {
        "pooled_points": torch.stack([item[:minimum] for item in pooled_points_per_batch]),
        "pooled_features": torch.stack([item[:minimum] for item in pooled_features_per_batch]),
        "retained_counts": torch.stack([item[:minimum] for item in counts_per_batch]),
        "retained_coordinates": torch.stack(
            [item[:minimum] for item in coordinates_per_batch]
        ),
        "inverse": inverse_per_batch,
        "unique_keys": unique_keys_per_batch,
        "voxel_counts": torch.tensor(
            [item.shape[0] for item in unique_keys_per_batch],
            device=points.device,
            dtype=torch.int64,
        ),
        "start": torch.stack(starts),
        "end": torch.stack(ends),
        "extents": torch.stack(extents_per_batch),
    }


def artificial_cases(device: torch.device) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    def tensor(values: list[list[float]]) -> torch.Tensor:
        return torch.tensor(values, device=device, dtype=torch.float32)

    cases: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        (
            "all_points_same_voxel",
            tensor([[0, 0, 0], [0.1, 0.2, 0.3], [0.4, 0.1, 0.2]])[None],
            tensor([1.0, 1.0, 1.0]),
        ),
        (
            "every_point_different_voxel",
            tensor([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]])[None],
            tensor([1.0, 1.0, 1.0]),
        ),
        (
            "two_voxels",
            tensor([[0, 0, 0], [0.2, 0.1, 0], [1.2, 0, 0], [1.4, 0.2, 0]])[None],
            tensor([1.0, 1.0, 1.0]),
        ),
        (
            "voxel_boundaries",
            tensor(
                [
                    [0, 0, 0],
                    [0.9999999, 0, 0],
                    [1.0, 0, 0],
                    [1.0000001, 0, 0],
                    [2.0, 0, 0],
                ]
            )[None],
            tensor([1.0, 1.0, 1.0]),
        ),
        (
            "negative_coordinates",
            tensor([[-2.0, -1.0, -0.2], [-1.1, -0.2, -0.1], [-0.1, 0.1, 0.2]])[None],
            tensor([0.5, 0.5, 0.5]),
        ),
        (
            "duplicate_points",
            tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]])[None],
            tensor([0.5, 0.5, 0.5]),
        ),
        (
            "multi_batch",
            torch.stack(
                (
                    tensor([[0, 0, 0], [0.2, 0, 0], [1.2, 0, 0], [2.2, 0, 0]]),
                    tensor([[-3, 0, 0], [-2.8, 0, 0], [-1.8, 0, 0], [-0.8, 0, 0]]),
                )
            ),
            tensor([1.0, 1.0, 1.0]),
        ),
        (
            "unequal_voxel_counts_across_batch",
            torch.stack(
                (
                    tensor([[0, 0, 0], [0.1, 0, 0], [1.1, 0, 0], [2.1, 0, 0]]),
                    tensor([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [1.1, 0, 0]]),
                )
            ),
            tensor([1.0, 1.0, 1.0]),
        ),
    ]
    return cases


def validate_case(
    name: str, points: torch.Tensor, voxel_size: torch.Tensor
) -> dict[str, Any]:
    feature_values = torch.arange(
        points.shape[0] * points.shape[1] * 5,
        device=points.device,
        dtype=torch.float32,
    ).reshape(points.shape[0], points.shape[1], 5)
    features = (feature_values - 9.5) / 7.0
    reference = reference_pool(points, features, voxel_size)
    actual = standard_voxel_pool_with_metadata(points, features, voxel_size)

    checks: dict[str, bool] = {
        "voxel_count": torch.equal(reference["voxel_counts"], actual.voxel_count_per_batch),
        "start": torch.equal(reference["start"], actual.start),
        "end": torch.equal(reference["end"], actual.end),
        "extents": torch.equal(reference["extents"], actual.extents),
        "coordinates": torch.equal(
            reference["retained_coordinates"], actual.retained_voxel_coordinates
        ),
        "counts": torch.equal(reference["retained_counts"], actual.retained_voxel_counts),
        "pooled_xyz": torch.allclose(
            reference["pooled_points"], actual.pooled_points, rtol=RTOL, atol=ATOL
        ),
        "pooled_features": torch.allclose(
            reference["pooled_features"], actual.pooled_features, rtol=RTOL, atol=ATOL
        ),
        "finite": bool(
            torch.isfinite(actual.pooled_points).all()
            and torch.isfinite(actual.pooled_features).all()
        ),
    }
    membership_checks: list[bool] = []
    direct_inverse_checks: list[bool] = []
    key_checks: list[bool] = []
    for batch_index, reference_inverse in enumerate(reference["inverse"]):
        actual_inverse = actual.point_to_voxel[batch_index]
        membership_checks.append(
            torch.equal(
                reference_inverse[:, None] == reference_inverse[None, :],
                actual_inverse[:, None] == actual_inverse[None, :],
            )
        )
        direct_inverse_checks.append(torch.equal(reference_inverse, actual_inverse))
        actual_keys = actual.unique_local_keys[
            actual.unique_batch_ids == batch_index
        ]
        key_checks.append(torch.equal(reference["unique_keys"][batch_index], actual_keys))
    checks["cluster_membership_semantic"] = all(membership_checks)
    checks["point_to_voxel_direct"] = all(direct_inverse_checks)
    checks["sorted_unique_keys"] = all(key_checks)
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise AssertionError(f"{name} failed checks: {failed}")
    return {
        "name": name,
        "batch_size": int(points.shape[0]),
        "points_per_batch": int(points.shape[1]),
        "voxel_size": voxel_size.detach().cpu().tolist(),
        "voxel_count_per_batch": actual.voxel_count_per_batch.detach().cpu().tolist(),
        "retained_voxel_count": int(actual.pooled_points.shape[1]),
        "pooled_xyz_max_abs_error": float(
            (reference["pooled_points"] - actual.pooled_points).abs().max().item()
        ),
        "pooled_features_max_abs_error": float(
            (reference["pooled_features"] - actual.pooled_features).abs().max().item()
        ),
        "checks": checks,
        "status": "passed",
    }


def main() -> int:
    args = parse_args()
    run_dir = make_run_dir(args.run_dir)
    output_path = run_dir / "voxel_pool_artificial_tests.json"
    log_path = run_dir / "voxel_pool_artificial_tests.log"
    payload: dict[str, Any] = {
        "status": "started",
        "device": args.device,
        "rtol": RTOL,
        "atol": ATOL,
        "implementation": str(PROJECT_ROOT / "deployment" / "onnx_voxel_pool.py"),
        "reference": "torch_cluster.grid_cluster 1.6.3",
        "cases": [],
    }
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        lines = [f"run_dir={run_dir}", f"device={device}"]
        for name, points, voxel_size in artificial_cases(device):
            result = validate_case(name, points, voxel_size)
            payload["cases"].append(result)
            lines.append(json.dumps(result, ensure_ascii=False))
            print(f"PASS {name}: voxels={result['voxel_count_per_batch']}")
        payload["status"] = "STANDARD_OPS_VOXEL_POOL_EQUIVALENCE_PASSED"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("STANDARD_OPS_VOXEL_POOL_EQUIVALENCE_PASSED")
        print(f"ARTIFACT_DIR={run_dir}")
        return 0
    except Exception as exc:
        payload.update(
            {
                "status": "STANDARD_OPS_VOXEL_POOL_EQUIVALENCE_FAILED",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log_path.write_text(payload["traceback"], encoding="utf-8")
        print("STANDARD_OPS_VOXEL_POOL_EQUIVALENCE_FAILED")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
