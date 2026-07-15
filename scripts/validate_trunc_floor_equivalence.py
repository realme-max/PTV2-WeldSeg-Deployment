"""Audit trunc/floor equivalence for GCN_res weld voxel coordinates.

This script is intentionally independent from ONNX export.  It proves the
restricted-domain replacement used by deployment voxel pooling by checking all
90 weld files, the fixed 54/18/18 sub-dataset splits, and the six fixed NPZ
evaluation samples.  It does not modify the deployment implementation.
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
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "trunc_floor_equivalence"
VOXEL_SIZES = (0.06, 0.13, 0.325, 0.8125)
SPLITS = ("train", "val", "test")
FIXED_NPZ_NAMES = (
    "val_00_weld_7",
    "val_01_weld_61",
    "val_02_weld_49",
    "test_00_weld_65",
    "test_01_weld_30",
    "test_02_weld_28",
)


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
    logger = logging.getLogger(f"trunc_floor_equivalence.{run_dir.name}")
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


def load_split(split: str) -> list[Path]:
    split_path = SPLIT_ROOT / f"sub_shuffled_{split}_file_list.json"
    entries = json.loads(split_path.read_text(encoding="utf-8"))
    paths: list[Path] = []
    for entry in entries:
        parts = Path(entry).parts
        if len(parts) < 3 or parts[-2] != "000001":
            raise ValueError(f"Unexpected split entry in {split_path}: {entry!r}")
        path = POINT_ROOT / f"{parts[-1]}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path.resolve())
    return paths


def normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    normalized = xyz.astype(np.float32, copy=True)
    normalized -= normalized.mean(axis=0, keepdims=True)
    radius = np.sqrt(np.sum(normalized**2, axis=1)).max()
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"Invalid normalization radius: {radius}")
    normalized /= radius
    return normalized


def mixed_radix_keys(
    coordinates: torch.Tensor, extents: torch.Tensor
) -> torch.Tensor:
    strides = torch.stack(
        (
            torch.ones_like(extents[0]),
            extents[0],
            extents[0] * extents[1],
        )
    )
    return torch.sum(coordinates * strides.reshape(1, 3), dim=1)


def audit_xyz(
    xyz_numpy: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, Any]:
    if xyz_numpy.ndim != 2 or xyz_numpy.shape[1] != 3:
        raise ValueError(f"Expected XYZ [N,3], got {xyz_numpy.shape}")
    xyz = torch.as_tensor(xyz_numpy, dtype=torch.float32, device=device)
    if not bool(torch.isfinite(xyz).all()):
        raise FloatingPointError("XYZ contains NaN or Inf")

    start = torch.amin(xyz, dim=0)
    end = torch.amax(xyz, dim=0)
    shifted = xyz - start
    shifted_min = float(torch.amin(shifted).item())
    shifted_nonnegative = bool(torch.all(shifted >= 0).item())
    per_size: dict[str, Any] = {}

    for scalar_size in VOXEL_SIZES:
        voxel_size = torch.full(
            (3,), scalar_size, dtype=torch.float32, device=device
        )
        scaled = shifted / voxel_size
        scaled_min = float(torch.amin(scaled).item())
        scaled_nonnegative = bool(torch.all(scaled >= 0).item())

        trunc_coordinates = torch.trunc(scaled).to(torch.int64)
        floor_coordinates = torch.floor(scaled).to(torch.int64)
        trunc_extents = (
            torch.trunc((end - start) / voxel_size).to(torch.int64) + 1
        )
        floor_extents = (
            torch.floor((end - start) / voxel_size).to(torch.int64) + 1
        )

        trunc_keys = mixed_radix_keys(trunc_coordinates, trunc_extents)
        floor_keys = mixed_radix_keys(floor_coordinates, floor_extents)
        trunc_unique, trunc_inverse = torch.unique(
            trunc_keys, sorted=True, return_inverse=True
        )
        floor_unique, floor_inverse = torch.unique(
            floor_keys, sorted=True, return_inverse=True
        )

        checks = {
            "scaled_nonnegative": scaled_nonnegative,
            "voxel_coordinates_equal": bool(
                torch.equal(trunc_coordinates, floor_coordinates)
            ),
            "extents_equal": bool(torch.equal(trunc_extents, floor_extents)),
            "unique_voxel_count_equal": trunc_unique.numel() == floor_unique.numel(),
            "sorted_unique_keys_equal": bool(torch.equal(trunc_unique, floor_unique)),
            "point_to_voxel_mapping_equal": bool(
                torch.equal(trunc_inverse, floor_inverse)
            ),
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise AssertionError(
                f"trunc/floor mismatch for voxel_size={scalar_size}: {failed}"
            )
        per_size[str(scalar_size)] = {
            "scaled_min": scaled_min,
            "voxel_count": int(trunc_unique.numel()),
            "checks": checks,
        }

    if not shifted_nonnegative:
        raise AssertionError(f"shifted contains negative values; min={shifted_min}")
    return {
        "point_count": int(xyz.shape[0]),
        "shifted_min": shifted_min,
        "shifted_nonnegative": shifted_nonnegative,
        "voxel_sizes": per_size,
    }


def update_summary(
    summary: dict[str, Any], sample_name: str, result: dict[str, Any]
) -> None:
    summary["sample_count"] += 1
    summary["point_count"] += result["point_count"]
    summary["minimum_shifted"] = min(
        summary["minimum_shifted"], result["shifted_min"]
    )
    summary["all_shifted_nonnegative"] = (
        summary["all_shifted_nonnegative"] and result["shifted_nonnegative"]
    )
    for size, size_result in result["voxel_sizes"].items():
        size_summary = summary["voxel_sizes"][size]
        size_summary["minimum_scaled"] = min(
            size_summary["minimum_scaled"], size_result["scaled_min"]
        )
        size_summary["minimum_voxel_count"] = min(
            size_summary["minimum_voxel_count"], size_result["voxel_count"]
        )
        size_summary["maximum_voxel_count"] = max(
            size_summary["maximum_voxel_count"], size_result["voxel_count"]
        )
        for check, passed in size_result["checks"].items():
            size_summary["checks"][check] = size_summary["checks"][check] and passed
    summary["sample_names"].append(sample_name)


def new_summary() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "point_count": 0,
        "minimum_shifted": float("inf"),
        "all_shifted_nonnegative": True,
        "sample_names": [],
        "voxel_sizes": {
            str(size): {
                "minimum_scaled": float("inf"),
                "minimum_voxel_count": sys.maxsize,
                "maximum_voxel_count": 0,
                "checks": {
                    "scaled_nonnegative": True,
                    "voxel_coordinates_equal": True,
                    "extents_equal": True,
                    "unique_voxel_count_equal": True,
                    "sorted_unique_keys_equal": True,
                    "point_to_voxel_mapping_equal": True,
                },
            }
            for size in VOXEL_SIZES
        },
    }


def artificial_scalar_test(device: torch.device) -> dict[str, Any]:
    values = torch.tensor(
        [0.0, 0.1, 1.9, 3.99, -0.1, -1.9],
        dtype=torch.float32,
        device=device,
    )
    trunc_values = torch.trunc(values)
    floor_values = torch.floor(values)
    positive_equal = bool(torch.equal(trunc_values[:4], floor_values[:4]))
    negative_different = bool(torch.all(trunc_values[4:] != floor_values[4:]).item())
    if not positive_equal or not negative_different:
        raise AssertionError("Artificial scalar trunc/floor expectations failed")
    return {
        "input": values.cpu().tolist(),
        "trunc": trunc_values.cpu().tolist(),
        "floor": floor_values.cpu().tolist(),
        "nonnegative_values_equal": positive_equal,
        "negative_values_different": negative_different,
    }


def main() -> int:
    args = parse_args()
    run_dir = make_run_dir(args.run_dir)
    logger = make_logger(run_dir)
    result_path = run_dir / "trunc_floor_equivalence.json"
    payload: dict[str, Any] = {
        "status": "started",
        "project_root": str(PROJECT_ROOT),
        "device": args.device,
        "voxel_sizes": list(VOXEL_SIZES),
        "definition": "shifted = xyz - amin(xyz, dim=0)",
        "summaries": {},
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

        payload["artificial_scalars"] = artificial_scalar_test(device)
        logger.info("Artificial scalar test passed: nonnegative equal, negative different")

        split_paths = {split: load_split(split) for split in SPLITS}
        all_split_paths = [path for split in SPLITS for path in split_paths[split]]
        if len(all_split_paths) != 90 or len(set(all_split_paths)) != 90:
            raise AssertionError(
                f"Expected 90 unique split files, got {len(all_split_paths)} entries "
                f"and {len(set(all_split_paths))} unique"
            )
        actual_files = sorted(POINT_ROOT.glob("weld_*.txt"))
        if len(actual_files) != 90 or set(map(Path.resolve, actual_files)) != set(all_split_paths):
            raise AssertionError("The 54/18/18 sub splits do not cover exactly weld_1..weld_90")

        raw_all_summary = new_summary()
        normalized_summaries = {split: new_summary() for split in SPLITS}
        for split in SPLITS:
            for path in split_paths[split]:
                data = np.loadtxt(path, dtype=np.float32)
                if data.ndim != 2 or data.shape[1] != 4:
                    raise ValueError(f"Expected four columns in {path}, got {data.shape}")
                raw_xyz = data[:, :3]
                raw_result = audit_xyz(raw_xyz, device=device)
                update_summary(raw_all_summary, path.stem, raw_result)
                normalized_result = audit_xyz(normalize_xyz(raw_xyz), device=device)
                update_summary(normalized_summaries[split], path.stem, normalized_result)

        payload["summaries"]["all_90_raw_xyz"] = raw_all_summary
        for split in SPLITS:
            payload["summaries"][f"normalized_{split}"] = normalized_summaries[split]
            logger.info(
                "%s: samples=%d points=%d shifted_min=%g scaled_min=%s",
                split,
                normalized_summaries[split]["sample_count"],
                normalized_summaries[split]["point_count"],
                normalized_summaries[split]["minimum_shifted"],
                {
                    size: values["minimum_scaled"]
                    for size, values in normalized_summaries[split]["voxel_sizes"].items()
                },
            )

        fixed_summary = new_summary()
        fixed_details: dict[str, Any] = {}
        for name in FIXED_NPZ_NAMES:
            path = PREDICTIONS_ROOT / f"{name}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path) as data:
                xyz = np.asarray(data["normalized_xyz"], dtype=np.float32)
            result = audit_xyz(xyz, device=device)
            fixed_details[name] = result
            update_summary(fixed_summary, name, result)
        payload["summaries"]["fixed_6_npz"] = fixed_summary
        payload["fixed_npz_details"] = fixed_details

        all_summaries_passed = all(
            summary["all_shifted_nonnegative"]
            and all(
                all(size_result["checks"].values())
                for size_result in summary["voxel_sizes"].values()
            )
            for summary in payload["summaries"].values()
        )
        if not all_summaries_passed:
            raise AssertionError("At least one real-data trunc/floor check failed")

        payload["proof_scope"] = {
            "all_actual_weld_files": 90,
            "train_samples": len(split_paths["train"]),
            "val_samples": len(split_paths["val"]),
            "test_samples": len(split_paths["test"]),
            "fixed_npz_samples": len(FIXED_NPZ_NAMES),
            "all_actual_domain_checks_passed": True,
            "replacement_is_not_valid_for_arbitrary_negative_scaled_values": True,
        }
        payload["status"] = "TRUNC_FLOOR_EQUIVALENCE_PASSED"
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("TRUNC_FLOOR_EQUIVALENCE_PASSED")
        print(f"ARTIFACT_DIR={run_dir}")
        print("TRUNC_FLOOR_EQUIVALENCE_PASSED")
        return 0
    except Exception as exc:
        payload.update(
            {
                "status": "TRUNC_FLOOR_EQUIVALENCE_FAILED",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.exception("TRUNC_FLOOR_EQUIVALENCE_FAILED")
        print("TRUNC_FLOOR_EQUIVALENCE_FAILED")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
