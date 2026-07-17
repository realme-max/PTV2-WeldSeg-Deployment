"""Validate the CPU reference and CUB plugin against torch.unique(sorted=True)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    results = []
    for case in comparison["cases"]:
        keys = torch.tensor(case["keys"], dtype=torch.int64, device="cpu")
        values, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        expected_values = values.tolist()
        expected_inverse = inverse.tolist()
        cpu_match = (
            expected_values == case["cpu_reference_values"]
            and expected_inverse == case["cpu_reference_inverse"]
        )
        plugin_match = (
            expected_values == case["plugin_values"]
            and expected_inverse == case["plugin_inverse"]
            and len(expected_values) == case["plugin_count"]
        )
        results.append(
            {
                "name": case["name"],
                "n": case["n"],
                "torch_unique_count": len(expected_values),
                "cpu_reference_matches_torch": cpu_match,
                "plugin_matches_torch": plugin_match,
                "passed": cpu_match and plugin_match,
            }
        )
    payload = {
        "torch_version": torch.__version__,
        "device": "cpu",
        "sorted": True,
        "return_inverse": True,
        "case_count": len(results),
        "all_passed": all(item["passed"] for item in results),
        "cases": results,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"TORCH_REFERENCE_CASES={len(results)}")
    print(f"TORCH_REFERENCE_ALL_PASSED={payload['all_passed']}")
    print("VOXEL_UNIQUE_CUB_TORCH_REFERENCE_PASSED" if payload["all_passed"] else "VOXEL_UNIQUE_CUB_TORCH_REFERENCE_FAILED")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
