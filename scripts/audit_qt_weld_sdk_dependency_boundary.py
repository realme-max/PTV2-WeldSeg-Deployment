"""Audit the Phase 10A Qt layer for SDK-only source dependencies."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QT_ROOT = PROJECT_ROOT / "deployment" / "qt_weld_app"
SOURCE_SUFFIXES = {".h", ".hpp", ".cpp", ".cc", ".cxx"}
FORBIDDEN_INCLUDES = {
    "NvInfer.h",
    "NvInferPlugin.h",
    "cuda_runtime_api.h",
    "Windows.h",
    "PointCloudLoader.h",
    "PointSampler.h",
    "FeatureBuilder.h",
    "KnnGraphBuilder.h",
    "TensorRTInference.h",
    "SegmentationPostProcessor.h",
    "WeldGeometryExtractor.h",
}
FORBIDDEN_PATH_MARKERS = {
    "strict_fp32_voxelunique_cub.plan",
    "VoxelUniqueCubPlugin.dll",
    r"E:\GRP-PTv2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_files = sorted(
        path for path in QT_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
    )
    include_pattern = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
    violations: list[dict[str, object]] = []
    includes: dict[str, list[str]] = {}

    for path in source_files:
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        file_includes = include_pattern.findall(text)
        includes[relative] = file_includes
        for include in file_includes:
            leaf = Path(include).name
            if leaf in FORBIDDEN_INCLUDES:
                violations.append({
                    "file": relative,
                    "kind": "forbidden_include",
                    "value": include,
                })
        for marker in FORBIDDEN_PATH_MARKERS:
            if marker in text:
                violations.append({
                    "file": relative,
                    "kind": "hard_coded_production_path_or_artifact",
                    "value": marker,
                })

    cmake_path = QT_ROOT / "CMakeLists.txt"
    cmake_text = cmake_path.read_text(encoding="utf-8")
    cmake_checks = {
        "application_links_ptv2_weld_sdk": "ptv2_weld_sdk" in cmake_text,
        "automoc_enabled": "CMAKE_AUTOMOC ON" in cmake_text,
        "autouic_enabled": "CMAKE_AUTOUIC ON" in cmake_text,
        "autorcc_enabled": "CMAKE_AUTORCC ON" in cmake_text,
        "cxx17_enabled": "CMAKE_CXX_STANDARD 17" in cmake_text,
    }
    passed = not violations and all(cmake_checks.values())
    report = {
        "status": "PASS" if passed else "FAILED",
        "scope": str(QT_ROOT),
        "source_file_count": len(source_files),
        "forbidden_includes": sorted(FORBIDDEN_INCLUDES),
        "violations": violations,
        "cmake_checks": cmake_checks,
        "includes_by_file": includes,
        "conclusion": (
            "Qt source consumes only Qt, Phase 10A view/controller headers, and public WeldDetector SDK headers."
            if passed else
            "The Qt SDK-only dependency boundary is violated."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        print("SDK_ONLY_DEPENDENCY_AUDIT_FAILED", file=sys.stderr)
        return 1
    print("SDK_ONLY_DEPENDENCY_AUDIT_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
