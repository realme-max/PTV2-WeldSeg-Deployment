"""Finalize Phase 2C.1 correctness evidence without running benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import onnx
import tensorrt as trt
import torch


EXPECTED_ORIGINAL_ONNX_SHA256 = (
    "20aa7ba21a52c6497e0ce10676edae599def203bbddd4ca063b7abccdeeb5198"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return (completed.stdout + completed.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--original-onnx", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    original_onnx = args.original_onnx.resolve()
    comparison_path = run_dir / "comparison.json"
    torch_path = run_dir / "torch_reference_validation.json"
    diagnostic_onnx = run_dir / "voxel_unique_correctness_dynamic.onnx"
    plan_path = run_dir / "voxel_unique_correctness.plan"

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    torch_validation = json.loads(torch_path.read_text(encoding="utf-8"))
    model = onnx.load_model(str(diagnostic_onnx), load_external_data=False)
    custom_nodes = [
        {"name": node.name, "op_type": node.op_type, "domain": node.domain}
        for node in model.graph.node
    ]
    diagnostic_inputs = [value.name for value in model.graph.input]
    diagnostic_outputs = [value.name for value in model.graph.output]
    source_dir = run_dir / "source"
    source_hashes = {
        str(path.relative_to(source_dir)).replace("\\", "/"): sha256(path)
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    original_hash = sha256(original_onnx)
    required_cases = {
        "all_same",
        "all_unique",
        "sorted",
        "reversed",
        "repeated_groups",
    }
    present_cases = {case["name"] for case in comparison["cases"]}
    random_sizes = sorted(
        {
            case["n"]
            for case in comparison["cases"]
            if case["name"].startswith("random_n")
        }
    )

    environment = {
        "phase": "TensorRT Phase 2C.1 VoxelUnique correctness",
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "tensorrt_version": trt.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available()
        else None,
        "cudnn_version": torch.backends.cudnn.version(),
        "onnx_version": onnx.__version__,
        "nvcc_path": str(args.nvcc.resolve()),
        "nvcc_version_output": command_output([str(args.nvcc.resolve()), "--version"]),
        "diagnostic_onnx": str(diagnostic_onnx),
        "diagnostic_onnx_sha256": sha256(diagnostic_onnx),
        "diagnostic_onnx_nodes": custom_nodes,
        "diagnostic_onnx_inputs": diagnostic_inputs,
        "diagnostic_onnx_outputs": diagnostic_outputs,
        "correctness_engine": str(plan_path),
        "correctness_engine_sha256": sha256(plan_path),
        "correctness_engine_scope": "isolated VoxelUnique test network only",
        "original_gcn_res_onnx": str(original_onnx),
        "original_gcn_res_onnx_sha256": original_hash,
        "original_gcn_res_onnx_expected_sha256": EXPECTED_ORIGINAL_ONNX_SHA256,
        "original_gcn_res_onnx_unchanged": original_hash
        == EXPECTED_ORIGINAL_ONNX_SHA256,
        "source_sha256": source_hashes,
    }
    (run_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    all_passed = (
        comparison["all_passed"]
        and torch_validation["all_passed"]
        and required_cases.issubset(present_cases)
        and random_sizes == [4, 8, 32, 2048]
        and environment["original_gcn_res_onnx_unchanged"]
        and custom_nodes
        == [
            {
                "name": "VoxelUniqueCorrectnessNode",
                "op_type": "VoxelUnique",
                "domain": "com.tensorrt.ptv2",
            }
        ]
        and diagnostic_inputs == ["voxel_key"]
        and diagnostic_outputs
        == ["voxel_count", "unique_values", "inverse_indices"]
    )

    rows = []
    for case in comparison["cases"]:
        rows.append(
            f"| {case['name']} | {case['n']} | {case['plugin_count']} | "
            f"{'PASS' if case['passed'] else 'FAIL'} |"
        )
    report = f"""# TensorRT VoxelUniquePlugin correctness

## Scope

This Phase 2C.1 experiment validates a real, intentionally unoptimized
`VoxelUnique` IPluginV3 implementation in an isolated TensorRT network. It does
not load or modify the GCN_res ONNX, checkpoint, deployment model, or training
code. It does not run FP16 or a benchmark.

## Contract

- Input: `voxel_key`, INT64 `[N]`
- Outputs: `voxel_count`, INT32 scalar; `unique_values`, INT64 `[M]`;
  `inverse_indices`, INT64 `[N]`
- Semantics: `torch.unique(sorted=True, return_inverse=True)`
- Dynamic bound: `1 <= M <= N`, declared through an IPluginV3 size tensor

## Reference and implementation

The C++ CPU reference sorts a copy of the keys, removes adjacent duplicates,
then uses binary search to produce inverse indices. The CUDA correctness kernel
uses one thread, performs exact INT64 de-duplication, insertion-sorts the unique
values, and maps each key to its sorted unique index. This serial algorithm is
deliberately not a performance implementation.

Each Plugin result was compared first with the C++ CPU reference and then with
an independent CPU PyTorch `torch.unique` result.

## Results

- Cases: {comparison['case_count']}
- Random sizes: {random_sizes}
- C++ reference vs Plugin: {'PASS' if comparison['all_passed'] else 'FAIL'}
- PyTorch reference vs Plugin: {'PASS' if torch_validation['all_passed'] else 'FAIL'}
- Dynamic `M` shape equals `voxel_count`: {'PASS' if all(case['shape_match'] for case in comparison['cases']) else 'FAIL'}
- Original GCN_res ONNX SHA-256 unchanged: {'PASS' if environment['original_gcn_res_onnx_unchanged'] else 'FAIL'}

| Case | N | M | Result |
|---|---:|---:|---|
{chr(10).join(rows)}

## Environment

- TensorRT: {trt.__version__}
- PyTorch: {torch.__version__}
- CUDA runtime reported by PyTorch: {torch.version.cuda}
- GPU: {environment['gpu_name']}
- Compute capability: {environment['compute_capability']}
- Correctness ONNX: `{diagnostic_onnx}`
- Correctness Engine: `{plan_path}`

## Evidence

- `comparison.json`: complete inputs and C++/Plugin outputs for every case
- `torch_reference_validation.json`: independent PyTorch comparison
- `correctness_run.log`: parser, builder, runtime, and per-case status
- `environment.json`: version, path, source hash, and original-ONNX hash evidence
- `source/`: exact source snapshot used by this run

## Conclusion

{'VOXEL_UNIQUE_PLUGIN_CORRECTNESS_PASSED' if all_passed else 'VOXEL_UNIQUE_PLUGIN_CORRECTNESS_FAILED'}
"""
    (run_dir / "test_report.md").write_text(report, encoding="utf-8")
    print(f"FINAL_ARTIFACT_VALIDATION={all_passed}")
    print("VOXEL_UNIQUE_PLUGIN_CORRECTNESS_PASSED" if all_passed else "VOXEL_UNIQUE_PLUGIN_CORRECTNESS_FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
