"""Qualify and package the Phase 8C CUB engine as a task-equivalent production baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for item in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import gcn_res_tensorrt_phase8d_common as common  # noqa: E402


WORKER = SCRIPTS_ROOT / "phase8d_gcn_res_worker.py"
PRODUCTION_RUNNER = SCRIPTS_ROOT / "run_gcn_res_tensorrt_production.py"
SELECTOR = SCRIPTS_ROOT / "select_gcn_res_tensorrt_baseline.py"
ALLOWED_PHASE8D_CHANGES = {
    "README.md",
    "docs/context_handoff.md",
    "docs/tensorrt_phase8d_production_baseline.md",
    "docs/tensorrt_production_runbook.md",
    "deployment/tensorrt/current_baseline.json",
    "scripts/gcn_res_tensorrt_phase8d_common.py",
    "scripts/phase8d_gcn_res_worker.py",
    "scripts/qualify_gcn_res_tensorrt_production_baseline.py",
    "scripts/run_gcn_res_tensorrt_production.py",
    "scripts/select_gcn_res_tensorrt_baseline.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts/gcn_res_tensorrt")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--cold-start-runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measurements", type=int, default=100)
    parser.add_argument("--determinism-runs", type=int, default=100)
    parser.add_argument("--soak-iterations", type=int, default=5000)
    return parser.parse_args()


def git_precondition() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True,
            check=True, encoding="utf-8", errors="replace",
        ).stdout.strip()

    status = run("status", "--porcelain=v1")
    unexpected: list[str] = []
    for line in status.splitlines():
        path = line[3:].strip().replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path not in ALLOWED_PHASE8D_CHANGES:
            unexpected.append(line)
    head = run("log", "-1", "--oneline")
    diff_check = subprocess.run(
        ["git", "diff", "--check"], cwd=PROJECT_ROOT, text=True,
        capture_output=True, encoding="utf-8", errors="replace",
    )
    passed = (
        not unexpected
        and diff_check.returncode == 0
        and "feat(tensorrt): optimize VoxelUnique with CUB and validate engine" in head
    )
    return {
        "status": "PASS" if passed else "PHASE8D_WORKTREE_PRECONDITION_FAILED",
        "head": head,
        "git_status_porcelain": status,
        "unexpected_changes": unexpected,
        "git_diff_check_exit_code": diff_check.returncode,
        "git_diff_check_output": (diff_check.stdout + diff_check.stderr).strip(),
    }


def run_process(command: list[str], log_path: Path, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n$ {' '.join(command)}\n")
        stream.write(completed.stdout)
        stream.write(completed.stderr)
        stream.write(f"\nexit_code={completed.returncode} elapsed_seconds={elapsed:.6f}\n")
    return {
        "command": command,
        "exit_code": completed.returncode,
        "elapsed_seconds": elapsed,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def make_package(run_dir: Path, deployment_id: str, frozen: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    package = run_dir / "package"
    dirs = {
        "engine": package / "engine",
        "plugins": package / "plugins",
        "model": package / "model",
        "manifests": package / "manifests",
        "scripts": package / "scripts",
        "docs": package / "docs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=False)
    engine = dirs["engine"] / "strict_fp32_voxelunique_cub.plan"
    plugin = dirs["plugins"] / "VoxelUniqueCubPlugin.dll"
    onnx = dirs["model"] / "gcn_res_voxelunique_cub.onnx"
    shutil.copy2(common.CANDIDATE_ENGINE, engine)
    shutil.copy2(common.CANDIDATE_PLUGIN, plugin)
    shutil.copy2(common.CANDIDATE_ONNX, onnx)
    manifest = {
        "deployment_id": deployment_id,
        "status": "qualification_pending",
        "model": "GCN_res",
        "precision": "strict_fp32",
        "task_equivalence": True,
        "strict_numerical_equivalence": False,
        "numerical_exception": {
            "threshold": 0.0001,
            "passed_samples": 13,
            "total_samples": 18,
            "worst_sample": "weld_14",
            "worst_max_abs": 0.00012063980102539062,
            "status": "CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED",
        },
        "input_contract": {"points": [1, 2048, 4], "adj": [1, 2048, 2048], "dtype": "float32"},
        "output_contract": {"logits": [1, 2048, 2], "dtype": "float32"},
        "plugin": {
            "name": "VoxelUniqueCub", "version": "1",
            "namespace": "com.tensorrt.ptv2.experimental", "instances": 4,
        },
        "artifacts": {
            "engine": "engine/strict_fp32_voxelunique_cub.plan",
            "plugin": "plugins/VoxelUniqueCubPlugin.dll",
            "onnx": "model/gcn_res_voxelunique_cub.onnx",
        },
        "engine_sha256": common.sha256(engine),
        "plugin_sha256": common.sha256(plugin),
        "onnx_sha256": common.sha256(onnx),
        "checkpoint_sha256": frozen["checkpoint"]["sha256"],
        "tensor_rt_version": "11.1.0.106",
        "cuda_runtime": "12.8",
        "gpu": "NVIDIA GeForce RTX 5060",
        "compute_capability": "12.0",
        "compatibility": {
            "tensorrt": "11.1.0.106", "cuda_runtime": "12.8",
            "gpu": "NVIDIA GeForce RTX 5060", "compute_capability": "12.0",
            "os": "Windows x64", "python": "3.11",
        },
        "qualification": {"promotion_status": "PENDING"},
        "rollback_deployment_id": "gcn-res-trt-strict-fp32-baseline-20260716",
    }
    manifest_path = dirs["manifests"] / "deployment_manifest.json"
    common.dump_json(manifest_path, manifest)
    compatibility = manifest["compatibility"] | {
        "required_dlls": ["VoxelUniqueCubPlugin.dll", "TensorRT 11.1 runtime DLLs", "CUDA 12.8 runtime DLLs"],
        "engine_portability": "TensorRT engines are GPU/driver/TensorRT-version specific; validate on target before use.",
    }
    common.dump_json(dirs["manifests"] / "compatibility.json", compatibility)
    rollback = {
        "deployment_id": "gcn-res-trt-strict-fp32-baseline-20260716",
        "manifest_type": "rollback",
        "status": "rollback_baseline_available",
        "model": "GCN_res",
        "precision": "strict_fp32",
        "task_equivalence": True,
        "engine_path": str(common.BASELINE_ENGINE.resolve()),
        "engine_sha256": common.EXPECTED_HASHES["baseline_engine"],
        "plugin_path": str(common.BASELINE_PLUGIN.resolve()),
        "plugin_sha256": common.EXPECTED_HASHES["baseline_plugin"],
        "reason": "Retained validated baseline; slower but available for explicit manual rollback if the CUB package has compatibility issues.",
        "automatic_rollback": False,
        "switch_method": "Run select_gcn_res_tensorrt_baseline.py with an explicit validated manifest.",
        "numerical_exception_on_candidate": manifest["numerical_exception"],
    }
    common.dump_json(dirs["manifests"] / "rollback_manifest.json", rollback)
    (dirs["scripts"] / "verify_package.ps1").write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "$PackageRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path\n"
        "$ChecksumFile = Join-Path $PackageRoot 'manifests\\checksums.sha256'\n"
        "foreach ($Line in Get-Content -LiteralPath $ChecksumFile) {\n"
        "  if ($Line -notmatch '^([0-9a-f]{64})  (.+)$') { throw \"Invalid checksum line: $Line\" }\n"
        "  $Expected = $Matches[1]\n"
        "  $Relative = $Matches[2] -replace '/', '\\'\n"
        "  $Target = Join-Path $PackageRoot $Relative\n"
        "  if (-not (Test-Path -LiteralPath $Target -PathType Leaf)) { throw \"Missing package file: $Relative\" }\n"
        "  $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Target).Hash.ToLowerInvariant()\n"
        "  if ($Actual -ne $Expected) { throw \"SHA-256 mismatch: $Relative\" }\n"
        "}\n"
        "Write-Output 'TENSORRT_PRODUCTION_PACKAGE_CHECKSUMS_PASSED'\n",
        encoding="utf-8",
    )
    (dirs["scripts"] / "run_inference.ps1").write_text(
        "param([Parameter(Mandatory=$true)][string]$Points,[Parameter(Mandatory=$true)][string]$Adj,[Parameter(Mandatory=$true)][string]$Output)\n"
        f"& '{common.PYTHON}' '{PRODUCTION_RUNNER}' --manifest (Join-Path $PSScriptRoot '..\\manifests\\deployment_manifest.json') --points $Points --adj $Adj --output $Output\n"
        "exit $LASTEXITCODE\n",
        encoding="utf-8",
    )
    (dirs["scripts"] / "run_smoke.ps1").write_text(
        "param([Parameter(Mandatory=$true)][string]$Points,[Parameter(Mandatory=$true)][string]$Adj,[string]$Output=(Join-Path $env:TEMP 'gcn_res_trt_smoke.npy'))\n"
        "& (Join-Path $PSScriptRoot 'run_inference.ps1') -Points $Points -Adj $Adj -Output $Output\n"
        "exit $LASTEXITCODE\n",
        encoding="utf-8",
    )
    (dirs["docs"] / "KNOWN_LIMITATIONS.md").write_text(
        "# Known limitations\n\n- Fixed B=1, N=2048, FP32 only.\n- RTX 5060 / SM 12.0 and TensorRT 11.1.0.106 qualified.\n"
        "- Task outputs are equivalent, but strict per-sample max-absolute numerical threshold passed only 13/18; worst weld_14 = 1.206398010254e-4.\n"
        "- No automatic PyTorch fallback and no automatic rollback.\n",
        encoding="utf-8",
    )
    (dirs["docs"] / "ROLLBACK.md").write_text(
        "# Rollback\n\nRollback is manual and manifest-driven. The previous engine and plugin remain untouched. "
        "Use `select_gcn_res_tensorrt_baseline.py --manifest <validated-old-manifest>`; never replace files in place.\n",
        encoding="utf-8",
    )
    (dirs["docs"] / "README.md").write_text(
        "# GCN_res TensorRT strict-FP32 production package\n\nRun `scripts/run_inference.ps1`. The runner validates artifact hashes, runtime/GPU compatibility and fixed input contracts before loading the engine. "
        "This is a task-equivalent baseline with an explicit strict numerical exception; see `KNOWN_LIMITATIONS.md`.\n",
        encoding="utf-8",
    )
    return package, manifest_path, manifest


def extract_input_arrays(input_dir: Path, manifest: dict[str, Any], target_dir: Path) -> dict[str, dict[str, Path]]:
    target_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Path]] = {}
    for record in manifest["samples"]:
        points, adj, _ = common.load_frozen_sample(input_dir, record)
        points_path = target_dir / f"{record['sample_id']}_points.npy"
        adj_path = target_dir / f"{record['sample_id']}_adj.npy"
        np.save(points_path, points, allow_pickle=False)
        np.save(adj_path, adj, allow_pickle=False)
        outputs[record["sample_id"]] = {"points": points_path, "adj": adj_path}
    return outputs


def worker_command(mode: str, runtime: str, input_dir: Path, input_manifest: Path, output: Path, **kwargs: Any) -> list[str]:
    command = [str(common.PYTHON), str(WORKER), "--mode", mode, "--runtime", runtime,
               "--input-dir", str(input_dir), "--input-manifest", str(input_manifest), "--output", str(output)]
    if runtime == "candidate":
        command += ["--engine", str(kwargs.pop("engine")), "--plugin", str(kwargs.pop("plugin"))]
    elif runtime == "baseline":
        command += ["--engine", str(common.BASELINE_ENGINE), "--plugin", str(common.BASELINE_PLUGIN)]
    for key, value in kwargs.items():
        command += [f"--{key.replace('_', '-')}", str(value)]
    return command


def aggregate_regression(run_dir: Path, manifest: dict[str, Any], result_paths: dict[str, Path]) -> dict[str, Any]:
    results = {name: common.load_json(path) for name, path in result_paths.items()}
    by_runtime = {name: {item["sample_id"]: item for item in payload["samples"]} for name, payload in results.items()}
    per_sample: list[dict[str, Any]] = []
    all_labels: list[np.ndarray] = []
    predictions: dict[str, list[np.ndarray]] = {name: [] for name in results}
    for record in manifest["samples"]:
        sample_id = record["sample_id"]
        _, _, labels = common.load_frozen_sample(run_dir / "validation_inputs", record)
        all_labels.append(labels.reshape(-1))
        logits = {
            name: np.load(by_runtime[name][sample_id]["logits_path"], allow_pickle=False)
            for name in results
        }
        pred = {name: np.argmax(value, axis=-1).astype(np.int64) for name, value in logits.items()}
        for name in results:
            predictions[name].append(pred[name].reshape(-1))
        candidate_vs_pt = common.numerical_comparison(logits["pytorch"], logits["candidate"])
        candidate_vs_base = common.numerical_comparison(logits["baseline"], logits["candidate"])
        per_sample.append({
            "sample_id": sample_id,
            "candidate_vs_pytorch": candidate_vs_pt,
            "candidate_vs_baseline": candidate_vs_base,
            "candidate_vs_pytorch_labels_exact": bool(np.array_equal(pred["candidate"], pred["pytorch"])),
            "candidate_vs_baseline_labels_exact": bool(np.array_equal(pred["candidate"], pred["baseline"])),
            "strict_threshold_passed": bool(candidate_vs_pt["max_abs_error"] < 1e-4),
        })
    labels_all = np.concatenate(all_labels)
    aggregate_metrics = {name: common.segmentation_metrics(labels_all, np.concatenate(items)) for name, items in predictions.items()}
    worst = max(per_sample, key=lambda item: item["candidate_vs_pytorch"]["max_abs_error"])
    report = {
        "statuses": ["CANDIDATE_RUNTIME_VALIDATION_PASSED", "CANDIDATE_TASK_LEVEL_EQUIVALENCE_CONFIRMED", "CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED"],
        "total_samples": len(per_sample),
        "runtime_passed_samples": len(per_sample),
        "strict_threshold_passed_samples": sum(item["strict_threshold_passed"] for item in per_sample),
        "strict_threshold": 0.0001,
        "all_candidate_vs_pytorch_labels_exact": all(item["candidate_vs_pytorch_labels_exact"] for item in per_sample),
        "all_candidate_vs_baseline_labels_exact": all(item["candidate_vs_baseline_labels_exact"] for item in per_sample),
        "metrics": aggregate_metrics,
        "metric_deltas_candidate_vs_pytorch": {key: abs(float(aggregate_metrics["candidate"][key]) - float(aggregate_metrics["pytorch"][key])) for key in ("overall_accuracy", "miou", "weld_seam_precision", "weld_seam_recall", "weld_seam_f1")},
        "worst_sample": worst["sample_id"],
        "worst_max_abs": worst["candidate_vs_pytorch"]["max_abs_error"],
        "per_sample": per_sample,
    }
    common.dump_json(run_dir / "production_accuracy_regression.json", report)
    return report


def aggregate_performance(run_dir: Path, round_results: list[dict[str, Any]]) -> dict[str, Any]:
    runtimes = ("pytorch", "baseline", "candidate")
    aggregate: dict[str, Any] = {}
    per_sample_rows: list[dict[str, Any]] = []
    for runtime in runtimes:
        payloads = [item["payload"] for item in round_results if item["runtime"] == runtime]
        pure = [value for payload in payloads for sample in payload["per_sample"] for value in sample["pure_samples_ms"]]
        e2e = [value for payload in payloads for sample in payload["per_sample"] for value in sample["e2e_samples_ms"]]
        sample_names = [item["sample_id"] for item in payloads[0]["per_sample"]]
        sample_summary: dict[str, Any] = {}
        for name in sample_names:
            rows = [sample for payload in payloads for sample in payload["per_sample"] if sample["sample_id"] == name]
            sample_pure = [value for row in rows for value in row["pure_samples_ms"]]
            sample_e2e = [value for row in rows for value in row["e2e_samples_ms"]]
            sample_summary[name] = {"pure": common.latency_statistics(sample_pure), "e2e": common.latency_statistics(sample_e2e)}
            per_sample_rows.append({"runtime": runtime, "sample_id": name, "pure_mean_ms": sample_summary[name]["pure"]["mean"], "pure_p99_ms": sample_summary[name]["pure"]["p99"], "e2e_mean_ms": sample_summary[name]["e2e"]["mean"], "e2e_p99_ms": sample_summary[name]["e2e"]["p99"]})
        aggregate[runtime] = {
            "pure": common.latency_statistics(pure), "e2e": common.latency_statistics(e2e),
            "sample_macro_pure_mean_ms": float(np.mean([item["pure"]["mean"] for item in sample_summary.values()])),
            "sample_macro_e2e_mean_ms": float(np.mean([item["e2e"]["mean"] for item in sample_summary.values()])),
            "fastest_sample": min(sample_summary, key=lambda name: sample_summary[name]["pure"]["mean"]),
            "slowest_sample": max(sample_summary, key=lambda name: sample_summary[name]["pure"]["mean"]),
            "worst_p99_sample": max(sample_summary, key=lambda name: sample_summary[name]["pure"]["p99"]),
            "per_sample": sample_summary,
        }
    candidate_slower = [name for name, stats in aggregate["candidate"]["per_sample"].items() if stats["pure"]["mean"] >= aggregate["pytorch"]["per_sample"][name]["pure"]["mean"]]
    performance = {
        "rounds": [{key: value for key, value in item.items() if key != "payload"} for item in round_results],
        "aggregate": aggregate,
        "speedups": {
            "candidate_vs_baseline_pure": aggregate["baseline"]["pure"]["mean"] / aggregate["candidate"]["pure"]["mean"],
            "candidate_vs_pytorch_pure": aggregate["pytorch"]["pure"]["mean"] / aggregate["candidate"]["pure"]["mean"],
            "candidate_vs_pytorch_e2e_host_to_host": aggregate["pytorch"]["e2e"]["mean"] / aggregate["candidate"]["e2e"]["mean"],
        },
        "candidate_samples_slower_than_pytorch_pure": candidate_slower,
    }
    performance["promotion_thresholds_passed"] = bool(
        aggregate["candidate"]["pure"]["mean"] < aggregate["baseline"]["pure"]["mean"]
        and aggregate["candidate"]["pure"]["mean"] < aggregate["pytorch"]["pure"]["mean"]
        and aggregate["candidate"]["e2e"]["mean"] < aggregate["pytorch"]["e2e"]["mean"]
    )
    common.dump_json(run_dir / "production_performance_18_samples.json", performance)
    with (run_dir / "per_sample_latency.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_sample_rows[0]))
        writer.writeheader(); writer.writerows(per_sample_rows)
    return performance


def run_negative_tests(run_dir: Path, package: Path, manifest_path: Path, arrays: dict[str, dict[str, Path]], log: Path) -> dict[str, Any]:
    root = run_dir / "negative_cases"
    root.mkdir()
    base_manifest = common.load_json(manifest_path)
    sample = arrays["weld_65"]
    scenarios: list[tuple[str, str]] = [
        ("plugin_missing", "plugin_missing"), ("plugin_hash_mismatch", "plugin_corrupt"),
        ("engine_hash_mismatch", "engine_corrupt"), ("engine_truncated", "engine_truncated"),
        ("points_shape_wrong", "points_shape"), ("adj_shape_wrong", "adj_shape"),
        ("dtype_wrong", "dtype"), ("compatibility_mismatch", "compatibility"),
    ]
    records: list[dict[str, Any]] = []
    for name, mode in scenarios:
        case_package = root / name / "package"
        for folder in ("engine", "plugins", "model", "manifests"):
            (case_package / folder).mkdir(parents=True, exist_ok=True)
        shutil.copy2(package / base_manifest["artifacts"]["engine"], case_package / base_manifest["artifacts"]["engine"])
        shutil.copy2(package / base_manifest["artifacts"]["plugin"], case_package / base_manifest["artifacts"]["plugin"])
        shutil.copy2(package / base_manifest["artifacts"]["onnx"], case_package / base_manifest["artifacts"]["onnx"])
        manifest = json.loads(json.dumps(base_manifest))
        points, adj = sample["points"], sample["adj"]
        if mode == "plugin_missing":
            (case_package / manifest["artifacts"]["plugin"]).unlink()
        elif mode == "plugin_corrupt":
            with (case_package / manifest["artifacts"]["plugin"]).open("ab") as stream: stream.write(b"phase8d-negative")
        elif mode in {"engine_corrupt", "engine_truncated"}:
            path = case_package / manifest["artifacts"]["engine"]
            data = path.read_bytes()
            path.write_bytes((data + b"phase8d-negative") if mode == "engine_corrupt" else data[:4096])
        elif mode == "points_shape":
            value = np.load(points, allow_pickle=False)[:, :1024, :]
            points = root / name / "bad_points.npy"; np.save(points, value, allow_pickle=False)
        elif mode == "adj_shape":
            value = np.load(adj, allow_pickle=False)[:, :1024, :1024]
            adj = root / name / "bad_adj.npy"; np.save(adj, value, allow_pickle=False)
        elif mode == "dtype":
            value = np.load(points, allow_pickle=False).astype(np.float64)
            points = root / name / "bad_dtype_points.npy"; np.save(points, value, allow_pickle=False)
        elif mode == "compatibility":
            manifest["compatibility"]["compute_capability"] = "99.9"
        case_manifest = case_package / "manifests/deployment_manifest.json"
        common.dump_json(case_manifest, manifest)
        output = root / name / "should_not_exist.npy"
        command = [str(common.PYTHON), str(PRODUCTION_RUNNER), "--qualification-mode", "--manifest", str(case_manifest), "--points", str(points), "--adj", str(adj), "--output", str(output)]
        executed = run_process(command, log, timeout=180)
        summary_path = output.with_suffix(output.suffix + ".json")
        summary = common.load_json(summary_path) if summary_path.is_file() else {}
        passed = executed["exit_code"] != 0 and not output.exists() and summary.get("inference_executed") is False and bool(summary.get("error"))
        records.append({"scenario": name, "passed": passed, "exit_code": executed["exit_code"], "error": summary.get("error"), "inference_executed": summary.get("inference_executed"), "output_created": output.exists()})
    report = {"status": "PASS" if all(item["passed"] for item in records) else "FAILED", "total": len(records), "passed": sum(item["passed"] for item in records), "cases": records}
    common.dump_json(run_dir / "negative_test_report.json", report)
    return report


def write_soak_csv(run_dir: Path, soak: dict[str, Any]) -> None:
    memory_rows = []
    latency_rows = []
    for item in soak["snapshots"]:
        memory_rows.append({"iteration": item["iteration"], "sample_id": item["sample_id"], **item["memory"], **{f"gpu_{key}": value for key, value in item["telemetry"].items() if key != "raw"}})
        latency_rows.append({"iteration": item["iteration"], "sample_id": item["sample_id"], "rolling_latency_mean_ms": item["rolling_latency_mean_ms"]})
    for path, rows in ((run_dir / "soak_memory.csv", memory_rows), (run_dir / "soak_latency.csv", latency_rows)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def finalize_package(run_dir: Path, package: Path, manifest_path: Path, summary: dict[str, Any]) -> None:
    manifest = common.load_json(manifest_path)
    manifest["status"] = "production_baseline"
    manifest["numerical_exception"].update({
        "passed_samples": summary["accuracy"]["strict_threshold_passed_samples"],
        "worst_sample": summary["accuracy"]["worst_sample"],
        "worst_max_abs": summary["accuracy"]["worst_max_abs"],
    })
    manifest["qualification"] = {
        "status": "PASSED_WITH_NUMERICAL_EXCEPTION",
        "qualification_status": "TENSORRT_CUB_PRODUCTION_QUALIFICATION_PASSED_WITH_NUMERICAL_EXCEPTION",
        "promotion_status": "TENSORRT_CUB_STRICT_FP32_TASK_EQUIVALENT_BASELINE_PROMOTED",
        "strict_numerical_status": "CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED",
        "qualified_at": common.now_iso(),
        "summary_path": "../../production_qualification_summary.json",
    }
    common.dump_json(manifest_path, manifest)
    qualification = {
        "deployment_id": manifest["deployment_id"],
        "status": "PASSED_WITH_NUMERICAL_EXCEPTION",
        "hard_conditions": summary["conditions"],
        "numerical_exception": manifest["numerical_exception"],
    }
    common.dump_json(package / "manifests/qualification.json", qualification)
    shutil.copy2(manifest_path, run_dir / "deployment_manifest.json")
    for name in ("compatibility.json", "qualification.json", "rollback_manifest.json"):
        shutil.copy2(package / "manifests" / name, run_dir / name)
    inventory: list[dict[str, Any]] = []
    for path in sorted(item for item in package.rglob("*") if item.is_file()):
        if path.name == "checksums.sha256":
            continue
        inventory.append({"path": path.relative_to(package).as_posix(), "size_bytes": path.stat().st_size, "sha256": common.sha256(path)})
    common.dump_json(run_dir / "package_inventory.json", {"files": inventory, "total_files": len(inventory), "total_bytes": sum(item["size_bytes"] for item in inventory)})
    lines = [f"{item['sha256']}  {item['path']}" for item in inventory]
    (package / "manifests/checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(package / "manifests/checksums.sha256", run_dir / "checksums.sha256")


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_phase8d_production_baseline"
    run_dir = args.output_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logs = {name: run_dir / name for name in ("cold_start.log", "performance.log", "determinism.log", "soak.log", "accuracy.log", "negative_tests.log", "promotion.log")}
    initial = {"status": "PRODUCTION_BASELINE_PROMOTION_BLOCKED", "run_dir": str(run_dir), "started_at": common.now_iso()}
    common.dump_json(run_dir / "production_qualification_summary.json", initial)
    try:
        precondition = git_precondition()
        common.dump_json(run_dir / "worktree_precondition.json", precondition)
        if precondition["status"] != "PASS":
            raise RuntimeError("PHASE8D_WORKTREE_PRECONDITION_FAILED")
        frozen_objects = common.frozen_objects()
        common.dump_json(run_dir / "frozen_objects.json", frozen_objects)
        common.dump_json(run_dir / "environment.json", common.environment_snapshot())
        input_dir = run_dir / "validation_inputs"
        input_manifest = common.prepare_frozen_inputs(input_dir)
        input_manifest_path = input_dir / "input_manifest.json"
        deployment_id = f"gcn-res-trt-cub-strict-fp32-{run_id.split('_phase8d')[0]}"
        package, package_manifest, manifest = make_package(run_dir, deployment_id, frozen_objects)
        arrays = extract_input_arrays(input_dir, input_manifest, run_dir / "input_arrays")

        # Three fresh processes establish the authoritative 18-sample regression references.
        regression_paths: dict[str, Path] = {}
        for runtime in ("pytorch", "baseline", "candidate"):
            output = run_dir / f"regression_{runtime}.json"
            regression_paths[runtime] = output
            kwargs: dict[str, Any] = {"output_dir": run_dir / f"regression_{runtime}_outputs"}
            if runtime == "candidate": kwargs.update(engine=package / manifest["artifacts"]["engine"], plugin=package / manifest["artifacts"]["plugin"])
            command = worker_command("regression", runtime, input_dir, input_manifest_path, output, **kwargs)
            result = run_process(command, logs["accuracy.log"], timeout=900)
            if result["exit_code"] != 0:
                raise RuntimeError(f"{runtime} regression failed")
        accuracy = aggregate_regression(run_dir, input_manifest, regression_paths)

        # Package-path smoke on four fixed samples through the public fail-closed entry point.
        smoke_records = []
        candidate_reference_dir = run_dir / "regression_candidate_outputs"
        for sample_id in ("weld_65", "weld_5", "weld_12", "weld_14"):
            output = run_dir / "package_smoke" / f"{sample_id}_logits.npy"
            command = [str(common.PYTHON), str(PRODUCTION_RUNNER), "--qualification-mode", "--manifest", str(package_manifest), "--points", str(arrays[sample_id]["points"]), "--adj", str(arrays[sample_id]["adj"]), "--output", str(output)]
            executed = run_process(command, logs["accuracy.log"], timeout=300)
            summary = common.load_json(output.with_suffix(output.suffix + ".json"))
            expected = np.load(candidate_reference_dir / f"{sample_id}_prediction.npy", allow_pickle=False)
            actual = np.argmax(np.load(output, allow_pickle=False), axis=-1) if output.is_file() else np.empty(0)
            smoke_records.append({"sample_id": sample_id, "exit_code": executed["exit_code"], "summary": summary, "labels_match_phase8c_candidate_path": bool(np.array_equal(actual, expected))})
        smoke_passed = all(item["exit_code"] == 0 and item["summary"].get("runtime_status") == "PASS" and item["summary"].get("runtime_plugin_instances") == 4 and item["labels_match_phase8c_candidate_path"] for item in smoke_records)
        common.dump_json(run_dir / "production_package_smoke.json", {"status": "PASS" if smoke_passed else "FAILED", "samples": smoke_records})

        # Ten true cold starts: each subprocess imports and registers only the CUB plugin.
        cold_records = []
        for iteration in range(args.cold_start_runs):
            output = run_dir / "cold_start_runs" / f"run_{iteration + 1:02d}.json"
            command = worker_command("cold-start", "candidate", input_dir, input_manifest_path, output,
                                     engine=package / manifest["artifacts"]["engine"], plugin=package / manifest["artifacts"]["plugin"], sample_id="weld_65")
            executed = run_process(command, logs["cold_start.log"], timeout=300)
            payload = common.load_json(output)
            payload["exit_code"] = executed["exit_code"]
            cold_records.append(payload)
        cold = {
            "status": "PASS" if len(cold_records) == args.cold_start_runs and all(
                item.get("status") == "PASS"
                and item["exit_code"] == 0
                and item.get("output_finite") is True
                and item.get("error_recorder_errors") == 0
                and item.get("runtime_plugin_instances") == 4
                for item in cold_records
            ) else "FAILED",
            "successful_runs": sum(item.get("status") == "PASS" and item["exit_code"] == 0 for item in cold_records),
            "total_runs": args.cold_start_runs,
            "unique_logits_hashes": len({item.get("output_sha256") for item in cold_records}),
            "labels_hash_stable": len({item.get("predicted_labels_sha256") for item in cold_records}) == 1,
            "note": "Cold-start acceptance is process/runtime success. Bitwise logits and label determinism are classified by the separate 100-run determinism test.",
            "runs": cold_records,
        }
        common.dump_json(run_dir / "cold_start_results.json", cold)

        # Three independent rounds, rotating backend order.
        names = [item["sample_id"] for item in input_manifest["samples"]]
        random.Random(common.SEED).shuffle(names)
        round_orders = [
            ("pytorch", "baseline", "candidate"),
            ("candidate", "pytorch", "baseline"),
            ("baseline", "candidate", "pytorch"),
        ]
        round_results: list[dict[str, Any]] = []
        for round_index, backend_order in enumerate(round_orders, start=1):
            for runtime in backend_order:
                output = run_dir / "performance_rounds" / f"round_{round_index}_{runtime}.json"
                kwargs = {"sample_order": ",".join(names), "warmup": args.warmup, "iterations": args.measurements}
                if runtime == "candidate": kwargs.update(engine=package / manifest["artifacts"]["engine"], plugin=package / manifest["artifacts"]["plugin"])
                command = worker_command("benchmark", runtime, input_dir, input_manifest_path, output, **kwargs)
                executed = run_process(command, logs["performance.log"], timeout=1800)
                if executed["exit_code"] != 0:
                    raise RuntimeError(f"Performance round {round_index} {runtime} failed")
                round_results.append({"round": round_index, "runtime": runtime, "backend_order": list(backend_order), "result_path": str(output), "payload": common.load_json(output)})
        performance = aggregate_performance(run_dir, round_results)

        determinism_path = run_dir / "determinism_report.json"
        command = worker_command("determinism", "candidate", input_dir, input_manifest_path, determinism_path,
                                 engine=package / manifest["artifacts"]["engine"], plugin=package / manifest["artifacts"]["plugin"], samples="weld_65,weld_5,weld_14", iterations=args.determinism_runs)
        deterministic_process = run_process(command, logs["determinism.log"], timeout=900)
        determinism = common.load_json(determinism_path)
        if deterministic_process["exit_code"] != 0:
            raise RuntimeError("Determinism worker failed")

        soak_path = run_dir / "soak_test.json"
        command = worker_command("soak", "candidate", input_dir, input_manifest_path, soak_path,
                                 engine=package / manifest["artifacts"]["engine"], plugin=package / manifest["artifacts"]["plugin"],
                                 soak_iterations=args.soak_iterations, reference_dir=candidate_reference_dir)
        soak_process = run_process(command, logs["soak.log"], timeout=1800)
        soak = common.load_json(soak_path)
        if soak_process["exit_code"] != 0:
            raise RuntimeError("Soak worker failed")
        write_soak_csv(run_dir, soak)

        negative = run_negative_tests(run_dir, package, package_manifest, arrays, logs["negative_tests.log"])

        hashes_after = common.frozen_objects()
        hashes_unchanged = frozen_objects == hashes_after
        conditions = {
            "candidate_hashes_correct": hashes_unchanged,
            "cold_start_10_of_10": cold["status"] == "PASS" and cold["successful_runs"] == 10,
            "runtime_18_of_18": accuracy["runtime_passed_samples"] == 18,
            "task_metrics_exact": accuracy["all_candidate_vs_pytorch_labels_exact"] and accuracy["all_candidate_vs_baseline_labels_exact"] and all(value == 0.0 for value in accuracy["metric_deltas_candidate_vs_pytorch"].values()),
            "package_smoke": smoke_passed,
            "candidate_faster_than_baseline_and_pytorch": performance["promotion_thresholds_passed"],
            "deterministic_labels": determinism["status"] in {"DETERMINISTIC_LOGITS_CONFIRMED", "DETERMINISTIC_LABELS_ONLY"},
            "soak_5000": soak.get("status") == "PASS" and soak.get("successful_enqueues") == 5000 and not soak.get("monotonic_memory_growth_detected") and not soak.get("obvious_latency_degradation_detected"),
            "negative_tests": negative["status"] == "PASS" and negative["passed"] == 8,
            "rollback_manifest_valid": (
                common.load_json(package / "manifests/rollback_manifest.json")["engine_sha256"] == common.EXPECTED_HASHES["baseline_engine"]
                and common.load_json(package / "manifests/rollback_manifest.json")["plugin_sha256"] == common.EXPECTED_HASHES["baseline_plugin"]
            ),
            "numerical_exception_explicit": accuracy["strict_threshold_passed_samples"] < 18 and accuracy["strict_threshold_passed_samples"] == 13,
        }
        promoted = all(conditions.values())
        summary = {
            "status": "TENSORRT_CUB_PRODUCTION_QUALIFICATION_PASSED_WITH_NUMERICAL_EXCEPTION" if promoted else "PRODUCTION_BASELINE_PROMOTION_BLOCKED",
            "promotion_status": "TENSORRT_CUB_STRICT_FP32_TASK_EQUIVALENT_BASELINE_PROMOTED" if promoted else "PRODUCTION_BASELINE_PROMOTION_BLOCKED",
            "strict_numerical_status": "CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED",
            "run_dir": str(run_dir), "deployment_id": deployment_id,
            "completed_at": common.now_iso(), "conditions": conditions,
            "frozen_objects": frozen_objects, "cold_start": cold,
            "performance": performance, "determinism": determinism,
            "soak": soak, "accuracy": accuracy, "negative_tests": negative,
        }
        common.dump_json(run_dir / "production_qualification_summary.json", summary)
        if not promoted:
            (run_dir / "phase8d_summary.md").write_text("# Phase 8D\n\nPRODUCTION_BASELINE_PROMOTION_BLOCKED\n\nSee `production_qualification_summary.json`.\n", encoding="utf-8")
            print("PRODUCTION_BASELINE_PROMOTION_BLOCKED")
            return 2
        finalize_package(run_dir, package, package_manifest, summary)
        select = run_process([str(common.PYTHON), str(SELECTOR), "--manifest", str(package_manifest)], logs["promotion.log"], timeout=120)
        if select["exit_code"] != 0:
            raise RuntimeError("Production baseline pointer selection failed")
        phase_summary = (
            "# TensorRT Phase 8D production baseline\n\n"
            "TENSORRT_CUB_PRODUCTION_QUALIFICATION_PASSED_WITH_NUMERICAL_EXCEPTION\n\n"
            "TENSORRT_CUB_STRICT_FP32_TASK_EQUIVALENT_BASELINE_PROMOTED\n\n"
            "CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED\n\n"
            f"Deployment: `{deployment_id}`\n\nEngine SHA-256: `{manifest['engine_sha256']}`\n\n"
            f"Plugin SHA-256: `{manifest['plugin_sha256']}`\n"
        )
        (run_dir / "phase8d_summary.md").write_text(phase_summary, encoding="utf-8")
        print("TENSORRT_CUB_PRODUCTION_QUALIFICATION_PASSED_WITH_NUMERICAL_EXCEPTION")
        print("TENSORRT_CUB_STRICT_FP32_TASK_EQUIVALENT_BASELINE_PROMOTED")
        print("CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED")
        return 0
    except Exception as exc:
        initial.update({"exception_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "failed_at": common.now_iso()})
        common.dump_json(run_dir / "production_qualification_summary.json", initial)
        with logs["promotion.log"].open("a", encoding="utf-8") as stream:
            stream.write(traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
        print("PRODUCTION_BASELINE_PROMOTION_BLOCKED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
