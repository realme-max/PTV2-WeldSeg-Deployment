"""Shared, fail-closed runtime helpers for TensorRT Phase 8D qualification."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import build_gcn_res_tensorrt_fp32 as phase4  # noqa: E402
import compare_tensorrt_pytorch_logits as parity  # noqa: E402
import evaluate_gcn_res_checkpoint as evaluation  # noqa: E402
import gcn_res_tensorrt_cub_common as cub_common  # noqa: E402
import run_gcn_res_tensorrt_fp32_inference as phase5  # noqa: E402


PYTHON = PROJECT_ROOT / ".venv_ptv2" / "Scripts" / "python.exe"
PHASE8C_DIR = PROJECT_ROOT / "artifacts/gcn_res_tensorrt/20260717_155708_684630_phase8c_candidate_engine"
CANDIDATE_ENGINE = PHASE8C_DIR / "strict_fp32_voxelunique_cub_candidate.plan"
CANDIDATE_PLUGIN = PHASE8C_DIR / "VoxelUniqueCubPlugin.dll"
CANDIDATE_ONNX = PHASE8C_DIR / "gcn_res_voxelunique_cub_candidate.onnx"
BASELINE_ENGINE = cub_common.BASELINE_ENGINE
BASELINE_PLUGIN = cub_common.BASELINE_PLUGIN
CHECKPOINT = cub_common.CHECKPOINT
SPLIT_ROOT = PROJECT_ROOT / "data/weld/train_test_split"

EXPECTED_HASHES = {
    "candidate_engine": "a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299",
    "candidate_plugin": "6641ec147e8eac10206a5c60ba1c1390c398d4b59e32a6a618a37046360ec348",
    "candidate_onnx": "16ca5c16c330e6572b1730e80da724231a28b68872a3203c21240348d4d89299",
    "baseline_engine": "b76563580089bbc0684e7a8e1edda6028c2d89fd91397c22b4efbd89ecd7fc2c",
    "baseline_plugin": "60c48c400cc926f6aa183de62298a203916b1add6393a3483a332624da39d0ab",
    "checkpoint": "311bddf3607d76e6b7ded450b8419bf6ae98f34f50578608b3e6a1c1c3e58d21",
}
EXPECTED_IO = cub_common.EXPECTED_IO
NUM_POINTS = 2048
SEED = 42


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return actual


def frozen_objects() -> dict[str, dict[str, Any]]:
    paths = {
        "candidate_engine": CANDIDATE_ENGINE,
        "candidate_plugin": CANDIDATE_PLUGIN,
        "candidate_onnx": CANDIDATE_ONNX,
        "baseline_engine": BASELINE_ENGINE,
        "baseline_plugin": BASELINE_PLUGIN,
        "checkpoint": CHECKPOINT,
    }
    return {
        name: {
            "path": str(path.resolve()),
            "sha256": assert_hash(path, EXPECTED_HASHES[name], name),
            "size_bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }


def latency_statistics(values_ms: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(list(values_ms), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Latency samples must be finite and non-empty")
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return {
        "unit": "ms",
        "count": int(values.size),
        "mean": mean,
        "median": float(np.median(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "std": std,
        "min": float(values.min()),
        "max": float(values.max()),
        "coefficient_of_variation": float(std / mean) if mean else 0.0,
    }


def command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    return (completed.stdout + completed.stderr).strip()


def environment_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "timestamp": now_iso(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "nvidia_smi": command_output([
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap,memory.total",
            "--format=csv,noheader",
        ]),
        "pip_check": command_output([sys.executable, "-m", "pip", "check"]),
    }
    try:
        import torch

        result.update({
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        })
    except Exception as exc:
        result["torch_environment_error"] = f"{type(exc).__name__}: {exc}"
    dll_handles = []
    try:
        # The SDK ZIP is intentionally not placed on the permanent system PATH.
        # Reproduce the production process' temporary DLL search policy here.
        dll_handles = cub_common.configure_dll_search(CANDIDATE_PLUGIN)
        import tensorrt as trt

        result["tensorrt"] = trt.__version__
        result["tensorrt_dll_search"] = "temporary process-local SDK/CUDA/plugin directories"
    except Exception as exc:
        result["tensorrt_environment_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for handle in dll_handles:
            handle.close()
    return result


def gpu_telemetry() -> dict[str, Any]:
    query = command_output([
        "nvidia-smi",
        "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ])
    parts = [item.strip() for item in query.split(",")]
    result: dict[str, Any] = {"raw": query}
    if len(parts) >= 4:
        result.update({
            "temperature_c": int(parts[0]),
            "utilization_percent": int(parts[1]),
            "memory_used_mib": int(parts[2]),
            "memory_free_mib": int(parts[3]),
        })
    return result


def split_names() -> list[str]:
    path = SPLIT_ROOT / "sub_shuffled_test_file_list.json"
    entries = load_json(path)
    names = [Path(str(item).replace("\\", "/")).stem for item in entries]
    if len(names) != 18 or len(set(names)) != 18:
        raise RuntimeError(f"Expected 18 unique test samples, got {names}")
    return names


def prepare_frozen_inputs(output_dir: Path) -> dict[str, Any]:
    """Create one deterministic, hash-addressed input cache for all Phase 8D processes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "input_manifest.json"
    expected_names = split_names()
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        if [item["sample_id"] for item in manifest.get("samples", [])] == expected_names:
            valid = True
            for item in manifest["samples"]:
                path = output_dir / item["file"]
                valid = valid and path.is_file() and sha256(path) == item["file_sha256"]
            if valid:
                return manifest

    import torch

    dataset = evaluation.FixedWeldEvaluationDataset("test")
    actual_names = [record.sample_name for record in dataset.records]
    if actual_names != expected_names:
        raise RuntimeError(f"Dataset order mismatch: {actual_names} != {expected_names}")
    samples: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        xyz = sample["normalized_xyz"].unsqueeze(0).to(torch.float32)
        points = np.ascontiguousarray(evaluation.make_model_input(xyz).numpy(), dtype=np.float32)
        adjacency_tensor, adjacency_ms = evaluation.build_adjacency_cpu(xyz)
        adjacency = np.ascontiguousarray(adjacency_tensor.numpy(), dtype=np.float32)
        labels = np.ascontiguousarray(sample["labels"].unsqueeze(0).numpy(), dtype=np.int64)
        indices = np.ascontiguousarray(sample["sample_indices"].numpy(), dtype=np.int64)
        if points.shape != (1, 2048, 4) or adjacency.shape != (1, 2048, 2048):
            raise RuntimeError(f"{sample['sample_name']}: invalid frozen input contract")
        if not np.isfinite(points).all() or not np.isfinite(adjacency).all():
            raise FloatingPointError(f"{sample['sample_name']}: non-finite frozen input")
        filename = f"{index:02d}_{sample['sample_name']}.npz"
        path = output_dir / filename
        np.savez_compressed(
            path,
            points=points,
            adj=adjacency,
            labels=labels,
            sample_indices=indices,
        )
        samples.append({
            "index": index,
            "sample_id": str(sample["sample_name"]),
            "logical_path": str(sample["logical_path"]),
            "file": filename,
            "file_sha256": sha256(path),
            "points_sha256": array_sha256(points),
            "adj_sha256": array_sha256(adjacency),
            "labels_sha256": array_sha256(labels),
            "sample_indices_sha256": array_sha256(indices),
            "adjacency_cpu_ms": float(adjacency_ms),
        })
    manifest = {
        "created_at": now_iso(),
        "seed": SEED,
        "num_points": NUM_POINTS,
        "k_neighbors": 6,
        "split": "test",
        "total_samples": len(samples),
        "split_json": str((SPLIT_ROOT / "sub_shuffled_test_file_list.json").resolve()),
        "split_json_sha256": sha256(SPLIT_ROOT / "sub_shuffled_test_file_list.json"),
        "samples": samples,
    }
    dump_json(manifest_path, manifest)
    return manifest


def load_frozen_sample(input_dir: Path, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = input_dir / record["file"]
    if sha256(path) != record["file_sha256"]:
        raise RuntimeError(f"Frozen input file hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        points = np.ascontiguousarray(payload["points"], dtype=np.float32)
        adj = np.ascontiguousarray(payload["adj"], dtype=np.float32)
        labels = np.ascontiguousarray(payload["labels"], dtype=np.int64)
    if points.shape != (1, 2048, 4) or adj.shape != (1, 2048, 2048) or labels.shape != (1, 2048):
        raise RuntimeError(f"Frozen input shape mismatch: {path}")
    if array_sha256(points) != record["points_sha256"] or array_sha256(adj) != record["adj_sha256"]:
        raise RuntimeError(f"Frozen input array hash mismatch: {path}")
    return points, adj, labels


class TensorRTRunner:
    """One-engine runner. A process must instantiate either baseline or candidate, never both."""

    def __init__(self, engine_path: Path, plugin_path: Path, kind: str) -> None:
        if kind not in {"candidate", "baseline"}:
            raise ValueError(kind)
        self.kind = kind
        self.engine_path = engine_path.resolve()
        self.plugin_path = plugin_path.resolve()
        self.dll_handles = cub_common.configure_dll_search(self.plugin_path)
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self.trt, self.cudart = trt, cudart
        phase5.cuda_call(cudart, "cudaSetDevice(0)", cudart.cudaSetDevice(0))
        self.logger = trt.Logger(trt.Logger.WARNING)
        if not trt.init_libnvinfer_plugins(self.logger, ""):
            raise RuntimeError("TensorRT standard plugin initialization failed")
        plugin_started = time.perf_counter()
        if kind == "candidate":
            self.library, info = cub_common.load_cub_plugin(self.plugin_path)
            counter_name = "getVoxelUniqueCubRuntimeCreationCount"
            if not info["registered"]:
                raise RuntimeError("VoxelUniqueCub creator registration failed")
        else:
            self.library, info = phase4.load_plugin_library(self.plugin_path)
            counter_name = "getVoxelUniqueRuntimeCreationCount"
            if not info["registration_function_returned"]:
                raise RuntimeError("VoxelUnique creator registration failed")
        self.plugin_load_ms = (time.perf_counter() - plugin_started) * 1000.0
        self.plugin_info = info
        ErrorRecorder = phase5.make_error_recorder_class(trt)
        self.recorder = ErrorRecorder()
        self.runtime = trt.Runtime(self.logger)
        self.runtime.error_recorder = self.recorder
        before = int(getattr(self.library, counter_name)())
        started = time.perf_counter()
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        self.deserialize_ms = (time.perf_counter() - started) * 1000.0
        if self.engine is None:
            raise RuntimeError("deserialize_cuda_engine returned None")
        self.engine.error_recorder = self.recorder
        after = int(getattr(self.library, counter_name)())
        self.runtime_plugin_instances = after - before
        if self.runtime_plugin_instances != 4:
            raise RuntimeError(f"Expected four runtime plugin instances, got {self.runtime_plugin_instances}")
        started = time.perf_counter()
        self.context = self.engine.create_execution_context()
        self.context_creation_ms = (time.perf_counter() - started) * 1000.0
        if self.context is None:
            raise RuntimeError("create_execution_context returned None")
        self.context.error_recorder = self.recorder
        self.io = cub_common.engine_io(trt, self.engine)
        self.host_logits = np.empty((1, 2048, 2), dtype=np.float32)
        self.stream = phase5.cuda_call(cudart, "cudaStreamCreate", cudart.cudaStreamCreate())[0]
        sizes = {"points": 1 * 2048 * 4 * 4, "adj": 1 * 2048 * 2048 * 4, "logits": self.host_logits.nbytes}
        self.pointers: dict[str, int] = {}
        for name, size in sizes.items():
            self.pointers[name] = int(phase5.cuda_call(cudart, f"cudaMalloc({name})", cudart.cudaMalloc(size))[0])
            if not self.context.set_tensor_address(name, self.pointers[name]):
                raise RuntimeError(f"set_tensor_address({name}) failed")

    def copy_inputs(self, points: np.ndarray, adj: np.ndarray) -> None:
        for name, array in (("points", points), ("adj", adj)):
            phase5.cuda_call(self.cudart, f"H2D {name}", self.cudart.cudaMemcpyAsync(
                self.pointers[name], int(array.ctypes.data), int(array.nbytes),
                self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream,
            ))

    def enqueue(self) -> None:
        if not self.context.execute_async_v3(stream_handle=int(self.stream)):
            raise RuntimeError("execute_async_v3 returned false")

    def copy_output(self) -> None:
        phase5.cuda_call(self.cudart, "D2H logits", self.cudart.cudaMemcpyAsync(
            int(self.host_logits.ctypes.data), self.pointers["logits"], int(self.host_logits.nbytes),
            self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream,
        ))

    def synchronize(self) -> None:
        phase5.cuda_call(self.cudart, "cudaStreamSynchronize", self.cudart.cudaStreamSynchronize(self.stream))
        if self.recorder.num_errors:
            raise RuntimeError(f"TensorRT ErrorRecorder: {self.recorder.serializable()}")

    def infer(self, points: np.ndarray, adj: np.ndarray) -> np.ndarray:
        self.copy_inputs(points, adj)
        self.enqueue()
        self.copy_output()
        self.synchronize()
        output = np.ascontiguousarray(self.host_logits.copy(), dtype=np.float32)
        if output.shape != (1, 2048, 2) or not np.isfinite(output).all():
            raise RuntimeError("TensorRT output contract or finite check failed")
        return output

    def benchmark(self, points: np.ndarray, adj: np.ndarray, warmup: int, iterations: int) -> dict[str, Any]:
        self.copy_inputs(points, adj)
        self.synchronize()
        for _ in range(warmup):
            self.enqueue()
        self.synchronize()
        start_event = phase5.cuda_call(self.cudart, "cudaEventCreate(start)", self.cudart.cudaEventCreate())[0]
        stop_event = phase5.cuda_call(self.cudart, "cudaEventCreate(stop)", self.cudart.cudaEventCreate())[0]
        pure: list[float] = []
        try:
            for _ in range(iterations):
                phase5.cuda_call(self.cudart, "event record start", self.cudart.cudaEventRecord(start_event, self.stream))
                self.enqueue()
                phase5.cuda_call(self.cudart, "event record stop", self.cudart.cudaEventRecord(stop_event, self.stream))
                phase5.cuda_call(self.cudart, "event sync stop", self.cudart.cudaEventSynchronize(stop_event))
                pure.append(float(phase5.cuda_call(self.cudart, "event elapsed", self.cudart.cudaEventElapsedTime(start_event, stop_event))[0]))
        finally:
            phase5.cuda_call(self.cudart, "destroy start event", self.cudart.cudaEventDestroy(start_event))
            phase5.cuda_call(self.cudart, "destroy stop event", self.cudart.cudaEventDestroy(stop_event))
        for _ in range(warmup):
            self.copy_inputs(points, adj); self.enqueue(); self.copy_output(); self.synchronize()
        e2e: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter()
            self.copy_inputs(points, adj); self.enqueue(); self.copy_output(); self.synchronize()
            e2e.append((time.perf_counter() - started) * 1000.0)
        output = np.ascontiguousarray(self.host_logits.copy(), dtype=np.float32)
        if not np.isfinite(output).all():
            raise FloatingPointError("TensorRT benchmark produced NaN/Inf")
        return {"pure_samples_ms": pure, "e2e_samples_ms": e2e, "output_sha256": array_sha256(output)}

    def memory_info(self) -> dict[str, int]:
        free_bytes, total_bytes = phase5.cuda_call(self.cudart, "cudaMemGetInfo", self.cudart.cudaMemGetInfo())
        return {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes), "used_bytes": int(total_bytes - free_bytes)}

    def close(self) -> dict[str, Any]:
        for pointer in reversed(list(self.pointers.values())):
            phase5.cuda_call(self.cudart, "cudaFree", self.cudart.cudaFree(pointer))
        self.pointers.clear()
        phase5.cuda_call(self.cudart, "cudaStreamDestroy", self.cudart.cudaStreamDestroy(self.stream))
        result = {"num_errors": int(self.recorder.num_errors), "errors": self.recorder.serializable()}
        del self.context, self.engine, self.runtime, self.recorder
        gc.collect()
        for handle in self.dll_handles:
            handle.close()
        return result


class PyTorchRunner:
    def __init__(self, checkpoint_path: Path = CHECKPOINT) -> None:
        import torch
        from types import SimpleNamespace
        from deployment.gcn_res_onnx_model import GCNResStandardOps
        from deployment.gcn_res_onnx_wrapper import GCNResOnnxWrapper

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA unavailable")
        self.torch = torch
        self.old = {
            "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn": bool(torch.backends.cudnn.allow_tf32),
            "precision": torch.get_float32_matmul_precision(),
        }
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload["model_state_dict"]
        model = GCNResStandardOps(SimpleNamespace(num_class=2), in_dim=4)
        self.strict_load = str(model.load_state_dict(state, strict=True))
        self.wrapper = GCNResOnnxWrapper(model).to("cuda:0").eval()
        del payload, state, model

    def infer(self, points: np.ndarray, adj: np.ndarray) -> np.ndarray:
        torch = self.torch
        points_gpu = torch.from_numpy(points).to("cuda:0")
        adj_gpu = torch.from_numpy(adj).to("cuda:0")
        with torch.inference_mode():
            output = self.wrapper(points_gpu, adj_gpu)
        torch.cuda.synchronize(0)
        result = np.ascontiguousarray(output.detach().cpu().numpy(), dtype=np.float32)
        if result.shape != (1, 2048, 2) or not np.isfinite(result).all():
            raise RuntimeError("PyTorch output contract or finite check failed")
        return result

    def benchmark(self, points: np.ndarray, adj: np.ndarray, warmup: int, iterations: int) -> dict[str, Any]:
        torch = self.torch
        points_gpu = torch.from_numpy(points).to("cuda:0")
        adj_gpu = torch.from_numpy(adj).to("cuda:0")
        with torch.inference_mode():
            for _ in range(warmup):
                output = self.wrapper(points_gpu, adj_gpu)
            torch.cuda.synchronize(0)
            pure: list[float] = []
            for _ in range(iterations):
                torch.cuda.synchronize(0)
                started = time.perf_counter()
                output = self.wrapper(points_gpu, adj_gpu)
                torch.cuda.synchronize(0)
                pure.append((time.perf_counter() - started) * 1000.0)
            for _ in range(warmup):
                p = torch.from_numpy(points).to("cuda:0")
                a = torch.from_numpy(adj).to("cuda:0")
                host = self.wrapper(p, a).detach().cpu().numpy()
            torch.cuda.synchronize(0)
            e2e: list[float] = []
            for _ in range(iterations):
                started = time.perf_counter()
                p = torch.from_numpy(points).to("cuda:0")
                a = torch.from_numpy(adj).to("cuda:0")
                host = self.wrapper(p, a).detach().cpu().numpy()
                torch.cuda.synchronize(0)
                e2e.append((time.perf_counter() - started) * 1000.0)
        host = np.ascontiguousarray(host, dtype=np.float32)
        if not np.isfinite(host).all():
            raise FloatingPointError("PyTorch benchmark produced NaN/Inf")
        return {"pure_samples_ms": pure, "e2e_samples_ms": e2e, "output_sha256": array_sha256(host)}

    def close(self) -> None:
        torch = self.torch
        del self.wrapper
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(0)
        torch.backends.cuda.matmul.allow_tf32 = self.old["matmul"]
        torch.backends.cudnn.allow_tf32 = self.old["cudnn"]
        torch.set_float32_matmul_precision(self.old["precision"])


def segmentation_metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    return parity.segmentation_metrics(labels.reshape(-1), prediction.reshape(-1))


def numerical_comparison(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(difference)
    left = reference.astype(np.float64).reshape(-1)
    right = actual.astype(np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    labels_left = np.argmax(reference, axis=-1)
    labels_right = np.argmax(actual, axis=-1)
    return {
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "cosine_similarity": float(np.dot(left, right) / denominator),
        "matching_points": int((labels_left == labels_right).sum()),
        "total_points": int(labels_left.size),
        "agreement": float((labels_left == labels_right).mean()),
        "outputs_finite": bool(np.isfinite(reference).all() and np.isfinite(actual).all()),
    }
