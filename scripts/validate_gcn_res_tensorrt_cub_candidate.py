"""Phase 8C runtime, plugin-intermediate, and 18-sample regression validation."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import traceback
from pathlib import Path
from typing import Any

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
import gcn_res_tensorrt_cub_common as common  # noqa: E402
import run_gcn_res_tensorrt_fp32_inference as phase5  # noqa: E402
import smoke_test_gcn_res_tensorrt_engine as phase7a  # noqa: E402
import validate_gcn_res_tensorrt_strict_fp32_multisample as phase6  # noqa: E402

BASELINE_ISOLATED_ENGINE = PROJECT_ROOT / "artifacts/tensorrt_plugin_prototype/20260715_203305_357432_correctness/voxel_unique_correctness.plan"
CUB_ISOLATED_ENGINE = common.PHASE8B_DIR / "voxel_unique_cub.plan"
MAX_N = 2048
SELECTED_PLUGIN_SAMPLES = ("weld_65", "weld_5", "weld_12", "weld_14")


def memory_snapshot(cudart: Any, label: str) -> dict[str, Any]:
    free_bytes, total_bytes = phase5.cuda_call(cudart, f"cudaMemGetInfo({label})", cudart.cudaMemGetInfo())
    return {
        "label": label,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "used_bytes": int(total_bytes - free_bytes),
        "method": "cudaMemGetInfo lifecycle snapshot; not an instantaneous in-kernel peak",
    }


class FullEngineRunner:
    def __init__(
        self,
        trt: Any,
        cudart: Any,
        engine_path: Path,
        library: Any,
        counter_name: str,
        label: str,
        collect_memory: bool = False,
    ) -> None:
        self.trt, self.cudart, self.label = trt, cudart, label
        self.library = library
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.memory: list[dict[str, Any]] = []
        if collect_memory:
            self.memory.append(memory_snapshot(cudart, f"{label}_before_deserialize"))
        ErrorRecorder = phase5.make_error_recorder_class(trt)
        self.recorder = ErrorRecorder()
        self.runtime = trt.Runtime(self.logger)
        self.runtime.error_recorder = self.recorder
        before_count = int(getattr(library, counter_name)())
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"{label}: deserialize_cuda_engine returned None")
        self.engine.error_recorder = self.recorder
        after_count = int(getattr(library, counter_name)())
        if after_count - before_count != 4:
            raise RuntimeError(f"{label}: expected 4 runtime plugin creations, got delta={after_count-before_count}")
        if collect_memory:
            self.memory.append(memory_snapshot(cudart, f"{label}_after_deserialize"))
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"{label}: create_execution_context returned None")
        self.context.error_recorder = self.recorder
        if collect_memory:
            self.memory.append(memory_snapshot(cudart, f"{label}_after_context"))
        self.io = common.engine_io(trt, self.engine)
        self.logits = np.empty((1, 2048, 2), dtype=np.float32)
        self.pointers: dict[str, int] = {}
        self.stream = phase5.cuda_call(cudart, f"{label}: cudaStreamCreate", cudart.cudaStreamCreate())[0]
        sizes = {
            "points": 1 * 2048 * 4 * 4,
            "adj": 1 * 2048 * 2048 * 4,
            "logits": int(self.logits.nbytes),
        }
        for name, byte_count in sizes.items():
            self.pointers[name] = int(phase5.cuda_call(cudart, f"{label}: cudaMalloc({name})", cudart.cudaMalloc(byte_count))[0])
            if not self.context.set_tensor_address(name, self.pointers[name]):
                raise RuntimeError(f"{label}: set_tensor_address({name}) failed")
        if collect_memory:
            self.memory.append(memory_snapshot(cudart, f"{label}_after_device_buffers"))
        self.metadata = {
            "label": label,
            "engine_path": str(engine_path.resolve()),
            "engine_sha256": common.sha256(engine_path),
            "engine_size_bytes": engine_path.stat().st_size,
            "runtime_creation_delta": after_count - before_count,
            "io": self.io,
            "buffer_bytes": sizes,
        }

    def infer(self, points: np.ndarray, adj: np.ndarray) -> np.ndarray:
        for name, array in (("points", points), ("adj", adj)):
            phase5.cuda_call(self.cudart, f"{self.label}: H2D {name}", self.cudart.cudaMemcpyAsync(
                self.pointers[name], int(array.ctypes.data), int(array.nbytes),
                self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
        if not self.context.execute_async_v3(stream_handle=int(self.stream)):
            raise RuntimeError(f"{self.label}: execute_async_v3 returned false")
        phase5.cuda_call(self.cudart, f"{self.label}: D2H logits", self.cudart.cudaMemcpyAsync(
            int(self.logits.ctypes.data), self.pointers["logits"], int(self.logits.nbytes),
            self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
        phase5.cuda_call(self.cudart, f"{self.label}: synchronize", self.cudart.cudaStreamSynchronize(self.stream))
        if self.recorder.num_errors:
            raise RuntimeError(f"{self.label}: ErrorRecorder={self.recorder.serializable()}")
        if self.logits.shape != (1, 2048, 2) or self.logits.dtype != np.float32 or not np.isfinite(self.logits).all():
            raise RuntimeError(f"{self.label}: invalid logits")
        return np.ascontiguousarray(self.logits.copy())

    def close(self) -> dict[str, Any]:
        for name, pointer in reversed(list(self.pointers.items())):
            phase5.cuda_call(self.cudart, f"{self.label}: cudaFree({name})", self.cudart.cudaFree(pointer))
        self.pointers.clear()
        phase5.cuda_call(self.cudart, f"{self.label}: cudaStreamDestroy", self.cudart.cudaStreamDestroy(self.stream))
        result = {"num_errors": self.recorder.num_errors, "errors": self.recorder.serializable()}
        del self.context, self.engine, self.runtime, self.recorder
        return result


class IsolatedUniqueRunner:
    def __init__(self, trt: Any, cudart: Any, engine_path: Path, label: str) -> None:
        self.trt, self.cudart, self.label = trt, cudart, label

        class OutputAllocator(trt.IOutputAllocator):
            def __init__(allocator_self, pointer: int, capacity: int) -> None:
                trt.IOutputAllocator.__init__(allocator_self)
                allocator_self.pointer = int(pointer)
                allocator_self.capacity = int(capacity)
                allocator_self.shape = None

            def reallocate_output(allocator_self, tensor_name: str, memory: int, size: int, alignment: int) -> int:
                del tensor_name, memory, alignment
                return allocator_self.pointer if int(size) <= allocator_self.capacity else 0

            def reallocate_output_async(allocator_self, tensor_name: str, memory: int, size: int, alignment: int, stream_handle: int) -> int:
                del tensor_name, memory, alignment, stream_handle
                return allocator_self.pointer if int(size) <= allocator_self.capacity else 0

            def notify_shape(allocator_self, tensor_name: str, shape: Any) -> None:
                del tensor_name
                allocator_self.shape = [int(value) for value in shape]

        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"{label}: isolated engine deserialization failed")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"{label}: isolated context creation failed")
        self.host_count = np.empty((1,), dtype=np.int32)
        self.host_values = np.empty((MAX_N,), dtype=np.int64)
        self.host_inverse = np.empty((MAX_N,), dtype=np.int64)
        sizes = {"voxel_key": MAX_N * 8, "voxel_count": 4, "unique_values": MAX_N * 8, "inverse_indices": MAX_N * 8}
        self.pointers = {name: int(phase5.cuda_call(cudart, f"{label}: cudaMalloc({name})", cudart.cudaMalloc(size))[0]) for name, size in sizes.items()}
        for name, pointer in self.pointers.items():
            if not self.context.set_tensor_address(name, pointer):
                raise RuntimeError(f"{label}: set_tensor_address({name}) failed")
        self.allocator = OutputAllocator(self.pointers["unique_values"], sizes["unique_values"])
        if not self.context.set_output_allocator("unique_values", self.allocator):
            raise RuntimeError(f"{label}: set_output_allocator failed")
        self.stream = phase5.cuda_call(cudart, f"{label}: cudaStreamCreate", cudart.cudaStreamCreate())[0]

    def infer(self, keys: np.ndarray) -> dict[str, Any]:
        keys = np.ascontiguousarray(keys, dtype=np.int64).reshape(-1)
        n = int(keys.size)
        if not 1 <= n <= MAX_N:
            raise ValueError(f"{self.label}: N={n} outside [1,2048]")
        self.allocator.shape = None
        if not self.context.set_input_shape("voxel_key", (n,)):
            raise RuntimeError(f"{self.label}: set_input_shape({n}) failed")
        phase5.cuda_call(self.cudart, f"{self.label}: H2D", self.cudart.cudaMemcpyAsync(
            self.pointers["voxel_key"], int(keys.ctypes.data), int(keys.nbytes),
            self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
        if not self.context.execute_async_v3(stream_handle=int(self.stream)):
            raise RuntimeError(f"{self.label}: isolated enqueue failed")
        for name, array, byte_count in (
            ("voxel_count", self.host_count, self.host_count.nbytes),
            ("unique_values", self.host_values, self.host_values.nbytes),
            ("inverse_indices", self.host_inverse, n * 8),
        ):
            phase5.cuda_call(self.cudart, f"{self.label}: D2H {name}", self.cudart.cudaMemcpyAsync(
                int(array.ctypes.data), self.pointers[name], int(byte_count),
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
        phase5.cuda_call(self.cudart, f"{self.label}: sync", self.cudart.cudaStreamSynchronize(self.stream))
        m = int(self.host_count[0])
        return {
            "voxel_count": m,
            "unique_values": self.host_values[:m].copy(),
            "inverse_indices": self.host_inverse[:n].copy(),
            "runtime_output_shape": list(self.allocator.shape or []),
        }

    def close(self) -> None:
        for name, pointer in reversed(list(self.pointers.items())):
            phase5.cuda_call(self.cudart, f"{self.label}: cudaFree({name})", self.cudart.cudaFree(pointer))
        phase5.cuda_call(self.cudart, f"{self.label}: destroy stream", self.cudart.cudaStreamDestroy(self.stream))
        del self.context, self.engine, self.runtime


def capture_stage_keys(torch_runner: phase6.PyTorchStrictFP32Runner, points: np.ndarray, adj: np.ndarray) -> list[np.ndarray]:
    import torch
    import deployment.gcn_res_onnx_model as model_module
    from deployment.onnx_voxel_pool import standard_voxel_pool_with_metadata

    original_pool = model_module.standard_voxel_pool
    keys_by_stage: list[np.ndarray] = []

    def capture_pool(xyz: torch.Tensor, features: torch.Tensor, voxel_size: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = standard_voxel_pool_with_metadata(xyz, features, voxel_size)
        raw_keys = result.unique_local_keys.index_select(0, result.point_to_voxel.reshape(-1))
        keys_by_stage.append(np.ascontiguousarray(raw_keys.detach().cpu().numpy(), dtype=np.int64))
        return result.pooled_points, result.pooled_features

    model_module.standard_voxel_pool = capture_pool
    try:
        _ = torch_runner.infer(points, adj)
    finally:
        model_module.standard_voxel_pool = original_pool
    if len(keys_by_stage) != 4:
        raise RuntimeError(f"Expected four voxel stages, captured {len(keys_by_stage)}")
    return keys_by_stage


def load_sample(dataset: Any, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    sample = dataset[index]
    xyz = sample["normalized_xyz"].unsqueeze(0).to(torch.float32)
    points = np.ascontiguousarray(evaluation.make_model_input(xyz).numpy(), dtype=np.float32)
    adj_tensor, adj_seconds = evaluation.build_adjacency_cpu(xyz)
    adj = np.ascontiguousarray(adj_tensor.numpy(), dtype=np.float32)
    labels = np.ascontiguousarray(sample["labels"].unsqueeze(0).numpy(), dtype=np.int64)
    return points, adj, labels, {"sample_id": sample["sample_name"], "logical_path": sample["logical_path"], "adjacency_cpu_seconds": adj_seconds}


def metric_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    keys = ("overall_accuracy", "weld_seam_iou", "background_iou", "miou", "weld_seam_precision", "weld_seam_recall", "weld_seam_f1")
    return {key: abs(float(left[key]) - float(right[key])) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    candidate_engine = run_dir / "strict_fp32_voxelunique_cub_candidate.plan"
    candidate_plugin = run_dir / "VoxelUniqueCubPlugin.dll"
    protected_before = common.protected_snapshot()
    candidate_hashes_before = {"engine": common.sha256(candidate_engine), "plugin": common.sha256(candidate_plugin)}
    summary: dict[str, Any] = {"status": "VOXELUNIQUE_CUB_CANDIDATE_RUNTIME_OR_REGRESSION_FAILED"}
    common.dump_json(run_dir / "candidate_multisample_parity.json", summary)
    handles: list[Any] = []
    torch_runner = candidate_runner = baseline_runner = baseline_unique = cub_unique = None
    try:
        handles = common.configure_dll_search(candidate_plugin)
        # Baseline DLL lives in a separate directory and is loaded only for three-way regression.
        if hasattr(__import__("os"), "add_dll_directory"):
            handles.append(__import__("os").add_dll_directory(str(common.BASELINE_PLUGIN.parent)))
        import tensorrt as trt
        import torch
        from cuda.bindings import runtime as cudart

        phase5.cuda_call(cudart, "cudaSetDevice(0)", cudart.cudaSetDevice(0))
        if not trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), ""):
            raise RuntimeError("init_libnvinfer_plugins returned false")
        cub_library, cub_info = common.load_cub_plugin(candidate_plugin)
        baseline_library, baseline_info = phase4.load_plugin_library(common.BASELINE_PLUGIN)
        if not cub_info["registered"] or not baseline_info["registration_function_returned"]:
            raise RuntimeError("Plugin registration failed")

        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        old_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        old_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
        old_precision = torch.get_float32_matmul_precision()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        dataset = evaluation.FixedWeldEvaluationDataset("test")
        if len(dataset) != 18:
            raise RuntimeError(f"Expected 18 test samples, got {len(dataset)}")
        torch_runner = phase6.PyTorchStrictFP32Runner(common.CHECKPOINT)
        candidate_runner = FullEngineRunner(trt, cudart, candidate_engine, cub_library, "getVoxelUniqueCubRuntimeCreationCount", "candidate", True)

        fixed_points, fixed_adj, fixed_meta = phase7a.read_fixed_input(common.PHASE6_DIR / "per_sample_results.json")
        smoke_logits = candidate_runner.infer(fixed_points, fixed_adj)
        smoke = {
            "status": "CANDIDATE_RUNTIME_VALIDATION_PASSED",
            "sample": fixed_meta,
            "deserialize": "PASS", "context_creation": "PASS", "enqueueV3": "PASS",
            "output_shape": list(smoke_logits.shape), "output_dtype": str(smoke_logits.dtype),
            "output_finite": bool(np.isfinite(smoke_logits).all()),
            "error_recorder_errors": candidate_runner.recorder.num_errors,
            "runtime_plugin_instances": 4,
            "runtime_plugin_identity": "com.tensorrt.ptv2.experimental::VoxelUniqueCub:1",
            "logits": {"min": float(smoke_logits.min()), "max": float(smoke_logits.max()), "mean": float(smoke_logits.mean()), "std": float(smoke_logits.std())},
        }
        common.dump_json(run_dir / "runtime_smoke_report.json", smoke)

        baseline_runner = FullEngineRunner(trt, cudart, common.BASELINE_ENGINE, baseline_library, "getVoxelUniqueRuntimeCreationCount", "baseline")
        baseline_unique = IsolatedUniqueRunner(trt, cudart, BASELINE_ISOLATED_ENGINE, "baseline_unique")
        cub_unique = IsolatedUniqueRunner(trt, cudart, CUB_ISOLATED_ENGINE, "cub_unique")
        sample_index = {record.sample_name: index for index, record in enumerate(dataset.records)}
        plugin_records: list[dict[str, Any]] = []
        for sample_id in SELECTED_PLUGIN_SAMPLES:
            points, adj, _, _ = load_sample(dataset, sample_index[sample_id])
            for stage, keys in enumerate(capture_stage_keys(torch_runner, points, adj), start=1):
                baseline_output = baseline_unique.infer(keys)
                candidate_output = cub_unique.infer(keys)
                record = {
                    "sample_id": sample_id, "stage": f"tdb_{stage}", "input_n": int(keys.size),
                    "input_sha256": phase5.array_sha256(keys),
                    "baseline_voxel_count": baseline_output["voxel_count"],
                    "candidate_voxel_count": candidate_output["voxel_count"],
                    "voxel_count_exact": baseline_output["voxel_count"] == candidate_output["voxel_count"],
                    "unique_values_exact": bool(np.array_equal(baseline_output["unique_values"], candidate_output["unique_values"])),
                    "inverse_indices_exact": bool(np.array_equal(baseline_output["inverse_indices"], candidate_output["inverse_indices"])),
                    "runtime_output_shape_exact": baseline_output["runtime_output_shape"] == candidate_output["runtime_output_shape"],
                    "baseline_runtime_shape": baseline_output["runtime_output_shape"],
                    "candidate_runtime_shape": candidate_output["runtime_output_shape"],
                }
                record["passed"] = all(record[key] for key in ("voxel_count_exact", "unique_values_exact", "inverse_indices_exact", "runtime_output_shape_exact"))
                plugin_records.append(record)
        plugin_parity = {
            "status": "CANDIDATE_PLUGIN_INTERMEDIATE_PARITY_PASSED" if all(item["passed"] for item in plugin_records) else "CANDIDATE_PLUGIN_INTERMEDIATE_PARITY_FAILED",
            "samples": list(SELECTED_PLUGIN_SAMPLES), "stages_per_sample": 4,
            "comparisons": len(plugin_records), "passed_comparisons": sum(item["passed"] for item in plugin_records),
            "all_elements_bitwise_exact": all(item["passed"] for item in plugin_records), "records": plugin_records,
            "method": "Same real per-stage key tensor passed through isolated baseline and candidate TensorRT engines; formal engines unchanged.",
        }
        common.dump_json(run_dir / "plugin_intermediate_parity.json", plugin_parity)
        if not plugin_parity["all_elements_bitwise_exact"]:
            raise RuntimeError("Candidate/baseline plugin intermediate parity failed")

        per_sample: list[dict[str, Any]] = []
        labels_all: list[np.ndarray] = []
        pt_preds_all: list[np.ndarray] = []
        baseline_preds_all: list[np.ndarray] = []
        candidate_preds_all: list[np.ndarray] = []
        for index in range(len(dataset)):
            points, adj, labels, meta = load_sample(dataset, index)
            pt_logits = torch_runner.infer(points, adj)
            baseline_logits = baseline_runner.infer(points, adj)
            candidate_logits = candidate_runner.infer(points, adj)
            candidate_vs_pt = phase6.numerical_comparison(pt_logits, candidate_logits)
            candidate_vs_baseline = phase6.numerical_comparison(baseline_logits, candidate_logits)
            pt_pred = np.argmax(pt_logits, axis=-1)
            baseline_pred = np.argmax(baseline_logits, axis=-1)
            candidate_pred = np.argmax(candidate_logits, axis=-1)
            candidate_metrics = parity.segmentation_metrics(labels, candidate_pred)
            record = {
                "sample_index": index, **meta,
                "points_sha256": phase5.array_sha256(points), "adj_sha256": phase5.array_sha256(adj),
                "candidate_vs_pytorch": candidate_vs_pt,
                "candidate_vs_baseline_tensorrt": candidate_vs_baseline,
                "candidate_accuracy": candidate_metrics["overall_accuracy"],
                "candidate_miou": candidate_metrics["miou"],
                "candidate_weld_f1": candidate_metrics["weld_seam_f1"],
                "finite": bool(candidate_vs_pt["outputs_finite"] and candidate_vs_baseline["outputs_finite"]),
            }
            per_sample.append(record)
            labels_all.append(labels.reshape(-1)); pt_preds_all.append(pt_pred.reshape(-1)); baseline_preds_all.append(baseline_pred.reshape(-1)); candidate_preds_all.append(candidate_pred.reshape(-1))
            print(f"{index+1:02d}/18 {meta['sample_id']}: cand-vs-pt max={candidate_vs_pt['max_abs_error']:.9e}, cand-vs-base max={candidate_vs_baseline['max_abs_error']:.9e}, agreement={candidate_vs_pt['agreement']:.6f}")

        labels_cat = np.concatenate(labels_all)
        pt_metrics = parity.segmentation_metrics(labels_cat, np.concatenate(pt_preds_all))
        baseline_metrics = parity.segmentation_metrics(labels_cat, np.concatenate(baseline_preds_all))
        candidate_metrics = parity.segmentation_metrics(labels_cat, np.concatenate(candidate_preds_all))
        candidate_vs_pt_deltas = metric_deltas(pt_metrics, candidate_metrics)
        candidate_vs_baseline_deltas = metric_deltas(baseline_metrics, candidate_metrics)
        runtime_pass = len(per_sample) == 18 and all(item["finite"] for item in per_sample) and candidate_runner.recorder.num_errors == 0
        task_pass = all(item["candidate_vs_baseline_tensorrt"]["agreement"] == 1.0 for item in per_sample) and max(candidate_vs_baseline_deltas.values()) < 1e-5
        strict_pass = all(item["candidate_vs_pytorch"]["max_abs_error"] < 1e-4 for item in per_sample)
        summary = {
            "status": "VOXELUNIQUE_CUB_CANDIDATE_ENGINE_REGRESSION_CORE_COMPLETED" if runtime_pass and task_pass else "VOXELUNIQUE_CUB_CANDIDATE_ENGINE_REGRESSION_FAILED",
            "runtime_status": "CANDIDATE_RUNTIME_VALIDATION_PASSED" if runtime_pass else "CANDIDATE_RUNTIME_VALIDATION_FAILED",
            "task_status": "CANDIDATE_TASK_LEVEL_EQUIVALENCE_CONFIRMED" if task_pass else "CANDIDATE_TASK_LEVEL_EQUIVALENCE_FAILED",
            "strict_numerical_status": "CANDIDATE_STRICT_NUMERICAL_THRESHOLD_PASSED" if strict_pass else "CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED",
            "strict_threshold": {"per_sample_max_abs_error_strictly_less_than": 1e-4, "not_relaxed": True},
            "total_samples": 18, "runtime_passed_samples": sum(item["finite"] for item in per_sample),
            "strict_threshold_passed_samples": sum(item["candidate_vs_pytorch"]["max_abs_error"] < 1e-4 for item in per_sample),
            "worst_candidate_vs_pytorch": max(per_sample, key=lambda item: item["candidate_vs_pytorch"]["max_abs_error"]),
            "worst_candidate_vs_baseline": max(per_sample, key=lambda item: item["candidate_vs_baseline_tensorrt"]["max_abs_error"]),
            "pytorch_metrics": pt_metrics, "baseline_tensorrt_metrics": baseline_metrics, "candidate_metrics": candidate_metrics,
            "candidate_vs_pytorch_metric_deltas": candidate_vs_pt_deltas,
            "candidate_vs_baseline_metric_deltas": candidate_vs_baseline_deltas,
            "all_candidate_vs_baseline_labels_exact": all(item["candidate_vs_baseline_tensorrt"]["agreement"] == 1.0 for item in per_sample),
            "all_candidate_vs_pytorch_labels_exact": all(item["candidate_vs_pytorch"]["agreement"] == 1.0 for item in per_sample),
            "plugin_intermediate_status": plugin_parity["status"],
            "candidate_engine": candidate_runner.metadata, "baseline_engine": baseline_runner.metadata,
        }
        common.dump_json(run_dir / "candidate_multisample_parity.json", summary)
        common.dump_json(run_dir / "accuracy_summary.json", {
            "pytorch": pt_metrics, "baseline_tensorrt": baseline_metrics, "candidate_tensorrt": candidate_metrics,
            "candidate_vs_pytorch_delta": candidate_vs_pt_deltas, "candidate_vs_baseline_delta": candidate_vs_baseline_deltas,
        })
        fieldnames = ["sample_index", "sample_id", "max_abs_vs_pytorch", "mean_abs_vs_pytorch", "rmse_vs_pytorch", "cosine_vs_pytorch", "agreement_vs_pytorch", "max_abs_vs_baseline", "agreement_vs_baseline", "candidate_accuracy", "candidate_miou", "candidate_weld_f1", "strict_threshold_passed"]
        with (run_dir / "per_sample_results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames); writer.writeheader()
            for item in per_sample:
                a, b = item["candidate_vs_pytorch"], item["candidate_vs_baseline_tensorrt"]
                writer.writerow({"sample_index": item["sample_index"], "sample_id": item["sample_id"], "max_abs_vs_pytorch": a["max_abs_error"], "mean_abs_vs_pytorch": a["mean_abs_error"], "rmse_vs_pytorch": a["rmse"], "cosine_vs_pytorch": a["cosine_similarity"], "agreement_vs_pytorch": a["agreement"], "max_abs_vs_baseline": b["max_abs_error"], "agreement_vs_baseline": b["agreement"], "candidate_accuracy": item["candidate_accuracy"], "candidate_miou": item["candidate_miou"], "candidate_weld_f1": item["candidate_weld_f1"], "strict_threshold_passed": a["max_abs_error"] < 1e-4})

        workspace = json.loads((common.PHASE8B_DIR / "workspace_layout.json").read_text(encoding="utf-8"))
        memory = {
            "method": "Isolated-process cudaMemGetInfo lifecycle snapshots; not an in-kernel transient peak.",
            "candidate_engine_size_bytes": candidate_engine.stat().st_size,
            "candidate_lifecycle_snapshots": candidate_runner.memory,
            "candidate_plugin_workspace_per_instance_bytes_at_n2048": workspace["total_bytes"],
            "candidate_plugin_instances": 4,
            "candidate_plugin_workspace_upper_bound_sum_bytes": 4 * workspace["total_bytes"],
            "workspace_note": "Per-instance workspace is visible in plugin configuration; cudaMemGetInfo cannot isolate transient TensorRT workspace reuse.",
        }
        common.dump_json(run_dir / "memory_summary.json", memory)
        if not runtime_pass or not task_pass:
            raise RuntimeError(f"Core regression failed: runtime={runtime_pass}, task={task_pass}")
        if common.protected_snapshot() != protected_before or {"engine": common.sha256(candidate_engine), "plugin": common.sha256(candidate_plugin)} != candidate_hashes_before:
            raise RuntimeError("Protected or candidate artifacts changed during validation")
        print(summary["runtime_status"])
        print(summary["task_status"])
        print(summary["strict_numerical_status"])
        print("VOXELUNIQUE_CUB_CANDIDATE_CORE_REGRESSION_COMPLETED")
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.set_float32_matmul_precision(old_precision)
        return 0
    except Exception as exc:
        summary["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        common.dump_json(run_dir / "candidate_multisample_parity.json", summary)
        print(traceback.format_exc(), file=sys.stderr)
        print("VOXELUNIQUE_CUB_CANDIDATE_ENGINE_FAILED")
        return 1
    finally:
        for runner in (cub_unique, baseline_unique):
            if runner is not None:
                try: runner.close()
                except Exception: pass
        for runner in (baseline_runner, candidate_runner):
            if runner is not None:
                try: runner.close()
                except Exception: pass
        if torch_runner is not None:
            try: torch_runner.close()
            except Exception: pass
        gc.collect()
        _ = handles


if __name__ == "__main__":
    raise SystemExit(main())
