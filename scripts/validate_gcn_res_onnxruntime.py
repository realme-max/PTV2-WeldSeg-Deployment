"""Validate exported GCN_res deployment ONNX against its PyTorch reference."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import onnx
import onnxruntime as ort
RTOL = 1e-4
ATOL = 1e-5
MIN_LABEL_AGREEMENT = 0.9999
MODEL_NAME = "gcn_res_deploy_fp32_opset18.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def make_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"gcn_res_onnxruntime.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "onnxruntime_validation.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def compare_logits(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    if reference.shape != (1, 2048, 2) or actual.shape != reference.shape:
        raise RuntimeError(f"Logits shape mismatch: reference={reference.shape}, actual={actual.shape}")
    finite = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
    absolute = np.abs(actual - reference)
    relative = absolute / np.maximum(np.abs(reference), 1e-8)
    reference_labels = np.argmax(reference, axis=-1)
    actual_labels = np.argmax(actual, axis=-1)
    agreement = float(np.mean(reference_labels == actual_labels))
    reference_probability = softmax(reference)
    actual_probability = softmax(actual)
    result = {
        "max_abs_error": float(np.max(absolute)),
        "mean_abs_error": float(np.mean(absolute)),
        "max_relative_error": float(np.max(relative)),
        "predicted_label_agreement": agreement,
        "weld_seam_probability_max_abs_error": float(
            np.max(np.abs(actual_probability[..., 0] - reference_probability[..., 0]))
        ),
        "weld_seam_probability_mean_abs_error": float(
            np.mean(np.abs(actual_probability[..., 0] - reference_probability[..., 0]))
        ),
        "background_probability_max_abs_error": float(
            np.max(np.abs(actual_probability[..., 1] - reference_probability[..., 1]))
        ),
        "background_probability_mean_abs_error": float(
            np.mean(np.abs(actual_probability[..., 1] - reference_probability[..., 1]))
        ),
        "outputs_finite": finite,
        "logits_allclose_rtol_1e-4_atol_1e-5": bool(
            np.allclose(actual, reference, rtol=RTOL, atol=ATOL)
        ),
    }
    if not finite or not result["logits_allclose_rtol_1e-4_atol_1e-5"] or agreement < MIN_LABEL_AGREEMENT:
        raise RuntimeError(f"ONNX Runtime parity threshold failed: {result}")
    return result


def create_session(model_path: Path, provider: str) -> ort.InferenceSession:
    providers = [provider]
    if provider == "CUDAExecutionProvider":
        providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(str(model_path), providers=providers)
    if session.get_providers()[0] != provider:
        raise RuntimeError(f"Requested provider {provider} is not active: {session.get_providers()}")
    return session


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    model_path = run_dir / MODEL_NAME
    logger = make_logger(run_dir)
    if not model_path.is_file():
        raise FileNotFoundError(f"Exported ONNX not found: {model_path}")
    onnx.checker.check_model(onnx.load(model_path), full_check=True)
    available = ort.get_available_providers()
    requested = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        requested.append("CUDAExecutionProvider")
    logger.info("ONNX Runtime %s available providers=%s", ort.__version__, available)

    input_path = run_dir / "export_input.npz"
    reference_path = run_dir / "pytorch_deploy_reference.npz"
    if not input_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError(
            f"Export input/reference missing: {input_path}, {reference_path}"
        )
    with np.load(input_path) as source:
        points = np.asarray(source["points"], dtype=np.float32)
        adjacency = np.asarray(source["adj"], dtype=np.float32)
    with np.load(reference_path) as source:
        reference = np.asarray(source["logits"], dtype=np.float32)
    if points.shape != (1, 2048, 4) or adjacency.shape != (1, 2048, 2048):
        raise RuntimeError(f"Fixed input shape mismatch: points={points.shape}, adj={adjacency.shape}")

    provider_results: dict[str, Any] = {}

    for provider in requested:
        session = create_session(model_path, provider)
        inputs_meta = [(item.name, item.shape, item.type) for item in session.get_inputs()]
        outputs_meta = [(item.name, item.shape, item.type) for item in session.get_outputs()]
        if inputs_meta != [
            ("points", [1, 2048, 4], "tensor(float)"),
            ("adj", [1, 2048, 2048], "tensor(float)"),
        ] or outputs_meta != [("logits", [1, 2048, 2], "tensor(float)")]:
            raise RuntimeError(f"ORT interface mismatch: inputs={inputs_meta}, outputs={outputs_meta}")
        feed = {"points": points, "adj": adjacency}
        actual = session.run(["logits"], feed)[0]
        comparison = compare_logits(reference, actual)
        comparison.update({"sample": "val_00_weld_7", "provider": provider})
        logger.info("Parity %s val_00_weld_7: %s", provider, comparison)
        provider_results[provider] = {
            "sample": comparison,
            "passed": True,
        }
    result = {
        "status": "GCN_RES_ONNXRUNTIME_PARITY_PASSED",
        "onnxruntime_version": ort.__version__,
        "available_providers": available,
        "thresholds": {"rtol": RTOL, "atol": ATOL, "minimum_label_agreement": MIN_LABEL_AGREEMENT},
        "providers": provider_results,
    }
    save_json(run_dir / "onnxruntime_validation.json", result)
    logger.info("GCN_RES_ONNXRUNTIME_PARITY_PASSED artifact_dir=%s", run_dir)
    print(f"GCN_RES_ONNXRUNTIME_PARITY_PASSED\nARTIFACT_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
