# TensorRT Phase 8D: Production Baseline Qualification

## Outcome

The CUB-based strict-FP32 candidate passed production qualification as a **task-equivalent baseline with an explicit numerical exception**.

```text
TENSORRT_CUB_PRODUCTION_QUALIFICATION_PASSED_WITH_NUMERICAL_EXCEPTION
TENSORRT_CUB_STRICT_FP32_TASK_EQUIVALENT_BASELINE_PROMOTED
CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED
```

This result does **not** claim strict numerical equivalence. No model, checkpoint, ONNX mathematics, plugin algorithm, precision policy, or TensorRT tactic was changed during Phase 8D.

Formal artifacts are under:

```text
artifacts/gcn_res_tensorrt/
20260717_173128_144483_phase8d_production_baseline/
```

## Promoted deployment

| Item | Value |
|---|---|
| Deployment ID | `gcn-res-trt-cub-strict-fp32-20260717_173128_144483` |
| Engine | `package/engine/strict_fp32_voxelunique_cub.plan` |
| Engine SHA-256 | `a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299` |
| Plugin | `package/plugins/VoxelUniqueCubPlugin.dll` |
| Plugin SHA-256 | `6641ec147e8eac10206a5c60ba1c1390c398d4b59e32a6a618a37046360ec348` |
| ONNX | `package/model/gcn_res_voxelunique_cub.onnx` |
| ONNX SHA-256 | `16ca5c16c330e6572b1730e80da724231a28b68872a3203c21240348d4d89299` |
| Checkpoint SHA-256 | `311bddf3607d76e6b7ded450b8419bf6ae98f34f50578608b3e6a1c1c3e58d21` |
| TensorRT | `11.1.0.106` |
| CUDA runtime | `12.8` |
| GPU | NVIDIA GeForce RTX 5060, SM `12.0` |
| Precision | strict FP32; TF32/FP16/INT8 disabled |

I/O remains fixed:

```text
points  float32 [1,2048,4]
adj     float32 [1,2048,2048]
logits  float32 [1,2048,2]
```

The package contains 13 versioned files. `checksums.sha256` covers every package file except itself. Package verification and a final default-mode production inference both passed after promotion.

## Qualification evidence

### Precondition and integrity

- Milestone commit: `8f012ac feat(tensorrt): optimize VoxelUnique with CUB and validate engine`.
- Worktree precondition passed before qualification; only Phase 8D implementation files were permitted afterward.
- Candidate and retained baseline hashes matched before and after qualification.
- The previous engine and plugin were not moved, renamed, overwritten, or deleted.

### Cold start

Ten independent Python processes explicitly loaded only `VoxelUniqueCubPlugin.dll`, initialized TensorRT standard plugins, deserialized the candidate engine, created a context, and inferred `weld_65`.

| Measurement | Mean | Min | Max |
|---|---:|---:|---:|
| Plugin load | 4.2648 ms | 3.7697 ms | 5.2853 ms |
| Engine deserialize | 34.9371 ms | 32.1485 ms | 42.8714 ms |
| Context creation | 387.1734 ms | 375.7315 ms | 400.6203 ms |
| First inference | 38.6541 ms | 37.6097 ms | 39.7999 ms |
| Total startup | 675.8311 ms | 657.9445 ms | 694.1443 ms |

Result: 10/10 processes exited successfully, all outputs were finite, all ErrorRecorders remained at zero, and every engine created exactly four CUB plugin instances. Predicted-label hashes were stable. Logits hashes were not bitwise stable and were intentionally not made a cold-start acceptance condition.

### Package-path smoke

`weld_65`, `weld_5`, `weld_12`, and `weld_14` were rerun through the versioned package path. Engine/plugin hashes, I/O, four plugin instances, finite logits, ErrorRecorder zero, and labels against the saved candidate references all passed.

### Full test regression

All 18 fixed test samples used seed 42, 2048 points, `k=6`, identical normalized points, and identical adjacency matrices. PyTorch, old TensorRT baseline, and the CUB candidate ran in separate processes.

- Runtime: 18/18 passed for every backend.
- Candidate vs PyTorch labels: 100% for all samples.
- Candidate vs baseline TensorRT labels: 100% for all samples.
- Accuracy, mIoU, precision, recall, and F1 deltas: exactly zero.
- Test mIoU: `0.936308797729007`.
- Weld-seam F1: `0.946799161765700`.
- Strict per-sample `max_abs < 1e-4`: 13/18.
- Worst sample: `weld_14`.
- Worst max absolute error: `1.2302398681640625e-4`.

The numerical exception is first-class in the deployment manifest, package limitations, qualification record, and rollback manifest.

### Three-round, 18-sample latency distribution

Each backend ran in an independent process. Backend order rotated per round. Each sample used 20 warmups and 100 measurements per round.

| Runtime | Pure mean | Pure P50 | Pure P95 | Pure P99 | Host-to-host E2E mean | E2E P95 |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch strict FP32 | 20.9776 ms | 20.3802 ms | 24.5455 ms | 29.0834 ms | 23.8599 ms | 27.4369 ms |
| Old TensorRT baseline | 41.1458 ms | 39.9305 ms | 48.9264 ms | 49.9237 ms | 44.0713 ms | 51.7711 ms |
| CUB candidate | 4.8209 ms | 4.7589 ms | 5.7756 ms | 6.5597 ms | 7.8194 ms | 9.1425 ms |

Speedups:

- Candidate vs old TensorRT pure: `8.5348x`.
- Candidate vs PyTorch pure: `4.3514x`.
- Candidate vs PyTorch host-to-host E2E: `3.0514x`.
- Candidate samples slower than PyTorch pure: none of 18.

Pure/model-only and host-to-host E2E results are reported separately. No TensorRT E2E number is compared to PyTorch pure as if they were the same scope.

### Determinism

`weld_65`, `weld_5`, and `weld_14` each ran 100 times with identical inputs.

```text
DETERMINISTIC_LABELS_ONLY
```

Labels were bitwise stable for every run. Logits were not bitwise stable; the maximum repeated-run absolute difference was `7.8678131103515625e-6`. This is recorded rather than relabeled as bitwise determinism.

### Soak

The candidate completed 5000/5000 enqueue operations while cycling all 18 samples.

- Finite outputs and reference labels: 5000/5000.
- ErrorRecorder errors: 0.
- Mean/P50/P95/P99 pure latency: `5.6729 / 5.4735 / 6.8165 / 7.4628 ms`.
- Monotonic memory growth detected: false.
- Obvious latency degradation detected: false.
- Last-ten vs first-ten rolling-latency ratio: `0.9131`.

`cudaMemGetInfo` and `nvidia-smi` are observational lifecycle samples, not exact allocator or kernel peak measurements.

### Negative paths

All 8/8 fail-closed cases returned nonzero, reported a concrete error, did not create logits, did not execute inference, and did not fall back:

1. Plugin missing;
2. Plugin hash mismatch;
3. Engine hash mismatch;
4. Truncated engine;
5. Invalid points shape;
6. Invalid adjacency shape;
7. Invalid dtype;
8. GPU compatibility mismatch.

## Rollback

The retained old baseline remains:

- Engine SHA-256: `b76563580089bbc0684e7a8e1edda6028c2d89fd91397c22b4efbd89ecd7fc2c`.
- Plugin SHA-256: `60c48c400cc926f6aa183de62298a203916b1add6393a3483a332624da39d0ab`.

Rollback is never automatic. It requires an explicit validated manifest selection through `scripts/select_gcn_res_tensorrt_baseline.py`; artifact files are not replaced or deleted.

## Boundaries

Phase 8D did not perform FP16, INT8, model/ONNX/plugin-algorithm changes, engine tactic rebuilds, C++ integration, robot integration, safety-threshold changes, or automatic fallback. See `docs/tensorrt_production_runbook.md` for operation.
