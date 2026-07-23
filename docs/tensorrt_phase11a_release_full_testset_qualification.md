# TensorRT Phase 11A: Release Package Full Test-Set Qualification

## 1. Objective

Phase 11A validates the frozen production deployment pipeline on the complete
18-sample weld test split:

```text
TXT point cloud
  -> production C++ preprocessing
  -> TensorRT Strict FP32 inference
  -> C++ post-processing
  -> metrics, geometry and Release-package exports
```

This phase is qualification only. It did not rebuild the Engine, modify the
Plugin, alter the model or checkpoint, change precision, change the split, or
weaken any threshold.

The authoritative artifact root is:

```text
artifacts/gcn_res_tensorrt/
20260723_202430_351197_phase11a_release_full_testset_qualification/
```

## 2. Frozen production assets

Deployment ID:
`gcn-res-trt-cub-strict-fp32-20260717_173128_144483`.

| Asset | SHA-256 |
|---|---|
| TensorRT Engine | `a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299` |
| VoxelUniqueCub Plugin | `6641ec147e8eac10206a5c60ba1c1390c398d4b59e32a6a618a37046360ec348` |
| deployment ONNX | `16ca5c16c330e6572b1730e80da724231a28b68872a3203c21240348d4d89299` |
| GCN_res checkpoint | `311bddf3607d76e6b7ded450b8419bf6ae98f34f50578608b3e6a1c1c3e58d21` |

Precision remained Strict FP32: TF32, FP16 and INT8 were disabled. The
requested Phase 10C.1 package no longer existed at the recorded path, so the
newer qualified Phase 10C.2 package was used:

```text
D:\PTV2_Weld_App_0.1.1_Phase10C2_QUALIFIED_20260723_194357
```

Its Engine and Plugin hashes exactly match the frozen production assets.

## 3. Authoritative test split

The source of truth is
`data/weld/train_test_split/sub_shuffled_test_file_list.json`, SHA-256
`d7464be1b9e0efc6e02bb2490e78e9cf13b10368a2be9bdf0bd86d64f40aff76`.
The original JSON order was preserved:

1. `weld_65`
2. `weld_30`
3. `weld_28`
4. `weld_81`
5. `weld_88`
6. `weld_5`
7. `weld_55`
8. `weld_76`
9. `weld_12`
10. `weld_70`
11. `weld_14`
12. `weld_18`
13. `weld_29`
14. `weld_32`
15. `weld_36`
16. `weld_4`
17. `weld_15`
18. `weld_82`

The train/val/test counts are 54/18/18 with zero overlap. All 18 test paths
are unique and present. Every test TXT has 2048 finite, four-column rows and
labels restricted to 0/1.

## 4. Execution route

The added `weld_sdk_testset_qualification` C++17 runner initializes
`WeldDetector` once, then reuses the TensorRT Runtime, Engine, execution
context, CUDA stream and buffers while iterating the authoritative manifest.
It emits machine-readable JSONL records for preprocessing, inference,
post-processing, timings and process resources.

The orchestration script builds the VS2022 x64 Release target, reconstructs
the exact C++ sampled input for Python references, runs the repository SDK
backend, launches the package from an external D-drive runtime root, validates
exports, and performs the stability and rejection probes.

Actual build toolchain evidence from `configure.log`:

- Visual Studio 2022 x64 Release;
- MSVC 19.38.33130 at
  `D:/vs2022/Community/VC/Tools/MSVC/14.38.33130/bin/Hostx64/x64/cl.exe`;
- Windows SDK 10.0.22621;
- CUDA Toolkit 12.8.93.

The generic `compiler` field in `environment.json` reflects an older `cl.exe`
found on PATH and is not the compiler selected by CMake.

## 5. Ground-truth isolation

The TXT fourth column is used only as sampled ground truth after prediction.
Model features are:

```text
[normalized_x, normalized_y, normalized_z, constant_one]
```

The audit confirmed that labels are not passed to the feature builder, every
fourth feature equals one, and sampled ground-truth labels follow the exact
C++ sampled indices. No label leakage was detected.

## 6. Per-sample results

All 18 samples completed with 2048 sampled points, finite logits and geometry,
successful SDK status, and zero TensorRT ErrorRecorder errors.

| Sample | Accuracy | Weld P | Weld R | Weld F1 | mIoU | Predicted weld points | Strict max abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| weld_65 | 0.982422 | 0.947368 | 0.887892 | 0.916667 | 0.913347 | 209 | 0.000067711 |
| weld_30 | 0.979004 | 0.978365 | 0.970203 | 0.974267 | 0.957489 | 832 | 0.000062466 |
| weld_28 | 0.982910 | 0.982290 | 0.976526 | 0.979400 | 0.965424 | 847 | 0.000056267 |
| weld_81 | 0.988281 | 0.930693 | 0.949495 | 0.940000 | 0.936945 | 202 | 0.000073433 |
| weld_88 | 0.982910 | 0.805882 | 0.985612 | 0.886731 | 0.889098 | 170 | 0.000104904 |
| weld_5 | 0.979004 | 0.948718 | 0.877470 | 0.911704 | 0.907094 | 234 | 0.000092030 |
| weld_55 | 0.963867 | 0.941057 | 0.911417 | 0.926000 | 0.907755 | 492 | 0.000064135 |
| weld_76 | 0.985352 | 0.928571 | 0.919192 | 0.923858 | 0.921207 | 196 | 0.000080585 |
| weld_12 | 0.981934 | 0.981221 | 0.863636 | 0.918681 | 0.914737 | 213 | 0.000098228 |
| weld_70 | 0.972168 | 0.922222 | 0.794258 | 0.853470 | 0.857054 | 180 | 0.000070572 |
| weld_14 | 0.987793 | 0.957854 | 0.946970 | 0.952381 | 0.947593 | 261 | 0.000099659 |
| weld_18 | 0.977051 | 0.926641 | 0.895522 | 0.910816 | 0.905121 | 259 | 0.000105381 |
| weld_29 | 0.977051 | 0.977011 | 0.969213 | 0.973097 | 0.954186 | 870 | 0.000050545 |
| weld_32 | 0.979492 | 0.978443 | 0.971463 | 0.974940 | 0.958494 | 835 | 0.000055313 |
| weld_36 | 0.973633 | 0.962353 | 0.973810 | 0.968047 | 0.947085 | 850 | 0.000059128 |
| weld_4 | 0.981445 | 0.962963 | 0.889734 | 0.924901 | 0.919673 | 243 | 0.000122070 |
| weld_15 | 0.977051 | 0.959091 | 0.847390 | 0.899787 | 0.896122 | 220 | 0.000101566 |
| weld_82 | 0.974609 | 0.949438 | 0.797170 | 0.866667 | 0.868516 | 178 | 0.000071049 |

The machine-readable per-sample CSV/JSONL files are the numerical source of
truth; this table is a six-decimal presentation.

## 7. Aggregate metrics

Label semantics are class 0 = `weld_seam`, class 1 = `background`.
The global confusion matrix uses rows = ground truth and columns = prediction:

```text
[[ 7000,   475],
 [  291, 29098]]
```

| Metric | Global point-level | Mean per sample |
|---|---:|---:|
| Accuracy | 0.979220920 | 0.979220920 |
| Weld precision | 0.960087779 | 0.946676877 |
| Weld recall | 0.936454849 | 0.912609567 |
| Weld F1 | 0.948124069 | 0.927856360 |
| Weld IoU | 0.901364924 | 0.867570305 |
| Background IoU | 0.974350388 | 0.973200696 |
| mIoU | 0.937857656 | 0.920385500 |

The measured global mIoU and weld F1 are close to the historical values
(approximately 0.9363 and 0.9468), but the actual Phase 11A results above are
authoritative for the production C++ preprocessing route.

## 8. Python/TensorRT comparison

Fresh PyTorch CUDA and TensorRT inference used the exact sampled indices,
features and k=6 adjacency reconstructed from the production C++ run.
TensorRT, the C++ SDK and Python TensorRT produced identical labels for all
18 samples. PyTorch and TensorRT also had 100% point-label agreement on every
sample. The comparison is therefore a task-level same-input pass with the
strict numerical exception described next.

## 9. Strict numerical exception

The unchanged criterion is per-sample `max_abs_error < 1e-4`.

Current official Phase 11A same-input run:

- 14/18 passed;
- failures: `weld_88`, `weld_18`, `weld_4`, `weld_15`;
- worst sample: `weld_4`;
- worst maximum absolute logit error: `0.0001220703125`.

Historical Phase 8D frozen result remains separately recorded:

- 13/18 passed;
- failures: `weld_5`, `weld_12`, `weld_14`, `weld_4`, `weld_15`;
- worst sample: `weld_14`;
- worst maximum absolute logit error:
  `0.00012302398681640625`.

An immediately preceding parity-only process produced a slightly different
near-threshold pass set, consistent with the already documented CUDA
scatter/reduction FP32 noise. Labels remained exact. Consequently, no
bitwise-determinism or strict numerical-equivalence claim is made:

```text
CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED
```

## 10. Repeated-run determinism

The complete 18-sample sequence ran three times in one process. For every
sample, labels and weld counts were exact, geometry maximum difference was
zero, status remained successful, and ErrorRecorder remained zero. Logits
were not captured in a form that supports a bitwise-determinism claim, so the
claim is limited to task outputs.

## 11. Cold-start results

Five fresh-process cold starts on `weld_65` passed:

- startup success: 5/5;
- mean initialization: 642.309980 ms;
- mean first detect: 66.290621 ms;
- mean process wall time: 1027.558900 ms;
- Engine SHA, Plugin load and context creation passed;
- ErrorRecorder errors: zero;
- no crash or intermittent DLL-load failure.

## 12. Soak results

The single-process soak ran 18 samples × 20 rounds = 360 detections:

- successful detections: 360/360;
- Engine initializations: 1;
- Plugin loads: 1;
- labels task-deterministic;
- ErrorRecorder errors: zero;
- working-set growth: 8,683,520 bytes, followed by a plateau;
- private-memory change: -15,597,568 bytes;
- GPU-used growth: 0 bytes;
- handle growth: 0;
- thread growth: 1.

No monotonic resource growth or runtime instability was observed over this
bounded interval. This is not a proof that all memory leaks are absent.

## 13. Timing methodology

Warm timing initializes the runtime once, performs one warm-up, preserves the
authoritative JSON sample order, measures all 18 samples, and excludes cold
start, export and Qt rendering.

| Stage | Mean ms | Median ms | Min ms | Max ms | P95 ms |
|---|---:|---:|---:|---:|---:|
| Load cloud | 1.601206 | 1.531350 | 1.489100 | 2.374200 | 2.081035 |
| Sampling | 0.025761 | 0.025100 | 0.024500 | 0.032100 | 0.030995 |
| Feature build | 0.017639 | 0.016800 | 0.016300 | 0.023300 | 0.022110 |
| CPU adjacency | 18.512883 | 18.202100 | 17.797600 | 20.234000 | 19.989369 |
| TensorRT CUDA | 4.927732 | 4.902560 | 4.371744 | 5.588736 | 5.386858 |
| TensorRT wall | 7.836667 | 7.777400 | 6.947000 | 9.290300 | 8.721990 |
| Post-process | 0.089272 | 0.078850 | 0.073200 | 0.134300 | 0.128265 |
| Total SDK detect | 28.106839 | 27.671100 | 26.663000 | 31.079300 | 31.022689 |

## 14. Bottleneck analysis

CPU k=6 adjacency construction is the primary measured bottleneck:

- adjacency: 18.512883 ms, 65.8661% of warm SDK detect;
- TensorRT wall inference: 7.836667 ms, 27.8817%;
- cloud loading: 5.6969%;
- all other measured stages combined: below 1%.

No KNN or other performance optimization was performed in Phase 11A.

## 15. Package/source equivalence

Repository-built SDK output was compared against the Phase 10C.2 package
launched from:

```text
D:\PTV2_Phase11A_Runtime_20260723_202430_351197
```

That root is outside the repository, build tree and artifact build directory.
Inputs were copied to the D-drive runtime root. Across all 18 samples:

- labels were exact;
- geometry was within 1e-5;
- Engine and Plugin hashes were exact;
- no hidden repository input dependency was observed.

## 16. Export validation

The packaged application produced and validated all 18 export sets:

- `weld_result.json` parsed;
- `weld_points.ply` vertex counts matched predicted weld counts and all PLY
  labels were weld class 0;
- `prediction.txt` contained 2048 rows;
- manifest hashes matched;
- task IDs matched;
- values were finite;
- no partial temporary directory remained;
- `detection_view.png` was present and validated for all 18.

## 17. Fail-closed validation

All 15 required probes passed: missing Engine, missing Plugin, wrong Engine
SHA, wrong Plugin SHA, missing cloud, malformed TXT, fewer than 2048 points,
NaN coordinate, nonfinite logits, no-weld output, unwritable export
directory, corrupted manifest, missing `qwindows.dll`, missing Plugin
dependency, and repeated initialization. Rejected cases did not crash,
silently fall back or produce a successful fake output.

## 18. GUI subset smoke

The selected cases were:

- `weld_88`: lowest weld ratio;
- `weld_29`: highest weld ratio;
- `weld_14`: historical Phase 8D strict worst case;
- `weld_65`: qualified reference.

For all four, the product-smoke route rendered 2048 points, respected the
class-color contract, exposed finite bbox/centroid/PCA fields and exported
successfully. Qualified Phase 10C.1 scroll and resize stress evidence was
reused for the data-independent layout boundary.

## 19. Known limitations and Phase 8D regression

The sample manifest and all production asset hashes match Phase 8D, but the
task baseline does not:

| Global metric | Phase 11A minus Phase 8D |
|---|---:|
| Accuracy | +0.000569661 |
| mIoU | +0.001548858 |
| Weld precision | -0.004512772 |
| Weld recall | +0.006811945 |
| Weld F1 | +0.001324907 |

The first confirmed cause is preprocessing order:

- Phase 8D froze a NumPy-generated point permutation;
- the production C++ SDK uses `std::mt19937` plus `std::shuffle`;
- every test file contains exactly 2048 rows, so the selected point set is the
  same but its order differs;
- the deployed network is measurably order-sensitive;
- prediction mismatches mapped back to original rows are nonzero for every
  sample.

This is not an Engine, Plugin, ONNX or checkpoint regression. It is a mismatch
between the historical approved preprocessing/input ordering and the current
production SDK contract. Phase 11A was not authorized to change either
contract, so the mismatch remains an explicit blocker. A future phase must
choose and freeze one authoritative sampling/order contract, then requalify
without silently replacing the Phase 8D evidence.

## 20. Final qualification decision

Operational qualification passed: build, 18/18 inference, finite outputs,
zero ErrorRecorder errors, same-process repetition, 5/5 cold starts, 360/360
soak, package/source equivalence, all exports, all fail-closed probes and GUI
subset smoke.

Production qualification nevertheless requires task metrics to match the
approved Phase 8D baseline. They do not, because the frozen Phase 8D and
production C++ preprocessing order contracts differ. Therefore Phase 11A
cannot promote or reaffirm task equivalence.

```text
PHASE_11A_RELEASE_FULL_TESTSET_QUALIFICATION_BLOCKED
TENSORRT_RELEASE_FULL_TESTSET_TASK_EQUIVALENCE_FAILED
CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED
```

No threshold was weakened, no difficult sample was removed, and no strict
numerical-equivalence claim is made.
