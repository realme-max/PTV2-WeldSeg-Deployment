# TensorRT Phase 8A：VoxelUnique Baseline and Optimization Assessment

更新时间：2026-07-17（Asia/Shanghai）

## 1. 状态

本轮遵循“不要立即重写，先进行baseline分析”的约束，完成A1隔离profiling、A2源码
审计和A3优化方案评估；没有进入CUDA改写、重编译或TensorRT回归。因此状态为：

```text
VOXELUNIQUE_ANALYSIS_COMPLETED
```

不是 `VOXELUNIQUE_OPTIMIZATION_COMPLETED`。

## 2. 新增文件

- `scripts/profile_voxelunique_kernel.py`
- `scripts/benchmark_voxelunique_plugin.py`
- `docs/voxelunique_kernel_analysis.md`
- `docs/tensorrt_phase8a_voxelunique_optimization.md`

产物：

`artifacts/gcn_res_tensorrt/20260717_133149_811270_phase8a_voxelunique/`

包含：

- `baseline_profile.json`
- `voxelunique_kernel_baseline.json`
- `optimized_profile.json`（`NOT_RUN`，没有伪造优化结果）
- `correctness_report.json`
- `latency_comparison.json`
- `benchmark_inputs.npz`
- `environment.json`
- `phase8a_summary.json`

## 3. Baseline摘要

| Input | N | M | Kernel mean | Copy mean | Total mean |
|---|---:|---:|---:|---:|---:|
| random keys `[0,512)` | 2048 | 499 | `37.258966 ms` | `0.190753 ms` | `37.449719 ms` |
| weld_65 tdb_1 keys | 2048 | 397 | `28.847813 ms` | `0.189470 ms` | `29.037282 ms` |

真实key隔离kernel与Phase 7C完整Engine tdb_1的 `30.222106 ms`接近，证明Plugin内部
串行CUDA算法是主因。

## 4. 当前实现结论

- 1个block、1个thread；
- 1个kernel launch；
- serial deduplication；
- insertion sort；
- serial inverse lookup；
- 复杂度 `O(N·M + M²)`；
- 无atomic；
- 无CPU fallback；
- 无Plugin内部global synchronize；
- 内存拷贝仅约0.19ms，不是主要瓶颈。

## 5. 优化评估

第一推荐为CUB radix sort pairs + boundary detection + prefix scan + inverse scatter。
GPU hash因sorted=True合同仍需额外排序，不作为第一选择；block-local fusion作为CUB基线
未达5ms后的第二步。

当前真实kernel达到 `<5 ms` 需要至少 `5.77x` speedup/`82.67%`降幅。算法上具备
可行性，但本轮未实现，不能声称目标已经达到。

## 6. 正确性和安全边界

当前baseline Plugin正确性证据覆盖随机N=4/8/32/2048、all same、all unique、sorted、
reversed、repeated groups和INT64 extremes，18/18通过。本轮两个profile输入也通过
count/values/inverse/runtime-shape比较。

执行前后正式 Engine、正式 ONNX、Plugin DLL、Plugin源码、checkpoint和单Plugin
correctness Engine哈希均未变化。没有运行正式Engine，没有Parser/build/parity/latency
回归，没有FP16、INT8或C++部署。

详细算法和下一阶段建议见：`docs/voxelunique_kernel_analysis.md`。

```text
VOXELUNIQUE_ANALYSIS_COMPLETED
```
