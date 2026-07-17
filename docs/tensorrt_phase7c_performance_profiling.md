# TensorRT Phase 7C：Layer-level Performance Profiling

更新时间：2026-07-17（Asia/Shanghai）

## 1. 结论

正式 Strict FP32 Engine 已通过 TensorRT `IProfiler` 完成 layer-level profiling：

```text
TENSORRT_PERFORMANCE_PROFILING_COMPLETED
```

核心结论：当前 Engine 的主要耗时不是 GEMM、Scatter/Gather 或普通 shape-copy，
而是第一级 `/model/tdb_1/Unique` 的 VoxelUnique IPluginV3。4 个 VoxelUnique 实例
合计占 profile layer time 的 `80.0083%`，其中 tdb_1 单层占 `76.3516%`。

按本阶段 A/B/C/D 分类，主要瓶颈为：

```text
C. dynamic shape overhead
```

更精确的工程表述是：**runtime-size VoxelUnique Plugin execution dominates**。
DynamicShape 标记总占比 `83.7667%`，但扣除 VoxelUnique 后其余动态 shape/copy 路径
只占 `3.7584%`；因此不能把问题泛化为所有 TensorRT dynamic shape 层都很慢。

## 2. 脚本与产物

新增脚本：

`scripts/profile_gcn_res_tensorrt_engine.py`

正式成功产物：

`artifacts/gcn_res_tensorrt/20260717_131821_006323_phase7c_profiling/`

包含：

- `layer_profile.json`
- `top50_layers.csv`
- `plugin_profile.json`
- `gemm_profile.json`
- `profiling_summary.md`

执行命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe `
  E:\GRP-PTv2\scripts\profile_gcn_res_tensorrt_engine.py
```

第一次采集在完成 profiler 回调后，由于诊断脚本将名称中包含
`VoxelUnique_voxel_count...DeviceToShapeHostCopy` 的4个 size-tensor copy层误分类为
Plugin，从而触发“必须恰好4个Plugin实例”的保护检查并失败。该诊断产物位于：

`artifacts/gcn_res_tensorrt/20260717_131746_211324_phase7c_profiling/`

该问题不涉及 Engine 或 profiler 本身。仅将分类条件收紧为
`LayerType=PluginV3 && PluginType=VoxelUnique` 后，使用新 run ID 重新完成正式采集。

## 3. 测试条件与计时口径

- Engine：`strict_fp32.plan`
- Engine SHA-256：`b76563580089bbc0684e7a8e1edda6028c2d89fd91397c22b4efbd89ecd7fc2c`
- 样本：`weld_65`
- points：FP32 `[1,2048,4]`
- adj：FP32 `[1,2048,2048]`
- 固定输入哈希：与 Phase 7A/7B 完全一致
- warmup：`100`，此时未挂载 profiler
- profile iterations：`100`
- profiler：TensorRT `IProfiler`
- `context.enqueue_emits_profile = True`
- 输入在 warmup 前复制到 device，此后保持常驻
- profiled iterations 不含 H2D/D2H
- profile 结束后只复制一次 logits 检查有限性

IProfiler 共记录 `57,000` 次 callback，即 `570` 个 Engine layer × `100` 次；全部
570个profile layer均与正式 `engine_inspector.json` 精确匹配，无未匹配层。

## 4. TensorRT 总耗时

IProfiler 汇总的每次平均 layer time：

```text
39.582810 ms
```

Phase 7B CUDA Event pure enqueue mean 为 `37.064278 ms`。Profiler 值高约 `6.80%`，
这是挂载逐层回调后的 profiling instrumentation 口径，不能用它替换 Phase 7B 的生产
延迟基线；它用于层间耗时占比和瓶颈定位。

## 5. Top latency layers

| Rank | Layer | Type | Avg ms/inference | Share |
|---:|---|---|---:|---:|
| 1 | `/model/tdb_1/Unique` | PluginV3 | `30.222106` | `76.351594%` |
| 2 | `/model/tdb_2/Unique` | PluginV3 | `1.389235` | `3.509693%` |
| 3 | `[trainStation9]` | TrainStation | `0.456279` | `1.152720%` |
| 4 | `[trainStation2]` | TrainStation | `0.211753` | `0.534963%` |
| 5 | `__myl_SqrtReshBott_myl45_10` | kgen | `0.209499` | `0.529268%` |
| 6 | `/model/ptb_0/linear_1/MatMul_myl45_13` | gemm | `0.195754` | `0.494544%` |
| 7 | `tdb_1 voxel_count DeviceToShapeHostCopy` | DeviceToShapeHost | `0.151147` | `0.381849%` |
| 8 | `ptb_0 GVA v/k fused MatMul` | gemm | `0.145142` | `0.366680%` |
| 9 | `[trainStation4]` | TrainStation | `0.134774` | `0.340486%` |
| 10 | `[trainStation3]` | TrainStation | `0.130723` | `0.330252%` |

完整 Top 50 见 `top50_layers.csv`。

## 6. VoxelUnique Plugin 专项

| Instance | Avg ms/inference | Share |
|---|---:|---:|
| `/model/tdb_1/Unique` | `30.222106` | `76.351594%` |
| `/model/tdb_2/Unique` | `1.389235` | `3.509693%` |
| `/model/tdb_3/Unique` | `0.051625` | `0.130422%` |
| `/model/tdb_4/Unique` | `0.006565` | `0.016585%` |
| **合计** | **`31.669531`** | **`80.008294%`** |

tdb_1 和 tdb_2 两个实例合计占 `79.861287%`。随 voxel hierarchy 点数下降，后两级
Unique 耗时快速下降；证据将瓶颈定位到前级大 N runtime unique，而不是“4个Plugin
实例数量”本身。

## 7. GEMM、Scatter/Gather 与动态 shape

| Category | Avg ms/inference | Share |
|---|---:|---:|
| GEMM | `2.151586` | `5.435657%` |
| Scatter | `0.354988` | `0.896823%` |
| Gather | `0.411112` | `1.038613%` |
| Scatter + Gather | `0.766100` | `1.935437%` |
| Reduce | `0.503573` | `1.272201%` |
| Shuffle/Reshape | `0.569834` | `1.439599%` |
| DynamicShape including VoxelUnique | `33.157198` | `83.766661%` |
| DynamicShape excluding VoxelUnique | `1.487668` | `3.758368%` |

Inspector 中共有 `86` 个 GEMM，profiling 同样关联到 `86` 个 GEMM。全部 GEMM 的
input/constants/output datatype 为 `Float`，tactic 使用
`f32f32_f32f32_f32` 形式；Inspector 全图没有 `Half` datatype，也没有 `tf32` tactic
token：

- TF32 GEMM count：`0`
- FP16 layer/datatype count：`0`
- FP16/INT8 Builder flag：disabled

因此当前耗时结果确实来自 Strict FP32 Engine，不是隐式 TF32/FP16。

## 8. 瓶颈分类

### A. Compute bound

不是当前主要分类。GEMM只占 `5.44%`；IProfiler 本身也不提供 FLOP utilization，
不能证明 VoxelUnique 内核属于算术 compute-bound。

### B. Memory bound

不是由 Scatter/Gather 证据支持的主要分类，两者合计仅 `1.94%`。VoxelUnique 内部是否
memory-bound 需要 Nsight Compute bandwidth/occupancy 指标，本阶段不能仅凭 layer time
下结论。

### C. Dynamic shape overhead

是本阶段主要分类。更具体地说，是 runtime-size VoxelUnique Plugin，尤其 tdb_1，
而不是一般的 Shape/Expand/DeviceToShapeHost 层。

### D. Kernel launch overhead

存在明显碎片化：`530/570` 个 layer 的平均耗时小于 `0.05 ms`。但这些小层合计未超过
VoxelUnique 的80%主导占比，因此 kernel launch/fragmentation 是次要因素而非首要瓶颈。

## 9. 完整性与停止线

Engine、ONNX、Plugin DLL、Plugin源文件、checkpoint和Engine Inspector JSON的执行
前后SHA-256一致。TensorRT ErrorRecorder为0，logits shape为 `[1,2048,2]` 且全部有限。

本阶段没有修改或重建 Engine，没有修改 ONNX、Plugin、checkpoint、Builder config，
没有执行 FP16、INT8 或 benchmark optimization。当前只完成瓶颈归因并停止。

```text
TENSORRT_PERFORMANCE_PROFILING_COMPLETED
```
