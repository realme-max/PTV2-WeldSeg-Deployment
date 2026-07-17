# TensorRT Phase 8C：VoxelUniqueCub 候选 Engine 集成与回归

## 1. 结论

本阶段已完成实验版 `VoxelUniqueCub` 在完整 GCN_res Strict FP32 Engine 中的集成、正确性回归和性能回归。

```text
VOXELUNIQUE_CUB_CANDIDATE_ENGINE_REGRESSION_COMPLETED
TENSORRT_CUB_PLUGIN_OPTIMIZATION_CONFIRMED
TENSORRT_CUB_END_TO_END_ACCELERATION_CONFIRMED
```

候选 Engine 的 Parser、Builder、Runtime 均通过；4 个真实样本、4 个 TDB stage 的 Plugin 中间结果 16/16 逐元素完全一致；固定 test split 的 18/18 个样本正常运行，候选 TensorRT 与基线 TensorRT 的标签和任务指标完全一致。

原有 `per-sample max_abs < 1e-4` 标准没有放宽。候选与 PyTorch 的 18 个样本中 13 个满足该严格阈值，最差 `weld_14` 为 `1.206398010254e-4`，因此必须保留：

```text
CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED
```

这不影响 runtime 成功和任务级等价的独立结论。

## 2. 受保护对象与候选对象

正式对象均未覆盖或原地修改：

| 对象 | 路径 | SHA-256 |
|---|---|---|
| 正式 TensorRT-ready ONNX | `artifacts/gcn_res_tensorrt/20260716_190125_699274_dds_reshape_rewrite/dds_reshape_rewritten.onnx` | `f71a585ded3f348a193e37395a124ab8a425ff72fc09ae92114d41872c460b98` |
| 基线 Strict FP32 Engine | `artifacts/gcn_res_tensorrt/20260716_224643_531592_strict_fp32/strict_fp32.plan` | `b76563580089bbc0684e7a8e1edda6028c2d89fd91397c22b4efbd89ecd7fc2c` |
| 基线 Plugin DLL | `artifacts/tensorrt_plugin_library/build_cuda128/Release/ptv2_voxel_unique_plugin.dll` | `60c48c400cc926f6aa183de62298a203916b1add6393a3483a332624da39d0ab` |
| checkpoint | `models/testParameters/GCN_res/best_model.pth` | `311bddf3607d76e6b7ded450b8419bf6ae98f34f50578608b3e6a1c1c3e58d21` |

候选产物目录：

`artifacts/gcn_res_tensorrt/20260717_155708_684630_phase8c_candidate_engine/`

| 候选对象 | SHA-256 / 大小 |
|---|---|
| `gcn_res_voxelunique_cub_candidate.onnx` | `16ca5c16c330e6572b1730e80da724231a28b68872a3203c21240348d4d89299` |
| `strict_fp32_voxelunique_cub_candidate.plan` | `a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299` / 30,277,300 bytes |
| `VoxelUniqueCubPlugin.dll` | `6641ec147e8eac10206a5c60ba1c1390c398d4b59e32a6a618a37046360ec348` |

Phase 8B 的 `278.38x` 是隔离 Plugin 测试结果；Phase 8C 下文的数值才是完整 Engine 结果，二者没有混用。

## 3. 派生 ONNX 审计

正式 ONNX 中 `/model/tdb_1/Unique` 至 `/model/tdb_4/Unique` 四个节点映射为 `com.tensorrt.ptv2.experimental::VoxelUniqueCub`。节点输入、输出、顺序、DDS size-tensor 连接、dtype 和计算属性保持不变。为使 TensorRT 按实验 Creator 身份查找 Plugin，还必须同步解析身份属性 `plugin_namespace`，并加入 custom-domain opset import；这是 Plugin 身份元数据，不改变数学计算。

审计结果：

- 节点数：`2663 → 2663`
- initializer 数：`381 → 381`
- graph I/O：不变
- 非 Plugin 节点结构哈希：不变
- initializer 内容哈希：不变
- 归一化后的拓扑与 tensor contract：不变
- `onnx.checker.check_model()`：PASS

证据见 `derived_onnx_audit.json`。

## 4. Parser、Builder 与 Engine 结构

- TensorRT：`11.1.0.106`
- GPU：NVIDIA GeForce RTX 5060，SM `12.0`
- Parser errors：`0`
- `VoxelUniqueCub` Creator 调用：`4`
- 基线自定义 Creator：候选 Parser 独立进程中未加载
- workspace：`4 GiB`
- TF32 / FP16 / INT8：全部关闭
- Engine deserialize：PASS
- Inspector layer count：`574`
- `PluginV3 / VoxelUniqueCub`：恰好 `4` 个
- TF32 tactic：`0`

I/O 契约：

| Tensor | Mode | dtype | shape |
|---|---|---|---|
| `points` | INPUT | FP32 | `[1,2048,4]` |
| `adj` | INPUT | FP32 | `[1,2048,2048]` |
| `logits` | OUTPUT | FP32 | `[1,2048,2]` |

## 5. Runtime smoke 与 Plugin 中间回归

固定 `weld_65` 输入 hash 与 Phase 6/7 一致：

- points：`b9f7ace14e74b05b076fa5d0f5e1226a0c1e84530336d785e8c3391f90a57063`
- adj：`a543b9f287b8bbca844bc36b5af72bec70e964095a35e7c68d8279a40e31cf12`

Engine deserialize、context creation、`enqueueV3` 全部 PASS；输出为 FP32 `[1,2048,2]`，全部有限，ErrorRecorder errors 为 `0`，runtime 实例为 4 个 `VoxelUniqueCub`。

Plugin 中间回归使用 `weld_65`、`weld_5`、`weld_12`、`weld_14`，逐级比较 `tdb_1`～`tdb_4` 的 `voxel_count`、`unique_values`、`inverse_indices` 和 runtime output shape。16/16 组比较全部逐元素完全一致。正式候选 Engine 没有为了暴露中间输出而修改；比较使用相同真实 stage key 的基线/候选隔离诊断 Engine。

## 6. 18 样本回归

固定条件：test split 18 个样本、seed 42、N=2048、k=6、同一 checkpoint、同一 points/adj 预处理、PyTorch Strict FP32。

```text
CANDIDATE_RUNTIME_VALIDATION_PASSED
CANDIDATE_TASK_LEVEL_EQUIVALENCE_CONFIRMED
CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED
```

- runtime 成功：`18/18`
- 候选与基线 TensorRT 标签一致率：每个样本 `100%`
- 候选与 PyTorch 标签一致率：每个样本 `100%`
- 候选与基线任务指标 delta：`0`
- 候选与基线最大 logits 误差最差值：`7.62939453125e-6`
- 候选与 PyTorch 严格阈值通过：`13/18`
- 候选与 PyTorch 最差样本：`weld_14`，max abs `1.206398010254e-4`

| Accuracy | mIoU | Weld precision | Weld recall | Weld F1 |
|---:|---:|---:|---:|---:|
| 0.978651259 | 0.936308798 | 0.964600551 | 0.929642905 | 0.946799162 |

标签语义保持为 class 0 = weld_seam，class 1 = background。

## 7. 完整模型延迟

固定 `weld_65`，warmup 100，measurement 1000。三个 runtime 分别在独立进程中执行；TensorRT pure 使用 CUDA Event 且数据常驻 device；E2E 包含 H2D、`enqueueV3`、D2H 和同步。

| Runtime | Scope | Mean ms | Speedup vs baseline TRT | Speedup vs PyTorch |
|---|---|---:|---:|---:|
| PyTorch CUDA Strict FP32 | forward | 20.412148 | — | 1.000× |
| Baseline TensorRT | pure | 37.199287 | 1.000× | 0.5487× |
| Baseline TensorRT | E2E | 40.029700 | 1.000× | 0.5099× |
| Candidate TensorRT CUB | pure | 4.701311 | **7.9125×** | **4.3418×** |
| Candidate TensorRT CUB | E2E | 7.599190 | **5.2676×** | **2.6861×** |

因此候选不仅显著快于基线 TensorRT，也在本次固定测试条件下快于 PyTorch。隔离 Plugin 的 Phase 8B speedup 没有被当作完整模型 speedup。

## 8. Layer profiling 与新瓶颈

固定 `weld_65`，warmup 100，IProfiler 100 次：

| 指标 | Baseline | Candidate | 改善 |
|---|---:|---:|---:|
| Profiled full-engine layer time | 39.582810 ms | 7.529873 ms | 5.2568× |
| 4 个 Unique Plugin 总耗时 | 31.669531 ms | 0.137512 ms | 230.3032× |
| Plugin 占比 | 80.0083% | 1.8262% | -78.1821 pp |
| `tdb_1` Plugin | 30.222106 ms | 0.045858 ms | 659.0425× |

候选侧：GEMM `2.118945 ms / 28.1405%`，Scatter `0.335248 ms / 4.4522%`，Gather `0.405046 ms / 5.3792%`，DynamicShape 含 Plugin `1.400148 ms / 18.5946%`，扣除 Plugin 后 `1.262635 ms / 16.7683%`。

旧的 VoxelUnique 瓶颈已经消除。574 层中 540 层小于 `0.05 ms`，Top 单层 `[trainStation9]` 仅 `0.388343 ms / 5.1574%`；新的主特征是大量小 kernel 的碎片化执行/launch 开销。

IProfiler 会引入回调开销，因此 7.529873 ms 不应替代 Phase 7B 方法得到的 4.701311 ms pure latency；profiling 数字用于层级归因。

## 9. 显存

- 候选 Engine 文件：30,277,300 bytes
- 候选隔离进程 `cudaMemGetInfo` 最大生命周期增量：383,778,816 bytes
- 基线同口径增量：383,778,816 bytes
- Plugin workspace：49,920 bytes/实例（N=2048）
- 4 实例简单上界和：199,680 bytes

`cudaMemGetInfo` 是生命周期快照，不是 kernel 内部瞬时显存峰值；TensorRT 可复用 workspace，不能从快照中单独反推出四个 Plugin 同时占用 199,680 bytes。

## 10. 新增脚本与停止边界

新增：

- `scripts/gcn_res_tensorrt_cub_common.py`
- `scripts/prepare_gcn_res_voxelunique_cub_candidate.py`
- `scripts/build_gcn_res_tensorrt_cub_candidate.py`
- `scripts/validate_gcn_res_tensorrt_cub_candidate.py`
- `scripts/benchmark_gcn_res_tensorrt_cub_candidate.py`
- `scripts/profile_gcn_res_tensorrt_cub_candidate.py`
- `scripts/finalize_gcn_res_tensorrt_phase8c.py`

完整结构化结果位于 Phase 8C 产物目录。候选仍是实验产物，本阶段没有自动替换正式 Strict FP32 Engine；没有执行 FP16、INT8、C++ 应用集成，也没有修改 checkpoint、模型结构、采样或邻接逻辑。
