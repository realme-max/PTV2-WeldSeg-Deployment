# PTV2-WeldSeg-Deployment 项目交接

## Phase 8B：VoxelUnique CUB 隔离优化（2026-07-17）

新增独立实验 Plugin `com.tensorrt.ptv2.experimental::VoxelUniqueCub` version 1，位于
`deployment/tensorrt_voxel_unique_plugin_cub/`。实现使用 CUB signed-INT64 radix sort pairs、run
boundary、inclusive scan 和 inverse scatter；最大 TensorRT workspace 为 `49,920 bytes`，`enqueue()`
内没有动态 CUDA 分配、host round-trip 或同步。

独立 correctness Engine 的 21 个 case 全部通过；count、values、inverse 和 DDS runtime shape 同时与
C++ CPU reference 及 `torch.unique(sorted=True, return_inverse=True)` 完全一致。

| Input | Baseline kernel mean | CUB kernel mean | Speedup |
|---|---:|---:|---:|
| random keys | `37.258966 ms` | `0.117754 ms` | `316.41x` |
| weld_65 tdb_1 keys | `28.847813 ms` | `0.103627 ms` | `278.38x` |

真实 key 已满足 `<5 ms` 与至少 `5x` 两个门槛。实验产物位于
`artifacts/gcn_res_tensorrt/20260717_151544_915303_phase8b_voxelunique_cub/`。正式 ONNX、Strict FP32
Engine、baseline Plugin DLL 和 checkpoint 的冻结 SHA-256 均未变化。

本阶段按要求停止，没有接入正式图、重建正式 Engine、执行 18 样本 parity、全模型 benchmark、
FP16/INT8 或 C++ 应用部署。下一步需单独授权 Phase 8C candidate 集成和完整回归。

详细报告：`docs/tensorrt_phase8b_voxelunique_cub.md`。

```text
VOXEL_UNIQUE_CUB_CORRECTNESS_PASSED
VOXEL_UNIQUE_CUB_BENCHMARK_PASSED
VOXELUNIQUE_CUB_ISOLATED_OPTIMIZATION_COMPLETED
```

更新时间：2026-07-16（Asia/Shanghai）

## 1. 最终模型与接口

- 原模型：`models/testParameters/GCN_res/model.py`
- checkpoint：`models/testParameters/GCN_res/best_model.pth`
- deployment 模型：`deployment/gcn_res_onnx_model.py`
- class 0：`weld_seam`
- class 1：`background`

固定接口：

```text
points: FP32 [1,2048,4]
adj:    FP32 [1,2048,2048]
logits: FP32 [1,2048,2]
```

固定评估基线：test mIoU `0.936309`，weld F1 `0.946799`。

## 2. 环境

- 虚拟环境：`E:\GRP-PTv2\.venv_ptv2`
- Python：3.11.8
- PyTorch：2.7.1+cu128
- cuDNN：9.7.1
- CUDA Toolkit：12.8 Update 1
- TensorRT SDK/Python：11.1.0.106
- onnxruntime-gpu：1.26.0
- cuda-python / cuda-bindings：12.8.0
- torch_geometric：2.8.0
- GPU：NVIDIA GeForce RTX 5060（SM 12.0）
- `pip check`：No broken requirements found

TensorRT SDK：

`D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106`

## 3. TensorRT 派生图与 Engine

当前 TensorRT 派生 ONNX：

`artifacts/gcn_res_tensorrt/20260716_190125_699274_dds_reshape_rewrite/dds_reshape_rewritten.onnx`

SHA-256：`f71a585ded3f348a193e37395a124ab8a425ff72fc09ae92114d41872c460b98`

当前 FP32 Engine：

`artifacts/gcn_res_tensorrt/20260716_190125_699274_dds_reshape_rewrite/gcn_res_dds_reshape_fp32_b1_n2048.plan`

SHA-256：`7a856d1aa50628360d4acd5ee384fcd5042a8087a3112a361ee3c47db9e7326b`

已完成：

- `Unique → com.tensorrt.ptv2::VoxelUnique`；
- 16 个 constant-false If folding；
- 8 个已审计 DDS Reshape 等价改写为 `Unsqueeze(axis=0)`；
- ONNX checker、shape inference、TensorRT Parser；
- 4 GiB workspace 的 FP32 Engine build；
- Engine 保存、反序列化和 I/O 验证；
- Build/Runtime/Inspector 确认 4 个 VoxelUnique 实例。

```text
TENSORRT_DDS_RESHAPE_REWRITE_ENGINE_BUILD_PASSED
TENSORRT_FP32_ENGINE_BUILD_PASSED
```

## 4. Phase 5 FP32 Runtime 与最终输出 parity

固定样本：`val_00_weld_7`。

输入：

`artifacts/gcn_res_onnx/20260715_onnx_after_cdist_fp32_opset18/export_input.npz`

TensorRT Runtime 已成功反序列化 Engine、创建 execution context、分配 CUDA buffer，
并通过一次 `execute_async_v3/enqueueV3` 完成推理；ErrorRecorder 为 0 errors，输出均有限。

| 指标 | TensorRT FP32 vs PyTorch CUDA |
|---|---:|
| max absolute error | `1.444053650e-02` |
| mean absolute error | `1.960213384e-03` |
| RMSE | `2.894863672e-03` |
| cosine similarity | `0.999999752275` |
| label agreement | `2048/2048 = 100%` |

单样本分类结果完全一致：mIoU `0.903673335`，weld seam F1 `0.909433962`。
但 `max_abs_error` 未满足建议的 `<1e-4` 条件，因此状态保持：

```text
TENSORRT_FP32_NUMERICAL_PARITY_FAILED
```

Phase 5 产物：

`artifacts/gcn_res_tensorrt/20260716_212305_673121_fp32_inference/`

报告：`docs/tensorrt_phase5_fp32_parity.md`。

## 5. Phase 5B 中间张量定位

只读诊断脚本：

`scripts/locate_gcn_res_tensorrt_first_divergence.py`

最终诊断产物：

`artifacts/gcn_res_tensorrt/20260716_215859_539272_intermediate_parity/`

正式 Engine 仅暴露 `points/adj/logits`，没有 debug tensor。诊断采用同一未修改 ONNX
临时构建只驻留内存的 FP32 Engine，把选定 ITensor 暴露为额外输出；未保存该 Engine。
正式 Engine、ONNX、Plugin DLL 和 checkpoint 在执行前后 SHA-256 均未变化。

定位结果：

- `linear_1` 输出 bitwise 一致，max abs `0`；
- 首个捕获到的数值分叉为 `ptb_0` 输出：max abs `1.035153866e-03`，
  mean abs `1.334132176e-04`，cosine `0.999999917845`；
- `VoxelUnique` 的 `voxel_count=518`、518 个 `unique_values`、2048 个
  `inverse_indices` 全部 bitwise 一致；
- Scatter 的 `unique_batch_ids`、`voxel_point_counts`、
  `voxel_count_per_batch` 全部 bitwise 一致；
- 坐标 Scatter add 的 max abs 为 `9.536743164e-07`，仍在容差内；
- max Scatter 的输入特征在进入 Scatter 前已存在 max abs `3.173947334e-03`
  的上游差异，因此 pooled-feature 差异不是 VoxelUnique 或 Scatter 首次引入。

结论：当前首个已定位边界在 `ptb_0`，早于 `tdb_1 → VoxelUnique → Scatter → pooling`。
本轮没有尝试修复。

```text
FIRST_TENSORRT_PYTORCH_DIVERGENCE_FOUND
```

## 6. Phase 5C：ptb_0 内部分叉

诊断脚本：

`scripts/locate_ptb0_tensorrt_first_divergence.py`

最终产物：

`artifacts/gcn_res_tensorrt/20260716_221423_053311_ptb0_parity/`

共比较 24 个有序边界，覆盖 distance、TopK、Gather、relative encoding、Q/K/V、
attention logits、Softmax、aggregation、BN/ReLU、Linear₂ 和 residual Add。

- `stem_linear_features` bitwise 一致；
- 首个非 bitwise 算术差异：`/model/ptb_0/ReduceSum` 的 `distance_squared`，
  max abs `4.768371582e-07`，仍满足 `rtol=1e-5, atol=1e-6`；
- `TopK` 有 `11534/32768` 个 index ID 不同，但 2048 点中只有 1291 个唯一 XYZ，
  所有错位 index 选到的 XYZ 均逐元素一致，`neighbors_xyz` max abs 为 `0`；
- 因此 TopK 差异是重复坐标并列邻居的排序差异，不改变邻域几何；
- 首个超浮点容差的特征边界：`ptb0_linear1_features`，
  max abs `3.492087126e-04`，而其输入 `stem_linear_features` bitwise 一致；
- 后续 relative encoding、QKV、attention 和 residual 继续传播并放大该差异；
- 正式 Engine、ONNX、Plugin 和 checkpoint 前后哈希一致，未尝试修复。

```text
FIRST_PTB0_INTERNAL_DIVERGENCE_FOUND
```

## 7. Phase 5D：ptb_0.linear_1 差异归因

分析脚本：

`scripts/analyze_ptb0_linear1_precision.py`

产物：

`artifacts/gcn_res_tensorrt/20260716_222700_127628_linear1_analysis/`

只读提取并证明：

- `X` 为 `[1,2048,48]` FP32，PyTorch/TensorRT dump bitwise 一致；
- checkpoint `W` 为 `[48,48]`，ONNX MatMul initializer 与 `W.T` bitwise 一致；
- ONNX bias 与 checkpoint bias bitwise 一致；
- NumPy FP64 参考下，PyTorch max abs 为 `3.368786462e-07`；
- TensorRT max abs 为 `3.492322137e-04`；
- PyTorch Phase 5C 输出与显式关闭 TF32 的 CUDA `F.linear` bitwise 一致；
- TensorRT Phase 5C 输出与显式开启 TF32 的 CUDA `F.linear` bitwise 一致。

正式 Engine Inspector 显示：

```text
layer_type = gemm
tactic = sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize128x64x16_stage6_warpsize2x2x1_tensor16x8x8
```

该 layer 将 ONNX `MatMul + Add(bias)` 融合；I/O 和常量均为 Float/FP32，
但 tactic 的 `f32f32_tf32f32_f32` 明确使用 TF32 tensor-core 乘法与 FP32
累加/输出。Inspector 未发现独立 weight-reformat layer，权重常量以 row-major Float
呈现；内部 tactic packing 不单独可见。

结论：`ptb_0.linear_1` 差异已直接归因于 TensorRT 默认 TF32 GEMM tactic，
而 PyTorch 验证路径使用 full-FP32 CUDA matmul。不是 X/W/b、转置、VoxelUnique、
Scatter 或 TopK 几何错误。本轮未修改 Builder flag 或重新构建 Engine。

```text
LINEAR1_TENSORRT_DIFF_ATTRIBUTED
```

## 8. Phase 5E：Strict FP32 Engine 与 parity

执行脚本：

`scripts/build_validate_gcn_res_tensorrt_strict_fp32.py`

最终产物：

`artifacts/gcn_res_tensorrt/20260716_224643_531592_strict_fp32/`

本轮以同一 `dds_reshape_rewritten.onnx`、VoxelUnique Plugin、checkpoint 和固定输入为
真源，仅在新的 Builder config 中显式执行 `config.clear_flag(trt.BuilderFlag.TF32)`；
FP16、INT8、benchmark 均保持关闭。源 ONNX、Plugin DLL 和 checkpoint 的执行前后
SHA-256 完全一致。

Strict FP32 Engine：

`strict_fp32.plan`

SHA-256：`b76563580089bbc0684e7a8e1edda6028c2d89fd91397c22b4efbd89ecd7fc2c`

构建和 Inspector 结果：

- TensorRT Parser：PASS，error count `0`；
- Engine build、保存和反序列化：PASS；
- VoxelUnique：build/runtime/Inspector 均为 `4` 个实例；
- Inspector 共发现 `86` 个 GEMM layer，其中含 `tf32` 的 tactic 为 `0`；
- `ptb_0.linear_1` 的新 tactic 为：

```text
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x64x8_stage3_warpsize1x2x1_ffma_aligna4_alignc4
```

PyTorch 参考侧同样显式关闭 CUDA matmul/cuDNN TF32，并使用 `highest` float32 matmul
precision；参考推理后已恢复原进程设置。TensorRT 仅执行一次 `enqueueV3`，未做预热或
benchmark，ErrorRecorder 为 `0` errors。

| 指标 | TensorRT Strict FP32 vs PyTorch CUDA FP32 |
|---|---:|
| max absolute error | `9.775161743e-05` |
| mean absolute error | `2.132453346e-05` |
| RMSE | `2.841241238e-05` |
| cosine similarity | `0.999999999982` |
| label agreement | `2048/2048 = 100%` |

输出均为有限值，满足本轮 `max_abs_error < 1e-4`、cosine `> 0.9999`、label
agreement `>= 99.99%` 的验收条件。关闭 TF32 后，原 FP32 Engine 的
`1.444053650e-02` 最大误差下降到 `9.775161743e-05`。

```text
TENSORRT_STRICT_FP32_PARITY_PASSED
```

## 9. Phase 6：Strict FP32 全 test split 验证

执行脚本：

`scripts/validate_gcn_res_tensorrt_strict_fp32_multisample.py`

产物：

`artifacts/gcn_res_tensorrt/20260717_110500_836041_strict_fp32_multisample/`

固定 test split 的18个样本均完成 PyTorch CUDA 和 TensorRT Strict FP32 推理。每个样本
均使用 seed 42 的固定2048点采样、相同归一化结果和相同 `k=6` 邻接矩阵。两套 logits
已逐样本保存。

- TensorRT Runtime/ErrorRecorder：PASS，`0` errors；
- 所有输出有限；
- 18/18 样本 point-wise label agreement 均为 `100%`；
- cosine 条件 18/18 通过，最差为 `0.999999999979929`；
- `max_abs_error < 1e-4` 条件 13/18 通过；
- 超限样本：`weld_5`、`weld_12`、`weld_14`、`weld_4`、`weld_15`；
- 最差样本：`weld_14`，max abs `1.230239868164e-04`；
- 每样本 max abs 平均值：`8.540683322483e-05`；
- 聚合 PyTorch/TensorRT mIoU 均为 `0.936308797729`；
- 聚合 PyTorch/TensorRT weld seam F1 均为 `0.946799161766`；
- mIoU/F1 绝对差均为 `0`；
- Engine、ONNX、Plugin、checkpoint 的执行前后 SHA-256 一致。

由于5个样本超过预先规定的 logits 最大绝对误差条件，未调整容差，最终状态为：

```text
TENSORRT_STRICT_FP32_MULTISAMPLE_VALIDATION_FAILED
```

详细报告：`docs/tensorrt_phase6_multisample_validation.md`。

## 10. Phase 6B：Strict FP32 残余误差归因

执行脚本：

`scripts/attribute_gcn_res_tensorrt_strict_fp32_residual_error.py`

产物：

`artifacts/gcn_res_tensorrt/20260717_113127_107178_residual_error_attribution/`

分析样本为 `weld_14`、`weld_5`、`weld_12`。正式 Engine 未标记任何 debug tensor，
因此使用同一只读 ONNX、Plugin、4 GiB workspace 和相同 Strict-FP32 策略，在内存中
构建不落盘诊断 Engine，将19个主要 stage tensor临时暴露为输出。正式 Engine、ONNX、
Plugin、checkpoint 与正式 Builder 配置均未修改。

主要结论：

- 三个样本的 `stem_linear` 均 bitwise 一致；
- 三个样本的首个分叉和首个正向误差放大均为 `ptb_0`；
- `ptb_0` max abs 为 `7.7486e-06 ~ 9.3877e-06`；
- `gcn_0` 随后稳定将误差放大约 `2.7061e-05 ~ 2.9027e-05`；
- transition-down 不是首个误差来源，且多个down stage会降低max error；
- 最差样本 `weld_14` 的最大单级放大发生在 `ptb_8`，delta
  `3.564357758e-05`；
- 三样本全局最大单级放大发生在 `weld_5` 的 `segmentation_head_input`，delta
  `4.351139069e-05`；
- 诊断/正式 TensorRT logits max abs 仅为 `2.861e-06 ~ 5.245e-06`，标签一致率
  均为100%，说明临时输出导致的 tactic扰动远小于待归因残差；
- 正式 TensorRT/PyTorch logits max abs 为 `1.1158e-04 ~ 1.2207e-04`。

归因：残余 Strict-FP32 差异起源于 `ptb_0`，随后由 `gcn_0` 和decoder/head阶段累计、
放大；证据不支持 VoxelUnique 或 transition-down pooling 是首个来源。本轮没有进一步
展开 `ptb_0` 内部算子，也没有修复。

```text
RESIDUAL_ERROR_ATTRIBUTION_COMPLETED
```

## 11. Phase 7A：Engine Benchmark Preparation

新增脚本：

`scripts/smoke_test_gcn_res_tensorrt_engine.py`

成功产物：

`artifacts/gcn_res_tensorrt/20260717_115222_307208_phase7a_engine_prepare/`

正式 Strict FP32 Engine 已完成只读 metadata/Inspector 检查和一次 Runtime smoke：

- Engine SHA-256：`b76563580089bbc0684e7a8e1edda6028c2d89fd91397c22b4efbd89ecd7fc2c`；
- TensorRT `11.1.0.106`、CUDA Runtime `12.8`、RTX 5060 SM `12.0`；
- Inspector layer count `570`，optimization profile `1`；
- 恰好 4 个 `PluginV3 / VoxelUnique` 层，runtime plugin instance count 也为 `4`；
- I/O 为 points FP32 `[1,2048,4]`、adj FP32 `[1,2048,2048]`、logits FP32
  `[1,2048,2]`；
- 固定 test 样本 `weld_65` 的 points/adj/sample indices 哈希与 Phase 6 完全一致；
- Engine deserialize、context creation、单次 `enqueueV3` 均 PASS；
- logits 全部有限，ErrorRecorder errors `0`；
- Engine、ONNX、Plugin DLL、Plugin 源码与 checkpoint 在执行前后均未变化；
- `pip check` 无冲突。

本阶段没有 warmup、重复推理、计时、显存采样、parity、accuracy 或 Engine build。

```text
ENGINE_BENCHMARK_PREPARATION_COMPLETED
```

详细报告：`docs/tensorrt_phase7a_engine_preparation.md`。

## 12. Phase 7B：Latency Benchmark

新增脚本：

- `scripts/benchmark_gcn_res_pytorch_latency.py`
- `scripts/benchmark_gcn_res_tensorrt_latency.py`

产物：

`artifacts/gcn_res_tensorrt/20260717_122000_phase7b_latency_benchmark/`

固定 `weld_65`、points `[1,2048,4]`、adj `[1,2048,2048]`，输入哈希与 Phase 7A
完全一致。PyTorch 和 TensorRT 分别在独立进程中完成100次warmup和1000次正式测试。

| Runtime | Mean ms | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|
| PyTorch CUDA forward | `20.7122` | `20.4034` | `23.1942` | `24.9172` |
| TensorRT pure enqueue | `37.0643` | `37.0274` | `38.2124` | `38.8241` |
| TensorRT pageable-host E2E | `39.6064` | `39.5497` | `40.6932` | `41.4386` |

以 `PyTorch mean / TensorRT mean` 定义 speedup：pure 为 `0.558817x`，E2E 为
`0.522950x`。当前 Strict FP32 TensorRT pure latency 比 PyTorch 高 `78.95%`，没有
取得加速。

显存口径：PyTorch allocator benchmark peak allocated `170,183,680` bytes、peak
reserved `190,840,832` bytes；TensorRT 隔离进程 `cudaMemGetInfo` 生命周期快照相对
基线最大观测增量 `383,778,816` bytes。后者不是 kernel 内部瞬时 peak。

Engine、ONNX、Plugin、checkpoint 和 Plugin 源码执行前后均未变化；TensorRT
ErrorRecorder 为0，输出有限，未执行 accuracy regression。

```text
TENSORRT_LATENCY_BENCHMARK_COMPLETED
```

详细报告：`docs/tensorrt_phase7b_latency_benchmark.md`。

## 13. Phase 7C：TensorRT Layer Profiling

新增脚本：

`scripts/profile_gcn_res_tensorrt_engine.py`

正式产物：

`artifacts/gcn_res_tensorrt/20260717_131821_006323_phase7c_profiling/`

使用固定 `weld_65` 和 Phase 7A/7B 相同输入哈希，执行100次无profiler warmup，再挂载
TensorRT `IProfiler`执行100次profile。输入常驻device，profile不含H2D/D2H。

主要结果：

- IProfiler平均layer time总和：`39.582810 ms/inference`；
- 570/570 Engine layers精确关联Inspector，callback共57,000次；
- Top1 `/model/tdb_1/Unique`：`30.222106 ms`、`76.351594%`；
- Top2 `/model/tdb_2/Unique`：`1.389235 ms`、`3.509693%`；
- 4个VoxelUnique合计：`31.669531 ms`、`80.008294%`；
- GEMM：`2.151586 ms`、`5.435657%`；
- Scatter/Gather合计：`0.766100 ms`、`1.935437%`；
- DynamicShape含Plugin：`83.766661%`；扣除VoxelUnique后为`3.758368%`；
- 86个GEMM均为Float，TF32 GEMM为0，全Inspector没有Half datatype；
- 530/570层平均小于0.05 ms，存在次要的碎片化/launch开销，但不改变Plugin主导结论。

瓶颈分类为 `C. dynamic shape overhead`，更精确地说是前两级、尤其tdb_1的
runtime-size VoxelUnique Plugin execution。现有证据不支持GEMM compute、Scatter/Gather
memory access或一般shape-copy是主要瓶颈。

```text
TENSORRT_PERFORMANCE_PROFILING_COMPLETED
```

详细报告：`docs/tensorrt_phase7c_performance_profiling.md`。

## 14. Phase 8A：VoxelUnique Baseline Analysis

新增脚本：

- `scripts/profile_voxelunique_kernel.py`
- `scripts/benchmark_voxelunique_plugin.py`

产物：

`artifacts/gcn_res_tensorrt/20260717_133149_811270_phase8a_voxelunique/`

复用既有单VoxelUnique correctness Engine，不构建新Engine、不经过完整PTV2。随机N=2048
key（M=499）和真实`weld_65/tdb_1` key（M=397）均执行100次warmup和1000次CUDA
Event分段测量。

| Input | Plugin execution mean | Copy mean | Total mean |
|---|---:|---:|---:|
| random | `37.258966 ms` | `0.190753 ms` | `37.449719 ms` |
| weld_65 tdb_1 | `28.847813 ms` | `0.189470 ms` | `29.037282 ms` |

当前CUDA源码为一个block/一个thread，串行去重、插入排序和inverse lookup，总复杂度
`O(N·M + M²)`。每次enqueue只有一个kernel，无atomic、CPU fallback或Plugin内部global
synchronize。真实隔离kernel与Phase 7C完整Engine tdb_1的`30.222106 ms`接近，确认
Plugin串行算法是主因，约0.19ms的数据复制不是瓶颈。

当前正确性证据覆盖随机N=4/8/32/2048和全部要求边界，18/18通过。优先优化建议为
CUB radix sort pairs + boundary flags + scan + inverse scatter；GPU hash因sorted=True仍需
排序，不作为第一选择；block-local fusion作为后备。

本轮按“不要立即重写”约束只完成分析，`optimized_profile.json`为`NOT_RUN`，没有修改或
重编译Plugin，也没有执行正式TensorRT回归。

```text
VOXELUNIQUE_ANALYSIS_COMPLETED
```

详细报告：

- `docs/voxelunique_kernel_analysis.md`
- `docs/tensorrt_phase8a_voxelunique_optimization.md`

## 15. 当前停止线与下一步

Phase 8A baseline分析已完成。本轮停止，不进入：

- CUDA kernel改写或Plugin重编译；
- 正式Engine重建；
- Parser/runtime/parity/latency回归；
- FP16 / INT8；
- C++部署或其他模型优化。

若后续明确授权Phase 8B，应在新源码和新DLL路径实现CUB版本，保留当前baseline DLL，
先通过完整correctness矩阵，再做isolated benchmark；只有两者通过后才允许构建candidate
Strict FP32 Engine并执行18样本回归。

## 16. Phase 8C：VoxelUniqueCub 候选 Engine 集成与回归

正式对象保持不变，所有候选文件位于：

`artifacts/gcn_res_tensorrt/20260717_155708_684630_phase8c_candidate_engine/`

候选 Strict FP32 Engine SHA-256 为 `a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299`。TensorRT Parser、Builder、deserialize、`weld_65` runtime smoke 均 PASS；Inspector 为 574 层、4 个 `PluginV3 / VoxelUniqueCub`，TF32/FP16/INT8 均关闭。

正确性结论：

- `weld_65/5/12/14 × tdb_1~4` 的 count/values/inverse/runtime shape：16/16 完全一致；
- 固定 test split runtime：18/18；
- 候选与基线 TensorRT 标签及任务指标：完全一致；
- 候选与 PyTorch 标签一致率：18/18 均 100%；
- 原严格阈值未放宽：13/18 的 max abs `<1e-4`，最差 `weld_14 = 1.206398010254e-4`，因此保留 `CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED`。

完整模型 latency（`weld_65`，100 warmup，1000 measurement）：

- PyTorch CUDA：20.412148 ms；
- baseline TensorRT pure/E2E：37.199287 / 40.029700 ms；
- candidate TensorRT pure/E2E：4.701311 / 7.599190 ms；
- candidate 相对 baseline：7.9125× pure、5.2676× E2E；
- candidate 相对 PyTorch：4.3418× pure、2.6861× E2E。

IProfiler 显示 4 个 Plugin 总耗时从 31.669531 ms 降至 0.137512 ms，`tdb_1` 从 30.222106 ms 降至 0.045858 ms。Plugin 占比从 80.0083% 降至 1.8262%；新的主要特征是 540/574 个层低于 0.05 ms 的碎片化 kernel launch/执行开销。

```text
VOXELUNIQUE_CUB_CANDIDATE_ENGINE_REGRESSION_COMPLETED
TENSORRT_CUB_PLUGIN_OPTIMIZATION_CONFIRMED
TENSORRT_CUB_END_TO_END_ACCELERATION_CONFIRMED
```

## 17. Phase 8D: Production baseline qualification

Formal artifacts: `artifacts/gcn_res_tensorrt/20260717_173128_144483_phase8d_production_baseline/`.

```text
TENSORRT_CUB_PRODUCTION_QUALIFICATION_PASSED_WITH_NUMERICAL_EXCEPTION
TENSORRT_CUB_STRICT_FP32_TASK_EQUIVALENT_BASELINE_PROMOTED
CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED
```

Key evidence:

- Candidate Engine SHA-256: `a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299`.
- CUB Plugin SHA-256: `6641ec147e8eac10206a5c60ba1c1390c398d4b59e32a6a618a37046360ec348`.
- Cold start: 10/10 independent processes passed; four plugin instances and zero ErrorRecorder errors each.
- Full regression: 18/18 runtime and 100% candidate/PyTorch/baseline label agreement; all task metric deltas are zero.
- Strict numerical threshold remains 13/18; worst `weld_14`, max abs `1.2302398681640625e-4`.
- Three-round aggregate: candidate pure `4.8209 ms`, E2E `7.8194 ms`; `8.5348x` vs old TensorRT pure, `4.3514x` vs PyTorch pure, and `3.0514x` vs PyTorch host-to-host E2E.
- No candidate sample was slower than PyTorch pure.
- Determinism: `DETERMINISTIC_LABELS_ONLY`; labels stable, logits max repeated-run difference `7.8678131103515625e-6`.
- Soak: 5000/5000 passed, finite outputs/reference labels, zero ErrorRecorder errors, no monotonic memory growth or obvious latency degradation.
- Negative paths: 8/8 fail closed with no inference or fallback.
- Package checksums and final default-mode production inference passed.

The active pointer is `deployment/tensorrt/current_baseline.json`. The old baseline Engine/Plugin remain untouched and rollback is an explicit manifest switch only. No FP16, INT8, C++ integration, model/ONNX/plugin-algorithm change, robot integration, or safety-threshold change was performed.

Detailed references:

- `docs/tensorrt_phase8d_production_baseline.md`
- `docs/tensorrt_production_runbook.md`

详细报告：`docs/tensorrt_phase8c_candidate_engine_regression.md`。候选尚未替换正式 Engine；下一步需单独授权，且本阶段未执行 FP16、INT8 或 C++ 集成。
