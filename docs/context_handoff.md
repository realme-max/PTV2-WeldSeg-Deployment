# PTV2-WeldSeg-Deployment 项目交接

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

## 9. 当前停止线与下一步

Phase 5E 已完成 Strict FP32 单样本推理和 PyTorch parity。本轮按要求停止，不进入：

- FP16 / INT8；
- benchmark 或 kernel 优化；
- C++ 部署；
- 修改 ONNX、Plugin、checkpoint、模型、graph rewrite 或验收阈值。

需要注意：当前 max absolute error 虽通过 `1e-4`，但距离阈值较近。若后续获得授权，
应先在既有固定样本集合上复核 Strict FP32 parity，再决定是否将该 Engine 作为正式部署
基线；该复核不属于本轮执行范围。
