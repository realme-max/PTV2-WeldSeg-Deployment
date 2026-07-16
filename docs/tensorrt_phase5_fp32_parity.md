# TensorRT Phase 5：FP32 Runtime 与 PyTorch 数值对齐

更新时间：2026-07-16（Asia/Shanghai）

## 1. 结论

TensorRT FP32 Runtime 已成功完成一次固定输入推理，但相对 PyTorch
deployment CUDA baseline 的 logits 最大绝对误差超过本阶段建议阈值：

```text
TensorRT Runtime inference: PASSED
TensorRT ErrorRecorder errors: 0
outputs finite: true
point-wise label agreement: 2048 / 2048 = 100%
max_absolute_error: 1.444053650e-02
cosine_similarity: 0.999999752275

TENSORRT_FP32_NUMERICAL_PARITY_FAILED
```

因此本轮不进入 FP16、INT8、benchmark 或 C++ 部署，也没有修改 ONNX、
Plugin、checkpoint 或容差。

## 2. 固定对象

- Engine：`artifacts/gcn_res_tensorrt/20260716_190125_699274_dds_reshape_rewrite/gcn_res_dds_reshape_fp32_b1_n2048.plan`
- Engine SHA-256：`7a856d1aa50628360d4acd5ee384fcd5042a8087a3112a361ee3c47db9e7326b`
- 派生 ONNX：`artifacts/gcn_res_tensorrt/20260716_190125_699274_dds_reshape_rewrite/dds_reshape_rewritten.onnx`
- ONNX SHA-256：`f71a585ded3f348a193e37395a124ab8a425ff72fc09ae92114d41872c460b98`
- checkpoint：`models/testParameters/GCN_res/best_model.pth`
- 输入：`artifacts/gcn_res_onnx/20260715_onnx_after_cdist_fp32_opset18/export_input.npz`
- 样本：`val_00_weld_7`
- 输入文件 SHA-256：`41a6d03a017223f8a3681b102832d964a959d36e8691ea155d1bed5cf4d4a471`

输入数组完全相同且为 C-contiguous FP32：

| Tensor | Shape | Bytes | Array SHA-256 |
|---|---:|---:|---|
| points | `[1,2048,4]` | 32,768 | `f9b272d41dddea0d9d8e813f0605e4aae652b19496928392f9f36db51e28bc24` |
| adj | `[1,2048,2048]` | 16,777,216 | `0e712ec9ce04b617b6b4de50cbfb17f706cae979d44aa52fa1366a068ce85220` |

## 3. Runtime 实现

新增脚本：

- `scripts/run_gcn_res_tensorrt_fp32_inference.py`
- `scripts/compare_tensorrt_pytorch_logits.py`

Runtime 路径：

1. 注册 TensorRT standard plugins 与 `VoxelUniquePluginCreator`；
2. `Runtime.deserialize_cuda_engine()`；
3. 创建一个 execution context；
4. 使用 `cuda-python 12.8.0` 的 `cuda.bindings.runtime` 创建显式 CUDA stream；
5. 使用 `cudaMalloc` 分配 points、adj、logits buffer；
6. 使用 `cudaMemcpyAsync` 完成 H2D；
7. 使用 `set_tensor_address()` 绑定三个 tensor；
8. 只调用一次 `execute_async_v3()`，即 enqueueV3；
9. 使用同一 stream 执行 logits D2H 并同步；
10. 释放全部 device buffer 并销毁 stream。

没有 warmup、循环或 benchmark。单次执行及 D2H 的观测时间为
`0.096920 s`，该数值不作为性能基准。

CUDA buffer：

| Tensor | Direction | Bytes |
|---|---|---:|
| points | H2D | 32,768 |
| adj | H2D | 16,777,216 |
| logits | D2H | 16,384 |

TensorRT ErrorRecorder 同时挂载到 Runtime、Engine 和 Context，最终记录
`0` 条错误且未 overflow。

## 4. Engine I/O 验证

| Tensor | Mode | Dtype | Engine/Context shape |
|---|---|---|---|
| points | INPUT | FLOAT | `[1,2048,4]` |
| adj | INPUT | FLOAT | `[1,2048,2048]` |
| logits | OUTPUT | FLOAT | `[1,2048,2]` |

四个 `VoxelUnique` runtime instance 均成功创建。

## 5. PyTorch baseline

使用 `deployment/gcn_res_onnx_model.py::GCNResStandardOps` 和
`deployment/gcn_res_onnx_wrapper.py::GCNResOnnxWrapper`：

- `in_dim=4`；
- `num_class=2`；
- checkpoint `load_state_dict(strict=True)`；
- CUDA FP32；
- 与 TensorRT 完全相同的 points 和 adj；
- 输出 shape `[1,2048,2]`；
- 输出全部有限。

## 6. PyTorch vs TensorRT

| 指标 | 结果 |
|---|---:|
| max absolute error | `1.444053650e-02` |
| mean absolute error | `1.960213384e-03` |
| RMSE | `2.894863672e-03` |
| cosine similarity | `0.999999752275` |
| max relative error，分母下限 `1e-8` | `5.887851214e-02` |
| mean relative error，分母下限 `1e-8` | `9.129480492e-04` |
| relative L2 error | `8.309721070e-04` |
| matching points | `2048` |
| total points | `2048` |
| label agreement | `1.0` |

建议验收条件为 `max_abs_error < 1e-4` 且
`cosine_similarity > 0.9999`。cosine 和标签一致率通过，但最大绝对误差未通过，
所以不能标记 parity pass。

## 7. 单样本分类指标

标签语义：class 0 为 `weld_seam`，class 1 为 `background`。
由于 TensorRT 和 PyTorch 的预测标签完全相同，两者指标一致：

| 指标 | TensorRT / PyTorch |
|---|---:|
| overall accuracy | `0.9765625` |
| weld seam IoU | `0.833910035` |
| background IoU | `0.973436635` |
| mIoU | `0.903673335` |
| weld seam precision | `0.979674797` |
| weld seam recall | `0.848591549` |
| weld seam F1 | `0.909433962` |

混淆矩阵定义为行=ground truth、列=prediction：

```text
[[ 241,   43],
 [   5, 1759]]
```

## 8. 只读离线定位

没有再次运行模型，只比较已保存数组：

| 比较 | max abs | mean abs | cosine | label agreement |
|---|---:|---:|---:|---:|
| 新 PyTorch vs 历史 PyTorch deployment reference | `1.907349e-06` | `2.413658e-07` | `0.9999999999999954` | `1.0` |
| TensorRT vs 历史 PyTorch reference | `1.444006e-02` | `1.960200e-03` | `0.9999997522857026` | `1.0` |
| TensorRT vs 新 PyTorch reference | `1.444054e-02` | `1.960213e-03` | `0.9999997522749307` | `1.0` |

这排除了本次 PyTorch scatter reduction 抖动作为 `1.44e-2` 误差的主要解释。
当前证据只能把差异定位到 TensorRT 执行路径内部；尚无中间张量证据可以在
`VoxelUnique Plugin`、voxel mapping、Scatter、DDS rewrite 或 TensorRT tactic
之间确定首个分叉点，不能武断归因。

## 9. 产物

目录：

`artifacts/gcn_res_tensorrt/20260716_212305_673121_fp32_inference/`

- `tensorrt_logits.npy`
- `pytorch_logits.npy`
- `inference_summary.json`
- `runtime_environment.json`
- `parity_report.json`
- `offline_reference_diagnosis.json`

## 10. 当前停止状态

```text
TENSORRT Runtime: PASSED
PyTorch baseline: PASSED
TensorRT ErrorRecorder: 0 errors
Numerical parity: FAILED
FP16: NOT RUN
INT8: NOT RUN
Benchmark: NOT RUN
C++ deployment: NOT RUN

TENSORRT_FP32_NUMERICAL_PARITY_FAILED
```

下一步应先设计只读的中间张量 parity 定位方案，从第一级
`VoxelUnique → Scatter/pooling → Reshape/Unsqueeze` 开始确定首个运行时分叉，
再决定是否需要处理 Plugin、Scatter 或 TensorRT tactic。
