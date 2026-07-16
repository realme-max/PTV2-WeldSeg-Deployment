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
- GPU：NVIDIA GeForce RTX 5060，SM 12.0
- `pip check`：No broken requirements found

TensorRT SDK：

`D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106`

## 3. TensorRT 部署图与 Engine

当前 TensorRT 派生 ONNX：

`artifacts/gcn_res_tensorrt/20260716_190125_699274_dds_reshape_rewrite/dds_reshape_rewritten.onnx`

SHA-256：

`f71a585ded3f348a193e37395a124ab8a425ff72fc09ae92114d41872c460b98`

当前 FP32 Engine：

`artifacts/gcn_res_tensorrt/20260716_190125_699274_dds_reshape_rewrite/gcn_res_dds_reshape_fp32_b1_n2048.plan`

SHA-256：

`7a856d1aa50628360d4acd5ee384fcd5042a8087a3112a361ee3c47db9e7326b`

已完成：

- `Unique → com.tensorrt.ptv2::VoxelUnique`；
- 16 个 constant-false If folding；
- 8 个已审计 DDS Reshape 等价改写为 `Unsqueeze(axis=0)`；
- ONNX checker 与 shape inference；
- TensorRT Parser，0 errors；
- 4 GiB FP32 Builder；
- Engine 保存、反序列化及 I/O 检查；
- Build/Runtime/Inspector 均确认 4 个 VoxelUnique 实例。

```text
TENSORRT_DDS_RESHAPE_REWRITE_ENGINE_BUILD_PASSED
TENSORRT_FP32_ENGINE_BUILD_PASSED
```

## 4. Phase 5 Runtime 与 parity

固定样本：`val_00_weld_7`。

输入：

`artifacts/gcn_res_onnx/20260715_onnx_after_cdist_fp32_opset18/export_input.npz`

Runtime 已使用 cuda-python 显式分配 device buffer、创建 stream，并只调用一次
`execute_async_v3/enqueueV3`。TensorRT ErrorRecorder 为 0 errors，输出全部有限。

PyTorch deployment CUDA 与 TensorRT FP32：

| 指标 | 结果 |
|---|---:|
| max absolute error | `1.444053650e-02` |
| mean absolute error | `1.960213384e-03` |
| RMSE | `2.894863672e-03` |
| cosine similarity | `0.999999752275` |
| label agreement | `2048/2048 = 100%` |

单样本 TensorRT 与 PyTorch 分类结果完全相同：mIoU `0.903673335`，
weld seam F1 `0.909433962`。

但是 `max_abs_error` 没有满足建议的 `<1e-4` 条件，因此状态必须保持：

```text
TENSORRT_FP32_NUMERICAL_PARITY_FAILED
```

Phase 5 产物：

`artifacts/gcn_res_tensorrt/20260716_212305_673121_fp32_inference/`

完整报告：`docs/tensorrt_phase5_fp32_parity.md`。

## 5. 已完成的离线定位

新 PyTorch baseline 与历史 PyTorch deployment reference 的最大误差只有
`1.907349e-06`；TensorRT 对两份 PyTorch reference 的最大误差均约
`1.444e-02`。这说明差异不是本次 PyTorch baseline 抖动造成的。

目前没有中间张量证据可以确定首个分叉发生在 VoxelUnique、voxel mapping、
Scatter、DDS rewrite 还是 TensorRT tactic，禁止无证据修改任何一项。

## 6. 唯一下一步

先设计并执行 TensorRT 中间张量的只读 parity 定位，从第一级
`VoxelUnique → Scatter/pooling → Unsqueeze` 开始确定首个运行时数值分叉。

在 FP32 parity 通过前，禁止进入：

- FP16；
- INT8；
- benchmark 或 kernel 优化；
- C++ 部署；
- 修改 ONNX、Plugin、checkpoint 或验收阈值。
