# PTV2-WeldSeg-Deployment 项目交接

更新时间：2026-07-16（Asia/Shanghai）

## 1. 最终模型与任务

- 项目：`E:\GRP-PTv2`
- 模型：`models/testParameters/GCN_res/model.py`
- checkpoint：`models/testParameters/GCN_res/best_model.pth`
- class 0：`weld_seam`
- class 1：`background`

固定接口：

```text
points: FP32 [1, 2048, 4]
adj:    FP32 [1, 2048, 2048]
logits: FP32 [1, 2048, 2]
```

固定评估基线：test mIoU `0.936309`，weld F1 `0.946799`。

## 2. PyTorch 与 deployment 状态

deployment模型：`deployment/gcn_res_onnx_model.py`

```text
CHECKPOINT_AND_FORWARD_VALIDATION_PASSED
TRUNC_FLOOR_EQUIVALENCE_PASSED
SCATTER_REDUCE_AMIN_EQUIVALENCE_PASSED
SCATTER_REDUCE_AMAX_EQUIVALENCE_PASSED
STANDARD_OPS_VOXEL_POOL_EQUIVALENCE_PASSED
```

CUDA FP32 scatter reduction存在非bitwise归约顺序误差；最终logits最大绝对误差
约 `2.503395e-06`，标签一致率100%。

## 3. ONNX 与 ONNX Runtime

原部署ONNX：

```text
artifacts/gcn_res_onnx/20260715_onnx_after_cdist_fp32_opset18/
gcn_res_deploy_fp32_opset18.onnx
```

```text
GCN_RES_ONNX_EXPORT_PASSED
ONNX graph validation: PASSED
```

ORT CPU标签一致率100%，但logits未满足既定allclose容差。

ORT CUDA必须保留以下真实状态：

```text
ONNXRUNTIME_CUDA_PARITY_NOT_COMPLETED
ORT_CUDA_PROFILING_NO_EVENTS_FLUSHED
ORT_CUDA_EXTREME_LATENCY_REPRODUCED
```

CUDA EP推理出现可复现的极端延迟，因此ORT CUDA parity没有完成。不得把它写成
已通过，也不要再次运行长时间ORT CUDA inference。

## 4. 当前环境

- 虚拟环境：`E:\GRP-PTv2\.venv_ptv2`
- Python：3.11.8
- PyTorch：2.7.1+cu128
- PyTorch CUDA Runtime：12.8
- cuDNN：9.7.1
- onnxruntime-gpu：1.26.0
- GPU：NVIDIA GeForce RTX 5060，SM 12.0
- Driver：610.74
- CUDA Toolkit：12.8 Update 1，nvcc 12.8.93
- CUDA_PATH：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`
- TensorRT SDK/Python：11.1.0.106
- pip check：No broken requirements found

TensorRT SDK：

```text
D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106
```

新终端使用TensorRT前临时设置：

```powershell
$env:TENSORRT_ROOT = 'D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106'
$env:PATH = "$env:TENSORRT_ROOT\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;$env:PATH"
```

C++/CUDA必须使用VS2022 x64工具链；普通PATH中还存在旧VS2015，不能使用。

## 5. TensorRT graph/parser进度

已完成：

- `Unique`语义与TDB依赖审计；
- IPluginV3 size tensor最小原型；
- Custom ONNX → Plugin Creator → Parser链路；
- `VoxelUniquePlugin`正确性测试；
- 4个`ai.onnx::Unique`替换为
  `com.tensorrt.ptv2::VoxelUnique`；
- 标准`ScatterElements` Plugin注册；
- 16个constant-false `If`等价性审计与折叠；
- ONNX checker；
- TensorRT Parser，0 errors。

当前TensorRT部署ONNX：

```text
artifacts/gcn_res_tensorrt/20260715_213934_180785_if_folded/if_folded.onnx
```

SHA-256：

```text
f0ca962b4e46e7495d40c7f23387c8dffbd4ca88e580452408f2fd9da85bc5ba
```

## 6. TensorRT Phase 4结果

Phase 4 Parser再次通过，四个`VoxelUnique` BUILD实例均创建成功，但FP32 Engine
构建失败：

```text
TENSORRT_FP32_ENGINE_BUILD_FAILED
```

首个TensorRT原生错误：

```text
convertExplicitDDSPluginToImplicit.cpp:149
Error Code 2: Internal Error
Assertion nodeIdxToDDSOutputIndices.count(i) ==
          nodeIdxToSizeTensors.count(i) failed
```

判断：

- 与`VoxelUnique`运行时长度/size tensor集成直接相关；
- 属于DDS/shape-tensor转换阶段；
- 不是Parser回归；
- 没有证据指向ScatterElements；
- 不是workspace不足，因此没有8 GiB重试；
- 尚未进入tactic失败；
- 没有SM120 kernel/capability错误。

运行目录：

```text
artifacts/gcn_res_tensorrt/20260716_171437_360317_fp32_engine_build/
```

完整报告：`docs/tensorrt_phase4_fp32_engine_build.md`

## 7. 当前严格状态

```text
Parser: PASSED (0 errors)
FP32 Engine build: FAILED
Serialized engine: NOT GENERATED
Engine deserialization: NOT RUN
Engine I/O validation: NOT RUN
Engine Inspector: NOT RUN
TensorRT inference: NOT RUN
TensorRT parity: NOT RUN
```

不得写成`TENSORRT_FP32_ENGINE_BUILD_PASSED`或TensorRT inference/parity已通过。

## 8. 唯一下一步

在重新构建Engine前，先审计TensorRT 11.1 IPluginV3多输出DDS协议与
`declareSizeTensor()`映射，确认 `voxel_count`、`unique_values[M]`、
`inverse_indices[N]` 三个输出的size tensor表达是否满足Builder要求。

当前禁止直接重试、增加workspace、执行TensorRT inference、FP16、INT8或benchmark。
