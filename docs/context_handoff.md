# PTV2-WeldSeg-Deployment 项目交接

更新日期：2026-07-15（Asia/Shanghai）

## 1. 项目与最终模型

目标链路：

```text
PyTorch → ONNX → TensorRT FP32 Engine → TensorRT Python parity → C++ GPU inference
```

- 项目：`E:\GRP-PTv2`
- 模型：`models/testParameters/GCN_res/model.py`
- checkpoint：`models/testParameters/GCN_res/best_model.pth`
- 标签 0：`weld_seam`
- 标签 1：`background`

固定接口：

```text
points: float32 [1, 2048, 4]
adj:    float32 [1, 2048, 2048]
logits: float32 [1, 2048, 2]
```

测试基线：test mIoU 0.936309，weld F1 0.946799。

## 2. PyTorch 与 deployment 状态

deployment 模型：`deployment/gcn_res_onnx_model.py`

```text
TRUNC_FLOOR_EQUIVALENCE_PASSED
SCATTER_REDUCE_AMIN_EQUIVALENCE_PASSED
SCATTER_REDUCE_AMAX_EQUIVALENCE_PASSED
STANDARD_OPS_VOXEL_POOL_EQUIVALENCE_PASSED
```

CUDA FP32 scatter reduction 存在非 bitwise 的归约顺序误差；最终 logits 最大绝对误差约 `2.503395e-06`，标签一致率 100%。

## 3. ONNX 与 ORT 状态

现有 ONNX：

`artifacts/gcn_res_onnx/20260715_onnx_after_cdist_fp32_opset18/gcn_res_deploy_fp32_opset18.onnx`

```text
GCN_RES_ONNX_EXPORT_PASSED
ONNX graph validation: PASSED
```

ORT CPU label agreement 为 100%，但 logits 未达到既定 allclose 容差。

ORT CUDA 准确状态：

```text
ONNXRUNTIME_CUDA_PARITY_NOT_COMPLETED
ORT_CUDA_PROFILING_NO_EVENTS_FLUSHED
ORT_CUDA_EXTREME_LATENCY_REPRODUCED
```

必须保留结论：

```text
ORT CUDA parity was not completed because CUDA EP inference showed reproducible extreme latency.
```

不得把 ORT CUDA parity 记为已通过，也不要再次运行长时间 ORT CUDA inference。

## 4. 当前 Python/GPU/CUDA 环境

- 虚拟环境：`E:\GRP-PTv2\.venv_ptv2`
- Python：3.11.8
- PyTorch：2.7.1+cu128
- PyTorch CUDA Runtime：12.8
- cuDNN：9.7.1
- onnxruntime-gpu：1.26.0
- GPU：NVIDIA GeForce RTX 5060，SM 12.0
- driver：610.74
- CUDA Toolkit：12.8 Update 1
- `nvcc`：V12.8.93
- `CUDA_PATH`：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`
- `pip check`：No broken requirements found

PyTorch、ORT 和全部 PyG CUDA 扩展保持原锁定版本。

## 5. TensorRT 环境

TensorRT SDK 根目录：

```text
D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106
```

当前状态：

```text
TensorRT SDK: 11.1.0.106 Windows x64 CUDA 12
TensorRT Python: 11.1.0.106
trtexec: TensorRT v110100 build 106
TENSORRT_PYTHON_BUILDER_AVAILABLE
TENSORRT_ENVIRONMENT_READY
```

完整 SDK 已确认包含：

- `bin/trtexec.exe`；
- `include/NvInfer.h`；
- `include/NvOnnxParser.h`；
- `lib/nvinfer_11.lib`；
- `lib/nvonnxparser_11.lib`；
- `bin/nvinfer_11.dll`；
- `bin/nvonnxparser_11.dll`；
- `bin/nvinfer_builder_resource_sm120_11.dll`。

Python binding 使用 SDK 自带 `cp311` wheel，以 `--no-deps` 安装；只新增 `tensorrt==11.1.0.106`，其他包未变化。

TensorRT `bin` 没有永久加入 User/Machine PATH。每个新 PowerShell 在使用 TensorRT 前应执行：

```powershell
$env:TENSORRT_ROOT = 'D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106'
$env:PATH = "$env:TENSORRT_ROOT\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;$env:PATH"
```

不设置该临时 PATH 时，Python import 会因找不到 `nvinfer_11.dll` 失败；设置后 import 和 Builder 已验证成功。

## 6. C++ 工具链

- Visual Studio Community 2022 17.8.2；
- Desktop development with C++：已安装；
- MSVC x64 19.38.33130；
- CMake 4.1.0。

当前普通 PATH 的 `cl.exe` 优先命中旧 VS2015 x86。后续 C++ 构建必须使用 VS2022 Developer Command Prompt 或 `vcvars64.bat`。

## 7. TensorRT Phase 1 Parser 结果

TensorRT 11.1.0.106 Python Parser 已对现有 ONNX 完成只读解析：

```text
TENSORRT_ONNX_PARSER_FAILED
FIRST_BLOCKING_OPERATOR=Unique
FIRST_BLOCKING_NODE=/model/tdb_1/Unique
```

Parser 共报告 4 个 `UNSUPPORTED_NODE`：

- `/model/tdb_1/Unique`，ONNX node 153；
- `/model/tdb_2/Unique`，ONNX node 495；
- `/model/tdb_3/Unique`，ONNX node 869；
- `/model/tdb_4/Unique`，ONNX node 1243。

两个输入已正确识别为 `points [1,2048,4]` 和 `adj [1,2048,2048]`，但 parser 未完成，network output 数为 0。没有创建 builder config，没有构建 Engine，没有运行 inference。

`trtexec` 11.1 没有真正的 parser-only 参数；`--skipInference` 仍会先构建 Engine，因此本阶段只保存 `trtexec --help`，没有用它加载 ONNX。

运行目录：

`artifacts/gcn_res_tensorrt/20260715_164224_298664_parser_audit/`

详细报告：`docs/tensorrt_phase1_parser_audit.md`

## 8. 下一步唯一入口

在任何 Engine build 之前，先对四个 ONNX `Unique` 的排序、inverse mapping、counts 和下游依赖做数学语义审计。禁止直接删除或绕过 voxel key 去重；只有标准算子等价实现完成逐层验证后，才允许重新导出 ONNX 并再次执行 Parser。

当前仍禁止 TensorRT Engine build、TensorRT inference、FP16 和 benchmark；没有修改 ONNX、模型、checkpoint、数据或容差。

环境记录：`docs/tensorrt_phase0_parser_build_audit.md`
