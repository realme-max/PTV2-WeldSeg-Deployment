# TensorRT Phase 0：环境安装与 Parser/Build 前置审计

## 1. 当前结论

更新日期：2026-07-15（Asia/Shanghai）

CUDA Toolkit 12.8 Update 1 和 TensorRT 11.1.0.106 Windows x64 CUDA 12 完整 SDK 已安装/配置完成。TensorRT Python wheel 来自同一 SDK，版本完全一致，Python Builder 创建测试通过。

```text
TENSORRT_ENVIRONMENT_READY
```

该状态表示 TensorRT parser/build 所需环境已准备好，不表示 ONNX 已被 TensorRT 解析，也不表示 Engine 已生成。本轮没有执行 ONNX Parser、Engine build、TensorRT inference、ORT CUDA inference、FP16 或 benchmark。

项目状态：

```text
ONNX export: PASSED
ONNX graph validation: PASSED
ORT CPU numerical validation: label agreement 100%
ORT CUDA numerical parity: BLOCKED_BY_EXTREME_LATENCY
TensorRT environment: READY
TensorRT ONNX parser: NOT RUN
TensorRT FP32 engine build: NOT RUN
```

## 2. CUDA Toolkit 验收

| 项目 | 实测结果 |
|---|---|
| CUDA Toolkit | 12.8 Update 1 |
| `nvcc` | release 12.8, V12.8.93 |
| CUDA Toolkit 根目录 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8` |
| `nvcc.exe` | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe` |
| Machine `CUDA_PATH` | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8` |
| User PATH | 已加入 CUDA `v12.8\bin`，未加入 `libnvvp` |
| GPU | NVIDIA GeForce RTX 5060，SM 12.0 |
| NVIDIA driver | 610.74 |

安装器因为 Machine PATH 过长，未能自动加入 CUDA 路径。审计后只在 User PATH 前置以下一项，未清理或改写 Machine PATH：

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
```

绝对路径验证：

```text
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```

## 3. TensorRT 完整 SDK

SDK 根目录：

```text
D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106
```

下载包：

```text
TensorRT-Enterprise-11.1.0.106-Windows-amd64-cuda-12.9-Release-external.zip
```

该包是 TensorRT 11.1 的 CUDA 12 Windows x64 完整 SDK。文件名中的 `cuda-12.9` 表示 CUDA 12 构建所使用的 Toolkit 基线；当前 CUDA 12.8 Update 1 满足 RTX 5060/SM 12.0 的 CUDA 12 环境要求。

SDK 目录检查：

| 项目 | 路径/状态 |
|---|---|
| `bin` | 存在 |
| `include` | 存在 |
| `lib` | 存在 |
| `python` | 存在 |
| `trtexec.exe` | `...\bin\trtexec.exe` |
| `NvInfer.h` | `...\include\NvInfer.h` |
| `NvOnnxParser.h` | `...\include\NvOnnxParser.h` |
| TensorRT import libraries | `...\lib\nvinfer_11.lib` 等 |
| ONNX parser import library | `...\lib\nvonnxparser_11.lib` |
| TensorRT DLL | `...\bin\nvinfer_11.dll` |
| ONNX parser DLL | `...\bin\nvonnxparser_11.dll` |
| SM 12.0 builder resource | `...\bin\nvinfer_builder_resource_sm120_11.dll` |

头文件版本宏：

```text
TRT_MAJOR_ENTERPRISE 11
TRT_MINOR_ENTERPRISE 1
TRT_PATCH_ENTERPRISE 0
TRT_BUILD_ENTERPRISE 106
```

## 4. `trtexec` 验证

完整路径：

```text
D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106\bin\trtexec.exe
```

在临时 PATH 中加入 TensorRT `bin` 和 CUDA `bin` 后，`trtexec --help` 返回码为 0，版本横幅为：

```text
TensorRT.trtexec [TensorRT v110100] [b106]
```

该版本的 `trtexec --version` 会先打印版本横幅，随后因为没有模型参数而显示帮助并返回 1；因此环境 smoke test 使用该版本实际支持的 `--help`，未把返回 1 误报为 SDK 不可用。

TensorRT 没有永久加入 User/Machine PATH。每次使用前应在当前 PowerShell 临时执行：

```powershell
$env:TENSORRT_ROOT = 'D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106'
$env:PATH = "$env:TENSORRT_ROOT\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;$env:PATH"
```

## 5. TensorRT Python binding

安装来源：SDK 自带 CPython 3.11 wheel。

```text
D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106\python\tensorrt-11.1.0.106-cp311-none-win_amd64.whl
```

执行方式：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -m pip install --no-deps <上述 wheel>
```

结果：

```text
Successfully installed tensorrt-11.1.0.106
tensorrt_version 11.1.0.106
TENSORRT_PYTHON_BUILDER_AVAILABLE
```

只安装 Full Runtime wheel，没有安装 Lean 或 Dispatch wheel，没有访问 PyPI 解析 TensorRT 依赖。

由于按要求没有永久加入 TensorRT `bin`，普通新终端中直接 `import tensorrt` 会提示找不到 `nvinfer_11.dll`。这不是 wheel 或 SDK 版本失败；执行上一节的当前终端临时 PATH 配置后，import 和 Builder 均通过。

## 6. 安装前后依赖检查

安装前：

```text
python -m pip check
No broken requirements found.
```

安装后：

```text
python -m pip check
No broken requirements found.
```

`pip freeze` 的唯一新增项为本地 SDK wheel：

```text
tensorrt @ file:///D:/NVIDIA_GeForce5060/TensorRT-11.1.0/TensorRT-11.1.0.106/python/tensorrt-11.1.0.106-cp311-none-win_amd64.whl
```

锁定依赖复查：

| 包 | 验收版本 |
|---|---|
| PyTorch | 2.7.1+cu128 |
| PyTorch CUDA Runtime | 12.8 |
| cuDNN | 9.7.1 / 90701 |
| onnxruntime-gpu | 1.26.0 |
| torch-geometric | 2.8.0 |
| torch_cluster | 1.6.3+pt27cu128 |
| torch_scatter | 2.1.2+pt27cu128 |
| torch_sparse | 0.6.18+pt27cu128 |
| pyg-lib | 0.5.0+pt27cu128 |
| torch_spline_conv | 1.2.2+pt27cu128 |

没有升级、降级或重装上述包。

## 7. C++ 工具链注意事项

- Visual Studio Community 2022：17.8.2。
- “使用 C++ 的桌面开发”工作负载：已安装。
- 可用 x64 MSVC：19.38.33130。
- CMake：4.1.0。

普通 PowerShell 的 `where cl` 仍优先命中旧 VS2015 x86 编译器。后续 C++ 项目必须使用 VS2022 Developer Command Prompt，或先运行：

```text
D:\vs2022\Community\VC\Auxiliary\Build\vcvars64.bat
```

## 8. 本轮明确未执行

- 没有运行 TensorRT ONNX Parser；
- 没有运行 `trtexec --onnx`；
- 没有构建 FP32/FP16 Engine；
- 没有执行 TensorRT inference；
- 没有重新运行 ORT CUDA inference；
- 没有修改 ONNX、模型、checkpoint、数据或容差。

## 9. 下一步

下一阶段从现有固定 ONNX 开始 TensorRT Python Parser 只读审计，并保存真实 parser error。Parser 完全通过后，才允许执行固定 FP32、B=1、N=2048 的 `trtexec --skipInference` / Engine build。

```text
TENSORRT_ENVIRONMENT_READY
```

