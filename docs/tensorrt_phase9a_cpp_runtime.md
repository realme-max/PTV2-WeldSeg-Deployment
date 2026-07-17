# TensorRT Phase 9A: C++ Runtime Minimal Inference

## 1. 目标与结论

Phase 9A 在不修改 checkpoint、ONNX、Engine、VoxelUniqueCub CUDA 算法、Builder 配置和现有 Python 推理逻辑的前提下，新增了 Windows C++17 TensorRT Runtime 后端，并使用 Phase 8D 正式生产包完成最小推理闭环。

```text
TENSORRT_CPP_RUNTIME_MINIMAL_INFERENCE_COMPLETED
```

验证链路为：

```text
raw FP32 binary
  -> C++ TensorRT Runtime
  -> explicit VoxelUniqueCub Plugin registration
  -> Engine deserialize
  -> CUDA H2D
  -> setTensorAddress
  -> enqueueV3
  -> CUDA D2H
  -> logits.bin
```

本阶段没有进入 Qt、Robot、PCL、GUI、FP16、INT8 或 Engine rebuild。

## 2. 目录结构

```text
deployment/tensorrt_runtime/
├── CMakeLists.txt
├── README.md
├── examples/
│   └── README.md
├── include/
│   ├── CudaBufferManager.h
│   ├── PluginLoader.h
│   └── TensorRTInference.h
└── src/
    ├── CudaBufferManager.cpp
    ├── PluginLoader.cpp
    ├── TensorRTInference.cpp
    └── main.cpp

scripts/
└── validate_gcn_res_tensorrt_cpp_runtime.py
```

核心职责：

- `PluginLoader`：调用 `initLibNvInferPlugins`，使用 `LoadLibraryW`/`GetProcAddress` 加载生产 Plugin，调用 `initVoxelUniqueCubPlugin`，验证 `VoxelUniqueCub / 1 / com.tensorrt.ptv2.experimental` Creator，并在 Runtime 资源释放后注销 Creator 和卸载 DLL。
- `CudaBufferManager`：管理 `points`、`adj`、`logits` 三块 CUDA device memory，检查每次 `cudaMalloc`、`cudaMemcpyAsync` 和释放路径。
- `TensorRTInference`：验证 Engine SHA-256，创建 Runtime、反序列化 Engine、创建 Context、验证固定 I/O 契约、绑定地址、执行 `enqueueV3`、同步并检查 ErrorRecorder。
- `trt_runtime_demo`：读取无头部 FP32 binary，严格检查字节数，输出 `logits.bin` 和运行记录。

## 3. 生产依赖与冻结哈希

Deployment ID：`gcn-res-trt-cub-strict-fp32-20260717_173128_144483`。

| 对象 | 路径 | SHA-256 |
|---|---|---|
| Engine | `package/engine/strict_fp32_voxelunique_cub.plan` | `a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299` |
| Plugin | `package/plugins/VoxelUniqueCubPlugin.dll` | `6641ec147e8eac10206a5c60ba1c1390c398d4b59e32a6a618a37046360ec348` |
| ONNX（未修改、未读取执行） | `package/model/gcn_res_voxelunique_cub.onnx` | `16ca5c16c330e6572b1730e80da724231a28b68872a3203c21240348d4d89299` |

C++ Runtime 强制要求调用方传入 Engine SHA-256。实际哈希不一致时，在加载 Plugin 和反序列化 Engine 前失败。

## 4. 固定 I/O contract

| Tensor | Mode | dtype | shape | elements | bytes |
|---|---|---|---|---:|---:|
| `points` | INPUT | FP32 | `[1,2048,4]` | 8192 | 32768 |
| `adj` | INPUT | FP32 | `[1,2048,2048]` | 4194304 | 16777216 |
| `logits` | OUTPUT | FP32 | `[1,2048,2]` | 4096 | 16384 |

`.bin` 为 little-endian、连续、无 header 的 `float32`。C++ Runtime 不依赖 NumPy；验证脚本只负责从冻结 Phase 8D `.npy/.npz` 生成 binary 和比较结果。

## 5. CMake 编译

实际环境：

- Windows 10 `10.0.19045`，Windows SDK `10.0.22621.0`；
- Visual Studio 2022，MSVC `19.38.33130.0`，x64，toolset `v143`；
- CMake `4.1.0`；
- CUDA Toolkit `12.8.93`；
- TensorRT SDK `11.1.0.106`。

实际命令：

```powershell
cmake -S E:\GRP-PTv2\deployment\tensorrt_runtime `
  -B E:\GRP-PTv2\artifacts\gcn_res_tensorrt\phase9a_cpp_runtime_build `
  -G "Visual Studio 17 2022" -A x64 -T v143 `
  -DTENSORRT_ROOT="D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106" `
  -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

cmake --build E:\GRP-PTv2\artifacts\gcn_res_tensorrt\phase9a_cpp_runtime_build `
  --config Release --parallel 4
```

最终结果：`Release/trt_runtime_demo.exe` 构建成功。运行所需 `nvinfer_11.dll`、`nvinfer_plugin_11.dll` 和 `cudart64_12.dll` 被复制到 exe 目录。全局 vcpkg applocal hook 因当前 PATH 不含 `pwsh.exe` 打印了一条非致命提示，但 MSBuild 返回 0，目标和运行 DLL 均完整，随后实际 Runtime 验证通过。

## 6. 运行方式

```powershell
trt_runtime_demo.exe `
  --engine <package>\engine\strict_fp32_voxelunique_cub.plan `
  --plugin <package>\plugins\VoxelUniqueCubPlugin.dll `
  --engine-sha256 a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299 `
  --points weld_65_points.bin `
  --adj weld_65_adj.bin `
  --output cpp_logits.bin `
  --runtime-json cpp_runtime_summary.json `
  --benchmark-json cpp_runtime_benchmark.json `
  --warmup 100 `
  --iterations 100
```

实际验证编排命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe `
  E:\GRP-PTv2\scripts\validate_gcn_res_tensorrt_cpp_runtime.py `
  --exe E:\GRP-PTv2\artifacts\gcn_res_tensorrt\phase9a_cpp_runtime_build\Release\trt_runtime_demo.exe `
  --warmup 100 --iterations 100
```

最终产物：`artifacts/gcn_res_tensorrt/20260717_212521_216041_phase9a_cpp_runtime/`。

## 7. Runtime 与数值验证

Runtime 结果：

- TensorRT：`11.1.0.106`；
- Engine deserialize：PASS；
- Context creation：PASS；
- Engine I/O contract：PASS；
- 已注册 Creator 总数：68；
- `VoxelUniqueCub` Runtime 实例：4；
- `enqueueV3`：PASS；
- ErrorRecorder errors：0；
- logits shape/dtype：`[1,2048,2] / float32`；
- logits finite：true。

同一冻结 `weld_65` 输入上的 Python TensorRT 与 C++ TensorRT 比较：

| 指标 | 结果 |
|---|---:|
| max abs error | `3.5762786865234375e-06` |
| mean abs error | `3.5584344004746526e-07` |
| matching points | `2048 / 2048` |
| label agreement | `100%` |
| Python/C++ mIoU | `0.9235089884843775 / 0.9235089884843775` |
| mIoU delta | `0.0` |
| Python/C++ weld F1 | `0.9262086513994912 / 0.9262086513994912` |
| weld F1 delta | `0.0` |

该结果确认 C++ Runtime 与 Python TensorRT 对该固定样本任务级一致，但不改变 Phase 8D 已记录的 `CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED`，也不宣称 Engine 相对 PyTorch 获得严格数值等价。

## 8. Phase 9A 轻量计时

本结果仅为 C++ Runtime 的 100 warmup + 100 inference 功能性计时，不替代 Phase 8D 正式 benchmark：

| 项目 | 结果 |
|---|---:|
| initialization | `623.7955 ms` |
| plugin load | `1.1889 ms` |
| Engine deserialize | `27.6010 ms` |
| context creation | `410.9331 ms` |
| CUDA Event inference mean | `5.0713 ms` |
| CUDA Event inference P50 / P95 | `5.0012 / 5.9260 ms` |
| host E2E mean | `8.2245 ms` |
| host E2E P50 / P95 | `8.0459 / 9.7061 ms` |

## 9. Fail-closed 验证

以下 5 个真实负向用例均以非零状态退出，未产生推理结果，也不存在 PyTorch fallback：

- Engine 不存在；
- Plugin 不存在；
- DLL 可加载但缺少必需 Plugin exports；
- Engine SHA-256 不匹配；
- `points.bin` 字节数错误（32764，而非 32768）。

所有 CUDA 调用、`setTensorAddress`、`enqueueV3`、同步和 TensorRT ErrorRecorder 均有返回值检查。没有人为注入 GPU 故障，因此 CUDA 失败路径为代码级覆盖，而非硬件故障实测。

## 10. 已知限制

- 仅支持 Windows x64、TensorRT 11.1.0.106、CUDA 12.8 和当前 SM120 生产 Engine；
- 固定 `B=1, N=2048, FP32`，不支持 dynamic batch/points；
- 输入已经完成归一化、类别通道拼接和 CPU 邻接矩阵构建；本阶段不包含点云 TXT 预处理；
- API 当前为最小同步闭环，不包含多 context、并发、零拷贝或服务化；
- binary 文件没有版本 header，必须由上层严格遵守 manifest contract；
- 未接入 Qt、PCL、Robot 或 GUI；
- 未执行 FP16、INT8、Engine rebuild 或 Phase 9B/Qt 集成。

```text
TENSORRT_CPP_RUNTIME_MINIMAL_INFERENCE_COMPLETED
```
