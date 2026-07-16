# TensorRT Phase 4：GCN_res FP32 Engine Build

更新时间：2026-07-16（Asia/Shanghai）

## 1. 结论摘要

本轮完成了固定 shape、FP32 TensorRT Engine 构建尝试，但 Builder 在序列化
Engine 前触发 TensorRT 内部 DDS（data-dependent shape）size-tensor 断言，因此
按停止条件终止：

```text
TENSORRT_FP32_ENGINE_BUILD_FAILED
```

已确认：

- ONNX Parser 成功，错误数为 0；
- TensorRT 标准插件初始化成功；
- `VoxelUnique` version `1` Creator 注册并回查成功；
- 标准 `ScatterElements` version `2` Creator 回查成功；
- 四个 `VoxelUnique` BUILD Plugin 实例均已创建；
- Builder 确实开始执行，但没有产生 serialized engine；
- 没有 `.plan`；
- 没有反序列化 Engine；
- 没有创建 execution context；
- 没有分配模型输入/输出显存；
- 没有执行 TensorRT inference、parity、benchmark、FP16 或 INT8。

## 2. 输入模型

输入 ONNX：

```text
E:\GRP-PTv2\artifacts\gcn_res_tensorrt\
20260715_213934_180785_if_folded\if_folded.onnx
```

SHA-256：

```text
f0ca962b4e46e7495d40c7f23387c8dffbd4ca88e580452408f2fd9da85bc5ba
```

固定接口：

```text
points  FP32 [1, 2048, 4]
adj     FP32 [1, 2048, 2048]
logits  FP32 [1, 2048, 2]
```

本轮没有修改该 ONNX。

## 3. 环境

| 项目 | 实际值 |
|---|---|
| OS | Windows 10 10.0.19045 |
| Python | 3.11.8 |
| Python executable | `E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe` |
| TensorRT | 11.1.0.106 |
| CUDA Toolkit | 12.8 Update 1，nvcc 12.8.93 |
| PyTorch | 2.7.1+cu128 |
| cuDNN | 9.7.1 |
| GPU | NVIDIA GeForce RTX 5060 |
| Compute capability | 12.0 / SM120 |
| Driver | 610.74 |
| pip check | No broken requirements found |

TensorRT SDK：

```text
D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106
```

插件 DLL 使用 VS2022 x64、MSVC 19.38、CUDA 12.8、`sm_120` 构建。DLL 仅增加
Creator 注册桥，插件算法直接复用已经完成正确性验证的：

```text
tests/tensorrt_voxel_unique_correctness/VoxelUniqueCorrectnessPlugin.cu
```

## 4. 构建配置

```text
precision: FP32
workspace: 4 GiB (4294967296 bytes)
points shape: [1,2048,4]
adj shape: [1,2048,2048]
optimization profile: 未创建（输入为静态 shape）
FP16: 未启用
INT8: 未启用
sparsity: 未启用
DLA: 未启用
refit: 未启用
version compatible: 未启用
weight streaming: 未启用
TF32: 保持 TensorRT 11.1 默认值，没有主动修改
timeout: 1800 秒
```

TensorRT 11.1 Python API 已不再暴露旧的 `BuilderFlag.FP16`、
`BuilderFlag.INT8` 和 `EXPLICIT_BATCH`；显式 batch 为强制默认。脚本记录了API
可用性，没有通过替代方式启用低精度。

构建在独立子进程中执行，stdout/stderr 完整写入 `builder_verbose.log`。本轮只
启动一个 Builder 子进程，没有并发 Builder。

## 5. Parser 与插件注册

| 检查 | 结果 |
|---|---:|
| Standard plugins initialized | PASS |
| `VoxelUnique` v1 Creator | PASS |
| `ScatterElements` v2 Creator | PASS |
| Parser success | PASS |
| Parser errors | 0 |
| VoxelUnique BUILD instances | 4 |

因此本次失败不是 Parser 回归，也不是 Creator 缺失。

## 6. Engine Build 结果

执行目录：

```text
E:\GRP-PTv2\artifacts\gcn_res_tensorrt\
20260716_171437_360317_fp32_engine_build\
```

Builder 执行约 `7.544391` 秒后，`build_serialized_network()` 返回 `None`。
TensorRT 首个原生错误为：

```text
[convertExplicitDDSPluginToImplicit.cpp::
nvinfer1::builder::convertExplicitDDSPluginToImplicit::149]
Error Code 2: Internal Error
(Assertion nodeIdxToDDSOutputIndices.count(i) ==
nodeIdxToSizeTensors.count(i) failed.)
```

失败分类：

```text
TENSORRT_FP32_ENGINE_BUILD_FAILED
```

## 7. 根因分类

| 问题类别 | 判断 | 证据 |
|---|---|---|
| VoxelUnique size tensor | 是 | 失败发生在 TensorRT 的 explicit DDS plugin → implicit size-tensor 转换 |
| Shape tensor / data-dependent output | 是 | 断言直接比较 DDS output 与 size tensor 映射 |
| ScatterElements | 无证据 | Parser已越过Scatter，Builder错误未指向Scatter |
| Tactic profiling | 否 | 在 DDS 图转换阶段失败，尚无 tactic 失败信息 |
| Workspace不足 | 否 | 没有 workspace/OOM 错误；内部断言与4 GiB大小无关 |
| SM120不兼容 | 无证据 | 插件已按sm_120编译并注册，未出现kernel/capability错误 |

原生错误没有给出具体 ONNX node 名称，只给出 TensorRT 内部 DDS 转换 pass。
结合网络中只有 `VoxelUnique` 使用 `declareSizeTensor()`，可判定错误与该插件的
运行时输出长度集成有关；这是基于日志的定位，不代表已经证明 TensorRT 本身或
插件哪一侧应如何修改。

## 8. 为什么没有尝试 8 GiB

用户允许的8 GiB重试仅适用于明确的 workspace不足。当前首错是确定的 DDS
size-tensor内部断言，因此增加 workspace 不会针对根因。本轮没有启动第二个
Builder，也没有无限重试。

## 9. 结构验证状态

由于没有 serialized engine：

- `deserializeCudaEngine()`：未执行；
- Engine I/O枚举：未执行；
- Engine Inspector：未执行；
- VoxelUnique RUNTIME实例：未创建；
- `.plan`：不存在；
- Engine SHA-256：不可用。

`engine_io.json` 和 `engine_inspector.json` 仅记录“不可用/未执行”，不是成功
结果。`engine_sha256.txt` 明确记录 `NOT_GENERATED`。

## 10. 产物

- `builder_verbose.log`：TensorRT完整日志；
- `build_summary.json`：状态、首错与分类；
- `build_config.json`：精度、workspace及禁用策略；
- `environment.json`：系统、CUDA、TensorRT、GPU和依赖状态；
- `plugin_registry.json`：Creator注册表证据；
- `onnx_sha256.txt`：输入ONNX哈希；
- `engine_io.json`：未执行说明；
- `engine_inspector.json`：未执行说明；
- `engine_sha256.txt`：`NOT_GENERATED`；
- `ptv2_voxel_unique_plugin.dll`：本轮加载的插件库副本。

## 11. 下一步

下一阶段应先单独审计 TensorRT 11.1 对 IPluginV3 多输出 DDS 的 size tensor
约束，重点检查：

1. `declareSizeTensor(outputIndex, opt, upper)` 的 `outputIndex` 与三个插件输出的
   对应关系；
2. ONNX Parser生成的Plugin V3层中，哪个输出被标记为 data-dependent；
3. `voxel_count` 标量是否被TensorRT识别为运行时 size tensor；
4. 一个插件同时输出 `count`、`values[M]`、`inverse[N]` 时的合法DDS拓扑；
5. TensorRT 11.1是否要求size tensor直接作为网络/插件的特定输出。

在完成该审计前，不应重新构建Engine，也不应改动workspace、精度或容差。

