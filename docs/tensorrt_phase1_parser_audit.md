# TensorRT Phase 1：GCN_res ONNX Parser 审计

## 1. 结论摘要

审计日期：2026-07-15（Asia/Shanghai）

TensorRT 11.1.0.106 Python ONNX Parser 已对现有固定 ONNX 完成一次只读解析。Parser 返回失败，共报告 4 条错误；四条均为四级 voxel 下采样中的 ONNX `Unique` 节点不受支持。

```text
TENSORRT_ONNX_PARSER_FAILED
```

首个确定阻塞点：

```text
node_index=153
node_name=/model/tdb_1/Unique
operator=Unique
error_code=ErrorCode.UNSUPPORTED_NODE
source=onnxOpCheckers.cpp:1227 checkUnique
```

当前不能进入 FP32 Engine build。本阶段没有创建 builder config，没有调用 Engine build API，没有运行 `trtexec --onnx`，也没有执行 inference、FP16 或 benchmark。

## 2. 新增脚本与执行命令

新增只读脚本：

`scripts/audit_gcn_res_tensorrt_parser.py`

执行命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe `
  E:\GRP-PTv2\scripts\audit_gcn_res_tensorrt_parser.py `
  --onnx E:\GRP-PTv2\artifacts\gcn_res_onnx\20260715_onnx_after_cdist_fp32_opset18\gcn_res_deploy_fp32_opset18.onnx `
  --tensorrt-root D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106 `
  --cuda-root "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
```

脚本通过子进程捕获 TensorRT 原生 VERBOSE stdout/stderr，确保完整日志写入 `parser_verbose.log`。Worker 中只创建 `Builder`、`Network`、`OnnxParser` 并调用 `parser.parse_from_file()`。

## 3. 环境信息

| 项目 | 实测结果 |
|---|---|
| Windows | Windows 10 10.0.19045，64-bit |
| Python | 3.11.8 |
| TensorRT Python | 11.1.0.106 |
| TensorRT SDK | 11.1.0.106 Windows x64 CUDA 12 |
| CUDA Toolkit | 12.8 Update 1，nvcc V12.8.93 |
| PyTorch | 2.7.1+cu128 |
| PyTorch CUDA Runtime | 12.8 |
| cuDNN | 9.7.1 / 90701 |
| GPU | NVIDIA GeForce RTX 5060 |
| Compute capability | 12.0 |

TensorRT 11.1 已移除 `NetworkDefinitionCreationFlag.EXPLICIT_BATCH`；显式 batch 已成为网络默认语义。因此脚本先检测该枚举，在当前版本使用：

```python
network = builder.create_network(0)
```

记录值：

```text
explicit_batch_api=implicit_explicit_batch_default_in_tensorrt_11
network_creation_flags=0
```

这只是 TensorRT 11.1 API 兼容处理，没有修改 ONNX shape 或数学语义。

## 4. ONNX 信息

| 项目 | 值 |
|---|---|
| 路径 | `artifacts/gcn_res_onnx/20260715_onnx_after_cdist_fp32_opset18/gcn_res_deploy_fp32_opset18.onnx` |
| 文件大小 | 25,135,572 bytes |
| SHA-256 | `20aa7ba21a52c6497e0ce10676edae599def203bbddd4ca063b7abccdeeb5198` |
| ONNX IR | 8 |
| Opset | 18 |
| 节点数 | 2,647 |

重点算子数量：

| 算子 | 数量 | 本轮 Parser 状态 |
|---|---:|---|
| Unique | 4 | **明确不支持，4 个错误** |
| NonZero | 8 | 未报告独立 parser error，尚不能证明可 build |
| ScatterElements | 20 | 未报告独立 parser error，尚不能证明可 build |
| GatherND | 8 | 未报告独立 parser error，尚不能证明可 build |
| GatherElements | 18 | 未报告独立 parser error，尚不能证明可 build |
| TopK | 13 | 未报告独立 parser error，尚不能证明可 build |
| ReduceMin | 8 | 未报告独立 parser error，尚不能证明可 build |
| ReduceMax | 4 | 未报告独立 parser error，尚不能证明可 build |
| Shape | 165 | 未报告独立 parser error，尚不能证明可 build |
| Expand | 34 | 未报告独立 parser error，尚不能证明可 build |

“未报告 parser error”不能等同于“TensorRT Builder 完全支持”。当前图没有成功解析到完整 network output，Builder 也没有运行。

## 5. Parser 结果

解析耗时所在命令总进程约 4.5 秒，Parser 正常返回失败状态，没有卡死。

```text
parser_success=false
num_parser_errors=4
num_layers=245
num_inputs=2
num_outputs=0
engine_build_attempted=false
inference_attempted=false
```

已识别输入：

| 名称 | TensorRT dtype | Shape |
|---|---|---|
| points | DataType.FLOAT | `[1, 2048, 4]` |
| adj | DataType.FLOAT | `[1, 2048, 2048]` |

输入名称、类型和 shape 符合预期。由于 parser 在 `Unique` 处失败，`logits` 没有成为 network output：

```text
num_outputs=0
io_matches_expected=false
```

这不是输出被常量固化，而是网络没有解析完成。

## 6. 完整 Parser errors

混合错误分类结果：

```text
UNSUPPORTED_OPERATOR=4
UNSUPPORTED_ATTRIBUTE=0
DYNAMIC_SHAPE_ERROR=0
SHAPE_TENSOR_ERROR=0
DATA_DEPENDENT_SHAPE=0
PLUGIN_REQUIRED=0
UNKNOWN=0
```

| # | ONNX node index | Node name | Operator | Error code | 分类 |
|---:|---:|---|---|---|---|
| 0 | 153 | `/model/tdb_1/Unique` | Unique | `UNSUPPORTED_NODE` | `UNSUPPORTED_OPERATOR` |
| 1 | 495 | `/model/tdb_2/Unique` | Unique | `UNSUPPORTED_NODE` | `UNSUPPORTED_OPERATOR` |
| 2 | 869 | `/model/tdb_3/Unique` | Unique | `UNSUPPORTED_NODE` | `UNSUPPORTED_OPERATOR` |
| 3 | 1243 | `/model/tdb_4/Unique` | Unique | `UNSUPPORTED_NODE` | `UNSUPPORTED_OPERATOR` |

四条错误均来自：

```text
source_file=onnxOpCheckers.cpp
source_function=checkUnique
source_line=1227
description=false
```

这四个节点对应部署 voxel pooling 四级 hierarchy 中的 key 去重。`Unique` 用于生成 materialized voxel 集合和 point-to-voxel inverse mapping；它不是可直接删除的冗余节点。

## 7. `trtexec` parser-only 审计

执行并保存了完整：

```powershell
trtexec --help
```

TensorRT 11.1.0.106 `trtexec` 没有提供真正的 parser-only 参数。它提供 `--skipInference`，但帮助文本明确说明：

```text
Exit after the engine has been built and skip inference perf measurement
```

因此本阶段没有对 ONNX 运行 `trtexec --onnx ... --skipInference`，因为那会违反“不要构建 Engine”的约束。

```text
trtexec help return code=0
parser_only_available=false
skip_inference_available=true
onnx_invocation_attempted=false
```

## 8. 审计产物

运行目录：

`artifacts/gcn_res_tensorrt/20260715_164224_298664_parser_audit/`

| 文件 | 用途 |
|---|---|
| `parser_verbose.log` | TensorRT VERBOSE 完整解析日志，421,470 bytes |
| `parser_errors.json` | 4 条结构化 parser error |
| `parser_summary.json` | Parser、I/O、ONNX算子和 trtexec 审计摘要 |
| `environment.json` | TensorRT/CUDA/Python/GPU/ONNX环境证据 |
| `trtexec_help.log` | TensorRT 11.1 trtexec 完整帮助日志 |

## 9. 是否可以进入 FP32 Engine build

**不可以。**

原因：`parser_success=false`，四个 `Unique` 节点是确定的 parser 阻塞点，网络没有完整输出。此时运行 Builder 不会形成可信的固定 FP32 Engine 可行性结论。

## 10. 下一步建议

下一阶段应先单独审计四个 `Unique` 节点的完整 ONNX 语义：

1. 检查 `sorted`、`axis` 属性和四个输出中实际被消费的输出；
2. 明确 voxel key 的 dtype、排序规则、inverse mapping 和 counts 依赖；
3. 评估是否存在由 TensorRT 标准层可表达且经过逐层数值验证的等价实现；
4. 若标准层无法表达，再单独评估 TensorRT plugin，不能在未审计语义前删除或绕过 `Unique`；
5. 只有新的部署实现完成独立等价验证后，才允许重新导出 ONNX 并重跑 Parser。

本阶段到此停止：

```text
TENSORRT_ONNX_PARSER_FAILED
FIRST_BLOCKING_OPERATOR=Unique
FIRST_BLOCKING_NODE=/model/tdb_1/Unique
```

