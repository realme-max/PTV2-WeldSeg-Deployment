# TensorRT Phase 7A：Engine Benchmark Preparation

更新时间：2026-07-17（Asia/Shanghai）

## 1. 结论

正式 Strict FP32 Engine 已完成只读元数据检查和一次 TensorRT Runtime smoke
inference。Engine 反序列化、execution context 创建、固定 I/O 合同、4 个
VoxelUnique IPluginV3 实例、固定输入哈希、`enqueueV3`、有限输出和 ErrorRecorder
均通过：

```text
ENGINE_BENCHMARK_PREPARATION_COMPLETED
```

本阶段没有重建或修改 Engine，没有修改 ONNX、Plugin、checkpoint 或 deployment
模型；没有执行 latency、throughput、显存、parity、accuracy、FP16 或 INT8 测试。

## 2. 新增入口与产物

新增脚本：

`scripts/smoke_test_gcn_res_tensorrt_engine.py`

成功产物目录：

`artifacts/gcn_res_tensorrt/20260717_115222_307208_phase7a_engine_prepare/`

包含：

- `engine_metadata.json`
- `environment.json`
- `smoke_test_result.json`
- `phase7a_report.md`

执行命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe `
  E:\GRP-PTv2\scripts\smoke_test_gcn_res_tensorrt_engine.py
```

实现检查期间的第一次执行在 enqueue 前因新脚本误读 I/O 元数据字段
`shape`（实际字段为 `engine_shape`）而停止，产物保存在：

`artifacts/gcn_res_tensorrt/20260717_115142_089353_phase7a_engine_prepare/`

该次没有执行推理，也没有修改受保护对象。仅修正新脚本的元数据字段读取后，使用新
run ID 从头完成上述正式 smoke。

## 3. 正式 Engine 元数据

- Engine：`artifacts/gcn_res_tensorrt/20260716_224643_531592_strict_fp32/strict_fp32.plan`
- Engine SHA-256：`b76563580089bbc0684e7a8e1edda6028c2d89fd91397c22b4efbd89ecd7fc2c`
- 文件大小：`30,346,644` bytes
- TensorRT：`11.1.0.106`
- CUDA Runtime：`12.8`（Runtime API `12080`）
- GPU：NVIDIA GeForce RTX 5060，SM `12.0`
- 精度：Strict FP32
- TF32：disabled
- FP16：disabled
- INT8：disabled
- Inspector layer count：`570`
- Inspector GEMM count：`86`
- TF32 GEMM count：`0`
- Optimization profiles：`1`

固定 profile 0：

| Tensor | Min | Opt | Max |
|---|---|---|---|
| points | `[1,2048,4]` | `[1,2048,4]` | `[1,2048,4]` |
| adj | `[1,2048,2048]` | `[1,2048,2048]` | `[1,2048,2048]` |

Engine I/O：

| Tensor | Mode | Dtype | Shape | Location |
|---|---|---|---|---|
| points | INPUT | FLOAT | `[1,2048,4]` | DEVICE |
| adj | INPUT | FLOAT | `[1,2048,2048]` | DEVICE |
| logits | OUTPUT | FLOAT | `[1,2048,2]` | DEVICE |

## 4. VoxelUnique Plugin 检查

Plugin Creator：

- name：`VoxelUnique`
- version：`1`
- namespace：`com.tensorrt.ptv2`
- Python interface：`IPluginCreatorV3One`

反序列化时创建 `4` 个 runtime plugin instance。Inspector 中恰好存在以下 4 个
`LayerType=PluginV3`、`PluginType=VoxelUnique` 层：

1. `/model/tdb_1/Unique`
2. `/model/tdb_2/Unique`
3. `/model/tdb_3/Unique`
4. `/model/tdb_4/Unique`

Inspector 中另外可见的 4 个 `DeviceToShapeHost` 层是 voxel count size tensor 的
数据搬运层，不计作 VoxelUnique Plugin 实例。

## 5. 固定 smoke 输入

没有生成随机输入。脚本使用固定 test split 的第一个样本 `weld_65`，严格复用历史
评估入口的 seed `42`、2048 点采样、XYZ 归一化、常量类别通道和 `k=6` 邻接构建
逻辑。

输入哈希与 Phase 6 保存的 `per_sample_results.json` 完全一致：

- points SHA-256：`b9f7ace14e74b05b076fa5d0f5e1226a0c1e84530336d785e8c3391f90a57063`
- adj SHA-256：`a543b9f287b8bbca844bc36b5af72bec70e964095a35e7c68d8279a40e31cf12`
- sample indices SHA-256：`845a63985495a98943c872bde19111a5d2020e2bc05e40bdf3ffd48f525ea4c7`

## 6. 一次 Runtime smoke 结果

| Check | Result |
|---|---|
| deserialize | PASS |
| context creation | PASS |
| enqueueV3 | PASS |
| enqueue count | `1` |
| output shape | `[1,2048,2]` |
| output dtype | `float32` |
| output finite | `true` |
| ErrorRecorder errors | `0` |

Logits 仅记录基础统计，不用于 parity 或 accuracy：

- min：`-7.554868698120117`
- max：`9.246336936950684`
- mean：`0.07580175250768661`
- std：`3.218564033508301`

显式 buffer 大小：

- points：`32,768` bytes
- adj：`16,777,216` bytes
- logits：`16,384` bytes

上述大小是张量合同推导值，不是显存 benchmark。

## 7. 环境与完整性边界

- Python：`3.11.8`
- PyTorch：`2.7.1+cu128`
- PyTorch CUDA Runtime：`12.8`
- cuDNN：`9.7.1`
- TensorRT：`11.1.0.106`
- NVIDIA driver：`610.74`
- GPU：NVIDIA GeForce RTX 5060，SM `12.0`
- `pip check`：`No broken requirements found.`

执行前后以下哈希均未变化：

- Engine：`b76563580089bbc0684e7a8e1edda6028c2d89fd91397c22b4efbd89ecd7fc2c`
- ONNX：`f71a585ded3f348a193e37395a124ab8a425ff72fc09ae92114d41872c460b98`
- Plugin DLL：`60c48c400cc926f6aa183de62298a203916b1add6393a3483a332624da39d0ab`
- checkpoint：`311bddf3607d76e6b7ded450b8419bf6ae98f34f50578608b3e6a1c1c3e58d21`
- 4 个 VoxelUnique Plugin 源文件：执行前后 manifest 完全一致。

## 8. 后续 Benchmark 入口

Phase 7A 已建立后续 benchmark 可复用的正式 Engine、固定 I/O、固定样本、输入哈希、
Plugin 注册、Runtime/ErrorRecorder 和显式 CUDA buffer/stream 入口。后续只有在获得明确
授权后，才能在此基础上增加 warmup、重复次数、CUDA event 计时、吞吐或显存统计。

```text
ENGINE_BENCHMARK_PREPARATION_COMPLETED
```
