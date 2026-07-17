# TensorRT Phase 7B：Latency Benchmark

更新时间：2026-07-17（Asia/Shanghai）

## 1. 结论

PyTorch CUDA deployment 和 TensorRT Strict FP32 Engine 均在固定 `weld_65`
输入上完成 `100` 次 warmup 和 `1000` 次正式测量：

```text
TENSORRT_LATENCY_BENCHMARK_COMPLETED
```

当前 Strict FP32 TensorRT 图没有取得延迟加速。TensorRT pure enqueue mean 为
`37.0643 ms`，PyTorch CUDA pure forward mean 为 `20.7122 ms`；定义
`speedup = PyTorch latency / TensorRT latency` 时，pure speedup 为 `0.5588x`，
端到端 speedup 为 `0.5229x`。本阶段只记录真实结果，没有修改 Engine、图、Plugin、
tactic 或精度配置。

## 2. 脚本与产物

新增脚本：

- `scripts/benchmark_gcn_res_pytorch_latency.py`
- `scripts/benchmark_gcn_res_tensorrt_latency.py`

产物目录：

`artifacts/gcn_res_tensorrt/20260717_122000_phase7b_latency_benchmark/`

主要文件：

- `pytorch_latency.json`
- `tensorrt_latency.json`
- `memory_summary.json`
- `environment.json`
- `benchmark_summary.json`

执行命令：

```powershell
$run = "E:\GRP-PTv2\artifacts\gcn_res_tensorrt\20260717_122000_phase7b_latency_benchmark"

E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe `
  E:\GRP-PTv2\scripts\benchmark_gcn_res_pytorch_latency.py `
  --run-dir $run

E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe `
  E:\GRP-PTv2\scripts\benchmark_gcn_res_tensorrt_latency.py `
  --run-dir $run
```

两个框架在独立 Python 进程中运行，避免 PyTorch caching allocator 与 TensorRT
Runtime/context 的内存统计互相污染。

## 3. 固定测试条件

- 样本：test split index 0，`weld_65`
- seed：`42`
- points：FP32 `[1,2048,4]`
- adj：FP32 `[1,2048,2048]`
- logits：FP32 `[1,2048,2]`
- 邻接：`k=6`
- warmup：`100`
- benchmark：`1000`
- GPU：NVIDIA GeForce RTX 5060，SM `12.0`
- TensorRT：`11.1.0.106`
- PyTorch：`2.7.1+cu128`
- CUDA Runtime：`12.8`
- NVIDIA driver：`610.74`

输入哈希严格复用 Phase 7A：

- points：`b9f7ace14e74b05b076fa5d0f5e1226a0c1e84530336d785e8c3391f90a57063`
- adj：`a543b9f287b8bbca844bc36b5af72bec70e964095a35e7c68d8279a40e31cf12`
- sample indices：`845a63985495a98943c872bde19111a5d2020e2bc05e40bdf3ffd48f525ea4c7`

正式 Engine：

- 路径：`artifacts/gcn_res_tensorrt/20260716_224643_531592_strict_fp32/strict_fp32.plan`
- SHA-256：`b76563580089bbc0684e7a8e1edda6028c2d89fd91397c22b4efbd89ecd7fc2c`
- TF32：disabled
- FP16：disabled
- INT8：disabled
- VoxelUnique runtime instances：`4`

## 4. 测量口径

### PyTorch CUDA

points 和 adj 在 warmup 前复制到 `cuda:0`，正式测试只统计 deployment wrapper 的
model forward。每次测量前执行一次 `torch.cuda.synchronize()`，随后用
`time.perf_counter()` 开始计时；forward 后再次 synchronize，再停止计时。

PyTorch 显式使用：

- `torch.backends.cuda.matmul.allow_tf32 = False`
- `torch.backends.cudnn.allow_tf32 = False`
- `torch.set_float32_matmul_precision("highest")`
- inference mode

### TensorRT pure inference

points 和 adj 预先常驻 device buffer。CUDA Event 在同一 execution stream 上包围单次
`execute_async_v3/enqueueV3`，不包含 H2D、D2H 或 CPU wall-clock 提交时间。

### TensorRT end-to-end

使用 pageable NumPy host buffer，每次统计：

1. points H2D；
2. adj H2D；
3. `enqueueV3`；
4. logits D2H；
5. stream synchronize。

端到端使用 `time.perf_counter()` wall clock。该结果对应当前脚本的 pageable-host
数据路径，不代表后续 pinned memory 或 C++ pipeline 的最终性能。

## 5. Latency 结果

单位均为毫秒。

| Runtime | Mean | Median/P50 | P95 | P99 | Std | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch CUDA forward | `20.7122` | `20.4034` | `23.1942` | `24.9172` | `1.2432` | `18.6156` | `27.8567` |
| TensorRT pure enqueue | `37.0643` | `37.0274` | `38.2124` | `38.8241` | `0.6639` | `35.3648` | `40.1118` |
| TensorRT end-to-end | `39.6064` | `39.5497` | `40.6932` | `41.4386` | `0.6484` | `37.9063` | `44.4197` |

Speedup 定义：

`PyTorch mean latency / TensorRT mean latency`

| Comparison | Speedup | Interpretation |
|---|---:|---|
| TensorRT pure vs PyTorch | `0.558817x` | TensorRT 延迟高 `78.95%` |
| TensorRT E2E vs PyTorch | `0.522950x` | TensorRT 延迟高 `91.22%` |

这只说明当前 Strict FP32、动态 voxel/DDS、VoxelUnique Plugin 和当前 TensorRT tactic
组合下的实测性能。Phase 7B 没有继续做逐层 profiling 或性能归因。

## 6. 显存结果

### PyTorch allocator

| Metric | Bytes | MiB |
|---|---:|---:|
| after setup allocated | `41,714,688` | `39.78` |
| after warmup allocated | `50,250,752` | `47.92` |
| benchmark peak allocated | `170,183,680` | `162.30` |
| benchmark peak reserved | `190,840,832` | `182.00` |

PyTorch 在 warmup 后调用 `reset_peak_memory_stats()`，因此 peak 统计覆盖正式 1000 次
benchmark，并保留当时已存在的模型和输入 allocation。

### TensorRT Runtime

- Engine 文件大小：`30,346,644` bytes（`28.94 MiB`）
- deserialize 后观测增量：`33,554,432` bytes（`32 MiB`）
- execution context 后观测增量：`367,001,600` bytes（`350 MiB`）
- device buffers 后及测试期间最大观测增量：`383,778,816` bytes（`366 MiB`）

TensorRT 数据来自隔离进程内多个生命周期边界的 `cudaMemGetInfo` 快照，增量以
`cudaSetDevice` 后的进程基线为参照。`cudaMemGetInfo` 是全局 free-memory 采样，因此
`366 MiB` 是本次生命周期快照的最大观测增量，不是 kernel 内部瞬时 allocator peak。

## 7. 完整性与停止线

执行前后以下对象哈希一致：

- Engine：`b76563580089bbc0684e7a8e1edda6028c2d89fd91397c22b4efbd89ecd7fc2c`
- ONNX：`f71a585ded3f348a193e37395a124ab8a425ff72fc09ae92114d41872c460b98`
- Plugin DLL：`60c48c400cc926f6aa183de62298a203916b1add6393a3483a332624da39d0ab`
- checkpoint：`311bddf3607d76e6b7ded450b8419bf6ae98f34f50578608b3e6a1c1c3e58d21`
- VoxelUnique Plugin 源码 manifest：一致

TensorRT ErrorRecorder errors 为 `0`，输出 shape/dtype 正确且全部有限；`pip check`
为 `No broken requirements found.`。

本轮没有执行 accuracy regression、FP16、INT8、C++ 部署、Engine rebuild、图或 Plugin
优化。由于 TensorRT pure inference 慢于 PyTorch，本阶段只记录该事实并停止。

```text
TENSORRT_LATENCY_BENCHMARK_COMPLETED
```
