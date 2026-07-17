# TensorRT Phase 8B：VoxelUnique CUB 隔离优化

更新时间：2026-07-17（Asia/Shanghai）

## 1. 结论

已完成独立、可回退的 `VoxelUniqueCub` 实验插件，实现、正确性矩阵和隔离性能测试均通过：

```text
VOXEL_UNIQUE_CUB_CORRECTNESS_PASSED
VOXEL_UNIQUE_CUB_BENCHMARK_PASSED
VOXELUNIQUE_CUB_ISOLATED_OPTIMIZATION_COMPLETED
```

真实 `weld_65/tdb_1` key 的 Plugin CUDA 时间由 `28.847813 ms` 降至 `0.103627 ms`，约 `278.38x` 加速，满足 `<5 ms` 和至少 `5x` 两项门槛。

本轮结果仍是隔离实验：没有替换正式 Plugin DLL，没有修改正式 ONNX，没有重建或执行正式 GCN_res Engine。

## 2. 版本隔离

新增实验实现：

- `deployment/tensorrt_voxel_unique_plugin_cub/VoxelUniqueCubPlugin.h`
- `deployment/tensorrt_voxel_unique_plugin_cub/VoxelUniqueCubPlugin.cpp`
- `deployment/tensorrt_voxel_unique_plugin_cub/VoxelUniqueCubPlugin.cu`
- `deployment/tensorrt_voxel_unique_plugin_cub/VoxelUniqueCubPluginLibrary.cpp`
- `deployment/tensorrt_voxel_unique_plugin_cub/CMakeLists.txt`

实验 Plugin 标识：

| 字段 | 值 |
|---|---|
| name | `VoxelUniqueCub` |
| version | `1` |
| namespace | `com.tensorrt.ptv2.experimental` |
| CUDA architecture | `SM 12.0` |
| 支持 N | `1 <= N <= 2048` |

正式图中的 `com.tensorrt.ptv2::VoxelUnique` 和基线目录均未改变。

## 3. CUDA 算法

输入为 `INT64 keys[N]`，输出合同保持为：

- `voxel_count`: `INT32` scalar；
- `unique_values`: `INT64[M]`；
- `inverse_indices`: `INT64[N]`。

数据流：

1. CUDA kernel 生成 `original_indices = [0, 1, ..., N-1]`；
2. `cub::DeviceRadixSort::SortPairs` 对 `(signed INT64 key, INT32 original index)` 排序；
3. CUDA kernel 标记相邻 key 的 run boundary；
4. `cub::DeviceScan::InclusiveSum` 生成一基 run id，写输出时减一得到 `[0, M-1]`；
5. CUDA kernel 同时写入 `unique_values`、按原始位置恢复的 `inverse_indices` 和 `voxel_count`。

实现包含 3 个显式辅助 kernel，加上 CUB radix sort 和 scan 的内部 kernel。`enqueue()` 正常路径没有 `cudaMalloc`、`cudaFree`、host round-trip 或 CUDA 同步，只在末尾使用非同步的 `cudaPeekAtLastError()`；CUB 调用返回码均被检查。

第一次内部测试发现 exclusive scan 直接产生的重复 key inverse id 会偏大。该未通过结果保留在 `correctness_run_attempt1_failed.log`；实现随后改为 inclusive scan 并显式转为零基 id，完整测试从头重跑后通过。

## 4. signed INT64 排序语义

使用 CUDA Toolkit 12.8 的 `cub::DeviceRadixSort::SortPairs`，key 类型直接为 `int64_t`。CUB 本地头文件 `cub/device/device_radix_sort.cuh` 明确说明：对 signed integral key 会翻转 sign bit 后执行 radix sort，因此其升序顺序等价于有符号整数比较，而不是按原始 unsigned bit pattern 排序。

本轮测试覆盖负数、0、正数、`INT64_MIN`、`INT64_MAX` 以及三者混合和重复。这些 case 的 Plugin 结果与 `std::sort<int64_t>` CPU reference、`torch.unique(sorted=True, return_inverse=True)` 均逐元素完全一致。

## 5. TensorRT workspace

`getWorkspaceSize()` 为 profile 最大值 `N=2048` 返回 `49,920 bytes`。所有分区按 256 bytes 对齐：

| 分区 | offset | bytes |
|---|---:|---:|
| sorted keys | 0 | 16,384 |
| original indices | 16,384 | 8,192 |
| sorted original indices | 24,576 | 8,192 |
| boundary flags | 32,768 | 8,192 |
| unique ids | 40,960 | 8,192 |
| shared CUB temporary | 49,152 | 767 |
| aligned total | — | 49,920 |

CUB 查询结果为 radix sort `1 byte`、inclusive scan `767 bytes`；两者按执行时序复用同一临时分区。完整证据见 `workspace_layout.json`。

另外对每个整数 `N=1..2048` 执行了 host-side layout 查询，最大 workspace 确实出现在 `N=2048`，没有发现中间 N 的临时空间超过 profile 最大值；结果见 `workspace_range_audit.txt`。

## 6. 独立正确性验证

新增测试目录：`tests/tensorrt_voxel_unique_cub_correctness/`。独立诊断 ONNX 只包含一个动态 N `VoxelUniqueCub` 节点，TensorRT Parser、FP32 Builder、Engine deserialize 和 runtime 均通过。

测试共 21 个 case：随机 `N=4/8/32/2048` 每种 3 组，以及 N=1、all same、all unique、already sorted、reversed、repeated groups、mixed signs、INT64 extremes 和 mixed min/max。

| 项目 | 结果 |
|---|---|
| case | `21/21 PASS` |
| `voxel_count` | 完全一致 |
| `unique_values` | bitwise 一致 |
| `inverse_indices` | 完全一致 |
| runtime output shape M | 完全一致 |
| C++ CPU reference vs Torch | 完全一致 |
| Plugin vs Torch | 完全一致 |

## 7. 隔离性能

复用 Phase 8A 相同输入和相同 CUDA Event 边界，执行 100 次 warmup、1,000 次正式测量。

冻结输入：

- random seed 42、范围 `[0,512)`：SHA-256 `628b105b98123a300108114a014fdc2b04ef229e5f1015c1ebec89946bd99f92`；
- `weld_65/tdb_1`：SHA-256 `9b22f88e82bd59cfd0411e417ed487fae8a9a8959f5dec5a5694596cce80dae8`，`M=397`。

| 输入 | 基线 kernel mean | CUB kernel mean | CUB P50 | CUB P95 | speedup |
|---|---:|---:|---:|---:|---:|
| random (`M=499`) | `37.258966 ms` | `0.117754 ms` | `0.106224 ms` | `0.183176 ms` | `316.41x` |
| weld_65 (`M=397`) | `28.847813 ms` | `0.103627 ms` | `0.095040 ms` | `0.147613 ms` | `278.38x` |

`weld_65` 的含 H2D、Plugin、D2H 的 total mean 为 `0.256374 ms`。性能测试结束后再次校验 count/values/inverse/runtime shape，结果仍全部正确。

## 8. 环境与产物

环境：Windows 10 `10.0.19045`、Python `3.11.8`、PyTorch `2.7.1+cu128`、CUDA Toolkit `12.8.93`、TensorRT `11.1.0.106`、RTX 5060 SM `12.0`。

产物目录：`artifacts/gcn_res_tensorrt/20260717_151544_915303_phase8b_voxelunique_cub/`。

主要文件：

- `VoxelUniqueCubPlugin.dll`
- `voxel_unique_cub.onnx`
- `voxel_unique_cub.plan`
- `environment.json`
- `build_config.json`
- `workspace_layout.json`
- `correctness_report.json`
- `per_case_comparison.json`
- `torch_reference_validation.json`
- `baseline_vs_cub_latency.json`
- `cub_kernel_profile.json`
- `benchmark_summary.md`
- `build.log`
- `correctness_run.log`
- `benchmark_run.log`

## 9. 完整性与停止线

执行后复核以下对象 SHA-256 与冻结值一致：

- 正式 ONNX：`f71a585ded3f348a193e37395a124ab8a425ff72fc09ae92114d41872c460b98`；
- Strict FP32 Engine：`b76563580089bbc0684e7a8e1edda6028c2d89fd91397c22b4efbd89ecd7fc2c`；
- baseline Plugin DLL：`60c48c400cc926f6aa183de62298a203916b1add6393a3483a332624da39d0ab`；
- checkpoint：`311bddf3607d76e6b7ded450b8419bf6ae98f34f50578608b3e6a1c1c3e58d21`。

`pip check` 为 `No broken requirements found.`。本轮没有执行正式 Engine rebuild、18 样本 parity、全模型 latency/profile、FP16、INT8 或 C++ 应用部署。

下一阶段如获授权，应进入 Phase 8C：将实验 Plugin 接入新的派生 ONNX 和 candidate Strict FP32 Engine，再完整执行 Parser、Builder、Runtime、18 样本 parity 及 Phase 7B/7C 性能回归。

```text
VOXELUNIQUE_CUB_ISOLATED_OPTIMIZATION_COMPLETED
```
