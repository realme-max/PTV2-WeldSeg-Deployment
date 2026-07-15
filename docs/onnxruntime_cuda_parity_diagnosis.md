# ONNX Runtime CUDA parity 异常慢诊断

## 1. 结论摘要

目标 ONNX 能创建以 `CUDAExecutionProvider` 为第一 Provider 的 Session，但固定单样本推理在约 29 分 25 秒内仍未返回。进程持续响应、GPU 持续 100%，因此不是完整 CPU fallback，也不是进程崩溃。

两次开启 `SessionOptions.enable_profiling=True` 的诊断均在 warmup 内复现异常慢：

- 第一次在 5 分钟诊断窗口后尝试控制信号终止；
- 第二次在 180 秒设置 `RunOptions.terminate=True`，但活动执行段不响应协作取消；
- 两次都无法执行 `session.end_profiling()`，ORT 创建的 profile JSON 均为 0 字节；
- 因此没有可验证的节点耗时事件，不能伪造“Top 20 实测耗时节点”。

当前证据支持“ORT CUDA 执行路径存在严重 kernel/Provider 分区效率问题”，但尚不能仅凭 0 字节 profile 将责任精确归到某一个节点。最高风险区域是动态 voxel 流程中的 `Unique/NonZero/ScatterElements` 及其 CUDA/CPU 数据搬运，以及两组全分辨率 pairwise-distance 标准算子链。

没有修改 ONNX graph、deployment、checkpoint、数据或容差；没有进入 TensorRT。

## 2. 验证对象

- ONNX：`artifacts/gcn_res_onnx/20260715_onnx_after_cdist_fp32_opset18/gcn_res_deploy_fp32_opset18.onnx`
- 输入：`artifacts/gcn_res_onnx/20260715_onnx_after_cdist_fp32_opset18/export_input.npz`
- ORT：`onnxruntime-gpu==1.26.0`
- PyTorch：`2.7.1+cu128`
- GPU：NVIDIA GeForce RTX 5060
- Provider 请求顺序：`CUDAExecutionProvider, CPUExecutionProvider`
- 实际 Session Provider：`CUDAExecutionProvider, CPUExecutionProvider`

新增只读 profiling 脚本：

- `scripts/profile_gcn_res_onnxruntime_cuda.py`

该脚本只读取既有 ONNX/NPZ，开启 ORT profiling，计划执行一次 warmup 和一次正式 inference。由于 warmup 未返回，正式 inference 没有启动。

## 3. 原 parity inference 终止前状态

记录时间：2026-07-15 15:13:41 +08:00。

| 项目 | 结果 |
|---|---:|
| PID | 35816 |
| inference/process elapsed | 1764.920 s（约 29 分 25 秒） |
| CPU time | 1755.781 s |
| process responding | True |
| process working set | 982.008 MiB |
| process private memory | 2349.352 MiB |
| GPU utilization | 100% |
| GPU memory utilization | 1% |
| GPU memory used | 2780 MiB / 8151 MiB |
| GPU power | 38.74 W |
| GPU temperature | 45°C |

向现有 PTY 发送 Ctrl+C 后，原 parity 进程成功退出；没有启动替代 parity Session，也没有删除任何 artifact。

## 4. 是否发生 CUDA EP fallback

没有发生“整个模型回退到 CPU”。证据：

```text
session.get_providers() = ['CUDAExecutionProvider', 'CPUExecutionProvider']
GPU utilization = 100%
GPU memory used ≈ 2.7 GiB
```

但 ORT Session 创建时明确警告：

```text
36 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider.
```

这说明图被分区后存在 CUDA/CPU 边界和数据搬运。由于 profile 未 flush，当前不能逐节点确认哪些节点实际落在 CPU；因此结论是“无完整 fallback，但存在节点级 Provider 分区/搬运”。

## 5. Profiling 执行结果

### 5.1 第一次 profiling

- 目录：`artifacts/gcn_res_onnxruntime_cuda/20260715_1515_cuda_profile_diagnosis/`
- CUDA Session 创建成功；
- warmup 超过 5 分钟未返回；
- 正式 inference 未启动；
- 多次 Ctrl+C/Ctrl+Break 未使活动 ORT 执行段正常返回；
- profile JSON：`ort_cuda_profile_2026-07-15_15-15-34_933.json`；
- profile JSON 大小：0 bytes。

### 5.2 协作取消 profiling

- 目录：`artifacts/gcn_res_onnxruntime_cuda/20260715_1523_cuda_profile_cooperative_timeout/`
- CUDA Session 创建成功；
- warmup 超过 180 秒；
- 设置 `RunOptions.terminate=True`；
- 活动执行段未响应终止标志；
- 正式 inference 未启动；
- profile JSON：`ort_cuda_profile_2026-07-15_15-24-10_238.json`；
- profile JSON 大小：0 bytes。

ORT profile 在 `end_profiling()` 时完成写入。由于执行卡在活动 CUDA 段且无法正常回到 Python `finally`，两个文件都没有事件，无法计算 warmup/正式 inference 的完整耗时，也无法从 profile 提取实测节点耗时。

## 6. Top 20 实测节点

不可用。原因不是缺少分析代码，而是两个 ORT profile JSON 均为 0 字节、没有任何 `Node` 事件。

以下表格是基于 ONNX shape inference 的“静态高工作量候选 Top 20”，不是实测耗时排名。Provider 和执行时间均不得伪造，统一标记为不可测。

| # | 节点名称 | op_type | 输出 shape / 元素数 | Provider | 实测耗时 |
|---:|---|---|---|---|---|
| 1 | `/model/ptb_0/Sub` | Sub | `[1,2048,2048,3]` / 12,582,912 | 不可测 | 不可测 |
| 2 | `/model/ptb_0/Mul` | Mul | `[1,2048,2048,3]` / 12,582,912 | 不可测 | 不可测 |
| 3 | `/model/ptb_9/Sub` | Sub | `[1,2048,2048,3]` / 12,582,912 | 不可测 | 不可测 |
| 4 | `/model/ptb_9/Mul` | Mul | `[1,2048,2048,3]` / 12,582,912 | 不可测 | 不可测 |
| 5 | `/model/ptb_0/ReduceSum` | ReduceSum | `[1,2048,2048]` / 4,194,304 | 不可测 | 不可测 |
| 6 | `/model/ptb_0/Sqrt` | Sqrt | `[1,2048,2048]` / 4,194,304 | 不可测 | 不可测 |
| 7 | `/model/ptb_9/ReduceSum` | ReduceSum | `[1,2048,2048]` / 4,194,304 | 不可测 | 不可测 |
| 8 | `/model/ptb_9/Sqrt` | Sqrt | `[1,2048,2048]` / 4,194,304 | 不可测 | 不可测 |
| 9 | `/model/ptb_0/gva/delta_mult/linear_1/MatMul` | MatMul | `[1,2048,16,48]` / 1,572,864 | 不可测 | 不可测 |
| 10 | `/model/ptb_0/gva/delta_mult/linear_2/MatMul` | MatMul | `[1,2048,16,48]` / 1,572,864 | 不可测 | 不可测 |
| 11 | `/model/ptb_0/gva/delta_bias/linear_1/MatMul` | MatMul | `[1,2048,16,48]` / 1,572,864 | 不可测 | 不可测 |
| 12 | `/model/ptb_0/gva/delta_bias/linear_2/MatMul` | MatMul | `[1,2048,16,48]` / 1,572,864 | 不可测 | 不可测 |
| 13 | `/model/ptb_0/gva/k/MatMul` | MatMul | `[1,2048,16,48]` / 1,572,864 | 不可测 | 不可测 |
| 14 | `/model/ptb_0/gva/v/MatMul` | MatMul | `[1,2048,16,48]` / 1,572,864 | 不可测 | 不可测 |
| 15 | `/model/ptb_0/gva/Sub` | Sub | `[1,2048,16,48]` / 1,572,864 | 不可测 | 不可测 |
| 16 | `/model/ptb_0/gva/Mul` | Mul | `[1,2048,16,48]` / 1,572,864 | 不可测 | 不可测 |
| 17 | `/model/ptb_0/gva/Mul_1` | Mul | `[1,2048,16,2,24]` / 1,572,864 | 不可测 | 不可测 |
| 18 | `/model/ptb_9/gva/delta_mult/linear_1/MatMul` | MatMul | `[1,2048,16,48]` / 1,572,864 | 不可测 | 不可测 |
| 19 | `/model/ptb_9/gva/delta_mult/linear_2/MatMul` | MatMul | `[1,2048,16,48]` / 1,572,864 | 不可测 | 不可测 |
| 20 | `/model/ptb_9/gva/delta_bias/linear_1/MatMul` | MatMul | `[1,2048,16,48]` / 1,572,864 | 不可测 | 不可测 |

## 7. 重点算子审计

图总节点数：2647。

| 算子/类别 | 数量 | 风险说明 |
|---|---:|---|
| pairwise distance `Sqrt` | 13 | 每次代表一组标准欧氏距离链；全分辨率 stage 产生平方级张量 |
| Sub | 51 | 含 `ptb_0/ptb_9` 的 `[1,2048,2048,3]` 广播差值 |
| Mul | 81 | 含 pairwise distance 平方与 attention 运算 |
| ReduceSum | 34 | 含 pairwise distance 坐标维归约 |
| TopK | 13 | KNN/插值近邻选择 |
| Gather | 100 | 动态索引和特征传播 |
| GatherElements | 18 | 动态索引 |
| GatherND | 8 | advanced indexing 导出结果 |
| ScatterElements | 20 | 四级 voxel pooling；动态 index/shape/reduction |
| Unique | 4 | 每级 voxel key 去重，输出维度数据依赖 |
| NonZero | 8 | 数据依赖 shape，可能造成 Provider 边界 |
| ReduceMin | 8 | voxel 边界/最小值聚合 |
| ReduceMax | 4 | voxel feature max pooling |
| Expand | 34 | 广播索引/特征张量 |
| MatMul | 110 | attention、MLP、GCN |
| Shape | 165 | 大量动态 shape 传播与控制张量 |

当前 ONNX 中没有单独的 `CDist` 节点。它已展开为：

```text
Unsqueeze → Sub → Mul → ReduceSum → Sqrt
```

全分辨率 `ptb_0` 和 `ptb_9` 各自形成约 48 MiB 的 FP32 差值/平方中间张量（12,582,912 个元素），随后生成约 16 MiB 的距离矩阵（4,194,304 个元素）。这会带来显著显存带宽和 kernel launch 压力，但仅凭张量规模仍不足以解释 30 分钟级时延。

四级 voxel pooling 共包含 20 个 `ScatterElements`、4 个 `Unique`、8 个 `NonZero`，且大量输出 shape 为数据依赖。结合 36 个 Memcpy 节点警告，这一动态索引/归约链是更需要优先定位的 Provider 分区风险。

## 8. 原因判断

当前可以确认：

1. CUDA EP 已激活，并非整个图 CPU fallback；
2. 图存在节点级 CUDA/CPU 分区和至少 36 个边界 Memcpy；
3. GPU 在长时间内保持 100%，进程持续响应；
4. ORT 的 `RunOptions.terminate` 在活动执行段内也无法及时生效；
5. 同一输入的 ORT CPU EP 能在短时间内返回，而 CUDA EP 超过 29 分钟未返回。

因此可以判断为 **ORT CUDA 执行路径的严重效率/调度问题**，而不是输入文件、Session Provider 顺序或完整 fallback 问题。

尚不能确认：

- 首个长耗时节点究竟是 pairwise-distance 链、动态 Scatter/Unique 链还是某个 CUDA kernel；
- 36 个 Memcpy 的逐节点 Provider 归属；
- Top 20 实测耗时，因为 profile 未生成事件。

## 9. 下一步建议

在不修改当前 ONNX 的前提下，下一步应优先获得“活动节点”证据，而不是再次盲等完整推理：

1. 使用 ONNX Runtime verbose node execution 日志或 Windows ETW，记录进入长时间执行前的最后一个节点；
2. 使用 NVIDIA Nsight Systems 对同一进程采样，识别持续占用 GPU 的 kernel 名称和对应 launch；
3. 在获得单独授权后，生成仅用于诊断的分阶段/截断 ONNX 副本，对 `tdb_1` voxel 链与 `ptb_0` pairwise-distance 链分别计时；原 ONNX 保持不变；
4. 导出 ORT optimized graph 并读取节点 Provider assignment，确认 36 个 Memcpy 的边界；
5. 在定位前，不应将当前 ORT CUDA 路径视为可部署性能基线，也不应进入 TensorRT。

## 10. 状态

```text
ONNXRUNTIME_CUDA_PARITY_NOT_COMPLETED
ORT_CUDA_PROFILING_NO_EVENTS_FLUSHED
ORT_CUDA_EXTREME_LATENCY_REPRODUCED
```
