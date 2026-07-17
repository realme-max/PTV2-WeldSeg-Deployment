# VoxelUnique CUDA Kernel Baseline Analysis

更新时间：2026-07-17（Asia/Shanghai）

## 1. 分析范围

本报告对应 TensorRT Phase 8A 的 baseline-only 范围。已完成：

- 当前 VoxelUnique CUDA 源码审计；
- 单 Plugin Engine 隔离 profiling；
- 随机 N=2048 key 与真实 `weld_65/tdb_1` key 各1000次 CUDA Event测量；
- 当前实现正确性证据复核；
- radix sort、hash和fusion方案评估。

没有修改、重编译或替换 CUDA Plugin，没有重建正式 Engine，也没有执行所谓“优化后”
性能或回归测试。

## 2. 当前实现位置与接口

Plugin 头文件：

`tests/tensorrt_voxel_unique_correctness/VoxelUniqueCorrectnessPlugin.h`

CUDA 实现：

`tests/tensorrt_voxel_unique_correctness/VoxelUniqueCorrectnessPlugin.cu`

正式 DLL 包装与注册：

`deployment/tensorrt_voxel_unique_plugin/VoxelUniquePluginLibrary.cpp`

Plugin 合同不变：

- input：`keys INT64[N]`
- output 0：`voxel_count INT32[]`
- output 1：`unique_values INT64[M]`
- output 2：`inverse_indices INT64[N]`
- `unique_values` 必须按 signed INT64 升序，即 `sorted=True`
- `inverse_indices[i]` 必须指向 `keys[i]` 在排序后 unique values 中的位置
- M 是运行时 size tensor。

## 3. 当前算法

源码已经明确标注为：

`VoxelUniqueCorrectnessPlugin.cu:15`

```cpp
// Intentionally unoptimized Phase 2C.1 implementation. One CUDA thread
// performs serial deduplication, insertion sort, and inverse lookup.
```

`voxelUniqueSerialKernel` 位于第17行。算法分为三段：

1. **串行去重**（第27行开始）
   - 依次读取每个 `keys[i]`；
   - 线性扫描当前 `uniqueValues[0:uniqueCount]`；
   - 未找到时追加到 uniqueValues。
2. **串行插入排序**（第44行开始）
   - 对首次出现顺序的 M 个 unique values 执行 insertion sort；
   - 得到 signed INT64 升序输出。
3. **串行 inverse lookup**（第56行开始）
   - 对每个 input key 再次线性扫描排序后的 M 个值；
   - 写入 inverse index。

`enqueue()` 位于第178行；实际 launch 为第196行：

```cpp
voxelUniqueSerialKernel<<<1, 1, 0, stream>>>(...)
```

因此当前实现为一个 block、一个 thread、一个 kernel launch。

## 4. 复杂度与执行特征

设输入点数为 N、唯一 key 数为 M：

- 串行去重：`O(N·M)`，M接近N时退化为 `O(N²)`；
- 插入排序：平均/最坏 `O(M²)`；
- inverse lookup：`O(N·M)`；
- 总体：`O(N·M + M²)`，M与N同阶时为 `O(N²)`；
- 输出内存：`O(N)`；
- 并行度：1个CUDA thread。

其他实现特征：

| 检查项 | 结论 |
|---|---|
| CUDA kernel数量 | 每次 enqueue 1个 |
| atomic | 无 |
| global synchronization | Plugin内部无；enqueue只调用 `cudaPeekAtLastError()` |
| CPU fallback | 无 |
| CPU unique/sort | 无 |
| CUDA stream | 使用TensorRT传入stream |
| workspace | 当前kernel不使用workspace |
| memory allocation in enqueue | 无 |
| memory access | 单线程反复线性读取global keys/uniqueValues并写输出 |

Plugin 内部没有 `cudaDeviceSynchronize` 或 `cudaStreamSynchronize`。但是
`voxel_count` 同时承担 TensorRT DDS size tensor，完整 Engine 仍存在size-tensor相关的
device/host shape管理；Phase 7C已证明这些普通shape-copy只占很小比例。

## 5. 隔离 baseline 方法

测量复用既有单Plugin correctness Engine：

`artifacts/tensorrt_plugin_prototype/20260715_203305_357432_correctness/voxel_unique_correctness.plan`

SHA-256：

`e7939a4ba0f4ddf40c9efd3eb4b9188d6c1582acefc88c22f105a1a230a25b75`

该Engine只有：

```text
voxel_key -> VoxelUnique -> voxel_count / unique_values / inverse_indices
```

没有经过完整 PTV2，也没有构建新 Engine。每个 case 执行：

- warmup：100次；
- benchmark：1000次；
- CUDA Event 0→1：H2D input；
- CUDA Event 1→2：Plugin `enqueueV3` device执行区间；
- CUDA Event 2→3：D2H三个输出；
- CUDA Event 0→3：合计。

每轮复制 `49,156 bytes`：input `16,384`、count `4`、values capacity `16,384`、
inverse `16,384`。

## 6. Baseline结果

### 随机 voxel key

- seed：42
- 分布：uniform integer `[0,512)`
- N：2048
- M：499

| Metric | Mean | P50 | P95 | P99 |
|---|---:|---:|---:|---:|
| Plugin execution | `37.258966 ms` | `37.252144` | `38.096261` | `38.496507` |
| H2D + D2H | `0.190753 ms` | `0.176592` | `0.267846` | `0.380934` |
| Total | `37.449719 ms` | `37.432831` | `38.275508` | `38.687685` |

### 真实 weld_65 / tdb_1 key

真实key严格来自 Phase 7A固定points、`voxel_size=0.06`、per-axis min/max和：

```text
key = x + extent_x*y + extent_x*extent_y*z
```

- N：2048
- M：397
- extents：`[32,12,7]`

| Metric | Mean | P50 | P95 | P99 |
|---|---:|---:|---:|---:|
| Plugin execution | `28.847813 ms` | `28.861120` | `29.491200` | `29.859653` |
| H2D + D2H | `0.189470 ms` | `0.175376` | `0.263264` | `0.368064` |
| Total | `29.037282 ms` | `29.046544` | `29.692297` | `30.038145` |

Phase 7C完整 Engine中的 `/model/tdb_1/Unique` 为 `30.222106 ms`。隔离真实key的
`28.847813 ms` 与其接近，证明瓶颈主要在Plugin执行本身，而不是完整网络的GEMM、
Scatter/Gather或输入输出拷贝。

M从397增加到499时，kernel mean从28.85 ms增加到37.26 ms，也与当前
`O(N·M + M²)` 串行复杂度一致。

## 7. 正确性基线

当前Plugin已有18个通过用例，已复核覆盖：

- 随机 N：4、8、32、2048；
- all same；
- all unique；
- sorted；
- reversed；
- repeated groups；
- INT64 min/max extremes。

所有用例的 count、sorted unique values、inverse indices和runtime M shape均匹配CPU
reference。本轮两个N=2048 profile case在测量后也再次验证了相同合同。

这些结果只证明当前 baseline Plugin正确。后续每个优化版本都必须重新跑完整用例，
不能继承baseline结论。

## 8. 优化方案评估

### 方案1：CUB radix sort + boundary/scan（优先）

建议数据流：

```text
keys + original_index
  -> CUB DeviceRadixSort::SortPairs
  -> boundary flags on sorted keys
  -> exclusive prefix scan -> unique_id
  -> write unique_values at boundaries
  -> inverse_indices[original_index] = unique_id
  -> write voxel_count M
```

优点：

- 从单线程二次算法转为GPU并行 radix sort，复杂度接近 `O(N·key_bits)`；
- 排序输出天然满足 `sorted=True`；
- original index pair可精确恢复原输入顺序的inverse indices；
- N上限2048很小，CUB路径预计远低于5ms目标。

必须验证：

- signed INT64排序，包括 `INT64_MIN/INT64_MAX`；
- CUB signed-key bit transform是否与 `torch.unique(sorted=True)`一致；
- duplicate key的boundary和scan off-by-one；
- M=1和M=N；
- temp workspace必须预先计算并由Plugin/TensorRT管理；
- enqueue中禁止 `cudaMalloc`、host round-trip和global synchronize；
- DDS `voxel_count`、unique output allocator与runtime M保持原协议。

### 方案2：GPU hash voxelization

Hash表可把发现unique key降到期望 `O(N)`，但当前合同要求全局signed升序。Hash结果最终
仍需排序unique keys并重新映射inverse；还要处理冲突、容量、确定性和极值key。因此它
不是第一版最小风险方案，除非未来允许改变排序合同——当前明确不允许。

### 方案3：kernel fusion / block-local radix sort

N最大仅2048，可考虑 CUB `BlockRadixSort`：一个或少量block在shared memory中完成key
与original index排序，再融合boundary、unique write和inverse scatter。它可减少CUB
device算法的多个launch，适合在方案1正确但仍未达到5ms时继续评估。

风险是寄存器/shared-memory占用、跨block协调、signed INT64顺序和DDS count写出。
不应在没有先建立CUB多kernel正确基线前直接采用高度融合版本。

## 9. 性能目标评估

真实隔离kernel当前为 `28.847813 ms`。达到 `<5 ms` 需要：

- 延迟降低至少 `82.67%`；
- kernel speedup至少 `5.77x`。

若完整Engine其他耗时不变，并把tdb_1从 `30.2221 ms`降到 `5 ms`，Phase 7B pure
latency的理想简单估算为：

```text
37.0643 - 30.2221 + 5 = 11.8422 ms
```

对应当前TensorRT自身约 `3.13x` 改善。这个估算没有计入优化后tactic、workspace或DDS
行为变化，不能当成验收结果。

基于N=2048和当前单线程二次实现，`<5 ms`在算法层面具有较强可行性，但必须由优化后
实测确认。本轮没有优化实现，因此 `optimized_profile.json` 明确记录为 `NOT_RUN`。

## 10. 推荐下一阶段

若授权Phase 8B，建议严格按以下顺序：

1. 在新Plugin源码/新DLL路径实现CUB sort-pairs版本，不覆盖baseline DLL；
2. 先跑N=4/8/32/2048和全部边界正确性；
3. 再运行本报告同口径1000次isolated benchmark；
4. 达到正确性后才用原ONNX重新构建独立candidate Strict FP32 Engine；
5. 执行Parser、build、runtime、18样本parity和Phase 7B/7C回归；
6. 仅在CUB路径仍高于5ms时评估block-local fusion。

```text
VOXELUNIQUE_ANALYSIS_COMPLETED
```
