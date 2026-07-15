# scatter_reduce amin include_self 等价性审计

## 1. 结论

当前 deployment voxel pooling 中用于计算 `unique_batch_ids` 的：

```python
zeros.scatter_reduce(
    dim=0,
    index=inverse_global,
    src=point_batch_ids,
    reduce="amin",
    include_self=False,
)
```

在当前实际定义域内，可数学等价替换为：

```python
full(torch.iinfo(torch.int64).max).scatter_reduce(
    dim=0,
    index=inverse_global,
    src=point_batch_ids,
    reduce="amin",
    include_self=True,
)
```

`point_batch_ids` 是 int64，不能存储 IEEE `+inf`。因此使用 `9223372036854775807` 作为类型保持的正无穷单位元；所有有效 batch id 都位于 `[0, B-1]`，严格小于该值。

```text
SCATTER_REDUCE_AMIN_EQUIVALENCE_PASSED
```

## 2. 原始输入语义

代码位置为修改前的 `deployment/onnx_voxel_pool.py:104-108`。

| 参数 | 当前含义 |
|---|---|
| target/self | int64 零张量 `[C]`，`C = len(unique_global_keys)` |
| dim | `0` |
| index | `inverse_global [B*N]`，由 `torch.unique(..., return_inverse=True)` 生成 |
| src | `point_batch_ids [B*N]`，取值范围 `[0,B-1]` |
| reduce | `amin` |
| include_self | 原实现为 `False` |
| output | `unique_batch_ids [C]`，每个实际 voxel 所属 batch |

其中 `global_keys` 已使用每个 batch 的 capacity offset 消除跨 batch key 冲突。因此一个 unique global voxel 内的所有点具有相同 batch id，`amin` 的结果就是该 voxel 的 batch id。

## 3. 空 voxel 可达性证明

令：

```text
unique_global_keys, inverse_global = unique(flat_global_keys, return_inverse=True)
C = len(unique_global_keys)
```

`inverse_global` 的定义是把每个输入元素映射到它在 `unique_global_keys` 中的位置。`unique_global_keys` 中的每个元素都来自至少一个 `flat_global_keys` 元素，所以对所有 `j in [0,C-1]`：

```text
exists i: inverse_global[i] = j
```

因此 `[C]` 目标张量中的每个槽至少接收一个 source，不存在 materialized empty voxel。

人工分配未被 index 引用的槽时，两种 raw 输出确实不同：

```text
index       = [0, 2]
src         = [0, 1]
old output  = [0, 0, 1, 0]
new output  = [0, INT64_MAX, 1, INT64_MAX]
```

已占用槽完全一致，空槽不同。该测试明确限定了等价范围：如果未来不再以 `unique_global_keys` 的长度创建 target，或者人为分配稠密空 voxel，则本替换不能直接沿用。

## 4. 数学等价

对任意实际 target voxel `j`，source 集合记为：

```text
S_j = {point_batch_ids[i] | inverse_global[i] = j}
```

上节已经证明 `S_j` 非空。原结果为：

```text
old[j] = min(S_j)
```

令 int64 最大值为 `I`。由于所有 `s in S_j` 都满足 `s < I`：

```text
new[j] = min(I, S_j) = min(S_j) = old[j]
```

因此在当前 materialized unique voxel 定义域内，`include_self=True` 与正无穷单位元保持数学一致。

## 5. 独立测试脚本

新增：

- `scripts/validate_scatter_reduce_amin_equivalence.py`

命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe scripts\validate_scatter_reduce_amin_equivalence.py --device cuda:0
```

产物：

- `artifacts/scatter_reduce_amin_equivalence/20260715_111108_160112_audit/scatter_reduce_amin_equivalence.json`
- `artifacts/scatter_reduce_amin_equivalence/20260715_111108_160112_audit/validation.log`

## 6. 人工测试

覆盖：

- 一个 voxel 多个点；
- 一个 voxel 一个点；
- 人工空 target slot；
- 接近 int64 上界的有效 source；
- 极值坐标；
- 负坐标；
- multi-batch。

共完成 20 次完整 pooling 比较。所有实际 materialized voxel 的 batch id、key、坐标、cluster mapping 和 pooled features 完全一致。人工空槽的差异按第 3 节处理，不属于当前可达状态。

## 7. 实际数据测试

全部 90 个 `weld_1.txt`～`weld_90.txt` 按项目实际归一化逻辑加载，并分别在四级 voxel size `0.06、0.13、0.325、0.8125` 下比较。

| 范围 | 样本 | pooling 对比 | materialized empty voxel | pooled feature 最大误差 | pooled point 最大误差 |
|---|---:|---:|---:|---:|---:|
| 90 个 weld 文件 | 90 | 360 | 0 | 0.0 | `1.788139e-07` |
| 6 个固定 NPZ | 6 | 24 | 0 | 0.0 | `1.788139e-07` |

以下离散结果全部逐元素完全一致：

- voxel min/start 坐标；
- extents；
- unique global/local voxel keys；
- unique voxel 数量；
- unique batch ids；
- voxel point counts；
- point-to-voxel mapping；
- retained voxel coordinates。

pooled point 的微小差异来自两次独立 CUDA FP32 `scatter_add` 执行，低于既有 `rtol=1e-5, atol=1e-6`；pooled features 为 bitwise 一致。

## 8. 条件修改

独立审计通过后，已修改 `deployment/onnx_voxel_pool.py` 中唯一的 batch-id `amin`：

- target 初值：int64 零改为 `torch.iinfo(torch.int64).max`；
- `include_self=False` 改为 `include_self=True`；
- `src/index/dim/reduce` 均未改变；
- dtype 仍为 int64；
- feature pooling 的 `amax(include_self=False)` 未修改。

未修改原始模型、checkpoint、数据、网络结构或其他数学逻辑。

## 9. STANDARD_OPS_VOXEL_POOL_EQUIVALENCE 复验

命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe scripts\validate_voxel_pool_equivalence.py --device cuda:0
```

产物：

- `artifacts/gcn_res_standard_ops/20260715_111135_928847_equivalence/`

8 类标准 voxel pooling 用例全部通过：

```text
STANDARD_OPS_VOXEL_POOL_EQUIVALENCE_PASSED
```

## 10. 停止边界

本阶段没有执行 ONNX export、ONNX Runtime 或 TensorRT。`artifacts/gcn_res_onnx` 中仍无成功生成的 `.onnx` 文件。

```text
SCATTER_REDUCE_AMIN_EQUIVALENCE_PASSED
STANDARD_OPS_VOXEL_POOL_EQUIVALENCE_PASSED
ONNX_EXPORT_NOT_RUN
```
