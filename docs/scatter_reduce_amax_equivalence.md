# scatter_reduce amax include_self 等价性审计

## 1. 结论

当前 deployment voxel feature pooling 中的：

```python
torch.zeros([C, F], dtype=torch.float32).scatter_reduce(
    dim=0,
    index=feature_index,
    src=flat_features,
    reduce="amax",
    include_self=False,
)
```

在当前实际定义域内，可数学等价替换为：

```python
torch.full([C, F], -inf, dtype=torch.float32).scatter_reduce(
    dim=0,
    index=feature_index,
    src=flat_features,
    reduce="amax",
    include_self=True,
)
```

人工测试、全部 90 个 weld 文件、四级 voxel size 和 6 个固定 NPZ 均通过。

```text
SCATTER_REDUCE_AMAX_EQUIVALENCE_PASSED
```

## 2. 当前 feature pooling 输入

修改前代码位于 `deployment/onnx_voxel_pool.py:129-142`。

| 项目 | 含义 |
|---|---|
| `points_features` | FP32 `[B,N,F]` |
| `flat_features` / src | FP32 `[B*N,F]` |
| `inverse_global` | int64 `[B*N]`，来自 `torch.unique(flat_global_keys, return_inverse=True)` |
| `feature_index` | int64 `[B*N,F]`，由 `inverse_global.unsqueeze(1).expand(-1,F)` 得到 |
| target | FP32 `[C,F]`，`C=len(unique_global_keys)` |
| dim | `0` |
| reduce | `amax` |
| 原始初值 | `0`，但由 `include_self=False` 排除 |
| 输出 | 每个 materialized voxel、每个 feature channel 的最大值 |

四级实际 feature shape：

| TransitionDown | voxel size | 输出 feature dim |
|---|---:|---:|
| tdb_1 | 0.06 | 96 |
| tdb_2 | 0.13 | 192 |
| tdb_3 | 0.325 | 384 |
| tdb_4 | 0.8125 | 512 |

部署模型与 checkpoint 使用 FP32。审计数据中的 feature 全部有限，无 NaN/Inf。

## 3. 非空条件

`C` 不是稠密空间网格容量，而是 `unique_global_keys` 的实际元素数量。`inverse_global` 将每个输入点映射到一个 unique key，且每个 unique key 至少由一个输入点产生。因此：

```text
for every voxel j in [0,C-1]:
    exists point i such that inverse_global[i] = j
```

`feature_index` 只是把同一个 voxel index 沿 F 个 channel 展开，所以每个 `[j,f]` target 元素也至少接收一个 source feature。

全部真实测试中：

```text
materialized empty voxel count = 0
```

如果未来改成预分配稠密空 voxel target，本结论需要重新审计。

## 4. 数学证明

对任意 materialized voxel `j` 和 feature channel `f`，定义非空 source 集合：

```text
S(j,f) = {flat_features[i,f] | inverse_global[i] = j}
```

原结果：

```text
old(j,f) = max(S(j,f))
```

新结果：

```text
new(j,f) = max(-inf, S(j,f))
```

对任意非空有限 FP32 集合 `S`：

```text
max(-inf, S) = max(S)
```

因此：

```text
new(j,f) = old(j,f)
```

该证明同时覆盖正 feature、全负 feature、极大/极小有限 feature 和 multi-batch。使用零值配合 `include_self=True` 不等价，因为它会错误截断全负 feature；必须使用 `-inf`。

## 5. 独立测试脚本

新增：

- `scripts/validate_scatter_reduce_amax_equivalence.py`

命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe scripts\validate_scatter_reduce_amax_equivalence.py --device cuda:0
```

产物：

- `artifacts/scatter_reduce_amax_equivalence/20260715_112602_851933_audit/scatter_reduce_amax_equivalence.json`
- `artifacts/scatter_reduce_amax_equivalence/20260715_112602_851933_audit/validation.log`

脚本不导出 ONNX。

## 6. 人工测试

覆盖 5 类：

- 一个 voxel 多个 feature；
- 一个 voxel 单 feature；
- 极大/极小有限 FP32 feature；
- 全负 feature；
- multi-batch。

所有离散结果完全一致，pooled features 满足 `allclose(rtol=1e-5, atol=1e-6)`，并且实际为 bitwise 一致。

## 7. 真实数据测试

真实 XYZ 和标签用于生成有限 FP32 feature tensor，并严格匹配四级实际 feature dim `96/192/384/512`。等价证明本身对 feature 具体来源和值分布无依赖。

| 范围 | 样本数 | 四级对比数 | feature dims | 空 voxel | pooled feature 最大误差 |
|---|---:|---:|---|---:|---:|
| `weld_1.txt`～`weld_90.txt` | 90 | 360 | 96/192/384/512 | 0 | 0.0 |
| 6 个固定 NPZ | 6 | 24 | 96/192/384/512 | 0 | 0.0 |

完全一致的离散项：

- voxel start/min；
- extents；
- unique global/local keys；
- unique batch ids；
- voxel counts；
- point-to-voxel mapping；
- retained voxel coordinates。

pooled feature：

```text
allclose = True
bitwise equal = True
max_abs_error = 0.0
```

## 8. 条件修改

只有在独立审计成功后，才修改 `deployment/onnx_voxel_pool.py`：

- target 从 `torch.zeros([C,F])` 改为 `torch.full([C,F], -inf)`；
- `include_self=False` 改为 `include_self=True`；
- src、index、dim、reduce、dtype 和输出 shape 均未改变。

未修改原始模型、checkpoint、数据、网络结构或其他 deployment 数学逻辑。

## 9. STANDARD_OPS_VOXEL_POOL_EQUIVALENCE 复验

命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe scripts\validate_voxel_pool_equivalence.py --device cuda:0
```

产物：

- `artifacts/gcn_res_standard_ops/20260715_112637_916964_equivalence/`

8 类标准 voxel pooling 测试全部通过：

```text
STANDARD_OPS_VOXEL_POOL_EQUIVALENCE_PASSED
```

## 10. 停止边界

本阶段未执行：

- ONNX export；
- ONNX Runtime；
- TensorRT。

```text
SCATTER_REDUCE_AMAX_EQUIVALENCE_PASSED
STANDARD_OPS_VOXEL_POOL_EQUIVALENCE_PASSED
ONNX_EXPORT_NOT_RUN
ONNXRUNTIME_NOT_RUN
TENSORRT_NOT_RUN
```
