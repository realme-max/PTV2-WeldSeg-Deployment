# GCN_res trunc / floor ONNX 等价性审计

## 1. 结论摘要

在当前 GCN_res weld 部署模型的 voxel 定义域内，可以将：

```python
torch.trunc((xyz - start) / voxel_size)
```

替换为：

```python
torch.floor((xyz - start) / voxel_size)
```

成立条件是当前实现中固定满足：

- `start = amin(xyz, dim=points)`，即每个 batch、每个坐标轴独立取最小值；
- 四级 `voxel_size` 均严格为正：`0.06、0.13、0.325、0.8125`；
- 输入坐标全部为有限 FP32 数值。

本次已检查全部 90 个实际 weld 文件、54/18/18 子划分和 6 个固定 NPZ。实际定义域内的 voxel coordinates、extents、voxel 数量、排序 key 和 point-to-voxel mapping 全部完全一致。

```text
TRUNC_FLOOR_EQUIVALENCE_PASSED
```

## 2. 数学证明

对 batch 内任一点 `i` 和坐标轴 `d`：

```text
start[d] = min_i xyz[i,d]
```

根据最小值定义：

```text
xyz[i,d] >= start[d]
shifted[i,d] = xyz[i,d] - start[d] >= 0
```

由于当前四级 `voxel_size[d] > 0`：

```text
q[i,d] = shifted[i,d] / voxel_size[d] >= 0
```

对任意非负实数 `q`，向零截断和向下取整相同：

```text
trunc(q) = floor(q), q >= 0
```

因此当前 voxel 坐标满足：

```text
trunc((xyz - start) / voxel_size)
= floor((xyz - start) / voxel_size)
```

第二处 extents 计算同理。由于 `end = max(xyz)`，所以 `end - start >= 0`，从而：

```text
trunc((end - start) / voxel_size) + 1
= floor((end - start) / voxel_size) + 1
```

该证明不适用于任意外部 `start` 或负的 scaled values。如果未来允许调用者传入不等于坐标最小值的 start，必须重新审计。

## 3. 独立测试脚本

新增：

- `scripts/validate_trunc_floor_equivalence.py`

脚本只进行等价性审计，不导出 ONNX，不运行 ONNX Runtime 或 TensorRT。它检查：

1. 人工正负标量的 trunc/floor 行为；
2. 全部 90 个 TXT 的原始 XYZ；
3. 使用项目评估预处理逻辑得到的 train/val/test 归一化 XYZ；
4. 6 个历史评估固定 NPZ 中的 `normalized_xyz`；
5. 四种 voxel size 下的坐标、extents、mixed-radix key、unique voxel count 和 inverse mapping。

执行命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe scripts\validate_trunc_floor_equivalence.py --device cuda:0
```

审计产物：

- `artifacts/trunc_floor_equivalence/20260715_100132_231416_audit/trunc_floor_equivalence.json`
- `artifacts/trunc_floor_equivalence/20260715_100132_231416_audit/validation.log`

## 4. 人工标量结果

FP32 测试结果：

| 输入 | trunc | floor | 是否相同 |
|---:|---:|---:|---|
| 0 | 0 | 0 | 是 |
| 0.1 | 0 | 0 | 是 |
| 1.9 | 1 | 1 | 是 |
| 3.99 | 3 | 3 | 是 |
| -0.1 | -0 | -1 | 否 |
| -1.9 | -1 | -2 | 否 |

负数测试证明 `trunc` 与 `floor` 并非全定义域等价。本次替换成立的关键不是经验假设，而是当前实现先执行 `xyz - amin(xyz)`，使实际输入商恒为非负数。

## 5. 实际数据定义域检查

数据路径与数量：

| 范围 | 样本数 | 检查点数 | `shifted.min()` | 四种 voxel size 的最小商 |
|---|---:|---:|---:|---:|
| 全部原始 `weld_1.txt`～`weld_90.txt` | 90 | 184320 | 0.0 | 0.0 |
| sub train 归一化输入 | 54 | 110592 | 0.0 | 0.0 |
| sub val 归一化输入 | 18 | 36864 | 0.0 | 0.0 |
| sub test 归一化输入 | 18 | 36864 | 0.0 | 0.0 |
| 6 个固定 NPZ | 6 | 12288 | 0.0 | 0.0 |

子划分来源：

- `data/weld/train_test_split/sub_shuffled_train_file_list.json`
- `data/weld/train_test_split/sub_shuffled_val_file_list.json`
- `data/weld/train_test_split/sub_shuffled_test_file_list.json`

6 个固定 NPZ：

- `val_00_weld_7`
- `val_01_weld_61`
- `val_02_weld_49`
- `test_00_weld_65`
- `test_01_weld_30`
- `test_02_weld_28`

全部数据还满足：

- scaled values 全部 `>= 0`；
- trunc/floor voxel coordinates 逐元素完全一致；
- trunc/floor extents 完全一致；
- unique voxel count 完全一致；
- 排序后的 mixed-radix unique keys 完全一致；
- point-to-voxel inverse mapping 完全一致。

## 6. 部署实现修改

只有在上述独立审计成功后，才修改 `deployment/onnx_voxel_pool.py`：

- 原第 71 行 voxel coordinates：`torch.trunc` 改为 `torch.floor`；
- 原第 72 行 extents：`torch.trunc` 改为 `torch.floor`；
- 注释同步限定为 `start=per-axis minimum` 且 voxel size 为正的部署定义域。

没有修改：

- `models/testParameters/GCN_res/model.py`
- `models/testParameters/GCN_res/best_model.pth`
- 数据或划分 JSON
- checkpoint 或 benchmark
- 网络结构、参数、容差

修改后扫描 `deployment` 目录，不再存在 `torch.trunc` 或 `aten::trunc`。

## 7. STANDARD_OPS_VOXEL_POOL_EQUIVALENCE 复验

执行命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe scripts\validate_voxel_pool_equivalence.py --device cuda:0
```

产物：

- `artifacts/gcn_res_standard_ops/20260715_100156_656252_equivalence/`

8 类人工用例全部通过：

- all points in one voxel；
- every point in a different voxel；
- two voxels；
- voxel boundaries；
- negative coordinates；
- duplicate points；
- multi batch；
- unequal voxel counts across batch。

结果：

```text
STANDARD_OPS_VOXEL_POOL_EQUIVALENCE_PASSED
```

## 8. GCN_RES_DEPLOYMENT_MODEL_PARITY 复验

执行命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe scripts\validate_gcn_res_deployment_equivalence.py --device cuda:0
```

产物：

- `artifacts/gcn_res_standard_ops/20260715_100206_461255_deployment_parity/`

checkpoint 对 original/deployment 均 `strict=True` 加载成功，无 key mapping。既有脚本保持原严格逐层阈值，并在首个失败样本停止，因此本次只执行了 `val_00_weld_7`。

四级 voxel pooling 均通过，离散结果全部完全一致：

| 层 | voxel 数 | coordinates/counts/membership/keys | pooled feature 最大绝对误差 |
|---|---:|---|---:|
| tdb_1 | 518 | 全部完全一致 | 0.0 |
| tdb_2 | 129 | 全部完全一致 | 0.0 |
| tdb_3 | 24 | 全部完全一致 | 0.0 |
| tdb_4 | 4 | 全部完全一致 | 0.0 |

最终 logits 仍满足既定工程验收：

| 指标 | 结果 |
|---|---:|
| shape | `[1,2048,2]` |
| max absolute error | 0.000002503395 |
| mean absolute error | 0.000000221164 |
| max relative error | 0.000017549602 |
| logits `rtol=1e-4, atol=1e-5` | 通过 |
| predicted-label agreement | 100% |
| finite | 是 |

原严格逐层检查仍在 `ptb_8`、`tub_9`、`ptb_9` 报告失败，对应 feature 最大绝对误差分别为：

- `ptb_8`: `8.404255e-06`
- `tub_9`: `4.887581e-06`
- `ptb_9`: `4.768372e-06`

因此原验证脚本的状态字符串仍为：

```text
GCN_RES_DEPLOYMENT_MODEL_PARITY_FAILED
```

这与替换前已经确认的 CUDA FP32 scatter reduction 非 bitwise 噪声一致。本次没有调整逐层容差。关键证据是所有四级 voxel 离散分组与 pooled features 完全一致，且最终 logits 工程容差和标签一致率通过；没有出现由 trunc/floor 引起的新分叉。

## 9. 停止边界

按本阶段要求，本次没有执行：

- ONNX export；
- `onnx.checker` 或 shape inference；
- ONNX Runtime；
- TensorRT。

本阶段只证明并应用了 `trunc -> floor` 的受限定义域数学等价替换。下一次 ONNX 导出必须由后续任务显式启动。

## 10. 最终状态

```text
TRUNC_FLOOR_EQUIVALENCE_PASSED
STANDARD_OPS_VOXEL_POOL_EQUIVALENCE_PASSED
GCN_RES_DEPLOYMENT_MODEL_PARITY_FAILED  # 已知严格逐层 CUDA reduction 噪声；最终 logits 通过
ONNX_EXPORT_NOT_RUN
ONNXRUNTIME_NOT_RUN
TENSORRT_NOT_RUN
```
