# GCN_res 标准算子 voxel pooling 等价性报告

## 1. 结论摘要

本轮已完成部署侧标准 PyTorch 算子实现及两级验证，但**没有达到重新导出 ONNX 的全部前置条件**。

- 人工 voxel pooling 测试：8/8 通过。
- 原始 checkpoint：原模型与部署模型均 `strict=True` 加载成功；434 个 state_dict key 完全一致，无参数重命名或映射，加载后的张量逐项 bitwise 相同。
- 第一个真实固定样本 `val_00_weld_7`：四个 TransitionDown 的 voxel 数、voxel 坐标、成员关系、point-to-voxel、点数、pooled XYZ 和 pooled features 均通过。
- 同一样本最终 logits：满足最终 logits 容差，最大绝对误差 `1.9073486328125e-06`，预测标签一致率 `1.00000000`。
- 但是逐层严格容差 `rtol=1e-5, atol=1e-6` 在 `ptb_8`、`tub_9`、`ptb_9` 失败。因此遵守停止条件，没有运行其余 5 个真实样本，也没有重新导出 ONNX、运行 ONNX Runtime 或进入 TensorRT。

本轮状态：

```text
STANDARD_OPS_VOXEL_POOL_EQUIVALENCE_PASSED
GCN_RES_DEPLOYMENT_MODEL_PARITY_FAILED
```

## 2. 未修改的数学真源

本轮未修改以下文件：

| 文件 | 作用 | 本轮记录的 SHA-256 |
|---|---|---|
| `models/testParameters/GCN_res/model.py` | 原始网络真源 | `4618F4E1C119FC36B0EF5135C656F00CD97D80C1DF0F11A3CFC9711AC15DDD55` |
| `models/testParameters/GCN_res/ptv2_utils.py` | 原始 voxel pooling 与 PTV2 工具 | `AC8E629F8F35066219EA1D81B39FEB7D8B858A563C6225B472C0F0D78B503420` |
| `models/testParameters/GCN_res/best_model.pth` | 历史 checkpoint | `311BDDF3607D76E6B7DED450B8419BF6AE98F34F50578608B3E6A1C1C3E58D21` |

工作目录不是 Git 仓库，因此无法用 Git 基线证明“修改前后 diff”；以上 hash 是本轮验证时的完整性记录。三个原始文件的时间戳仍分别为 2024-12-06、2024-12-01、2024-12-06。

## 3. 原始 voxel pooling 语义

### 3.1 入口、grid size 与 shape

原始入口是 `models/testParameters/GCN_res/ptv2_utils.py:53` 的 `partition_based_pooling`，由 `TransitionDownBlock.forward`（同文件 `:101-108`）调用。模型中的四级配置位于 `models/testParameters/GCN_res/model.py:87,91,95,99`：

| Stage | 输入 features | 输出 features | voxel size |
|---|---:|---:|---|
| `tdb_1` | `[B,N0,48]` | `[B,N1,96]` | `[0.06,0.06,0.06]` |
| `tdb_2` | `[B,N1,96]` | `[B,N2,192]` | `[0.13,0.13,0.13]` |
| `tdb_3` | `[B,N2,192]` | `[B,N3,384]` | `[0.325,0.325,0.325]` |
| `tdb_4` | `[B,N3,384]` | `[B,N4,512]` | `[0.8125,0.8125,0.8125]` |

每级先执行 `Linear → BatchNorm1d → ReLU`，再对 XYZ 和变换后的 features 做 pooling。

### 3.2 torch_cluster::grid 的精确编号规则

原代码在 `ptv2_utils.py:67-70` 对 batch 做 Python 循环，并在每个样本上独立调用 `grid_cluster(points[i], size)`，没有显式传入 `start/end`。依据已安装的 torch_cluster 1.6.3 实现：

对 batch `b`、轴 `d∈{x,y,z}`：

```text
start[b,d] = min_i p[b,i,d]
end[b,d]   = max_i p[b,i,d]
q[b,i,d]   = trunc((p[b,i,d] - start[b,d]) / size[d])
extent[b,d]= trunc((end[b,d] - start[b,d]) / size[d]) + 1
```

底层 C++/CUDA 的浮点到 `int64` 转换是向零截断。由于默认 `start` 是该 batch 的最小值，差值非负，所以当前调用下 `trunc` 与 `floor` 相同；部署实现仍显式使用 `trunc`，没有把任意显式边界情形错误地概括为 floor。

三维坐标以 x 为最快变化轴编码：

```text
key = qx + extent_x * qy + extent_x * extent_y * qz
```

原始代码没有把 batch id 编入 key，因为每个 batch 在 Python 循环中独立编号。因此不同 batch 可以出现相同整数 key，但不会彼此聚合。

### 3.3 unique、inverse、聚合与映射

`ptv2_utils.py:70` 调用：

```python
unique_clusters, inverse_indices = cluster_indices.unique(return_inverse=True)
```

本环境的 CPU/CUDA 实测均返回升序 unique key；`inverse_indices[i]` 是原点 `i` 所属的升序 unique voxel 下标。设占用 voxel 数为 `C`，则：

- `unique_clusters`: `[C]`，按 key 升序。
- `inverse_indices`: `[N]`，即 point-to-voxel 映射。
- voxel-to-point：所有满足 `inverse_indices[i] = c` 的点集合；原代码没有单独返回稀疏列表。
- `scatter_reduce_(reduce="amax", include_self=False)`（`ptv2_utils.py:73-75`）：逐 voxel、逐通道最大值，输出 `[C,F]`。
- `scatter_add_`（`:77-78`）：逐 voxel 求 XYZ 和，输出 `[C,3]`。
- `bincount(..., minlength=C)`（`:79`）：得到每个已占用 voxel 的点数 `[C]`。
- 代表点（`:80`）：XYZ 和除以点数，即 voxel 内点的算术平均，输出 `[C,3]`。

`unique` 只产生实际被点占用的 voxel，因此 `C` 个输出 voxel 的 count 都大于 0，不存在被显式物化的空 voxel，也不存在除零。

### 3.4 多 batch 的特殊裁剪

原始代码先得到每个 batch 的 `C_b` 个 voxel，然后在 `ptv2_utils.py:85-90` 计算：

```text
M = min_b C_b
```

并只保留每个 batch 按升序 key 排列的前 `M` 个 voxel，最终输出：

- pooled XYZ `[B,M,3]`
- pooled features `[B,M,F]`

这不是 padding，也不是随机采样。被裁掉的是 key 排序靠后的 voxel。voxel 顺序会影响后续 `cdist/topk` 中距离相等时的 tie-breaking，也必须保证 XYZ 与 features 的顺序成对一致；因此部署实现保留原顺序，不能只保证 shape 相同。

## 4. 部署侧算法

实现位置：`deployment/onnx_voxel_pool.py:46-205`。

部署实现一次处理 `[B,N,3]`，但保持每个 batch 独立边界和编号：

1. `:64-72`：逐 batch 求 `start/end`，使用 `trunc` 和原公式计算 int64 voxel coordinates 与 extents。
2. `:76-87`：按 x-fastest 的混合进制计算 local key。
3. 为 batch `b` 定义容量 `capacity_b = extent_x * extent_y * extent_z`，以及互不重叠的前缀偏移 `offset_b = Σ(j<b) capacity_j`，构造 `global_key = offset_b + local_key`。
4. 因 `0 <= local_key < capacity_b`，不同 batch 的 key 区间严格不相交，不使用哈希，不存在哈希冲突。坐标、extent、key、offset 和 inverse 全部使用 int64，避免 int32 溢出。
5. `:92-110`：对 global key 做 sorted unique，并恢复 batch id、local key 与 local inverse。
6. `:111-137`：用标准 `scatter_add` 替代 `bincount`，XYZ 求和后除 count；features 使用标准 `scatter_reduce(amax, include_self=False)`。
7. `:138-181`：恢复每个 batch 的局部 voxel 序号，执行与原代码相同的 `min(C_b)` 裁剪，并恢复可审计的 voxel coordinates/counts。

`deployment/gcn_res_onnx_model.py:40-59` 用这个 pooling 替换 TransitionDown；其余网络按原模块树实现。`_voxel_size` 是 `persistent=False` buffer，不进入 state_dict，所以 checkpoint key 集合没有增加。

部署文件的静态搜索未发现代码级 `torch_cluster`、`torch_scatter`、NumPy、sklearn、`.cpu()` 或 `.numpy()` 调用；搜索命中的 `torch_cluster` 仅存在于解释原语义的注释/docstring。

## 5. 与原实现的实现差异

| 项目 | 原实现 | 部署实现 | 数学语义 |
|---|---|---|---|
| batch 处理 | Python `for`，逐样本 grid | batch key 加无冲突 offset 后一次 unique | 相同 |
| voxel key | torch_cluster CUDA/CPU kernel | 标准 tensor arithmetic + int64 | 相同 |
| count | `torch.bincount` | `zeros.scatter_add(ones)` | 相同 |
| feature pooling | `scatter_reduce_ amax` | `scatter_reduce amax` | 相同 |
| XYZ pooling | `scatter_add_ / count` | `scatter_add / count` | 相同 |
| 多 batch 输出 | Python list，裁到最小 C | local rank mask，裁到最小 C | 相同 |
| batch index in interpolation | 原代码构造默认 CPU `arange` | 在 indices 所在 device 构造 | 索引值相同；避免设备往返 |

没有删除、绕过或弱化 voxel pooling；没有改成简单平均池化、固定下采样或随机采样。

## 6. 人工测试

脚本：`scripts/validate_voxel_pool_equivalence.py`。

命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe scripts\validate_voxel_pool_equivalence.py --run-dir artifacts\gcn_res_standard_ops\20260714_172215_000000_equivalence_retry
```

运行设备：RTX 5060 / `cuda:0`。离散结果要求完全一致，浮点结果要求 `allclose(rtol=1e-5, atol=1e-6)`。

| 用例 | 原/部署 voxel 数 | 结果 |
|---|---|---|
| 所有点在同一 voxel | `[1]` | 通过 |
| 每点不同 voxel | `[4]` | 通过 |
| 两组 voxel | `[2]` | 通过 |
| voxel 边界附近 | `[3]` | 通过 |
| 负坐标 | `[3]` | 通过 |
| 重复点 | `[2]` | 通过 |
| 多 batch | `[3,3]` | 通过 |
| batch 内点数分布不均且 voxel 数不同 | `[3,2]` | 通过，正确裁为 2 |

每个用例均验证 voxel coordinates、unique key、voxel count、cluster membership（通过点对同簇关系验证，允许编号重排）、point-to-voxel、voxel 点数、pooled XYZ、pooled features 和有限性。

产物：`artifacts/gcn_res_standard_ops/20260714_172215_000000_equivalence_retry/voxel_pool_artificial_tests.json`。

## 7. checkpoint 验证

脚本：`scripts/validate_gcn_res_deployment_equivalence.py`。

- checkpoint 字段可读，epoch 为 124。
- 原模型 `strict=True`: `<All keys matched successfully>`。
- 部署模型 `strict=True`: `<All keys matched successfully>`。
- state_dict key 数：434。
- 无 key remapping、复制或重命名。
- 两个模型加载后的同名张量逐项 `torch.equal=True`。
- `linear_1.weight = [48,4]`，`mlp.weight = [2,48]`。
- checkpoint tensor 全部有限。

## 8. 真实固定样本对齐结果

### 8.1 停止位置

按停止条件，验证在第一个固定样本 `val_00_weld_7` 逐层失败后立即停止。因此本轮实际状态是：

| 固定样本 | 状态 |
|---|---|
| `val_00_weld_7` | 已运行；voxel 全通过，但后段逐层容差失败 |
| `val_01_weld_61` | 未运行：停止条件 |
| `val_02_weld_49` | 未运行：停止条件 |
| `test_00_weld_65` | 未运行：停止条件 |
| `test_01_weld_30` | 未运行：停止条件 |
| `test_02_weld_28` | 未运行：停止条件 |

所以不能写成“6 个真实样本通过”。

### 8.2 四级 voxel pooling

`val_00_weld_7` 的四级结果：

| Stage | voxel 数 | coords/membership/count/key | pooled XYZ max abs | pooled features max abs | 结果 |
|---|---:|---|---:|---:|---|
| `tdb_1` | 518 | 全部完全一致 | `5.960464e-08` | `0` | 通过 |
| `tdb_2` | 129 | 全部完全一致 | `2.980232e-08` | `0` | 通过 |
| `tdb_3` | 24 | 全部完全一致 | `5.960464e-08` | `0` | 通过 |
| `tdb_4` | 4 | 全部完全一致 | `0` | `0` | 通过 |

### 8.3 各网络阶段最大误差

以下来自保留完整诊断的停止运行。`Passed` 使用逐层 `rtol=1e-5, atol=1e-6`：

| Stage | Max abs error | Max relative error | Passed |
|---|---:|---:|---|
| `linear_1` | `0` | `0` | 是 |
| `ptb_0` | `0` | `0` | 是 |
| `gcn_0` | `0` | `0` | 是 |
| `tdb_1` | `1.192093e-07` | `1.644738e-07` | 是 |
| `ptb_1` | `5.140901e-07` | `5.705519e-04` | 是 |
| `tdb_2` | `8.940697e-07` | `1.454194e-04` | 是 |
| `ptb_2` | `8.940697e-07` | `1.355730e-02` | 是 |
| `tdb_3` | `2.384186e-07` | `8.619222e-05` | 是 |
| `ptb_3` | `2.384186e-07` | `8.447460e-05` | 是 |
| `tdb_4` | `1.192093e-07` | `7.457592e-05` | 是 |
| `ptb_4` | `1.192093e-07` | `7.457102e-05` | 是 |
| `tub_6` | `2.384186e-07` | `5.198267e-04` | 是 |
| `ptb_6` | `3.417954e-07` | `3.311311e-04` | 是 |
| `tub_7` | `7.152557e-07` | `6.797742e-05` | 是 |
| `ptb_7` | `7.152557e-07` | `7.696913e-04` | 是 |
| `tub_8` | `9.536743e-07` | `3.909424e-04` | 是 |
| `ptb_8` | `9.059906e-06` | `1.310616e-03` | **否** |
| `tub_9` | `8.463860e-06` | `2.531931e-04` | **否** |
| `ptb_9` | `9.298325e-06` | `1.233806e-03` | **否** |
| `mlp/logits` | `1.907349e-06` | `1.256061e-05` | 是（使用最终 logits 容差） |

小的 pooled XYZ 舍入差异经过后续 `cdist/topk/attention` 累积，在后段超过了逐层严格容差。两次失败运行的最终最大绝对误差分别为 `4.291534e-06` 和 `1.907349e-06`，说明当前 CUDA reduction/后续邻域路径还存在微小运行间差异；不能把单次最终 logits 通过替代六样本逐层验收。

### 8.4 最终 logits 与概率（已运行样本）

`val_00_weld_7`：

- shape：原/部署均 `[1,2048,2]`。
- max absolute error：`1.9073486328125e-06`。
- mean absolute error：`2.1181676856940612e-07`。
- max relative error：`1.25606147776125e-05`。
- logits `allclose(rtol=1e-4, atol=1e-5)`：通过。
- predicted label agreement：`1.00000000`。
- weld_seam（class 0）probability max/mean error：`1.788139e-07 / 6.460268e-09`。
- background（class 1）probability max/mean error：`2.086163e-07 / 6.140453e-09`。
- 原/部署 logits 均无 NaN/Inf。

这些结果只代表第一个样本，不能外推为六样本成功。

## 9. 验证脚本首次诊断异常

首次真实验证在额外重放 TransitionDown 的诊断代码处遇到：

```text
RuntimeError: Inference tensors cannot be saved for backward.
```

位置是 `compare_transition_pool` 对 hook 捕获的 inference tensor 再调用 Linear。它发生在任何等价性判断前，不是模型数值失败。修正仅将这段只读诊断重放包入 `torch.inference_mode()`，未修改模型或算法。完整 traceback 保存在：

`artifacts/gcn_res_standard_ops/20260714_171836_824396_equivalence/deployment_equivalence.json`

之后的新运行才产生本报告的实际数值失败结论。

## 10. 是否具备重新导出条件

不具备。用户定义的六项前置条件中：

| 条件 | 状态 |
|---|---|
| 人工测试全部通过 | 通过 |
| 6 个真实样本逐层关键张量对齐 | **未通过；首样本后段失败并停止** |
| 6 个样本最终 logits 达到容差 | 未执行完成 |
| 6 个样本标签一致率达到要求 | 未执行完成 |
| 无 NaN/Inf | 已运行部分通过 |
| 原模型/checkpoint/数据未修改 | 通过 |

因此本轮明确没有调用新的 opset 18 导出，也没有产生新 ONNX 文件。`onnx.checker`、shape inference、PythonOp/ATen/custom domain/Constant/双输入依赖检查均未运行；ONNX Runtime 和 TensorRT 也未运行。

## 11. 产物与修改文件

新增：

- `deployment/onnx_voxel_pool.py`
- `deployment/gcn_res_onnx_model.py`
- `scripts/validate_voxel_pool_equivalence.py`
- `scripts/validate_gcn_res_deployment_equivalence.py`
- `docs/gcn_res_standard_ops_equivalence.md`

关键产物：

- 人工测试通过：`artifacts/gcn_res_standard_ops/20260714_172215_000000_equivalence_retry/voxel_pool_artificial_tests.json`
- 首次真实数值失败日志：同目录下 `deployment_equivalence.log/json`
- 保留完整逐层误差的停止运行：`artifacts/gcn_res_standard_ops/20260714_172300_000000_failed_stage_diagnostic/deployment_equivalence.log/json`

原始模型、原始 `ptv2_utils.py`、checkpoint、数据和划分文件均未修改。

## 12. 最终标志

```text
STANDARD_OPS_VOXEL_POOL_EQUIVALENCE_PASSED
GCN_RES_DEPLOYMENT_MODEL_PARITY_FAILED
```
