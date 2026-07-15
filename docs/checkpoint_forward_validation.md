# GCN_res Checkpoint 与 RTX 5060 Forward 验证记录

执行日期：2026-07-14  
环境：`E:\GRP-PTv2\.venv_ptv2`  
目标模型：`models/testParameters/GCN_res/model.py`  
目标 checkpoint：`models/testParameters/GCN_res/best_model.pth`  
历史初次状态：前置依赖联合导入曾因 Hydra 1.2.0 / Python 3.11 不兼容而停止。获准升级 Hydra 1.3.2 后的最终成功结果见第 8 节。

## 1. 安装的剩余依赖版本

本阶段新安装的主要包：

| 包 | 版本 | 安装状态 |
|---|---:|---|
| torch_geometric | 2.8.0 | 成功 |
| tqdm | 4.68.4 | 成功（安装 torch_geometric 时安装） |
| hydra-core | 1.2.0 | 安装成功，Python 3.11 导入失败 |
| omegaconf | 2.3.1 | 安装成功，因停止条件未完成导入验证 |
| scikit-learn | 1.9.0 | 安装及导入成功 |
| torchinfo | 1.8.0 | 安装成功，未纳入本轮联合 import 列表 |
| torchsummary | 1.5.1 | 安装成功，未纳入本轮联合 import 列表 |
| onnx | 1.22.0 | 安装成功，因停止条件未完成导入验证 |
| onnxruntime | 1.27.0 | 安装成功，因停止条件未完成导入验证 |

未改变的锁定包：

```text
Python 3.11.8
numpy 2.4.4
torch 2.7.1+cu128
torch_cluster 1.6.3+pt27cu128
torch_scatter 2.1.2+pt27cu128
torch_sparse 0.6.18+pt27cu128
pyg_lib 0.5.0+pt27cu128
torch_spline_conv 1.2.2+pt27cu128
scipy 1.17.1
```

## 2. 联合导入结果

成功导入并打印版本、路径：

- numpy 2.4.4
- torch 2.7.1+cu128
- torch_geometric 2.8.0
- torch_cluster 1.6.3+pt27cu128
- torch_scatter 2.1.2+pt27cu128
- torch_sparse 0.6.18+pt27cu128
- scipy 1.17.1
- scikit-learn 1.9.0

失败：

```text
hydra-core 1.2.0
ValueError: mutable default <class 'hydra.conf.JobConf.JobConfig.OverrideDirname'>
for field override_dirname is not allowed: use default_factory
```

完整 traceback 已记录于 `docs/local_environment_setup_result.md` 第 9.5 节。

失败发生后按要求停止，因此 OmegaConf、ONNX 和 ONNX Runtime 的导入未继续执行。没有发现要求触发 NumPy 降级的明确错误。

## 3. Checkpoint 元数据

本轮未使用 PyTorch 重新读取 checkpoint，因为前置依赖联合导入未通过。此前静态审计结果仍为：

```text
path: models/testParameters/GCN_res/best_model.pth
epoch: 124
train_acc: 0.9735565185546875
test_acc: 0.96408203125
class_avg_iou: 0.8890682997702295
inctance_avg_iou: 0.8890682997702295
linear_1.weight: (48, 4)
mlp.weight: (2, 48)
```

这些是 `docs/model_checkpoint_input_audit.md` 的无 PyTorch元数据审计结果，不是本轮 `torch.load()` 验收结果。

## 4. Strict load 结果

状态：**未执行**。

以下验收尚未完成：

- checkpoint 字段的 PyTorch 读取。
- `linear_1.weight == (48,4)` 运行时断言。
- `mlp.weight == (2,48)` 运行时断言。
- `model.load_state_dict(..., strict=True)`。
- checkpoint 全 tensor NaN/Inf 检查。

## 5. CUDA forward 输入输出 shape

状态：**未执行**。

预定但尚未运行：

```text
B=1, N=2048, K=6
points:     [1,2048,4] cuda:0
adj:        [1,2048,2048] cuda:0
points_xyz: [1,2048,3] cuda:0
logits:     [1,2048,2] cuda:0
```

## 6. GPU 显存峰值

状态：**未测量**。没有执行模型 forward，因此不能提供有效的 `torch.cuda.max_memory_allocated()` 数值。

## 7. 完整验收结论

```text
DEPENDENCY_JOINT_IMPORT_PASSED: NO
CHECKPOINT_STRICT_LOAD_PASSED: NOT RUN
CUDA_FORWARD_PASSED: NOT RUN
CHECKPOINT_AND_FORWARD_VALIDATION_PASSED: NOT EMITTED
```

当前唯一阻塞是严格指定的 `hydra-core==1.2.0` 在 Python 3.11.8 中导入失败。没有修改 PyTorch、CUDA、PyG 扩展、NumPy、模型、checkpoint 或项目源码。

## 8. Hydra 修复后的最终 Checkpoint/Forward 验收

### 8.1 最终环境

只将 Hydra 升级为 1.3.2，其余锁定版本全部保持：

```text
Python 3.11.8
torch 2.7.1+cu128
CUDA Runtime 12.8
torch_geometric 2.8.0
torch_cluster 1.6.3+pt27cu128
torch_scatter 2.1.2+pt27cu128
torch_sparse 0.6.18+pt27cu128
pyg_lib 0.5.0+pt27cu128
torch_spline_conv 1.2.2+pt27cu128
numpy 2.4.4
scipy 1.17.1
hydra-core 1.3.2
omegaconf 2.3.1
```

```text
pip check: No broken requirements found.
DEPENDENCY_JOINT_IMPORT_PASSED
HYDRA_COMPOSE_VALIDATION_PASSED
```

Hydra 成功组合所有顶层任务配置。目标 `partseg_v2_improved` 解析为：

```text
model.name=Nico_v2_GCN_Drop
batch_size=4
epoch=200
num_point=2048
```

compose 过程中有缺少 `_self_` 的迁移 warning，但在 `version_base="1.2"` 下无 compose 错误且目标值与 YAML 一致；没有修改配置。

### 8.2 Checkpoint 元数据

```text
path E:\GRP-PTv2\models\testParameters\GCN_res\best_model.pth
bytes 74340448
fields:
  epoch
  train_acc
  test_acc
  class_avg_iou
  inctance_avg_iou
  model_state_dict
  optimizer_state_dict

epoch 124
train_acc 0.9735565185546875
test_acc 0.96408203125
class_avg_iou 0.8890682997702295
inctance_avg_iou 0.8890682997702295
state entries 434
linear_1.weight (48, 4)
mlp.weight (2, 48)
```

### 8.3 Strict load 与 tensor 有限性

模型按以下方式构建：

```text
models.testParameters.GCN_res.model.PTV2Segmentation
cfg.num_class=2
in_dim=4
```

结果：

```text
strict load result <All keys matched successfully>
all checkpoint tensor count 1349
all checkpoint tensor numel 18483530
nonfinite tensor paths []
CHECKPOINT_STRICT_LOAD_AND_FINITE_CHECK_PASSED
```

结论：checkpoint 的模型权重、buffer 和 optimizer state 中全部 torch tensor 均为有限值。

### 8.4 CUDA forward 输入输出

固定输入条件：

```text
B=1, N=2048, K=6
model device=cuda:0
```

运行结果：

| 张量 | shape | dtype | device |
|---|---|---|---|
| points | `(1,2048,4)` | torch.float32 | cuda:0 |
| adj | `(1,2048,2048)` | torch.float32 | cuda:0 |
| points_xyz | `(1,2048,3)` | torch.float32 | cuda:0 |
| logits | `(1,2048,2)` | torch.float32 | cuda:0 |

```text
logits finite True
```

### 8.5 GPU 显存峰值

```text
forward 前已分配显存: 50,257,408 bytes
forward 前已保留显存: 81,788,928 bytes
forward 峰值已分配:   144,028,160 bytes = 137.355957 MiB
forward 增量峰值:     93,770,752 bytes = 89.426758 MiB
forward 峰值已保留:   180,355,072 bytes = 172.0 MiB
```

### 8.6 Forward 耗时

```text
CUDA Event: 619.2323608398438 ms
Wall clock: 627.8989999991609 ms
```

该数值是一次完整验证 forward，不是预热后的吞吐 benchmark。

### 8.7 最终验收

```text
DEPENDENCY_JOINT_IMPORT_PASSED
HYDRA_COMPOSE_VALIDATION_PASSED
CHECKPOINT_STRICT_LOAD_AND_FINITE_CHECK_PASSED
CHECKPOINT_AND_FORWARD_VALIDATION_PASSED
```

所有要求的 checkpoint 与 RTX 5060 CUDA forward 验收项均已通过。没有修改模型、checkpoint、训练脚本或 YAML，也没有启动正式训练。
