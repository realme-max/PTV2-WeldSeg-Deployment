# GRP-PTv2 模型、Checkpoint 与输入维度审计

审计日期：2026-07-14  
项目路径：`E:\GRP-PTv2`  
前置报告：`docs/local_python_environment_audit.md`  
操作边界：本次未安装依赖、未加载训练任务、未修改环境、模型源码、训练脚本或配置文件；只新增本报告。

## 1. 结论摘要

本轮分析已经解决“weld 输入到底是 4 维还是 19 维”的静态证据问题：

- 两个本地 checkpoint 的 `linear_1.weight` 都是 `(48, 4)`，因此它们对应的真实网络输入都是 4 维，不是 19 维。
- weld 的网络输入 `[B,N,4]` 是“归一化 XYZ 三维 + weld 类别 one-hot 一维”；TXT 的第四列点级分割标签不会作为网络特征输入。
- 19 维来自 ShapeNet part segmentation：`XYZ 3维 + 16类 category one-hot = 19维`。它不是 weld 的 normals、额外几何特征或某种已实现的工业特征融合。
- 当前默认分支 `models/Nico_v2_GCN_Drop/model.py` 是在历史 4 维 weld 分支上后改成 19 维的 ShapeNet 版本，因而与本地两个 weld checkpoint 均不兼容。
- `GCN_LFA_res` checkpoint 虽然指标字段更高并含 LFA 参数，但 forward 将 LFA 输出丢弃；checkpoint 的 Adam 状态也证明所有 LFA 参数从未得到优化器状态，即没有参与损失反传。

因此，当前短训练与后续部署应只保留一个规范基线：

> **唯一推荐模型分支：`models/testParameters/GCN_res`**  
> **唯一推荐输入：`points [B,N,4]` 与 `adj [B,N,N]`**  
> **唯一推荐 checkpoint：`models/testParameters/GCN_res/best_model.pth`**

这里的“推荐用于部署”含义是：将该分支作为唯一可信的 PyTorch 结构/数值基线，再针对 ONNX/TensorRT 做等价改造；并不表示当前源码已经可以直接无损转换 TensorRT。

## 2. 分析方法与证据边界

当前 Python 环境没有 PyTorch，不能直接执行 `torch.load()`、构造模型或调用 `load_state_dict(strict=True)`。但 `.pth` 文件是 PyTorch zip serialization 格式，本次使用 Python 标准库做了以下只读检查：

1. 计算文件大小、时间戳和 SHA-256。
2. 检查 zip 目录、`data.pkl`、byte order 和序列化版本。
3. 使用严格白名单的元数据解析器，只允许 `OrderedDict`、NumPy scalar 和 PyTorch tensor rebuild 元数据，不导入 PyTorch、不读取 tensor 数值、不允许任意 checkpoint 类执行。
4. 提取 checkpoint 顶层字段、state_dict 键、tensor shape、保存设备以及 optimizer 参数状态映射。

这已经足以确定第一层输入维度、输出类别数、模块命名和 LFA 是否获得梯度；仍需安装 PyTorch 后验证的内容包括：

- `load_state_dict(strict=True)` 是否完全通过。
- tensor 数值是否完整、是否存在 NaN/Inf。
- checkpoint 与当前 `ptv2_utils.py` 的运行行为是否一致。
- RTX 5060 上的完整 forward、显存与 CUDA 扩展行为。
- PyTorch 输出与历史指标能否复现。

由于保存逻辑只写入 `classifier.state_dict()`，checkpoint 本身没有保存 Python 模块名或模型类；“对应哪个源码”的结论来自 state_dict 指纹、同目录源码、时间戳和训练保存逻辑的联合证据。

## 3. 两个 checkpoint 的实际内容

### 3.1 `GCN_res/best_model.pth`

文件：`models/testParameters/GCN_res/best_model.pth`

```text
文件大小：74,340,448 bytes
修改时间：2024-12-06 20:44:18
SHA-256：311BDDF3607D76E6B7DED450B8419BF6AE98F34F50578608B3E6A1C1C3E58D21
序列化版本：3
byte order：little
tensor data blobs：1349
保存设备：cuda:0
```

顶层字段：

```text
epoch = 124
train_acc = 0.9735565185546875
test_acc = 0.96408203125
class_avg_iou = 0.8890682997702295
inctance_avg_iou = 0.8890682997702295
model_state_dict
optimizer_state_dict
```

关键 state_dict 指纹：

```text
state entries                         434
linear_1.weight                       (48, 4)
linear_1.bias                         (48,)
gcn_0.linear.weight                   (48, 48)
fpn_c1.weight                         (48, 48, 1)
fpn_c1_linear.weight                  (48, 48)
residual_linear.weight                (48, 48, 1)
mlp.weight                            (2, 48)
mlp.bias                              (2,)
LFA.* entries                         0
lfa_* entries                         0
```

与源码的对应关系：

- 同目录 `models/testParameters/GCN_res/model.py:78-80` 定义 `PTV2Segmentation(cfg, in_dim=4)` 和 `Linear(4,48)`。
- `models/testParameters/GCN_res/model.py:84` 注册 `gcn_0`，`:142` 在 forward 中实际调用。
- `models/testParameters/GCN_res/model.py:104-116` 定义 FPN 和 residual 模块；`:170-199` 实际把 FPN c1～c4 加到上采样特征。
- `models/testParameters/GCN_res/model.py:131` 定义 `Linear(48, num_class)`；checkpoint 的输出权重为 `(2,48)`。
- 同目录 `model.py` 修改时间为 `2024-12-06 19:34:06`，checkpoint 在约 70 分钟后保存，时间关系与实验快照相符。

结论：这是一个 4 维输入、2 类输出、实际使用一层 GCN 和 FPN 加法融合的 weld checkpoint；与同目录 `GCN_res/model.py` 高度匹配。最终“完全匹配”仍需 PyTorch strict load 确认。

### 3.2 `GCN_LFA_res/best_model.pth`

文件：`models/testParameters/GCN_LFA_res/best_model.pth`

```text
文件大小：79,883,266 bytes
修改时间：2024-12-08 00:08:09
SHA-256：638E815F8C90C14F19D29A20368886FD7A86B52FE8F3AE6591D5B48B380EE676
序列化版本：3
byte order：little
tensor data blobs：1398
保存设备：cuda:0
```

顶层字段：

```text
epoch = 138
train_acc = 0.9784317016601562
test_acc = 0.97416015625
class_avg_iou = 0.9221081736985471
inctance_avg_iou = 0.9221081736985471
model_state_dict
optimizer_state_dict
```

关键 state_dict 指纹：

```text
state entries                         483
linear_1.weight                       (48, 4)
LFA.mlp.0.weight                      (4, 8)
LFA.mlp.2.weight                      (4, 4)
gcn_0.linear.weight                   (48, 48)
lfa_0.mlp.0.weight                    (48, 96, 1)
fpn_c1.weight                         (48, 48, 1)
mlp.weight                            (2, 48)
LFA.* entries                         4
lfa_0～lfa_4 entries                  45（其中可训练参数 30）
```

表面上它含有前置 `LFA` 和五级 `lfa_0～lfa_4`，但执行图与名字不一致：

- `models/testParameters/GCN_LFA_res/model.py:202` 先执行 `out = linear_1(points_features)`。
- `models/testParameters/GCN_LFA_res/model.py:204` 随后执行 `points_features = self.LFA(points_features)`。
- `models/testParameters/GCN_LFA_res/model.py:207` 送入主干的仍是先前的 `out`，不是 LFA 更新后的 `points_features`。
- `models/testParameters/GCN_LFA_res/model.py:208,215,221,227,233` 的 `lfa_0～lfa_4` 调用全部被注释。

optimizer state 提供了进一步的训练证据：

```text
全部模型可训练参数：363
有 Adam state 的参数：305
LFA.* 参数：4；有 Adam state：0
lfa_* 参数：30；有 Adam state：0
gcn_* 参数：2；有 Adam state：2
linear_1 参数：2；有 Adam state：2
```

Adam 通常只在参数获得梯度并执行 step 后创建该参数的 state。LFA 参数在 138 个 epoch 的 checkpoint 中全部没有 optimizer state，与“输出未进入 loss 图”完全一致。因此历史注释“加入 LFA 后有提升”不能由当前执行图支持；更高的 checkpoint 指标可能来自随机初始化、数据采样或训练运行差异，不能归因于 LFA。

结论：该 checkpoint 与同目录源码的参数注册结构高度匹配，但其有效计算图本质上仍是 GCN + FPN；它额外执行了一次结果被丢弃的 O(B·N²) / Python 循环 LFA，并保存大量从未训练的参数，不应作为规范短训练或部署基线。

### 3.3 checkpoint 保存逻辑

当前改进训练入口 `train_partseg_weld_V2improved.py:282-295` 在 `inctance_avg_iou` 改善时保存：

```python
state = {
    'epoch': epoch,
    'train_acc': train_instance_acc,
    'test_acc': test_metrics['accuracy'],
    'class_avg_iou': test_metrics['class_avg_iou'],
    'inctance_avg_iou': test_metrics['inctance_avg_iou'],
    'model_state_dict': classifier.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(state, 'best_model.pth')
```

两个 checkpoint 的顶层字段与该 schema 一致。但仓库没有 Git 历史或实验日志可以证明 2024 年保存时使用的训练脚本与当前文件逐字相同，所以准确的历史命令、随机种子和数据划分仍需向原作者确认。

## 4. weld 的真实输入维度

### 4.1 原始 TXT 与网络输入不是同一个“四列”概念

原始 weld TXT 是：

```text
[x, y, z, segmentation_label]
```

`WeldDataset.__getitem__` 位于 `dataset.py:334-357`：

- `normal_channel=False` 时，`:343-344` 只取 `data[:,0:3]` 作为 `point_set`。
- `:347` 单独取 `data[:,-1]` 为 `seg` 监督标签。
- `:350` 只对 XYZ 做 `pc_normalize`。
- `:352-355` 重采样到固定 `N` 点。

Dataset 返回：

```text
point_set: [N,3]，归一化 XYZ
cls:       [1]，点云所属对象类别；weld 数据只有类别索引 0
seg:       [N]，每点 0/1 标签
```

训练入口 `train_partseg_weld_V2improved.py:190,194` 再把 `point_set` 与对象类别 one-hot 拼接：

```python
torch.cat([
    points,                                      # [B,N,3]
    to_categorical(label, 1).repeat(1,N,1)      # [B,N,1]
], dim=-1)
```

`data/weld/synsetoffset2category.txt` 只有：

```text
weld    000001
```

因此 one-hot 永远是 `1.0`。checkpoint 对应的网络输入为：

```text
points shape = [B,N,4]
feature 0..2 = 归一化 x,y,z
feature 3    = 单类别 weld one-hot，恒为 1
```

第四维不是点级 0/1 segmentation label。标签只进入 loss，不能拼入推理输入。

### 4.2 第二个输入：邻接矩阵

`GCN_res.PTV2Segmentation.forward` 的签名是 `(points, adj)`。训练入口 `train_partseg_weld_V2improved.py:47-63` 使用原始 `[B,N,3]` XYZ 通过 scikit-learn `kneighbors_graph(k=6, include_self=False)` 构造稠密邻接矩阵：

```text
adj shape = [B,N,N]
adj dtype = float32
每行约 6 个 1，其余为 0
```

于是推荐模型的真实输入接口是：

```text
points: [B,N,4]
adj:    [B,N,N]
```

输出由 `model.py:203-205` 生成：

```text
points_xyz: [B,N,3]
logits:     [B,N,2]
```

当前短训练配置 `num_point=2048`，所以固定测试实例通常是：

```text
points [B,2048,4]
adj    [B,2048,2048]
logits [B,2048,2]
```

## 5. 19 维输入的真实来源

项目中没有发现 weld 使用 normals 或额外 15 维几何特征的实现。19 维来源可由两条完整代码链确认。

### 5.1 原始 ShapeNet part segmentation 链

`train_partseg.py:56-64`：

```python
TRAIN_DATASET = PartNormalDataset(...)
args.input_dim = (6 if args.normal else 3) + 16
args.num_class = 50
num_category = 16
```

当 `normal=False`：

```text
XYZ 3 + ShapeNet category one-hot 16 = 19
```

当 `normal=True`：

```text
XYZ + normals 6 + category one-hot 16 = 22
```

### 5.2 名称误导的 ShapeNet 实验入口

`dataset_train_partseg_weld_V2improved.py` 虽然文件名含 weld，但实际活动代码是：

- `:133-136` 使用 `PartNormalDataset`，weld Dataset 行被注释。
- `:140` 使用 `(3 or 6) + 16`。
- `:143-144` 使用 50 个 part labels、16 个对象类别。

当前默认分支 `models/Nico_v2_GCN_Drop/model.py:143-148` 正是在原 4 维版本上改成：

```python
# def __init__(self, cfg, in_dim=4):
def __init__(self, cfg, in_dim=19):
    self.linear_1 = nn.Linear(19, 48)
    self.LFA = LocalFeatureAggregationBest(input_dim=19, output_dim=19, k=8)
```

同目录源码与历史 `models/testParameters/GCN_LFA_res/model.py` 的 diff 仅有少量关键改动，其中就包括 `4 -> 19`、禁用前置 LFA 和修改一个 PT block 的 K。因此 19 是后续 ShapeNet 适配遗留到默认 weld 配置的结果。

### 5.3 为什么 `args.input_dim=4` 没有修正默认模型

训练入口 `train_partseg_weld_V2improved.py:107` 设置 `args.input_dim=4`，但 `:115` 仅调用：

```python
PTV2Segmentation(args)
```

`Nico_v2_GCN_Drop.PTV2Segmentation` 没有读取 `cfg.input_dim`，而是使用构造函数默认参数 `in_dim=19`。所以 `args.input_dim=4` 只是配置对象上的一个未被该模型消费的字段，不能改变第一层。

结论：不应直接把当前 19 改成 4 并宣称问题解决。正确做法是先将 4 维 checkpoint 与它的原始 4 维源码快照绑定；本报告推荐的正是 `testParameters/GCN_res` 这一对。

## 6. 主要模型分支对比

本表只分析语义分割类 `PTV2Segmentation`，不讨论同文件里的 classifier。

“残差 FPN”指 c1～c4 的 FPN 投影被实际加到 decoder 特征；多个分支虽然注册了 `residual_linear` 并计算 `fpn_c5`，但这两部分没有被真正使用。

| 模型分支 | 默认 `in_dim` / 第一层 | forward 参数 | 实际 GCN | 实际 LFA | 残差 FPN | checkpoint 关系 | 短训练 | ONNX/TensorRT |
|---|---|---|---|---|---|---|---|---|
| `models/Nico` | 4 / `Linear(4,48)`，`:54-56` | `(points)`，`:89` | 否 | 否 | 否；仅 U-Net skip | 无本地匹配 checkpoint | 可做无 GCN baseline，但不是本项目最佳证据链 | 相对简单，但仍依赖 PTV2 `torch_cluster`/动态体素操作；无权重基线 |
| `models/Nico_v2_GCN` | 4 / `Linear(4,48)`，`:69-71` | `(points,adj)`，`:106` | **否**；`:112` 调用被注释 | 否 | 否 | 无；名字与执行图不符 | 不推荐，`adj` 传入却未使用 | 不推荐作为规范分支 |
| `models/Nico_v2_GCN_LFA` | 4 / `Linear(4,48)`，`:131-136` | `(points,adj)`，`:188` | 是，且在 PTB0 前，`:195` | **否**；`:191` 被注释 | 是，`:223-253` | 与 GCN_res 相近但多 `LFA.*`，GCN 位置也不同；不能视为精确匹配 | 可运行候选，但无本地精确 checkpoint | 外部稠密 adj + PTV2 自定义操作；死 LFA 参数增加混乱 |
| `models/Nico_v2_GCN_Drop` | **19** / `Linear(19,48)`，`:144-148` | `(points,adj)`，`:204` | 是，PTB0 后，`:213` | 否；输入 LFA 和 lfa0～4 调用均注释 | 是，`:241-270` | 是 `GCN_LFA_res` 的后续编辑版，但首层/LFA shape 已不兼容；无本地 19 维 checkpoint | **不可用于当前 weld 短训练** | 当前默认分支不应导出；“Drop”也没有活动 Dropout，`:202` 被注释 |
| `models/Nico_v2_preGCN_Drop` | 4 / `Linear(4,48)`，`:100-104` | `(points,adj)`，`:162` | 是，PTB0 后，`:171` | **是**；`:165` LFA 输出在 `:166` 进入 Linear | 是，`:199-228` | 无本地匹配 checkpoint；不能拿 LFA 未训练的 `GCN_LFA_res` 直接替代 | 不作为第一轮短训练；Python 双层循环 LFA 成本高 | 不适合直接部署，前置 LFA 含 `cdist/topk` 和按点 Python 循环 |
| `models/Nico_v2_GCN_ONNX` | 4 / `Linear(4,48)`，`:161-166`；死 LFA 仍固定 19 | `(points)`，`:222` | 是，内部计算 adj | 否；所有 LFA 调用注释 | 是，`:260-290` | 无本地匹配 checkpoint；本地两份 strict load 都会缺键或 shape 冲突 | 不适合作为训练唯一分支 | 是实验性导出分支，但 `compute_adjacency` 在 forward 中执行 `.cpu().numpy()` 和 sklearn，trace 会固化/脱离输入，当前不可靠 |
| `models/testParameters/model.py` | 4 / `Linear(4,48)`，`:54-56` | `(points)`，`:89` | 否 | 否 | 否 | 无 | 与 `Nico` 基线重复，不推荐另立分支 | 同 Nico |
| `models/testParameters/Nico` | 4 / `Linear(4,48)`，`:54-56` | `(points)`，`:89` | 否 | 否 | 否 | 无 | 重复快照，不推荐 | 同 Nico |
| `models/testParameters/res` | 4 / `Linear(4,48)`，`:143-147` | `(points,adj)`，`:202` | 否；`:211` 注释 | 否；全部调用注释 | 是，`:239-268` | 无 | 可作为 FPN 消融实验，不是当前目标 | 传入但不使用 adj，注册大量死模块，不适合作为规范部署分支 |
| `models/testParameters/GCN_res` | 4 / `Linear(4,48)`，`:78-80` | `(points,adj)`，`:135` | **是**，PTB0 后，`:142` | 否；没有注册 LFA 参数 | **是**，`:170-199` | **精确候选：`GCN_res/best_model.pth`** | **唯一推荐** | **作为唯一数值基线推荐**；仍需解决外部 adj 与 `torch_cluster` 导出 |
| `models/testParameters/GCN_LFA_res` | 4 / `Linear(4,48)`，`:141-145` | `(points,adj)`，`:200` | 是，PTB0 后，`:209` | **无有效贡献**；`:204` 结果被丢弃，lfa0～4 注释 | 是，`:237-266` | 精确候选：`GCN_LFA_res/best_model.pth` | 不推荐；额外计算和未训练参数 | 不推荐；LFA Python 循环无输出贡献且妨碍导出 |

## 7. checkpoint 与当前主要分支的兼容性

### 7.1 能确定的精确候选

| checkpoint | 唯一合理源码候选 | 证据 |
|---|---|---|
| `GCN_res/best_model.pth` | `models/testParameters/GCN_res/model.py` | 首层 `(48,4)`、GCN、20 个 FPN state、无 LFA/lfa state、输出 `(2,48)` 全部吻合；同目录源码早于 checkpoint 约 70 分钟 |
| `GCN_LFA_res/best_model.pth` | `models/testParameters/GCN_LFA_res/model.py` | 首层 `(48,4)`、4 个 LFA state、45 个 lfa state、GCN/FPN/输出全部吻合；同目录源码早于 checkpoint 约 3 小时 |

### 7.2 为什么不能把 checkpoint 直接交给当前默认分支

`GCN_LFA_res` checkpoint：

```text
linear_1.weight      (48,4)
LFA.mlp.0.weight     (4,8)
LFA.mlp.2.weight     (4,4)
```

当前 `Nico_v2_GCN_Drop` 预期：

```text
linear_1.weight      (48,19)
LFA.mlp.0.weight     (19,38)
LFA.mlp.2.weight     (19,19)
```

即便 LFA 在 forward 中被注释，`load_state_dict(strict=True)` 仍会检查这些已注册参数的 shape，因此会失败。`strict=False` 也不会自动忽略相同键名的 shape mismatch。

### 7.3 为什么不选择指标更高的 `GCN_LFA_res`

它的保存指标确实高于 `GCN_res`，但：

1. LFA 输出没有连接到后续网络。
2. 所有 LFA 参数没有 Adam state，证明没有从 loss 获得优化。
3. 五级 lfa 模块全部只注册不调用。
4. 前置 LFA 仍执行 `torch.cdist` 和双层 Python 循环，浪费训练与推理时间。
5. 两个 checkpoint 的有效主干都是 GCN + FPN，指标差异无法由 LFA 解释。

所以更高历史数值不能抵消结构不确定性。规范基线应优先选择结构干净、checkpoint 对应明确的 `GCN_res`。

## 8. ONNX/TensorRT 影响

选择 `GCN_res` 并不回避部署问题，而是先固定正确的 PyTorch 数值基线。该分支后续部署至少有两类工作：

1. `forward(points, adj)` 需要决定邻接矩阵是：
   - 由 C++/CUDA 预处理生成后作为 TensorRT 第二输入；或
   - 用可导出的 tensor-only kNN/图构建替代 sklearn；或
   - 重新定义等价的稀疏 GCN 插件。
2. `models/testParameters/GCN_res/ptv2_utils.py:68-79` 使用 `torch_cluster.grid.grid_cluster`、动态 `unique`、`scatter_reduce`、`bincount` 和 Python batch 循环。这些是 ONNX/TensorRT 的主要算子与动态 shape 风险。

当前 `Nico_v2_GCN_ONNX` 不能直接作为答案：

- `model.py:23-29` 在 forward 内把 tensor 转 CPU/NumPy 后调用 sklearn 构图；ONNX tracing 不会得到输入相关的通用 kNN 图。
- `export2ONNX.py:127-132` 只传一个 `dummy_input`，但 dynamic axes 中出现未定义的 `indices`；模型又返回 `(points_xyz, out)` 两个值，却只配置一个 output name。
- `export2ONNX.py:75` 指向原作者 D 盘的另一个 ONNX checkpoint，本机两份 checkpoint 都不是该文件。
- ONNX 分支注册的死 LFA 是 19 维，与本地 4 维 `GCN_LFA_res` checkpoint 仍有 shape 冲突。

因此后续应从 `GCN_res` 做 PyTorch 对齐测试，再建立部署适配层；不能把实验性 ONNX 分支当作模型真源。

## 9. 最终建议

### 9.1 推荐用于短训练的唯一模型分支

```text
models/testParameters/GCN_res/model.py
models/testParameters/GCN_res/ptv2_utils.py
class: PTV2Segmentation
```

理由：4 维 weld 输入明确、真实使用 GCN、真实使用 FPN、没有 LFA 死参数、存在结构指纹完全对应的本地 checkpoint，也是后续建立 PyTorch/ONNX/TensorRT 数值链最清晰的起点。

注意：当前 Hydra 配置没有直接选择这个嵌套目录的配置项，训练入口的 `shutil.copy('models/{model.name}/model.py')` 也不能直接处理带点号的嵌套模块名。短训练前需要在“允许修改代码/配置”的后续阶段设计一个明确入口；本轮没有修改。

### 9.2 推荐输入 shape

```text
points: float32 [B,N,4]
  [:,:,0:3] = 按 WeldDataset.pc_normalize 处理后的 XYZ
  [:,:,3]   = weld 对象类别 one-hot，恒为 1.0

adj: float32 [B,N,N]
  基于原始三维 points 的 k=6、include_self=False 邻接矩阵

outputs:
  points_xyz: float32 [B,N,3]
  logits:     float32 [B,N,2]
```

当前短训练建议固定 `N=2048`，先不要在第一轮同时引入动态点数。

### 9.3 推荐 checkpoint

```text
E:\GRP-PTv2\models\testParameters\GCN_res\best_model.pth
SHA-256:
311BDDF3607D76E6B7DED450B8419BF6AE98F34F50578608B3E6A1C1C3E58D21
```

使用方式建议：安装 PyTorch 后先 strict load 和 forward 验证；短训练若目标是验证训练链，可分别测试“从该 checkpoint 恢复 2～5 epoch”和“固定种子从头训练 2～5 epoch”，但两者的实验结论应分开记录。

### 9.4 仍需向师兄/原作者确认的问题

1. 两份 checkpoint 当年分别使用了哪一个训练脚本、配置文件、随机种子和原始 train/val/test JSON？
2. `GCN_LFA_res` 中 LFA 输出被丢弃是编码错误，还是当时有另一份未保存进仓库的正确源码？
3. 注释里的 `0.97451 / 0.92211` 是否就是本地 `epoch=138` checkpoint 的 test accuracy / mIoU；是否有独立复测记录？
4. 当前 `Nico_v2_GCN_Drop` 改为 19 维是否只为 ShapeNet 50 类实验？是否存在对应的 19 维 checkpoint？
5. 工业推理最终是否必须保留第 4 维恒 1 category one-hot，还是允许在重新训练后把它从模型接口中移除？对历史 checkpoint 必须保留。
6. 工业端是否已有 kNN/邻接矩阵 CUDA 实现？若没有，后续 TensorRT 是接受稠密 `adj` 第二输入，还是开发图构建插件？
7. `models/testParameters/GCN_res` 是否就是希望保留的“GCN + residual FPN 最佳模型”，还是还有未拷入仓库的日志目录和源码快照？
8. 本地两个 checkpoint 是否允许视为可信文件并用 `torch.load(..., weights_only=False)` 读取完整指标/优化器状态？

### 9.5 安装 PyTorch 后必须执行的 checkpoint 验证脚本

以下脚本只读 checkpoint，不训练、不保存文件。它先验证两份 checkpoint 与同目录源码 strict match，再验证推荐模型的第一层、输出层和一次 CUDA forward。执行前需要 PyTorch、`torch-cluster` 及项目所需扩展均已正确安装。

```python
from pathlib import Path
from types import SimpleNamespace

import torch

from models.testParameters.GCN_res.model import (
    PTV2Segmentation as GCNResModel,
)
from models.testParameters.GCN_LFA_res.model import (
    PTV2Segmentation as GCNLFAResModel,
)


ROOT = Path(r"E:\GRP-PTv2")
CFG = SimpleNamespace(num_class=2)

CASES = [
    (
        "GCN_res",
        GCNResModel,
        ROOT / "models/testParameters/GCN_res/best_model.pth",
    ),
    (
        "GCN_LFA_res",
        GCNLFAResModel,
        ROOT / "models/testParameters/GCN_LFA_res/best_model.pth",
    ),
]


def trusted_torch_load(path: Path):
    # 仅对已经确认来源可信的本地 checkpoint 使用 weights_only=False。
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # 兼容还没有 weights_only 参数的旧版 PyTorch。
        return torch.load(path, map_location="cpu")


def check_checkpoint(name, model_cls, path):
    print(f"\n=== {name} ===")
    checkpoint = trusted_torch_load(path)
    print("checkpoint fields:", list(checkpoint))
    for key in (
        "epoch",
        "train_acc",
        "test_acc",
        "class_avg_iou",
        "inctance_avg_iou",
    ):
        print(key, checkpoint.get(key))

    state_dict = checkpoint["model_state_dict"]
    print("linear_1.weight", tuple(state_dict["linear_1.weight"].shape))
    print("mlp.weight", tuple(state_dict["mlp.weight"].shape))

    assert tuple(state_dict["linear_1.weight"].shape) == (48, 4)
    assert tuple(state_dict["mlp.weight"].shape) == (2, 48)

    model = model_cls(CFG, in_dim=4)
    result = model.load_state_dict(state_dict, strict=True)
    print("strict load:", result)

    bad = [
        key
        for key, value in state_dict.items()
        if torch.is_tensor(value) and not torch.isfinite(value).all()
    ]
    assert not bad, f"non-finite checkpoint tensors: {bad}"
    return checkpoint, model


loaded = {}
for case in CASES:
    checkpoint, model = check_checkpoint(*case)
    loaded[case[0]] = (checkpoint, model)


# 推荐分支的实际 CUDA forward。
assert torch.cuda.is_available()
device = torch.device("cuda:0")
model = loaded["GCN_res"][1].to(device).eval()

B, N, K = 1, 2048, 6
xyz = torch.randn(B, N, 3, device=device)
xyz = xyz - xyz.mean(dim=1, keepdim=True)
scale = torch.linalg.vector_norm(xyz, dim=-1).amax(dim=1, keepdim=True)
xyz = xyz / scale.unsqueeze(-1).clamp_min(1e-12)
category_one_hot = torch.ones(B, N, 1, device=device)
points = torch.cat([xyz, category_one_hot], dim=-1)

# 仅用于 smoke test 的 tensor-only kNN 邻接矩阵。
distance = torch.cdist(xyz, xyz)
neighbor_index = distance.topk(K + 1, largest=False).indices[:, :, 1:]
adj = torch.zeros(B, N, N, dtype=points.dtype, device=device)
adj.scatter_(2, neighbor_index, 1.0)

with torch.inference_mode():
    points_xyz, logits = model(points, adj)
    torch.cuda.synchronize()

print("points input:", points.shape, points.dtype, points.device)
print("adj input:", adj.shape, adj.dtype, adj.device)
print("points_xyz output:", points_xyz.shape)
print("logits output:", logits.shape)
print("finite logits:", torch.isfinite(logits).all().item())

assert points.shape == (B, N, 4)
assert adj.shape == (B, N, N)
assert points_xyz.shape == (B, N, 3)
assert logits.shape == (B, N, 2)
assert torch.isfinite(logits).all()

print("\nCHECKPOINT_AND_FORWARD_VALIDATION_PASSED")
```

只有在推荐 checkpoint 对 `GCN_res` 完成 strict load、CUDA forward 输出 `[B,2048,2]` 且数值有限后，才应进入 2～5 epoch 短训练阶段。
