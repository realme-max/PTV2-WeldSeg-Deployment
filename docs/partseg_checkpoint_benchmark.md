# PartSeg checkpoints 统一评估报告

## 1. 最终结论

本轮扫描 `models/partseg/*/best_model.pth`，共发现 7 个 checkpoint：

- 6 个满足当前焊缝输入输出契约并完成固定 val/test evaluation；
- 1 个为 19 维输入、50 类输出，标记为 `incompatible_with_weld_input`，未强行修改或评估；
- 0 个 compatible checkpoint 加载或推理失败；
- 所有 checkpoint tensor 均未发现 NaN/Inf；
- 所有已评估模型均 `load_state_dict(strict=True)` 成功，没有随机权重回退。

按用户指定的排序规则——第一优先 test mIoU、第二优先 test weld F1、第三优先 ONNX 部署难度——最终推荐继续使用：

```text
models/testParameters/GCN_res/best_model.pth
```

GCN_res 的 test mIoU 为 `0.936309`，test weld F1 为 `0.946799`，均高于 partseg 第一名 `Nico_v2_GCN_LFA` 的 `0.912241 / 0.926708`。

最终状态：

```text
PARTSEG_CHECKPOINT_BENCHMARK_COMPLETED
```

## 2. 固定评估协议

- 数据：`E:\GRP-PTv2\data\weld`
- val：`sub_shuffled_val_file_list.json`，18 个样本
- test：`sub_shuffled_test_file_list.json`，18 个样本
- seed：42
- num points：2048
- batch size：1
- KNN adjacency：k=6
- device：`cuda:0` / RTX 5060
- dtype：FP32
- 输入：normalized XYZ + 恒为 1 的 weld category one-hot，即 `[1,2048,4]`
- 标签：class 0 = `weld_seam`，class 1 = `background`
- loss：CrossEntropyLoss
- 只执行 `model.eval()` + `torch.inference_mode()`，没有训练或微调。

数据加载、归一化和确定性采样复用已验证的 `scripts/evaluate_gcn_res_checkpoint.py::FixedWeldEvaluationDataset`。GCN_res 一行直接读取其最终验证 baseline：

`artifacts/gcn_res_evaluation/20260714_160831_945091_historical_checkpoint/metrics.json`

没有重复训练或改写 baseline。

## 3. 实际命令与产物

脚本：`scripts/evaluate_all_partseg_checkpoints.py`

执行命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe `
  E:\GRP-PTv2\scripts\evaluate_all_partseg_checkpoints.py `
  --run-id 20260715_094500_fixed_sub_splits_final
```

成功标志：

```text
PARTSEG_CHECKPOINT_BENCHMARK_COMPLETED
```

产物目录：

`artifacts/partseg_checkpoint_benchmark/20260715_094500_fixed_sub_splits_final/`

其中包括：

- `benchmark.json`：checkpoint 元数据、完整 val/test 指标、模型来源、strict load、部署难度和排名；
- `summary.csv`：统一汇总表；
- `run.log`：每个样本的 evaluation 日志。

## 4. 模型和 checkpoint 结构审计

所有 checkpoint 保存目录都存在 `model.py`，类名均为 `PTV2Segmentation`。但是这些保存目录都没有同时保存 `ptv2_utils.py`，不能脱离其他目录独立复现。

| Model | Saved model.py | Forward | Input | Output | Parameters | state_dict keys | Status |
|---|---|---|---:|---:|---:|---:|---|
| `Nico_v2_GCN_Drop` | `models/partseg/Nico_v2_GCN_Drop/model.py` | checkpoint 不兼容，未调用 | 19 | 50 | 7,576,275 | 483 | `incompatible_with_weld_input` |
| `Nico_v2_GCN_L` | `models/partseg/Nico_v2_GCN_L/model.py` | `(points, adj)` | 4 | 2 | 8,328,898 | 452 | `evaluated` |
| `Nico_v2_GCN_LFA` | `models/partseg/Nico_v2_GCN_LFA/model.py` | `(points, adj)` | 4 | 2 | 6,193,258 | 438 | `evaluated` |
| `Nico_v2_GCN_L_Res` | `models/partseg/Nico_v2_GCN_L_Res/model.py` | `(points, adj)` | 4 | 2 | 6,193,202 | 434 | `evaluated` |
| `Nico_v2_GCN_ONNX` | `models/partseg/Nico_v2_GCN_ONNX/model.py` | `(points)` | 4 | 2 | 7,573,203 | 483 | `evaluated` |
| `Nico_v2_GCN_Res_LFA` | `models/partseg/Nico_v2_GCN_Res_LFA/model.py` | `(points, adj)` | 4 | 2 | 6,193,258 | 438 | `evaluated` |
| `Nico_v2_preGCN_Drop` | `models/partseg/Nico_v2_preGCN_Drop/model.py` | `(points, adj)` | 4 | 2 | 6,193,258 | 438 | `evaluated` |
| `GCN_res` baseline | `models/testParameters/GCN_res/model.py` | `(points, adj)` | 4 | 2 | 6,193,202 | 434 | `existing_verified_baseline` |

这里的 Input/Output 来自 checkpoint 的 `linear_1.weight` 和 `mlp.weight`，不是仅根据构造函数默认值猜测。Parameters 是模型 trainable parameter 数量；state_dict keys 包含参数和 BatchNorm buffers。

### 4.1 Nico_v2_GCN_Drop 不兼容原因

该 checkpoint 的真实形状是：

```text
linear_1.weight = [48,19]
mlp.weight      = [50,48]
```

它是 19 维输入、50 类输出的历史 checkpoint，不是当前 4 维、2 类焊缝 checkpoint，因此没有尝试补零、裁剪输入、替换 head 或非严格加载。

保存目录的 `model.py` 已与 checkpoint 结构不一致；当前 `models/Nico_v2_GCN_Drop/model.py` 可以按 19→50 严格加载，参数数为 7,576,275。脚本只用该训练模块核实参数数和 strict load，绝未用它评估焊缝数据。

### 4.2 模型来源完整性

| Model | saved model.py 与当前 `models/<name>/model.py` | ptv2_utils 来源 |
|---|---|---|
| `Nico_v2_GCN_Drop` | 不同 | 不评估 |
| `Nico_v2_GCN_L` | 相同 | 对应训练模块 |
| `Nico_v2_GCN_LFA` | 不同 | 对应训练模块 |
| `Nico_v2_GCN_L_Res` | 相同 | 对应训练模块 |
| `Nico_v2_GCN_ONNX` | 相同 | 对应训练模块 |
| `Nico_v2_GCN_Res_LFA` | 当前根模型目录已不存在 | 推断使用 sibling `Nico_v2_GCN_LFA/ptv2_utils.py` |
| `Nico_v2_preGCN_Drop` | 相同 | 对应训练模块 |

评估始终执行 checkpoint 目录保存的 `model.py`。当其中相对导入的 `.ptv2_utils` 缺失时，脚本显式加载表中的 utility 文件并在 JSON 中记录路径与 SHA-256。

`Nico_v2_GCN_Res_LFA` 的原始根模型/utility 已缺失，其结果依赖 sibling utility 推断，复现证据弱于其他模型。即使接受该推断，其 test mIoU 也只有 `0.901462`，不会影响最终第一名。

## 5. Val 指标

| Model | Loss | Accuracy | Weld seam IoU | Background IoU | mIoU | Weld Precision | Weld Recall | Weld F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Nico_v2_GCN_L` | 0.132370 | 0.947890 | 0.804538 | 0.933660 | 0.869099 | 0.858617 | 0.927399 | 0.891683 |
| `Nico_v2_GCN_LFA` | 0.110681 | 0.957981 | 0.841161 | 0.945954 | 0.893558 | 0.869976 | 0.962116 | 0.913729 |
| `Nico_v2_GCN_L_Res` | 0.109347 | 0.956923 | 0.834962 | 0.944918 | 0.889940 | 0.879956 | 0.942294 | 0.910059 |
| `Nico_v2_GCN_ONNX` | 2.105306 | 0.291097 | 0.245910 | 0.077941 | 0.161925 | 0.245938 | 0.999531 | 0.394747 |
| `Nico_v2_GCN_Res_LFA` | 0.127290 | 0.947754 | 0.807169 | 0.933130 | 0.870149 | 0.846493 | 0.945578 | 0.893296 |
| `Nico_v2_preGCN_Drop` | 0.105988 | 0.956028 | 0.833059 | 0.943666 | 0.888363 | 0.872317 | 0.948745 | 0.908927 |
| **`GCN_res` baseline** | 0.109621 | **0.959961** | 0.838530 | **0.949450** | **0.893990** | **0.925725** | 0.899015 | 0.912174 |

Val 上 `Nico_v2_GCN_LFA` 的 weld IoU、recall 和 weld F1 略高于 GCN_res，但 GCN_res 的 overall accuracy、background IoU、mIoU 和 weld precision 更高。最终排序第一指标是 test mIoU，不按 val 单项选择。

## 6. Test 指标

| Model | Loss | Accuracy | Weld seam IoU | Background IoU | mIoU | Weld Precision | Weld Recall | Weld F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Nico_v2_GCN_L` | 0.089555 | 0.968180 | 0.859386 | 0.960498 | 0.909942 | 0.898596 | 0.951679 | 0.924376 |
| `Nico_v2_GCN_LFA` | 0.084320 | 0.968750 | 0.863426 | 0.961056 | 0.912241 | 0.889798 | **0.966813** | 0.926708 |
| `Nico_v2_GCN_L_Res` | 0.083506 | 0.967773 | 0.860021 | 0.959817 | 0.909919 | 0.884406 | **0.968937** | 0.924743 |
| `Nico_v2_GCN_ONNX` | 2.327901 | 0.272054 | 0.219186 | 0.085098 | 0.152142 | 0.219186 | 1.000000 | 0.359562 |
| `Nico_v2_GCN_Res_LFA` | 0.090917 | 0.964464 | 0.847177 | 0.955746 | 0.901462 | 0.874834 | 0.964025 | 0.917267 |
| `Nico_v2_preGCN_Drop` | 0.082124 | 0.967773 | 0.857878 | 0.959991 | 0.908934 | 0.896711 | 0.951945 | 0.923503 |
| **`GCN_res` baseline** | **0.062328** | **0.978651** | **0.898973** | **0.973645** | **0.936309** | **0.964601** | 0.929643 | **0.946799** |

较高 weld recall 不代表综合性能更好。部分 LFA 模型通过预测更多焊缝点获得高 recall，但 precision、weld IoU、background IoU 和整体 mIoU 下降。GCN_res 在 test 上的 weld recall 不是最高，但 precision、IoU、F1 和 mIoU 明显最佳。

## 7. 统一汇总与排序

汇总表中的 `weld_f1` 指 test weld F1。

| Rank | model_name | checkpoint_path | input_dim | params | val_mIoU | test_mIoU | weld_f1 | ONNX ease | status |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | **`GCN_res_reference_baseline`** | `models/testParameters/GCN_res/best_model.pth` | 4 | 6,193,202 | 0.893990 | **0.936309** | **0.946799** | difficult | `existing_verified_baseline` |
| 2 | `Nico_v2_GCN_LFA` | `models/partseg/Nico_v2_GCN_LFA/best_model.pth` | 4 | 6,193,258 | 0.893558 | 0.912241 | 0.926708 | very difficult | `evaluated` |
| 3 | `Nico_v2_GCN_L` | `models/partseg/Nico_v2_GCN_L/best_model.pth` | 4 | 8,328,898 | 0.869099 | 0.909942 | 0.924376 | difficult | `evaluated` |
| 4 | `Nico_v2_GCN_L_Res` | `models/partseg/Nico_v2_GCN_L_Res/best_model.pth` | 4 | 6,193,202 | 0.889940 | 0.909919 | 0.924743 | difficult | `evaluated` |
| 5 | `Nico_v2_preGCN_Drop` | `models/partseg/Nico_v2_preGCN_Drop/best_model.pth` | 4 | 6,193,258 | 0.888363 | 0.908934 | 0.923503 | very difficult | `evaluated` |
| 6 | `Nico_v2_GCN_Res_LFA` | `models/partseg/Nico_v2_GCN_Res_LFA/best_model.pth` | 4 | 6,193,258 | 0.870149 | 0.901462 | 0.917267 | very difficult | `evaluated` |
| 7 | `Nico_v2_GCN_ONNX` | `models/partseg/Nico_v2_GCN_ONNX/best_model.pth` | 4 | 7,573,203 | 0.161925 | 0.152142 | 0.359562 | very difficult | `evaluated` |
| — | `Nico_v2_GCN_Drop` | `models/partseg/Nico_v2_GCN_Drop/best_model.pth` | 19 | 7,576,275 | — | — | — | not ranked | `incompatible_with_weld_input` |

排名 3 和 4 严格遵循第一排序项：`Nico_v2_GCN_L` 的 test mIoU `0.909942319` 略高于 `Nico_v2_GCN_L_Res` 的 `0.909919280`。只有 test mIoU 相同时才比较 weld F1。

## 8. ONNX 部署难度

本轮没有导出 ONNX，以下仅是基于实际代码的静态部署难度，用作第三排序项：

| Model | 难度 | 主要证据 |
|---|---|---|
| `GCN_res` | difficult | 原始 voxel pooling 使用 torch_cluster；标准算子版本的人工 pooling 已通过，但全模型逐层 parity 尚未通过 |
| `Nico_v2_GCN_L` | difficult | PTV2 utility 仍依赖 torch_cluster voxel pooling |
| `Nico_v2_GCN_L_Res` | difficult | PTV2 utility 仍依赖 torch_cluster voxel pooling |
| `Nico_v2_GCN_LFA` | very difficult | torch_cluster，加上 active LFA Python batch/point 循环 |
| `Nico_v2_GCN_Res_LFA` | very difficult | torch_cluster、active LFA Python 循环，且原始 utility 来源缺失 |
| `Nico_v2_preGCN_Drop` | very difficult | torch_cluster，加上前置 LFA Python batch/point 循环 |
| `Nico_v2_GCN_ONNX` | very difficult | `forward(points)` 内部执行 sklearn adjacency、CPU NumPy round-trip；其名称含 ONNX 不代表可直接部署 |

`Nico_v2_GCN_ONNX` 不仅精度最低，而且把邻接图构建放在模型内部的 Python/sklearn 路径，不符合当前双输入 `points + adj` 的部署契约，因此不能因为目录名带 ONNX 就优先选择。

## 9. 最终部署模型建议

最终模型仍选择：

```text
Model source: models/testParameters/GCN_res/model.py
Checkpoint:   models/testParameters/GCN_res/best_model.pth
Input:        points [1,2048,4], adj [1,2048,2048]
Output:       logits [1,2048,2]
```

理由：

1. test mIoU 排名第一，比 partseg 第一名高 `0.024068`。
2. test weld F1 排名第一，比 partseg 第一名高 `0.020091`。
3. test accuracy、weld IoU、background IoU 和 weld precision 全部第一。
4. 已有最完整的 checkpoint、PyTorch baseline、逐样本结果、性能测试和标签语义证据。
5. 尽管当前 ONNX 标准算子全模型 parity 尚未通过，但问题已经定位到 CUDA reduction 数值传播；其他候选没有展示更低的部署风险，也没有更高精度。

本轮没有进入 ONNX，没有修改 deployment，没有修改任何模型、checkpoint、配置、数据或划分 JSON。

## 10. 验收标志

```text
PARTSEG_CHECKPOINT_BENCHMARK_COMPLETED
```
