# GCN_res 历史 checkpoint 推理与评估基线

## 1. 结论摘要

历史 `GCN_res` checkpoint 已在固定 weld 子数据集 val/test 划分上完成 PyTorch CUDA 推理、指标计算、逐样本导出、性能基准和复现性验证。

```text
GCN_RES_CHECKPOINT_EVALUATION_PASSED
```

- 模型：`models/testParameters/GCN_res/model.py`
- checkpoint：`models/testParameters/GCN_res/best_model.pth`
- 运行 ID：`20260714_160831_945091_historical_checkpoint`
- 产物目录：`artifacts/gcn_res_evaluation/20260714_160831_945091_historical_checkpoint/`
- val：18 个样本，mIoU 0.893990，overall accuracy 0.959961。
- test：18 个样本，mIoU 0.936309，overall accuracy 0.978651。
- 已确认标签语义：class 0 / `weld_seam` 为焊缝，class 1 / `background` 为背景。
- val 焊缝 F1 为 0.912174；test 焊缝 F1 为 0.946799。
- 整体 GPU 峰值显存：122.363281 MiB。
- checkpoint 重新加载后同一样本推理两次：全部输出有限，`allclose(rtol=1e-5, atol=1e-6)` 通过，最大绝对差 `2.3841858e-6`。

没有修改原始模型、Dataset、JSON、checkpoint 或训练脚本。

## 2. 新增文件与实际命令

新增：

- `scripts/evaluate_gcn_res_checkpoint.py`
- `docs/gcn_res_checkpoint_evaluation.md`

静态检查：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -c "from pathlib import Path; compile(Path(r'E:\GRP-PTv2\scripts\evaluate_gcn_res_checkpoint.py').read_text(encoding='utf-8'), r'E:\GRP-PTv2\scripts\evaluate_gcn_res_checkpoint.py', 'exec'); print('SYNTAX_PASSED')"
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -c "import runpy; runpy.run_path(r'E:\GRP-PTv2\scripts\evaluate_gcn_res_checkpoint.py', run_name='not_main'); print('SCRIPT_IMPORT_PASSED')"
```

结果：

```text
SYNTAX_PASSED
SCRIPT_IMPORT_PASSED
```

评估命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe E:\GRP-PTv2\scripts\evaluate_gcn_res_checkpoint.py
```

## 3. 固定配置

| 项目 | 值 |
|---|---|
| seed | 42 |
| num_point | 2048 |
| k_neighbors | 6 |
| batch_size | 1 |
| num_workers | 0 |
| device | `cuda:0` |
| val JSON | `sub_shuffled_val_file_list.json`，18 个 |
| test JSON | `sub_shuffled_test_file_list.json`，18 个 |
| points | `[1,2048,4]` |
| adjacency | `[1,2048,2048]` |
| logits | `[1,2048,2]` |

val/test JSON 顺序保持不变，两个集合无交叠。每个样本使用固定 seed、split offset 和样本索引执行有放回采样，因此后续 PyTorch/ONNX/TensorRT 对齐可以复用相同 2048 点及 `sample_indices`。

模型第 4 维输入是恒为 1 的 weld 对象类别 one-hot；前三维是中心化并按最大点半径缩放后的 XYZ。

## 4. 标签语义核对

正式确认后的统一映射是：

- **class 0 / `weld_seam`：TXT 第 4 列标签 0，表示焊缝。**
- **class 1 / `background`：TXT 第 4 列标签 1，表示背景。**

| 模型类别 | 语义名称 | 数据含义 |
|---|---|---|
| class 0 / `weld_seam` | 焊缝 | TXT 第 4 列原始整数标签 `0` |
| class 1 / `background` | 背景 | TXT 第 4 列原始整数标签 `1` |

代码与数据读取证据：

- `dataset.py:342-347` 读取 TXT，并把最后一列原样转换为逐点 `seg`。
- `dataset.py:325` 定义 `self.seg_classes = {'weld': [0, 1]}`。
- `train_partseg_weld_V2improved.py:31` 同样定义 weld 分割标签集合 `[0,1]`。

正式语义由 CloudCompare 视觉证据确认：

- CloudCompare 直接加载 TXT 第 4 列作为 Scalar Field。
- 色标显示 0 对应蓝色，1 对应红色。
- 蓝色窄带区域为空间中的焊缝。
- 红色大面积区域为空间中的背景。
- 因此 `label 0 = weld_seam`，`label 1 = background`。

原始评估把 class 1 / `background` 当作二分类正类，所以原 `precision`、`recall`、`f1` 实际是背景指标。本次后处理没有删除这些数值，而是将它们明确迁移为 `background_precision`、`background_recall`、`background_f1`，并根据现有混淆矩阵补算焊缝指标。

## 5. checkpoint 加载结果

checkpoint 使用 `load_state_dict(strict=True)` 加载，结果：

```text
<All keys matched successfully>
```

| 字段 | 值 |
|---|---:|
| epoch | 124 |
| train_acc | 0.9735565186 |
| test_acc | 0.9640820313 |
| class_avg_iou | 0.8890682998 |
| inctance_avg_iou | 0.8890682998 |
| `linear_1.weight` | `[48,4]` |
| `mlp.weight` | `[2,48]` |
| 所有 checkpoint tensor 有限 | 是 |

加载失败时脚本会抛出异常并停止，不存在随机初始化后继续评估的分支。

## 6. val/test 汇总指标

混淆矩阵按“行=ground truth，列=prediction”排列。背景指标保留原始 class 1 / `background` 正类计算结果；焊缝指标以 class 0 / `weld_seam` 为正类补算。

| Split | Avg loss | Overall accuracy | Weld seam IoU | Background IoU | mIoU | Weld seam Precision | Weld seam Recall | Weld seam F1 | Background Precision | Background Recall | Background F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.109621 | 0.959961 | 0.838530 | 0.949450 | 0.893990 | 0.925725 | 0.899015 | 0.912174 | 0.969878 | 0.978298 | 0.974070 |
| test | 0.062328 | 0.978651 | 0.898973 | 0.973645 | 0.936309 | 0.964601 | 0.929643 | 0.946799 | 0.982097 | 0.991238 | 0.986646 |

以 class 0 / `weld_seam` 为正类时：

```text
TP = confusion_matrix[0,0]
FN = confusion_matrix[0,1]
FP = confusion_matrix[1,0]
TN = confusion_matrix[1,1]
```

- val：TP=7665、FN=861、FP=615、TN=27723；焊缝 Precision=0.925725、Recall=0.899015、F1=0.912174。
- test：TP=7003、FN=530、FP=257、TN=29074；焊缝 Precision=0.964601、Recall=0.929643、F1=0.946799。

val confusion matrix：

```text
                         predicted weld_seam  predicted background
ground truth weld_seam                  7665                   861
ground truth background                  615                 27723
```

test confusion matrix：

```text
                         predicted weld_seam  predicted background
ground truth weld_seam                  7003                   530
ground truth background                  257                 29074
```

val 共评估 36,864 点；test 共评估 36,864 点。

正类语义的修正不会改变以下指标：overall accuracy、class 0 / weld seam IoU、class 1 / background IoU、mIoU 和 confusion matrix。它只改变 Precision/Recall/F1 的命名与要观察的正类，并新增了焊缝正类指标。

## 7. 逐样本结果与预测 NPZ

`per_sample_metrics.csv` 共 36 行：val 18 行、test 18 行。字段包含：

- `sample_name`
- `split`
- `loss`
- `accuracy`
- `weld_seam_iou`
- `background_iou`
- `miou`
- `predicted_background_points`
- `ground_truth_background_points`
- `weld_seam_precision`
- `weld_seam_recall`
- `weld_seam_f1`
- `background_precision`
- `background_recall`
- `background_f1`

原 CSV 中的 `precision`、`recall`、`f1` 数值按 class 1 / `background` 正类计算，现已原值迁移为 `background_precision`、`background_recall`、`background_f1`。逐样本焊缝指标通过原背景 TP、预测背景点数、真值背景点数和固定 2048 点重建每个样本的混淆矩阵后计算；36 个样本按 split 汇总得到的混淆矩阵与原 val/test 汇总矩阵完全一致。

按 JSON 原始顺序固定导出的 6 个样本：

- val：`weld_7`、`weld_61`、`weld_49`
- test：`weld_65`、`weld_30`、`weld_28`

每个 NPZ 包含：

| 字段 | shape |
|---|---|
| `original_xyz` | `[2048,3]` |
| `normalized_xyz` | `[2048,3]` |
| `ground_truth_labels` | `[2048]` |
| `predicted_labels` | `[2048]` |
| `class_1_probability` | `[2048]`；等价语义为 `background_probability` |
| `logits` | `[2048,2]` |
| `sample_indices` | `[2048]` |
| `sample_name` | scalar |
| `split` | scalar |

现有 NPZ 没有重新生成或修改，其中 `class_1_probability = background_probability`。为保持本次 PyTorch 基线文件内容及校验对象不变，字段名仍是 `class_1_probability`；后续生成新格式 NPZ 时建议改名为 `background_probability`。NPZ 数值有限值检查通过。保存 `sample_indices` 是为了在 ONNX/TensorRT 阶段直接复用相同采样点，避免采样差异污染数值对齐结果。

## 8. 推理性能基准

固定样本：val 第一个样本 `weld_7`。预热 10 轮，正式统计 50 轮，单位均为毫秒。

| 阶段 | Mean | P50/Median | P95 | Min | Max |
|---|---:|---:|---:|---:|---:|
| CPU 邻接矩阵 | 24.624 | 24.677 | 26.989 | 20.272 | 27.831 |
| Host-to-Device | 1.401 | 1.369 | 1.544 | 1.340 | 1.692 |
| 纯 CUDA model forward | 17.557 | 17.202 | 21.123 | 15.554 | 24.874 |
| 完整单样本端到端 | 52.217 | 51.379 | 58.157 | 47.429 | 60.169 |

计时边界：

- CPU 邻接：`sklearn.neighbors.kneighbors_graph` 和 dense float32 转换。
- H2D：points 与 `[1,2048,2048]` 稠密邻接矩阵的 CUDA Event 拷贝时间。
- 纯 forward：只包含 `GCN_res(points, adj)`，不包含有限值检查、softmax 或后处理。
- 端到端：包含 TXT 读取、固定采样、归一化、邻接构建、pin memory、H2D、模型 forward、argmax 和 D2H。

正式评估 GPU 峰值显存为 122.363281 MiB；benchmark 统计阶段峰值为 122.218262 MiB。

## 9. 可复现性验证

评估结束后重新实例化 `GCN_res`，再次对历史 checkpoint 执行 strict 加载，并对固定 val 样本 `weld_7` 连续推理两次。

| 检查 | 结果 |
|---|---|
| strict reload | 所有 key 匹配 |
| 两次输出均无 NaN/Inf | 是 |
| bitwise exact | 否 |
| `allclose(rtol=1e-5, atol=1e-6)` | 是 |
| 最大绝对差 | `2.384185791015625e-6` |

RTX 5060 上两次 CUDA 输出存在极小的浮点调度差异，但处于明确记录的合理容差内。后续 ONNX/TensorRT 对齐不应错误地要求 bitwise equality，建议同时报告最大绝对误差、最大相对误差、预测标签一致率和 background probability（NPZ 旧字段 `class_1_probability`）误差。

## 10. 产物清单

运行目录：

```text
artifacts/gcn_res_evaluation/20260714_160831_945091_historical_checkpoint/
```

包含：

- `metrics.json`
- `confusion_matrix.csv`
- `per_sample_metrics.csv`
- `benchmark.json`
- `config_resolved.yaml`
- `run.log`
- `predictions/val_00_weld_7.npz`
- `predictions/val_01_weld_61.npz`
- `predictions/val_02_weld_49.npz`
- `predictions/test_00_weld_65.npz`
- `predictions/test_01_weld_30.npz`
- `predictions/test_02_weld_28.npz`

这套产物可直接作为后续 ONNX Runtime 与 TensorRT 的固定输入、PyTorch logits/probability 参考和性能基线。

## 11. 标签语义修正记录

本次只执行现有评估结果的语义修正和基于混淆矩阵的指标后处理，没有重新训练、重新推理或修改任何 logits、预测标签、NPZ、checkpoint、模型源码、数据或划分 JSON。

修改文件：

- `docs/gcn_res_checkpoint_evaluation.md`
- `artifacts/gcn_res_evaluation/20260714_160831_945091_historical_checkpoint/metrics.json`
- `artifacts/gcn_res_evaluation/20260714_160831_945091_historical_checkpoint/per_sample_metrics.csv`
- `artifacts/gcn_res_evaluation/20260714_160831_945091_historical_checkpoint/confusion_matrix.csv`
- `artifacts/gcn_res_evaluation/20260714_160831_945091_historical_checkpoint/config_resolved.yaml`

汇总字段迁移：

- `class_0_iou` → `weld_seam_iou`
- `class_1_iou` → `background_iou`
- `precision` → `background_precision`
- `recall` → `background_recall`
- `f1` → `background_f1`
- 新增 `weld_seam_precision`、`weld_seam_recall`、`weld_seam_f1`

逐样本 CSV 还将 `predicted_positive_points`、`ground_truth_positive_points` 明确重命名为 `predicted_background_points`、`ground_truth_background_points`。

兼容策略：没有在 `metrics.json` 和逐样本 CSV 中重复保留含糊旧字段；旧字段的数值完整保存在新的语义字段中，`metrics.json.metric_field_migration` 记录了逐项迁移关系及 `legacy_ambiguous_fields_retained=false`。原始运行日志作为执行时的历史记录保持不变，其语义说明由本报告、更新后的 `metrics.json` 和 `config_resolved.yaml` 覆盖。

最终焊缝指标：

- val：Precision 0.925725，Recall 0.899015，F1 0.912174。
- test：Precision 0.964601，Recall 0.929643，F1 0.946799。

```text
LABEL_SEMANTICS_AND_WELD_METRICS_CORRECTED
```
