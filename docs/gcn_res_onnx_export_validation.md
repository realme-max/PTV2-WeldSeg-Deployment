# GCN_res 固定 ONNX 导出与 ONNX Runtime 对齐报告

## 1. 本轮结论

第一版固定 ONNX 导出在 opset 17 阶段失败，并已按停止条件终止后续流程。

```text
GCN_RES_ONNX_EXPORT_FAILED
GCN_RES_ONNXRUNTIME_PARITY_NOT_RUN
```

实际首个阻塞算子：

```text
torch_cluster::grid
```

完整异常：

```text
torch.onnx.errors.UnsupportedOperatorError: ONNX export failed on an operator
with unrecognized namespace torch_cluster::grid. If you are trying to export
a custom operator, make sure you registered it with the right domain and version.
```

这不是 checkpoint、输入 shape、CUDA 或 wrapper 错误。固定 PyTorch 前向已经成功，失败发生在 `torch.onnx.export` 将 Torch IR 转换为标准 ONNX 节点时。

本轮没有：

- 修改 `models/testParameters/GCN_res/model.py` 或 `ptv2_utils.py`；
- 使用 `Nico_v2_GCN_ONNX` 作为模型真源；
- 注册仓库旧代码中的自定义 GridCluster symbolic；
- 启用 ATen fallback；
- 把 voxel/grid 聚类结果静默固化为常量；
- 尝试 opset 18；
- 运行 ONNX Runtime；
- 进入 TensorRT。

因此两个成功标志均未输出：

```text
GCN_RES_ONNX_EXPORT_PASSED                 # 未达到
GCN_RES_ONNXRUNTIME_PARITY_PASSED          # 未执行
```

## 2. 新增文件

- `deployment/gcn_res_onnx_wrapper.py`
- `scripts/export_gcn_res_onnx.py`
- `scripts/validate_gcn_res_onnxruntime.py`
- `docs/gcn_res_onnx_export_validation.md`

静态检查结果：

```text
SYNTAX_PASSED
SCRIPT_IMPORT_PASSED
```

wrapper 禁止项扫描没有发现 sklearn、NumPy 转换、CPU KNN、重新采样、softmax 或 argmax。

## 3. wrapper 接口

`deployment/gcn_res_onnx_wrapper.py:9-18` 的 `GCNResOnnxWrapper` 只执行：

```python
_, logits = self.model(points, adj)
return logits
```

固定接口设计：

| 类型 | 名称 | dtype | shape |
|---|---|---|---|
| input | `points` | float32 | `[1,2048,4]` |
| input | `adj` | float32 | `[1,2048,2048]` |
| output | `logits` | float32 | `[1,2048,2]` |

`points_xyz` 只在 wrapper 边界被丢弃，原始 `GCN_res.PTV2Segmentation` 的内部执行和参数没有变化。邻接矩阵仍是外部输入，wrapper 内没有构建 KNN。

## 4. 固定导出执行

实际命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe E:\GRP-PTv2\scripts\export_gcn_res_onnx.py
```

运行 ID：

```text
20260714_165954_346592_fixed_b1_n2048
```

运行目录：

```text
artifacts/gcn_res_onnx/20260714_165954_346592_fixed_b1_n2048/
```

导出前已通过：

- Python：`E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe`
- PyTorch：`2.7.1+cu128`
- CUDA Runtime：12.8
- GPU：NVIDIA GeForce RTX 5060，capability `(12,0)`
- checkpoint：`strict=True`，所有 key 匹配
- checkpoint `linear_1.weight`：`[48,4]`
- checkpoint `mlp.weight`：`[2,48]`
- 输入 points：`float32 [1,2048,4]`，全部有限
- 输入 adj：`float32 [1,2048,2048]`，全部有限，共 12,288 个非零邻接项，即 `2048×6`
- PyTorch logits：`float32 [1,2048,2]`，全部有限
- 新 PyTorch reference 与历史评估 NPZ logits 最大绝对差：`2.86102294921875e-6`

固定导出参数：B=1、N=2048、FP32、CUDA:0、opset 17、无 dynamic axes、无 ATen fallback、无自定义 symbolic。

## 5. 导出可行性审计

### 5.1 活跃的 TransitionDown / voxel pooling 链路

`models/testParameters/GCN_res/model.py:87-100` 创建四个 TransitionDown：

| 层 | grid size | 输入特征 | 线性后特征 | pooling 输出 |
|---|---|---|---|---|
| `tdb_1` | 0.06 | `[1,N0,48]`, N0=2048 | `[1,N0,96]` | `[1,N1,3]`, `[1,N1,96]` |
| `tdb_2` | 0.13 | `[1,N1,96]` | `[1,N1,192]` | `[1,N2,3]`, `[1,N2,192]` |
| `tdb_3` | 0.325 | `[1,N2,192]` | `[1,N2,384]` | `[1,N3,3]`, `[1,N3,384]` |
| `tdb_4` | 0.8125 | `[1,N3,384]` | `[1,N3,512]` | `[1,N4,3]`, `[1,N4,512]` |

其中 N1～N4 是输入坐标决定的 voxel 数量。固定 B=1、N0=2048 并不能固定这些内部点数。

| 操作 | 文件与行号 | 输入 → 输出 | ONNX/固定 shape 判断 | 等价替换或 Plugin 判断 |
|---|---|---|---|---|
| `torch_cluster.grid_cluster` | `ptv2_utils.py:69` | `[Ni,3]` + size `[3]` → cluster id `[Ni]` | **实际导出首个失败点**；没有标准 PyTorch ONNX symbolic。固定输入 shape 不能消除对坐标值的依赖 | 若保持双输入接口，需要显式标准算子重写或 ORT custom op；未来 TensorRT 很可能需要 Plugin 或在网络外预计算 |
| `unique(return_inverse=True)` | `ptv2_utils.py:70` | cluster id `[Ni]` → unique `[Ci]`、inverse `[Ni]` | ONNX 有 `Unique`，但 Ci 动态；当前未实际走到该节点的导出验证 | ORT 可能执行；TensorRT 支持需单独验证，可能与 grid pooling 合并为 Plugin |
| `scatter_reduce_(amax)` | `ptv2_utils.py:74-75` | features `[Ni,Fi]` + inverse `[Ni]` → `[Ci,Fi]` | opset 17 对 max reduction 表达受限；opset 18 的 ScatterElements 才更接近该 reduction，但仍需 exporter 和数值验证 | 可做严格等价标准算子实现；TensorRT 可能需要 Plugin |
| `scatter_add_` | `ptv2_utils.py:77-78` | xyz `[Ni,3]` + inverse `[Ni]` → sums `[Ci,3]` | 可映射 scatter-add 类标准节点，但动态 Ci 和 exporter 支持仍需验证 | 可能标准化；TensorRT 是否原生支持需验证 |
| `bincount` | `ptv2_utils.py:79` | inverse `[Ni]` → count `[Ci]` | PyTorch legacy ONNX exporter通常没有直接标准 symbolic；当前尚未实际走到该失败点 | 可用“ones + scatter-add”做数学等价替换；未来也可并入 pooling Plugin |
| 除以 cluster count | `ptv2_utils.py:80` | sums `[Ci,3]` / counts `[Ci,1]` → centroid `[Ci,3]` | Div 本身可支持；依赖前序动态张量 | 不需要独立 Plugin |
| Python batch loop | `ptv2_utils.py:64,68` | 对 B 个样本逐个 pooling | B=1 时 tracer 会展开一次；不能支持通用动态 batch | 第一版固定 B=1 可接受，但必须防止数据流被常量化 |
| Python `min` 与切片 | `ptv2_utils.py:85-87` | 多个 `[Ci,*]` 裁到 `min(Ci)` | B=1 时数学上没有跨 batch 裁剪，但 tracer 对 Python size 的处理存在常量固化风险 | 固定 B=1 可在部署等价实现中消除跨 batch min；B>1 需显式定义语义 |
| `stack` | `ptv2_utils.py:89-90` | 列表 → `[B,Cmin,*]` | 固定 B=1 容易；Cmin 仍动态 | 标准 ONNX 支持，前序 shape 是关键 |

### 5.2 PointTransformer 邻域与插值链路

| 操作 | 文件与行号 | 输入 → 输出 | ONNX/固定 shape 判断 | 等价替换或 Plugin 判断 |
|---|---|---|---|---|
| `torch.cdist`（PT block） | `ptv2_utils.py:213` | `[1,Ni,3] × [1,Ni,3]` → `[1,Ni,Ni]` | ONNX 没有名为 CDist 的基础节点；PyTorch exporter可能分解为标准算术和 Reduce。内部 Ni 动态增加风险 | 可标准分解，通常不需 Plugin，但显存/性能需评估 |
| `topk` | `ptv2_utils.py:214` | `[1,Ni,Ni]` → indices `[1,Ni,K]` | ONNX TopK 支持；各 block 的 K 是构造时常量 | 预计可原生表达，TensorRT 支持需后续验证 |
| `torch.gather` + reshape | `ptv2_utils.py:40-51,215,218` | points/features + `[1,Ni,K]` → neighbours `[1,Ni,K,C]` | Gather/reshape 可表达；动态 Ni 要保持正确 | 通常不需要 Plugin |
| `torch.cdist`（上采样） | `ptv2_utils.py:147` | high xyz `[1,Nh,3]` × low xyz `[1,Nl,3]` → `[1,Nh,Nl]` | 与上述 CDist 相同；Nh/Nl 来自动态 voxel 层 | 可标准分解，性能需验证 |
| `topk(k=1)` | `ptv2_utils.py:150`，调用在 `:126` | `[1,Nh,Nl]` → value/index `[1,Nh,1]` | TopK 支持，k 固定 | 通常不需要 Plugin |
| `arange` + flatten +高级索引 | `ptv2_utils.py:154-159` | features2 与动态 indices → `[1,Nh,C]` | 可能导出为 GatherND/reshape；B=1 有利，但内部动态 shape 仍需验证 | 预计可等价标准化，不应静默改索引顺序 |
| Python `int(c/groups)` | `ptv2_utils.py:189,192` | channel reshape | 导出出现 TracerWarning；C 和 groups 是模型固定常量，所以本固定模型中可视为安全常量 | 不需要 Plugin，但导出后需 shape 与数值验证 |

### 5.3 GCN 与非活跃代码

- `model.py:70-73` 的 GCN 是 `torch.matmul(adj, x)` 加 Linear。输入 adj 为 `[1,2048,2048]`，x 为 `[1,2048,48]`，输出 `[1,2048,48]`；MatMul/Linear 是标准 ONNX 操作。导出成功后仍必须用图依赖检查确认 logits 依赖 `adj`，避免 adj 被错误固化或裁掉。
- `model.py:11-61` 的 `KNNLayer/LocalFeatureAggregation` 包含 `cdist/topk/gather`，但 `PTV2Segmentation` 中对应 LFA 调用被注释，本次实际前向不经过该分支，不是当前导出阻塞点。

## 6. 为什么没有自动尝试 opset 18

opset 18 可能改善 `scatter_reduce_(amax)` 的标准节点表达能力，但不会为 `torch_cluster::grid` 自动增加 PyTorch symbolic。当前失败发生在进入 Unique/Scatter/Bincount 转换验证之前。

因此直接把 `opset_version=17` 改为 18，只会再次遇到同一个未识别命名空间，不能解决首个阻塞。按“失败先分析、禁止静默改变数学语义”的原则，本轮没有进行无意义重试。

## 7. 旧 ONNX 代码为何不能使用

仓库 `export2ONNX.py:28-29,78` 注册：

```python
CustomNamespace::GridCluster
```

这会把问题推迟为 ONNX Runtime/TensorRT 自定义算子实现问题，并产生自定义 domain。当前验收明确要求检查并拒绝自定义域，且本机没有对应 ORT custom kernel，因此该方案没有被使用。

旧脚本同时使用 `Nico_v2_GCN_ONNX`、原作者 D 盘路径和不同调用接口，也不符合本次模型真源与固定双输入要求。

## 8. 导出后检查状态

由于 `.onnx` 文件未生成，下列检查均未执行，不能写成通过：

| 检查 | 状态 |
|---|---|
| `onnx.checker.check_model` | 未执行：无 ONNX 文件 |
| ONNX shape inference | 未执行 |
| 输入/输出 name、shape、dtype | 设计已固定，未从 ONNX 实证 |
| PythonOp 检查 | 未执行 |
| ATen fallback 检查 | 导出配置禁止 fallback；无成品图可扫描 |
| 自定义域检查 | 未执行；导出时未注册任何自定义 symbolic |
| 大型 Constant 检查 | 未执行 |
| adj graph input / initializer / dependency 检查 | 未执行 |
| 输出是否常量固化 | 未执行 |

## 9. ONNX Runtime 对齐与性能状态

`scripts/validate_gcn_res_onnxruntime.py` 已实现：

- 读取固定 6 个 NPZ；
- 用 normalized XYZ 重建 points 和 k=6 dense adj；
- 比较 logits max/mean absolute error、max relative error；
- 比较预测标签一致率；
- 分别比较 weld seam 与 background probability；
- 验收 `rtol=1e-4, atol=1e-5` 和标签一致率 ≥99.99%；
- CPU/CUDA ExecutionProvider（若可用）各预热10轮、正式50轮；
- 输出 mean、P50、P95、min、max。

当前 ONNX Runtime 1.27.0 可用 provider 为：

```text
AzureExecutionProvider
CPUExecutionProvider
```

本机当前没有 `CUDAExecutionProvider`。更重要的是 ONNX 导出已失败，所以验证脚本没有运行，未产生任何数值对齐或 ORT 性能结论。

## 10. 失败运行产物

已保存：

- `export_input.npz`
  - points `[1,2048,4]` FP32
  - adj `[1,2048,2048]` FP32
  - normalized XYZ、sample indices、ground truth
- `pytorch_reference.npz`
  - logits `[1,2048,2]` FP32
  - 历史评估 baseline logits
- `config_resolved.yaml`
  - 固定配置、失败类型、失败消息和完整 traceback
- `export.log`
  - 执行命令环境、strict load、shape 和异常栈

未生成：

- `gcn_res_fixed_b1_n2048.onnx`
- ONNX Runtime parity CSV/JSON
- ONNX Runtime benchmark JSON

## 11. 下一步需要显式选择的技术路线

当前不能在不做额外设计的情况下继续到 ORT/TensorRT。可选路线均需要单独授权和数值证明：

1. **部署侧标准 ONNX 等价实现**：不改原模型源码，在 deployment 模块中严格复现 grid assignment、Unique、max pooling、centroid mean 和 count；先用 PyTorch 在6个固定样本逐层/最终 logits 对齐，再尝试 opset 18。难点是动态 cluster 数和 B=1 trace 的数据依赖。
2. **扩大外部输入接口**：在网络外预计算每层 voxel cluster/inverse mapping，并作为额外输入传入。可避免 `torch_cluster::grid` 进入 ONNX，但不再是当前仅 points+adj 的双输入接口。
3. **自定义算子 / Plugin**：为 grid pooling 链路实现 ONNX Runtime custom op，并在未来实现 TensorRT Plugin。数学语义最直接，但违反本轮“无自定义域”的可控标准图要求，维护成本最高。
4. **更换可部署网络结构并重新训练**：当前明确禁止，且会改变模型/checkpoint，不属于本阶段。

在选定路线并完成数学等价验证前，不应把已有 `Nico_v2_GCN_ONNX` 或自定义 GridCluster 节点当作可部署成功。

# 第二轮标准算子等价实现状态（2026-07-14）

第一轮 `torch_cluster::grid` 导出失败后，已按授权在 deployment 目录实现标准 PyTorch 算子的 voxel pooling，并完成分层验证。最新结论以 `docs/gcn_res_standard_ops_equivalence.md` 为准：

- `STANDARD_OPS_VOXEL_POOL_EQUIVALENCE_PASSED`：8 类人工用例全部通过。
- `GCN_RES_DEPLOYMENT_MODEL_PARITY_FAILED`：首个真实样本的四级 voxel pooling 全部通过，最终 logits 也满足最终容差且标签一致率 100%，但 `ptb_8`、`tub_9`、`ptb_9` 未达到逐层 `rtol=1e-5, atol=1e-6`。
- 按停止条件，其余 5 个真实样本未运行。
- **没有重新执行 opset 18 ONNX 导出**，没有生成第二轮 ONNX，也没有运行 ONNX Runtime/TensorRT。

当前仍然有效的导出状态为：

```text
GCN_RES_ONNX_EXPORT_FAILED
GCN_RES_ONNXRUNTIME_PARITY_NOT_RUN
```

第二轮等价性完整数据位于：

- `artifacts/gcn_res_standard_ops/20260714_172215_000000_equivalence_retry/`
- `artifacts/gcn_res_standard_ops/20260714_172300_000000_failed_stage_diagnostic/`

# 第三轮 deployment 模型 opset 18 导出（2026-07-15）

## 12. 本轮范围与固定配置

本轮按授权直接使用部署侧模型 `deployment/gcn_res_onnx_model.py` 作为导出源，原始模型、历史 checkpoint、数据和 benchmark 结果均未修改。

| 项目 | 实际值 |
|---|---|
| 原始数学真源 | `models/testParameters/GCN_res/model.py` |
| ONNX 导出源 | `deployment/gcn_res_onnx_model.py` |
| 标准算子 voxel pooling | `deployment/onnx_voxel_pool.py` |
| checkpoint | `models/testParameters/GCN_res/best_model.pth` |
| 固定样本 | `val_00_weld_7` |
| points | FP32 `[1,2048,4]` |
| adj | FP32 `[1,2048,2048]` |
| logits | FP32 `[1,2048,2]` |
| opset | 18 |
| dynamic axes | 禁用 |
| ATen fallback | 禁用 |
| 自定义 symbolic | 未注册 |
| run id | `20260715_deploy_fp32_opset18` |

实际命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe scripts\export_gcn_res_onnx.py --run-id 20260715_deploy_fp32_opset18
```

## 13. 导出前验证与已保存基线

导出前步骤全部成功：

- 运行环境为 PyTorch `2.7.1+cu128`、CUDA Runtime `12.8`、RTX 5060，计算能力 `(12,0)`。
- checkpoint 使用 `load_state_dict(strict=True)` 加载，结果为 `<All keys matched successfully>`。
- 固定真实样本成功构造 points `[1,2048,4]` 和 adj `[1,2048,2048]`。
- deployment 模型前向成功，生成 logits `[1,2048,2]`。
- 输入、邻接矩阵和 PyTorch deployment logits 已在调用 ONNX exporter 前落盘，因此导出失败没有丢失参考数据。

本轮目录：

`artifacts/gcn_res_onnx/20260715_deploy_fp32_opset18/`

已生成：

- `export_input.npz`：固定样本 points、adj 及相关样本信息。
- `pytorch_deploy_reference.npz`：deployment 模型的 FP32 logits 参考输出。
- `config_resolved.yaml`：固定配置、失败类型、失败消息和完整 traceback。
- `export.log`：环境、strict load、输入输出 shape 和异常栈。

未生成：

- `gcn_res_deploy_fp32_opset18.onnx`

## 14. 首个确定阻塞算子

本轮导出在 PyTorch legacy ONNX exporter 的图转换阶段失败，首个确定阻塞算子为：

```text
torch.onnx.errors.UnsupportedOperatorError:
Exporting the operator 'aten::trunc' to ONNX opset version 18 is not supported.
```

该算子来自部署侧 voxel key 计算：

- `deployment/onnx_voxel_pool.py:71`：`voxel_coordinates = torch.trunc(shifted / voxel_size).to(torch.int64)`
- `deployment/onnx_voxel_pool.py:72`：`extents = torch.trunc((end - start) / voxel_size.reshape(1, 3)).to(torch.int64) + 1`

这不是原先的 `torch_cluster::grid` 阻塞；标准算子部署实现已经成功绕开该自定义扩展。当前问题是 PyTorch 2.7.1 legacy exporter 没有为 `aten::trunc` 提供 opset 18 symbolic 转换。失败发生在 ONNX 文件生成之前。

`torch.trunc` 是当前部署实现中用于保持原 `torch_cluster.grid` 整数转换语义的明确操作。本轮禁止修改模型，因此没有将其静默替换为 `floor`、整数 `Cast`、自定义 symbolic 或其他表达，也没有改用另一套 exporter。任何替换都必须先证明边界、负坐标和整数转换规则与当前实现一致，再重新执行已有等价性测试。

## 15. 导出后检查与 ORT 状态

由于目标 `.onnx` 文件没有生成，以下检查均无法执行，不能标记为通过：

| 检查 | 状态 |
|---|---|
| `onnx.checker.check_model` | 未执行：无 ONNX 文件 |
| ONNX shape inference | 未执行 |
| PythonOp 扫描 | 未执行 |
| ATen 节点扫描 | 未执行 |
| `torch_cluster` 节点扫描 | 未执行 |
| custom domain 扫描 | 未执行 |
| 大型 Constant/initializer 扫描 | 未执行 |
| points/adj 是否被固化 | 未执行 |
| logits 对两个输入的依赖性 | 未执行 |
| ONNX Runtime 数值对齐 | 未运行 |

因此本轮没有产生 max/mean absolute error、max relative error、weld/background probability error 或 predicted-label agreement。没有把缺失结果写成零误差，也没有运行 TensorRT。

## 16. 本轮结论与后续修复边界

本轮严格执行了“遇到首个阻塞算子即停止”的要求。可以确认：

1. checkpoint、真实输入和 deployment PyTorch 参考输出有效且已保存。
2. 导出已越过上一轮的 `torch_cluster::grid` 问题。
3. 当前第一个确定阻塞点是 `deployment/onnx_voxel_pool.py:71` 的 `aten::trunc`；第 72 行存在同类操作，但 exporter 在首个节点即停止。
4. 当前没有可供 checker 或 ONNX Runtime 使用的模型文件。
5. 本轮没有修改 checkpoint、原始模型、deployment 数学逻辑、数据或 benchmark 结果。

后续若获授权，应只针对 `trunc` 的 ONNX 等价表达开展独立语义审计，重点覆盖负坐标、voxel 边界、`start=min(xyz)` 后非负性以及 toward-zero 与 floor 的差异；通过人工用例和真实样本对齐后，才可再次导出。不能在未经证明时仅为生成 ONNX 而替换该操作。

```text
GCN_RES_ONNX_EXPORT_FAILED
GCN_RES_ONNXRUNTIME_PARITY_NOT_RUN
FIRST_BLOCKING_OPERATOR=aten::trunc
```
