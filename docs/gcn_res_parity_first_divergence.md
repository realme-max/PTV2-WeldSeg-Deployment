# GCN_res deployment parity 首个数值分叉定位报告

## 1. 结论

本轮只执行只读 forward hook 和张量诊断，没有修改任何模型实现、checkpoint、数据或容差，也没有继续 ONNX。

必须区分三种“首个分叉”：

1. **最早的非零数值差异**：`tdb_1` 的 pooled XYZ，即第一次 voxel pooling 的 XYZ `scatter_add / count` 输出。
   - shape：`[1,518,3]`
   - max abs：`1.1920928955078125e-07`
   - mean abs：`2.397226861461377e-10`
   - max relative：`1.6447380346562568e-07`
   - `allclose(rtol=1e-5, atol=1e-6)`：通过
   - 同级 pooled features `[1,518,96]`：bitwise 完全一致。
2. **最早超过既定 allclose 容差的内部算子**：`ptb_1.gva.delta_mult.batch_norm`。
   - shape：`[1,96,518,16]`
   - max abs：`2.3245811462402344e-06`
   - mean abs：`1.2602337839950906e-08`
   - max relative：`1.6957897692918777e-02`
   - allclose：失败
   - `delta_bias.batch_norm` 同样失败，但 `ptb_1` 最终 feature 输出仍满足 allclose。
3. **第一次在后续主路径中保持失败状态的输出**：FPN stage 2 的 `fpn_c2`，在进入 `ptb_8` 之前已经失败。
   - `fpn_c2 Conv1d` `[1,48,518]`：max abs `1.2576580047607422e-05`，allclose 失败。
   - `fpn_c2_linear` `[1,518,96]`：max abs `8.955597877502441e-06`，allclose 失败。
   - `tub_8 + fpn_c2` 融合结果、也就是 `ptb_8` 的输入 feature `[1,518,96]`：max abs `8.955597877502441e-06`，allclose 失败。

因此，不能把 `ptb_8` 解释为误差的源头。`ptb_8` 是首次被上一条 FPN 分支的超差输入直接污染的 PTV2 block；其 residual 会保留这部分差异。

## 2. 固定条件与检查方法

- 样本：`artifacts/gcn_res_evaluation/20260714_160831_945091_historical_checkpoint/predictions/val_00_weld_7.npz`
- original：`models/testParameters/GCN_res/model.py::PTV2Segmentation`
- deployment：`deployment/gcn_res_onnx_model.py::GCNResStandardOps`
- checkpoint：`models/testParameters/GCN_res/best_model.pth`
- checkpoint 加载：两边均 `strict=True`，无 key mapping。
- 输入：points `[1,2048,4]`、adj `[1,2048,2048]`，FP32，`cuda:0`。
- GPU：NVIDIA GeForce RTX 5060，capability `(12,0)`。
- seed：42；TF32 关闭；cuDNN benchmark 关闭。
- 固定判定：`torch.allclose(rtol=1e-5, atol=1e-6)`，未调整容差。

本轮捕获和展开的全部浮点张量均为有限值，没有 NaN/Inf。

通过 forward hook 捕获 original/deployment 对应模块的输入与输出；对模块内部没有独立 `nn.Module` 的 `cdist/topk/gather/interpolation/residual`，使用各自已加载模块和已捕获输入在 `torch.inference_mode()` 下只读重放。未向磁盘保存或替换任何中间 tensor。

相关原始代码位置：

- voxel XYZ 求和与均值：`models/testParameters/GCN_res/ptv2_utils.py:78-80`
- positional BatchNorm：`models/testParameters/GCN_res/ptv2_utils.py:13,33`
- PTV2 KNN/gather：`models/testParameters/GCN_res/ptv2_utils.py:200-224`
- FPN c2：`models/testParameters/GCN_res/model.py:105,110,171,177`
- `tub_8 + fpn_c2 → ptb_8`：`models/testParameters/GCN_res/model.py:194-196`
- `tub_9 → ptb_9`：`models/testParameters/GCN_res/model.py:198-200`

部署对应位置：

- voxel XYZ 求和与均值：`deployment/onnx_voxel_pool.py:113-120`
- positional BatchNorm：`deployment/gcn_res_onnx_model.py:21,26`
- FPN c2 与后续路径：`deployment/gcn_res_onnx_model.py:215,220,259,264,279-288`

## 3. voxel 输出和四级 encoder feature

同一行的 shape 对 original 和 deployment 均相同。

### 3.1 voxel XYZ

| 输出 | Shape | Max abs | Mean abs | Max relative | Allclose |
|---|---|---:|---:|---:|---|
| `tdb_1.xyz` | `[1,518,3]` | `1.192093e-07` | `2.397227e-10` | `1.644738e-07` | 通过 |
| `tdb_2.xyz` | `[1,129,3]` | `5.960464e-08` | `5.799709e-10` | `1.934661e-07` | 通过 |
| `tdb_3.xyz` | `[1,24,3]` | `5.960464e-08` | `2.590241e-09` | `1.163497e-07` | 通过 |
| `tdb_4.xyz` | `[1,4,3]` | `7.450581e-09` | `6.208817e-10` | `6.047989e-08` | 通过 |

### 3.2 voxel pooled features

| 输出 | Shape | Max abs | Mean abs | Max relative | Allclose |
|---|---|---:|---:|---:|---|
| `tdb_1.features` | `[1,518,96]` | `0` | `0` | `0` | 通过，bitwise 相同 |
| `tdb_2.features` | `[1,129,192]` | `7.152557e-07` | `6.425731e-09` | `1.049666e-04` | 通过 |
| `tdb_3.features` | `[1,24,384]` | `2.980232e-07` | `5.538507e-09` | `3.033889e-05` | 通过 |
| `tdb_4.features` | `[1,4,512]` | `1.490116e-07` | `4.880256e-09` | `7.373435e-05` | 通过 |

`tdb_1.features` 完全相同而 `tdb_1.xyz` 首次出现约一个 FP32 ULP 的差异，说明最初差异只在 XYZ 求和/均值路径，不在 `amax` feature pooling。

### 3.3 stage 1～4 PTV2 features

| 输出 | Shape | Max abs | Mean abs | Max relative | Allclose |
|---|---|---:|---:|---:|---|
| stage 1 / `ptb_1.features` | `[1,518,96]` | `8.344650e-07` | `5.423477e-09` | `3.085467e-04` | 通过 |
| stage 2 / `ptb_2.features` | `[1,129,192]` | `8.344650e-07` | `7.611333e-09` | `2.066681e-03` | 通过 |
| stage 3 / `ptb_3.features` | `[1,24,384]` | `2.384186e-07` | `5.538523e-09` | `3.028591e-05` | 通过 |
| stage 4 / `ptb_4.features` | `[1,4,512]` | `1.192093e-07` | `4.880256e-09` | `7.372951e-05` | 通过 |

相对误差在接近零的元素上会很大；allclose 结果按固定的绝对与相对组合条件计算，不能只依据 max relative 判断。

## 4. 最早内部 allclose 失败：ptb_1 positional BatchNorm

`ptb_1` 输入 features 来自 `tdb_1` 的 max pooling，仍 bitwise 相同；只有 XYZ 带 `1.192093e-07` 的差异。

| `ptb_1` 内部张量 | Shape | Max abs | Mean abs | Max relative | Allclose |
|---|---|---:|---:|---:|---|
| feature `linear_1` | `[1,518,96]` | `0` | `0` | `0` | 通过 |
| delta_mult `linear_1` | `[1,518,16,96]` | `7.450581e-08` | `3.822951e-10` | `3.318584e-03` | 通过 |
| delta_mult `batch_norm` | `[1,96,518,16]` | `2.324581e-06` | `1.260234e-08` | `1.695790e-02` | **失败** |
| delta_mult `linear_2` | `[1,518,16,96]` | `2.622604e-06` | `6.385700e-09` | `1.574008e-03` | **失败** |
| delta_bias `linear_1` | `[1,518,16,96]` | `9.313226e-08` | `3.711681e-10` | `7.951070e-02` | 通过 |
| delta_bias `batch_norm` | `[1,96,518,16]` | `2.160668e-06` | `1.181350e-08` | `9.352978e-01` | **失败** |
| q / k / v | 相应 `[1,518,{1或16},96]` | `0` | `0` | `0` | 通过 |
| attention conv | `[1,2,518,16]` | `4.291534e-06` | `2.509245e-08` | `4.341915e-05` | **失败** |
| softmax | `[1,2,518,16]` | `7.301569e-07` | `2.969391e-09` | `7.594515e-06` | 通过 |
| GVA output linear | `[1,518,96]` | `1.668930e-06` | `7.218818e-09` | `1.590917e-03` | 通过 |
| block `linear_2` | `[1,518,96]` | `1.788139e-06` | `8.142043e-09` | `3.447305e-03` | 通过 |
| block 最终 feature | `[1,518,96]` | 运行间约 `8e-07～1.4e-06` | 约 `5e-09～7e-09` | 运行相关 | 通过 |

这里 BatchNorm 处于 eval 模式，使用 checkpoint 的 running statistics 和 affine 参数。微小 positional input 差异被固定缩放放大，造成第一个严格 allclose 失败；后续 softmax、线性映射和 residual 又把 block 最终输出压回容差内。因此这是“内部首个失败”，不是“持续到 logits 的第一条失败输出”。

## 5. ptb_8 之前的 FPN 分叉

针对 `ptb_8` 入口的独立 hook 证明：

| 张量 | Shape | Max abs | Mean abs | Max relative | Allclose |
|---|---|---:|---:|---:|---|
| `ptb_1.features`（本次 focused run） | `[1,518,96]` | `1.430511e-06` | `6.952864e-09` | `9.923448e-04` | 通过 |
| `fpn_c2 Conv1d` | `[1,48,518]` | `1.257658e-05` | `2.225795e-08` | `5.255236e-04` | **失败** |
| `fpn_c2_linear` | `[1,518,96]` | `8.955598e-06` | `1.063526e-08` | `1.193714e-03` | **失败** |
| `tub_8.features` | `[1,518,96]` | `2.622604e-06` | `3.700371e-08` | `5.850971e-04` | 通过 |
| `tub_8 + fpn_c2` | `[1,518,96]` | `8.955598e-06` | `4.660704e-08` | `8.074283e-04` | **失败** |
| `ptb_8` pre-hook feature | `[1,518,96]` | `8.955598e-06` | `4.660704e-08` | `8.074283e-04` | **失败** |

重构的 `tub_8 + fpn_c2` 与两边各自 pre-hook 捕获的 `ptb_8` 输入均 `torch.equal=True`，所以没有遗漏中间操作。`fpn_c2` 的 Conv1d 对 stage 1 的微小 feature 差异进行线性组合；权重放大和部分输出接近零时的抵消，使该分支首次持续超差。

## 6. ptb_8 内部展开

`ptb_8` 的输入已经失败。以下展示主要内部节点：

| 张量 | Shape | Max abs | Mean abs | Max relative | Allclose |
|---|---|---:|---:|---:|---|
| input XYZ | `[1,518,3]` | `1.192093e-07` | `2.397227e-10` | `1.644738e-07` | 通过 |
| input features / residual | `[1,518,96]` | `8.955598e-06` | `3.870993e-08` | `2.692308e-02` | **失败** |
| pairwise distances | `[1,518,518]` | `6.482005e-07` | `1.010778e-09` | `1.413390e-05` | 通过 |
| topk indices | `[1,518,4]` | mismatch `0` | — | — | 完全一致 |
| gathered neighbour XYZ | `[1,518,4,3]` | `1.192093e-07` | `2.780783e-10` | `1.644738e-07` | 通过 |
| block `linear_1` | `[1,518,96]` | `4.053116e-06` | `4.619317e-08` | `3.797468e-03` | **失败** |
| gathered neighbour features | `[1,518,4,96]` | `4.053116e-06` | `4.565743e-08` | `3.797468e-03` | **失败** |
| delta_mult BatchNorm | `[1,96,518,4]` | `2.771616e-06` | `1.084342e-08` | `2.691146e-03` | **失败** |
| delta_bias BatchNorm | `[1,96,518,4]` | `2.264977e-06` | `1.143594e-08` | `1.713873e-03` | **失败** |
| q | `[1,518,1,96]` | `3.218651e-06` | `1.694066e-08` | `1.162759e-03` | 通过 |
| k | `[1,518,4,96]` | `3.576279e-06` | `1.575483e-08` | `9.233069e-04` | **失败** |
| v | `[1,518,4,96]` | `2.861023e-06` | `4.708098e-08` | `2.5e-01` | **失败** |
| vector attention | `[1,518,4,96]` | `5.483627e-06` | `1.162436e-08` | `2.867036e-03` | **失败** |
| attention softmax | `[1,2,518,4]` | `1.549721e-06` | `8.508404e-09` | `6.234668e-06` | 通过 |
| feature aggregation | `[1,518,96]` | `2.622604e-06` | `7.466087e-08` | `9.471422e-04` | **失败** |
| GVA BatchNorm | `[1,518,96]` | `3.576279e-06` | `4.842825e-08` | `2.105893e-03` | **失败** |
| GVA linear | `[1,518,96]` | `1.251698e-06` | `2.749352e-08` | `7.239653e-03` | 通过 |
| block `linear_2` | `[1,518,96]` | `1.430511e-06` | `2.460464e-08` | `5.585994e-03` | 通过 |
| residual branch | `[1,518,96]` | `8.955598e-06` | `3.870993e-08` | `2.692308e-02` | **失败** |
| `ptb_8` output feature | `[1,518,96]` | `9.059906e-06` | `5.631993e-08` | `7.554582e-04` | **失败** |

结论：邻域编号没有变化。`ptb_8` 的主要持续误差来自已经失败的 residual/input feature；attention 分支也受到 feature 和微小 XYZ 差异影响，但没有发生离散 KNN 跳变。

## 7. tub_9 内部展开

| 张量 | Shape | Max abs | Mean abs | Max relative | Allclose |
|---|---|---:|---:|---:|---|
| low XYZ 输入 | `[1,518,3]` | `1.192093e-07` | `2.397227e-10` | `1.644738e-07` | 通过 |
| low feature 输入 | `[1,518,96]` | `9.059906e-06` | `5.631993e-08` | `7.554582e-04` | **失败** |
| skip XYZ | `[1,2048,3]` | `0` | `0` | `0` | 完全一致 |
| skip features | `[1,2048,48]` | `0` | `0` | `0` | 完全一致 |
| `linear_1a` | `[1,518,48]` | `6.780028e-06` | `7.838076e-08` | `8.690784e-04` | **失败** |
| low BatchNorm/ReLU | `[1,518,48]` | `8.505769e-06` | `3.863853e-08` | `2.176875e-03` | **失败** |
| interpolation distances | `[1,2048,518]` | `7.053837e-06` | `5.501135e-10` | `1.668110e-03` | **失败** |
| interpolation topk indices | `[1,2048,1]` | mismatch `0` | — | — | 完全一致 |
| normalized weights（k=1） | `[1,2048,1]` | `0` | `0` | `0` | 完全一致 |
| gathered/interpolated feature | `[1,2048,48]` | `8.505769e-06` | `4.003232e-08` | `6.456300e-04` | **失败** |
| skip `linear_1b/BN/ReLU` | `[1,2048,48]` | `0` | `0` | `0` | 完全一致 |
| residual sum / `tub_9` output | `[1,2048,48]` | `8.583069e-06` | `4.006470e-08` | `2.473077e-04` | **失败** |

`k=1` 时归一化权重恒为 1；插值映射完全一致。`tub_9` 只是把 `ptb_8` 已存在的 feature 差异传播到 2048 点，不是新的索引分叉。

## 8. ptb_9 内部展开

| 张量 | Shape | Max abs | Mean abs | Max relative | Allclose |
|---|---|---:|---:|---:|---|
| input XYZ | `[1,2048,3]` | `0` | `0` | `0` | 完全一致 |
| input features / residual | `[1,2048,48]` | `8.583069e-06` | `3.998808e-08` | `2.523659e-02` | **失败** |
| distances | `[1,2048,2048]` | `0` | `0` | `0` | 完全一致 |
| topk indices | `[1,2048,16]` | mismatch `0` | — | — | 完全一致 |
| positional delta_mult/delta_bias 全链 | 对应 `[1,2048,16,48]` | `0` | `0` | `0` | 完全一致 |
| block `linear_1` | `[1,2048,48]` | `3.695488e-06` | `5.898390e-08` | `9.468318e-03` | **失败** |
| q | `[1,2048,1,48]` | `2.264977e-06` | `3.639614e-08` | `1.192946e-02` | **失败** |
| k | `[1,2048,16,48]` | `3.099442e-06` | `3.846032e-08` | `1.333333e-03` | 通过 |
| v | `[1,2048,16,48]` | `2.503395e-06` | `5.846254e-08` | `7.253232e-03` | 通过 |
| feature aggregation | `[1,2048,48]` | `1.525879e-05` | `2.517762e-07` | `1.422289e-03` | 通过 |
| GVA BatchNorm | `[1,2048,48]` | `2.384186e-06` | `5.697045e-08` | `8.299895e-02` | 通过 |
| block `linear_2` | `[1,2048,48]` | `9.536743e-07` | `6.451302e-08` | `2.152389e-03` | 通过 |
| residual branch | `[1,2048,48]` | `8.583069e-06` | `3.998808e-08` | `2.523659e-02` | **失败** |
| `ptb_9` output | `[1,2048,48]` | `9.298325e-06` | `9.178962e-08` | `9.799118e-04` | **失败** |

这里 XYZ 和全部位置编码均 bitwise 相同；差异完全来自输入 feature。虽然 feature aggregation 的 max abs 达到 `1.525879e-05`，其元素尺度使组合 allclose 仍通过。最终失败仍由 residual 输入主导。

## 9. segmentation head 与 logits

| 输出 | Shape | Max abs | Mean abs | Max relative | Allclose |
|---|---|---:|---:|---:|---|
| segmentation head 输入（`ptb_9.features`） | `[1,2048,48]` | `9.298325e-06` | `9.178962e-08` | `9.799118e-04` | **失败** |
| segmentation head / `mlp` 输出 | `[1,2048,2]` | `2.384186e-06` | `1.890885e-07` | `6.267029e-06` | 通过 |
| model 返回 logits | `[1,2048,2]` | `2.384186e-06` | `1.890885e-07` | `6.267029e-06` | 通过 |

head 的线性投影降低了 48 维 feature 差异，所以最终 logits 重新落回既定容差，但这不能抵消逐层 parity 已失败的事实。

## 10. scatter、voxel 顺序与映射专项检查

### 10.1 排除 voxel 排序和 cluster mapping 错误

第一次 pooling 的专项结果：

- unique voxel keys：完全一致。
- voxel coordinates：完全一致。
- voxel point counts：完全一致。
- point-to-voxel inverse：完全一致。
- cluster membership：完全一致。
- pooled features：bitwise 完全一致。
- 只有 pooled XYZ mean 存在约 `1.192093e-07` 差异。

因此不存在 voxel 顺序变化、hash 冲突、batch key 冲突或点被分到不同 voxel 的证据。

### 10.2 排除 gather/index 错误

- `ptb_8` topk `[1,518,4]`：mismatch count `0`。
- `tub_9` interpolation topk `[1,2048,1]`：mismatch count `0`。
- `ptb_9` topk `[1,2048,16]`：mismatch count `0`。

所以三层的 gather/index 映射完全一致。

### 10.3 FP 聚合顺序证据

将相同的第一次 pooling 输入只在诊断中复制到 CPU 后，original pooling 和 deployment pooling 的 pooled XYZ `torch.equal=True`，max abs 为 `0`。CUDA 上则为 `1.192093e-07`。

同一实现、相同输入在 CUDA 上连续执行两次也出现微小变化：

- original pooling 自身重复：pooled XYZ max abs `2.980232e-08`。
- deployment pooling 自身重复：pooled XYZ max abs `1.192093e-07`。
- original 完整 logits 自身重复：max abs `3.814697e-06`。
- deployment 完整 logits 自身重复：max abs `2.980232e-06`。

这是强证据表明最初差异来自 CUDA `scatter_add` 浮点原子累加的执行/归约顺序。FP32 加法不满足结合律；相同 voxel 成员和相同数学求和在不同原子调度顺序下可能相差一个或数个 ULP。该判断是根据 CPU bitwise 对齐、CUDA 自身重复差异、离散映射完全一致所作的证据性推断，不是对 CUDA kernel 调度的直接 trace。

误差传播链为：

```text
tdb_1 CUDA XYZ scatter_add/mean（首个非零差异）
  → ptb_1 positional BatchNorm（首个内部 allclose 失败）
  → stage 1 feature 仍整体通过
  → fpn_c2 Conv1d/Linear（首个持续失败分支）
  → tub_8 + FPN 融合输入已失败
  → ptb_8 residual + attention
  → tub_9 interpolation（索引相同）
  → ptb_9 residual（邻域相同）
  → segmentation head
  → logits 重新落入最终容差
```

## 11. 原因分析

结论不是 deployment 数学逻辑、voxel 排序或索引映射错误。当前证据支持以下原因：

1. original 使用 `scatter_add_`，deployment 使用标准 `scatter_add`，两者在 CUDA 上对同一 voxel 内多个 FP32 XYZ 执行并行原子累加。
2. 原始和部署的 voxel 成员、key、inverse、count 都相同，但 CUDA 求和结果有约一个 FP32 ULP 的运行间/实现间差异。
3. positional BatchNorm 的固定 affine scale 将 XYZ 微差放大；某些内部张量先超过严格 allclose。
4. stage 1 最终 feature 虽通过，但进入 FPN c2 的 Conv1d 后，线性组合和接近零处的抵消使差异持续超过容差。
5. 后续 topk/gather/interpolation 索引全部相同，没有发生离散路径跳变；误差沿相同计算图传播，主要由 residual 保留。

## 12. 修复建议（本轮均未实施）

在“不修改任何实现、不调整容差”的当前约束下，没有可直接实施的修复；本报告只给出后续技术选择：

1. **确定性 segmented reduction**：在 deployment 侧按 `(voxel key, original point index)` 建立固定顺序，再用固定 reduction tree 计算 XYZ sum。这保持 voxel mean 的数学定义，但要验证能否用 ONNX/TensorRT 标准算子表达，并重新做六样本逐层 parity。
2. **与历史 CUDA 累加顺序完全一致**：理论上需复现 torch 原始 scatter kernel 的调度/归约细节；GPU 原子调度本身不保证稳定，通常无法仅靠标准 ONNX 算子保证 bitwise 一致。
3. **重新定义数值验收政策**：把同一 original CUDA 模型自身重复运行的误差包络纳入判定。这属于验收规则变化，当前明确禁止调整容差，因此必须另行授权，不能在本轮使用。
4. **在网络外提供预计算 voxel 结果**：可绕开 ONNX 内部 reduction 差异，但会改变部署输入接口，也需要单独授权。
5. 不建议修改历史 checkpoint、BatchNorm 参数、FPN 权重或重新训练来掩盖该差异；这些都会改变模型真源。

在作出上述路线选择前，保持当前状态：

```text
STANDARD_OPS_VOXEL_POOL_EQUIVALENCE_PASSED
GCN_RES_DEPLOYMENT_MODEL_PARITY_FAILED
ONNX_NOT_ATTEMPTED
```
