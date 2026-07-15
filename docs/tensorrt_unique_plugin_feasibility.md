# TensorRT Unique Plugin 可行性评估

## 1. 结论摘要

当前增量 parser 审计只确认一种项目侧阻塞算子：ONNX `Unique`。

- 四个 parser error 全部来自 `tdb_1`～`tdb_4` 的 `Unique`。
- TensorRT 11.1 `supports_operator()` 对图内其他算子均返回 `True`；该接口可能存在 false-positive，因此不能据此证明下游整图一定可解析。
- 注册 NVIDIA TensorRT 标准 Plugin 后，实际图中使用的 `ScatterElements` reduction 组合均通过独立 parser probe。
- TensorRT 11.1 的 `IPluginV3` 能通过 size tensor 表达由输入数据决定的输出长度，所以实现运行时 `M=number_of_unique_voxels` 在接口层面有依据。
- 但当前标准 ONNX `Unique` 会在 TensorRT 内置 `checkUnique` 阶段直接失败，不能仅把自定义 DLL 放入 registry 就让现有 ONNX 原样通过。

状态：

```text
ONLY_CONFIRMED_BLOCKING_OPERATOR=Unique
UNIQUE_PLUGIN_FEASIBILITY=CONDITIONALLY_FEASIBLE
CURRENT_ONNX_DROP_IN_PLUGIN=NOT_FEASIBLE
SPLIT_VOXEL_HIERARCHY_NOT_YET_REQUIRED
```

## 2. Incremental parser audit

审计使用三种互相独立的 parser 视角：

1. `parse_from_file()`：4条错误，全部为 `Unique`；
2. `supports_model_v2()`：模型不支持，被四个 Unique 划分为5个 subgraph；
3. `supports_operator()`：图中唯一返回 `False` 的 op type 是 `Unique`。

`supports_model_v2()` 返回的5段均为 `supported=False`。因此当前结论严格限定为“唯一**已确认**的阻塞算子是 Unique”，而不是“已经证明替换 Unique 后整图必然通过”。

增量审计产物：

```text
artifacts/gcn_res_tensorrt/20260715_171941_668015_incremental_parser_audit/
  incremental_parser_audit.json
  incremental_parser_report.md
```

## 3. ScatterElements 属性级检查

原图共有20个 `ScatterElements`，四级 TDB 每级5个：

| reduction | 每级数量 | 主要 dtype | 总数 |
|---|---:|---|---:|
| `min` | 1 | INT64 | 4 |
| `add` | 3 | INT64 / FLOAT | 12 |
| `max` | 1 | FLOAT | 4 |

未注册标准 Plugin 时，reduction probe 报告：

```text
ScatterReduction plugin was not found in the plugin registry
```

调用 TensorRT SDK 自带的：

```python
trt.init_libnvinfer_plugins(logger, "")
```

之后，下列 opset 18 parser-only probe 全部通过：

```text
ScatterElements(reduction=none, FLOAT)
ScatterElements(reduction=add, INT64)
ScatterElements(reduction=add, FLOAT)
ScatterElements(reduction=min, INT64)
ScatterElements(reduction=max, FLOAT)
```

这说明 Scatter reduction 当前依赖 NVIDIA 随 SDK 提供的标准 Plugin，但不要求项目自行实现第二种 Plugin。后续所有 parser/build 入口都必须先注册标准 Plugin。

## 4. 四个 Unique 的实际接口

四个节点均满足：

- 输入：一维 INT64 voxel key，长度为当前 stage 点数 `N_i`；
- `axis`：未设置，即对一维输入整体执行 Unique；
- `sorted=1`；
- `values`：INT64 `[M_i]`，内容本身未进入特征计算，但其 shape 用于取得 voxel 数；
- `indices`：未使用；
- `inverse_indices`：INT64 `[N_i]`，用于 point-to-voxel mapping；
- `counts`：未使用。

其中：

```text
M_i = number of materialized unique voxels
0 < M_i <= N_i
```

`inverse_indices` 随后影响 voxel count、mean pooled XYZ、max pooled feature，并继续影响 attention、geometry、Transition Up 和最终 `logits`。

## 5. TensorRT 11.1 对数据依赖输出 shape 的能力

本机 TensorRT 11.1.0.106 SDK 头文件给出了明确接口依据：

```text
D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106\include\NvInferRuntime.h
```

关键位置：

- 第313～336行：`IExprBuilder::declareSizeTensor()`；
- 第315行：允许 Plugin 输出维度不能仅由输入维度计算；
- 第317～321行：以 NonZero 的运行时 `K` 为示例；
- 第328～330行：必须提供用于 autotuning 的 opt 和内存分配 upper bound；
- 第913～914行：size tensor 输出必须为 INT32；
- 第932～943行：size tensor 输出必须是0D，并由 `getOutputShapes()` 声明。

因此可以设计一个 Plugin 输出0D INT32 的 `M_i`，并用它作为其他输出第一维：

```text
input:
  voxel_key       INT64 [N_i]

outputs:
  voxel_count     INT32 []       # size tensor, runtime value M_i
  unique_values   INT64 [M_i]
  inverse_indices INT64 [N_i]
```

理论 upper bound 可设为 `N_i`。opt 值需要根据固定 weld 数据统计或采用保守表达式，且不能依赖另一个 size tensor。

## 6. 为什么不能直接给当前 ONNX 投放一个 Unique DLL

当前 ONNX 使用标准域的 ONNX `Unique`，TensorRT 在：

```text
checkUnique → UNSUPPORTED_NODE: false
```

阶段直接拒绝。该路径不会自动把标准 `Unique` 转交给同名自定义 Plugin。

此外，ONNX `Unique` 的四个标准输出是：

```text
values[M]
indices[M]
inverse_indices[N]
counts[M]
```

它没有额外的0D `M` 输出，而 TensorRT `declareSizeTensor()` 要求某个 Plugin 输出专门作为0D INT32 size tensor。要正确使用该机制，未来至少需要以下一种受控变化：

1. 将标准 `Unique` 替换为自定义 domain 节点，并增加0D `voxel_count` 输出；
2. 用一个更大的自定义 voxel-pooling 节点替换 `Unique + ScatterElements + shape/index` 子图；
3. 不经过 ONNX parser，使用 TensorRT Network API 手工拼接 Plugin 和 backbone。

以上都会改变部署图或构图流程，因此在当前“禁止修改 ONNX/模型、禁止实现 Plugin”的阶段均未执行。

## 7. Plugin 粒度选择

### 方案 P1：Unique-only Plugin

职责：

- 对 INT64 voxel key 排序；
- 生成 sorted unique key；
- 生成与 sorted key 顺序一致的 inverse mapping；
- 输出运行时 voxel count。

优点：

- 数学边界小；
- 可直接复用现有标准算子 voxel aggregation；
- 对齐测试可复用已有 Unique/voxel mapping 审计。

风险：

- 仍需部署图重写以增加 size tensor；
- 后续动态 `Shape/ScatterElements/NonZero/If` 链只有在 parser 越过 Unique 后才能得到整图证明；
- 必须保证 sorted unique 顺序和 inverse ID 完全符合当前 ONNX 语义。

### 方案 P2：Voxel-pooling Plugin

职责扩大为：

```text
voxel key + Unique + count + pooled XYZ + pooled feature
```

建议接口：

```text
inputs:
  xyz             FLOAT [N_i,3]
  feature         FLOAT [N_i,C_i]

outputs:
  voxel_count     INT32 []
  pooled_xyz      FLOAT [M_i,3]
  pooled_feature  FLOAT [M_i,C_i]
  inverse_indices INT64 [N_i]   # 仅在后续确实需要时保留
```

优点：减少 parser 需要处理的数据依赖 shape/scatter 子图。

代价：Plugin 数学范围明显变大，需要同时证明排序、mean pooling、max pooling和浮点归约误差；实现及验证成本高于 P1。

当前优先评估 P1，但不立即实现。

## 8. 与拆分 voxel hierarchy 路线的关系

四级 mapping 不能全部由原始 points 一次性外部生成：

```text
mapping_2 depends on pooled_xyz_1
mapping_3 depends on pooled_xyz_2
mapping_4 depends on pooled_xyz_3
```

如果完全外移 voxel hierarchy，必须选择：

- CPU/CUDA preprocess 与多个 TensorRT stage 交替执行；或
- 将整个四级 voxel hierarchy 和相关 pooling 移出 TensorRT，只把已经形成的多尺度 tensors 输入 backbone；或
- 构建多个 engine，在各 stage 之间由宿主/CUDA 代码生成下一级 mapping。

这会显著增加输入接口、显存管理、CUDA stream 同步和 C++ 集成复杂度。因此当前不因图里“存在多个动态算子”就立即拆分；路线切换应以“多个**已确认不支持**算子”或 Unique Plugin 接口验证失败为依据。

## 9. 下一步门槛

建议下一阶段仍不实现完整 Plugin，只验证两个最小问题：

1. TensorRT 11.1 IPluginV3 size-tensor 最小网络能否在 Windows/SM12.0 上完成 parser/network 表达；
2. 自定义节点替换标准 Unique 后，parser 是否继续暴露新的不支持算子。

第二项必然需要生成独立的诊断 ONNX 副本或手工 TensorRT network，必须获得后续修改诊断图的授权；不得覆盖当前 ONNX。

只有满足以下条件才进入完整 Unique Plugin：

```text
size tensor prototype passed
custom node parser registration passed
next parser blocker inventory completed
Unique mapping semantics can be reproduced exactly
```

失败或出现多个新的动态阻塞时，转向：

```text
voxel hierarchy outside TensorRT
        +
TensorRT backbone / segmentation head
```

PLUGIN_FEASIBILITY_AUDIT_COMPLETED

