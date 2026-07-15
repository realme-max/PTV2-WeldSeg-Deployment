# GCN_res weld 子数据集短训练报告

## 1. 执行结论

本轮在 dry run 阶段失败，并已按“任何一步失败时停止”的要求终止后续操作。未执行 2 epoch 正式训练，也未尝试绕过错误或自动改动 batch size、模型、输入、精度。

- 执行状态：`DRY_RUN_FAILED`
- 失败原因：直接执行 `scripts/train_gcn_res_subdataset.py` 时，Python 的模块搜索路径没有包含项目根目录，导致无法导入 `models.testParameters.GCN_res.model`。
- 已通过的前置检查：配置解析、脚本语法、CUDA/GPU识别、子数据集划分加载、三个集合无交叠。
- 2 epoch 训练：未执行。
- 原始模型、原始训练脚本、原始 JSON、原始 checkpoint：均未修改或覆盖。

## 2. 本轮新增文件

- `scripts/train_gcn_res_subdataset.py`
  - 独立训练入口。
  - 数据读取固定为 `sub_shuffled_{train,val,test}_file_list.json`。
  - 支持 `from_scratch` 与 `resume_checkpoint` 两种模式。
  - 约束输入为 `[B,2048,4]`、邻接矩阵为 `[B,2048,2048]`、输出为 `[B,2048,2]`。
  - 包含有限值检查、OOM停止、strict checkpoint 加载、训练指标及独立产物目录逻辑。
- `config/gcn_res_subdataset_smoke.yaml`
  - 默认参数：seed 42、batch size 2、epochs 2、N 2048、k 6、Adam、learning rate 0.001、workers 0、`cuda:0`。
- `docs/gcn_res_subdataset_smoke_training.md`
  - 本报告。

## 3. 静态检查

使用解释器：

```text
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe
```

执行命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -c "from pathlib import Path; compile(Path(r'scripts/train_gcn_res_subdataset.py').read_text(encoding='utf-8'), r'scripts/train_gcn_res_subdataset.py', 'exec'); print('SYNTAX_OK')"
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -c "from omegaconf import OmegaConf; c=OmegaConf.load(r'config/gcn_res_subdataset_smoke.yaml'); print(OmegaConf.to_yaml(c, resolve=True))"
```

结果：

```text
SYNTAX_OK
```

配置解析成功，关键值为：

```text
mode: from_scratch
batch_size: 2
epochs: 2
num_point: 2048
k_neighbors: 6
device: cuda:0
split_prefix: sub_shuffled
num_class: 2
in_dim: 4
```

## 4. 数据划分

实际读取文件：

- `data/weld/train_test_split/sub_shuffled_train_file_list.json`
- `data/weld/train_test_split/sub_shuffled_val_file_list.json`
- `data/weld/train_test_split/sub_shuffled_test_file_list.json`

dry run 日志确认：

```text
Sub-dataset splits train=54 val=18 test=18; files are disjoint
```

因此划分数量为 train 54、val 18、test 18，三个集合无重复。未读取原始 `shuffled_*` 划分。

## 5. dry run

实际命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe scripts\train_gcn_res_subdataset.py --config config\gcn_res_subdataset_smoke.yaml --mode from_scratch --dry-run
```

环境识别结果：

```text
python=E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe
torch=2.7.1+cu128
cuda_runtime=12.8
gpu=NVIDIA GeForce RTX 5060
capability=(12, 0)
```

完整错误：

```text
2026-07-14 15:53:35,010 | ERROR | RUN_FAILED
Traceback (most recent call last):
  File "E:\GRP-PTv2\scripts\train_gcn_res_subdataset.py", line 776, in main
    model, optimizer, device, checkpoint_metadata = make_model_and_optimizer(cfg, logger)
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\GRP-PTv2\scripts\train_gcn_res_subdataset.py", line 316, in make_model_and_optimizer
    from models.testParameters.GCN_res.model import PTV2Segmentation
ModuleNotFoundError: No module named 'models'
Traceback (most recent call last):
  File "E:\GRP-PTv2\scripts\train_gcn_res_subdataset.py", line 815, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "E:\GRP-PTv2\scripts\train_gcn_res_subdataset.py", line 776, in main
    model, optimizer, device, checkpoint_metadata = make_model_and_optimizer(cfg, logger)
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\GRP-PTv2\scripts\train_gcn_res_subdataset.py", line 316, in make_model_and_optimizer
    from models.testParameters.GCN_res.model import PTV2Segmentation
ModuleNotFoundError: No module named 'models'
```

失败发生在模型导入之前，所以以下步骤均未执行：单 batch 取样、邻接矩阵生成、模型前向、loss、backward、optimizer step。

## 6. 输入输出 shape

训练入口已经实施并检查以下固定契约，但本轮因模型导入失败，尚无运行时 shape 观测值：

- Dataset points：`[B,2048,3]`
- weld category one-hot：`[B,2048,1]`，值恒为 1
- 模型 points：`[B,2048,4]`
- 邻接矩阵：`[B,2048,2048]`，`k=6`
- logits：`[B,2048,2]`
- seg：`[B,2048]`

## 7. 训练指标、显存与耗时

由于 dry run 在模型导入阶段终止：

- dry run loss：未产生
- dry run 邻接矩阵 CPU 构图耗时：未产生
- dry run GPU 峰值显存：未产生
- 2 epoch train/val loss、accuracy、mIoU：未产生
- 2 epoch epoch 耗时、CPU 构图耗时、GPU 峰值显存：未产生

## 8. 产物和 checkpoint 验证

本次失败运行目录：

```text
artifacts/subdataset_smoke/20260714_155334_965473_from_scratch_dryrun/
```

已生成：

- `config_resolved.yaml`
- `run.log`

未生成（符合失败即停止的安全策略）：

- `last_model.pth`
- `best_model.pth`
- `metrics.json`

因此本轮没有可执行的训练 checkpoint 保存及 strict 重新加载验证。

## 9. 后续恢复点

唯一阻塞点是独立脚本启动时的项目根目录模块解析。修复或改用可稳定解析项目包的启动方式后，应从 dry run 重新开始；只有 dry run 完整通过后，才能执行 `from_scratch`、2 epoch、batch size 2 的正式短训练。不得复用本次失败目录，应生成新的独立 `run_id`。

## 10. 模块解析修复与重新验证（2026-07-14）

### 10.1 修复范围

本次只修改了：

- `scripts/train_gcn_res_subdataset.py`
- `docs/gcn_res_subdataset_smoke_training.md`

未修改 YAML、原始 Dataset、JSON 划分、原始模型、原始训练脚本、checkpoint 或 Python 环境。

入口现在在所有项目内 import 之前执行：

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

assert (PROJECT_ROOT / "models").is_dir()
assert (PROJECT_ROOT / "config").is_dir()
assert (PROJECT_ROOT / "data").is_dir()
```

路径完全由 `__file__` 动态计算，没有硬编码项目路径、修改工作目录、复制 `models` 或要求设置 `PYTHONPATH`。启动日志新增 `PROJECT_ROOT`、`sys.path[0]` 和当前工作目录记录。

### 10.2 静态导入验证

实际命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -c "import runpy; runpy.run_path(r'E:\GRP-PTv2\scripts\train_gcn_res_subdataset.py', run_name='not_main'); print('SCRIPT_IMPORT_PASSED')"
```

结果：

```text
SCRIPT_IMPORT_PASSED
```

### 10.3 新 dry run

实际命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe E:\GRP-PTv2\scripts\train_gcn_res_subdataset.py --config E:\GRP-PTv2\config\gcn_res_subdataset_smoke.yaml --mode from_scratch --dry-run
```

新 run ID：

```text
20260714_155725_569825_from_scratch_dryrun
```

启动路径日志：

```text
PROJECT_ROOT=E:\GRP-PTv2
sys.path[0]=E:\GRP-PTv2
cwd=E:\GRP-PTv2
```

执行结果：

```text
DRY_RUN_PASSED
```

dry run 完成了 Dataset 加载、单 batch、CPU 邻接矩阵构建、CUDA forward、交叉熵 loss、有限值检查、backward、optimizer step、checkpoint 保存和 strict 重新加载。

| 项目 | 实际结果 |
|---|---:|
| mode | `from_scratch` |
| points | `[2,2048,4]`, `cuda:0` |
| adjacency | `[2,2048,2048]`, `cuda:0` |
| logits | `[2,2048,2]`, `cuda:0` |
| loss | 0.669112 |
| accuracy | 0.634766 |
| mIoU | 0.449118 |
| 单步耗时 | 1.154074 秒 |
| CPU 邻接构图耗时 | 0.053753 秒 |
| GPU 峰值显存 | 454.970703 MiB |
| last checkpoint strict 重载 | 所有 key 匹配 |
| best checkpoint strict 重载 | 所有 key 匹配 |

dry run 产物目录：

```text
artifacts/subdataset_smoke/20260714_155725_569825_from_scratch_dryrun/
```

包含 `last_model.pth`、`best_model.pth`、`metrics.json`、`config_resolved.yaml` 和 `run.log`。

### 10.4 from_scratch 两轮训练

dry run 成功后执行：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe E:\GRP-PTv2\scripts\train_gcn_res_subdataset.py --config E:\GRP-PTv2\config\gcn_res_subdataset_smoke.yaml --mode from_scratch --epochs 2 --batch-size 2
```

新 run ID：

```text
20260714_155739_892803_from_scratch_train
```

数据划分仍为 train 54、val 18、test 18，三者无交叠。每个阶段首个 batch 的实际 shape 均为 points `[2,2048,4]`、adjacency `[2,2048,2048]`、logits `[2,2048,2]`，设备均为 `cuda:0`。

| Epoch | Train loss | Train accuracy | Val loss | Val accuracy | Val mIoU | Epoch 耗时 | CPU 邻接构图 | GPU 峰值显存 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.391042 | 0.805420 | 0.791259 | 0.768717 | 0.384359 | 5.431336 秒 | 1.766135 秒 | 530.422363 MiB |
| 2 | 0.276538 | 0.875696 | 0.601904 | 0.783773 | 0.430208 | 4.532272 秒 | 1.783647 秒 | 534.182617 MiB |

其中 CPU 构图耗时分解为：

- Epoch 1：train 1.325134 秒，val 0.441001 秒。
- Epoch 2：train 1.347591 秒，val 0.436056 秒。

训练成功标志：

```text
SMOKE_TRAINING_PASSED
```

训练产物目录：

```text
artifacts/subdataset_smoke/20260714_155739_892803_from_scratch_train/
```

包含：

- `last_model.pth`
- `best_model.pth`
- `metrics.json`
- `config_resolved.yaml`
- `run.log`

训练结束后，`last_model.pth` 与 `best_model.pth` 均使用同一 `GCN_res`、`in_dim=4` 模型执行 strict 重新加载，结果均为：

```text
<All keys matched successfully>
```

### 10.5 最终结论

`GCN_res` 在 weld 子数据集上的独立短训练闭环已经跑通：数据加载、4 维输入拼接、k=6 稠密邻接矩阵、RTX 5060 forward/backward、Adam 更新、两轮 train/val、指标记录、独立 checkpoint 保存及 strict 重载全部成功。所有新产物均位于 `artifacts/subdataset_smoke`，没有写入或覆盖 `models/testParameters/GCN_res/best_model.pth`。
