# GRP-PTv2 Windows 本机 Python / GPU 环境审计

审计日期：2026-07-14（Asia/Shanghai）  
项目路径：`E:\GRP-PTv2`  
审计范围：Windows、Python、PyTorch、NVIDIA/CUDA、C++ 工具链、项目依赖与最小可运行性。  
操作边界：本次未安装、卸载或升级任何软件，未修改环境变量、虚拟环境或项目源码；只执行只读检查并新增本报告。

## 1. 结论摘要

当前环境**不能直接启动短训练**。RTX 5060 已被 Windows 和 NVIDIA 驱动正确识别，驱动版本也足够新；真正的直接阻塞是当前 Python 3.11 环境中没有安装 PyTorch、Hydra、OmegaConf、PyG/`torch-cluster` 等项目依赖。仓库中的 `.venv` 来自另一台电脑，虽然其中的 `python.exe --version` 能显示 3.8.19，但解释器初始化会因原机器路径和标准库缺失而失败，不能使用。

此外还发现两个与环境无关、会阻止默认短训练的项目问题：

1. 训练入口将当前 weld 输入构造成 4 维特征，但默认模型 `Nico_v2_GCN_Drop.PTV2Segmentation` 的第一层固定为 19 维，静态检查已确认不匹配。
2. `WeldDataset` 和训练入口仍固定使用原始划分及原作者 D 盘数据路径。原始 train/val JSON 共引用 36 个不存在的文件；新建的 54/18/18 子集 JSON 已通过验证，但当前 `WeldDataset` 尚未选择它们。

环境层面的总体判断：

| 项目 | 状态 | 结论 |
|---|---|---|
| RTX 5060 / NVIDIA 驱动 | 正常 | GPU 为 compute capability 12.0，驱动 610.74 |
| 当前 Python | 可用但为全局环境 | Python 3.11.8，不在 venv/Conda 中 |
| 仓库 `.venv` | 无效 | 来自 `C:\Users\Someone\.conda\envs\pointtransformer`，无法初始化标准库 |
| PyTorch GPU | 未配置 | 当前 Python 中没有 `torch`，GPU 运算无法测试 |
| CUDA Toolkit / `nvcc` | 未安装 | 训练预编译 PyTorch wheel 时不一定需要；后续 CUDA/C++ 编译与 TensorRT 需要 |
| PyG 点云扩展 | 未安装 | `torch_cluster`、`torch_geometric` 等全部不可导入 |
| C++ 工具链 | 部分就绪 | VS2022 x64 编译器存在；当前 PATH 却选中 VS2015 x86 `cl.exe` |
| CMake / Ninja | CMake 有，Ninja 无 | CMake 4.1.0；Ninja 未安装 |
| TensorRT | 未安装/不可发现 | `trtexec` 不存在，常见 TensorRT 目录也不存在 |
| 项目短训练 | 当前阻塞 | 依赖、路径/划分和输入维度均需后续处理 |

## 2. 当前 Python 环境

### 2.1 系统与终端

- 系统：Windows 10 家庭中文版 64 位，版本 `10.0.19045`，Build `19045`。
- 当前终端：Windows PowerShell `5.1.19041.6456`，Desktop edition，宿主为 `ConsoleHost`。

对应只读命令：

```powershell
Get-CimInstance Win32_OperatingSystem
$PSVersionTable
```

### 2.2 Python、pip、Conda 与虚拟环境

`where.exe python`：

```text
C:\Users\wlj\AppData\Local\Programs\Python\Python311\python.exe
C:\Users\wlj\AppData\Local\Microsoft\WindowsApps\python.exe
```

第二项是 WindowsApps 应用执行别名，不是第二套完整 Python。`py -0p` 只列出一套已注册解释器：

```text
 -V:3.11 * C:\Users\wlj\AppData\Local\Programs\Python\Python311\python.exe
```

当前解释器与 pip：

```text
Python 3.11.8
Executable: C:\Users\wlj\AppData\Local\Programs\Python\Python311\python.exe
sys.prefix: C:\Users\wlj\AppData\Local\Programs\Python\Python311
sys.base_prefix: C:\Users\wlj\AppData\Local\Programs\Python\Python311
pip 26.1.2 from C:\Users\wlj\AppData\Local\Programs\Python\Python311\Lib\site-packages\pip (python 3.11)
```

`VIRTUAL_ENV` 和 `CONDA_PREFIX` 均为空，`sys.prefix == sys.base_prefix`，因此当前**不在虚拟环境中**。`where.exe conda` 无结果，`conda --version` 和 `conda env list` 都返回 PowerShell `CommandNotFoundException`，本机当前 PATH 中无 Conda。

PATH 中有一套真实 Python 和一个 WindowsApps stub，当前命令解析到真实 Python，暂未出现多版本解释器抢占；但后续应始终使用 `python -m pip`，并在新建隔离环境后复查 `where.exe python`，避免 WindowsApps 别名造成误判。

### 2.3 仓库 `.venv` 有效性

文件：`E:\GRP-PTv2\.venv\pyvenv.cfg`

```text
home = C:\Users\Someone\.conda\envs\pointtransformer
implementation = CPython
version_info = 3.8.19.final.0
include-system-site-packages = false
base-prefix = C:\Users\Someone\.conda\envs\pointtransformer
base-executable = C:\Users\Someone\.conda\envs\pointtransformer\pythonw.exe
```

`E:\GRP-PTv2\.venv\Scripts\python.exe --version` 会打印 `Python 3.8.19`，但执行任意 `-c` 代码时失败，完整核心异常为：

```text
Python path configuration:
  sys.base_prefix = 'C:\Users\Someone\.conda\envs\pointtransformer'
  sys.path = [
    'C:\Users\Someone\.conda\envs\pointtransformer\python38.zip',
    'C:\Users\Someone\.conda\envs\pointtransformer\DLLs',
    'C:\Users\Someone\.conda\envs\pointtransformer\lib',
    'E:\GRP-PTv2',
  ]
Fatal Python error: init_fs_encoding: failed to get the Python codec of the filesystem encoding
ModuleNotFoundError: No module named 'encodings'
```

结论：该 `.venv` 是从其他电脑复制过来的虚拟环境，基解释器绝对路径已经失效，且本地环境内不含可供启动的完整标准库。**不要直接修补或继续使用它**；本次也未删除或重建它。

## 3. 当前 GPU 和 CUDA 环境

### 3.1 NVIDIA GPU 与驱动

`nvidia-smi` 关键输出：

```text
NVIDIA-SMI 610.74
Driver Version: 610.74
CUDA UMD Version: 13.3
GPU 0: NVIDIA GeForce RTX 5060
WDDM
Memory: 8151 MiB
```

`nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,pci.bus_id --format=csv`：

```text
NVIDIA GeForce RTX 5060, 610.74, 12.0, 8151 MiB, 00000000:01:00.0
```

因此 RTX 5060 被系统正确识别，GPU compute capability 为 `12.0`。

### 3.2 必须区分的三种 CUDA 版本

| 名称 | 本机实测 | 含义 |
|---|---|---|
| `nvidia-smi` CUDA / CUDA UMD | 13.3 | 驱动当前支持的最高 CUDA Driver API/UMD 能力，不代表已安装 Toolkit |
| CUDA Toolkit | 未发现 | `nvcc` 不存在，`CUDA_PATH` 为空，默认 Toolkit 目录不存在 |
| PyTorch CUDA Runtime | 无法取得 | 当前 Python 未安装 PyTorch，`torch.version.cuda` 无法执行 |

CUDA Toolkit 检查结果：

```text
where.exe nvcc
INFO: Could not find files for the given pattern(s).

nvcc --version
CommandNotFoundException: The term 'nvcc' is not recognized ...

$env:CUDA_PATH
<empty>
```

`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA` 也不存在。这个结果不妨碍未来使用带 CUDA Runtime 的 PyTorch 预编译 wheel 进行训练，但会阻止本地 CUDA 扩展编译、CUDA C++ 编译和需要 Toolkit 的 TensorRT 开发流程。

### 3.3 Visual Studio、MSVC、CMake 与 Ninja

Visual Studio Installer 的 `vswhere.exe` 检出：

```text
Visual Studio Community 2022
installationVersion: 17.8.2
installationPath: D:\vs2022\Community
state: complete / launchable
component: Microsoft.VisualStudio.Component.VC.Tools.x86.x64
```

VS2022 的 x64 编译器确实存在：

```text
D:\vs2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64\cl.exe
用于 x64 的 Microsoft (R) C/C++ 优化编译器 19.38.33130 版
```

但是当前普通 PowerShell 的 `where.exe cl` 返回：

```text
D:\vs2015\VC\bin\cl.exe
Microsoft (R) C/C++ Optimizing Compiler Version 19.00.24215.1 for x86
```

因此不是“没有 MSVC”，而是**当前终端 PATH 选中了旧的 VS2015 x86 编译器**。CUDA 12+ 已移除 32 位编译支持，后续构建 CUDA/TensorRT C++ 项目时应从 VS2022 x64 Developer PowerShell/命令提示符初始化工具链。NVIDIA 的 CUDA 13.0 Windows 文档列出 VS2022/MSVC 193x 的原生 x86_64 支持，并说明 CUDA 13.0 已移除 VS2017、CUDA 12.0 起移除 32 位编译支持：[CUDA Windows 编译器支持](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-installation-guide-microsoft-windows/index.html)。

其他工具：

```text
CMake: D:\CMake-4.1.0\bin\cmake.exe
cmake version 4.1.0
Ninja: 未发现，ninja --version -> CommandNotFoundException
```

### 3.4 TensorRT 探测

```text
where.exe trtexec
INFO: Could not find files for the given pattern(s).

trtexec --version
CommandNotFoundException: The term 'trtexec' is not recognized ...
```

以下常见目录均不存在：

```text
C:\Program Files\NVIDIA Corporation\TensorRT : False
C:\TensorRT                                      : False
D:\TensorRT                                      : False
```

结论：当前没有可发现的 TensorRT SDK/CLI。NVIDIA 当前安装前置条件也明确要求受支持的 GPU、驱动和 CUDA Toolkit；Windows 的 zip 包是官方支持的安装方式之一：[TensorRT prerequisites](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/prerequisites.html)、[Windows zip installation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html)。

## 4. PyTorch GPU 验证结果

按要求使用当前 Python 执行 PyTorch 诊断脚本，结果在第一条 `import torch` 即失败：

```text
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
ModuleNotFoundError: No module named 'torch'
```

最小 GPU 矩阵乘法测试同样完整失败：

```text
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'torch'
```

所以以下项目目前不是“测试失败”，而是**尚无 PyTorch 可供测试**：

- `torch.__version__`
- `torch.version.cuda`
- `torch.cuda.is_available()`
- cuDNN 版本
- PyTorch 识别的 GPU 名称与 capability
- CUDA tensor 分配、矩阵乘法和显存分配

不能据此判断未来是否会出现 `no kernel image is available for execution`；必须在安装与 RTX 5060/SM 12.0 匹配的 PyTorch 及扩展后重新实测。

## 5. 项目依赖清单

### 5.1 实际 import 与 requirements 对比

仓库 `E:\GRP-PTv2\requirements.txt` 仅包含：

```text
numpy
torch
tqdm
hydra-core==1.2
omegaconf
```

对项目 Python 文件（排除 `.venv`）做 AST/import 搜索后，实际还出现 `scipy`、`sklearn`、`torch_geometric`、`torch_cluster`、`torchinfo`、`torchsummary`、`onnx`、`onnxruntime` 等。因而不能只依赖现有 `requirements.txt`。

当前全局 Python 3.11.8 的实测清单：

| 用户要求检查的依赖 | 当前状态 | 版本/异常 | 项目实际使用 |
|---|---|---|---|
| numpy | 已安装且可导入 | 1.26.4 | 是 |
| torch | 未安装 | `ModuleNotFoundError: No module named 'torch'` | 是，核心依赖 |
| torchvision | 未安装 | `ModuleNotFoundError: No module named 'torchvision'` | 当前代码未发现直接 import |
| tqdm | 未安装 | `ModuleNotFoundError: No module named 'tqdm'` | 是 |
| hydra-core (`hydra`) | 未安装 | `ModuleNotFoundError: No module named 'hydra'` | 是 |
| omegaconf | 未安装 | `ModuleNotFoundError: No module named 'omegaconf'` | 是 |
| scikit-learn (`sklearn`) | 未安装 | `ModuleNotFoundError: No module named 'sklearn'` | 是 |
| scipy | 未安装 | `ModuleNotFoundError: No module named 'scipy'` | 是 |
| torch-geometric | 未安装 | `ModuleNotFoundError: No module named 'torch_geometric'` | 是 |
| torch-cluster | 未安装 | `ModuleNotFoundError: No module named 'torch_cluster'` | 是，模型工具中直接使用 |
| torch-scatter | 未安装 | `ModuleNotFoundError: No module named 'torch_scatter'` | 当前代码未发现直接 import，但可能为 PyG 扩展依赖 |
| torch-sparse | 未安装 | `ModuleNotFoundError: No module named 'torch_sparse'` | 当前代码未发现直接 import，但可能为 PyG 扩展依赖 |
| torchinfo | 未安装 | `ModuleNotFoundError: No module named 'torchinfo'` | 是 |
| torchsummary | 未安装 | `ModuleNotFoundError: No module named 'torchsummary'` | 是 |
| onnx | 未安装 | `ModuleNotFoundError: No module named 'onnx'` | 是 |
| onnxruntime | 未安装 | `ModuleNotFoundError: No module named 'onnxruntime'` | 是 |
| onnxruntime-gpu | 未安装 | 无分发包；其导入名同为 `onnxruntime` | 后续 GPU ORT 验证才需要 |
| packaging | 已安装且可导入 | 24.0 | 当前项目未直接使用，安装/版本判断工具 |
| setuptools | 已安装且可导入 | 65.5.0 | 当前项目未直接使用，构建工具 |
| wheel | 未安装 | `ModuleNotFoundError: No module named 'wheel'` | 当前项目未直接使用，构建工具 |

由于核心依赖缺失，当前无法进行真实的版本组合兼容性判定。`numpy 1.26.4`、`packaging 24.0` 和 `setuptools 65.5.0` 在当前解释器中可以正常导入；这不代表它们已经与未来选择的 PyTorch/PyG 组合完成验证。

## 6. 点云扩展依赖状态

三个重点 import 的完整异常类型一致：

```text
import torch_cluster
ModuleNotFoundError: No module named 'torch_cluster'

import torch_scatter
ModuleNotFoundError: No module named 'torch_scatter'

import torch_geometric
ModuleNotFoundError: No module named 'torch_geometric'
```

本项目并非只通过 `torch_geometric` 间接使用扩展，而是在多个 `models/*/ptv2_utils.py` 中直接使用 `torch_cluster`，训练入口还从 `torch_geometric.nn` 导入图构建/近邻功能。因此必须确保 `torch_cluster` 的 wheel 与下列四项同时匹配：

- Windows x64
- Python ABI（当前候选为 CPython 3.11）
- 精确 PyTorch 版本
- 精确 CUDA Runtime 变体

PyG 官方安装文档要求扩展 wheel 与 `${TORCH}`、`${CUDA}` 组合匹配，并提供 `data.pyg.org` wheel 索引：[PyG installation](https://pytorch-geometric.readthedocs.io/en/stable/notes/installation.html)。当前环境没有任何扩展，所以尚未出现 ABI、CUDA capability 或 `no kernel image` 错误；也不能用“包名存在”代替安装后的实际 CUDA 运算验证。

## 7. 项目最小可运行性

### 7.1 语法与配置

为避免 `py_compile` 写入 `.pyc`，本次采用 Python 内置 `compile(source, path, 'exec')` 做等价的只读语法检查：

```text
dataset.py                              SYNTAX_OK
train_partseg_weld_V2improved.py        SYNTAX_OK
export2ONNX.py                          SYNTAX_OK
```

`config` 下所有 YAML 均可被当前已存在的 PyYAML `safe_load` 解析，结果为 `YAML_OK`。但是 Hydra 的真实 compose/import 无法进行：

```text
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'hydra'
```

因此只能确认 YAML 语法，不可声称 Hydra 配置已经成功组合。

### 7.2 Dataset 与模型导入

`dataset.py` 与 `WeldDataset` 的导入在 `dataset.py` 顶部失败：

```text
Traceback (most recent call last):
  ...
  File "E:\GRP-PTv2\dataset.py", line 4, in <module>
    from torch.utils.data import Dataset
ModuleNotFoundError: No module named 'torch'
```

默认模型 `models.Nico_v2_GCN_Drop.model` 也在首行 `import torch` 失败。这里是依赖缺失，并非已证明模块内部逻辑正确。

### 7.3 默认输入维度不一致

默认模型由 `E:\GRP-PTv2\config\partseg_v2_improved.yaml:16` 选择：

```yaml
- model: Nico_v2_GCN_Drop
```

训练入口：

- `E:\GRP-PTv2\train_partseg_weld_V2improved.py:107`：`args.input_dim = (3) + 1 = 4`（默认 `normal: False`）。
- `E:\GRP-PTv2\train_partseg_weld_V2improved.py:190`、`:194`：把 XYZ 点特征与 1 维类别 one-hot 拼接，形成 `[B, N, 4]`。
- `E:\GRP-PTv2\dataset.py:342-347`：`WeldDataset` 读取 TXT，点特征使用前三列，最后一列为分割标签。

默认模型：

- `E:\GRP-PTv2\models\Nico_v2_GCN_Drop\model.py:144`：`PTV2Segmentation.__init__(..., in_dim=19)`。
- `E:\GRP-PTv2\models\Nico_v2_GCN_Drop\model.py:146`：`nn.Linear(19, 48)`。
- `E:\GRP-PTv2\models\Nico_v2_GCN_Drop\model.py:148`：LFA 同样固定 `input_dim=19`。

训练构造模型时只传入 `args`（`train_partseg_weld_V2improved.py:115`），没有覆盖 `in_dim`。所以当前默认组合会把 `[B,N,4]` 输入送入 `Linear(19,48)`，属于确定的静态维度不匹配。正式训练前必须先确认 19 维设计意图、历史权重结构和实际数据特征，不能直接猜成 4 或人为补 15 列。

### 7.4 weld 数据与划分

目录 `E:\GRP-PTv2\data\weld` 存在，实际 weld 主文件为 `weld_1.txt` 到 `weld_90.txt`，共 90 个。子集验证脚本：

`E:\GRP-PTv2\data\weld\train_test_split\validate_sub_split.py`

运行结果：

```text
Sub-dataset validation PASSED
  seed: 42
  split counts: train=54, val=18, test=18
  referenced TXT files: 90
  total data rows: 184320
  labels: [0.0, 1.0]
```

原始划分实测：

| JSON | 引用数 | 缺失数 | 缺失范围 |
|---|---:|---:|---|
| `shuffled_train_file_list.json` | 76 | 27 | `weld_100.txt`～`weld_126.txt` |
| `shuffled_val_file_list.json` | 25 | 9 | `weld_91.txt`～`weld_99.txt` |
| `shuffled_test_file_list.json` | 5 | 0 | 无 |

新增子划分均通过存在性、互斥性、四列和二值标签验证：

- `sub_shuffled_train_file_list.json`：54
- `sub_shuffled_val_file_list.json`：18
- `sub_shuffled_test_file_list.json`：18

但 `E:\GRP-PTv2\dataset.py:286-290` 的 `WeldDataset.__init__` 仍固定打开原始 `shuffled_*_file_list.json`。因此子划分文件虽然有效，当前默认 Dataset 尚不会使用它们。

### 7.5 checkpoint

当前训练代码默认从工作目录读取/写入 `best_model.pth`：

- 加载：`E:\GRP-PTv2\train_partseg_weld_V2improved.py:121`
- 保存：`E:\GRP-PTv2\train_partseg_weld_V2improved.py:284`

仓库根目录没有 `best_model.pth`，也没有现成 `log` 目录。发现两个模型 checkpoint：

```text
E:\GRP-PTv2\models\testParameters\GCN_LFA_res\best_model.pth  79,883,266 bytes
E:\GRP-PTv2\models\testParameters\GCN_res\best_model.pth      74,340,448 bytes
```

由于当前没有 PyTorch，本次没有反序列化 checkpoint，不能确认其模型名称、输入维度、epoch 或 state dict 是否与默认模型一致。

### 7.6 原作者硬编码路径

至少以下当前关键入口含原作者 D 盘绝对路径：

- `E:\GRP-PTv2\train_partseg_weld_V2improved.py:97`：数据根目录。
- `E:\GRP-PTv2\export2ONNX.py:75`：checkpoint 路径。
- `E:\GRP-PTv2\export2ONNX.py:84`：数据根目录。
- `E:\GRP-PTv2\predict_partseg_weld.py:67-75`：checkpoint、输出和数据根目录。

训练入口的硬编码路径为 `D:/xlxlqqq/.../data/weld/`，与本机项目 `E:\GRP-PTv2\data\weld` 不一致。仓库内还有多个旧实验文件包含 D/Q 盘路径，后续应先区分“当前入口”和“历史备份”，再做集中配置化处理。

## 8. RTX 5060 兼容性判断

### 8.1 已经能够确认的事实

- GPU：RTX 5060，compute capability `12.0`，驱动能够正常枚举设备。
- 驱动：610.74，`nvidia-smi` 报告 CUDA UMD 13.3。
- NVIDIA 的架构矩阵将 compute capability 12.0 对应为 Blackwell，并将其首次 Toolkit 支持列为 CUDA 12.8：[CUDA Toolkit/driver/architecture matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html)。
- PyTorch 2.7 官方发布说明明确加入 Blackwell 支持和 CUDA 12.8 预编译 wheel；当前 PyTorch 发布矩阵还提供更新的 CUDA 13.x 选择：[PyTorch 2.7 release](https://pytorch.org/blog/pytorch-2-7/)、[PyTorch release compatibility matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)。
- 本机驱动不是当前限制因素；PyTorch、CUDA Runtime 与扩展根本尚未安装。

### 8.2 风险逐项判断

| 风险 | 当前判断 | 依据 |
|---|---|---|
| PyTorch 太旧、无 SM 12.0 支持 | 当前没有 PyTorch，必需选支持 Blackwell 的版本 | `import torch` 失败；GPU 实测 SM 12.0 |
| `no kernel image is available` | 尚未发生，但错误选用旧 CUDA/PyTorch/扩展 wheel 时风险高 | 当前没有可运行 CUDA kernel；需安装后实测 |
| capability 不匹配 | GPU 端为 12.0；软件端未知 | `nvidia-smi` 已确认 12.0，PyTorch/扩展缺失 |
| PyG 扩展 wheel 无对应版本 | 是主要安装风险 | 本项目直接依赖 `torch_cluster`；必须匹配 Windows/Python/Torch/CUDA 四元组 |
| 扩展需要本地编译 | 取决于所选组合是否有 Windows wheel | 当前无 Toolkit、Ninja，普通终端还选中 VS2015 x86 |
| MSVC/CUDA Toolkit 不兼容 | 当前构建环境不成立 | Toolkit 无；当前 `cl` 为 VS2015 x86，虽另有可用 VS2022 x64 |
| TensorRT 不支持 GPU | RTX 5060/SM 12.0 属于当前支持范围，但版本仍须按矩阵配套 | TensorRT 支持矩阵覆盖 SM 7.5+；TensorRT-RTX 矩阵明确列出 Blackwell SM 12.0：[TensorRT matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)、[TensorRT-RTX matrix](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/support-matrix-1/1.3.html) |

不能仅凭驱动支持 CUDA 13.3 就安装任意 CUDA 13.x 组件。PyTorch wheel 自带的 CUDA Runtime、PyG 扩展 wheel、未来 TensorRT 包与本地 Toolkit 必须选择成官方支持的一组组合。

## 9. 当前阻塞问题

按“离 2～5 epoch 短训练的距离”排序：

1. **没有可用项目虚拟环境**：当前为全局 Python；仓库 `.venv` 已损坏/不可迁移。
2. **核心依赖全部缺失**：PyTorch、Hydra、OmegaConf、tqdm、PyG、`torch_cluster` 等无法导入。
3. **默认输入维度确定不一致**：训练输入 4 维，默认模型入口 19 维。
4. **训练数据根目录硬编码到不存在的 D 盘路径**。
5. **Dataset 默认仍读取有缺失文件的原始 train/val JSON**，不会自动使用已验证的 sub JSON。
6. **checkpoint 使用关系不清楚**：根目录无默认 checkpoint；两个历史权重尚未验证对应模型结构。
7. **无法进行 RTX 5060 的 PyTorch/扩展 CUDA kernel 实测**，因为 torch 与扩展未安装。
8. **后续本地编译链不完整**：无 CUDA Toolkit、无 Ninja，当前终端 `cl` 指向 VS2015 x86。
9. **TensorRT 未安装**：无 `trtexec`，也没有可发现的 SDK 目录。

## 10. 推荐环境方案

本节只是建议，**本次没有执行安装或修改**。

### 10.1 训练环境

建议保留系统 Python 3.11.8，但不要继续在全局 site-packages 中堆叠依赖；后续新建干净、可重建的 Python 3.11 x64 虚拟环境。不要复用当前 `.venv`。

版本选择应先以 `torch_cluster` 的 Windows wheel 是否存在为约束，再锁定整套组合。两个合理方向：

- **保守方案**：Python 3.11 + 支持 Blackwell 的 PyTorch CUDA 12.8 版本 + `data.pyg.org` 上同一 Torch/CUDA 标签的 Windows CPython 3.11 扩展 wheel。该方向更接近项目旧代码依赖方式，也有利于避免本地编译扩展。
- **新版本方案**：Python 3.11 + 当前稳定 PyTorch CUDA 13.x + 完全匹配的 PyG/`torch_cluster` Windows wheel。只有在安装前确认 wheel 索引确实覆盖该精确组合时才采用。

不要混装例如“PyTorch cu128 + `torch_cluster` cu130”，也不要从不匹配的旧 Torch 版本复制 `.pyd` 文件。PyG 的官方安装页与 wheel 索引应作为版本锁定依据：[PyG installation](https://pytorch-geometric.readthedocs.io/en/stable/notes/installation.html)、[PyG wheel index](https://data.pyg.org/whl/)。

### 10.2 CUDA Toolkit 与 C++/TensorRT

- 仅跑 PyTorch 预编译 wheel 的短训练时，可以先不安装系统 CUDA Toolkit；以 `torch.version.cuda` 报告的 wheel Runtime 为准。
- 如果 PyG 没有匹配 wheel而必须源码编译，或进入 TensorRT C++/CUDA 阶段，则需要安装与目标版本矩阵一致的 CUDA Toolkit。
- 使用 `D:\vs2022\Community` 的 x64 工具链，不使用当前 PATH 中的 VS2015 x86 `cl.exe`。
- TensorRT 应在 ONNX 模型和动态 shape 策略确定后再选版本，并严格按 TensorRT support matrix 配套 Toolkit/驱动。TensorRT engine 通常不应被当成跨平台、跨 TensorRT 版本或跨 GPU 架构的通用文件；官方矩阵也明确说明 serialized engine 的平台/版本/硬件可移植性限制：[TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)。

## 11. 后续安装步骤建议

以下是后续实施顺序，不是本次已执行操作：

1. 明确将使用的短训练模型及其真实输入维度；核对两个现有 checkpoint 的模型结构，先解决 4/19 维矛盾。
2. 决定让 `WeldDataset` 使用 sub JSON 的方式，并把当前入口的数据根目录切换到本机路径；这是后续代码修改阶段的工作。
3. 在安装前先从 PyTorch 官方矩阵和 PyG wheel 索引锁定一个完整组合，尤其确认 Windows CPython 3.11 的 `torch_cluster` wheel。
4. 新建干净虚拟环境，先安装锁定的 PyTorch，再运行本报告第 12 节的 GPU 基础复查。
5. 安装纯 Python 项目依赖和完全匹配的 PyG 扩展；分别做 CPU import、CUDA `knn`/图构建、一次模型前向。
6. 仅在上一步全部通过后，使用 sub dataset 跑 2～5 epoch，并记录显存、loss、验证指标、checkpoint 可加载性。
7. 用相同 checkpoint 做 PyTorch 推理基线，再修复/验证 ONNX 导出和 ONNX Runtime 数值一致性。
8. 根据 ONNX 中实际算子、动态点数/批次需求选择 TensorRT 版本与 Toolkit；先用 `trtexec` 构建和验证，再实现 C++ Runtime。
9. 最后将预处理、输入 shape/profile、后处理和 label 映射固化到工业 C++ 软件接口。

## 12. 用于复查的命令

以下命令均为复查命令；创建环境或安装包的命令刻意没有写入，避免在版本组合尚未锁定时误操作。

### 12.1 Python 与终端

```powershell
where.exe python
where.exe pip
where.exe py
py -0p
python --version
python -c "import sys, os; print(sys.executable); print(sys.prefix); print(sys.base_prefix); print(sys.path); print(os.environ.get('VIRTUAL_ENV')); print(os.environ.get('CONDA_PREFIX'))"
python -m pip --version
conda --version
conda env list
E:\GRP-PTv2\.venv\Scripts\python.exe -c "import sys; print(sys.executable); print(sys.prefix)"
```

### 12.2 NVIDIA、CUDA 与编译工具

```powershell
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,pci.bus_id --format=csv
where.exe nvcc
nvcc --version
$env:CUDA_PATH
where.exe cl
cl
where.exe cmake
cmake --version
where.exe ninja
ninja --version
where.exe trtexec
trtexec --version
```

### 12.3 PyTorch GPU

```powershell
@'
import sys
import torch

print("Python:", sys.version)
print("Executable:", sys.executable)
print("Torch:", torch.__version__)
print("Torch CUDA Runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN:", torch.backends.cudnn.version())
print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print("GPU", i, torch.cuda.get_device_name(i))
    print("Capability", torch.cuda.get_device_capability(i))
    print("Total memory", torch.cuda.get_device_properties(i).total_memory)
'@ | python -
```

```powershell
@'
import torch
x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")
z = x @ y
torch.cuda.synchronize()
print(z.shape)
print(z.device)
print(torch.cuda.memory_allocated())
'@ | python -
```

### 12.4 点云扩展实际 CUDA 验证

```powershell
@'
import torch
import torch_cluster
import torch_scatter
import torch_geometric
from torch_cluster import knn

print("torch", torch.__version__, "runtime", torch.version.cuda)
print("torch_cluster", torch_cluster.__file__)
print("torch_scatter", torch_scatter.__file__)
print("torch_geometric", torch_geometric.__version__)

x = torch.randn(32, 3, device="cuda")
y = torch.randn(8, 3, device="cuda")
edge = knn(x, y, k=4)
torch.cuda.synchronize()
print(edge.shape, edge.device)
'@ | python -
```

### 12.5 项目静态与数据复查

```powershell
Set-Location E:\GRP-PTv2
python -c "from pathlib import Path; [compile(p.read_text(encoding='utf-8-sig'), str(p), 'exec') for p in map(Path, ['dataset.py','train_partseg_weld_V2improved.py','export2ONNX.py'])]; print('SYNTAX_OK')"
python -c "from dataset import WeldDataset; print(WeldDataset)"
python -c "import models.Nico_v2_GCN_Drop.model as m; print(m.PTV2Segmentation)"
python data\weld\train_test_split\validate_sub_split.py
rg -n "D:/|D:\\\\|Q:/|Q:\\\\" -g "*.py"
```

完成依赖安装后的复查应保留完整终端输出，尤其是 `torch.__version__`、`torch.version.cuda`、`torch.cuda.get_device_capability()`、扩展模块文件路径以及最小 `knn` CUDA 运算结果；这些输出才足以最终排除 RTX 5060 的 kernel/ABI 不兼容风险。
