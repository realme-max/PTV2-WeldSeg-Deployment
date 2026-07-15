# GRP-PTv2 Windows 本地训练环境安装结果

执行日期：2026-07-14（Asia/Shanghai）  
项目：`E:\GRP-PTv2`  
目标环境：`E:\GRP-PTv2\.venv_ptv2`  
目标组合：Python 3.11、PyTorch 2.7.1、CUDA Runtime 12.8、PyG pt27/cu128 Windows wheels  
历史初次状态：第四阶段曾失败并按要求停止；后续获准修正安装顺序及 Hydra 版本后已继续验证。当前最终状态见第 10 节。

## 1. 结果摘要

| 阶段 | 状态 | 结果 |
|---|---|---|
| 1. 创建环境 | 通过 | 新环境为 Python 3.11.8，解释器指向 `.venv_ptv2\Scripts\python.exe` |
| 2. 基础工具 | 通过 | pip 26.1.2、setuptools 83.0.0、wheel 0.47.0 |
| 3. PyTorch cu128 | 通过 | torch 2.7.1+cu128、torchvision 0.22.1+cu128；RTX 5060/SM 12.0 与 CUDA 矩阵乘法通过 |
| 4. PyG 二进制扩展 | **失败并停止** | 指定 Windows wheels 全部找到，但 `torch_sparse` 依赖 SciPy；`--no-index` 无法取得尚未安装的 SciPy |
| 5. torch_geometric/项目依赖 | 未执行 | 因第四阶段失败停止 |
| 6. torch_cluster CUDA KNN | 未执行 | `torch_cluster` 未安装 |
| 7. checkpoint/CUDA forward | 未执行 | PyG 扩展与项目依赖尚未完成 |

失败不是 PyTorch、CUDA、RTX 5060 或目标 wheel 不兼容。目标 `torch_cluster-1.6.3+pt27cu128-cp311-cp311-win_amd64.whl` 已从官方页面正确解析并下载；pip 在正式安装前进行依赖解析时，因为 SciPy 尚未安装且命令包含 `--no-index` 而退出。pip 没有部分安装任何 PyG 扩展。

按照任务约束，本次没有自行更换 PyTorch/CUDA/PyG 版本，没有从普通 PyPI 编译扩展，没有安装 CPU/cu118/cu121/cu124/cu130 变体，也没有在失败后提前执行第五阶段。

## 2. 第一阶段：创建环境

### 2.1 创建前检查

```powershell
Test-Path E:\GRP-PTv2\.venv_ptv2
```

输出：

```text
False
```

目标目录原先不存在，因此没有覆盖或删除已有环境。

### 2.2 创建命令

```powershell
python -m venv E:\GRP-PTv2\.venv_ptv2
```

创建成功。原损坏环境 `E:\GRP-PTv2\.venv` 未被使用、修改或删除。

### 2.3 PowerShell 激活限制

首次执行：

```powershell
& E:\GRP-PTv2\.venv_ptv2\Scripts\Activate.ps1
```

完整错误：

```text
File E:\GRP-PTv2\.venv_ptv2\Scripts\Activate.ps1 cannot be loaded because
running scripts is disabled on this system.
CategoryInfo          : SecurityError
FullyQualifiedErrorId : UnauthorizedAccess
```

这是当前 Windows PowerShell execution policy 对 `.ps1` 的限制，不是虚拟环境损坏。本次没有修改系统、用户或进程 execution policy，而是使用 `activate.bat` 做临时验证，后续所有安装命令都直接调用新环境的绝对解释器。

### 2.4 绝对解释器验证

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe --version
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -m pip --version
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -c "import sys; print(sys.executable); print(sys.prefix); print(sys.base_prefix)"
```

输出：

```text
Python 3.11.8
pip 24.0 from E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\pip (python 3.11)
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe
E:\GRP-PTv2\.venv_ptv2
C:\Users\wlj\AppData\Local\Programs\Python\Python311
VENV_ASSERTIONS_PASSED
```

满足 `sys.prefix != sys.base_prefix`，证明该解释器处于新虚拟环境。

### 2.5 `activate.bat` 验证

```powershell
cmd.exe /d /c "call E:\GRP-PTv2\.venv_ptv2\Scripts\activate.bat && where.exe python && python --version && python -m pip --version"
```

输出：

```text
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe
C:\Users\wlj\AppData\Local\Programs\Python\Python311\python.exe
C:\Users\wlj\AppData\Local\Microsoft\WindowsApps\python.exe
Python 3.11.8
pip 24.0 from E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\pip (python 3.11)
```

`where.exe python` 第一项符合要求。验证期间曾有一次附加 `cmd.exe` 内嵌 `python -c` 命令因 PowerShell/cmd 双层引号出现 `SyntaxError: unexpected EOF while parsing`；该附加命令未修改环境，随后以上简化命令和绝对解释器断言均成功。

第一阶段结论：**通过**。

## 3. 第二阶段：升级基础工具

执行命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

关键安装输出：

```text
Successfully installed packaging-26.2 pip-26.1.2 setuptools-83.0.0 wheel-0.47.0
```

复查：

```text
pip 26.1.2 from E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\pip (python 3.11)
setuptools 83.0.0
wheel 0.47.0
```

第二阶段结论：**通过**。

## 4. 第三阶段：安装并验证 PyTorch 2.7.1 cu128

### 4.1 安装命令

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -m pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128
```

pip 从官方 cu128 index 解析到：

```text
torch-2.7.1+cu128-cp311-cp311-win_amd64.whl
torchvision-0.22.1+cu128-cp311-cp311-win_amd64.whl
```

主 PyTorch wheel 大小约 3.27 GB，下载约 5 分 36 秒。最终输出：

```text
Successfully installed MarkupSafe-3.0.3 filelock-3.29.0 fsspec-2026.4.0
jinja2-3.1.6 mpmath-1.3.0 networkx-3.6.1 numpy-2.4.4 pillow-12.2.0
sympy-1.14.0 torch-2.7.1+cu128 torchvision-0.22.1+cu128
typing-extensions-4.15.0
```

这里由 torchvision 临时带入了 NumPy 2.4.4；原计划在第五阶段锁定为 NumPy 1.26.4，但第五阶段因停止条件未执行。因此当前环境中的 NumPy 仍是 2.4.4。

### 4.2 GPU 身份与 Runtime 验证

输出：

```text
Torch: 2.7.1+cu128
Torch CUDA Runtime: 12.8
CUDA available: True
GPU: NVIDIA GeForce RTX 5060
Capability: (12, 0)
cuDNN: 90701
PYTORCH_GPU_IDENTITY_PASSED
```

以下断言全部通过：

```text
torch.__version__.startswith("2.7.1")
torch.version.cuda == "12.8"
torch.cuda.is_available() is True
torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5060"
torch.cuda.get_device_capability(0) == (12, 0)
```

### 4.3 CUDA 矩阵乘法

测试：

```python
x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")
z = x @ y
torch.cuda.synchronize()
```

输出：

```text
shape: torch.Size([1024, 1024])
device: cuda:0
finite: True
memory_allocated: 21102592
PYTORCH_CUDA_MATMUL_PASSED
```

第三阶段结论：**通过**。PyTorch 2.7.1 cu128 可以在本机 RTX 5060 / SM 12.0 上执行 CUDA kernel。

## 5. 第四阶段：PyG 二进制扩展安装失败

### 5.1 严格执行的原始命令

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv --no-index -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
```

没有移除 `--no-index`，没有加入 `--no-deps`，没有更换页面、PyTorch 或 CUDA 版本。

### 5.2 完整命令输出

```text
Looking in links: https://data.pyg.org/whl/torch-2.7.1+cu128.html
Collecting pyg_lib
  Downloading pyg_lib-0.5.0+pt27cu128-cp311-cp311-win_amd64.whl (4.1 MB)
Collecting torch_scatter
  Downloading torch_scatter-2.1.2+pt27cu128-cp311-cp311-win_amd64.whl (3.6 MB)
Collecting torch_sparse
  Downloading torch_sparse-0.6.18+pt27cu128-cp311-cp311-win_amd64.whl (2.1 MB)
Collecting torch_cluster
  Downloading torch_cluster-1.6.3+pt27cu128-cp311-cp311-win_amd64.whl (1.6 MB)
Collecting torch_spline_conv
  Downloading torch_spline_conv-1.2.2+pt27cu128-cp311-cp311-win_amd64.whl (603 kB)
INFO: pip is looking at multiple versions of torch-sparse to determine which version is compatible with other requirements.
Collecting pyg_lib
  Downloading pyg_lib-0.4.0+pt27cu128-cp311-cp311-win_amd64.whl (3.7 MB)
ERROR: Could not find a version that satisfies the requirement scipy (from torch-sparse) (from versions: none)
ERROR: No matching distribution found for scipy
```

### 5.3 原因

`torch_sparse` 的 wheel metadata 声明依赖 SciPy。原阶段顺序在第四阶段安装 PyG 扩展，而 SciPy 被安排在第五阶段安装；同时第四阶段命令使用 `--no-index`，pip 只能查看 `data.pyg.org` 的 PyG wheel 页面，不能从普通 package index 获取 SciPy。因此 resolver 在安装事务提交前退出。

这不是以下问题：

- 不是缺少 `torch_cluster 1.6.3 pt27/cu128/cp311/win_amd64` wheel；该 wheel 已准确找到。
- 不是回退到了源码包；所有下载项都是 Windows `.whl`。
- 不是 CPU/cu118/cu121/cu124/cu130 包混装。
- 不是 RTX 5060、CUDA Runtime 12.8 或 PyTorch 2.7.1 验证失败。

### 5.4 失败后的只读确认

```powershell
python -m pip show pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv
```

输出：

```text
WARNING: Package(s) not found: pyg_lib, torch_cluster, torch_scatter,
torch_sparse, torch_spline_conv
```

模块探测：

```text
pyg_lib          None
torch_scatter    None
torch_sparse     None
torch_cluster    None
torch_spline_conv None
scipy            None
```

第四阶段结论：**失败**。按照“若某一步失败，停止后续安装”的要求，流程在这里终止。

## 6. 当前环境的最终状态

当前关键包：

```text
numpy==2.4.4
packaging==26.2
pip==26.1.2
setuptools==83.0.0
torch==2.7.1+cu128
torchvision==0.22.1+cu128
wheel==0.47.0
```

已确认可用：

- Python 3.11.8 虚拟环境。
- PyTorch 2.7.1 CUDA 12.8。
- RTX 5060 / compute capability 12.0。
- cuDNN 90701。
- 基础 CUDA tensor 与矩阵乘法。

尚未安装：

- SciPy 与计划中的项目依赖。
- pyg_lib、torch_scatter、torch_sparse、torch_cluster、torch_spline_conv。
- torch_geometric。
- Hydra、OmegaConf、scikit-learn、ONNX、ONNX Runtime 等项目依赖。

尚未执行：

- `torch_cluster` CUDA KNN 测试。
- `GCN_res` checkpoint 的 PyTorch strict load。
- `[1,2048,4] + [1,2048,2048] -> [1,2048,2]` CUDA forward。
- `CHECKPOINT_AND_FORWARD_VALIDATION_PASSED` 验收。

## 7. 继续执行所需的用户决策

要继续而不改变锁定的 PyTorch/CUDA/PyG wheel 版本，最小恢复方案是调整依赖安装顺序：先安装原本第五阶段已经指定的 `numpy==1.26.4` 和 `scipy`，再原样重试第四阶段的 `--no-index -f https://data.pyg.org/whl/torch-2.7.1+cu128.html` 命令。

该方案会改变用户指定的阶段顺序，因此本次没有自行执行。另一个技术选项是给 PyG wheel 命令增加 `--no-deps`，但这会改变用户提供的严格命令，且在 SciPy 尚未安装时留下依赖不完整状态，本报告不建议默认采用。

在获得继续指示前，应保留当前 `.venv_ptv2`；不需要删除或重建，因为 Python、PyTorch 与 RTX 5060 CUDA 验证已经成功。

## 8. 获准调整顺序后的续装结果

续装日期：2026-07-14  
授权内容：先单独安装 SciPy，再原样重试带 `--no-index` 的官方 PyG wheel 命令。  
续装最终状态：**成功；PyG pt27/cu128 Windows 扩展导入与 CUDA KNN 均通过。**

### 8.1 环境复查

续装前重新检查了绝对解释器、虚拟环境前缀、PyTorch 和 CUDA Runtime：

```text
executable E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe
prefix E:\GRP-PTv2\.venv_ptv2
base_prefix C:\Users\wlj\AppData\Local\Programs\Python\Python311
torch 2.7.1+cu128
cuda_runtime 12.8
cuda_available True
ENVIRONMENT_RECHECK_PASSED
pip 26.1.2 from E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\pip (python 3.11)
```

结论：续装操作确实作用于 `E:\GRP-PTv2\.venv_ptv2`；没有使用原损坏的 `.venv`，PyTorch/CUDA 锁定版本没有变化。

### 8.2 单独安装 SciPy

执行命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -m pip install scipy
```

完整关键输出：

```text
Collecting scipy
  Downloading scipy-1.17.1-cp311-cp311-win_amd64.whl.metadata (60 kB)
Requirement already satisfied: numpy<2.7,>=1.26.4 in .\.venv_ptv2\Lib\site-packages (from scipy) (2.4.4)
Downloading scipy-1.17.1-cp311-cp311-win_amd64.whl (36.6 MB)
Installing collected packages: scipy
Successfully installed scipy-1.17.1
```

导入验证：

```text
scipy 1.17.1
E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\scipy\__init__.py
```

SciPy 安装使用 CPython 3.11 Windows AMD64 wheel，没有源码编译。

### 8.3 原样重试 PyG 官方 wheel 命令

执行命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv --no-index -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
```

`--no-index` 被完整保留；没有使用普通 PyPI 获取 PyG 扩展，没有更换 PyTorch/CUDA 版本，也没有触发源码编译。

完整输出：

```text
Looking in links: https://data.pyg.org/whl/torch-2.7.1+cu128.html
Collecting pyg_lib
  Using cached pyg_lib-0.5.0+pt27cu128-cp311-cp311-win_amd64.whl (4.1 MB)
Collecting torch_scatter
  Using cached torch_scatter-2.1.2+pt27cu128-cp311-cp311-win_amd64.whl (3.6 MB)
Collecting torch_sparse
  Using cached torch_sparse-0.6.18+pt27cu128-cp311-cp311-win_amd64.whl (2.1 MB)
Collecting torch_cluster
  Using cached torch_cluster-1.6.3+pt27cu128-cp311-cp311-win_amd64.whl (1.6 MB)
Collecting torch_spline_conv
  Using cached torch_spline_conv-1.2.2+pt27cu128-cp311-cp311-win_amd64.whl (603 kB)
Requirement already satisfied: scipy in .\.venv_ptv2\Lib\site-packages (from torch_sparse) (1.17.1)
Requirement already satisfied: numpy<2.7,>=1.26.4 in .\.venv_ptv2\Lib\site-packages (from scipy->torch_sparse) (2.4.4)
Installing collected packages: torch_spline_conv, torch_scatter, pyg_lib, torch_sparse, torch_cluster
Successfully installed pyg_lib-0.5.0+pt27cu128 torch_cluster-1.6.3+pt27cu128
torch_scatter-2.1.2+pt27cu128 torch_sparse-0.6.18+pt27cu128
torch_spline_conv-1.2.2+pt27cu128
```

安装结果与锁定组合一致：

| 包 | 已安装版本 | wheel 标签 |
|---|---|---|
| pyg_lib | `0.5.0+pt27cu128` | CPython 3.11 / Windows AMD64 |
| torch_scatter | `2.1.2+pt27cu128` | CPython 3.11 / Windows AMD64 |
| torch_sparse | `0.6.18+pt27cu128` | CPython 3.11 / Windows AMD64 |
| torch_cluster | `1.6.3+pt27cu128` | CPython 3.11 / Windows AMD64 |
| torch_spline_conv | `1.2.2+pt27cu128` | CPython 3.11 / Windows AMD64 |

### 8.4 扩展导入验证

按要求执行 SciPy、torch_cluster、torch_scatter 和 torch_sparse 导入检查，输出：

```text
scipy 1.17.1
E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\torch_cluster\__init__.py
E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\torch_scatter\__init__.py
E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\torch_sparse\__init__.py
```

全部扩展联合导入及版本检查：

```text
pyg_lib 0.5.0+pt27cu128
torch_cluster 1.6.3+pt27cu128
torch_scatter 2.1.2+pt27cu128
torch_sparse 0.6.18+pt27cu128
torch_spline_conv 1.2.2+pt27cu128
```

再次检查 PyTorch 锁定版本：

```text
2.7.1+cu128
12.8
TORCH_LOCK_UNCHANGED
```

### 8.5 `torch_cluster` CUDA KNN 测试

执行逻辑：

```python
import torch
from torch_cluster import knn

x = torch.randn(32, 3, device="cuda")
y = torch.randn(8, 3, device="cuda")
edge = knn(x, y, k=4)
torch.cuda.synchronize()
```

完整结果：

```text
torch 2.7.1+cu128
runtime 12.8
gpu NVIDIA GeForce RTX 5060
capability (12, 0)
edge.shape torch.Size([2, 32])
edge.device cuda:0
edge.dtype torch.int64
edge range 0 30
TORCH_CLUSTER_CUDA_KNN_PASSED
```

验收结论：

- 扩展 DLL 能正常加载。
- CUDA KNN kernel 在 RTX 5060 / SM 12.0 上成功执行。
- 没有 `DLL load failed`。
- 没有 `no kernel image is available for execution`。
- 没有 `undefined symbol`。
- 没有 CUDA capability/ABI 不匹配错误。

### 8.6 修正后的阶段状态

原第四阶段现已由“失败”修正为：**成功**。失败原因和修正路径已经被完整保留，没有覆盖原始失败记录。

当前环境已经具备 PyTorch 2.7.1 cu128 和底层 PyG CUDA 扩展；但原计划第五阶段的 `torch_geometric`、Hydra、OmegaConf、scikit-learn、ONNX 等项目依赖，以及 NumPy 1.26.4 锁定尚未在本次续装中执行。checkpoint strict load 和项目完整 forward 也仍属于后续阶段。

## 9. 剩余依赖安装与联合导入结果

执行日期：2026-07-14  
本阶段最终状态：**依赖安装成功，但联合导入在 Hydra 1.2.0 / Python 3.11 兼容性错误处失败；已停止 checkpoint 与 forward。**

### 9.1 锁定环境复查

安装前使用绝对解释器复查：

```text
executable E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe
prefix E:\GRP-PTv2\.venv_ptv2
python 3.11.8 [MSC v.1937 64 bit (AMD64)]
torch 2.7.1+cu128
torch_cluster 1.6.3+pt27cu128
torch_scatter 2.1.2+pt27cu128
torch_sparse 0.6.18+pt27cu128
pyg_lib 0.5.0+pt27cu128
torch_spline_conv 1.2.2+pt27cu128
scipy 1.17.1
torch runtime 12.8
gpu NVIDIA GeForce RTX 5060
capability (12, 0)
LOCKED_ENVIRONMENT_RECHECK_PASSED
```

### 9.2 安装 `torch_geometric`

执行命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -m pip install torch_geometric
```

安装结果：

```text
Successfully installed aiohappyeyeballs-2.7.1 aiohttp-3.14.1
aiosignal-1.4.0 attrs-26.1.0 certifi-2026.6.17
charset_normalizer-3.4.9 colorama-0.4.6 frozenlist-1.8.0
idna-3.18 multidict-6.7.1 propcache-0.5.2 psutil-7.2.2
pyparsing-3.3.2 requests-2.34.2 torch_geometric-2.8.0
tqdm-4.68.4 urllib3-2.7.0 xxhash-3.8.1 yarl-1.24.2
```

安装后再次核对，PyTorch 及所有底层 PyG 扩展版本未变化，输出 `LOCKS_STILL_VALID`。

### 9.3 安装其余项目依赖

执行命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -m pip install tqdm hydra-core==1.2 omegaconf scikit-learn torchinfo torchsummary onnx onnxruntime
```

安装结果：

```text
Successfully installed PyYAML-6.0.3 antlr4-python3-runtime-4.9.3
flatbuffers-25.12.19 hydra-core-1.2.0 joblib-1.5.3
ml_dtypes-0.5.4 narwhals-2.24.0 omegaconf-2.3.1
onnx-1.22.0 onnxruntime-1.27.0 protobuf-7.35.1
scikit-learn-1.9.0 threadpoolctl-3.6.0
torchinfo-1.8.0 torchsummary-1.5.1
```

`antlr4-python3-runtime` 4.9.3 由 pip 从源码分发生成了纯 Python wheel；这不是 PyTorch/PyG CUDA 扩展的源码编译，也没有改变任何锁定的 PyTorch、CUDA 或 PyG 扩展版本。

NumPy 保持 `2.4.4`，没有执行降级。

### 9.4 联合导入成功项

使用绝对解释器依次联合导入，以下项目成功：

```text
numpy:           version=2.4.4
path=E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\numpy\__init__.py

torch:           version=2.7.1+cu128
path=E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\torch\__init__.py

torch_geometric: version=2.8.0
path=E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\torch_geometric\__init__.py

torch_cluster:   version=1.6.3+pt27cu128
path=E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\torch_cluster\__init__.py

torch_scatter:   version=2.1.2+pt27cu128
path=E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\torch_scatter\__init__.py

torch_sparse:    version=0.6.18+pt27cu128
path=E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\torch_sparse\__init__.py

scipy:           version=1.17.1
path=E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\scipy\__init__.py

sklearn:         version=1.9.0
path=E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\sklearn\__init__.py
```

这些结果表明 NumPy 2.4.4 与本阶段已执行的 torch_geometric、SciPy、scikit-learn 导入之间没有出现明确兼容性错误，因此没有触发 NumPy 调整条件。

### 9.5 Hydra 导入失败

失败包：`hydra-core==1.2.0`  
Python：3.11.8  
错误类型：`ValueError`  
错误位置：`hydra/conf/__init__.py` 中 `JobConf` 的 dataclass 定义。

完整 traceback：

```text
--- importing hydra ---
IMPORT_FAILED: hydra
Traceback (most recent call last):
  File "<stdin>", line 25, in <module>
  File "C:\Users\wlj\AppData\Local\Programs\Python\Python311\Lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 940, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\hydra\__init__.py", line 5, in <module>
    from hydra import utils
  File "E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\hydra\utils.py", line 8, in <module>
    import hydra._internal.instantiate._instantiate2
  File "E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\hydra\_internal\instantiate\_instantiate2.py", line 12, in <module>
    from hydra._internal.utils import _locate
  File "E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\hydra\_internal\utils.py", line 18, in <module>
    from hydra.core.utils import get_valid_filename, validate_config_path
  File "E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\hydra\core\utils.py", line 20, in <module>
    from hydra.core.hydra_config import HydraConfig
  File "E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\hydra\core\hydra_config.py", line 6, in <module>
    from hydra.conf import HydraConf
  File "E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\hydra\conf\__init__.py", line 46, in <module>
    class JobConf:
  File "E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\hydra\conf\__init__.py", line 75, in JobConf
    @dataclass
     ^^^^^^^^^
  File "C:\Users\wlj\AppData\Local\Programs\Python\Python311\Lib\dataclasses.py", line 1230, in dataclass
    return wrap(cls)
           ^^^^^^^^^
  File "C:\Users\wlj\AppData\Local\Programs\Python\Python311\Lib\dataclasses.py", line 1220, in wrap
    return _process_class(cls, init, repr, eq, order, unsafe_hash,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wlj\AppData\Local\Programs\Python\Python311\Lib\dataclasses.py", line 958, in _process_class
    cls_fields.append(_get_field(cls, name, type, kw_only))
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wlj\AppData\Local\Programs\Python\Python311\Lib\dataclasses.py", line 815, in _get_field
    raise ValueError(f'mutable default {type(f.default)} for field '
ValueError: mutable default <class 'hydra.conf.JobConf.JobConfig.OverrideDirname'> for field override_dirname is not allowed: use default_factory
```

测试程序在打印 traceback 后重新抛出异常，因此终端中同一完整 traceback 出现了两次；上方保留了一份完整调用链。

这是 Hydra 1.2.0 与 Python 3.11 dataclass 行为之间的明确兼容性错误，不是 NumPy 2.4.4 错误，也不是 PyTorch/CUDA/PyG 扩展错误。

### 9.6 停止结果

按照“任何一步失败都停止后续操作”的要求：

- 没有自行升级或降级 Hydra。
- 没有调整 NumPy。
- 没有继续执行 `omegaconf`、`onnx`、`onnxruntime` 的联合 import；这些包已安装，但本轮未完成导入验收。
- 没有加载 `GCN_res/best_model.pth`。
- 没有执行 `load_state_dict(strict=True)`。
- 没有执行固定 `[1,2048,4]` / `[1,2048,2048]` CUDA forward。
- 没有输出 `CHECKPOINT_AND_FORWARD_VALIDATION_PASSED`。

下一步需要用户明确授权如何解决 Hydra：例如允许将 `hydra-core` 从严格锁定的 1.2.0 调整到支持 Python 3.11 的版本，或允许项目绕开 Hydra 后单独执行 checkpoint 验证。本次没有自行选择方案。

## 10. Hydra 1.3.2 修复与最终环境验收

执行日期：2026-07-14  
授权变更：只允许将 `hydra-core` 从 1.2.0 升级到 1.3.2。  
最终状态：**环境、联合导入、Hydra compose、checkpoint strict load 和 RTX 5060 完整 forward 全部通过。**

### 10.1 Hydra 升级预检与安装

实际安装前先执行 pip dry-run。预检输出表明只会安装 `hydra-core-1.3.2`：

```text
Collecting hydra-core==1.3.2
Requirement already satisfied: omegaconf<2.4,>=2.2 (2.3.1)
Requirement already satisfied: antlr4-python3-runtime==4.9.* (4.9.3)
Requirement already satisfied: packaging (26.2)
Requirement already satisfied: PyYAML>=5.1.0 (6.0.3)
Would install hydra-core-1.3.2
```

预检没有计划修改 PyTorch、CUDA Runtime、torch_geometric、PyG 扩展、NumPy 或 SciPy，因此执行授权命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -m pip install --upgrade hydra-core==1.3.2
```

结果：

```text
Found existing installation: hydra-core 1.2.0
Successfully uninstalled hydra-core-1.2.0
Successfully installed hydra-core-1.3.2
```

### 10.2 Hydra/OmegaConf 与锁定版本复查

```text
hydra 1.3.2
hydra path E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\hydra\__init__.py
omegaconf 2.3.1
omegaconf path E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\omegaconf\__init__.py
```

全部锁定版本断言通过：

```text
torch 2.7.1+cu128
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
ALL_VERSION_LOCKS_PASSED
```

`python -m pip check`：

```text
No broken requirements found.
```

### 10.3 完整联合导入

所有要求的模块均成功导入：

| 模块 | 版本 | 模块路径 |
|---|---|---|
| numpy | 2.4.4 | `.venv_ptv2\Lib\site-packages\numpy\__init__.py` |
| torch | 2.7.1+cu128 | `.venv_ptv2\Lib\site-packages\torch\__init__.py` |
| torch_geometric | 2.8.0 | `.venv_ptv2\Lib\site-packages\torch_geometric\__init__.py` |
| torch_cluster | 1.6.3+pt27cu128 | `.venv_ptv2\Lib\site-packages\torch_cluster\__init__.py` |
| torch_scatter | 2.1.2+pt27cu128 | `.venv_ptv2\Lib\site-packages\torch_scatter\__init__.py` |
| torch_sparse | 0.6.18+pt27cu128 | `.venv_ptv2\Lib\site-packages\torch_sparse\__init__.py` |
| scipy | 1.17.1 | `.venv_ptv2\Lib\site-packages\scipy\__init__.py` |
| sklearn | 1.9.0 | `.venv_ptv2\Lib\site-packages\sklearn\__init__.py` |
| hydra | 1.3.2 | `.venv_ptv2\Lib\site-packages\hydra\__init__.py` |
| omegaconf | 2.3.1 | `.venv_ptv2\Lib\site-packages\omegaconf\__init__.py` |
| onnx | 1.22.0 | `.venv_ptv2\Lib\site-packages\onnx\__init__.py` |
| onnxruntime | 1.27.0 | `.venv_ptv2\Lib\site-packages\onnxruntime\__init__.py` |

终端标志：

```text
DEPENDENCY_JOINT_IMPORT_PASSED
No broken requirements found.
```

NumPy 继续保持 2.4.4，没有出现要求触发版本调整的兼容性错误。

### 10.4 Hydra compose 验证

使用 Hydra 1.3.2 的 compose API，并显式设置 `version_base="1.2"` 保持项目原 Hydra 1.2 语义。在不启动训练、不修改 YAML 的情况下，`config` 下 8 个顶层任务配置全部成功组合：

```text
COMPOSE_OK ONNXpartseg_v2_improved_predict: model=Nico_v2_GCN_ONNX
COMPOSE_OK cls: model=Menghao
COMPOSE_OK partseg: model=Hengshuang
COMPOSE_OK partseg-o: model=Hengshuang-o
COMPOSE_OK partsegSA: model=Hengshuang-o
COMPOSE_OK partseg_v2: model=Nico
COMPOSE_OK partseg_v2_improved: model=Nico_v2_GCN_Drop
COMPOSE_OK partseg_v2_improved_predict: model=Nico_v2_GCN
```

`config/partseg_v2_improved.yaml` 的解析结果：

```text
model.name Nico_v2_GCN_Drop
batch_size 4
epoch 200
num_point 2048
HYDRA_COMPOSE_VALIDATION_PASSED
```

Hydra 对这些顶层 YAML 发出 `Defaults list is missing _self_` 迁移警告。该警告没有导致 compose 失败；在 `version_base="1.2"` 下目标字段解析值与现有配置一致，未观察到实际配置值变化。本次没有为消除警告而修改任何 YAML。

### 10.5 Checkpoint strict load 与有限性

目标：

```text
model: models/testParameters/GCN_res/model.py::PTV2Segmentation
checkpoint: models/testParameters/GCN_res/best_model.pth
checkpoint bytes: 74340448
```

PyTorch 读取结果：

```text
checkpoint fields ['epoch', 'train_acc', 'test_acc', 'class_avg_iou',
                   'inctance_avg_iou', 'model_state_dict',
                   'optimizer_state_dict']
epoch 124
train_acc 0.9735565185546875
test_acc 0.96408203125
class_avg_iou 0.8890682997702295
inctance_avg_iou 0.8890682997702295
state entries 434
linear_1.weight (48, 4)
mlp.weight (2, 48)
strict load result <All keys matched successfully>
all checkpoint tensor count 1349
all checkpoint tensor numel 18483530
nonfinite tensor paths []
CHECKPOINT_STRICT_LOAD_AND_FINITE_CHECK_PASSED
```

检查遍历了 checkpoint 内的 model state、optimizer state 及嵌套容器中的全部 torch tensor，不只检查模型参数。所有 1349 个 tensor 均无 NaN/Inf。

### 10.6 RTX 5060 固定 shape 完整 forward

固定参数：

```text
B=1
N=2048
K=6
dtype=float32
device=cuda:0
```

输入输出：

```text
model.device cuda:0
points.shape (1, 2048, 4) dtype torch.float32 device cuda:0
adj.shape (1, 2048, 2048) dtype torch.float32 device cuda:0
points_xyz.shape (1, 2048, 3) dtype torch.float32 device cuda:0
logits.shape (1, 2048, 2) dtype torch.float32 device cuda:0
logits finite True
```

显存测量：

```text
baseline allocated bytes 50257408
baseline reserved bytes 81788928
peak allocated bytes 144028160
peak allocated MiB 137.35595703125
forward peak delta bytes 93770752
forward peak delta MiB 89.4267578125
peak reserved bytes 180355072
peak reserved MiB 172.0
```

耗时：

```text
forward CUDA event ms 619.2323608398438
forward wall ms 627.8989999991609
```

这是单次验证 forward 的时间，不是预热后多轮 benchmark 平均值。

最终终端标志：

```text
CHECKPOINT_AND_FORWARD_VALIDATION_PASSED
```

没有 DLL load、CUDA kernel image、capability、OOM 或非有限数值错误。

### 10.7 当前最终结论

环境已达到下一阶段短训练的基础运行条件：

- Python/PyTorch/CUDA/PyG 版本锁定保持不变。
- Hydra 1.3.2 可导入并能组合项目配置。
- NumPy 2.4.4 与当前依赖联合导入成功。
- `GCN_res` checkpoint strict load 成功。
- checkpoint 全部 tensor 有限。
- RTX 5060 上固定 2048 点完整 forward 成功。

本阶段没有启动正式训练，没有修改项目模型、训练代码或配置文件。
