# GCN_res ONNX Runtime CUDA Execution Provider 验证

## 1. 结论

检查日期：2026-07-15（Asia/Shanghai）

- 当前环境：`E:\GRP-PTv2\.venv_ptv2`
- 已安装：`onnxruntime==1.27.0`（CPU 包）
- 未安装：`onnxruntime-gpu`
- `ort.get_available_providers()`：`['AzureExecutionProvider', 'CPUExecutionProvider']`
- `CUDAExecutionProvider` 当前不可用，因此本轮没有执行 ORT CUDA parity。
- 没有安装、卸载或升级任何包，没有修改 ONNX、deployment、checkpoint 或数据。
- 没有进入 TensorRT。

状态：

```text
ONNXRUNTIME_CUDA_EP_NOT_AVAILABLE
ONNXRUNTIME_CUDA_PARITY_NOT_RUN
```

## 2. 验证对象

- ONNX：`artifacts/gcn_res_onnx/20260715_onnx_after_cdist_fp32_opset18/gcn_res_deploy_fp32_opset18.onnx`
- 固定输入：`artifacts/gcn_res_onnx/20260715_onnx_after_cdist_fp32_opset18/export_input.npz`
- PyTorch CUDA 参考：`artifacts/gcn_res_onnx/20260715_onnx_after_cdist_fp32_opset18/pytorch_deploy_reference.npz`
- 输入：`points float32 [1,2048,4]`、`adj float32 [1,2048,2048]`
- 输出：`logits float32 [1,2048,2]`

## 3. 当前 ORT 安装与 Provider 证据

执行命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -m pip show onnxruntime
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -m pip show onnxruntime-gpu
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -c "import onnxruntime as ort; print(ort.__version__); print(ort.__file__); print(ort.get_available_providers()); print(ort.get_all_providers())"
```

关键输出：

```text
onnxruntime==1.27.0
onnxruntime path=E:\GRP-PTv2\.venv_ptv2\Lib\site-packages\onnxruntime\__init__.py
WARNING: Package(s) not found: onnxruntime-gpu
available_providers=['AzureExecutionProvider', 'CPUExecutionProvider']
```

`get_all_providers()` 中出现 `CUDAExecutionProvider` 只表示 ORT API 知道该 Provider 名称，不代表当前安装包包含并能加载 CUDA EP。是否可用必须以 `get_available_providers()` 为准。

## 4. PyTorch CUDA 运行时

```text
Python environment: E:\GRP-PTv2\.venv_ptv2
PyTorch: 2.7.1+cu128
PyTorch CUDA Runtime: 12.8
cuDNN: 90701 (9.7.1)
CUDA available: True
GPU: NVIDIA GeForce RTX 5060
Capability: (12, 0)
```

这证明 PyTorch CUDA 参考侧可运行，但不能证明 CPU 版 ONNX Runtime 具备 CUDA EP。

## 5. CPU Execution Provider 基线（保留）

Provider：`CPUExecutionProvider`

| 指标 | 结果 |
|---|---:|
| max_abs_error | 7.562637329102e-04 |
| mean_abs_error | 9.532897092868e-05 |
| max_relative_error | 3.267243970186e-03 |
| weld_seam_probability_max_abs_error | 6.407499313354e-05 |
| weld_seam_probability_mean_abs_error | 2.487789515726e-06 |
| background_probability_max_abs_error | 6.416440010071e-05 |
| background_probability_mean_abs_error | 2.489928192517e-06 |
| predicted_label_agreement | 1.000000 (100%) |
| outputs_finite | True |
| logits allclose (`rtol=1e-4, atol=1e-5`) | False |

CPU EP 能生成有限输出且标签完全一致，但 logits 没有通过既定 allclose 阈值。

## 6. CUDA Execution Provider 结果

计划使用的 Provider 顺序：

```python
providers = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]
```

本轮结果：未执行。原因是 `onnxruntime-gpu` 未安装，`CUDAExecutionProvider` 不在 `ort.get_available_providers()` 中。没有把 CPU fallback 的结果伪装成 CUDA 结果。

待 GPU EP 可用后必须记录：

- 实际激活的 `session.get_providers()`，且第一项必须为 `CUDAExecutionProvider`
- max_abs_error
- mean_abs_error
- max_relative_error
- weld seam probability max/mean error
- background probability max/mean error
- predicted label agreement
- 输出有限性与 logits allclose

## 7. 安装方案（仅记录，未执行）

PyPI 已提供 `onnxruntime-gpu==1.27.0` 的 `cp311-win_amd64` wheel。官方 CUDA EP 文档说明 CUDA 12.x 构建可在 CUDA 12.x 环境使用，cuDNN 主版本必须一致；PyTorch 2.4 及以后 CUDA 12.x 版本使用 cuDNN 9.x。当前 PyTorch 2.7.1+cu128 / cuDNN 9.7.1 与该方向一致。

建议在获得环境修改授权后，先保存环境快照，再将 CPU ORT 包替换为同版本 GPU 包。不要让 `onnxruntime` 与 `onnxruntime-gpu` 两套发行包在同一环境长期并存，因为它们提供相同的 `onnxruntime` Python 模块。

```powershell
$python = 'E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe'

# 仅在后续获得明确授权后执行
& $python -m pip freeze > E:\GRP-PTv2\docs\venv_ptv2_before_ort_gpu.txt
& $python -m pip uninstall -y onnxruntime
& $python -m pip install onnxruntime-gpu==1.27.0
& $python -m pip check
```

随后先导入 PyTorch，使其 CUDA/cuDNN DLL 可供 ORT 使用，再检查 Provider：

```powershell
& $python -c "import torch; import onnxruntime as ort; ort.preload_dlls(); print(ort.__version__); print(ort.get_available_providers()); ort.print_debug_info()"
```

官方资料：

- ONNX Runtime CUDA EP：<https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>
- ONNX Runtime 安装说明：<https://onnxruntime.ai/docs/install/>
- `onnxruntime-gpu` PyPI wheel：<https://pypi.org/project/onnxruntime-gpu/>

## 8. CUDA parity 复查命令

GPU 包安装并确认 `CUDAExecutionProvider` 可用后，使用原始固定文件，不重新导出 ONNX：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe E:\GRP-PTv2\scripts\validate_gcn_res_onnxruntime.py `
  --run-dir E:\GRP-PTv2\artifacts\gcn_res_onnx\20260715_onnx_after_cdist_fp32_opset18
```

验证脚本必须确认：

```python
session = ort.InferenceSession(
    str(model_path),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
assert session.get_providers()[0] == "CUDAExecutionProvider"
```

如果 CUDA EP 创建失败或回退到 CPU，应保留完整错误并停止，不能把 fallback 结果记作 CUDA parity。
