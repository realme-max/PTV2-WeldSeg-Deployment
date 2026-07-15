# GCN_res ONNX Runtime CUDA parity

## 1. 结论

执行日期：2026-07-15（Asia/Shanghai）

阶段1成功完成，阶段2因 CUDA Runtime 主版本不匹配失败。已立即停止，阶段3 CUDA parity 未执行。

```text
ONNXRUNTIME_CUDA_EP_UNAVAILABLE
ONNXRUNTIME_CUDA_PARITY_NOT_RUN
```

没有修改 ONNX、deployment、原始模型、checkpoint、数据或容差，也没有进入 TensorRT。

## 2. 验证对象

- 环境：`E:\GRP-PTv2\.venv_ptv2`
- 环境备份：`docs/venv_before_onnxruntime_gpu.txt`
- ONNX：`artifacts/gcn_res_onnx/20260715_onnx_after_cdist_fp32_opset18/gcn_res_deploy_fp32_opset18.onnx`
- 输入：`artifacts/gcn_res_onnx/20260715_onnx_after_cdist_fp32_opset18/export_input.npz`
- PyTorch CUDA 参考：`artifacts/gcn_res_onnx/20260715_onnx_after_cdist_fp32_opset18/pytorch_deploy_reference.npz`

## 3. 阶段1：替换 ONNX Runtime

执行命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -m pip uninstall -y onnxruntime
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -m pip install onnxruntime-gpu==1.27.0
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe -m pip check
```

结果：

```text
Found existing installation: onnxruntime 1.27.0
Successfully uninstalled onnxruntime-1.27.0
Successfully installed onnxruntime-gpu-1.27.0
No broken requirements found.
```

安装使用 Windows CPython 3.11 wheel：`onnxruntime_gpu-1.27.0-cp311-cp311-win_amd64.whl`。

安装过程没有升级依赖，以下锁定项保持不变：

| 项目 | 结果 |
|---|---|
| PyTorch | `2.7.1+cu128` |
| PyTorch CUDA Runtime | `12.8` |
| cuDNN | `90701`（9.7.1） |
| NumPy | 已有 `2.4.4`，Requirement already satisfied |
| PyG / torch_cluster | 未执行任何修改命令 |

CPU 发行包 `onnxruntime` 已不存在，当前发行包为 `onnxruntime-gpu==1.27.0`。

## 4. 阶段2：CUDA Execution Provider 验证

PyTorch CUDA 侧正常：

```text
torch.cuda.is_available()=True
torch.version.cuda=12.8
torch.backends.cudnn.version()=90701
GPU=NVIDIA GeForce RTX 5060
```

执行 `ort.preload_dlls()` 时，ORT 输出：

```text
WARNING: The installed PyTorch 2.7.1+cu128 uses CUDA 12.x,
but onnxruntime-gpu is built with CUDA 13.x.
Please install PyTorch for CUDA 13.x to be compatible.
```

随后报告缺少：

```text
cublasLt64_13.dll
cublas64_13.dll
cufft64_12.dll
cudart64_13.dll
```

CUDA Provider DLL 加载失败：

```text
Error loading onnxruntime_providers_cuda.dll
which depends on cublasLt64_13.dll (Windows error 126)
```

虽然 `ort.get_available_providers()` 返回：

```text
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

但这只表示 GPU wheel 注册了 Provider。创建 Session 后的真实结果为：

```text
session.get_providers()=['CPUExecutionProvider']
```

保护检查因此抛出：

```text
RuntimeError: CUDAExecutionProvider is not first;
refusing CPU fallback: ['CPUExecutionProvider']
```

成功标志 `ONNXRUNTIME_CUDA_EP_AVAILABLE` 未输出。

## 5. 阶段3：CUDA parity

未执行。原因是 CUDA EP 没有成功激活，按停止条件禁止将 CPU fallback 当作 CUDA 结果继续。

因此以下指标本轮没有结果：

- max_abs_error
- mean_abs_error
- max_relative_error
- weld_seam probability max/mean error
- background probability max/mean error
- predicted_label_agreement
- outputs_finite
- logits allclose

之前的 CPU EP 基线不变：最大绝对误差 `7.562637e-04`、平均绝对误差 `9.532897e-05`、标签一致率 100%。该结果没有被覆盖或冒充为 CUDA EP。

## 6. 首个确定阻塞

```text
FIRST_BLOCKING_CONDITION=onnxruntime-gpu 1.27.0 requires CUDA 13.x DLLs,
but the locked PyTorch environment provides CUDA 12.8 DLLs
```

当前不能在保持 PyTorch `2.7.1+cu128` 不变的同时，直接使用此 `onnxruntime-gpu 1.27.0` Windows wheel 完成 CUDA EP 验证。

本轮未自行安装 CUDA 13、未更换 PyTorch、未改用其他 ORT 版本。下一步需要先明确选择兼容策略，再重新执行阶段2。

## 7. 产物

- `artifacts/gcn_res_onnxruntime_cuda/20260715_115138_ort_gpu_1_27_cuda_ep_failed/cuda_ep_validation.json`
- `artifacts/gcn_res_onnxruntime_cuda/20260715_115138_ort_gpu_1_27_cuda_ep_failed/run.log`
