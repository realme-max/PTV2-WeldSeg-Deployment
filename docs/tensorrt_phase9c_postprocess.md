# TensorRT Phase 9C：C++ 分割后处理与焊缝几何输出

## 1. 目标与状态

Phase 9C 在既有 Phase 9B C++ 点云输入和 Phase 9A TensorRT Runtime 之后增加了任务后处理：

```text
weld TXT
  -> C++ PointCloud Pipeline
  -> TensorRT logits [1,2048,2]
  -> argmax + confidence
  -> class 0 weld-seam extraction
  -> original sampled-coordinate recovery
  -> bbox / centroid / PCA length
  -> JSON / ASCII PLY / prediction TXT
```

最终状态：

```text
CPP_POSTPROCESS_PIPELINE_COMPLETED
```

本阶段没有修改或重新构建 checkpoint、ONNX、TensorRT Engine、VoxelUnique CUB Plugin 或 TensorRT Runtime 核心，也没有进入 Qt、PCL 可视化、Robot、FP16、INT8 或 CUDA kernel 优化。

## 2. 模块设计

新增 `deployment/postprocess/`：

- `SegmentationPostProcessor`
  - 强制 logits 合同为 4096 个有限 FP32 值，即 `[2048,2]`。
  - 标签语义固定为 `class 0 = weld_seam`、`class 1 = background`。
  - 按要求使用 `logit0 > logit1 ? 0 : 1`，相等时选择 background。
  - 用稳定的二分类 softmax 计算最大类别概率作为 confidence。
- `CoordinateRecovery`
  - 直接复制采样后的原始 `PointXYZL`，不增加归一化或反归一化。
  - 输出点顺序与模型输入的采样顺序一致。
- `WeldGeometryExtractor`
  - 仅对 label 0 点计算数量、比例、AABB 和中心点。
  - 使用 FP64 累加计算 3x3 population covariance。
  - 用对称 Jacobi 特征分解求主方向，投影范围作为 `length_mm`。
  - 无 weld 点时 fail closed。
- `ResultWriter`
  - 验证输出目录、统计值、weld 数量和推理耗时。
  - 生成 `weld_result.json`、`weld_points.ply` 和 `prediction.txt`。
  - 输出路径不可创建或不可写时失败，不生成 fallback 结果。

`deployment/weld_trt_app/` 通过 CMake 链接 `ptv2_postprocess`，同时继续直接复用 `ptv2_tensorrt_runtime`，没有复制 Runtime 实现。

## 3. 输出格式

### weld_result.json

```json
{
  "task_id": "weld_65",
  "total_points": 2048,
  "weld_points": 209,
  "weld_ratio": 0.10205078125,
  "center": [3.5052173, 0.26490718, 272.75027],
  "bbox": {
    "min": [-25.941799, -5.2818, 268.5667],
    "max": [30.82, 6.1363, 277.4646]
  },
  "length_mm": 57.1960526,
  "inference_ms": 35.0625
}
```

### weld_points.ply

ASCII PLY，只保存 weld-seam 点：

```text
x y z label confidence
```

PLY header 明确包含 `x/y/z/label/confidence` 属性；所有数据行 label 均为 0。

### prediction.txt

保持 Phase 9B 格式与采样顺序：

```text
x y z predicted_label
```

## 4. 编译方法

配置与构建命令：

```powershell
cmake -S E:\GRP-PTv2\deployment\weld_trt_app `
  -B E:\GRP-PTv2\artifacts\gcn_res_tensorrt\phase9b_weld_trt_build `
  -G "Visual Studio 17 2022" -A x64 -T v143 `
  -DTENSORRT_ROOT="D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106" `
  -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

cmake --build E:\GRP-PTv2\artifacts\gcn_res_tensorrt\phase9b_weld_trt_build `
  --config Release --parallel 4
```

结果：VS2022 x64 Release `PASS`，生成：

- `Release/weld_trt_demo.exe`
- `postprocess/Release/postprocess_failure_probe.exe`

构建结束时既有系统 vcpkg hook 仍输出非致命 `pwsh.exe is not recognized`，但所有目标成功生成且返回码为 0。

## 5. 运行方法

Phase 9C 目录输出方式：

```powershell
E:\GRP-PTv2\artifacts\gcn_res_tensorrt\phase9b_weld_trt_build\Release\weld_trt_demo.exe `
  --cloud E:\GRP-PTv2\data\weld\000001\weld_65.txt `
  --engine E:\GRP-PTv2\artifacts\gcn_res_tensorrt\20260717_173128_144483_phase8d_production_baseline\package\engine\strict_fp32_voxelunique_cub.plan `
  --plugin E:\GRP-PTv2\artifacts\gcn_res_tensorrt\20260717_173128_144483_phase8d_production_baseline\package\plugins\VoxelUniqueCubPlugin.dll `
  --output E:\GRP-PTv2\artifacts\gcn_res_tensorrt\phase9c_result
```

为兼容 Phase 9B，`--output ...\prediction.txt` 仍被接受；此时 JSON/PLY 写入该文件的父目录，prediction 保持用户给定路径。

## 6. Python/C++ 一致性验证

验证脚本：`scripts/validate_gcn_res_tensorrt_cpp_postprocess.py`。

正式成功产物：

```text
artifacts/gcn_res_tensorrt/20260717_221445_024382_phase9c_cpp_postprocess/
```

结果：

| 检查项 | 结果 |
|---|---:|
| logits | `[2048,2]`，全部有限 |
| Phase 9B vs Phase 9C labels | `2048/2048 = 100%` |
| Python argmax vs C++ labels | `2048/2048 = 100%` |
| Python/C++ weld point count | `209 / 209` |
| weld ratio error | `0.0` |
| center error | `0.0` |
| bbox min/max error | `0.0 / 0.0` |
| PCA length error | `0.0` |
| 最大 geometry error | `0.0 < 1e-5` |
| 坐标恢复最大误差 | `4.999999987e-7 < 1e-6` |
| PLY confidence 最大误差 | `4.969482603e-10` |
| PLY vertices | `209`，全部 label 0 |
| ErrorRecorder errors | `0` |

`WeldGeometryResult` 的字段按任务定义为 FP32，因此正式 parity 将独立 Python FP64 计算显式量化到 FP32 输出合同后比较。FP64 原值和量化误差同时保存在 `postprocess_parity.json` 中；没有调整 `1e-5` 阈值。中心点 Z 在 FP64 到 FP32 的合同量化误差为 `1.489374625e-5`，量化后 Python/C++ 输出完全一致。

Phase 9B 旧 CLI 形式也重新执行了完整回归，预处理 exact、标签一致率 100%、mIoU/F1 delta 0、6/6 旧异常路径 PASS。

## 7. Fail-closed 验证

以下 5 项通过独立 C++ probe 测试，均以退出码 1 结束并输出明确错误，未生成 fallback：

1. 空 logits；
2. logits 长度 4095；
3. NaN logits；
4. 没有 class 0 weld 点；
5. 输出路径为不可用目录（实际为普通文件）。

结果：`5/5 PASS`，`fallback = NONE`。

## 8. 当前限制

- 固定 B=1、N=2048、2 类 FP32 logits。
- 几何长度是第一版 PCA 主方向投影范围，不是沿焊缝曲线的弧长。
- 结果仅覆盖采样后的 2048 点，没有回写未采样点或原始百万点云。
- confidence 为模型 softmax 置信度，不是工业质量阈值或安全判据。
- 无焊缝点被视为错误并 fail closed，不输出“空结果”。
- 当前没有 Qt、PCL 可视化、Robot 接口、FP16、INT8 或 CUDA 后处理。
- Phase 8D 的严格数值例外继续有效：`CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED`。
