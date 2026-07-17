# TensorRT Phase 9B：C++ 点云预处理流水线

## 1. 目标与结果

Phase 9B 在 Phase 9A C++ TensorRT Runtime 前增加了独立的 CPU 点云预处理模块，实现以下闭环：

```text
weld TXT (x y z label)
  -> PointCloudLoader
  -> PointSampler (N=2048, seed=42)
  -> FeatureBuilder (normalized XYZ + constant 1)
  -> KnnGraphBuilder (k=6 dense adjacency)
  -> Phase 9A TensorRTInference
  -> prediction.txt
```

最终状态：

```text
CPP_POINTCLOUD_PIPELINE_COMPLETED
```

本阶段没有修改或重新构建 checkpoint、ONNX、TensorRT Engine、VoxelUnique Plugin、Python TensorRT Runtime，也没有进入 Qt、PCL 可视化、Robot、CUDA KNN、FP16 或 INT8。

## 2. 新增模块

- `deployment/pointcloud_pipeline/`
  - `PointCloudLoader`：读取并验证四列 TXT，拒绝文件缺失、列数错误、非法数值、NaN/Inf 和非 0/1 标签。
  - `PointSampler`：使用 `std::mt19937(seed=42)` 与 `std::shuffle` 无放回采样；输入少于 2048 点时失败。
  - `FeatureBuilder`：生成 FP32 `[1,2048,4]`。
  - `KnnGraphBuilder`：CPU 确定性构建 `k=6`、无 self-edge 的稠密 FP32 `[1,2048,2048]` 邻接矩阵。
- `deployment/weld_trt_app/`
  - `weld_trt_demo.exe` 串联点云预处理和现有 `deployment/tensorrt_runtime/`。
- `scripts/validate_gcn_res_tensorrt_cpp_pointcloud_pipeline.py`
  - 独立验证 C++/Python 预处理、TensorRT 输出、任务指标、重复运行确定性和 fail-closed 行为。

TensorRT Runtime 代码通过 CMake `add_subdirectory()` 直接复用，没有复制实现。

## 3. 数据格式与模型契约

输入 TXT 每个非空行必须恰好包含：

```text
x y z label
```

- `x/y/z`：有限浮点数。
- `label`：整数 `0` 或 `1`。
- `label 0 = weld_seam`，`label 1 = background`。

模型固定契约：

| Tensor | dtype | shape | 说明 |
|---|---|---|---|
| `points` | FP32 | `[1,2048,4]` | 归一化 XYZ + 常量 `1.0` |
| `adj` | FP32 | `[1,2048,2048]` | k=6 connectivity adjacency |
| `logits` | FP32 | `[1,2048,2]` | 每点两类 logits |

部署归一化严格复现当前 Python 基线：先用完整原始点云计算 XYZ 质心，所有点减质心，再除以最大点半径；随后按固定采样索引生成模型输入。`weld_65` 恰好有 2048 点，因此无放回采样只改变点顺序，不丢点。

KNN 以归一化前的 sampled XYZ 计算欧氏距离。全局平移和统一正比例缩放不改变邻居排序，因此与 Python 基线的归一化 XYZ KNN 语义一致；验证中邻接矩阵逐元素完全一致。

## 4. 编译方法

已使用 Visual Studio 2022 x64 Release、C++17、CUDA Toolkit 12.8 和 TensorRT 11.1.0.106 完成构建：

```powershell
cmake -S E:\GRP-PTv2\deployment\weld_trt_app `
  -B E:\GRP-PTv2\artifacts\gcn_res_tensorrt\phase9b_weld_trt_build `
  -G "Visual Studio 17 2022" -A x64 -T v143 `
  -DTENSORRT_ROOT="D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106" `
  -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

cmake --build E:\GRP-PTv2\artifacts\gcn_res_tensorrt\phase9b_weld_trt_build `
  --config Release --parallel 4
```

构建结果：`PASS`。可执行文件为：

```text
E:\GRP-PTv2\artifacts\gcn_res_tensorrt\phase9b_weld_trt_build\Release\weld_trt_demo.exe
```

构建结束时系统 vcpkg hook 输出了非致命的 `pwsh.exe is not recognized` 提示，但所有目标成功生成且 CMake 返回码为 0；该提示与 Phase 9A 相同，不影响本轮二进制和验证结果。

## 5. 运行方法

```powershell
E:\GRP-PTv2\artifacts\gcn_res_tensorrt\phase9b_weld_trt_build\Release\weld_trt_demo.exe `
  --cloud E:\GRP-PTv2\data\weld\000001\weld_65.txt `
  --engine E:\GRP-PTv2\artifacts\gcn_res_tensorrt\20260717_173128_144483_phase8d_production_baseline\package\engine\strict_fp32_voxelunique_cub.plan `
  --plugin E:\GRP-PTv2\artifacts\gcn_res_tensorrt\20260717_173128_144483_phase8d_production_baseline\package\plugins\VoxelUniqueCubPlugin.dll `
  --output E:\GRP-PTv2\artifacts\gcn_res_tensorrt\phase9b_smoke\prediction.txt `
  --report E:\GRP-PTv2\artifacts\gcn_res_tensorrt\phase9b_smoke\runtime_report.json
```

`prediction.txt` 共 2048 行，保持采样后的点顺序：

```text
x y z predicted_label
```

## 6. Python/C++ 一致性验证

正式验证产物：

```text
artifacts/gcn_res_tensorrt/20260717_215224_557179_phase9b_cpp_pointcloud_pipeline/
```

验证命令：

```powershell
E:\GRP-PTv2\.venv_ptv2\Scripts\python.exe `
  E:\GRP-PTv2\scripts\validate_gcn_res_tensorrt_cpp_pointcloud_pipeline.py `
  --exe E:\GRP-PTv2\artifacts\gcn_res_tensorrt\phase9b_weld_trt_build\Release\weld_trt_demo.exe
```

固定样本为 `weld_65`，测试结果：

| 检查项 | 结果 |
|---|---:|
| C++ points vs Python points | exact，max abs `0.0` |
| C++ adjacency vs sklearn adjacency | exact，mismatch `0` |
| logits shape | `[1,2048,2]` |
| logits finite | PASS |
| Python TRT vs C++ TRT max abs | `3.0994415283203125e-6` |
| mean abs | `3.5571292755776085e-7` |
| label agreement | `2048/2048 = 100%` |
| mIoU delta | `0.0` |
| weld seam F1 delta | `0.0` |
| 两次 C++ 采样索引 | exact |
| 两次 C++ predicted labels | exact |

任务指标在 Python TensorRT 和 C++ TensorRT 两侧完全一致：accuracy `0.982421875`、mIoU `0.9133471933471933`、weld seam F1 `0.9166666666666666`。

本轮单次冷启动记录为：load `1.6324 ms`、sample `0.0264 ms`、feature `0.0161 ms`、CPU KNN `17.7163 ms`、TensorRT CUDA inference `37.5332 ms`、inference wall `41.8911 ms`。这些是功能验证数据，不是正式 benchmark。

生产对象校验值：

- Engine SHA-256：`a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299`
- Plugin SHA-256：`6641ec147e8eac10206a5c60ba1c1390c398d4b59e32a6a618a37046360ec348`
- Runtime Plugin instances：`4`
- ErrorRecorder errors：`0`

## 7. Fail-closed 验证

以下 6 个场景全部返回非 0，且没有 fallback：

1. TXT 不存在；
2. 点数为 2047；
3. TXT 行格式错误；
4. 坐标为 NaN；
5. Engine 不存在；
6. Plugin 不存在。

结果：`6/6 PASS`，`fallback = NONE`。

## 8. 当前限制

- 固定 `B=1`、`N=2048`、`k=6`、FP32。
- 采样为无放回固定随机采样；输入不足 2048 点直接失败。
- KNN 使用 CPU `O(N^2)` 距离计算和稠密邻接矩阵，尚未做 CUDA 优化。
- 当前仅完成单样本功能和 Python/C++ 对齐验证，不替代 Phase 8D 正式性能与全测试集资格认证。
- 没有 Qt、PCL 可视化、Robot 接口、并发、多 batch、FP16 或 INT8。
- Phase 8D 的严格数值例外仍然有效：`CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED`；本阶段只证明固定样本 Python TensorRT 与 C++ TensorRT 的任务级一致性。
