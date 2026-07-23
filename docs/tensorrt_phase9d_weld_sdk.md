# TensorRT Phase 9D：WeldDetector C++ SDK

## 1. 目标与结果

Phase 9D 将现有部署链封装为单一 C++17 SDK 门面：

```text
Application
  -> WeldDetector SDK
      -> PointCloudLoader / PointSampler / FeatureBuilder / KnnGraphBuilder
      -> TensorRTInference / VoxelUniqueCub Plugin
      -> SegmentationPostProcessor / CoordinateRecovery / WeldGeometryExtractor
      -> WeldResult
```

调用方只需 include `WeldDetector.h`、`WeldConfig.h`、`WeldResult.h` 和 `WeldStatus.h`。TensorRT 对象、CUDA buffer、logits、预处理对象和后处理临时张量均由 Pimpl 隐藏。

```text
WELD_DETECTOR_SDK_COMPLETED
```

本阶段没有修改或重新构建 Engine、ONNX、checkpoint、VoxelUnique CUB Plugin、TensorRT Runtime 核心，也没有进入 FP16、INT8、CUDA kernel 优化或 Qt。

## 2. SDK 目录与依赖

新增 `deployment/weld_sdk/`，构建静态库：

```text
ptv2_weld_sdk.lib
```

CMake 依赖关系：

```text
ptv2_weld_sdk
  -> ptv2_pointcloud_pipeline
  -> ptv2_tensorrt_runtime
  -> ptv2_postprocess
```

SDK 没有复制任何底层实现。`weld_trt_app/main.cpp` 已改为只链接 `ptv2_weld_sdk`，源码不再 include Loader、TensorRT、CUDA 或 Postprocess 头文件。

## 3. API

### WeldConfig

```cpp
struct WeldConfig {
    std::string engine_path;
    std::string plugin_path;
    int input_points = 2048;
    int num_classes = 2;
    bool enable_geometry = true;
    std::string output_path; // optional Phase 9C compatibility output
};
```

`input_points` 和 `num_classes` 必须分别为 2048 和 2。`output_path` 为空时 SDK 只返回 `WeldResult`；非空时 SDK 内部复用 `ResultWriter` 生成 JSON、PLY 和 prediction TXT。以 `.txt` 结尾时保持 Phase 9B/9C prediction 路径兼容，否则解释为结果目录。

生产 Engine SHA-256 由 SDK 内部固定校验，不把完整性验证责任交给调用方。

### WeldResult

公开结果包括：

- `success`、`task_id`
- `total_points`、`weld_points`、`weld_ratio`
- `center[3]`、`bbox_min[3]`、`bbox_max[3]`
- `length_mm`、`inference_ms`
- `labels`

不公开 logits、TensorRT buffer 或 CUDA pointer。

### WeldStatus

| 状态 | 含义 |
|---|---|
| `SUCCESS` | 初始化或检测完成 |
| `INVALID_CONFIG` | 配置非法或未初始化 |
| `ENGINE_LOAD_FAILED` | Engine 文件、哈希或反序列化失败 |
| `PLUGIN_LOAD_FAILED` | Plugin 文件或加载验证失败 |
| `POINTCLOUD_LOAD_FAILED` | TXT 缺失或内容无法加载 |
| `PREPROCESS_FAILED` | 采样、特征或 KNN 失败 |
| `INFERENCE_FAILED` | TensorRT enqueue/runtime 失败 |
| `POSTPROCESS_FAILED` | 分类、坐标、几何或输出失败 |

SDK 以返回状态作为主要错误通道，并通过 `lastError()` 提供详细错误。内部不可避免的标准库异常会被边界捕获并转换为状态，不向调用方传播。

## 4. 调用示例

```cpp
#include "WeldConfig.h"
#include "WeldDetector.h"
#include "WeldResult.h"
#include "WeldStatus.h"

ptv2::weld::WeldConfig config;
config.engine_path = "package/engine/strict_fp32_voxelunique_cub.plan";
config.plugin_path = "package/plugins/VoxelUniqueCubPlugin.dll";

ptv2::weld::WeldDetector detector;
auto status = detector.initialize(config);
if (status != ptv2::weld::WeldStatus::SUCCESS) {
    // detector.lastError()
    return;
}

ptv2::weld::WeldResult result;
status = detector.detect("weld_65.txt", result);
if (status == ptv2::weld::WeldStatus::SUCCESS) {
    // result.labels, result.weld_points, result.length_mm, ...
}
```

一个 `WeldDetector` 实例拥有一个 TensorRT execution context，当前不是线程安全对象。多线程调用应每线程创建独立实例，或由调用方串行化同一实例。

## 5. 编译

```powershell
cmake -S E:\GRP-PTv2\deployment\weld_trt_app `
  -B E:\GRP-PTv2\artifacts\gcn_res_tensorrt\phase9d_weld_sdk_build `
  -G "Visual Studio 17 2022" -A x64 -T v143 `
  -DTENSORRT_ROOT="D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106" `
  -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

cmake --build E:\GRP-PTv2\artifacts\gcn_res_tensorrt\phase9d_weld_sdk_build `
  --config Release --parallel 4
```

VS2022 x64 Release 构建 `PASS`，生成：

- `weld_sdk/Release/ptv2_weld_sdk.lib`
- `weld_sdk/Release/sdk_smoke_test.exe`
- `Release/weld_trt_demo.exe`

已有 vcpkg hook 的非致命 `pwsh.exe is not recognized` 提示仍会出现，但所有目标成功生成，CMake 返回码为 0。

## 6. SDK smoke test

```powershell
sdk_smoke_test.exe `
  --engine package\engine\strict_fp32_voxelunique_cub.plan `
  --plugin package\plugins\VoxelUniqueCubPlugin.dll `
  --cloud data\weld\000001\weld_65.txt
```

结果：

| 字段 | 值 |
|---|---:|
| status | `SUCCESS` |
| success | `true` |
| task_id | `weld_65` |
| total_points | `2048` |
| labels | `2048` |
| weld_points | `209` |
| weld_ratio | `0.1020507812` |
| PCA length | `57.1960526 mm` |

标志：`SDK_SMOKE_TEST_PASSED`。

## 7. Phase 9C 兼容验证

验证脚本：`scripts/validate_gcn_res_tensorrt_weld_sdk.py`。

正式产物：

```text
artifacts/gcn_res_tensorrt/20260720_161056_633334_phase9d_weld_sdk/
```

| 检查项 | 结果 |
|---|---:|
| Phase 9C/9D matching labels | `2048/2048` |
| label agreement | `100%` |
| weld point count | `209 / 209` |
| weld ratio error | `2.500000068e-10` |
| center max error | `3.417968628e-7` |
| bbox max error | `4.257812520e-7` |
| PCA length error | `4.873047033e-8` |
| overall geometry max error | `4.257812520e-7 < 1e-5` |
| SDK-only app labels vs SDK smoke | exact |
| SDK-only app geometry max error | `4.257812520e-7` |

SDK 的可选输出同时生成 `weld_result.json`、`weld_points.ply` 和 `prediction.txt`，其中 prediction labels 与 `WeldResult.labels` 完全一致。

## 8. Fail-closed 验证

四项测试均非零退出，返回精确状态，无崩溃和 fallback：

| 场景 | 状态 |
|---|---|
| Engine 不存在 | `ENGINE_LOAD_FAILED` |
| Plugin 不存在 | `PLUGIN_LOAD_FAILED` |
| cloud 不存在 | `POINTCLOUD_LOAD_FAILED` |
| 点数 2047 | `PREPROCESS_FAILED` |

结果：`4/4 PASS`，`fallback = NONE`。

## 9. 当前边界

- SDK 固定适配当前生产 B=1、N=2048、2 类、FP32 Engine。
- 固定采样 seed 42、CPU k=6 KNN，与 Phase 9B/9C 保持一致。
- SDK 当前不提供并发 execution context 池。
- 没有 JSON 配置解析；`WeldConfig` 为未来扩展点。
- 没有暴露 logits 或内部 GPU 资源。
- Phase 8D 严格数值例外仍然有效：`CANDIDATE_STRICT_NUMERICAL_THRESHOLD_FAILED`。
