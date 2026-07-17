# GCN_res TensorRT C++ Runtime

This directory contains the Phase 9A Windows C++17 inference backend for the promoted GCN_res TensorRT production package. It deserializes the existing Engine; it does not build or modify an Engine.

## Fixed contract

| Tensor | Direction | Type | Shape | Binary elements |
|---|---|---|---|---:|
| `points` | input | FP32 | `[1,2048,4]` | 8192 |
| `adj` | input | FP32 | `[1,2048,2048]` | 4194304 |
| `logits` | output | FP32 | `[1,2048,2]` | 4096 |

All `.bin` files are raw contiguous little-endian `float32` arrays with no header and no NumPy dependency. File sizes are checked exactly before inference.

## Build with Visual Studio 2022

```powershell
cmake -S E:\GRP-PTv2\deployment\tensorrt_runtime `
  -B E:\GRP-PTv2\artifacts\gcn_res_tensorrt\phase9a_build `
  -G "Visual Studio 17 2022" -A x64 `
  -DTENSORRT_ROOT=D:\NVIDIA_GeForce5060\TensorRT-11.1.0\TensorRT-11.1.0.106 `
  -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

cmake --build E:\GRP-PTv2\artifacts\gcn_res_tensorrt\phase9a_build --config Release
```

The build directory is intentionally under ignored `artifacts/`. TensorRT and CUDA runtime DLLs are copied beside `trt_runtime_demo.exe`; the production `VoxelUniqueCubPlugin.dll` remains an explicit command-line dependency.

## Run

```powershell
trt_runtime_demo.exe `
  --engine package\engine\strict_fp32_voxelunique_cub.plan `
  --plugin package\plugins\VoxelUniqueCubPlugin.dll `
  --engine-sha256 a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299 `
  --points weld_65_points.bin `
  --adj weld_65_adj.bin `
  --output cpp_logits.bin `
  --runtime-json runtime_summary.json `
  --benchmark-json cpp_runtime_benchmark.json `
  --warmup 100 `
  --iterations 100
```

Initialization is fail-closed. Missing files, a mismatched Engine hash, an absent `VoxelUniqueCub/1/com.tensorrt.ptv2.experimental` Creator, a runtime plugin instance count other than four, an I/O contract mismatch, invalid binary sizes, CUDA errors, TensorRT ErrorRecorder entries, or `enqueueV3` failure terminate the process. There is no PyTorch fallback.
