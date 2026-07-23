# WeldDetector C++ SDK

`ptv2_weld_sdk` is the public C++17 façade for the production weld-segmentation deployment chain. Applications include only `WeldDetector.h`, `WeldConfig.h`, `WeldResult.h`, and `WeldStatus.h`; TensorRT, CUDA buffers, logits, sampling objects, and post-processing internals remain private.

The SDK is fail closed and returns a typed `WeldStatus`. A detector instance owns one TensorRT execution context and is intentionally not thread-safe; use one initialized instance per calling thread or serialize calls externally.

`WeldConfig::output_path` is optional. It preserves Phase 9C JSON/PLY/prediction output without exposing internal tensors. Leave it empty for result-only SDK usage.
