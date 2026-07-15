# VoxelUniquePrototype

Isolated TensorRT 11.1 `IPluginV3` experiment for a runtime output length.

The prototype uses:

- input `keys`: INT64 `[4]`;
- output `count`: INT32 scalar declared with `declareSizeTensor()`;
- output `values`: INT64 `[M]` where `M=count`;
- output `inverse`: INT64 `[4]`;
- `IOutputAllocator` for the data-dependent `values` output;
- a deliberately minimal CUDA kernel for the fixed test vector only.

It does not load or modify the GCN_res ONNX model and is not a production
Unique implementation.

