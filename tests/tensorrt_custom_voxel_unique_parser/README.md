# TensorRT Custom VoxelUnique Parser Prototype

Independent Phase 2B validation for:

```text
custom ONNX node
  -> TensorRT ONNX Parser
  -> IPluginCreatorV3One
  -> IPluginV3 layer
  -> serialized engine build
```

The test uses domain and namespace `com.tensorrt.ptv2`, plugin name
`VoxelUnique`, and version `1`. It never opens or changes the GCN_res ONNX.

The Plugin runtime method is intentionally a no-op because inference is out of
scope for this phase. Only parser integration and engine serialization are
validated.

