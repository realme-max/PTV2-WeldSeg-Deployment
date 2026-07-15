# GCN_res VoxelUnique parser integration

Phase 3 parser-only test for the derived GCN_res ONNX graph. The executable
registers the correctness-validated `VoxelUnique` IPluginV3 Creator and invokes
the TensorRT ONNX Parser. It intentionally never creates a builder config,
builds an engine, or runs inference.
