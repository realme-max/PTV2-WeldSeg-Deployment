# VoxelUniquePlugin correctness

Phase 2C.1 correctness-only implementation.

- CPU reference: sort + unique + lower_bound inverse mapping.
- Plugin: deliberately serial, single-thread CUDA implementation.
- Independent final oracle: PyTorch CPU `torch.unique(sorted=True, return_inverse=True)`.
- Dynamic profile: `N=1..2048`.
- No GCN_res ONNX, FP16, benchmark, or production optimization.

