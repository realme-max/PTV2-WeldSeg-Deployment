# Runtime input example

Phase 9A validation material is generated under ignored `artifacts/`; no dataset or binary tensor is committed here.

For `weld_65`, prepare these raw contiguous little-endian FP32 files:

- `weld_65_points.bin`: 8192 floats / 32768 bytes, shape `[1,2048,4]`.
- `weld_65_adj.bin`: 4194304 floats / 16777216 bytes, shape `[1,2048,2048]`.
- `cpp_logits.bin`: 4096 floats / 16384 bytes, shape `[1,2048,2]`.

Run `scripts/validate_gcn_res_tensorrt_cpp_runtime.py` to derive the two inputs from the frozen Phase 8D sample, execute the existing Python production runner, run the C++ backend, and write `runtime_compare.json`.
