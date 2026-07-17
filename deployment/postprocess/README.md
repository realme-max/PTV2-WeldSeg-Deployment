# C++ segmentation post-processing

This Phase 9C library converts fixed `[2048,2]` TensorRT logits into task results while preserving the deployed label semantics:

- class `0`: `weld_seam`
- class `1`: `background`

The library performs finite/shape validation, stable two-class softmax confidence, original sampled-coordinate recovery, weld-point extraction, bounding box and centroid statistics, PCA length estimation, and JSON/ASCII-PLY/prediction-TXT output. Invalid inputs fail closed; no fallback output is produced.
