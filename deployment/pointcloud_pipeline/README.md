# C++ weld point-cloud preprocessing

This Phase 9B library converts a weld TXT (`x y z label`) into the fixed GCN_res TensorRT input contract.

1. `PointCloudLoader` validates exactly four finite numeric columns and labels in `{0,1}` while recording count, bounds, and label counts.
2. `PointSampler` uses `std::mt19937(seed=42)` and sampling without replacement. Inputs with fewer than 2048 points fail.
3. `FeatureBuilder` reproduces the deployed preprocessing semantics: subtract the centroid of the complete cloud, divide by its maximum point radius, and append a constant `1.0` category channel.
4. `KnnGraphBuilder` deterministically creates a dense FP32 `[2048,2048]` connectivity graph with `k=6`, no self edges, and row-to-neighbor orientation.

The first implementation is CPU-only. It does not use PCL, CUDA KNN, visualization, or model-specific runtime code.
