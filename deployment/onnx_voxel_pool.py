"""Standard-PyTorch implementation of the historical GCN_res voxel pooling.

This module intentionally contains no torch_cluster/torch_scatter dependency and
does not move tensors to CPU.  The implementation mirrors torch_cluster 1.6.3
``grid`` semantics used by the historical model, including per-batch bounds,
mixed-radix voxel keys, sorted voxel order, max feature pooling, mean XYZ, and
the historical minimum-voxel-count crop across a batch.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class VoxelPoolResult(NamedTuple):
    pooled_points: torch.Tensor
    pooled_features: torch.Tensor
    retained_voxel_coordinates: torch.Tensor
    retained_voxel_counts: torch.Tensor
    point_to_voxel: torch.Tensor
    voxel_count_per_batch: torch.Tensor
    start: torch.Tensor
    end: torch.Tensor
    extents: torch.Tensor
    unique_local_keys: torch.Tensor
    unique_batch_ids: torch.Tensor
    retained_mask: torch.Tensor


def _validate_shapes(
    points: torch.Tensor, points_features: torch.Tensor, voxel_size: torch.Tensor
) -> None:
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape [B,N,3], got {tuple(points.shape)}")
    if points_features.ndim != 3 or points_features.shape[:2] != points.shape[:2]:
        raise ValueError(
            "points_features must have shape [B,N,F] with the same B,N as points, "
            f"got {tuple(points_features.shape)}"
        )
    if voxel_size.numel() != 3:
        raise ValueError(f"voxel_size must contain three values, got {tuple(voxel_size.shape)}")


def standard_voxel_pool_with_metadata(
    points: torch.Tensor,
    points_features: torch.Tensor,
    voxel_size: torch.Tensor,
) -> VoxelPoolResult:
    """Pool points with the exact ordering and aggregation rules of the source.

    ``point_to_voxel`` contains per-batch local voxel indices before the
    historical cross-batch crop.  A value greater than or equal to the returned
    pooled point count denotes a voxel discarded by that crop.
    """

    _validate_shapes(points, points_features, voxel_size)
    batch_size, points_per_batch, _ = points.shape
    voxel_size = voxel_size.to(device=points.device, dtype=points.dtype).reshape(1, 1, 3)

    # torch_cluster::grid uses per-call min/max when start/end are omitted.  The
    # historical implementation calls it once per batch item.
    start = torch.amin(points, dim=1)
    end = torch.amax(points, dim=1)
    shifted = points - start.unsqueeze(1)

    # C++/CUDA static_cast<int64_t> truncates toward zero.  Here start is the
    # per-axis minimum and voxel sizes are positive, so both shifted and the
    # extent numerator are non-negative.  On this restricted deployment domain
    # floor is exactly equivalent to trunc and has a standard ONNX lowering.
    voxel_coordinates = torch.floor(shifted / voxel_size).to(torch.int64)
    extents = torch.floor((end - start) / voxel_size.reshape(1, 3)).to(torch.int64) + 1

    # x is the fastest-changing dimension in torch_cluster 1.6.3:
    # key = x + extent_x*y + extent_x*extent_y*z.
    strides = torch.stack(
        (
            torch.ones_like(extents[:, 0]),
            extents[:, 0],
            extents[:, 0] * extents[:, 1],
        ),
        dim=1,
    )
    local_keys = torch.sum(voxel_coordinates * strides.unsqueeze(1), dim=2)

    # Make keys collision-free across batches without changing local ordering.
    capacities = torch.prod(extents, dim=1)
    offsets = torch.cumsum(capacities, dim=0) - capacities
    global_keys = local_keys + offsets.unsqueeze(1)

    flat_global_keys = global_keys.reshape(-1)
    unique_global_keys, inverse_global = torch.unique(
        flat_global_keys, sorted=True, return_inverse=True
    )
    total_voxels = unique_global_keys.shape[0]

    point_batch_ids = (
        torch.arange(batch_size, device=points.device, dtype=torch.int64)
        .reshape(batch_size, 1)
        .expand(batch_size, points_per_batch)
        .reshape(-1)
    )
    unique_batch_ids = torch.zeros(
        total_voxels, device=points.device, dtype=torch.int64
    ).scatter_reduce(
        0, inverse_global, point_batch_ids, reduce="amin", include_self=False
    )
    unique_local_keys = unique_global_keys - offsets.index_select(0, unique_batch_ids)

    count_int = torch.zeros(
        total_voxels, device=points.device, dtype=torch.int64
    ).scatter_add(0, inverse_global, torch.ones_like(inverse_global))
    count_float = count_int.to(points.dtype)

    flat_points = points.reshape(-1, 3)
    point_index = inverse_global.unsqueeze(1).expand(-1, 3)
    summed_points = torch.zeros(
        (total_voxels, 3), device=points.device, dtype=points.dtype
    ).scatter_add(0, point_index, flat_points)
    mean_points = summed_points / count_float.unsqueeze(1)

    feature_dim = points_features.shape[2]
    flat_features = points_features.reshape(-1, feature_dim)
    feature_index = inverse_global.unsqueeze(1).expand(-1, feature_dim)
    pooled_features_all = torch.zeros(
        (total_voxels, feature_dim),
        device=points_features.device,
        dtype=points_features.dtype,
    ).scatter_reduce(
        0,
        feature_index,
        flat_features,
        reduce="amax",
        include_self=False,
    )

    voxel_count_per_batch = torch.zeros(
        batch_size, device=points.device, dtype=torch.int64
    ).scatter_add(
        0, unique_batch_ids, torch.ones_like(unique_batch_ids)
    )
    batch_prefix = torch.cumsum(voxel_count_per_batch, dim=0) - voxel_count_per_batch
    local_unique_indices = (
        torch.arange(total_voxels, device=points.device, dtype=torch.int64)
        - batch_prefix.index_select(0, unique_batch_ids)
    )
    point_to_voxel = (
        inverse_global - batch_prefix.index_select(0, point_batch_ids)
    ).reshape(batch_size, points_per_batch)

    min_voxel_count = torch.amin(voxel_count_per_batch)
    retained_mask = local_unique_indices < min_voxel_count
    retained_points_flat = mean_points[retained_mask]
    retained_features_flat = pooled_features_all[retained_mask]
    retained_counts_flat = count_int[retained_mask]
    retained_keys_flat = unique_local_keys[retained_mask]
    retained_batch_ids = unique_batch_ids[retained_mask]

    retained_extents = extents.index_select(0, retained_batch_ids)
    retained_x = torch.remainder(retained_keys_flat, retained_extents[:, 0])
    retained_y = torch.remainder(
        torch.div(retained_keys_flat, retained_extents[:, 0], rounding_mode="floor"),
        retained_extents[:, 1],
    )
    retained_z = torch.div(
        retained_keys_flat,
        retained_extents[:, 0] * retained_extents[:, 1],
        rounding_mode="floor",
    )
    retained_coordinates_flat = torch.stack(
        (retained_x, retained_y, retained_z), dim=1
    )

    pooled_points = retained_points_flat.reshape(batch_size, min_voxel_count, 3)
    pooled_features = retained_features_flat.reshape(
        batch_size, min_voxel_count, feature_dim
    )
    retained_voxel_coordinates = retained_coordinates_flat.reshape(
        batch_size, min_voxel_count, 3
    )
    retained_voxel_counts = retained_counts_flat.reshape(batch_size, min_voxel_count)

    return VoxelPoolResult(
        pooled_points=pooled_points,
        pooled_features=pooled_features,
        retained_voxel_coordinates=retained_voxel_coordinates,
        retained_voxel_counts=retained_voxel_counts,
        point_to_voxel=point_to_voxel,
        voxel_count_per_batch=voxel_count_per_batch,
        start=start,
        end=end,
        extents=extents,
        unique_local_keys=unique_local_keys,
        unique_batch_ids=unique_batch_ids,
        retained_mask=retained_mask,
    )


def standard_voxel_pool(
    points: torch.Tensor,
    points_features: torch.Tensor,
    voxel_size: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ONNX-facing two-output form used by the deployment model."""

    result = standard_voxel_pool_with_metadata(points, points_features, voxel_size)
    return result.pooled_points, result.pooled_features
