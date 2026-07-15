"""GCN_res deployment model using only standard PyTorch tensor operations.

The module tree and parameter names intentionally mirror the historical
``models.testParameters.GCN_res.model.PTV2Segmentation`` model so its checkpoint
can be loaded directly with ``strict=True`` and no key remapping.
"""

from __future__ import annotations

import torch
from torch import nn

from deployment.onnx_voxel_pool import standard_voxel_pool


class PositionalEncoding(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(3, out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim)
        self.batch_norm = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, p_i: torch.Tensor, p_j: torch.Tensor) -> torch.Tensor:
        out = self.linear_1(p_i - p_j)
        out = self.batch_norm(out.permute(0, 3, 1, 2))
        out = self.relu(out.permute(0, 2, 3, 1))
        return self.linear_2(out)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    raw_size = idx.size()
    flat_idx = idx.reshape(raw_size[0], -1)
    selected = torch.gather(
        points, 1, flat_idx[..., None].expand(-1, -1, points.size(-1))
    )
    return selected.reshape(*raw_size, -1)


class TransitionDownBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid_size: list[float]) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.grid_size = grid_size
        self.register_buffer(
            "_voxel_size",
            torch.tensor(grid_size, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self, points_xyz: torch.Tensor, points_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.linear(points_features)
        features = self.norm(features.permute(0, 2, 1)).permute(0, 2, 1)
        features = self.relu(features)
        return standard_voxel_pool(points_xyz, features, self._voxel_size)


def interpolate(
    xyz1: torch.Tensor,
    features1: torch.Tensor,
    xyz2: torch.Tensor,
    features2: torch.Tensor,
    k: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.cdist(xyz1, xyz2, p=2)
    topk_results = torch.topk(distances, k=k, dim=2, largest=False)
    topk_weights = 1 / (topk_results.values + 1e-8)
    normalized_weights = topk_weights / topk_weights.sum(dim=2, keepdim=True)
    batch_indices = (
        torch.arange(
            topk_results.indices.shape[0],
            device=topk_results.indices.device,
            dtype=torch.int64,
        )
        .view(-1, 1)
        .expand(-1, topk_results.indices.shape[1] * k)
        .flatten()
    )
    selected = features2[batch_indices, topk_results.indices.flatten(), :]
    selected = selected * normalized_weights.flatten().unsqueeze(-1)
    selected = selected.reshape(-1, k, features2.shape[2]).sum(dim=1)
    interpolated = selected.view(features1.shape[0], features1.shape[1], features2.shape[2])
    return xyz1, interpolated


class TransitionUpBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear_1a = nn.Linear(in_dim, out_dim)
        self.linear_1b = nn.Linear(out_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(
        self,
        points_xyz: torch.Tensor,
        points_features: torch.Tensor,
        skipped_xyz: torch.Tensor,
        skipped_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.linear_1a(points_features)
        out = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.relu(out)
        interpolate_xyz, interpolate_features = interpolate(
            skipped_xyz, skipped_features, points_xyz, out, 1
        )
        skipped = self.linear_1b(skipped_features)
        skipped = self.bn(skipped.permute(0, 2, 1)).permute(0, 2, 1)
        skipped = self.relu(skipped)
        return interpolate_xyz, interpolate_features + skipped


class GroupVectorAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, groups: int) -> None:
        super().__init__()
        self.q = nn.Linear(in_dim, in_dim)
        self.k = nn.Linear(in_dim, in_dim)
        self.v = nn.Linear(in_dim, in_dim)
        self.conv_weights = nn.Conv2d(
            in_dim, out_dim, 1, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm1d(in_dim)
        self.delta_mult = PositionalEncoding(in_dim)
        self.delta_bias = PositionalEncoding(in_dim)
        self.softmax_1d = nn.Softmax(dim=1)
        self.linear = nn.Linear(in_dim, in_dim)
        self.out_dim = out_dim
        self.groups = groups

    def forward(
        self,
        points_xyz: torch.Tensor,
        points_features: torch.Tensor,
        neighbours_xyz: torch.Tensor,
        neighbours_features: torch.Tensor,
    ) -> torch.Tensor:
        delta_mult_out = self.delta_mult(points_xyz.unsqueeze(-2), neighbours_xyz)
        delta_bias_out = self.delta_bias(points_xyz.unsqueeze(-2), neighbours_xyz)
        q_out = self.q(points_features.unsqueeze(-2))
        k_out = self.k(neighbours_features)
        v_out = self.v(neighbours_features)
        vector_attention = delta_mult_out * (q_out - k_out) + delta_bias_out
        omega_out = self.conv_weights(vector_attention.permute(0, 3, 1, 2))
        omega_out = self.softmax_1d(omega_out)
        batch, height, width, channels = v_out.shape
        weight_encoding = (
            omega_out.permute(0, 2, 3, 1).unsqueeze(-1)
            * v_out.reshape(
                batch, height, width, self.groups, int(channels / self.groups)
            )
        ).reshape(batch, height, width, -1)
        out = torch.sum(weight_encoding, dim=2)
        out = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = torch.relu(out)
        return self.linear(out)


class PointTransformerV2Block(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, groups: int = 2, K: int = 16
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.gva = GroupVectorAttention(out_dim, groups, groups)
        self.linear_2 = nn.Linear(out_dim, in_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.k = K

    def forward(
        self, points_xyz: torch.Tensor, points_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = points_features.clone()
        distances = torch.cdist(points_xyz, points_xyz)
        _, indices = torch.topk(distances, self.k, largest=False)
        neighbours_xyz = index_points(points_xyz, indices)
        out = self.linear_1(points_features)
        neighbours_features = index_points(out, indices)
        out = self.gva(points_xyz, out, neighbours_xyz, neighbours_features)
        out = self.linear_2(out)
        out += residual
        return points_xyz, out


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.matmul(adj, x))


class GCNResStandardOps(nn.Module):
    """Parameter-compatible standard-operator deployment form of GCN_res."""

    def __init__(self, cfg: object, in_dim: int = 4) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, 48)
        self.ptb_0 = PointTransformerV2Block(in_dim=48, out_dim=48)
        self.gcn_0 = GraphConvolutionLayer(48, 48)
        self.tdb_1 = TransitionDownBlock(48, 96, [0.06] * 3)
        self.ptb_1 = PointTransformerV2Block(96, 96, K=16)
        self.tdb_2 = TransitionDownBlock(96, 192, [0.13] * 3)
        self.ptb_2 = PointTransformerV2Block(192, 192, K=2)
        self.tdb_3 = TransitionDownBlock(192, 384, [0.325] * 3)
        self.ptb_3 = PointTransformerV2Block(384, 384, K=1)
        self.tdb_4 = TransitionDownBlock(384, 512, [0.8125] * 3)
        self.ptb_4 = PointTransformerV2Block(512, 512, K=1)

        self.fpn_c1 = nn.Conv1d(48, 48, 1)
        self.fpn_c2 = nn.Conv1d(96, 48, 1)
        self.fpn_c3 = nn.Conv1d(192, 48, 1)
        self.fpn_c4 = nn.Conv1d(384, 48, 1)
        self.fpn_c5 = nn.Conv1d(512, 48, 1)
        self.fpn_c1_linear = nn.Linear(48, 48)
        self.fpn_c2_linear = nn.Linear(48, 96)
        self.fpn_c3_linear = nn.Linear(48, 192)
        self.fpn_c4_linear = nn.Linear(48, 384)
        self.fpn_c5_linear = nn.Linear(48, 512)
        self.residual_linear = nn.Conv1d(48, 48, 1)

        self.tub_6 = TransitionUpBlock(512, 384)
        self.ptb_6 = PointTransformerV2Block(384, 384, K=2)
        self.tub_7 = TransitionUpBlock(384, 192)
        self.ptb_7 = PointTransformerV2Block(192, 192, K=2)
        self.tub_8 = TransitionUpBlock(192, 96)
        self.ptb_8 = PointTransformerV2Block(96, 96, K=4)
        self.tub_9 = TransitionUpBlock(96, 48)
        self.ptb_9 = PointTransformerV2Block(48, 48, K=16)
        self.mlp = nn.Linear(48, int(getattr(cfg, "num_class")))
        self.activation = nn.ReLU()

    def forward(
        self, points: torch.Tensor, adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        points_xyz, points_features = points[:, :, :3], points
        out = self.linear_1(points_features)
        out_xyz, out_features = self.ptb_0(points_xyz, out)
        out_features = self.gcn_0(out_features, adj)
        skipped_0_xyz, skipped_0_features = out_xyz.clone(), out_features.clone()

        out_xyz, out_features = self.tdb_1(out_xyz, out_features)
        out_xyz, out_features = self.ptb_1(out_xyz, out_features)
        skipped_1_xyz, skipped_1_features = out_xyz.clone(), out_features.clone()
        out_xyz, out_features = self.tdb_2(out_xyz, out_features)
        out_xyz, out_features = self.ptb_2(out_xyz, out_features)
        skipped_2_xyz, skipped_2_features = out_xyz.clone(), out_features.clone()
        out_xyz, out_features = self.tdb_3(out_xyz, out_features)
        out_xyz, out_features = self.ptb_3(out_xyz, out_features)
        skipped_3_xyz, skipped_3_features = out_xyz.clone(), out_features.clone()
        out_xyz, out_features = self.tdb_4(out_xyz, out_features)
        out_xyz, out_features = self.ptb_4(out_xyz, out_features)

        fpn_out_c1 = self.fpn_c1(skipped_0_features.permute(0, 2, 1))
        fpn_out_c2 = self.fpn_c2(skipped_1_features.permute(0, 2, 1))
        fpn_out_c3 = self.fpn_c3(skipped_2_features.permute(0, 2, 1))
        fpn_out_c4 = self.fpn_c4(skipped_3_features.permute(0, 2, 1))
        fpn_out_c5 = self.fpn_c5(out_features.permute(0, 2, 1))
        fpn_out_c1 = self.fpn_c1_linear(fpn_out_c1.permute(0, 2, 1))
        fpn_out_c2 = self.fpn_c2_linear(fpn_out_c2.permute(0, 2, 1))
        fpn_out_c3 = self.fpn_c3_linear(fpn_out_c3.permute(0, 2, 1))
        fpn_out_c4 = self.fpn_c4_linear(fpn_out_c4.permute(0, 2, 1))
        fpn_out_c5 = self.fpn_c5_linear(fpn_out_c5.permute(0, 2, 1))

        out_xyz, out_features = self.tub_6(
            out_xyz, out_features, skipped_3_xyz, skipped_3_features
        )
        out_features = out_features + fpn_out_c4
        out_xyz, out_features = self.ptb_6(out_xyz, out_features)
        out_xyz, out_features = self.tub_7(
            out_xyz, out_features, skipped_2_xyz, skipped_2_features
        )
        out_features = out_features + fpn_out_c3
        out_xyz, out_features = self.ptb_7(out_xyz, out_features)
        out_xyz, out_features = self.tub_8(
            out_xyz, out_features, skipped_1_xyz, skipped_1_features
        )
        out_features = out_features + fpn_out_c2
        out_xyz, out_features = self.ptb_8(out_xyz, out_features)
        out_xyz, out_features = self.tub_9(
            out_xyz, out_features, skipped_0_xyz, skipped_0_features
        )
        out_features = out_features + fpn_out_c1
        out_xyz, out_features = self.ptb_9(out_xyz, out_features)
        return points_xyz, self.mlp(out_features)
