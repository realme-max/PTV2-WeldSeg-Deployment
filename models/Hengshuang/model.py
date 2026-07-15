import torch
import torch.nn as nn
from pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction
from .transformer import TransformerBlock

import torch_geometric.nn as gnn  # 需要 PyTorch Geometric 库
from torch_geometric.nn import knn_graph


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 使用 1x1 卷积调整 x 的通道数
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.fc = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x 维度: [batch_size, in_channels, num_points]
        batch_size, in_channels, num_points = x.size()

        avg_out = torch.mean(x, dim=-1)  # [B, C, N] -> [B, C]
        max_out, _  = torch.max(x, dim=-1)  # [B, C, N] -> [B, C]
        out = avg_out + max_out
        weights = self.relu(self.fc(out))  # [B, C] -> [B, C]
        weights = weights.unsqueeze(-1)
        out = x * weights

        return out  # [batch_size, out_channels, num_points]


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConvLayer, self).__init__()
        self.conv = gnn.GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index).relu()


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, cfg.model.transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        self.gcn_layers = nn.ModuleList()  # 添加 GCN 层列表
        self.gcn_projection = nn.ModuleList()     # 添加 全链接 层列表

        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))
            self.gcn_layers.append(GraphConvLayer(channel, channel))  # 对应的 GCN 层
            self.gcn_projection.append(nn.Linear(channel, channel))

        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            # edge_index_i = self.generate_edge_index(xyz)
            # points = self.gcn_layers[i](points, edge_index_i)  # 在 transformer 前后加入 GCN 处理
            # points = self.gcn_projection[i](points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))

        return points, xyz_and_feats

    def generate_edge_index(self, xyz, k=5):
        """
        生成新的 edge_index，根据 K 最近邻的方式连接点。

        :param xyz: 当前点云的坐标，形状为 (B, N, 3)，
                    B 是批大小，N 是点的数量，3 是每个点的坐标维度
        :param k: 每个点连接的最近邻点的数量
        :return: edge_index: 新的边索引，形状为 (2, E)，E 是边的数量
        """
        # 将 xyz 的形状调整为 (B * N, 3)，以便于 KNN 计算
        B, N, _ = xyz.size()
        xyz_reshaped = xyz.view(-1, 3)  # 形状变为 (B*N, 3)

        # 使用 KNN 计算边索引
        edge_index = knn_graph(xyz_reshaped, k=k, loop=False)  # 不考虑自环

        # edge_index 形状为 (2, E)，需要将其还原为 (2, E) 的形状
        # 每个批次的边索引需要考虑到批次大小
        # edge_index = edge_index + (torch.arange(B, device=xyz.device).view(-1, 1) * N)
        return edge_index


class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        self.nblocks = nblocks
    
    def forward(self, x):
        points, _ = self.backbone(x)
        res = self.fc2(points.mean(1))
        return res

class PointTransformerSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim

        # Fully connected layers after the backbone
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** nblocks)
        )

        # Transformer block for the highest level features
        self.transformer2 = TransformerBlock(32 * 2 ** nblocks, cfg.model.transformer_dim, nneighbor)

        # Initialize transition layers and transformers for lower levels
        self.nblocks = nblocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        self.attentions = nn.ModuleList()  # Attention mechanism

        for i in reversed(range(nblocks)):
            channel = 32 * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
            self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))

        for i in range(nblocks):
            channel = 32 * 4 ** i
            self.attentions.append(ChannelAttention(channel, channel))  # Adding attention module

        # Final fully connected layer for part segmentation
        self.fc3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)  # Output: num_class (part segmentation classes)
        )

    def forward(self, x):
        # Extract backbone features and final set of points
        points, xyz_and_feats = self.backbone(x)
        xyz = xyz_and_feats[-1][0]  # Get the coordinates from the last layer

        # Apply the transformer on the top level features
        points = self.transformer2(xyz, self.fc2(points))[0]

        # Reverse iterate through the blocks to upsample and apply transformers
        for i in range(self.nblocks):
            points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])  # Upsample features

            # Apply attention before feeding into transformer
            points = self.attentions[i](points)  # Channel attention mechanism
            points = self.transformers[i](xyz_and_feats[- i - 2][0], points)[0]

        # Final fully connected layers for segmentation prediction
        return self.fc3(points)




    