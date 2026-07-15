import torch
import torch.nn as nn
from .ptv2_utils import PointTransformerV2Block, TransitionDownBlock, TransitionUpBlock


class LocalFeatureAggregation(nn.Module):
    def __init__(self, input_dim, output_dim, k=8):
        """
        局部特征聚合模块，使用 RandLA-Net 的 MLP 方法。

        参数：
        - input_dim: 输入特征维度
        - output_dim: 输出特征维度
        - k: 每个点的邻居数量
        """
        super(LocalFeatureAggregation, self).__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),  # 对中心点和邻域特征的拼接做降维
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),  # 提取局部增强后的特征
        )
        self.output_dim = output_dim

    def forward(self, points):
        """
        前向传播方法。

        参数：
        - points: 输入点云特征，形状为 (B, N, C)

        返回：
        - 输出点云特征，形状为 (B, N, output_dim)
        """
        B, N, C = points.shape
        device = points.device

        # 初始化特征存储
        aggregated_features = torch.zeros(B, N, self.output_dim, device=device)

        for b in range(B):
            # 获取当前批次点
            cur_points = points[b]  # (N, C)

            # KNN 搜索 (基于欧氏距离)
            dist_matrix = torch.cdist(cur_points, cur_points)  # (N, N)
            knn_indices = dist_matrix.topk(self.k, largest=False).indices  # 最近 K 个点 (N, k)

            # 局部特征聚合
            for i in range(N):
                center_point = cur_points[i].unsqueeze(0)  # 中心点特征 (1, C)
                neighbors = cur_points[knn_indices[i]]  # 邻居点特征 (k, C)

                # 拼接中心点和邻居点的特征
                combined_features = torch.cat(
                    [neighbors - center_point, center_point.expand_as(neighbors)], dim=-1
                )  # (k, 2 * C)

                # 应用 MLP
                enhanced_features = self.mlp(combined_features)  # (k, output_dim)

                # 聚合增强后的特征 (例如取均值)
                aggregated_features[b, i] = enhanced_features.mean(dim=0)

        return aggregated_features



class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConvolutionLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, adj):
        # x: 输入特征，adj: 邻接矩阵
        out = torch.matmul(adj, x)  # 图卷积操作
        out = self.linear(out)
        return out

# 没有继承LFA模块，暂时无用
class PTV2Classifier(nn.Module):

    def __init__(self, n_classes=1, in_dim=3):
        super().__init__()
        self.linear = nn.Linear(in_dim, 48)
        self.ptb_0 = PointTransformerV2Block(in_dim=48, out_dim=48)
        self.gcn_0 = GraphConvolutionLayer(48, 48)  # 添加GCN层

        self.tdb_1 = TransitionDownBlock(in_dim=48, out_dim=96, grid_size=[0.06] * 3)
        self.ptb_1 = PointTransformerV2Block(in_dim=96, out_dim=96, K=4)

        self.tdb_2 = TransitionDownBlock(in_dim=96, out_dim=192, grid_size=[0.13] * 3)
        self.ptb_2 = PointTransformerV2Block(in_dim=192, out_dim=192, K=2)

        self.tdb_3 = TransitionDownBlock(in_dim=192, out_dim=384, grid_size=[0.325] * 3)
        self.ptb_3 = PointTransformerV2Block(in_dim=384, out_dim=384, K=1)

        self.tdb_4 = TransitionDownBlock(in_dim=384, out_dim=512, grid_size=[0.8125] * 3)
        self.ptb_4 = PointTransformerV2Block(in_dim=512, out_dim=512, K=1)

        # self.avg_pool = nn.AvgPool1d(1)
        self.mlp = nn.Linear(512, n_classes)

    def forward(self, points):
        points_xyz, points_features = points[:, :, :3], points
        out = self.linear(points_features)
        out_xyz, out_features = self.ptb_0(points_xyz, out)

        out_xyz, out_features = self.tdb_1(out_xyz, out_features)
        out_xyz, out_features = self.ptb_1(out_xyz, out_features)

        out_xyz, out_features = self.tdb_2(out_xyz, out_features)
        out_xyz, out_features = self.ptb_2(out_xyz, out_features)

        out_xyz, out_features = self.tdb_3(out_xyz, out_features)
        out_xyz, out_features = self.ptb_3(out_xyz, out_features)

        out_xyz, out_features = self.tdb_4(out_xyz, out_features)
        out_xyz, out_features = self.ptb_4(out_xyz, out_features)

        # out = self.avg_pool(out_features.permute(0, 2, 1))
        out = torch.mean(out_features, dim=1)  # average pooling here because we don't how many L is left

        out = self.mlp(out.squeeze(-1))

        return out


class PTV2Segmentation(nn.Module):
    def __init__(self, cfg, in_dim=4):
        super().__init__()

        self.LFA = LocalFeatureAggregation(input_dim=4, output_dim=4, k=8)

        self.linear_1 = nn.Linear(in_dim, 48)

        # 定义Point Transformer Blocks和GCN
        self.ptb_0 = PointTransformerV2Block(in_dim=48, out_dim=48)
        self.gcn_0 = GraphConvolutionLayer(48, 48)  # GCN操作

        self.tdb_1 = TransitionDownBlock(in_dim=48, out_dim=96, grid_size=[0.06] *  3)
        self.ptb_1 = PointTransformerV2Block(in_dim=96, out_dim=96, K=16)
        # self.gcn_1 = GraphConvolutionLayer(96, 96)

        self.tdb_2 = TransitionDownBlock(in_dim=96, out_dim=192, grid_size=[0.13] * 3)
        self.ptb_2 = PointTransformerV2Block(in_dim=192, out_dim=192, K=2)
        # self.gcn_2 = GraphConvolutionLayer(192, 192)

        self.tdb_3 = TransitionDownBlock(in_dim=192, out_dim=384, grid_size=[0.325] * 3)
        self.ptb_3 = PointTransformerV2Block(in_dim=384, out_dim=384, K=1)
        # self.gcn_3 = GraphConvolutionLayer(384, 384)

        self.tdb_4 = TransitionDownBlock(in_dim=384, out_dim=512, grid_size=[0.8125] * 3)
        self.ptb_4 = PointTransformerV2Block(in_dim=512, out_dim=512, K=1)
        # self.gcn_4 = GraphConvolutionLayer(512, 512)

        # 构建特征金字塔输出
        self.fpn_c1 = nn.Conv1d(48, 48, kernel_size=1)
        self.fpn_c2 = nn.Conv1d(96, 48, kernel_size=1)
        self.fpn_c3 = nn.Conv1d(192, 48, kernel_size=1)
        self.fpn_c4 = nn.Conv1d(384, 48, kernel_size=1)
        self.fpn_c5 = nn.Conv1d(512, 48, kernel_size=1)
        self.fpn_c1_linear = nn.Linear(48, 48)
        self.fpn_c2_linear = nn.Linear(48, 96)
        self.fpn_c3_linear = nn.Linear(48, 192)
        self.fpn_c4_linear = nn.Linear(48, 384)
        self.fpn_c5_linear = nn.Linear(48, 512)

        # 残差融合
        self.residual_linear = nn.Conv1d(48, 48, kernel_size=1)

        # 上采样层
        self.tub_6 = TransitionUpBlock(in_dim=512, out_dim=384)
        self.ptb_6 = PointTransformerV2Block(in_dim=384, out_dim=384, K=2)

        self.tub_7 = TransitionUpBlock(in_dim=384, out_dim=192)
        self.ptb_7 = PointTransformerV2Block(in_dim=192, out_dim=192, K=2)

        self.tub_8 = TransitionUpBlock(in_dim=192, out_dim=96)
        self.ptb_8 = PointTransformerV2Block(in_dim=96, out_dim=96, K=4)

        self.tub_9 = TransitionUpBlock(in_dim=96, out_dim=48)
        self.ptb_9 = PointTransformerV2Block(in_dim=48, out_dim=48, K=16)

        self.mlp = nn.Linear(48, int(cfg.num_class))

    def forward(self, points, adj):
        points_xyz, points_features = points[:, :, :3], points

        # points_features = self.LFA(points_features)

        out = self.linear_1(points_features)

        out_features = self.gcn_0(out, adj)

        # 第一层，包含GCN
        out_xyz, out_features = self.ptb_0(points_xyz, out_features)
        # out_features = self.gcn_0(out_features, adj)
        skipped_0_xyz, skipped_0_features = torch.clone(out_xyz), torch.clone(out_features)

        # 第二层
        out_xyz, out_features = self.tdb_1(out_xyz, out_features)
        out_xyz, out_features = self.ptb_1(out_xyz, out_features)
        skipped_1_xyz, skipped_1_features = torch.clone(out_xyz), torch.clone(out_features)

        # 第三层
        out_xyz, out_features = self.tdb_2(out_xyz, out_features)
        out_xyz, out_features = self.ptb_2(out_xyz, out_features)
        skipped_2_xyz, skipped_2_features = torch.clone(out_xyz), torch.clone(out_features)

        # 第四层
        out_xyz, out_features = self.tdb_3(out_xyz, out_features)
        out_xyz, out_features = self.ptb_3(out_xyz, out_features)
        skipped_3_xyz, skipped_3_features = torch.clone(out_xyz), torch.clone(out_features)

        # 第五层
        out_xyz, out_features = self.tdb_4(out_xyz, out_features)
        out_xyz, out_features = self.ptb_4(out_xyz, out_features)
        skipped_4_xyz, skipped_4_features = torch.clone(out_xyz), torch.clone(out_features)

        # 构建特征金字塔逐层融合
        fpn_out_c1 = self.fpn_c1(skipped_0_features.permute(0, 2, 1))
        fpn_out_c2 = self.fpn_c2(skipped_1_features.permute(0, 2, 1))
        fpn_out_c3 = self.fpn_c3(skipped_2_features.permute(0, 2, 1))
        fpn_out_c4 = self.fpn_c4(skipped_3_features.permute(0, 2, 1))
        fpn_out_c5 = self.fpn_c5(out_features.permute(0, 2, 1))

        fpn_out_c1 = self.fpn_c1_linear(fpn_out_c1.permute(0, 2, 1))  # 先 permute 再升维
        fpn_out_c2 = self.fpn_c2_linear(fpn_out_c2.permute(0, 2, 1))  # 先 permute 再升维
        fpn_out_c3 = self.fpn_c3_linear(fpn_out_c3.permute(0, 2, 1))  # 先 permute 再升维
        fpn_out_c4 = self.fpn_c4_linear(fpn_out_c4.permute(0, 2, 1))  # 先 permute 再升维
        fpn_out_c5 = self.fpn_c5_linear(fpn_out_c5.permute(0, 2, 1))  # 先 permute 再升维

        # 残差融合
        # residual_features = self.residual_linear(fpn_out_c5)

        # 上采样过程，逐层恢复空间分辨率并融合残差特征
        out_xyz, out_features = self.tub_6(out_xyz, out_features, skipped_3_xyz, skipped_3_features)
        out_features = out_features + fpn_out_c4  # 融合特征金字塔第4层
        out_xyz, out_features = self.ptb_6(out_xyz, out_features)

        out_xyz, out_features = self.tub_7(out_xyz, out_features, skipped_2_xyz, skipped_2_features)
        out_features = out_features + fpn_out_c3  # 融合特征金字塔第3层
        out_xyz, out_features = self.ptb_7(out_xyz, out_features)

        out_xyz, out_features = self.tub_8(out_xyz, out_features, skipped_1_xyz, skipped_1_features)
        out_features = out_features + fpn_out_c2  # 融合特征金字塔第2层
        out_xyz, out_features = self.ptb_8(out_xyz, out_features)

        out_xyz, out_features = self.tub_9(out_xyz, out_features, skipped_0_xyz, skipped_0_features)
        out_features = out_features + fpn_out_c1  # 融合特征金字塔第1层
        out_xyz, out_features = self.ptb_9(out_xyz, out_features)

        # 最终分类层
        out = self.mlp(out_features)

        return points_xyz, out