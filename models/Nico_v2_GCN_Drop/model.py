import torch
import torch.nn as nn
import torch.nn.functional as F
from .ptv2_utils import PointTransformerV2Block, TransitionDownBlock, TransitionUpBlock


## 截至20241202最好的模型
# 0.97021
# 0.90761
# 20241207加入前置LFA，测试
# 加入LFA后，有提升0.97451       0.92211


class KNNLayer(nn.Module):
    """K近邻选择层"""

    def __init__(self, K):
        super(KNNLayer, self).__init__()
        self.K = K

    def forward(self, points):
        # points: (B, N, 3)，B为batch size，N为点数，3为坐标维度
        B, N, _ = points.shape
        dist = torch.cdist(points, points)  # 计算所有点对之间的欧氏距离
        _, knn_idx = dist.topk(self.K, largest=False, dim=-1)  # 选择K个最近的邻居
        return knn_idx


class LocalFeatureAggregation(nn.Module):
    """局部特征聚合模块"""

    def __init__(self, in_channels, out_channels, K):
        super(LocalFeatureAggregation, self).__init__()
        self.K = K
        self.knn = KNNLayer(K)  # K近邻选择
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels * 2, out_channels, kernel_size=1),  # 邻居特征编码
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.attn_mlp = nn.Sequential(  # 注意力权重MLP
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.Softmax(dim=-1)
        )

    def forward(self, points_xyz, points_features):
        B, N, C = points_features.shape
        knn_idx = self.knn(points_xyz)  # (B, N, K)

        # 获取邻居特征
        knn_features = torch.gather(points_features, 1, knn_idx.unsqueeze(-1).expand(-1, -1, C))  # (B, N, K, C)
        knn_features = knn_features.view(B, N, self.K, C)

        # 将邻居特征与主点特征连接起来
        central_features = points_features.unsqueeze(2).expand(-1, -1, self.K, -1)  # (B, N, K, C)
        combined_features = torch.cat([central_features, knn_features], dim=-1)  # (B, N, K, 2*C)

        # 特征编码和加权求和
        combined_features = combined_features.view(B, N, 2 * C).permute(0, 2, 1)  # (B, 2*C, N)
        aggregated_features = self.mlp(combined_features)  # (B, out_channels, N)
        attn_weights = self.attn_mlp(aggregated_features)  # (B, out_channels, N)

        # 加权求和
        out_features = (aggregated_features * attn_weights).sum(dim=-1)  # (B, out_channels)
        return out_features


class LocalFeatureAggregationBest(nn.Module):
    def __init__(self, input_dim, output_dim, k=8):
        """
        局部特征聚合模块，使用 RandLA-Net 的 MLP 方法。

        参数：
        - input_dim: 输入特征维度
        - output_dim: 输出特征维度
        - k: 每个点的邻居数量
        """
        super(LocalFeatureAggregationBest, self).__init__()
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


class PTV2Segmentation(nn.Module):
    # def __init__(self, cfg, in_dim=4):
    def __init__(self, cfg, in_dim=19):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, 48)

        self.LFA = LocalFeatureAggregationBest(input_dim=19, output_dim=19, k=8)

        # 定义Point Transformer Blocks和GCN
        self.ptb_0 = PointTransformerV2Block(in_dim=48, out_dim=48)
        self.gcn_0 = GraphConvolutionLayer(48, 48)  # GCN操作
        self.lfa_0 = LocalFeatureAggregation(in_channels=48, out_channels=48, K=16)

        self.tdb_1 = TransitionDownBlock(in_dim=48, out_dim=96, grid_size=[0.06] * 3)
        self.ptb_1 = PointTransformerV2Block(in_dim=96, out_dim=96, K=16)
        self.lfa_1 = LocalFeatureAggregation(in_channels=96, out_channels=96, K=16)

        self.tdb_2 = TransitionDownBlock(in_dim=96, out_dim=192, grid_size=[0.13] * 3)
        self.ptb_2 = PointTransformerV2Block(in_dim=192, out_dim=192, K=2)
        self.lfa_2 = LocalFeatureAggregation(in_channels=192, out_channels=192, K=8)

        self.tdb_3 = TransitionDownBlock(in_dim=192, out_dim=384, grid_size=[0.325] * 3)
        self.ptb_3 = PointTransformerV2Block(in_dim=384, out_dim=384, K=1)
        self.lfa_3 = LocalFeatureAggregation(in_channels=384, out_channels=384, K=4)

        self.tdb_4 = TransitionDownBlock(in_dim=384, out_dim=512, grid_size=[0.8125] * 3)
        self.ptb_4 = PointTransformerV2Block(in_dim=512, out_dim=512, K=1)
        self.lfa_4 = LocalFeatureAggregation(in_channels=512, out_channels=512, K=4)

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
        self.ptb_6 = PointTransformerV2Block(in_dim=384, out_dim=384, K=1)
        # self.ptb_6 = PointTransformerV2Block(in_dim=384, out_dim=384, K=2)

        self.tub_7 = TransitionUpBlock(in_dim=384, out_dim=192)
        self.ptb_7 = PointTransformerV2Block(in_dim=192, out_dim=192, K=2)

        self.tub_8 = TransitionUpBlock(in_dim=192, out_dim=96)
        self.ptb_8 = PointTransformerV2Block(in_dim=96, out_dim=96, K=4)

        self.tub_9 = TransitionUpBlock(in_dim=96, out_dim=48)
        self.ptb_9 = PointTransformerV2Block(in_dim=48, out_dim=48, K=16)

        self.mlp = nn.Linear(48, int(cfg.num_class))
        self.activation = nn.ReLU()    # 激活函数
        # self.drop = nn.Dropout(p=0.5)  # 50% Dropout

    def forward(self, points, adj):
        points_xyz, points_features = points[:, :, :3], points
        out = self.linear_1(points_features)

        # points_features = self.LFA(points_features)

        # 第一层，包含GCN
        out_xyz, out_features = self.ptb_0(points_xyz, out)
        # out_xyz, out_features = self.lfa_0(out_xyz, out_features)
        out_features = self.gcn_0(out_features, adj)
        skipped_0_xyz, skipped_0_features = torch.clone(out_xyz), torch.clone(out_features)

        # 第二层
        out_xyz, out_features = self.tdb_1(out_xyz, out_features)
        out_xyz, out_features = self.ptb_1(out_xyz, out_features)
        # out_xyz, out_features = self.lfa_1(out_xyz, out_features)
        skipped_1_xyz, skipped_1_features = torch.clone(out_xyz), torch.clone(out_features)

        # 第三层
        out_xyz, out_features = self.tdb_2(out_xyz, out_features)
        out_xyz, out_features = self.ptb_2(out_xyz, out_features)
        # out_xyz, out_features = self.lfa_2(out_xyz, out_features)
        skipped_2_xyz, skipped_2_features = torch.clone(out_xyz), torch.clone(out_features)

        # 第四层
        out_xyz, out_features = self.tdb_3(out_xyz, out_features)
        out_xyz, out_features = self.ptb_3(out_xyz, out_features)
        # out_xyz, out_features = self.lfa_3(out_xyz, out_features)
        skipped_3_xyz, skipped_3_features = torch.clone(out_xyz), torch.clone(out_features)

        # 第五层
        out_xyz, out_features = self.tdb_4(out_xyz, out_features)
        out_xyz, out_features = self.ptb_4(out_xyz, out_features)
        # out_xyz, out_features = self.lfa_4(out_xyz, out_features)
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