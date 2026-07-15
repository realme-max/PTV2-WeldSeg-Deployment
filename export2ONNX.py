'''
porgrammer: xlxlqqq
date: 20250326
'''

import os
import torch
import logging
import importlib
import shutil
import numpy as np

from tqdm import tqdm
from dataset import WeldDataset
import hydra
import omegaconf
import torch_cluster

from torch.onnx import register_custom_op_symbolic
# import torch.onnx.symbolic_registry as sym_registry

from sklearn.neighbors import kneighbors_graph

seg_classes = {'weld': [0, 1]}

# 调整后的符号函数（假设接收 5 个参数）
# 定义自定义 ONNX 符号化函数
def grid_cluster_symbolic(g, x, size, start, end):
    return g.op("CustomNamespace::GridCluster", x, size, start, end)


def compute_adjacency(points, k=6):
    """
    - adj: 邻接矩阵
    """
    # 使用sklearn计算k最近邻图
    B, N, D = points.size()  # 获取批次大小、点数和特征维度
    adj_list = []

    for i in range(B):
        points_cpu = points[i].cpu().numpy() if points[i].is_cuda else points[i].numpy()
        adj_matrix = kneighbors_graph(points_cpu, n_neighbors=k, mode='connectivity', include_self=False)
        adj = adj_matrix.toarray()
        adj_tensor = torch.FloatTensor(adj).to(points.device)  # 转换为Torch张量并保持设备
        adj_list.append(adj_tensor)

    return torch.stack(adj_list)  # 返回形状为 (B, N, N) 的邻接矩阵

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def save_point_cloud_with_labels(point_cloud, labels, output_file):
    """保存分割后的点云，格式：X,Y,Z,Label"""
    assert point_cloud.shape[0] == labels.shape[0]
    output_data = np.hstack((point_cloud, labels.reshape(-1, 1)))
    np.savetxt(output_file, output_data, fmt='%.6f', delimiter=' ')


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

@hydra.main(config_path='config', config_name='ONNXpartseg_v2_improved_predict')
def main(args):
    model_path = "D:/xlxlqqq/document/pointcloud_net/Point-Transformers-master-official/log/partseg/Nico_v2_GCN_ONNX/best_model.pth"
    omegaconf.OmegaConf.set_struct(args, False)

    torch.onnx.register_custom_op_symbolic("torch_cluster::grid", grid_cluster_symbolic, 16)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    root = hydra.utils.to_absolute_path('D:/xlxlqqq/document/pointnet/Pointnet_Pointnet2_pytorch-master/data/weld/')

    TEST_DATASET = WeldDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    args.input_dim = (6 if args.normal else 3) + 1           ##########################
    args.num_class = 2
    num_category = 1                  ##################
    num_part = args.num_class
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PTV2Segmentation')(args).cuda()   ###########

    try:
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')

    classifier = classifier.eval()

    # 定义输入张量（根据点云数据格式调整）
    # 示例输入：假设输入为 (batch_size, num_points, 3) 的浮点张量
    dummy_input = torch.randn(1, 2048, 4).cuda()  # 以1024个点、每个点3个坐标为例
    adj = compute_adjacency(dummy_input).cuda()

    print(torch.ops.torch_cluster.grid)

    # 导出ONNX模型
    # 导出模型时声明自定义算子域
    # torch.onnx.export(
    #     classifier, (dummy_input, adj), "model.onnx",
    #     opset_version=16, verbose=True
    # )
    #
    # torch.onnx.export(
    #     classifier, dummy_input, "model.onnx",
    #     opset_version=16, verbose=True
    # )

    torch.onnx.export(classifier, dummy_input, "model.onnx",
                      opset_version=16,
                      input_names=["input"],
                      output_names=["output"],
                      dynamic_axes={"input": {1: "num_points", 2: "features"},
                                    "indices": {1: "num_points", 2: "features"}})


if __name__ == '__main__':
    main()


