"""
Author: Benny
Date: Nov 2019
暂时无法应用
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm
from dataset import WeldDataset
import hydra
import omegaconf

from torch_geometric.nn import knn_graph
from sklearn.neighbors import kneighbors_graph

seg_classes = {'weld': [0, 1]}


def load_point_cloud_from_txt(file_path):
    """加载txt格式的点云文件，假设文件包含点的XYZ坐标"""
    point_cloud = np.loadtxt(file_path, delimiter=' ')  # 假设逗号分隔符，你可以根据实际文件格式调整
    return point_cloud

def save_point_cloud_with_labels(point_cloud, labels, output_file):
    """保存分割后的点云，格式：X,Y,Z,Label"""
    assert point_cloud.shape[0] == labels.shape[0]
    output_data = np.hstack((point_cloud, labels.reshape(-1, 1)))
    np.savetxt(output_file, output_data, fmt='%.6f', delimiter=' ')


def compute_adjacency(points, k=6):
    # - adj: 邻接矩阵
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

@hydra.main(config_path='config', config_name='partseg_v2_improved')
def main(args):
    batch_size = 4
    cloud_file = []
    output_txt = []
    input_txt = []
    cloud_file.append("weld_15")
    cloud_file.append("weld_28")
    cloud_file.append("weld_59")
    cloud_file.append("weld_62")
    format = ".txt"

    model_path = "D:/xlxlqqq/document/pointcloud_net/Point-Transformers-master-official/log/partseg/log_backup/V2_GCN_L/best_model.pth"
    for cloud_file_i in cloud_file:
        input_txt.append("D:/xlxlqqq/document/pointcloud_net/Point-Transformers-master-official/data/test_input/" + cloud_file_i + format)
        output_txt.append("D:/xlxlqqq/document/pointcloud_net/Point-Transformers-master-official/data/test_output/" + cloud_file_i + "_after" + format)

    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    '''MODEL LOADING'''
    args.input_dim = (6 if args.normal else 3) + 1           ##########################
    args.num_class = 2
    num_category = 1                  ##################
    num_part = args.num_class
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PTV2Segmentation')(args).cuda()   ###########
    classifier.cuda()

    try:
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0

    classifier.eval()
    # 进行前向推理，获取分割结果
    with torch.no_grad():
        seg_label_to_cat = {}
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        points = np.zeros((4, 2048, 4))
        for i in range(batch_size):
            temp = load_point_cloud_from_txt(input_txt[i])
            points[i, :, :] = temp

        points, target = points[:, :, :3], points[:, :, 3]

        label = np.zeros((4, 1))
        points = torch.tensor(points).float().cuda()  # 添加batch维度并转换为tensor
        target = torch.tensor(target).long().cuda()
        label = torch.tensor(label).long().cuda()
        adj = compute_adjacency(points)

        cur_batch_size, NUM_POINT, _ = points.size()
        cat_points = torch.cat([points, to_categorical(label, num_category).repeat(1, points.shape[1], 1)], -1)
        seg_pred = classifier(cat_points, adj)
        seg_pred = seg_pred[1]  # 假设你的模型可以直接处理 (batch_size, num_points, num_features) 的输入
        cur_pred_val = seg_pred.cpu().data.numpy()
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((1, NUM_POINT)).astype(np.int32)
        target = target.cpu().data.numpy()

        for i in range(batch_size):
            cat = seg_label_to_cat[target[i, 0]]
            logits = cur_pred_val_logits[i, :, :]
            cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

        correct = np.sum(cur_pred_val == target)

    # 保存分割后的点云
    points = points.cpu().numpy().squeeze()
    cur_pred_val = cur_pred_val.squeeze()
    save_point_cloud_with_labels(points, cur_pred_val, output_txt)
    print(f"分割后的点云已保存到: {output_txt}")


if __name__ == '__main__':
    main()
