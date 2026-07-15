"""
Author: Benny
Date: Nov 2019
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
import torch.nn as nn
from torchsummary import summary


seg_classes = {'weld': [0, 1]}

def model_summary(model, inputs):
    # Forward 模拟调用，确保不会改变模型状态
    with torch.no_grad():
        output = model(*inputs)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params}")
    print(f"可训练参数量: {trainable_params}")
    return output


def compute_adjacency(points, k=6):
    """
    返回：
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


from torch_geometric.nn import knn


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
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    root = hydra.utils.to_absolute_path('D:/xlxlqqq/document/pointnet/Pointnet_Pointnet2_pytorch-master/data/weld/')

    TRAIN_DATASET = WeldDataset(root=root, npoints=args.num_point, split='trainval', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    TEST_DATASET = WeldDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    # args.input_dim = (6 if args.normal else 3) + 16
    args.input_dim = (6 if args.normal else 3) + 1           ##########################
    args.num_class = 2
    # num_category = 16
    num_category = 1                  ##################
    num_part = args.num_class
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PTV2Segmentation')(args).cuda()   ###########
    # classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerSeg')(args).cuda()    ###########
    criterion = torch.nn.CrossEntropyLoss()

    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0

    cat_points = torch.randn(4, 2048, 4).cuda()  # 根据模型的实际输入调整形状
    adj = torch.randn(4, 2048, 2048).cuda()  # 根据实际需求调整形状
    model_summary(classifier, (cat_points, adj))
    print(classifier.parameters())