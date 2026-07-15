
import torch

from torchinfo import summary

from models.testParameters.model import PTV2Segmentation
from models.Hengshuang_o.model import PointTransformerSeg

from argparse import Namespace

import argparse

def parse_args_V1():
    parser = argparse.ArgumentParser(description='Point Cloud GCN Training')

    parser.add_argument('--batch_size', type=int, default=4, help='Size of each batch')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--num_point', type=int, default=2048, help='Number of points in point cloud')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer type')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--normal', type=bool, default=False, help='Use normal vectors')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay factor')
    parser.add_argument('--step_size', type=int, default=20, help='Step size for learning rate decay')
    parser.add_argument('--input_dim', type=float, default=4, help='Learning rate decay factor')
    parser.add_argument('--num_class', type=int, default=2, help='Step size for learning rate decay')

    parser.add_argument('--nneighbor', type=int, default=16, help='Step size for learning rate decay')
    parser.add_argument('--nblocks', type=float, default=4, help='Learning rate decay factor')
    parser.add_argument('--transformer_dim', type=int, default=512, help='Step size for learning rate decay')

    return parser.parse_args()

def parse_args_V2():
    parser = argparse.ArgumentParser(description='Point Cloud GCN Training')

    parser.add_argument('--batch_size', type=int, default=4, help='Size of each batch')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--num_point', type=int, default=2048, help='Number of points in point cloud')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer type')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--normal', type=bool, default=False, help='Use normal vectors')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay factor')
    parser.add_argument('--step_size', type=int, default=20, help='Step size for learning rate decay')
    parser.add_argument('--input_dim', type=float, default=4, help='Learning rate decay factor')
    parser.add_argument('--num_class', type=int, default=2, help='Step size for learning rate decay')

    return parser.parse_args()


def main_V2Series():
    args = parse_args_V2()

    model = PTV2Segmentation(args).cuda()

    checkpoint = torch.load('./log/partseg/Nico/best_model.pth')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])

    cat_points = torch.randn(4, 2048, 4).cuda()  # 根据模型的实际输入调整形状
    adj = torch.randn(4, 2048, 2048).cuda()  # 根据实际需求调整形状
    # summary(model, [(4, 2048, 4), (4, 2048, 2048)])
    summary(model, (4, 2048, 4))

def main_V1Series():
    args = parse_args_V1()

    args.model = Namespace()
    args.model.nneighbor = 16  # 定义二级属性
    args.model.nblocks = 4  # 定义二级属性
    args.model.transformer_dim = 512  # 定义二级属性

    args.input_dim = (6 if args.normal else 3) + 1  ##########################
    args.num_class = 2

    num_category = 1  ##################
    num_part = args.num_class

    model = PointTransformerSeg(args).cuda()

    checkpoint = torch.load('./log/partseg/Hengshuang_o/best_model.pth')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])

    cat_points = torch.randn(4, 2048, 4).cuda()  # 根据模型的实际输入调整形状
    adj = torch.randn(4, 2048, 2048).cuda()  # 根据实际需求调整形状
    # summary(model, [(4, 2048, 4), (4, 2048, 2048)])
    summary(model, (4, 2048, 4))


if __name__ == "__main__":
    # main_V2Series()
    main_V1Series()