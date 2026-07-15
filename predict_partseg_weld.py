"""
只能用于PTV2 baseline的模型定义框架下，完成点云推理并保存为可视化点云
xlxlqqq
2024.10.10
成功运行
"""
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

from sklearn.neighbors import kneighbors_graph

seg_classes = {'weld': [0, 1]}


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

@hydra.main(config_path='config', config_name='partseg_v2_improved_predict')
def main(args):
    model_path = "D:/xlxlqqq/document/pointcloud_net/Point-Transformers-master-official/log/partseg/log_backup/V2_GCN_LFA_Res_Best/best_model.pth"
    save_path = "D:/xlxlqqq/document/pointcloud_net/Point-Transformers-master-official/data/test_output/"
    omegaconf.OmegaConf.set_struct(args, False)

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
    with torch.no_grad():
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        test_data_size = len(testDataLoader.dataset)
        for i in range(test_data_size):
            points = torch.tensor(testDataLoader.dataset[i][0]).unsqueeze(0)
            label = torch.tensor(testDataLoader.dataset[i][1]).unsqueeze(0)
            target = torch.tensor(testDataLoader.dataset[i][2]).unsqueeze(0)
            cloud_file = os.path.splitext(os.path.basename(testDataLoader.dataset.datapath[i][1]))[0]

            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            adj = compute_adjacency(points)
            testInput = torch.cat([points, to_categorical(label, num_category).repeat(1, points.shape[1], 1)], -1)
            seg_pred = classifier(testInput, adj)
            seg_pred = seg_pred[1] #############
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT) ).astype(np.int32)
            target = target.cpu().data.numpy()

            cat = seg_label_to_cat[target[0, 0]]
            logits = cur_pred_val_logits[0, :, :]
            cur_pred_val[0, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            output_txt = save_path + cloud_file + "_after" + ".txt"

            temp_points = points[0, :, :].cpu().numpy().squeeze()
            temp_cur_pred_val = cur_pred_val[0, :].squeeze()
            save_point_cloud_with_labels(temp_points, temp_cur_pred_val, output_txt)
            print(f"分割后的点云已保存到: {output_txt}")

if __name__ == '__main__':
    main()