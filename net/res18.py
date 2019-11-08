'''
@Author: your name
@Date: 2018-12-28 23:57:38
@LastEditTime: 2019-10-29 15:22:57
@LastEditors: your name
@Description: In User Settings Edit
@FilePath: /3D-Lung-nodules-detection-master/training/detector/res18.py
'''
# -*- coding: utf-8 -*-
"""
3D U-NET
特征提取path：1个preblock，2个forw_blocks，2个forw_blocks，3个forw_blocks，3个forw_blocks
反卷积path：3个back_blocks，3个back_blocks，1个out_put层
"""
import torch
from torch import nn
import sys
sys.path.append('../')
sys.path.append('./')
from net.layers import *

from configure.general import config_res18 as config


# config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38','d92998a73d4654a442e6d6ba15bbb827','990fbe3f0a1b53878669967b9afd1441','820245d8b211808bd18e78ff5be16fdb','adc3bbc63d40f8761c59be10f1e504c3',
#                       '417','077','188','876','057','087','130','468']

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True))

        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks_forw = [2, 2, 3, 3]  # 前向传播的完整网络
        num_blocks_back = [3, 3]  # 反向传播需要更新参数的blocks（相当于keras的trainable）
        self.featureNum_forw = [24, 32, 64, 64, 64]  # ？
        self.featureNum_back = [128, 64, 64]  # ？
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i + 1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i + 1], self.featureNum_forw[i + 1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    if i == 0:
                        addition = 3
                    else:
                        addition = 0
                    blocks.append(PostRes(self.featureNum_back[i + 1] + self.featureNum_forw[i + 2] + addition,
                                          self.featureNum_back[i]))
                else:
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2, stride=2)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.drop = nn.Dropout3d(p=0.5, inplace=False)
        self.output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size=1),
                                    nn.ReLU(),
                                    # nn.Dropout3d(p = 0.3),
                                    nn.Conv3d(64, 5 * len(config['anchors']), kernel_size=1))

    def forward(self, x, coord):
        out = self.preBlock(x)  # 16
        out_pool, indices0 = self.maxpool1(out)
        out1 = self.forw1(out_pool)  # 32
        out1_pool, indices1 = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)  # 64
        # out2 = self.drop(out2)
        out2_pool, indices2 = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)  # 96
        out3_pool, indices3 = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)  # 96
        # out4 = self.drop(out4)

        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))  # 96+96
        # comb3 = self.drop(comb3)
        rev2 = self.path2(comb3)

        comb2 = self.back2(torch.cat((rev2, out2, coord), 1))  # 64+64
        comb2 = self.drop(comb2)
        out = self.output(comb2)
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        # out = out.transpose(1, 4).transpose(1, 2).transpose(2, 3).contiguous()
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        # out = out.view(-1, 5)
        return out


def get_model():
    net = Net()
    loss = Loss(config['num_hard'])# num_hard = 2
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb