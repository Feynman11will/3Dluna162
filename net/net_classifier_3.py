'''
@Author: your name
@Date: 2018-12-28 23:57:38
@LastEditTime: 2019-11-01 14:31:42
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /3D-Lung-nodules-detection-master/training/classifier/net_classifier_3.py
'''
# -*- coding: utf-8 -*-
"""

"""
import torch
from torch import nn
from layers import *
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import rotate
import numpy as np
import os
import sys

sys.path.append('../')

from data.configure import config as config_training

config = {}
config['topk'] = 5
config['resample'] = None
config['datadir'] = config_training['preprocess_path']
config['preload_train'] = True
config['bboxpath'] = config_training['bbox_path']
config['labelfile'] = '../../work/labels.csv'
config['preload_val'] = True

config['padmask'] = False

config['crop_size'] = [96, 96, 96]
config['scaleLim'] = [0.85, 1.15]
config['radiusLim'] = [6, 100]
config['jitter_range'] = 0.15
config['isScale'] = True

config['random_sample'] = True
config['T'] = 1
config['topk'] = 5
config['stride'] = 4
config['augtype'] = {'flip': True, 'swap': False, 'rotate': False, 'scale': False}

config['detect_th'] = 0.05
config['conf_th'] = -1
config['nms_th'] = 0.05
config['filling_value'] = 160

config['startepoch'] = 20
config['lr_stage'] = np.array([50, 100, 140, 160])
config['lr'] = [0.01, 0.001, 0.0001, 0.00001]
config['miss_ratio'] = 1
config['miss_thresh'] = 0.03


class CaseNet(nn.Module):
    def __init__(self, topk, nodulenet):
        super(CaseNet, self).__init__()
        self.NoduleNet = nodulenet
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.baseline = nn.Parameter(torch.Tensor([-30.0]).float())
        self.Relu = nn.ReLU()

    def forward(self, xlist, coordlist):
        # xlist: b x topk x 1x 96 x 96 x 96 # top5 cube proposal 的维度
        # coordlist: b x topk x 3 x 24 x 24 x 24 # top5 cube proposal 的坐标维度

        '''
        input: b*topk 个roi区域 以及 coordlist  b x topk 坐标
            xlist: b x topk x 1x 96 x 96 x 96 # top5 cube proposal 的维度
            coordlist: b x topk x 3 x 24 x 24 x 24
        逻辑：
            使用目标检测网络对b*topk个目标区域检测
            noduleFeat 分类结果输出层的前一层，处理：对noduleFeat获取中心的centerfeat，维度[nxtopk, 64, 1, 1, 1 ]
                经过两个全连接层,最终输出的维度为1 得到out
                out shape = [b,topk]
            nodulePred 结节预测输出 形状为 [bxtopk,24,24,24,3,5] view as [b, topk, 24x24x24x3x5]
            casePred 为 对noduleFeat sigmod 偏移
            return : noduleFeat nodulePred out
        分类器的逻辑：
            1. 使用数据输入器根据目标检测器输出的topk个预测结果位置，在预测结果附近crop 出topk个目标作为疑似位置，cropsize=96
            2. 对该topk个roi进行分类
        '''
        xsize = xlist.size()
        corrdsize = coordlist.size()
        xlist = xlist.view(-1, xsize[2], xsize[3], xsize[4], xsize[5])
        coordlist = coordlist.view(-1, corrdsize[2], corrdsize[3], corrdsize[4], corrdsize[5])

        noduleFeat, nodulePred = self.NoduleNet(xlist, coordlist)
        nodulePred = nodulePred.contiguous().view(corrdsize[0], corrdsize[1], -1)

        # featshape = [nxtopk , 64 , 24 , 24 ,24]
        featshape = noduleFeat.size()
        # centerFeat shape = [nxtopk , 64]
        centerFeat = self.pool(noduleFeat[:, :, featshape[2] / 2 - 1:featshape[2] / 2 + 1,
                               featshape[3] / 2 - 1:featshape[3] / 2 + 1,
                               featshape[4] / 2 - 1:featshape[4] / 2 + 1])
        # centerFeat shape = [nxtopk , 64]
        centerFeat = centerFeat[:, :, 0, 0, 0]
        out = self.dropout(centerFeat)
        out = self.Relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        out = out.view(xsize[0], xsize[1])
        base_prob = torch.sigmoid(self.baseline)
        casePred = 1 - torch.prod(1 - out, dim=1) * (
                    1 - base_prob.expand(out.size()[0]))  # pd是dim=1的out层输出，pi是sigmoid处理后的各个结节out
        return nodulePred, casePred, out
