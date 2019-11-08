'''
@Author: your name
@Date: 2018-12-28 23:57:38
@LastEditTime: 2019-10-28 14:30:45
@LastEditors: Please set LastEditors
@Description: In User Settings Edi
@FilePath: /3D-Lung-nodules-detection-master/training/detector/data.py
'''
# -*- coding: utf-8 -*-
"""
Created on 2018/12/18 13:08

# n-net数据调整
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import collections
import random
from layers import iou
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate


class DataBowl3Detector(Dataset):
    def __init__(self, data_dir, split_path, config, phase='train', split_comber=None):
        """
        n-net输入数据调整
        :param data_dir:
        :param split_path: split_path这个变量，源码中是个存储name的numpy，事先早已经生成了，这里需要根据自己的训练数据做调整
        :param config:
        :param phase:
        :param split_comber:
        """
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.max_stride = config['max_stride']
        self.stride = config['stride']
        sizelim = config['sizelim'] / config['reso']
        sizelim2 = config['sizelim2'] / config['reso']
        sizelim3 = config['sizelim3'] / config['reso']
        self.blacklist = config['blacklist']
        self.isScale = config['aug_scale']
        self.r_rand = config['r_rand_crop']
        self.augtype = config['augtype']
        self.pad_value = config['pad_value']
        self.split_comber = split_comber

        idcs_list = np.load(split_path)

        # idcs_list = [f.split('_')[0] for f in os.listdir(split_path)]
        # idcs = sorted(set(idcs_list), key=idcs_list.index)
        idcs = np.sort(idcs_list)
        # if phase != 'test':
        #     idcs = [f for f in idcs if (f not in self.blacklist)]

        self.filenames = [os.path.join(data_dir, '%s_clean.npy' % idx) for idx in idcs]
        self.kagglenames = [f for f in self.filenames if len(f.split('/')[-1].split('_')[0]) > 20]
        self.lunanames = [f for f in self.filenames if len(f.split('/')[-1].split('_')[0]) < 20]

        labels = []

        for idx in idcs:
            # print(idx)
            path_ = os.path.join(data_dir, '%s_label.npy' % idx)

            l = np.load(path_, allow_pickle=True)
            if np.all(l == 0):
                l = np.array([])
            labels.append(l)

        self.sample_bboxes = labels
        if self.phase != 'test':
            self.bboxes = []
            for i, l in enumerate(labels):
                if len(l) > 0:
                    for t in l:
                        if t[3] > sizelim:
                            self.bboxes.append([np.concatenate([[i], t])])
                        if t[3] > sizelim2:
                            self.bboxes += [[np.concatenate([[i], t])]] * 2
                        if t[3] > sizelim3:
                            self.bboxes += [[np.concatenate([[i], t])]] * 4

            self.bboxes = np.concatenate(self.bboxes, axis=0)

        self.crop = Crop(config)
        self.label_mapping = LabelMapping(config, self.phase)

    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        isRandomImg = False
        if self.phase != 'test':
            if idx >= len(self.bboxes):
                isRandom = True
                idx = idx % len(self.bboxes)
                isRandomImg = np.random.randint(2)
            else:
                isRandom = False
        else:
            isRandom = False


        if self.phase != 'test':
            if not isRandomImg:
                bbox = self.bboxes[idx]
                filename = self.filenames[int(bbox[0])]
                imgs = np.load(filename)
                bboxes = self.sample_bboxes[int(bbox[0])]
                isScale = self.augtype['scale'] and (self.phase == 'train')

                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes, isScale, isRandom)
                if self.phase == 'train' and not isRandom:
                    sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
                                                            ifflip=self.augtype['flip'],
                                                            ifrotate=self.augtype['rotate'],
                                                            ifswap=self.augtype['swap'])
            else:
                randimid = np.random.randint(len(self.kagglenames))
                filename = self.kagglenames[randimid]
                imgs = np.load(filename)
                bboxes = self.sample_bboxes[randimid]
                isScale = self.augtype['scale'] and (self.phase == 'train')
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes, isScale=False, isRand=True)

            label = self.label_mapping(sample.shape[1:], target, bboxes)
            sample = (sample.astype(np.float32) - 128) / 128
            # TODO: PRINT1
            # print('===========================')
            # print('imgs.shape:', imgs.shape)
            # print('sample.shape:', sample.shape)
            # print('label.shape:',label.shape)
            # print('coord.shape:', coord.shape)
            # print('===========================')
            return torch.from_numpy(sample), torch.from_numpy(label), coord
        else:
            imgs = np.load(self.filenames[idx])#[1,nz,nh,nw]
            bboxes = self.sample_bboxes[idx]#[n*[4]]
            nz, nh, nw = imgs.shape[1:]
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            imgs = np.pad(imgs, [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',
                          constant_values=self.pad_value) #[1,4z,4h,4w]

            xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, imgs.shape[1] / self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[2] / self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[3] / self.stride), indexing='ij')
            coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')#[3,z,h,w]

            imgs, nzhw = self.split_comber.split(imgs) # [nz*nh*nw,4z,4h,4w]
            coord2, nzhw2 = self.split_comber.split(coord,
                                                    side_len=self.split_comber.side_len / self.stride,
                                                    max_stride=self.split_comber.max_stride / self.stride,
                                                    margin=self.split_comber.margin / self.stride)#[3,z,h,w]
            assert np.all(nzhw == nzhw2)
            imgs = (imgs.astype(np.float32) - 128) / 128

            # imgs:shape [z/36*w/36*h/36,1,208,208,208]  nzhw = [z/36, w/36, h/36]
            # coord2:shape =[z/36*w/36*h/36,3,52,52,52]  nzhw2=[z/36, w/36, h/36]
            # TODO: PRINT1
            print('imgs.shape:',imgs.shape)
            print('coord2.shape:',coord2.shape)

            return torch.from_numpy(imgs), bboxes, torch.from_numpy(coord2), np.array(nzhw)

    def __len__(self):
        if self.phase == 'train':
            return int (len(self.bboxes) / (1 - self.r_rand))
        elif self.phase == 'val':
            return len(self.bboxes)
        else:
            return len(self.sample_bboxes)


def augment(sample, target, bboxes, coord, ifflip=True, ifrotate=True, ifswap=True):
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand() * 180
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1 / 180 * np.pi), -np.sin(angle1 / 180 * np.pi)],
                               [np.sin(angle1 / 180 * np.pi), np.cos(angle1 / 180 * np.pi)]])
            # 对xy标签进行旋转 旋转矩阵与a进行矩阵乘
            newtarget[1:3] = np.dot(rotmat, target[1:3] - size / 2) + size / 2
            if np.all(newtarget[:3] > target[3]) and np.all(newtarget[:3] < np.array(sample.shape[1:4]) - newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample, angle1, axes=(2, 3), reshape=False)
                coord = rotate(coord, angle1, axes=(2, 3), reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat, box[1:3] - size / 2) + size / 2
            else:
                counter += 1
                if counter == 3:
                    break
    if ifswap:
        if sample.shape[1] == sample.shape[2] and sample.shape[1] == sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample, np.concatenate([[0], axisorder + 1]))
            coord = np.transpose(coord, np.concatenate([[0], axisorder + 1]))
            target[:3] = target[:3][axisorder]
            bboxes[:, :3] = bboxes[:, :3][:, axisorder]

    if ifflip:
        #         flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
        sample = np.ascontiguousarray(sample[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        coord = np.ascontiguousarray(coord[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        for ax in range(3):
            if flipid[ax] == -1:
                target[ax] = np.array(sample.shape[ax + 1]) - target[ax]
                bboxes[:, ax] = np.array(sample.shape[ax + 1]) - bboxes[:, ax]
    return sample, target, bboxes, coord


class Crop(object):
    def __init__(self, config):
        self.crop_size = config['crop_size']
        self.bound_size = config['bound_size']
        self.stride = config['stride']  # =4
        self.pad_value = config['pad_value']

    """
    0. __call__:函数的作用
        1. 使得类的实例对象可以像调用函数一样被调用
    1. 如果是多尺度：将目标进行多尺度的放大，缩放尺度的范围需要根据肺结节目标的大小自适应的放大
        这些bbox中存储着多个肺结节的bbox
        imgs, bbox[1:]单个目标, bboxes一张图像中的多个目标,isScale,isRandom
        在2d的目标检测中也可以
    2. bound_size = self.bound_size 这个bound_size =12
        python 中的copy()创建一个引用
        deepcopy为创建一个新的副本
    """

    def __call__(self, imgs, target, bboxes, isScale=False, isRand=False):
        if isScale:
            radiusLim = [8., 120.]
            scaleLim = [0.75, 1.25]
            scaleRange = [np.min([np.max([(radiusLim[0] / target[3]), scaleLim[0]]), 1])
                , np.max([np.min([(radiusLim[1] / target[3]), scaleLim[1]]), 1])]
            scale = np.random.rand() * (scaleRange[1] - scaleRange[0]) + scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float') / scale).astype('int')
        else:
            crop_size = self.crop_size
        bound_size = self.bound_size
        target = np.copy(target)
        bboxes = np.copy(bboxes)

        start = []
        """
        0. 确定剪裁区域的起始点
            if s>e:
                    start.append(np.random.randint(e,s))#!
            else:
                start.append(int(target[i])-crop_size[i]/2+np.random.randint(-bound_size/2,bound_size/2))
        """

        for i in range(3):
            if not isRand:
                r = target[3] / 2
                s = np.floor(target[i] - r) + 1 - bound_size
                e = np.ceil(target[i] + r) + 1 + bound_size - crop_size[i]  # crop_size = [128,128,128]
            else:
                s = np.max([imgs.shape[i + 1] - crop_size[i] / 2, imgs.shape[i + 1] / 2 + bound_size])
                e = np.min([crop_size[i] / 2, imgs.shape[i + 1] / 2 - bound_size])
                target = np.array([np.nan, np.nan, np.nan, np.nan])
            if s > e:
                start.append(np.random.randint(e, s))  # !
            else:
                start.append(int(target[i]) - crop_size[i] / 2 + np.random.randint(-bound_size / 2, bound_size / 2))

        normstart = np.array(start).astype('float32') / np.array(imgs.shape[1:]) - 0.5
        normsize = np.array(crop_size).astype('float32') / np.array(imgs.shape[1:])
        """
        1. 获取crop的范围normstart的范围是从[-corp_size/2, 512 - crop_size/2]
            xx,yy,zz的范围是从[normstart[0],normstart[0]+normsize[0]]
            coord 为np.concatenate(xx,yy,zz)
        2. 提取感兴趣区域 
            crop为图像的含有目标的感兴趣立方体区域
            对crop进行填充，填充后的形状确保均为127*127大小
        3. 
            如果start 超出了图像边界，左侧填充长度为 leftpad = -start[i]，否则就不填充
            如果crop超出了img_size,rightpad>0,就填充0，否则，就不填充    
            rightpad = max(0,start[i]+crop_size[i]-imgs.shape[i+1])
        """

        xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0] + normsize[0], self.crop_size[0] / self.stride),
                                 np.linspace(normstart[1], normstart[1] + normsize[1], self.crop_size[1] / self.stride),
                                 np.linspace(normstart[2], normstart[2] + normsize[2], self.crop_size[2] / self.stride),
                                 indexing='ij')

        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')

        pad = []
        pad.append([0, 0])
        for i in range(3):
            leftpad = max(0, -start[i])
            rightpad = max(0, start[i] + crop_size[i] - imgs.shape[i + 1])
            pad.append([leftpad, rightpad])
        crop = imgs[:,
               max(start[0], 0):min(start[0] + crop_size[0], imgs.shape[1]),
               max(start[1], 0):min(start[1] + crop_size[1], imgs.shape[2]),
               max(start[2], 0):min(start[2] + crop_size[2], imgs.shape[3])]
        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)

        '''
        4. 对 bboxes和targets进行处理，目标标签向左迁移start[i]个单位
            target[i] = target[i] - start[i] 
        '''

        for i in range(3):
            target[i] = target[i] - start[i]
        for i in range(len(bboxes)):
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j]
        '''
        5. 使用skimage.zoom对图像进行中心缩放，而后进行信息的填充处理
            5.1 newpad = self.crop_size[0]-crop.shape[1:][0]
            5.2 如果是缩小，就进行填充处理
            5.3 对targets值以及bboxes进行缩放处理
        '''
        if isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop, [1, scale, scale, scale], order=1)
            newpad = self.crop_size[0] - crop.shape[1:][0]
            if newpad < 0:
                crop = crop[:, :-newpad, :-newpad, :-newpad]
            elif newpad > 0:
                pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)
            for i in range(4):
                target[i] = target[i] * scale
            for i in range(len(bboxes)):
                for j in range(4):
                    bboxes[i][j] = bboxes[i][j] * scale
        return crop, target, bboxes, coord


class LabelMapping(object):
    def __init__(self, config, phase):
        self.stride = np.array(config['stride'])
        self.num_neg = int(config['num_neg'])
        self.th_neg = config['th_neg']
        self.anchors = np.asarray(config['anchors'])
        self.phase = phase
        if phase == 'train':
            self.th_pos = config['th_pos_train']
        elif phase == 'val':
            self.th_pos = config['th_pos_val']



    def __call__(self, input_size, target, bboxes):

        stride = self.stride
        num_neg = self.num_neg
        th_neg = self.th_neg
        anchors = self.anchors
        th_pos = self.th_pos

        output_size = []
        for i in range(3):
            # assert (input_size[i] % stride == 0)
            output_size.append(int(input_size[i] / stride))
        # anchors = [ 10.0, 30.0, 60.]
        '''
        1. output_size = input_size/stride 128/4=32
        2. 获得锚点框中心坐标oz,oh,ow
        3. 背景为[-1,-1,-1,-1,-1]
        '''

        label = -1 * np.ones(output_size + [len(anchors), 5], np.float32)
        offset = ((stride.astype('float')) - 1) / 2  # 1.5

        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

        '''
        4. 对于所有的目标
            select_samples 
            任务： bbox与anchor 两两配对，目标与anchor大小相近且b iou with anchor 大于阈值grid位置置零，
                  如果d与a差距较大，或者iou<阈值，依然为-1
                  输出的结果是[0,-1,-1,-1,-1]
            
        '''

        for bbox in bboxes:
            for i, anchor in enumerate(anchors):
                iz, ih, iw = select_samples(bbox, anchor, th_neg, oz, oh, ow)  # th_neg=0.02
                label[iz, ih, iw, i, 0] = 0
        '''
        6. 限制背景坐标点数量到800个
            6.1 self.phase == 'train' and self.num_neg > 0:
                neg_z, neg_h, neg_w, neg_a 为没有目标的相应维度坐标索引
                np.where 返回一个turple ，该turple元素为每个维度的索引值 长度为每个维度符合条件的长度的乘积
                random.sample(a,b)从列表a中随机抽取b个元素
            label清零
            label背景设置为-1
        '''



        if self.phase == 'train' and self.num_neg > 0:
            neg_z, neg_h, neg_w, neg_a = np.where(label[:, :, :, :, 0] == -1)
            neg_idcs = random.sample(range(len(neg_z)), min(num_neg, len(neg_z)))
            neg_z, neg_h, neg_w, neg_a = neg_z[neg_idcs], neg_h[neg_idcs], neg_w[neg_idcs], neg_a[neg_idcs]
            label[:, :, :, :, 0] = 0
            label[neg_z, neg_h, neg_w, neg_a, 0] = -1

        if np.isnan(target[0]):
            return label
        iz, ih, iw, ia = [], [], [], []
        '''
        7 对target进行处理
            7.1target 与anchors 一一匹配
            7.2 含有target目标的地方设置为 [1, dz, dh, dw, dd]
                idx = random.sample(range(len(iz)), 1)[0]
                pos = [iz[idx], ih[idx], iw[idx], ia[idx]] 随机选取一个在target目标区域的坐标
                dz = (target[0] - oz[pos[0]]) / anchors[pos[3]]
                同理可得dh dw dd
        '''
        for i, anchor in enumerate(anchors):
            iiz, iih, iiw = select_samples(target, anchor, th_pos, oz, oh, ow)#th_pos0.5
            iz.append(iiz)
            ih.append(iih)
            iw.append(iiw)
            ia.append(i * np.ones((len(iiz),), np.int64))
        iz = np.concatenate(iz, 0)
        ih = np.concatenate(ih, 0)
        iw = np.concatenate(iw, 0)
        ia = np.concatenate(ia, 0)
        flag = True
        if len(iz) == 0:
            pos = []
            for i in range(3):
                pos.append(max(0, int(np.round((target[i] - offset) / stride))))
            idx = np.argmin(np.abs(np.log(target[3] / anchors)))
            pos.append(idx)
            flag = False
        else:
            idx = random.sample(range(len(iz)), 1)[0]
            pos = [iz[idx], ih[idx], iw[idx], ia[idx]]
        dz = (target[0] - oz[pos[0]]) / anchors[pos[3]]
        dh = (target[1] - oh[pos[1]]) / anchors[pos[3]]
        dw = (target[2] - ow[pos[2]]) / anchors[pos[3]]
        dd = np.log(target[3] / anchors[pos[3]])
        label[pos[0], pos[1], pos[2], pos[3], :] = [1, dz, dh, dw, dd]
        return label


def select_samples(bbox, anchor, th, oz, oh, ow):
    '''
    0. 参数解析
        bbox, anchor, th, oz, oh, ow
        bbox为一个单个目标，anchor单个毛癫狂 th阈值 =0.02 ozohow anchor的中心坐标
    1. 实现逻辑
        1.1
    3. 功能分析
    '''
    '''
    1.1 
    '''
    z, h, w, d = bbox
    '''
    最大重合： d anchor的最小值
    最小重叠： d anchor的最大值：跳过
    对3个尺度的anchor box 进行匹配
    '''
    max_overlap = min(d, anchor)
    min_overlap = np.power(max(d, anchor), 3) * th / max_overlap / max_overlap
    '''
    2   如果重叠值反向
            返回全零
        否则：
        计算起始点s终止点e，从左向右，mz==1 is oz out [s,e], iz==1 is oz in [s,e] ,mz 与oz  同理计算\
        (mh ,ih) (mw,iw) ，为在bbox附近的中心坐标
    '''
    if min_overlap > max_overlap:
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    else:
        s = z - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = z + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mz = np.logical_and(oz >= s, oz <= e)
        iz = np.where(mz)[0]

        s = h - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = h + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)

        mh = np.logical_and(oh >= s, oh <= e)
        ih = np.where(mh)[0]

        s = w - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = w + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mw = np.logical_and(ow >= s, ow <= e)
        iw = np.where(mw)[0]

        if len(iz) == 0 or len(ih) == 0 or len(iw) == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)

        '''
        0<len(iz)<32
        iz ih iw 平铺 形状为 (lz,lh,lw) 最终形状为[lz*lh*lw] ,iz ih iw 为存在目标的mask
        使用该iz ih iw选择 锚点框 网格的坐标
        '''
        lz, lh, lw = len(iz), len(ih), len(iw)
        iz = iz.reshape((-1, 1, 1))
        ih = ih.reshape((1, -1, 1))
        iw = iw.reshape((1, 1, -1))
        iz = np.tile(iz, (1, lh, lw)).reshape((-1))
        ih = np.tile(ih, (lz, 1, lw)).reshape((-1))
        iw = np.tile(iw, (lz, lh, 1)).reshape((-1))
        centers = np.concatenate([
            oz[iz].reshape((-1, 1)),
            oh[ih].reshape((-1, 1)),
            ow[iw].reshape((-1, 1))], axis=1)

        '''
        锚点框起始点终止点s0 e0 计算
        目标球 起始点终止点计算
        交叉部分体积的计算 intersection 并集的计算union
        计算bbox 与anchorbox之间的iou
        选择iou大于阈值的iz ih 和iw 部分
        '''
        r0 = anchor / 2
        s0 = centers - r0
        e0 = centers + r0

        r1 = d / 2
        s1 = bbox[:3] - r1
        s1 = s1.reshape((1, -1))
        e1 = bbox[:3] + r1
        e1 = e1.reshape((1, -1))

        overlap = np.maximum(0, np.minimum(e0, e1) - np.maximum(s0, s1))

        intersection = overlap[:, 0] * overlap[:, 1] * overlap[:, 2]
        union = anchor * anchor * anchor + d * d * d - intersection

        iou = intersection / union

        mask = iou >= th
        # if th > 0.4:
        #   if np.sum(mask) == 0:
        #      print(['iou not large', iou.max()])
        # else:
        #    print(['iou large', iou[mask]])

        iz = iz[mask]
        ih = ih[mask]
        iw = iw[mask]
        return iz, ih, iw


def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

def collate_train(batch):
    l = len(batch)
    # print(l)
    ll = len(batch[0])
    # shape1 = batch[0][0].shape
    # shape2 = batch[0][1].shape
    # shape3 = batch[0][2].shape
    bsamples = torch.zeros([l]+list(batch[0][0].shape))
    blabels = torch.zeros([l]+list(batch[0][1].shape))
    bcoords = torch.zeros([l]+list(batch[0][2].shape))
    ss = []
    ls = []
    cs = []
    for idx , b in enumerate(batch):
        bsamples[idx] = b[0]
        blabels[idx] = b[1]
        bcoords[idx] = torch.from_numpy(b[2])

        ss.append(list(b[0].shape))
        ls.append(list(b[1].shape))
        cs.append(list(b[2].shape))

    ss = np.array(ss)
    ls = np.array(ls)
    cs = np.array(cs)
    assert len(np.unique(ss))==2,"形状发生错误1"
    assert len(np.unique(ls))== 3,"形状发生错误2"
    assert len(np.unique(cs)) ==2,"形状发生错误3"
    return bsamples,blabels,bcoords