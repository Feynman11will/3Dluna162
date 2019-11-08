'''
@Author: your name
@Date: 2018-12-28 23:57:38
@LastEditTime: 2019-10-28 14:30:45
@LastEditors: Please set LastEditors
@Description: In User Settings Edi
@FilePath: /3D-Lung-nodules-detection-master/training/detector/data.py
'''
# -*- coding: utf-8 -*-
from labelmapping import LabelMapping
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
from lib.crop import Crop


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
    bsamples = torch.zeros([l]+list(batch[0][0].shape))
    blabels = torch.zeros([l]+list(batch[0][1].shape))
    bcoords = torch.zeros([l]+list(batch[0][2].shape))
    ss = []
    ls = []
    cs = []
    # TODO:
    '''
    此处传来一个bug :b[0]形状不匹配
    RuntimeError: The expanded size of the tensor (128) must match the existing size (340) at non-singleton dimension 2.  
    Target sizes: [1, 128, 128, 128].  Tensor sizes: [128, 340, 128]
    此乃偶然现象, 
    解决方案：
        1. 使用batch中的其他图片代替。方案是取形状的众数，来代替异常
        2. 回到crop类，进行debug
    '''
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

    return bsamples, blabels, bcoords