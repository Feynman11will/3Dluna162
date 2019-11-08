# 鼻咽癌CT影像组学量化分析研究

import numpy as np
import random
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
            iiz, iih, iiw = select_samples(target, anchor, th_pos, oz, oh, ow)  # th_pos0.5
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

