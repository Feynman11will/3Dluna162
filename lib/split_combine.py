import torch
import numpy as np


class SplitComb():
    # sidelen = 144, 16, 16, margin = 32, 170

    def __init__(self, side_len, max_stride, stride, margin, pad_value):
        self.side_len = side_len
        self.max_stride = max_stride
        self.stride = stride
        self.margin = margin
        self.pad_value = pad_value

        '''
        0. __init__:数据初始化 side_len = 144, max_stride=16, stride= 16, margin = 32, pad_value=170
        1. split 方法
            0. 如果参数为空，参数设置为0.中初始化参数
            1. zhw三个方向上分块数量确定,并在三个维度的后方进行padding ，padding值为170
                nz = int(np.ceil(float(z) / side_len))
                return : nzhw = [nz, nh, nw]
            2. 使用3级for循环nz, nh, nw , 提取块列表，列表长度为 nz, nh, nw
                return ： splits
        2. combine :
            1中split方法操作的逆操作
        '''
    def split(self, data, side_len=None, max_stride=None, margin=None):
        if side_len == None:
            side_len = self.side_len
        if max_stride == None:
            max_stride = self.max_stride
        if margin == None:
            margin = self.margin

        assert (side_len > margin)
        assert (side_len % max_stride == 0)
        assert (margin % max_stride == 0)

        splits = []
        _, z, h, w = data.shape


        nz = int(np.ceil(float(z) / side_len))
        nh = int(np.ceil(float(h) / side_len))
        nw = int(np.ceil(float(w) / side_len))

        nzhw = [nz, nh, nw]
        self.nzhw = nzhw

        pad = [[0, 0],
               [margin, nz * side_len - z + margin],
               [margin, nh * side_len - h + margin],
               [margin, nw * side_len - w + margin]]
        data = np.pad(data, pad, 'edge')

        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len + 2 * margin
                    sh = ih * side_len
                    eh = (ih + 1) * side_len + 2 * margin
                    sw = iw * side_len
                    ew = (iw + 1) * side_len + 2 * margin
                    # shape [4z/144*4w/144*4h/144,1,208,208,208]  nzhw = [4z/144, 4w/144, 4h/144]
                    split = data[np.newaxis, :, sz:ez, sh:eh, sw:ew]
                    splits.append(split)

        splits = np.concatenate(splits, 0)
        return splits, nzhw

    '''
    combine :
        1. 得到输出特征图的大小
            side_len /= stride
            margin /= stride
        2. 
    '''
    def combine(self, output, nzhw=None, side_len=None, stride=None, margin=None):
        # [nzhw, 36, 36, 36,3,5]
        if side_len == None:
            side_len = self.side_len
        if stride == None:
            stride = self.stride
        if margin == None:
            margin = self.margin
        if nzhw == None:
            nz = self.nz
            nh = self.nh
            nw = self.nw
        else:
            nz, nh, nw = nzhw
        assert (side_len % stride == 0)
        assert (margin % stride == 0)

        side_len /= stride
        margin /= stride

        splits = []
        # splits = [nzhw, 36, 36, 36,3,5]
        for i in range(len(output)):
            splits.append(output[i])
        # output shape [nz*36, nh*36, nz*36, 3, 5]
        output = -1000000 * np.ones((
            nz * side_len,
            nh * side_len,
            nw * side_len,
            splits[0].shape[3],
            splits[0].shape[4]), np.float32)

        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len
                    sh = ih * side_len
                    eh = (ih + 1) * side_len
                    sw = iw * side_len
                    ew = (iw + 1) * side_len

                    split = splits[idx][margin:margin + side_len, margin:margin + side_len, margin:margin + side_len]
                    output[sz:ez, sh:eh, sw:ew] = split
                    idx += 1
                    output
                    # output shape: [nz * 36, nh * 36, nz * 36, 3, 5]
        return output
