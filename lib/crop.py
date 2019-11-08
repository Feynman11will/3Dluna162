import warnings
from pprint import pprint

import numpy as np
from scipy.ndimage import zoom

from mydebuger import MyDebuger

debuger = MyDebuger(debug = True,logOutPath = '/data2/wangll/03Log/log.txt')

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
        debuger.login('---------------begin---------------')

        if isScale:
            debuger.login('isScale : ', isScale)
            debuger.login('scale is: ', scale)
        debuger.login('crop_size is: ', crop_size)
            # pprint('crop_size is: '.format(crop_size),'\n')

        bound_size = self.bound_size
        target = np.copy(target)
        bboxes = np.copy(bboxes)

        start = []

        debuger.login('isrand is :', isRand,' target is :',target)

        for i in range(3):
            if not isRand:
                r = target[3] / 2
                s = np.floor(target[i] - r) + 1 - bound_size
                e = np.ceil(target[i] + r) + 1 + bound_size - crop_size[i]  # crop_size = [128,128,128]
            else:
                s = np.max([imgs.shape[i + 1] - crop_size[i] / 2, imgs.shape[i + 1] / 2 + bound_size])
                e = np.min([crop_size[i] / 2, imgs.shape[i + 1] / 2 - bound_size])
                target = np.array([np.nan, np.nan, np.nan, np.nan])
            debuger.login('s ',i, 'is ',s,'e ',i, 'is ',e)
                    # pprint('r{} is {};s{} is {};e{} is {};'.format(i,r,i,s,i,e))
            if s > e:
                start.append(np.random.randint(e, s))  # !
            else:
                start.append(int(target[i]) - crop_size[i] / 2 + np.random.randint(-bound_size / 2, bound_size / 2))
            debuger.login('start is:',start)


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


        debuger.login('pad size is:', pad)


        crop = imgs[:,
               max(start[0], 0):min(start[0] + crop_size[0], imgs.shape[1]),
               max(start[1], 0):min(start[1] + crop_size[1], imgs.shape[2]),
               max(start[2], 0):min(start[2] + crop_size[2], imgs.shape[3])]


        for i in range(len(pad)):
            for j in range(len(pad[i])):
                if pad[i][j] <=  crop_size[0] :
                    pad[i][j] = pad[i][j]
                else :
                    pad[i][j] = crop_size[0]
                    debuger.login('Execption happend')


        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)

        debuger.login('paded_size is :',crop.shape)

        '''
        4. 对 bboxes和targets进行处理，目标标签向左迁移start[i]个单位
            target[i] = target[i] - start[i] 
        '''

        for i in range(3):
            target[i] = target[i] - start[i]
        for i in range(len(bboxes)):
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j]

        debuger.login('crop target is :', target)
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

            debuger.login('After zoom crop size is:', crop.shape)

            if newpad < 0:
                crop = crop[:, :-newpad, :-newpad, :-newpad]

            elif newpad > 0:
                pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)

            debuger.login('After zoom+Pad crop size is:', crop.shape)
            for i in range(4):
                target[i] = target[i] * scale
            for i in range(len(bboxes)):
                for j in range(4):
                    bboxes[i][j] = bboxes[i][j] * scale
            debuger.login('After zoom+Pad target is:', target)

        debuger.logout()

        return crop, target, bboxes, coord
