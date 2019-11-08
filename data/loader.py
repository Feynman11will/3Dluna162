'''
加载一张图片
'''


import warnings

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom


def load_itk_image(filename):
    """s
    :param filename:
    :return:
    """
    itkimage = sitk.ReadImage(filename)  # 读取.mhd文件
    numpyImage = sitk.GetArrayFromImage(itkimage)  # 获取数据，自动从同名的.raw文件读取

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))  # 原始CT坐标系的坐标原点coords
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))  # 原始CT坐标系的体素距离
    transformM = np.array(list(reversed(itkimage.GetDirection())))
    isflip = False
    if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
        isflip = True

    return numpyImage, numpyOrigin, numpySpacing, isflip


def worldToVoxelCoord(worldCoord, origin, spacing):
    """
    世界坐标转换为真实坐标
    :param worldCoord:
    :param origin:
    :param spacing:
    :return:
    """
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def lumTrans(img):
    """
    灰度标准化（HU值），将HU值（[-1200, 600]）线性变换至0~255内的灰度值
    :param img:
    :return:
    """
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def resample(imgs, spacing, new_spacing, order=2):
    if len(imgs.shape) == 3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
        return imgs, true_spacing
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:, :, :, i]
            newslice, true_spacing = resample(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError('wrong shape')
