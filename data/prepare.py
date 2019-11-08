# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.io import loadmat
# import h5py
from scipy.ndimage.interpolation import zoom
from skimage import measure
import warnings
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from skimage.morphology import convex_hull_image
from multiprocessing import Pool
from functools import partial

from loader import worldToVoxelCoord, resample, lumTrans
from step1 import step1_python
import warnings
import pandas
from configure import config


def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1 = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1) > 0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2) > 2 * np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3, 1)
    dilatedMask = binary_dilation(convex_mask, structure=struct, iterations=10)
    return dilatedMask


# def savenpy(id):
id = 1
def savenpy(id, filelist, prep_folder, data_path, use_existing=True, annos=None):
    """
    stage数据集处理并保存为numpy的格式（单个filelist对应一个npy文件，包括label.npy和clean.npy）
    :param id: os.listdir的list的index
    :param filelist: os.listdir的结果，list格式
    :param prep_folder: stage-2数据集处理过后的保存地址
    :param data_path: stage-2数据集的下载地址
    :param use_existing:
    :return:
    """
    resolution = np.array([1, 1, 1])
    name = filelist[id]
    print(id)
    if use_existing:
        if os.path.exists(os.path.join(prep_folder, name + '_label.npy')) and os.path.exists(
                os.path.join(prep_folder, name + '_clean.npy')):
            print(name + ' had been done')
            return
    try:
        # step1.py中的3D图像数据预处理函数step1_python，用来返回：
        ##im：像素点的HU值
        ##m1：mask以内的像素
        ##m2：mask以外的像素
        ##spacing：像素距离
        im, m1, m2, spacing, origin, isflip = step1_python(os.path.join(data_path, name))

        ## 真实的mask边界
        Mask = m1 + m2
        ## mask世界CT坐标系到真实坐标系的转换
        newshape = np.round(np.array(Mask.shape) * spacing / resolution)
        ## mask真实坐标系下的位置坐标
        xx, yy, zz = np.where(Mask)
        ## 结节标注的边界，用一个刚好包裹下的cube-box代替，并padding部分像素
        box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
        box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
        box = np.floor(box).astype('int')
        margin = 5
        extendbox = np.vstack(
            [np.max([[0, 0, 0], box[:, 0] - margin], 0), np.min([newshape, box[:, 1] + 2 * margin], axis=0).T]).T
        extendbox = extendbox.astype('int')

        # 凸包+扩张情况下的mask提取，并把mask以外的像素灰度值均设为170，为避免肺结节分类干扰，扩张区域内的像素灰度值高于210（骨组织）也设为170
        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1 + dm2
        Mask = m1 + m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        im[np.isnan(im)] = -2000
        sliceim = lumTrans(im)
        sliceim = sliceim * dilatedMask + pad_value * (1 - dilatedMask).astype('uint8')
        bones = sliceim * extramask > bone_thresh
        sliceim[bones] = pad_value
        sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
        sliceim2 = sliceim1[extendbox[0, 0]:extendbox[0, 1],
                   extendbox[1, 0]:extendbox[1, 1],
                   extendbox[2, 0]:extendbox[2, 1]]
        sliceim = sliceim2[np.newaxis, ...]
        np.save(os.path.join(prep_folder, name + '_clean'), sliceim)
        # np.save(os.path.join(prep_folder,name+'_label'),np.array([[0,0,0,0]]))

        this_annos = np.copy(annos[annos[:, 0] == str(name)])  # 一行代表一个结节，一个病例可能对应多行标签
        label = []
        if len(this_annos) > 0:
            for c in this_annos:
                pos = worldToVoxelCoord(c[1:4][::-1], origin=origin, spacing=spacing)  # 将世界坐标转换为真实的体素坐标
                if isflip:
                    pos[1:] = Mask.shape[1:3] - pos[1:]
                label.append(np.concatenate([pos, [c[4] / spacing[1]]]))  # 这1段dataframe的操作没看懂？？？

        label = np.array(label)
        if len(label) == 0:
            label2 = np.array([[0, 0, 0, 0]])  # #若没有结节则设为全0，第1次读取label_numpy就是全设为0的问题
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)  # 对标签应用新的分辨率
            label2[3] = label2[3] * spacing[1] / resolution[1]  # 对直径应用新的分辨率
            label2[:3] = label2[:3] - np.expand_dims(extendbox[:, 0], 1)  # 将box外的长度砍掉，也就是相对于box的坐标
            label2 = label2[:4].T
        np.save(os.path.join(prep_folder, name + '_label.npy'), label2)
        print(name, '\n')
    except:
        print('bug in ' + name)
        raise
    print(name + ' done')


def full_prep(data_path, prep_folder, n_worker=6, use_existing=True, annos=None):
    """
    同时操作多个文件目录（data_path下生成的filelist），实现多进程multiprocessing，处理的预处理函数就是savenpy
    :param data_path: stage-2数据集的下载地址
    :param prep_folder: stage-2数据集处理过后的保存地址
    :param n_worker:
    :param use_existing:
    :return:
    """

    if (annos.all == None):
        print('未有输入标签')
        raise

    warnings.filterwarnings("ignore")
    if not os.path.exists(prep_folder):
        os.mkdir(prep_folder)

    print('starting preprocessing')
    pool = Pool(n_worker)
    filelist = [f for f in os.listdir(data_path)]
    filelist = [f for f in filelist if f.endswith('.mhd')]

    # 高阶嵌套函数的调用，用于固定1个或多个初始值，返回的是一个可调用的partial对象，partial(func,*args,**kw)
    partial_savenpy = partial(savenpy, filelist=filelist, prep_folder=prep_folder,
                              data_path=data_path, use_existing=use_existing, annos=annos)

    # for id, path in enumerate(filelist):
    #     if id==0:
    #         continue
    #     if id< 10:
    #         savenpy(id, filelist=filelist,prep_folder=prep_folder,data_path=data_path,use_existing=use_existing,annos=annos)
    #     else :
    #         break
    N = len(filelist)
    _ = pool.map(partial_savenpy, range(1, N))
    pool.close()
    pool.join()
    print('end preprocessing')
    return filelist


if __name__ == '__main__':
    luna_label = config['luna_label']
    prep_folder = config['preprocess_result_path']
    data_path = config['luna_segment']
    annos = np.array(pandas.read_csv(luna_label))
    full_prep(data_path, prep_folder, n_worker=6, use_existing=True, annos=annos)