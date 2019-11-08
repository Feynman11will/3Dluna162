from configure  import  config
import os
import pandas
from multiprocessing import Pool
from functools import partial
import numpy as np

from loader import load_itk_image, lumTrans, resample, worldToVoxelCoord

from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image


def process_mask(mask):
    """
    :param mask: 输入的mask
    :return:
    """
    """
    1. 对每一个slices的2d图像进行处理
        "1. 获得当前层的凸包，如果当潜藏的凸包面积过大，或者没有没有mask，则放弃凸包的提取，直接赋值原始的图像
    2. 对当前凸包mask进行膨胀处理，获得膨胀凸包
    """

    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1 = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1) > 0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2) > 1.5 * np.sum(mask1):  # 凸包生成的凸多边形太过分，掩盖了原始mask1的大致形状信息，则放弃凸包处理
                mask2 = mask1
        else:  # 没有mask目标
            mask2 = mask1
        convex_mask[i_layer] = mask2

    struct = generate_binary_structure(3, 1)
    dilatedMask = binary_dilation(convex_mask, structure=struct, iterations=10)
    return dilatedMask


def savenpy_luna(id, annos, filelist, luna_segment, luna_data, savepath):
    """
    与kaggle stage数据集的原理没什么差异
    :param id:
    :param annos:
    :param filelist:
    :param luna_segment:
    :param luna_data:
    :param savepath:
    :return:
    """
    islabel = True
    isClean = True
    resolution = np.array([1, 1, 1])
    """
    1. 加载Mask模板，得到两个片肺的mask,通过掩码来获得,对mask进行腐蚀膨胀操作；确定掩码的边界box,使用重采样后形状扩张，最终得到extendbox
    2. 样本处理
        2.1. 对mask进行凸包膨胀处理，得到两个肺的mask，m1 和m2 为两个肺部的mask, dilatedMask = dm1+dm2
        2.2. 异或操作^ 相当于p1!=p2
        2.3. 加载原始图像，
        2.4. 加载原始图像
        2.5. 阈值归一化到[0,255]
        2.6. 对mask内部的图片进行读取，对mask外部使用0进行填充，同时对骨头进行填充因为既有肺部组织，也有骨质
        2.7. 对图像进行重采样，分辨率为[1,1,1]
        2.8. 目标范围缩小在肺部
        2.9. 获取膨胀后、重采样的的bbox，对bbox进行重新剪裁保存在'_clean.npy'
    3. 标签处理
        3.1. 获取一个病例的多个肺结节标注
        3.2. 标签处理
            1. 对每一个多目标标签，进行世界坐标到相对坐标转换；翻转操作，得到label列表
            2. 如没有目标，标签值为全零
            3. 如果有目标：将目标框重采样，限制坐标到extendbox中，
            4. 保存
    4. 问题综合
        预处理将范围限定在肺部区域
        一个肺有多个肺结节
        肺结节的标签限定在肺部区域，一个标签中有多个目标
    """
    name = filelist[id]
    if os.path.join(name + '_clean.npy') in os.listdir(savepath):
        print('%s is already in savepath %s'%(name ,savepath))
        return

    Mask, origin, spacing, isflip = load_itk_image(os.path.join(luna_segment, name + '.mhd'))

    if isflip:
        Mask = Mask[:, ::-1, ::-1]
    newshape = np.round(np.array(Mask.shape) * spacing / resolution).astype('int')  # 获取mask在新分辨率下的尺寸

    m1 = Mask == 3  # LUNA16的掩码有两种值3和4？？？
    m2 = Mask == 4
    Mask = m1 + m2  # 将两种掩码合并

    xx, yy, zz = np.where(Mask)
    box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
    box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack(
        [np.max([[0, 0, 0], box[:, 0] - margin], 0), np.min([newshape, box[:, 1] + 2 * margin], axis=0).T]).T

    this_annos = np.copy(annos[annos[:, 0] == str(name)])

    if isClean:
        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1 + dm2
        Mask = m1 + m2

        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170
        '''

        '''
        sliceim, origin, spacing, isflip = load_itk_image(os.path.join(luna_data, name + '.mhd'))

        if isflip:
            sliceim = sliceim[:, ::-1, ::-1]
            print('flip!')
        sliceim = lumTrans(sliceim)
        sliceim = sliceim * dilatedMask + pad_value * (1 - dilatedMask).astype('uint8')
        bones = (sliceim * extramask) > bone_thresh
        sliceim[bones] = pad_value

        sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
        sliceim2 = sliceim1[extendbox[0, 0]:extendbox[0, 1],
                   extendbox[1, 0]:extendbox[1, 1],
                   extendbox[2, 0]:extendbox[2, 1]]
        sliceim = sliceim2[np.newaxis, ...]
        np.save(os.path.join(savepath, name + '_clean.npy'), sliceim)

    if islabel:
        this_annos = np.copy(annos[annos[:, 0] == str(name)])
        label = []
        if len(this_annos) > 0:
            for c in this_annos:
                pos = worldToVoxelCoord(c[1:4][::-1], origin=origin, spacing=spacing)  # 将世界坐标转换为真实的体素坐标
                if isflip:
                    pos[1:] = Mask.shape[1:3] - pos[1:]
                label.append(np.concatenate([pos, [c[4] / spacing[1]]]))  # 这1段dataframe的操作没看懂？？？

        label = np.array(label)
        if len(label) == 0:
            label2 = np.array([[0, 0, 0, 0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)  # 对标签应用新的分辨率
            label2[3] = label2[3] * spacing[1] / resolution[1]  # 对直径应用新的分辨率
            label2[:3] = label2[:3] - np.expand_dims(extendbox[:, 0], 1)  # 将box外的长度砍掉，也就是相对于box的坐标
            label2 = label2[:4].T
        np.save(os.path.join(savepath, name + '_label.npy'), label2)
    print(name)



def preprocess_luna(n_workers,luna_segment,savepath,luna_data,luna_label):
    """
    同时操作多个文件目录（data_path下生成的filelist），实现多进程multiprocessing，处理的预处理函数就是savenpy_luna
    :return:
    """

    finished_flag = '.flag_preprocessluna'
    print('starting preprocessing luna')
    # if not os.path.exists(finished_flag): # 先不管结束标志对代码的运行影响
    filelist = [f.split('.mhd')[0] for f in os.listdir(luna_data) if f.endswith('.mhd') ]
    annos = np.array(pandas.read_csv(luna_label))
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    pool = Pool(n_workers)


    partial_savenpy_luna = partial(savenpy_luna,annos=annos,filelist=filelist,
                                   luna_segment=luna_segment,luna_data=luna_data,savepath=savepath)

    N = len(filelist) # N=89
    # for id in range(N):
    #     savenpy_luna(id, annos = annos, filelist = filelist,luna_segment = luna_segment, luna_data = luna_data, savepath = savepath)

    print('file list length is %d'%N)
    _ = pool.map(partial_savenpy_luna, range(1, N))
    pool.close()# 关闭线程池
    pool.join()

    print('end preprocessing luna')
    return  filelist

def extractor(n_workers = 6):

    from tqdm import tqdm
    luna_segment = config['luna_segment']
    savepath = config['preprocess_result_path']
    luna_label = config['luna_label']
    print(luna_label)
    print(savepath)
    print(luna_segment)
    luna_data_parent_path = config['luna_data_parent_path']
    luna_data_subset = config['luna_data_subset']

    pbar = tqdm(luna_data_subset)

    for subset in pbar:
        luna_data = os.path.join(luna_data_parent_path, subset)
        pbar.set_description("Processing %s" % luna_data)
        preprocess_luna(n_workers=n_workers,luna_segment = luna_segment, savepath=savepath, luna_data=luna_data, luna_label=luna_label)


if __name__ =='__main__':
    extractor()