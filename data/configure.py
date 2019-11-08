# -*- coding: utf-8 -*-
'''
1. preprocess_result_path 存储npy文件输出路径
2. luna_segment : 肺部组织的mask
3. luna_label : annotations 文件路径
4. luna_data_parent_path : 原始数据路径父路径
5. luna_data_subset：luna_data_parent_path下文件的子路径子集和
'''
config = {
            'luna_segment':'/data2/wangll/02Dataset/LUNA2016_data/seg-lungs-LUNA16/',
            'preprocess_result_path':'/data2/wangll/02Dataset/LUNA2016tmp',
            'luna_label':'/data2/wangll/02Dataset/LUNA2016_data/CSVFILES/annotations.csv',
            'luna_data_parent_path':'/data2/wangll/02Dataset/LUNA2016_data',
            'luna_data_subset':['subset0','subset1','subset2','subset3','subset4','subset5','subset6','subset7','subset8','subset9'],
            'train_list':'../../data/train_list.npy',
            'val_list':'../../data/val_list.npy'
         }


