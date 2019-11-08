import os
import sys
from math import floor

import numpy as np
sys.path.append('../')

from data.configure import config



# full_list_path = config["preprocess_result_path"]

def split_path(full_list_path, split_ratio = 0.1):
    full_list = os.listdir(full_list_path)
    full_list = [ f.split('_')[0] for f in full_list if f.split('_')[1]!='label.npy' ]
    full_list = np.array(full_list)
    print(full_list)
    np.save('full_lsit.npy', full_list)

    l = len(full_list)
    val_length = floor(split_ratio*l)
    train_length = l-val_length
    train_list = []
    val_list = []

    for idx in range(train_length):
        train_list.append(full_list[idx])
    train_list = np.array(train_list)
    np.save('train_list.npy', train_list)

    for idx in range(train_length+ 1,l):
        val_list.append(full_list[idx])
    val_lsit = np.array(val_list)
    np.save('val_list.npy', val_lsit)


if __name__=='__main__':
    full_list_path = config["preprocess_result_path"]
    split_path(full_list_path =full_list_path)

    train_list = np.load('train_list.npy')
    val_list = np.load('val_list.npy')

    print(train_list)
    print('\n')
    print('\n')
    print(val_list)

    print('val_list')
