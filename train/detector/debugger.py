from torch.utils.data import DataLoader
from tqdm import trange, tqdm

import data_set

from data.configure import config as config_training

from configure.general import config_res18 as config

if __name__=='__main__':

    datadir = config_training['preprocess_result_path']
    dataset_train = data_set.DataBowl3Detector(
            datadir,
            config_training['train_list'],  # fix
            config,
            phase='train')

    train_loader = DataLoader(
        dataset_train,
        batch_size=8,
        shuffle=True,
        num_workers=1,
        collate_fn=data_set.collate_train,
        pin_memory=True)

    pbar = tqdm(train_loader)
    for i in trange(10):
        for j ,(data, target, coord) in enumerate(pbar):
            print('{}th batch in the {}th epoch:\n data shape is{}\n data shape is{}\n data shape is{}\n  '.format(j,i,data.shape,target.shape,coord.shape))
