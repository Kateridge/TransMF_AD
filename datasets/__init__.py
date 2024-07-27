from abc import ABC, abstractmethod
from monai.data import CacheDataset
import sys
from datasets.ADNI import *
from monai.data import DataLoader, Dataset, partition_dataset
import numpy as np
import os
import random


@abstractmethod
class CVDataset(ABC, CacheDataset):
    """
    Base class to generate cross validation datasets.

    """

    def __init__(
            self,
            data,
            transform,
            cache_num=sys.maxsize,
            cache_rate=1.0,
            num_workers=0,
    ) -> None:
        data = self._split_datalist(datalist=data)
        CacheDataset.__init__(
            self, data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers
        )

    def _split_datalist(self, datalist):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


def get_dataset(opt):
    if opt.dataset == 'ADNI12':
        print('----------------- Dataset -------------------')
        print('Loading ADNI. Train on ADNI1 and CNN_PET_ADCN on ADNI2.....')
        # load filenames and labels
        ADNI1 = ADNI(dataroot='/home/kateridge/Projects/Projects/Datasets/ADNI', label_filename='ADNI1_modality_complete.csv', task=opt.task)
        ADNI2 = ADNI(dataroot='/home/kateridge/Projects/Projects/Datasets/ADNI', label_filename='ADNI2_modality_complete.csv', task=opt.task)
        train_transforms, test_transforms = ADNI_transform()
        # divide ADNI1 into training set and validation set
        ADNI1_partitions = partition_dataset(data=ADNI1.data_dict, ratios=[0.8, 0.2], shuffle=True)
        train_dataset, val_dataset = ADNI1_partitions[0], ADNI1_partitions[1]
        # save dataset partitions
        save_dataset_partition(train_dataset, val_dataset, ADNI2.data_dict, os.path.join(opt.checkpoints_dir, opt.name))
        # get datasets
        train_dataset = Dataset(data=train_dataset, transform=train_transforms)
        val_dataset = Dataset(data=val_dataset, transform=test_transforms)
        test_dataset = Dataset(data=ADNI2.data_dict, transform=test_transforms)
        print('The number of training images = %d' % len(train_dataset))
        print('The number of val images = %d' % len(val_dataset))
        print('The number of CNN_PET_ADCN images = %d' % len(test_dataset))
        # get dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        return train_dataloader, val_dataloader, test_dataloader
    elif opt.dataset == 'ADNI':
        print('----------------- Dataset -------------------')
        print('Loading ADNI1+ADNI2.....')
        if opt.task == 'pretrain':
            ADNI_ALL = ADNI(dataroot='/home/kateridge/Projects/Projects/Datasets/ADNI', label_filename='ADNI.csv', task='ADCN')
            train_transforms, test_transforms = ADNI_transform()
            ADNI_partitions = partition_dataset(data=ADNI_ALL.data_dict, ratios=[0.8, 0.2], shuffle=True, seed=965)
            train_dataset, val_dataset = ADNI_partitions[0], ADNI_partitions[1]
            train_dataset = Dataset(data=train_dataset, transform=train_transforms)
            val_dataset = Dataset(data=val_dataset, transform=test_transforms)
            print('The number of training images = %d' % len(train_dataset))
            print('The number of val images = %d' % len(val_dataset))
            train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
            val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
            return train_dataloader, val_dataloader
        # load filenames and labels
        ADNI_ALL = ADNI(dataroot='/home/kateridge/Projects/Projects/Datasets/ADNI', label_filename='ADNI.csv', task=opt.task)
        train_transforms, test_transforms = ADNI_transform()
        # divide ADNI into training set, validation set and CNN_PET_ADCN set
        ADNI_partitions = partition_dataset(data=ADNI_ALL.data_dict, ratios=[0.6, 0.2, 0.2], shuffle=True)
        train_dataset, val_dataset, test_dataset = ADNI_partitions[0], ADNI_partitions[1], ADNI_partitions[2]
        # if opt.task == 'pMCIsMCI':
        #     ADNI_ADCN = ADNI(dataroot='/home/kateridge/Projects/Projects/Datasets/ADNI_SPM', label_filename='ADNI.csv',
        #                     task='ADCN')
        #     train_dataset += ADNI_ADCN.data_dict
        # save dataset partitions
        save_dataset_partition(train_dataset, val_dataset, test_dataset, os.path.join(opt.checkpoints_dir, opt.name))
        # get datasets
        train_dataset = Dataset(data=train_dataset, transform=train_transforms)
        val_dataset = Dataset(data=val_dataset, transform=test_transforms)
        test_dataset = Dataset(data=test_dataset, transform=test_transforms)
        print('The number of training images = %d' % len(train_dataset))
        print('The number of val images = %d' % len(val_dataset))
        print('The number of CNN_PET_ADCN images = %d' % len(test_dataset))
        # get dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        return train_dataloader, val_dataloader, test_dataloader


def save_dataset_partition(train, val, test, path):
    # save dic to npy file
    np.save(os.path.join(path, 'train.npy'), train)
    np.save(os.path.join(path, 'val.npy'), val)
    np.save(os.path.join(path, 'CNN_PET_ADCN.npy'), test)
    # if need to load npy
    # train_dic = np.load('train.npy', allow_pickle='TRUE').item()
