import os
import torch
import numpy as np
from torch.utils.data import SubsetRandomSampler
import itertools


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def getOptimizer(net_para, opt):
    # return torch.optim.Adam([
    #     {'params': (p for name, p in net.named_parameters() if 'bias' not in name),
    #      'weight_decay': weight_decay},
    #     {'params': (p for name, p in net.named_parameters() if 'bias' in name)}])
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net_para, lr=opt.lr, weight_decay=opt.weight_decay)
        lr_schedualer = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 26], gamma=0.1)
        return optimizer, lr_schedualer
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net_para, lr=opt.lr, weight_decay=opt.weight_decay)
        lr_schedualer = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 36], gamma=0.1)
        return optimizer, lr_schedualer


def cal_confusion_metrics(c_matrix):
    TP, FN, FP, TN = c_matrix[1, 1], c_matrix[1, 0], c_matrix[0, 1], c_matrix[0, 0]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    sen = TP / (TP + FN)
    spe = TN / (FP + TN)
    return sen, spe, f1


# split the input dataset to training set and validation set
def dataset_random_split(dataset, collate_fn, val_radio=.2, batch_size=1):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_radio * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                                               collate_fn=collate_fn)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler,
                                                    collate_fn=collate_fn)
    return train_loader, validation_loader, len(train_indices), len(val_indices)


def get_dataset_weights(dataset, train_idx):
    count_0 = 0
    count_1 = 0
    data = dataset.data
    for idx in range(len(dataset)):
        if data[idx]['label'] == 0:
            count_0 += 1
        elif data[idx]['label'] == 1:
            count_1 += 1
    weights = torch.FloatTensor([1/count_0, 1/count_1])
    print(f'negative class has {count_0} samples')
    print(f'positive class has {count_1} samples')
    return weights


class Logger():
    def __init__(self, log_dir):
        # create a logging file to store training losses
        self.log_name = os.path.join(log_dir, 'log.txt')
        with open(self.log_name, "a") as log_file:
            log_file.write(f'================ {self.log_name} ================\n')

    def print_message(self, msg):
        print(msg)
        with open(self.log_name, 'a') as log_file:
            log_file.write('%s\n' % msg)

    def print_message_nocli(self, msg):
        with open(self.log_name, 'a') as log_file:
            log_file.write('%s\n' % msg)
