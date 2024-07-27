import glob

import numpy as np
import torch
from datasets.ADNI import ADNI, ADNI_transform
from models.mymodel import model_single
from options.option import Option
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from monai.data import Dataset
import ignite
from ignite.metrics import Accuracy, Loss, Average, ConfusionMatrix
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.handlers import Checkpoint, global_step_from_engine, DiskSaver, create_lr_scheduler_with_warmup, LRScheduler
from utils.utils import getOptimizer, cal_confusion_metrics, mkdirs, get_dataset_weights
from torch.nn.functional import softmax
from utils.utils import Logger
import os
import random

if __name__ == '__main__':
    device = torch.device('cuda:{}'.format(0))
    # initialize options and create output directory
    opt = Option().parse()
    save_dir = os.path.join('./checkpoints', opt.name)
    # load ADNI dataset
    ADNI_data = ADNI(dataroot='/home/kateridge/Projects/Projects/Datasets/ADNI',
                     label_filename='ADNI.csv', task=opt.task).data_dict
    train_transforms, val_transforms = ADNI_transform(opt.aug)

    # prepare kfold splits
    num_fold = 5
    seed = 1
    if opt.task == 'ADCN':
        seed = 42
    elif opt.task == 'pMCIsMCI':
        seed = 996
    if opt.randint == 'True':
        seed = random.randint(1, 1000)
    print(f'The random seed is {seed}')
    kfold_splits = KFold(n_splits=num_fold, shuffle=True, random_state=seed)


    # get dataloaders according to splits
    def setup_dataflow(train_idx, test_idx):
        # further split training set and validation set
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=seed)

        train_data = [ADNI_data[i] for i in train_idx.tolist()]
        val_data = [ADNI_data[i] for i in val_idx.tolist()]
        test_data = [ADNI_data[i] for i in test_idx.tolist()]

        if opt.task == 'pMCIsMCI' and opt.extra_sample == 'True':
            ADNI_ADCN_data = ADNI(dataroot=opt.dataroot, label_filename='ADNI.csv', task='ADCN').data_dict
            train_data += ADNI_ADCN_data

        # create datasets
        train_dataset = Dataset(data=train_data, transform=train_transforms)
        val_dataset = Dataset(data=val_data, transform=val_transforms)
        test_dataset = Dataset(data=test_data, transform=val_transforms)
        print(f'Train Datasets: {len(train_dataset)}')
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size)

        weights = get_dataset_weights(train_dataset, train_idx)
        print(f'Val Datasets: {len(val_dataset)}')
        print(f'Test Datasets: {len(test_dataset)}')
        return train_loader, val_loader, test_loader, weights

    # initialize model, optimizer, loss
    def init_model(model):
        net_model = model_single(opt.dim).to(device)
        return net_model


    def train_model(train_dataloader, val_dataloader, test_dataloader, fold, weights):
        # create fold checkpoint directory
        save_path_fold = os.path.join(save_dir, str(fold))
        mkdirs(save_path_fold)
        logger = Logger(save_path_fold)
        # initialize model, optimizer and loss
        net_model = init_model('Single')
        optimizer, lr_schedualer = getOptimizer(net_model.parameters(), opt)
        # criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
        criterion = torch.nn.CrossEntropyLoss()

        # define train step
        def train_step(engine, batch):
            output_dic = {}
            # set model to train mode
            net_model.train()
            # decompose batch data
            MRI = batch['MRI'].to(device)
            label = batch['label'].to(device)
            output_dic['label'] = label
            # zero grad
            optimizer.zero_grad()

            # forward
            output_logits = net_model(MRI)
            output_dic['logits'] = output_logits
            all_loss = criterion(output_logits, label)
            output_dic['loss'] = all_loss.item()

            # backward
            all_loss.backward()

            # update param
            optimizer.step()
            return output_dic

        trainer_label = Engine(train_step)
        ProgressBar().attach(trainer_label)
        # lr_schedualer_handler = create_lr_scheduler_with_warmup(lr_schedualer, warmup_start_value=opt.lr * 0.1,
        #                                                         warmup_end_value=opt.lr, warmup_duration=3)
        # trainer_label.add_event_handler(Events.EPOCH_STARTED, lr_schedualer_handler)
        lr_schedualer_handler = LRScheduler(lr_schedualer)
        trainer_label.add_event_handler(Events.EPOCH_STARTED, lr_schedualer_handler)

        # define validation step
        def val_step(engine, batch):
            output_dic = {}
            # set model to val mode
            net_model.eval()

            with torch.no_grad():
                # decompose batch data
                MRI = batch['MRI'].to(device)
                label = batch['label'].to(device)
                output_dic['label'] = label

                # forward
                output_logits = net_model(MRI)
                output_dic['logits'] = output_logits
                all_loss = criterion(output_logits, label)
                output_dic['loss'] = all_loss.item()
                return output_dic

        evaluator = Engine(val_step)
        ProgressBar().attach(evaluator)

        class one_hot_transform:
            def __init__(self, target):
                self.target = target

            def __call__(self, output):
                y_pred, y = output[self.target], output['label']
                y_pred = torch.argmax(y_pred, dim=1).long()
                y_pred = ignite.utils.to_onehot(y_pred, 2)
                y = y.long()
                return y_pred, y

        # metrics
        train_metrics = {"accuracy": Accuracy(output_transform=lambda x: [x['logits'], x['label']]),
                         "loss": Average(output_transform=lambda x: x['loss'])}
        val_metrics = {"accuracy": Accuracy(output_transform=lambda x: [x['logits'], x['label']]),
                       "confusion": ConfusionMatrix(num_classes=2,
                                                    output_transform=one_hot_transform(target='logits')),
                       "auc": ROC_AUC(output_transform=lambda x: [softmax(x['logits'], dim=1)[:, -1], x['label']]),
                       "loss": Loss(criterion, output_transform=lambda x: [x['logits'], x['label']])}

        # attach metrics
        for name, metric in train_metrics.items():
            metric.attach(trainer_label, name)

        for name, metric in val_metrics.items():
            metric.attach(evaluator, name)

        # logging for training every epoch
        @trainer_label.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer_label):
            metrics = trainer_label.state.metrics
            logger.print_message('-------------------------------------------------')
            curr_lr = optimizer.param_groups[0]['lr']
            logger.print_message((f'Current learning rate: {curr_lr}'))
            logger.print_message(f"Training Results - Epoch[{trainer_label.state.epoch}] ")
            logger.print_message(f"loss: {metrics['loss']:.4f} accuracy: {metrics['accuracy']:.4f}")

        # logging for validation every epochw
        @trainer_label.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer_label):
            evaluator.run(val_dataloader)
            metrics = evaluator.state.metrics
            logger.print_message(f"Validation Results - Epoch[{trainer_label.state.epoch}] ")
            sen, spe, f1 = cal_confusion_metrics(metrics['confusion'])
            logger.print_message(f"loss: {metrics['loss']:.4f} accuracy: {metrics['accuracy']:.4f} "
                                 f"sensitivity: {sen:.4f} specificity: {spe:.4f} "
                                 f"f1 score: {f1:.4f} AUC: {metrics['auc']:.4f} ")

        # save model according to ACC or AUC
        checkpoint_saver = Checkpoint({'net_model': net_model},
                                      save_handler=DiskSaver(save_path_fold, require_empty=False),
                                      n_saved=1, filename_prefix='best_label', score_name='accuracy',
                                      global_step_transform=global_step_from_engine(trainer_label),
                                      greater_or_equal=True)
        evaluator.add_event_handler(Events.COMPLETED, checkpoint_saver)

        @trainer_label.on(Events.COMPLETED)
        def run_on_test(trainer_label):
            best_model_path = glob.glob(os.path.join(save_path_fold, 'best_label_net_model_*.pt'))[0]
            checkpoint_all = torch.load(best_model_path, map_location=device)
            Checkpoint.load_objects(to_load={'net_model': net_model}, checkpoint=checkpoint_all)
            logger.print_message(f'Load best model {best_model_path}')
            # detatch checkpoint saver handler
            evaluator.remove_event_handler(checkpoint_saver, Events.COMPLETED)
            # run evaluator
            evaluator.run(test_dataloader)
            metrics = evaluator.state.metrics
            logger.print_message('**************************************************************')
            logger.print_message(f"Test Results")
            sen, spe, f1 = cal_confusion_metrics(metrics['confusion'])
            logger.print_message(f"loss: {metrics['loss']:.4f} accuracy: {metrics['accuracy']:.4f} "
                                 f"sensitivity: {sen:.4f} specificity: {spe:.4f} "
                                 f"f1 score: {f1:.4f} AUC: {metrics['auc']:.4f} ")
            res_fold = [metrics['loss'], metrics['accuracy'], sen, spe, f1, metrics['auc']]
            evaluator.state.res_fold = res_fold

        trainer_label.run(train_dataloader, opt.stage1_epochs + opt.stage2_epochs)

        return evaluator.state.res_fold


    results = []
    logger_main = Logger(save_dir)
    for fold_idx, (train_idx, test_idx) in enumerate(kfold_splits.split(np.arange(len(ADNI_data)))):
        logger_main.print_message(f'************Fold {fold_idx}************')
        train_dataloader, val_dataloader, test_dataloader, weights = setup_dataflow(train_idx, test_idx)
        results.append(train_model(train_dataloader, val_dataloader, test_dataloader, fold_idx, weights))

    # calculate mean and std for each metrics
    results = np.array(results)
    res_mean = np.mean(results, axis=0)
    res_std = np.std(results, axis=0)
    logger_main.print_message(f'************Final Results************')
    logger_main.print_message(f'loss: {res_mean[0]:.4f} +- {res_std[0]:.4f}\n'
                              f'acc: {res_mean[1]:.4f} +- {res_std[1]:.4f}\n'
                              f'sen: {res_mean[2]:.4f} +- {res_std[2]:.4f}\n'
                              f'spe: {res_mean[3]:.4f} +- {res_std[3]:.4f}\n'
                              f'f1: {res_mean[4]:.4f} +- {res_std[4]:.4f}\n'
                              f'auc: {res_mean[5]:.4f} +- {res_std[5]:.4f}\n')
