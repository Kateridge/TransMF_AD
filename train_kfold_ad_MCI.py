import glob
import random
import numpy as np
import torch
from datasets.ADNI import ADNI, ADNI_transform
from models.mymodel import model_CNN, model_ad
from options.option import Option
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
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

if __name__ == '__main__':
    device = torch.device('cuda:{}'.format(0))
    # initialize options and create output directory
    opt = Option().parse()
    save_dir = os.path.join('./checkpoints', opt.name)
    # load ADNI dataset
    ADNI_MCI = ADNI(dataroot='/home/kateridge/Projects/Projects/Datasets/ADNI_SPM',
                label_filename='ADNI.csv', task=opt.task)
    train_transforms, val_transforms = ADNI_transform()
    ADNI_MCI_dataset = Dataset(data=ADNI_MCI.data_dict, transform=train_transforms)
    ADNI_ADCN = ADNI(dataroot='/home/kateridge/Projects/Projects/Datasets/ADNI_SPM', label_filename='ADNI.csv', task='ADCN')
    ADNI_MCI_len = len(ADNI_MCI_dataset)
    ADCN_len = len(ADNI_ADCN)
    # print(ADNI_MCI.data_dict.extend(ADNI_ADCN.data_dict))
    ADNI_train_dataset = Dataset(data=ADNI_MCI.data_dict + ADNI_ADCN.data_dict, transform=train_transforms)
    ADNI_test_dataset = Dataset(data=ADNI_MCI.data_dict + ADNI_ADCN.data_dict, transform=val_transforms)

    # prepare kfold splits
    num_fold = 5
    kfold_splits = KFold(n_splits=num_fold, shuffle=True, random_state=42)

    # get dataloaders according to splits
    def setup_dataflow(train_dataset, test_dataset, train_idx, test_idx):
        # further split training set and validation set
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)
        extra_idx = [x + ADNI_MCI_len for x in range(ADCN_len)]
        train_idx = np.concatenate([train_idx, np.array(extra_idx)], axis=0)

        # create subsampler
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, sampler=train_sampler, drop_last=True)
        val_loader = DataLoader(test_dataset, batch_size=opt.batch_size, sampler=val_sampler)
        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, sampler=test_sampler)

        # weights = get_dataset_weights(dataset, train_idx)
        return train_loader, val_loader, test_loader


    # initialize model, optimizer, loss
    def init_model(model):
        net_model = None
        if model == 'Transformer':
            net_model = model_ad(dim=opt.dim, depth=opt.trans_enc_depth, heads=8,
                                 dim_head=opt.dim // 8, mlp_dim=opt.dim * 4, dropout=opt.dropout).to(device)
        elif model == 'CNN':
            net_model = model_CNN(dim=opt.dim).to(device)
        return net_model


    def train_model(train_dataloader, val_dataloader, test_dataloader, fold):
        # create fold checkpoint directory
        save_path_fold = os.path.join(save_dir, str(fold))
        mkdirs(save_path_fold)
        logger = Logger(save_path_fold)
        # initialize model, optimizer and loss
        net_model = init_model('Transformer')
        optimizer, lr_schedualer = getOptimizer(net_model.parameters(), opt)
        # criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
        criterion = torch.nn.CrossEntropyLoss()
        res_fold = []

        # define train step
        def train_step(engine, batch):
            output_dic = {}
            # set model to train mode
            net_model.train()
            # decompose batch data
            MRI = batch['MRI'].to(device)
            PET = batch['PET'].to(device)
            label = batch['label'].to(device)
            output_dic['label'] = label
            # zero grad
            optimizer.zero_grad()

            # forward
            output_logits, D_MRI_logits, D_PET_logits = net_model(MRI, PET)
            output_dic['logits'] = output_logits
            output_dic['D_MRI_logits'] = D_MRI_logits
            output_dic['D_PET_logits'] = D_PET_logits

            ce_loss = criterion(output_logits, label)
            mri_gt = torch.ones([D_MRI_logits.shape[0]], dtype=torch.int64).to(MRI.device)
            pet_gt = torch.zeros([D_PET_logits.shape[0]], dtype=torch.int64).to(PET.device)

            output_dic['D_MRI_label'] = mri_gt
            output_dic['D_PET_label'] = pet_gt
            ad_loss = (criterion(D_MRI_logits, mri_gt) + criterion(D_PET_logits, pet_gt)) / 2
            # print(D_logits)
            output_dic['ce_loss'] = ce_loss.item()
            output_dic['ad_loss'] = ad_loss.item()

            # backward
            all_loss = ad_loss + ce_loss
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
                PET = batch['PET'].to(device)
                label = batch['label'].to(device)
                output_dic['label'] = label

                # forward
                output_logits, _, _ = net_model(MRI, PET)
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
                         "MRI_accuracy": Accuracy(output_transform=lambda x: [x['D_MRI_logits'], x['D_MRI_label']]),
                         "PET_accuracy": Accuracy(output_transform=lambda x: [x['D_PET_logits'], x['D_PET_label']]),
                         "ce_loss": Average(output_transform=lambda x: x['ce_loss']),
                         "ad_loss": Average(output_transform=lambda x: x['ad_loss'])}
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
            logger.print_message(f"ce_loss: {metrics['ce_loss']:.4f} "
                                 f"ad_loss: {metrics['ad_loss']:.4f} "
                                 f"accuracy: {metrics['accuracy']:.4f} "
                                 f"MRIaccuracy: {metrics['MRI_accuracy']:.4f} "
                                 f"PETaccuracy: {metrics['PET_accuracy']:.4f} ")

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
    for fold_idx, (train_idx, test_idx) in enumerate(kfold_splits.split(np.arange(len(ADNI_MCI_dataset)))):
        logger_main.print_message(f'************Fold {fold_idx}************')
        train_dataloader, val_dataloader, test_dataloader = setup_dataflow(ADNI_train_dataset, ADNI_test_dataset, train_idx, test_idx)
        results.append(train_model(train_dataloader, val_dataloader, test_dataloader, fold_idx))
        # log best metric every fold
        # logger_main.print_message(f'The best metrics for Fold {fold_idx} :')
        # sen, spe, f1 = cal_confusion_metrics(best_metrics['confusion'])
        # logger_main.print_message(f"loss: {best_metrics['loss']:.4f} accuracy: {best_metrics['accuracy']:.4f} "
        #                           f"sensitivity: {sen:.4f} specificity: {spe:.4f} "
        #                           f"f1 score: {f1:.4f} AUC: {best_metrics['auc']:.4f} ")
        # results.append([best_metrics['loss'], best_metrics['accuracy'], sen, spe, f1, best_metrics['auc']])
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
