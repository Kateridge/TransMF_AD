import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from options.option import Option
from datasets import get_dataset
from utils.utils import getOptimizer, cal_confusion_metrics, Logger
from models.mymodel import stage1_network, stage2_network, model_pretrain
from monai.data import DataLoader, Dataset
from ignite.metrics import Accuracy, Loss, Average, ConfusionMatrix
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.handlers import Checkpoint, global_step_from_engine, DiskSaver, create_lr_scheduler_with_warmup
import torch, ignite
from torch.nn.functional import softmax, adaptive_max_pool1d
from glob import glob
from einops import rearrange

if __name__ == '__main__':
    # get device GPU
    device = torch.device('cuda:{}'.format(0))

    # get options
    opt = Option().parse()
    save_dir = os.path.join('./checkpoints', opt.name)
    logger = Logger(opt)

    # get datasets and dataloaders
    if opt.task == 'pretrain':
        train_dataloader = get_dataset(opt)
    else:
        train_dataloader, val_dataloader, test_dataloader = get_dataset(opt)
    # torch.save({'train': train_dataloader, 'val': val_dataloader, 'CNN_PET_ADCN': test_dataloader}, 'dataloaders.pt')

    # get networks
    net_model = model_pretrain(dim=opt.dim, cl_dim=opt.dim, t=0.07, depth=opt.trans_enc_depth, heads=4,
                               dim_head=opt.dim // 4, mlp_dim=opt.dim * 4, dropout=opt.dropout).to(device)

    logger.print_message('----------------- Model Param -------------------')
    logger.print_message('Model: %.2fM*2' % (sum([param.nelement() for param in net_model.parameters()]) / 1e6))
    logger.print_message('----------------- Train Log -------------------')

    # get optimizers
    optimizer, lr_schedualer = getOptimizer(net_model.parameters(), opt)

    # get loss
    criterion = torch.nn.CrossEntropyLoss()


    # define train steps
    def pretrain_step(engine, batch):
        output_dic = {}
        # set model to train mode
        net_model.train()

        # decompose batch data
        MRI = batch['MRI'].to(device)
        PET = batch['PET'].to(device)
        # age = batch['age'].to(device)
        # b, _, _, _, _ = MRI.shape
        # age = age.reshape(b, 1)

        # zero grad
        optimizer.zero_grad()

        # forward
        loss_cl, loss_match = net_model.pretrain_forward(MRI, PET)
        loss = 0.2 * loss_cl + 0.8 * loss_match
        output_dic['loss_cl'] = loss_cl.item()
        output_dic['loss_match'] = loss_match.item()
        output_dic['loss'] = -loss.item()

        # backward
        loss.backward()

        # update param
        optimizer.step()
        return output_dic


    trainer = Engine(pretrain_step)
    ProgressBar().attach(trainer)


    class one_hot_transform:
        def __init__(self, target):
            self.target = target

        def __call__(self, output):
            y_pred, y = output[self.target], output['label']
            y_pred = torch.argmax(y_pred, dim=1).long()
            y_pred = ignite.utils.to_onehot(y_pred, 2)
            y = y.long()
            return y_pred, y


    # stage1 metrics
    train_metrics = {"loss_cl": Average(output_transform=lambda x: x['loss_cl']),
                     "loss_match": Average(output_transform=lambda x: x['loss_match']),
                     "loss": Average(output_transform=lambda x: x['loss'])}

    # attach train metrics
    for name, metric in train_metrics.items():
        metric.attach(trainer, name)


    # logging for training every epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        metrics = trainer.state.metrics
        logger.print_message('-------------------------------------------------')
        curr_lr = optimizer.param_groups[0]['lr']
        logger.print_message(f'Current learning rate: {curr_lr}')
        logger.print_message(f"Training Results - Epoch[{trainer.state.epoch}] ")
        logger.print_message(
            f"Total_loss: {metrics['loss']:.4f} loss_cl: {metrics['loss_cl']:.4f} loss_match: {metrics['loss_match']:.4f}")


    # save model according to loss
    checkpoint_saver = Checkpoint({'net_model': net_model}, save_handler=DiskSaver(save_dir, require_empty=False),
                                  n_saved=1, filename_prefix='best', score_name='loss')
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_saver)


    # training with label #
    ############################################################################################################
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
        output_logits = net_model(MRI, PET)
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
    lr_schedualer_handler = create_lr_scheduler_with_warmup(lr_schedualer, warmup_start_value=opt.lr * 0.2,
                                                            warmup_end_value=opt.lr, warmup_duration=3)
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
            output_logits = net_model(MRI, PET)
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
    checkpoint_saver = Checkpoint({'net_model': net_model}, save_handler=DiskSaver(save_dir, require_empty=False),
                                  n_saved=1, filename_prefix='best_label', score_name='accuracy',
                                  global_step_transform=global_step_from_engine(trainer_label))
    evaluator.add_event_handler(Events.COMPLETED, checkpoint_saver)


    @trainer_label.on(Events.STARTED)
    def load_best_pretrain_model(trainer_label):
        best_model_path = glob(os.path.join(save_dir, 'best_net_model_*.pt'))[0]
        checkpoint_all = torch.load(best_model_path, map_location=device)
        Checkpoint.load_objects(to_load={'net_model': net_model}, checkpoint=checkpoint_all)
        logger.print_message(f'Load best model {best_model_path}')

    @trainer_label.on(Events.COMPLETED)
    def run_on_test(trainer_label):
        best_model_path = glob(os.path.join(save_dir, 'best_label_net_model_*.pt'))[0]
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

    # run trainer
    if opt.mode == 'pretrain':
        trainer.run(train_dataloader, opt.stage1_epochs + opt.stage2_epochs)
    if opt.mode == 'train':
        trainer_label.run(train_dataloader, opt.stage1_epochs + opt.stage2_epochs)



