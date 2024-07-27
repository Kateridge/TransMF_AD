import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from options.option import Option
from datasets import get_dataset
from utils.utils import getOptimizer, cal_confusion_metrics, Logger
from models.mymodel import model_ad
from ignite.metrics import Accuracy, Loss, Average, ConfusionMatrix
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.handlers import Checkpoint, global_step_from_engine, DiskSaver, create_lr_scheduler_with_warmup, LRScheduler
import torch, ignite
from torch.nn.functional import softmax
from glob import glob

if __name__ == '__main__':
    # get device GPU
    device = torch.device('cuda:{}'.format(0))

    # get options
    opt = Option().parse()
    save_dir = os.path.join('./checkpoints', opt.name)
    logger = Logger(save_dir)

    # get datasets and dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataset(opt)

    # get networks
    net_model = model_ad(dim=opt.dim, depth=opt.trans_enc_depth, heads=8,
                         dim_head=opt.dim // 8, mlp_dim=opt.dim * 4, dropout=opt.dropout).to(device)

    logger.print_message('----------------- Model Param -------------------')
    logger.print_message('Model: %.2fM' % (sum([param.nelement() for param in net_model.parameters()]) / 1e6))
    logger.print_message('----------------- Train Log -------------------')

    # get optimizers
    optimizer, lr_schedualer = getOptimizer(net_model.parameters(), opt)

    # get loss
    criterion = torch.nn.CrossEntropyLoss()

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
            ce_loss = criterion(output_logits, label)
            output_dic['loss'] = ce_loss.item()
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
    checkpoint_saver = Checkpoint({'net_model': net_model}, save_handler=DiskSaver(save_dir, require_empty=False),
                                  n_saved=1, filename_prefix='best_label', score_name='accuracy',
                                  global_step_transform=global_step_from_engine(trainer_label),
                                  greater_or_equal=True)
    evaluator.add_event_handler(Events.COMPLETED, checkpoint_saver)


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


    trainer_label.run(train_dataloader, opt.stage1_epochs + opt.stage2_epochs)
