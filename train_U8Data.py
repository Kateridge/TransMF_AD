import os

from models.networks import sNet_Original

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from options.option import Option
from datasets.U8Data import U8Data, U8Data_transform
from utils.utils import getOptimizer, cal_confusion_metrics
from models.mymodel import stage1_network, stage2_network, stage1_resnet
from monai.data import DataLoader, Dataset
from monai.data import partition_dataset
from ignite.metrics import Accuracy, Loss, Average, ConfusionMatrix
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.handlers import Checkpoint, global_step_from_engine, DiskSaver, create_lr_scheduler_with_warmup, LRScheduler
import torch, ignite
from torch.nn.functional import softmax
from glob import glob
from utils.utils import getOptimizer, cal_confusion_metrics, Logger

if __name__ == '__main__':
    # get device GPU
    device = torch.device('cuda:{}'.format(0))

    # get options
    opt = Option().parse()
    save_dir = os.path.join('./checkpoints', opt.name)
    logger = Logger(save_dir)

    # get datasets and dataloaders
    U8Dataset = U8Data(dataroot='../Datasets/U8Data/', label_filename='av45_label.csv', task=opt.task)
    train_transforms, test_transforms = U8Data_transform()
    U8Dataset = partition_dataset(data=U8Dataset.data_dict, ratios=[0.6, 0.2, 0.2])
    train_dataset, val_dataset, test_dataset = U8Dataset[0], U8Dataset[1], U8Dataset[2]
    train_dataset = Dataset(data=train_dataset, transform=train_transforms)
    val_dataset = Dataset(data=val_dataset, transform=test_transforms)
    test_dataset = Dataset(data=test_dataset, transform=test_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)
    print('----------------- Dataset -------------------')
    print('The number of training images = %d' % train_size)
    print('The number of val images = %d' % val_size)
    print('The number of test images = %d\n' % test_size)

    # get networks
    net_model = sNet_Original(opt.dim).to(device)

    print('----------------- Model Param -------------------')
    print('Net : %.2fM*2' % (sum([param.nelement() for param in net_model.parameters()]) / 1e6))
    print('----------------- Train Log -------------------')

    # get optimizers
    optimizer, lr_schedualer = getOptimizer(net_model.parameters(), opt)

    # get loss
    weight_CE = torch.FloatTensor([1, 4]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_CE)

    # define train steps
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
        output_logits = net_model(PET)
        output_dic['logits'] = output_logits

        ce_loss = criterion(output_logits, label)

        # print(D_logits)
        output_dic['ce_loss'] = ce_loss.item()

        ce_loss.backward()

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
            output_logits = net_model(PET)
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
                     "ce_loss": Average(output_transform=lambda x: x['ce_loss'])}
    val_metrics = {"accuracy": Accuracy(output_transform=lambda x: [x['logits'], x['label']]),
                   "confusion": ConfusionMatrix(num_classes=2, output_transform=one_hot_transform(target='logits')),
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
                             f"accuracy: {metrics['accuracy']:.4f} ")


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
                                  n_saved=2, filename_prefix='best_label', score_name='accuracy',
                                  global_step_transform=global_step_from_engine(trainer_label),
                                  greater_or_equal=True)
    evaluator.add_event_handler(Events.COMPLETED, checkpoint_saver)

    trainer_label.run(train_dataloader, opt.stage1_epochs + opt.stage2_epochs)