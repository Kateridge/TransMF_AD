import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from options.option import Option
from datasets import get_dataset
from utils.utils import getOptimizer, cal_confusion_metrics
from models.mymodel import stage1_network, stage2_network, stage1_resnet
from monai.data import DataLoader, Dataset
from ignite.metrics import Accuracy, Loss, Average, ConfusionMatrix
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.handlers import Checkpoint, global_step_from_engine
import torch, ignite
from torch.nn.functional import softmax
from glob import glob

if __name__ == '__main__':
    # get device GPU
    device = torch.device('cuda:{}'.format(0))

    # get options
    opt = Option().parse()
    save_dir = os.path.join('./checkpoints', opt.name)

    # get datasets and dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataset(opt)
    # torch.save({'train': train_dataloader, 'val': val_dataloader, 'CNN_PET_ADCN': test_dataloader}, 'dataloaders.pt')

    # get networks
    net_mri_stage1 = stage1_network(dim=opt.dim, depth=opt.trans_enc_depth, dropout=opt.dropout,
                                    heads=4, dim_head=opt.dim // 4, mlp_dim=opt.dim * 4).to(device)
    net_pet_stage1 = stage1_network(dim=opt.dim, depth=opt.trans_enc_depth, dropout=opt.dropout,
                                    heads=4, dim_head=opt.dim // 4, mlp_dim=opt.dim * 4).to(device)
    net_stage2 = stage2_network(dim=opt.dim, depth=opt.cross_attn_depth, heads=4,
                                dim_head=opt.dim // 4, mlp_dim=opt.dim * 4, bottleneck_num=8,
                                dropout=opt.dropout).to(device)

    print('----------------- Model Param -------------------')
    print('Net stage1: %.2fM*2' % (sum([param.nelement() for param in net_mri_stage1.parameters()]) / 1e6))
    print('Net stage2: %.2fM\n' % (sum([param.nelement() for param in net_stage2.parameters()]) / 1e6))
    print('----------------- Train Log -------------------')

    # get optimizers
    net_mri_stage1_optimizer = getOptimizer(net_mri_stage1, opt.lr, opt.weight_decay)
    net_pet_stage1_optimizer = getOptimizer(net_pet_stage1, opt.lr, opt.weight_decay)
    net_stage2_optimizer = getOptimizer(net_stage2, opt.lr, opt.weight_decay)

    # get loss
    criterion = torch.nn.CrossEntropyLoss()

    # define train steps
    def train_step(engine, batch):
        output_dic = {}
        # set model to train mode
        net_mri_stage1.train()
        net_pet_stage1.train()
        net_stage2.train()

        # decompose batch data
        MRI = batch['MRI'].to(device)
        PET = batch['PET'].to(device)
        label = batch['label'].to(device)
        output_dic['label'] = label

        # zero grad
        net_mri_stage1_optimizer.zero_grad()
        net_pet_stage1_optimizer.zero_grad()
        net_stage2_optimizer.zero_grad()

        # forward
        mri_stage1_logits, _, mri_embeddings = net_mri_stage1(MRI)
        pet_stage1_logits, _, pet_embeddings = net_pet_stage1(PET)
        output_dic['mri_stage1_logits'] = mri_stage1_logits
        output_dic['pet_stage1_logits'] = pet_stage1_logits

        # calculate loss
        mri_stage1_loss = criterion(mri_stage1_logits, label)
        pet_stage1_loss = criterion(pet_stage1_logits, label)
        output_dic['mri_stage1_loss'] = mri_stage1_loss.item()
        output_dic['pet_stage1_loss'] = pet_stage1_loss.item()

        # backward
        if engine.state.epoch > opt.stage1_epochs:
            stage2_logits, _, _ = net_stage2(mri_embeddings, pet_embeddings)
            stage2_loss = criterion(stage2_logits, label)
            output_dic['stage2_logits'] = stage2_logits
            output_dic['stage2_loss'] = stage2_loss.item()
            stage2_loss.backward()
        else:
            mri_stage1_loss.backward()
            pet_stage1_loss.backward()

        # update param
        net_mri_stage1_optimizer.step()
        net_pet_stage1_optimizer.step()
        net_stage2_optimizer.step()

        return output_dic


    trainer = Engine(train_step)
    ProgressBar().attach(trainer)

    # define validation step
    def val_step(engine, batch):
        output_dic = {}
        # set model to val mode
        net_mri_stage1.eval()
        net_pet_stage1.eval()
        net_stage2.eval()

        with torch.no_grad():
            # decompose batch data
            MRI = batch['MRI'].to(device)
            PET = batch['PET'].to(device)
            label = batch['label'].to(device)
            output_dic['label'] = label

            # forward
            mri_stage1_logits, _, mri_embeddings = net_mri_stage1(MRI)
            pet_stage1_logits, _, pet_embeddings = net_pet_stage1(PET)
            output_dic['mri_stage1_logits'] = mri_stage1_logits
            output_dic['pet_stage1_logits'] = pet_stage1_logits
            if trainer.state.epoch > opt.stage1_epochs:
                stage2_logits, _, _ = net_stage2(mri_embeddings, pet_embeddings)
                output_dic['stage2_logits'] = stage2_logits
            return output_dic


    evaluator = Engine(val_step)
    ProgressBar().attach(evaluator)

    # define metrics
    # def one_hot_transform_mri(output):
    #     y_pred, y = output['mri_stage1_logits'], output['label']
    #     y_pred = torch.argmax(y_pred, dim=1).long()
    #     y_pred = ignite.utils.to_onehot(y_pred, 2)
    #     y = y.long()
    #     return y_pred, y
    #
    #
    # def one_hot_transform_pet(output):
    #     y_pred, y = output['pet_stage1_logits'], output['label']
    #     y_pred = torch.argmax(y_pred, dim=1).long()
    #     y_pred = ignite.utils.to_onehot(y_pred, 2)
    #     y = y.long()
    #     return y_pred, y

    # def one_hot_transform(output):
    #     y_pred, y = output['stage2_logits'], output['label']
    #     y_pred = torch.argmax(y_pred, dim=1).long()
    #     y_pred = ignite.utils.to_onehot(y_pred, 2)
    #     y = y.long()
    #     return y_pred, y

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
    train_metrics_stage1 = {"mri_accuracy": Accuracy(output_transform=lambda x: [x['mri_stage1_logits'], x['label']]),
                            "mri_loss": Average(output_transform=lambda x: x['mri_stage1_loss']),
                            "pet_accuracy": Accuracy(output_transform=lambda x: [x['pet_stage1_logits'], x['label']]),
                            "pet_loss": Average(output_transform=lambda x: x['pet_stage1_loss'])}
    val_metrics_stage1 = {"mri_accuracy": Accuracy(output_transform=lambda x: [x['mri_stage1_logits'], x['label']]),
                          "mri_confusion": ConfusionMatrix(num_classes=2, output_transform=one_hot_transform(target='mri_stage1_logits')),
                          "mri_auc": ROC_AUC(output_transform=lambda x: [softmax(x['mri_stage1_logits'], dim=1)[:, -1], x['label']]),
                          "mri_loss": Loss(criterion, output_transform=lambda x: [x['mri_stage1_logits'], x['label']]),
                          "pet_accuracy": Accuracy(output_transform=lambda x: [x['pet_stage1_logits'], x['label']]),
                          "pet_confusion": ConfusionMatrix(num_classes=2, output_transform=one_hot_transform(target='pet_stage1_logits')),
                          "pet_auc": ROC_AUC(output_transform=lambda x: [softmax(x['pet_stage1_logits'], dim=1)[:, -1], x['label']]),
                          "pet_loss": Loss(criterion, output_transform=lambda x: [x['pet_stage1_logits'], x['label']])}

    # attach stage1 metrics
    for name, metric in train_metrics_stage1.items():
        metric.attach(trainer, name)

    for name, metric in val_metrics_stage1.items():
        metric.attach(evaluator, name)

    # stage2 metrics
    train_metrics_stage2 = {"accuracy": Accuracy(output_transform=lambda x: [x['stage2_logits'], x['label']]),
                            "loss": Average(output_transform=lambda x: x['stage2_loss'])}
    val_metrics_stage2 = {"accuracy": Accuracy(output_transform=lambda x: [x['stage2_logits'], x['label']]),
                          "confusion": ConfusionMatrix(num_classes=2, output_transform=one_hot_transform(target='stage2_logits')),
                          "auc": ROC_AUC(output_transform=lambda x: [softmax(x['stage2_logits'], dim=1)[:, -1], x['label']]),
                          "loss": Loss(criterion, output_transform=lambda x: [x['stage2_logits'], x['label']])}

    # attach stage2 metrics
    @trainer.on(Events.EPOCH_STARTED(once=opt.stage1_epochs+1))
    def attach_stage2_metrics(trainer):
        # detach stage1 metrics
        for name, metric in train_metrics_stage1.items():
            metric.detach(trainer)
        for name, metric in val_metrics_stage1.items():
            metric.detach(evaluator)
        # attach stage2 metrics
        for name, metric in train_metrics_stage2.items():
            metric.attach(trainer, name)
        for name, metric in val_metrics_stage2.items():
            metric.attach(evaluator, name)
        # re-attach logging function for trainer, let the logging procedure run after new metric calculation
        trainer.remove_event_handler(log_training_results, Events.EPOCH_COMPLETED)
        trainer.remove_event_handler(log_validation_results, Events.EPOCH_COMPLETED)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

    # logging for training every epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        metrics = trainer.state.metrics
        print(f"Training Results - Epoch[{trainer.state.epoch}] ")
        if trainer.state.epoch > opt.stage1_epochs:
            print(f"loss: {metrics['loss']:.4f} accuracy: {metrics['accuracy']:.4f}")
        else:
            print(f"MRI loss: {metrics['mri_loss']:.4f} accuracy: {metrics['mri_accuracy']:.4f}")
            print(f"PET loss: {metrics['pet_loss']:.4f} accuracy: {metrics['pet_accuracy']:.4f}")

    # logging for validation every epochw
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_dataloader)
        metrics = evaluator.state.metrics
        print(f"Validation Results - Epoch[{trainer.state.epoch}] ")
        if trainer.state.epoch > opt.stage1_epochs:
            sen, spe, f1 = cal_confusion_metrics(metrics['confusion'])
            print(f"loss: {metrics['loss']:.4f} accuracy: {metrics['accuracy']:.4f} "
                  f"sensitivity: {sen:.4f} specificity: {spe:.4f} "
                  f"f1 score: {f1:.4f} AUC: {metrics['auc']:.4f} ")
        else:
            sen_mri, spe_mri, f1_mri = cal_confusion_metrics(metrics['mri_confusion'])
            sen_pet, spe_pet, f1_pet = cal_confusion_metrics(metrics['pet_confusion'])
            print(f"MRI loss: {metrics['mri_loss']:.4f} accuracy: {metrics['mri_accuracy']:.4f} "
                  f"sensitivity: {sen_mri:.4f} specificity: {spe_mri:.4f} "
                  f"f1 score: {f1_mri:.4f} AUC: {metrics['mri_auc']:.4f} ")
            print(f"PET loss: {metrics['pet_loss']:.4f} accuracy: {metrics['pet_accuracy']:.4f} "
                  f"sensitivity: {sen_pet:.4f} specificity: {spe_pet:.4f} "
                  f"f1 score: {f1_pet:.4f} AUC: {metrics['pet_auc']:.4f} ")


    # save model according to ACC or AUC
    checkpoint_saver_mri = Checkpoint({'net_mri_stage1': net_mri_stage1}, save_dir, n_saved=1,
                                      filename_prefix='best', score_name='mri_accuracy',
                                      global_step_transform=global_step_from_engine(trainer))
    checkpoint_saver_pet = Checkpoint({'net_pet_stage1': net_pet_stage1}, save_dir, n_saved=1,
                                      filename_prefix='best', score_name='pet_accuracy',
                                      global_step_transform=global_step_from_engine(trainer))
    checkpoint_saver_stage2 = Checkpoint({'net_mri_stage1': net_mri_stage1, 'net_pet_stage1': net_pet_stage1,
                                          'net_stage2': net_stage2}, save_dir, n_saved=1,
                                         filename_prefix='best', score_name='accuracy',
                                         global_step_transform=global_step_from_engine(trainer))
    evaluator.add_event_handler(Events.COMPLETED, checkpoint_saver_mri)
    evaluator.add_event_handler(Events.COMPLETED, checkpoint_saver_pet)

    # add save model handle for stage 2 and load best model for stage1 network
    @trainer.on(Events.EPOCH_STARTED(once=opt.stage1_epochs + 1))
    def attach_stage2_checkpoints():
        # remove old handers and add new handers
        evaluator.remove_event_handler(checkpoint_saver_mri, Events.COMPLETED)
        evaluator.remove_event_handler(checkpoint_saver_pet, Events.COMPLETED)
        evaluator.add_event_handler(Events.COMPLETED, checkpoint_saver_stage2)
        # load best model in stage1
        best_mri_stage1_path = glob(os.path.join(save_dir, 'best_net_mri_stage1_*.pt'))[0]
        best_pet_stage2_path = glob(os.path.join(save_dir, 'best_net_pet_stage1_*.pt'))[0]
        checkpoint_mri = torch.load(best_mri_stage1_path, map_location=device)
        checkpoint_pet = torch.load(best_pet_stage2_path, map_location=device)
        Checkpoint.load_objects(to_load={'net_mri_stage1': net_mri_stage1}, checkpoint=checkpoint_mri)
        Checkpoint.load_objects(to_load={'net_pet_stage1': net_pet_stage1}, checkpoint=checkpoint_pet)
        print(f'Load best MRI model in stage1 {best_mri_stage1_path}')
        print(f'Load best PET model in stage1 {best_pet_stage2_path}')


    trainer.run(train_dataloader, opt.stage1_epochs + opt.stage2_epochs)

    # CNN_PET_ADCN stage
    evaluator.run(test_dataloader)
    metrics = evaluator.state.metrics
    print(f"Test Results")
    sen_mri, spe_mri, f1_mri = cal_confusion_metrics(metrics['mri_confusion'])
    sen_pet, spe_pet, f1_pet = cal_confusion_metrics(metrics['pet_confusion'])
    print(f"MRI accuracy: {metrics['mri_accuracy']:.4f} "
          f"sensitivity: {sen_mri:.4f} specificity: {spe_mri:.4f} "
          f"f1 score: {f1_mri:.4f} AUC: {metrics['mri_auc']:.4f} ")
    print(f"PET accuracy: {metrics['pet_accuracy']:.4f} "
          f"sensitivity: {sen_pet:.4f} specificity: {spe_pet:.4f} "
          f"f1 score: {f1_pet:.4f} AUC: {metrics['pet_auc']:.4f} ")
    