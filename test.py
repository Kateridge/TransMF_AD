import os
from einops import rearrange

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from transformers import TFAutoModelForImageClassification

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from options.option import Option
from datasets import get_dataset
from utils.utils import getOptimizer, cal_confusion_metrics, Logger
from models.mymodel import model_CNN, model_pretrain, model_cl, model_ad, model_transformer
from ignite.metrics import Accuracy, Loss, Average, ConfusionMatrix
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.handlers import Checkpoint, global_step_from_engine, DiskSaver, create_lr_scheduler_with_warmup
from ignite.handlers.early_stopping import EarlyStopping
import torch, ignite
from torch.nn.functional import softmax, adaptive_avg_pool1d
from glob import glob
from sklearn.manifold import TSNE

if __name__ == '__main__':
    # get device GPU
    # device = torch.device('cuda:{}'.format(0))
    device = torch.device('cpu')
    # get options
    opt = Option().parse()
    # ********************* change CNN_PET_ADCN dir here !!! ****************************
    save_dir = os.path.join('./checkpoints', 'CNN_PET_ADCN')
    # ********************* change CNN_PET_ADCN dir here !!! ****************************

    # get datasets and dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataset(opt)
    # torch.save({'train': train_dataloader, 'val': val_dataloader, 'CNN_PET_ADCN': test_dataloader}, 'dataloaders.pt')

    # get networks
    # net_model = model_transformer(dim=opt.dim, depth=opt.trans_enc_depth, heads=4, dim_head=opt.dim // 4,
    #                               mlp_dim=opt.dim * 4, dropout=opt.dropout).to(device)
    net_model = model_CNN(dim=opt.dim).to(device)
    # net_model = model_ad(dim=opt.dim, depth=opt.trans_enc_depth, heads=4,
    #                      dim_head=opt.dim // 4, mlp_dim=opt.dim * 4, dropout=opt.dropout).to(device)
    best_model_path = glob(os.path.join(save_dir, 'best_label_net_model_*.pt'))[1]
    checkpoint_all = torch.load(best_model_path, map_location=device)
    Checkpoint.load_objects(to_load={'net_model': net_model}, checkpoint=checkpoint_all)
    print(f'Load best model {best_model_path}')

    # get loss
    criterion = torch.nn.CrossEntropyLoss()

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


    val_metrics = {"accuracy": Accuracy(output_transform=lambda x: [x['logits'], x['label']]),
                   "confusion": ConfusionMatrix(num_classes=2,
                                                output_transform=one_hot_transform(target='logits')),
                   "auc": ROC_AUC(output_transform=lambda x: [softmax(x['logits'], dim=1)[:, -1], x['label']]),
                   "loss": Loss(criterion, output_transform=lambda x: [x['logits'], x['label']])}

    for name, metric in val_metrics.items():
        metric.attach(evaluator, name)

    # logging for validation every epoch
    @evaluator.on(Events.COMPLETED)
    def log_validation_results(trainer_label):
        metrics = evaluator.state.metrics
        print(f"Validation Results - Epoch[{trainer_label.state.epoch}] ")
        sen, spe, f1 = cal_confusion_metrics(metrics['confusion'])
        print(f"loss: {metrics['loss']:.4f} accuracy: {metrics['accuracy']:.4f} "
                             f"sensitivity: {sen:.4f} specificity: {spe:.4f} "
                             f"f1 score: {f1:.4f} AUC: {metrics['auc']:.4f} ")

    # for item in net_model.named_parameters():
    #     print(item[0])

    # register hook to get intermidiate features
    mri_cnn_feats = np.zeros(shape=(0, 1024))
    pet_cnn_feats = np.zeros(shape=(0, 1024))


    def mri_hook(module, in_feat, out_feat):
        global mri_cnn_feats
        feat = out_feat.detach()
        feat = rearrange(feat, 'b c h w d -> b (c h w d)')
        # feat = adaptive_avg_pool1d(feat, 1).squeeze()  # b 128
        mri_cnn_feats = np.concatenate([mri_cnn_feats, feat.cpu().numpy()], axis=0)
        return None


    def pet_hook(module, in_feat, out_feat):
        global pet_cnn_feats
        feat = out_feat.detach()
        feat = rearrange(feat, 'b c h w d -> b (c h w d)')

        # feat = adaptive_avg_pool1d(feat, 1).squeeze()  # b 128
        pet_cnn_feats = np.concatenate([pet_cnn_feats, feat.cpu().numpy()], axis=0)
        return None

    net_model.mri_cnn.conv5.register_forward_hook(mri_hook)
    net_model.pet_cnn.conv5.register_forward_hook(pet_hook)
    # for (name, module) in net_model.named_modules():
    #     if name == 'fuse_transformer.layers.2.0':
    #         module.register_forward_hook(hook=mri_hook)
    # for (name, module) in net_model.named_modules():
    #     if name == 'fuse_transformer.layers.2.1':
    #         module.register_forward_hook(hook=pet_hook)

    evaluator.run(test_dataloader)

    # mri_cnn.conv5
    # pet_cnn.conv5
    # fuse_transformer.layers.5.0
    # fuse_transformer.layers.5.1
    # Create a two dimensional t-SNE projection of the embeddings
    tsne = TSNE(2, init='pca', learning_rate='auto')
    tsne_proj = tsne.fit_transform(np.concatenate([mri_cnn_feats, pet_cnn_feats], axis=0))
    # tsne_proj_mri = tsne.fit_transform(mri_cnn_feats)
    # tsne_proj_pet = tsne.fit_transform(pet_cnn_feats)


    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8, 8))
    num_samples = tsne_proj.shape[0]
    ax.scatter(tsne_proj[0:num_samples // 2, 0], tsne_proj[0:num_samples // 2, 1], label='MRI', alpha=0.5)
    ax.scatter(tsne_proj[num_samples // 2:, 0], tsne_proj[num_samples // 2:, 1], label='PET', alpha=0.5)
    # ax.scatter(tsne_proj_mri[:, 0], tsne_proj_mri[:, 1], label='MRI', alpha=0.5)
    # ax.scatter(tsne_proj_pet[:, 0], tsne_proj_pet[:, 1], label='PET', alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.savefig(os.path.join(save_dir, 'CNN_PET_ADCN.png'))
