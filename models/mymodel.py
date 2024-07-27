import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.networks import sNet, CrossTransformer, CrossTransformer_MOD_AVG
from torch.nn import AdaptiveAvgPool1d, AdaptiveMaxPool1d
from torch.autograd import Variable
from models.gradient_reversal import revgrad
import torch.nn.functional as F
from models.losses import FALoss, SupConLoss


class model_single(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # networks
        self.cnn = sNet(dim)
        self.avgpool = self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        # forward CNN
        feature_map = self.cnn(img)
        feature_map = self.avgpool(feature_map)
        feature_map = rearrange(feature_map, 'b c x y z -> b (c x y z)')
        logits = self.fc(feature_map)

        return logits


class model_CNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # networks
        self.mri_cnn = sNet(dim)
        self.pet_cnn = sNet(dim)
        self.transform = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            Rearrange('b c x y z -> b (c x y z)'),
        )
        self.fc = nn.Sequential(nn.Linear(dim * 2, 128), nn.ReLU(), nn.Linear(128, 2))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, mri, pet):
        mri_feat = self.mri_cnn(mri)
        pet_feat = self.pet_cnn(pet)
        mri_feat = self.transform(mri_feat)
        pet_feat = self.transform(pet_feat)
        logits = self.fc(torch.cat([mri_feat, pet_feat], dim=1))
        return logits


class model_transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        # networks
        self.mri_cnn = sNet(dim)
        self.pet_cnn = sNet(dim)
        self.fuse_transformer = CrossTransformer_MOD_AVG(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.fc_cls = nn.Sequential(nn.Linear(dim * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(512, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(64, 2))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, mri, pet):
        # forward stage1
        mri_embeddings = self.mri_cnn(mri)  # shape (b, d, x, y, z,)
        pet_embeddings = self.pet_cnn(pet)  # shape (b, d, x, y, z,)
        mri_embeddings = rearrange(mri_embeddings, 'b d x y z -> b (x y z) d')
        pet_embeddings = rearrange(pet_embeddings, 'b d x y z -> b (x y z) d')
        # forward stage2
        output_pos = self.fuse_transformer(mri_embeddings, pet_embeddings)  # shape (b, xyz, d)
        # output_pos = F.normalize(output_pos, dim=1)
        output_logits = self.fc_cls(output_pos)
        return output_logits


class model_transformer_res(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        # networks
        self.mri_cnn = sNet(dim)
        self.pet_cnn = sNet(dim)
        self.fuse_transformer = CrossTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.fc_cls = nn.Sequential(nn.Linear(dim * 2, 512), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(512, 64), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(64, 2))
        self.gap = nn.Sequential(Rearrange('b n d -> b d n'),
                                 AdaptiveAvgPool1d(1),
                                 Rearrange('b d n -> b (d n)'))
        self.gmp = nn.Sequential(Rearrange('b n d -> b d n'),
                                 AdaptiveMaxPool1d(1),
                                 Rearrange('b d n -> b (d n)'))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, mri, pet):
        # forward stage1
        mri_embeddings = self.mri_cnn(mri)  # shape (b, d, x, y, z,)
        pet_embeddings = self.pet_cnn(pet)  # shape (b, d, x, y, z,)
        mri_embeddings = rearrange(mri_embeddings, 'b d x y z -> b (x y z) d')
        pet_embeddings = rearrange(pet_embeddings, 'b d x y z -> b (x y z) d')
        # forward stage2
        mri_embeddings_fused, pet_embeddings_fused = self.fuse_transformer(mri_embeddings, pet_embeddings)  # shape (b, xyz, d)
        mri_embeddings_final = mri_embeddings_fused + mri_embeddings
        pet_embeddings_final = pet_embeddings_fused + pet_embeddings
        # forward mlp
        mri_cls_avg = self.gap(mri_embeddings_final)
        pet_cls_avg = self.gap(pet_embeddings_final)
        cls_token = torch.cat([mri_cls_avg,pet_cls_avg], dim=1)
        output_logits = self.fc_cls(cls_token)
        return output_logits


class model_CNN_ad(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # networks
        self.mri_cnn = sNet(dim)
        self.pet_cnn = sNet(dim)
        self.fc_cls = nn.Sequential(nn.Linear(dim * 2, 128), nn.ReLU(), nn.Linear(128, 2))
        self.gap = nn.Sequential(nn.AdaptiveAvgPool3d(1), Rearrange('b c x y z -> b (c x y z)'))
        self.D = nn.Sequential(nn.Linear(dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 2))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, mri, pet):
        # forward CNN
        mri_embeddings = self.mri_cnn(mri)  # shape (b, d, x, y, z,)
        pet_embeddings = self.pet_cnn(pet)  # shape (b, d, x, y, z,)

        alpha = torch.Tensor([2]).to(mri.device)
        # alpha = 1
        mri_embedding_vec = revgrad(self.gap(mri_embeddings), alpha)
        pet_embedding_vec = revgrad(self.gap(pet_embeddings), alpha)

        # forward discriminator
        D_MRI_logits = self.D(mri_embedding_vec)
        D_PET_logits = self.D(pet_embedding_vec)

        mri_feat = self.gap(mri_embeddings)
        pet_feat = self.gap(pet_embeddings)
        output_logits = self.fc_cls(torch.cat([mri_feat, pet_feat], dim=1))
        return output_logits, D_MRI_logits, D_PET_logits


class model_ad(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        # networks
        self.mri_cnn = sNet(dim)
        self.pet_cnn = sNet(dim)
        # self.cnn = sNet(dim)
        self.fuse_transformer = CrossTransformer_MOD_AVG(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.fc_cls = nn.Sequential(nn.Linear(dim * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(512, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(64, 2))
        self.gap = nn.Sequential(nn.AdaptiveAvgPool3d(1), Rearrange('b c x y z -> b (c x y z)'))
        self.D = nn.Sequential(nn.Linear(dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 2))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, mri, pet):
        # forward CNN
        mri_embeddings = self.mri_cnn(mri)  # shape (b, d, x, y, z,)
        pet_embeddings = self.pet_cnn(pet)  # shape (b, d, x, y, z,)

        alpha = torch.Tensor([2]).to(mri.device)
        mri_embedding_vec = revgrad(self.gap(mri_embeddings), alpha)
        pet_embedding_vec = revgrad(self.gap(pet_embeddings), alpha)

        # forward discriminator
        D_MRI_logits = self.D(mri_embedding_vec)
        D_PET_logits = self.D(pet_embedding_vec)

        # forward cross transformer
        mri_embeddings = rearrange(mri_embeddings, 'b d x y z -> b (x y z) d')
        pet_embeddings = rearrange(pet_embeddings, 'b d x y z -> b (x y z) d')
        output_pos = self.fuse_transformer(mri_embeddings, pet_embeddings)  # shape (b, xyz, d)
        output_logits = self.fc_cls(output_pos)
        return output_logits, D_MRI_logits, D_PET_logits

