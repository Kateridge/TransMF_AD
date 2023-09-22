import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from models.networks import CrossTransformer_MOD_AVG, sNetDSBN


class model_CNN_DSBN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # networks
        self.cnn = sNetDSBN(dim)
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
        mri_feat, pet_feat = self.cnn(mri, pet)
        mri_feat = self.transform(mri_feat)
        pet_feat = self.transform(pet_feat)
        logits = self.fc(torch.cat([mri_feat, pet_feat], dim=1))
        return logits


class model_CrossTransformer_DSBN(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        # networks
        self.cnn = sNetDSBN(dim)
        self.fuse_transformer = CrossTransformer_MOD_AVG(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.fc_cls = nn.Sequential(nn.Linear(dim * 4, 512), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(512, 64),  nn.ReLU(), nn.Dropout(0.5),
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
        mri_embeddings, pet_embeddings = self.cnn(mri, pet)
        mri_embeddings = rearrange(mri_embeddings, 'b d x y z -> b (x y z) d')
        pet_embeddings = rearrange(pet_embeddings, 'b d x y z -> b (x y z) d')
        # forward stage2
        output_pos = self.fuse_transformer(mri_embeddings, pet_embeddings)  # shape (b, xyz, d)
        # output_pos = F.normalize(output_pos, dim=1)
        output_logits = self.fc_cls(output_pos)
        return output_logits