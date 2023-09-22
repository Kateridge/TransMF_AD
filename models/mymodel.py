import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.networks import CrossTransformer_AVG, sNet, Transformer, PositionalEncoding1D, MultiModalityEncoder_Classic, \
    MultiModalityEncoder_Classic_AVG, CrossTransformer, SFCN, CrossTransformer_MOD_AVG, CrossTransformer_AVG_mod, \
    CrossTransformer_AVG_Single, sNetDSBN
from torch.nn import AdaptiveAvgPool1d, AdaptiveMaxPool1d
from torch.autograd import Variable
from models.gradient_reversal import revgrad
import torch.nn.functional as F
from models.resnet import generate_model
from models.losses import FALoss, SupConLoss


class stage1_network(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.cnn = sNet(dim)
        self.transformer_encoder = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.fc = nn.Linear(dim, 2)
        self.pe = PositionalEncoding1D(dim)

    def forward(self, img):
        # forward cnn
        cnn_logits, conv5_out = self.cnn(img)
        # feature to token
        token_embedding = rearrange(conv5_out, 'b c x y z -> b (x y z) c')
        b, _, _ = token_embedding.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        token_embedding = torch.cat((cls_tokens, token_embedding), dim=1)
        # add positional encoding
        # token_embedding += self.pe(token_embedding)
        # forward transformer encoder
        # token_embedding_encoded = self.transformer_encoder(token_embedding)
        # forward fc layer
        # cls_token = token_embedding_encoded[:, 0]
        # transformer_encoder_logits = self.fc(F.normalize(cls_token, p=2, dim=-1))
        # return cnn_logits, transformer_encoder_logits, token_embedding
        return cnn_logits, conv5_out, token_embedding

# class new_stage1_network(nn.Module):
#     def __init__(self, num_patch):
#         super().__init__()
#         self.cnn = patchNet()
#         self.fc = nn.Linear(num_patch * 128, 2)
#
#     def forward(self, patches):
#         patch_cnn_out = []
#         for patch in patches:
#             patch_cnn_out.append(rearrange(self.cnn(patch), 'b c x y z -> b (x y z) c'))
#         token_embedding = torch.cat(patch_cnn_out, dim=1)
#         logits = self.fc(rearrange(token_embedding, 'b n d -> b (n d)'))
#         return logits, token_embedding


# class stage2_network(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
#         super().__init__()
#         self.transformer_encoder = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.fc = nn.Linear(dim, 2)
#         self.pe = PositionalEncoding1D(dim)
#
#     def forward(self, token_embedding):
#         # add class embedding
#         b, _, _ = token_embedding.shape
#         cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
#         token_embedding = torch.cat((cls_tokens, token_embedding), dim=1)
#         # add positional encoding
#         token_embedding += self.pe(token_embedding)
#         # forward transformer encoder
#         token_embedding_encoded = self.transformer_encoder(token_embedding)
#         # forward fc layer
#         cls_token = token_embedding_encoded[:, 0]
#         transformer_encoder_logits = self.fc(cls_token)
#         return transformer_encoder_logits, token_embedding_encoded


# class stage3_network(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, dropout):
#         super().__init__()
#         self.cross_attention = CrossTransformer(dim, depth, heads, dim_head, dropout)  # 4中修改
#         self.fc = nn.Linear(512, 2)
#
#     def forward(self, mri_tokens, pet_tokens):
#         mri_tokens_final, pet_tokens_final = self.cross_attention(mri_tokens, pet_tokens)
#         mri_cls, pet_cls = map(lambda t: t[:, 0], (mri_tokens_final, pet_tokens_final))
#         return self.fc(torch.cat((mri_cls, pet_cls), dim=1))
class stage2_network(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, bottleneck_num, dropout):
        super().__init__()
        self.fuse_encoder = MultiModalityEncoder(dim, depth, heads, dim_head, mlp_dim, bottleneck_num, dropout)
        self.fc = nn.Linear(bottleneck_num * dim, 2)

    def forward(self, mri_tokens, pet_tokens):
        mri_tokens_final, pet_tokens_final, bn_tokens = self.fuse_encoder(mri_tokens, pet_tokens)
        bn_tokens = rearrange(bn_tokens, 'b n d -> b (n d)')
        mri_cls, pet_cls = map(lambda t: t[:, 0], (mri_tokens_final, pet_tokens_final))
        # return self.fc(F.normalize(torch.cat((mri_cls, pet_cls), dim=1), p=2, dim=1))
        # final_logits = self.fc(torch.cat((mri_cls, pet_cls), dim=1))
        final_logits = self.fc(bn_tokens)
        return final_logits, mri_cls, pet_cls


class stage1_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet10()
        # med3d = torch.load('./models/resnet_10_23dataset.pth')['state_dict']
        # print('load model from ./models/resnet_10_23dataset.pth')
        # med3d_state_dict = {}
        # for k, v in med3d.items():
        #     med3d_state_dict[k.split('module.')[-1]] = v
        # self.resnet.load_state_dict(med3d_state_dict)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.fc = nn.Linear(512, 2)

    def forward(self, img):
        x = self.resnet(img)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class stage1_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = resnet10()
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.fc = nn.Linear(512, 2)

    def forward(self, img):
        x = self.resnet(img)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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
        # mri_cls_max = self.gmp(mri_embeddings_final)
        pet_cls_avg = self.gap(pet_embeddings_final)
        # pet_cls_max = self.gmp(pet_embeddings_final)
        # cls_token = torch.cat([mri_cls_avg, mri_cls_max, pet_cls_avg, pet_cls_max], dim=1)
        cls_token = torch.cat([mri_cls_avg,pet_cls_avg], dim=1)
        output_logits = self.fc_cls(cls_token)
        return output_logits


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
        # alpha = 1
        mri_embedding_vec = revgrad(self.gap(mri_embeddings), alpha)
        pet_embedding_vec = revgrad(self.gap(pet_embeddings), alpha)
        # mri_embedding_vec = self.gap(mri_embeddings)
        # pet_embedding_vec = self.gap(pet_embeddings)

        # forward discriminator
        # mri_gt = torch.zeros([mri_embeddings.shape[0], 1]).to(mri_embeddings.device)
        # pet_gt = torch.ones([pet_embeddings.shape[0], 1]).to(mri_embeddings.device)
        # mri_ad_loss = self.ad_loss(self.D(mri_embedding_vec), mri_gt)
        # pet_ad_loss = self.ad_loss(self.D(pet_embedding_vec), pet_gt)
        # D_loss = self.ad_loss(self.D(torch.cat([mri_embedding_vec, pet_embedding_vec], dim=0)), gt)
        # D_loss = (mri_ad_loss + pet_ad_loss)/2
        D_MRI_logits = self.D(mri_embedding_vec)
        D_PET_logits = self.D(pet_embedding_vec)

        # forward cross transformer
        mri_embeddings = rearrange(mri_embeddings, 'b d x y z -> b (x y z) d')
        pet_embeddings = rearrange(pet_embeddings, 'b d x y z -> b (x y z) d')
        output_pos = self.fuse_transformer(mri_embeddings, pet_embeddings)  # shape (b, xyz, d)
        output_logits = self.fc_cls(output_pos)
        return output_logits, D_MRI_logits, D_PET_logits


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


class model_withSupCL(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        # networks
        self.mri_cnn = sNet(dim)
        self.pet_cnn = sNet(dim)
        self.fuse_transformer = CrossTransformer_AVG_mod(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.fc_cls = nn.Sequential(nn.Linear(dim * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(512, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(64, 2))
        self.fc_cl = nn.Sequential(nn.Linear(dim * 4, 128), nn.ReLU())
        self.loss_cl = SupConLoss()

    def forward(self, mri, pet, label):
        # forward CNN
        mri_embeddings = self.mri_cnn(mri)  # shape (b, d, x, y, z,)
        pet_embeddings = self.pet_cnn(pet)  # shape (b, d, x, y, z,)

        # CNN SupContrast

        # forward Transformer
        mri_embeddings_vec = rearrange(mri_embeddings, 'b d x y z -> b (x y z) d')
        pet_embeddings_vec = rearrange(pet_embeddings, 'b d x y z -> b (x y z) d')

        output_feats, mri_tokens, pet_tokens = self.fuse_transformer(mri_embeddings_vec, pet_embeddings_vec)  # shape (b, xyz, d)

        # Transformer SupContrast
        # loss_cl = (self.loss_cl(F.normalize(mri_tokens, dim=2), labels=label) + self.loss_cl(F.normalize(pet_tokens), labels=label)) / 2
        # forward fc layer
        output_logits = self.fc_cls(output_feats)
        output_cl = self.fc_cl(output_feats)
        loss_cl = self.loss_cl(output_cl, labels=label)
        return output_logits, loss_cl


class model_CNN_withcl(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # networks
        self.mri_cnn = sNet(dim)
        self.pet_cnn = sNet(dim)
        self.mri_proj = nn.Linear(dim, dim)
        self.pet_proj = nn.Linear(dim, dim)
        self.transform = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            Rearrange('b c x y z -> b (c x y z)'),
        )
        self.fc = nn.Linear(dim * 2, 2)
        self.t = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, mri, pet):
        mri_feat = self.mri_cnn(mri)
        pet_feat = self.pet_cnn(pet)
        mri_feat = self.transform(mri_feat)
        pet_feat = self.transform(pet_feat)

        mri_embeddings_vec = F.normalize(self.mri_proj(mri_feat), dim=-1)
        pet_embeddings_vec = F.normalize(self.pet_proj(pet_feat), dim=-1)
        mri_pet_similarity = mri_embeddings_vec @ pet_embeddings_vec.t() / self.t
        sim_targets = torch.zeros(mri_pet_similarity.size()).to(mri.device)
        sim_targets.fill_diagonal_(1)
        loss_mri2pet = -torch.sum(F.log_softmax(mri_pet_similarity, dim=1) * sim_targets, dim=1).mean()
        loss_pet2mri = -torch.sum(F.log_softmax(mri_pet_similarity, dim=0) * sim_targets, dim=0).mean()
        loss_cl = (loss_mri2pet + loss_pet2mri) / 2

        logits = self.fc(F.normalize(torch.cat([mri_feat, pet_feat], dim=1), dim=1))

        return logits, loss_cl


class model_cl(nn.Module):
    def __init__(self, dim, cl_dim, t, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        # networks
        self.mri_cnn = sNet(dim)
        self.pet_cnn = sNet(dim)
        self.fuse_transformer = CrossTransformer_AVG(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mri_proj = nn.Linear(dim, cl_dim)
        self.pet_proj = nn.Linear(dim, cl_dim)
        self.fc_cls_2 = nn.Sequential(nn.Linear(dim * 4, 32), nn.Linear(32, 2))
        # tools
        self.cl_transform = nn.Sequential(Rearrange('b n d -> b d n'),
                                          AdaptiveAvgPool1d(1),
                                          Rearrange('b d n -> b (d n)'))
        self.t = nn.Parameter(torch.ones([]) * t)

    def forward(self, mri, pet):
        # forward stage1
        mri_embeddings = self.mri_cnn(mri)  # shape (b, d, x, y, z,)
        pet_embeddings = self.pet_cnn(pet)  # shape (b, d, x, y, z,)
        mri_embeddings = rearrange(mri_embeddings, 'b d x y z -> b (x y z) d')
        pet_embeddings = rearrange(pet_embeddings, 'b d x y z -> b (x y z) d')

        # stage1 contrastive loss
        mri_embeddings_vec = self.cl_transform(mri_embeddings)  # shape (b, dim)
        pet_embeddings_vec = self.cl_transform(pet_embeddings)  # shape (b, dim)
        mri_embeddings_vec = F.normalize(self.mri_proj(mri_embeddings_vec), dim=-1)
        pet_embeddings_vec = F.normalize(self.pet_proj(pet_embeddings_vec), dim=-1)
        mri_pet_similarity = mri_embeddings_vec @ pet_embeddings_vec.t() / self.t

        sim_targets = torch.zeros(mri_pet_similarity.size()).to(mri.device)
        sim_targets.fill_diagonal_(1)
        loss_mri2pet = -torch.sum(F.log_softmax(mri_pet_similarity, dim=1) * sim_targets, dim=1).mean()
        loss_pet2mri = -torch.sum(F.log_softmax(mri_pet_similarity, dim=0) * sim_targets, dim=0).mean()
        loss_cl = (loss_mri2pet + loss_pet2mri) / 2

        output_pos = self.fuse_transformer(mri_embeddings, pet_embeddings)  # shape (b, xyz, d)
        output_logits = self.fc_cls_2(output_pos)

        return output_logits, loss_cl


class model_pretrain(nn.Module):
    def __init__(self, dim, cl_dim, t, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        # networks
        self.mri_cnn = sNet(dim)
        self.pet_cnn = sNet(dim)
        self.fuse_transformer = CrossTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mri_proj = nn.Linear(dim, cl_dim)
        self.pet_proj = nn.Linear(dim, cl_dim)
        self.fc_cls = nn.Linear(dim * 2, 2)
        # self.fc_cls_2 = nn.Sequential(nn.Linear(dim * 4, 32), nn.Linear(32, 2))
        self.fc_age = nn.Linear(dim * 2, 1)
        self.fc_match = nn.Linear(dim * 2, 2)
        self.fc_gender = nn.Linear(dim * 2, 2)
        # tools
        self.cl_transform = nn.Sequential(Rearrange('b n d -> b d n'),
                                          AdaptiveAvgPool1d(1),
                                          Rearrange('b d n -> b (d n)'))
        self.t = nn.Parameter(torch.ones([]) * t)

    def pretrain_forward(self, mri, pet):
        # forward stage1
        mri_embeddings = self.mri_cnn(mri)  # shape (b, d, x, y, z,)
        pet_embeddings = self.pet_cnn(pet)  # shape (b, d, x, y, z,)
        mri_embeddings = rearrange(mri_embeddings, 'b d x y z -> b (x y z) d')
        pet_embeddings = rearrange(pet_embeddings, 'b d x y z -> b (x y z) d')
        bs, _, _ = mri_embeddings.shape

        # stage1 contrastive loss
        mri_embeddings_vec = self.cl_transform(mri_embeddings)  # shape (b, dim)
        pet_embeddings_vec = self.cl_transform(pet_embeddings)  # shape (b, dim)
        mri_embeddings_vec = F.normalize(self.mri_proj(mri_embeddings_vec), dim=-1)
        pet_embeddings_vec = F.normalize(self.pet_proj(pet_embeddings_vec), dim=-1)
        mri_pet_similarity = mri_embeddings_vec @ pet_embeddings_vec.t() / self.t

        sim_targets = torch.zeros(mri_pet_similarity.size()).to(mri.device)
        sim_targets.fill_diagonal_(1)
        loss_mri2pet = -torch.sum(F.log_softmax(mri_pet_similarity, dim=1) * sim_targets, dim=1).mean()
        loss_pet2mri = -torch.sum(F.log_softmax(mri_pet_similarity, dim=0) * sim_targets, dim=0).mean()
        loss_cl = (loss_mri2pet + loss_pet2mri) / 2

        # stage2 mri and pet matching loss
        # forward positive features
        output_pos = self.fuse_transformer(mri_embeddings, pet_embeddings)  # shape (b, xyz, d)
        # select negative pairs
        with torch.no_grad():
            # weights_mri2pet = F.softmax(mri_pet_similarity, dim=1)
            # weights_pet2mri = F.softmax(mri_pet_similarity, dim=0).t()
            weights_mri2pet = torch.rand(mri_pet_similarity.size()).to(mri.device)
            weights_pet2mri = torch.rand(mri_pet_similarity.size()).to(mri.device)
            weights_mri2pet.fill_diagonal_(0)
            weights_pet2mri.fill_diagonal_(0)
        mri_embeddings_neg = []
        pet_embeddings_neg = []
        for b in range(bs):
            mri_neg_idx = torch.multinomial(weights_pet2mri[b, :], 1).item()
            pet_neg_idx = torch.multinomial(weights_mri2pet[b, :], 1).item()
            mri_embeddings_neg.append(mri_embeddings[mri_neg_idx])
            pet_embeddings_neg.append(pet_embeddings[pet_neg_idx])
        mri_embeddings_neg = torch.stack(mri_embeddings_neg, dim=0)
        pet_embeddings_neg = torch.stack(pet_embeddings_neg, dim=0)
        mri_embeddings_all = torch.cat([mri_embeddings, mri_embeddings_neg], dim=0)
        pet_embeddings_all = torch.cat([pet_embeddings_neg, pet_embeddings], dim=0)
        output_neg = self.fuse_transformer(mri_embeddings_all, pet_embeddings_all)
        output_logits = self.fc_match(torch.cat([output_pos, output_neg], dim=0))
        match_label = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(
            mri.device)
        loss_match = F.cross_entropy(output_logits, match_label)

        return loss_cl, loss_match

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


class model_transformer_resnet(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        # networks
        self.mri_cnn = generate_model(18)
        self.pet_cnn = generate_model(18)
        self.fuse_transformer = CrossTransformer_MOD_AVG(256, depth, heads, dim_head, mlp_dim, dropout)
        self.fc_cls = nn.Sequential(nn.Linear(256 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
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


class model_CNN_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        # networks
        self.mri_cnn = generate_model(18)
        self.pet_cnn = generate_model(18)
        self.transform = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            Rearrange('b c x y z -> b (c x y z)'),
        )
        self.fc = nn.Sequential(nn.Linear(256 * 2, 128), nn.ReLU(), nn.Linear(128, 2))
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
        # mri_embedding_vec = self.gap(mri_embeddings)
        # pet_embedding_vec = self.gap(pet_embeddings)

        # forward discriminator
        # mri_gt = torch.zeros([mri_embeddings.shape[0], 1]).to(mri_embeddings.device)
        # pet_gt = torch.ones([pet_embeddings.shape[0], 1]).to(mri_embeddings.device)
        # mri_ad_loss = self.ad_loss(self.D(mri_embedding_vec), mri_gt)
        # pet_ad_loss = self.ad_loss(self.D(pet_embedding_vec), pet_gt)
        # D_loss = self.ad_loss(self.D(torch.cat([mri_embedding_vec, pet_embedding_vec], dim=0)), gt)
        # D_loss = (mri_ad_loss + pet_ad_loss)/2
        D_MRI_logits = self.D(mri_embedding_vec)
        D_PET_logits = self.D(pet_embedding_vec)

        mri_feat = self.gap(mri_embeddings)
        pet_feat = self.gap(pet_embeddings)
        output_logits = self.fc_cls(torch.cat([mri_feat, pet_feat], dim=1))
        return output_logits, D_MRI_logits, D_PET_logits
