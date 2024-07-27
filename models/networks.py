from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import AdaptiveAvgPool1d, AdaptiveMaxPool1d


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class sNet(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, dim // 4, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim // 4),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 4, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim // 4),
            nn.LeakyReLU(),
            nn.Conv3d(dim // 4, dim // 2, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(dim // 2, dim // 2, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.LeakyReLU(),
            nn.Conv3d(dim // 2, dim // 1, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim // 1),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(dim // 1, dim * 2, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim * 2),
            nn.LeakyReLU(),
            nn.Conv3d(dim * 2, dim, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(dim),
            nn.LeakyReLU(),
            nn.AvgPool3d(2, stride=2)
        )

    def forward(self, mri):
        conv1_out = self.conv1(mri)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)

        return conv4_out


class SFCN(nn.Module):
    def __init__(self):
        super(SFCN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.block5 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

    def forward(self, x):
        # x_pooled = self.maxpool(x)
        print(x.shape)
        block1_out = self.block1(x)
        print(block1_out.shape)
        block2_out = self.block2(block1_out)
        print(block2_out.shape)
        block3_out = self.block3(block2_out)
        print(block3_out.shape)
        block4_out = self.block4(block3_out)
        print(block4_out.shape)
        block5_out = self.block5(block4_out)
        print(block5_out.shape)
        return block5_out


# pre-layernorm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# feedforward
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# attention
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _ = x.shape
        h = self.heads
        context = default(context, x)

        if kv_include_self:
            # cross attention requires CLS token includes itself as key / value
            context = torch.cat((x, context), dim=1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


# transformer encoder
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return self.norm(x)


class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, share=False):
        super().__init__()
        self.share = share
        self.layers = nn.ModuleList([])
        if self.share == True:
            for _ in range(depth):
                self.layers.append(Transformer(dim, 1, heads, dim_head, mlp_dim, dropout=dropout))
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Transformer(dim, 1, heads, dim_head, mlp_dim, dropout=dropout),
                    Transformer(dim, 1, heads, dim_head, mlp_dim, dropout=dropout)
                ]))

    def forward(self, mri_tokens, pet_tokens):
        for mri_enc, pet_enc in self.layers:
            mri_tokens = mri_enc(mri_tokens, context=torch.cat([mri_tokens, pet_tokens], dim=1)) + mri_tokens
            pet_tokens = pet_enc(pet_tokens, context=torch.cat([mri_tokens, pet_tokens], dim=1)) + pet_tokens
        return mri_tokens, pet_tokens


class CrossTransformer_MOD_AVG(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim, 1, heads, dim_head, mlp_dim, dropout=dropout),
                Transformer(dim, 1, heads, dim_head, mlp_dim, dropout=dropout)
            ]))
        self.gap = nn.Sequential(Rearrange('b n d -> b d n'),
                                 AdaptiveAvgPool1d(1),
                                 Rearrange('b d n -> b (d n)'))
        self.gmp = nn.Sequential(Rearrange('b n d -> b d n'),
                                 AdaptiveMaxPool1d(1),
                                 Rearrange('b d n -> b (d n)'))


    def forward(self, mri_tokens, pet_tokens):
        for mri_enc, pet_enc in self.layers:
            mri_tokens = mri_enc(mri_tokens, context=pet_tokens) + mri_tokens
            pet_tokens = pet_enc(pet_tokens, context=mri_tokens) + pet_tokens
        mri_cls_avg = self.gap(mri_tokens)
        mri_cls_max = self.gmp(mri_tokens)
        pet_cls_avg = self.gap(pet_tokens)
        pet_cls_max = self.gmp(pet_tokens)
        cls_token = torch.cat([mri_cls_avg,  pet_cls_avg, mri_cls_max, pet_cls_max], dim=1)
        return cls_token

