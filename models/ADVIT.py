import torch
from torch import nn
from vit_pytorch import ViT
from einops import rearrange


class ADVIT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_2d_mri = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1, 1, 25), stride=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2)),
            nn.Conv3d(32, 1, kernel_size=(1, 1, 25), stride=(1, 1, 1)),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))
        )
        self.to_2d_pet = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1, 1, 25), stride=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2)),
            nn.Conv3d(32, 1, kernel_size=(1, 1, 25), stride=(1, 1, 1)),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))
        )
        self.vit_mri = ViT(
            image_size=128,
            patch_size=16,
            num_classes=2,
            channels=1,
            dim=192,
            depth=6,
            heads=3,
            mlp_dim=768,
            dropout=0.1,
            emb_dropout=0.1)
        self.vit_pet = ViT(
            image_size=128,
            patch_size=16,
            num_classes=2,
            channels=1,
            dim=192,
            depth=6,
            heads=3,
            mlp_dim=768,
            dropout=0.1,
            emb_dropout=0.1)
        self.fc = nn.Linear(192*2, 2)

    def forward(self, mri,pet):
        mri_out = self.to_2d_mri(mri)
        pet_out = self.to_2d_pet(pet)
        mri_out = rearrange(mri_out, 'b c h w d -> b c h (w d)')
        pet_out = rearrange(pet_out, 'b c h w d -> b c h (w d)')
        mri_out = self.vit_mri(mri_out)
        pet_out = self.vit_pet(pet_out)
        logits = self.fc(torch.cat([mri_out, pet_out], dim=-1))

        return logits

# a = torch.randn([4,1,128,128,79])
# b = torch.randn([4,1,128,128,79])
# model = ADVIT()
# out = model(a,b)
# print(out.shape)

