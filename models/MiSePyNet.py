import torch
from torch import nn


class slice_cnn(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(1, 1, dim)),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(1, 1, (dim + 1) // 2)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=(1, 1, (dim + 1) // 2)),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(1, 1, (dim + 2) // 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=(1, 1, (dim + 2) // 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=(1, 1, (dim + 2) // 3)),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )

    def forward(self, img):
        conv1_out = self.conv1(img)
        conv2_out = self.conv2(img)
        conv3_out = self.conv3(img)
        # logits = self.fc(F.normalize(conv5_out, p=2, dim=-1))
        return conv1_out, conv2_out, conv3_out


class spatial_cnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(11, 11, 1), stride=(2, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 1)),
            nn.Conv3d(16, 32, kernel_size=(11, 11, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 1)),
            nn.Conv3d(32, 64, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(7, 7, 1), stride=(2, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1), padding=1),
            nn.Conv3d(16, 32, kernel_size=(7, 7, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1), padding=1),
            nn.Conv3d(32, 64, kernel_size=(7, 7, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1), padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(3, 3, 1), stride=(2, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1), padding=1),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1), padding=1),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1), padding=1)
        )

    def forward(self, slices1, slices2, slices3):
        conv1_out = self.conv1(slices1)
        conv2_out = self.conv1(slices2)
        conv3_out = self.conv1(slices3)
        # logits = self.fc(F.normalize(conv5_out, p=2, dim=-1))
        return (conv1_out + conv2_out + conv3_out)


class MiSePyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.slice_cnn_axial = slice_cnn(91)
        self.spatial_cnn_axial = spatial_cnn()
        self.slice_cnn_col = slice_cnn(109)
        self.spatial_cnn_col = spatial_cnn()
        self.slice_cnn_sag = slice_cnn(91)
        self.spatial_cnn_sag = spatial_cnn()
        # self.fc = nn.Sequential(
        #     nn.Linear(320, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(64, 2)
        # )

    def forward(self, img):
        # prepare 3 views
        img_axial = img
        img_cor = img.permute(0, 1, 2, 4, 3)
        img_sag = img.permute(0, 1, 4, 3, 2)

        # forward 3 views
        axial_conv1_out, axial_conv2_out, axial_conv3_out = self.slice_cnn_axial(img_axial)
        axial_out = self.spatial_cnn_axial(axial_conv1_out, axial_conv2_out, axial_conv3_out)
        col_conv1_out, col_conv2_out, col_conv3_out = self.slice_cnn_col(img_cor)
        col_out = self.spatial_cnn_col(col_conv1_out, col_conv2_out, col_conv3_out)
        sag_conv1_out, sag_conv2_out, sag_conv3_out = self.slice_cnn_sag(img_sag)
        sag_out = self.spatial_cnn_sag(sag_conv1_out, sag_conv2_out, sag_conv3_out)

        feat_out = torch.cat([axial_out.view(axial_out.shape[0], -1), col_out.view(col_out.shape[0], -1),
                              sag_out.view(sag_out.shape[0], -1)], dim=1)

        # forward fc layer
        # logits = self.fc(feat_out)
        return feat_out


class Mnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mri = MiSePyNet()
        self.pet = MiSePyNet()
        self.fc = nn.Sequential(
            nn.Linear(640, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, mri, pet):
        # prepare 3 views
        mri_feat = self.mri(mri)
        pet_feat = self.pet(pet)
        logits = torch.cat([mri_feat, pet_feat], dim=-1)
        # forward fc layer
        logits = self.fc(logits)
        return logits
