import pandas as pd
import os
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    SaveImaged,
    ScaleIntensityd,
    SpatialCropd,
    SpatialPadd,
    RandFlipd,
    EnsureTyped, RandRotated, RandZoomd,
)


class ADNI:
    def __init__(self, dataroot, label_filename, task):
        self.csv = pd.read_csv(os.path.join(dataroot, label_filename))
        self.labels = None
        self.label_dict = None
        self.data_dict = None
        mri_dir = os.path.join(dataroot, 'MRI')
        pet_dir = os.path.join(dataroot, 'PET')

        # get data and labels according to specific task
        if task == 'ADCN':
            self.labels = self.csv[(self.csv['Group'] == 'AD') | (self.csv['Group'] == 'CN')]
            self.label_dict = {'CN': 0, 'AD': 1}
        if task == 'pMCIsMCI':
            self.labels = self.csv[(self.csv['Group'] == 'pMCI') | (self.csv['Group'] == 'sMCI')]
            self.label_dict = {'sMCI': 0, 'pMCI': 1}
        if task == 'MCICN':
            self.labels = self.csv[
                (self.csv['Group'] == 'pMCI') | (self.csv['Group'] == 'sMCI') | (self.csv['Group'] == 'MCI') | (
                            self.csv['Group'] == 'CN')]
            self.label_dict = {'CN': 0, 'sMCI': 1, 'pMCI': 1, 'MCI': 1}

        # create data dic
        subject_list = self.labels['Subject'].tolist()
        label_list = self.labels['Group'].tolist()
        age_list = self.labels['Age'].tolist()
        self.data_dict = [{'MRI': os.path.join(mri_dir, subject_name + '.nii.gz'),
                           'PET': os.path.join(pet_dir, subject_name + '.nii.gz'),
                           'label': self.label_dict[subject_label],
                           'age': subject_age,
                           'Subject': subject_name}
                          for subject_name, subject_label, subject_age in zip(subject_list, label_list, age_list)]

    def __len__(self):
        return len(self.labels)

    def get_weights(self):
        label_list = []
        for item in self.data_dict:
            label_list.append(item['label'])
        return float(label_list.count(0)), float(label_list.count(1))


def ADNI_transform(aug='True'):
    if aug == 'True':
        train_transform = Compose([
                    LoadImaged(keys=['MRI', 'PET']),
                    EnsureChannelFirstd(keys=['MRI', 'PET']),
                    ScaleIntensityd(keys=['MRI', 'PET']),
                    # Augment
                    RandFlipd(keys=['MRI', 'PET'], prob=0.3, spatial_axis=0),
                    RandRotated(keys=['MRI', 'PET'], prob=0.3, range_x=0.05),
                    RandZoomd(keys=['MRI', 'PET'], prob=0.3, min_zoom=0.95, max_zoom=1),
                    EnsureTyped(keys=['MRI', 'PET'])
                ])
    else:
        train_transform = Compose([
                    LoadImaged(keys=['MRI', 'PET']),
                    EnsureChannelFirstd(keys=['MRI', 'PET']),
                    ScaleIntensityd(keys=['MRI', 'PET']),
                    EnsureTyped(keys=['MRI', 'PET'])
                ])
    test_transform = Compose([
                LoadImaged(keys=['MRI', 'PET']),
                EnsureChannelFirstd(keys=['MRI', 'PET']),
                ScaleIntensityd(keys=['MRI', 'PET']),
                EnsureTyped(keys=['MRI', 'PET'])
            ])
    return train_transform, test_transform


def ADNI_transform_Mnet(aug='True'):
    if aug == 'True':
        train_transform = Compose([
                    LoadImaged(keys=['MRI', 'PET']),
                    EnsureChannelFirstd(keys=['MRI', 'PET']),
                    ScaleIntensityd(keys=['MRI', 'PET']),
                    SpatialPadd(keys=['MRI', 'PET'], spatial_size=(91,109,91)),
                    # Augment
                    RandFlipd(keys=['MRI', 'PET'], prob=0.3, spatial_axis=0),
                    RandRotated(keys=['MRI', 'PET'], prob=0.3, range_x=0.05),
                    RandZoomd(keys=['MRI', 'PET'], prob=0.3, min_zoom=0.95, max_zoom=1),
                    EnsureTyped(keys=['MRI', 'PET'])
                ])
    else:
        train_transform = Compose([
                    LoadImaged(keys=['MRI', 'PET']),
                    EnsureChannelFirstd(keys=['MRI', 'PET']),
                    ScaleIntensityd(keys=['MRI', 'PET']),
                    SpatialPadd(keys=['MRI', 'PET'], spatial_size=(91, 109, 91)),
                    EnsureTyped(keys=['MRI', 'PET'])
                ])
    test_transform = Compose([
                LoadImaged(keys=['MRI', 'PET']),
                EnsureChannelFirstd(keys=['MRI', 'PET']),
                ScaleIntensityd(keys=['MRI', 'PET']),
                SpatialPadd(keys=['MRI', 'PET'], spatial_size=(91, 109, 91)),
                EnsureTyped(keys=['MRI', 'PET'])
            ])
    return train_transform, test_transform

def ADNI_transform_ADVIT(aug='True'):
    train_transform = Compose([
                    LoadImaged(keys=['MRI', 'PET']),
                    EnsureChannelFirstd(keys=['MRI', 'PET']),
                    ScaleIntensityd(keys=['MRI', 'PET']),
                    SpatialPadd(keys=['MRI', 'PET'], spatial_size=(128, 128, 79)),
                    EnsureTyped(keys=['MRI', 'PET'])
                ])
    test_transform = Compose([
                LoadImaged(keys=['MRI', 'PET']),
                EnsureChannelFirstd(keys=['MRI', 'PET']),
                ScaleIntensityd(keys=['MRI', 'PET']),
                SpatialPadd(keys=['MRI', 'PET'], spatial_size=(128, 128, 79)),
                EnsureTyped(keys=['MRI', 'PET'])
            ])
    return train_transform, test_transform

