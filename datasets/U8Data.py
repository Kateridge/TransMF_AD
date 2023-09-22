import pandas as pd
import os
import glob
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    SaveImaged,
    ScaleIntensityd,
    SpatialCropd,
    RandFlipd,
    EnsureTyped,
)


class U8Data:
    def __init__(self, dataroot, label_filename, task):
        self.csv = pd.read_csv(os.path.join(dataroot, label_filename))
        self.labels = []
        self.label_dict = []
        self.data_dict = []

        # get data and labels according to specific task
        if task == 'ADCN':
            self.labels = self.csv[(self.csv['Diag'] == 'AD') | (self.csv['Diag'] == 'NC') |
                                   (self.csv['Diag'] == 'NC(SCS1)') | (self.csv['Diag'] == 'NC(SCS2)') |
                                   (self.csv['Diag'] == 'NC(SCS3)') | (self.csv['Diag'] == 'SCD') |
                                   (self.csv['Diag'] == 'SCD1') | (self.csv['Diag'] == 'SCD2') |
                                   (self.csv['Diag'] == 'SCD3')]
            self.label_dict = {'NC':0, 'NC(SCS1)':0, 'NC(SCS2)':0, 'NC(SCS3)':0,
                               'SCD':0, 'SCD1':0, 'SCD2':0, 'SCD3':0, 'AD': 1}
        if task == 'MCICN':
            self.labels = self.csv[(self.csv['Diag'] == 'sMCI') | (self.csv['Diag'] == 'NC') |
                                   (self.csv['Diag'] == 'NC(SCS1)') | (self.csv['Diag'] == 'NC(SCS2)') |
                                   (self.csv['Diag'] == 'NC(SCS3)') | (self.csv['Diag'] == 'SCD') |
                                   (self.csv['Diag'] == 'SCD1') | (self.csv['Diag'] == 'SCD2') |
                                   (self.csv['Diag'] == 'SCD3') | (self.csv['Diag'] == 'MCI') |
                                   (self.csv['Diag'] == 'eMCI') | (self.csv['Diag'] == 'aMCI') |
                                   (self.csv['Diag'] == 'mMCI')]
            self.label_dict = {'NC':0, 'NC(SCS1)':0, 'NC(SCS2)':0, 'NC(SCS3)':0, 'SCD':0, 'SCD1':0, 'SCD2':0, 'SCD3':0,
                               'MCI': 1, 'sMCI':1, 'eMCI':1, 'aMCI':1, 'mMCI':1}

        if task == 'pMCIsMCI':
            self.labels = []
            print('ERROR: U8data does not support pMCIsMCI task!')

        if task == 'pretrain':
            self.labels = self.csv
            self.label_dict = {'NC':0, 'NC(SCS1)':0, 'NC(SCS2)':0, 'NC(SCS3)':0, 'SCD':0, 'SCD1':0, 'SCD2':0, 'SCD3':0,
                               'MCI': 1, 'sMCI':1, 'eMCI':1, 'aMCI':1, 'mMCI':1, 'AD': 1}


        # create data dic
        subject_list = self.labels['ID'].tolist()
        label_list = self.labels['Diag'].tolist()
        age_list = self.labels['Age'].tolist()

        self.data_dict = [{'MRI': os.path.join(dataroot, subject_name, 'MRI', 'U8T1brain.nii.gz'),
                           'PET': glob.glob(os.path.join(dataroot, subject_name, 'FDG', 'U8PET*.nii.gz'))[-1],
                           'label': self.label_dict[subject_label],
                           'age': subject_age,
                           'Subject': subject_name}
                          for subject_name, subject_label, subject_age in zip(subject_list, label_list, age_list)]
        # for subject_name, subject_label, subject_age in zip(subject_list, label_list, age_list):
        #     if subject_label == 'AMCI':
        #         print(subject_name)



    def __len__(self):
        return len(self.labels)

    def get_weights(self):
        label_list = []
        for item in self.data_dict:
            label_list.append(item['label'])
        return float(label_list.count(0)), float(label_list.count(1))


def U8Data_transform():
    train_transform = Compose(
        [
            LoadImaged(keys=['MRI', 'PET']),
            AddChanneld(keys=['MRI', 'PET']),
            ScaleIntensityd(keys=['MRI', 'PET']),
            SpatialCropd(keys=['MRI', 'PET'], roi_center=(90, 111, 81), roi_size=(148, 182, 162)),
            # RandFlipd(keys=['MRI', 'PET'], prob=0.3, spatial_axis=0),
            EnsureTyped(keys=['MRI', 'PET'])
        ]
    )
    test_transform = Compose(
        [
            LoadImaged(keys=['MRI', 'PET']),
            AddChanneld(keys=['MRI', 'PET']),
            ScaleIntensityd(keys=['MRI', 'PET']),
            SpatialCropd(keys=['MRI', 'PET'], roi_center=(90, 111, 81), roi_size=(148, 182, 162)),
            EnsureTyped(keys=['MRI', 'PET'])
        ]
    )
    return train_transform, test_transform


# U8_dataset = U8Data(dataroot='../../Datasets/U8Data', label_filename='label.csv', task='MCICN')
# train_transform, test_transform = U8Data_transform()
# print(len(U8_dataset))
