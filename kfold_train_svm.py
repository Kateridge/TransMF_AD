import numpy as np
import pandas as pd
import os
import glob
from MKLpy.preprocessing import normalization, rescale_01
from MKLpy import generators
from MKLpy.metrics import pairwise
from MKLpy.preprocessing import kernel_normalization


root_dir = '/media/kateridge/data/Datasets/ADNI_FDG/CAPS'
mri_paths = glob.glob(os.path.join(root_dir, 'subjects', '*', 'ses-M00', 't1', 'spm',
                                   'dartel', 'group-ADNI', 'atlas_statistics', '*AAL2*'))
mri_ROIs = []
pet_ROIs = []
y = []

csv = pd.read_csv('../Datasets/ADNI_Yongsheng/ADNI.csv')
label_map_dic = {'pMCI': 1, 'sMCI': 0}

for mri_path in mri_paths:
    sub_id_original = mri_path.split('sub-ADNI')[1].split('/ses-M00')[0]
    sub_id_original_list = sub_id_original.split('S')
    sub_id = sub_id_original_list[0] + '_S_' + sub_id_original_list[1]
    # get pet path
    pet_path = glob.glob(os.path.join(root_dir, 'subjects', 'sub-ADNI'+sub_id_original, 'ses-M00', 'pet', 'preprocessing',
                                   'group-ADNI', 'atlas_statistics', '*AAL2*'))[0]

    label = csv[csv['Subject'] == sub_id]
    if label.empty:
        continue
    label = label.iloc[0].at['Group']
    if label == 'AD' or label == 'CN' or label == 'MCI':
        continue
    label = label_map_dic[label]
    mri_roi_csv = pd.read_csv(mri_path, sep='\t')
    mri_rois = mri_roi_csv.iloc[:, 2].tolist()[1:]
    pet_roi_csv = pd.read_csv(pet_path, sep='\t')
    pet_rois = pet_roi_csv.iloc[:, 2].tolist()[1:]
    mri_ROIs.append(mri_rois)
    pet_ROIs.append(pet_rois)
    y.append(label)

print(len(mri_ROIs))
print(len(pet_ROIs))
print(len(y))
y = np.array(y)

X_mri = normalization(rescale_01(mri_ROIs))
X_pet = normalization(rescale_01(pet_ROIs))
KL = generators.Multiview_generator([X_mri, X_pet], kernel=pairwise.linear_kernel, include_identity=True)
KL_norm = [kernel_normalization(K) for K in KL]

# from MKLpy.model_selection import train_test_split
# KLtr, KLte, Ytr, Yte = train_test_split(KL_norm, y, test_size=.3, random_state=42)
# mkl = AverageMKL().fit(KLtr, Ytr)       #combine kernels and train the classifier
# y_preds  = mkl.predict(KLte)            #predict the output class
# y_scores = mkl.decision_function(KLte)  #returns the projection on the distance vector

# from sklearn.metrics import accuracy_score, roc_auc_score
# accuracy = accuracy_score(Yte, y_preds)
# roc_auc = roc_auc_score(Yte, y_scores)
# print('Accuracy score: %.4f, roc AUC score: %.4f' % (accuracy, roc_auc))

from sklearn.model_selection import StratifiedKFold as KFold
from MKLpy.algorithms import CKA
from sklearn.metrics import precision_recall_curve, average_precision_score,\
    roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, roc_auc_score
mkl = CKA()
n = len(y)
cv = KFold(5, random_state=996, shuffle=True)
results = []

def sensitivityCalc(Predictions, Labels):
    MCM = confusion_matrix(Labels, Predictions,
                                      sample_weight=None,
                                      labels=None)
    # MCM此处是 5 * 2 * 2的混淆矩阵（ndarray格式），5表示的是5分类

    # 切片操作，获取每一个类别各自的 tn, fp, tp, fn
    tn_sum = MCM[0, 0] # True Negative
    fp_sum = MCM[0, 1] # False Positive

    tp_sum = MCM[1, 1] # True Positive
    fn_sum = MCM[1, 0] # False Negative

    # 这里加1e-6，防止 0/0的情况计算得到nan，即tp_sum和fn_sum同时为0的情况
    Condition_negative = tp_sum + fn_sum + 1e-6

    sensitivity = tp_sum / Condition_negative
    macro_sensitivity = np.average(sensitivity, weights=None)

    micro_sensitivity = np.sum(tp_sum) / np.sum(tp_sum+fn_sum)

    return macro_sensitivity

def specificityCalc(Predictions, Labels):
    MCM = confusion_matrix(Labels, Predictions, sample_weight=None, labels=None)
    tn_sum = MCM[0, 0]
    fp_sum = MCM[0, 1]

    tp_sum = MCM[1, 1]
    fn_sum = MCM[1, 0]

    Condition_negative = tn_sum + fp_sum + 1e-6

    Specificity = tn_sum / Condition_negative
    macro_specificity = np.average(Specificity, weights=None)

    micro_specificity = np.sum(tn_sum) / np.sum(tn_sum+fp_sum)

    return macro_specificity


for train, test in cv.split(y, y):
    KLtr = [K[train][:, train] for K in KL]
    KLte = [K[test][:, train] for K in KL]
    clf = mkl.fit(KLtr, y[train])
    y_preds  = mkl.predict(KLte)
    y_scores = mkl.decision_function(KLte)
    accuracy = accuracy_score(y[test], y_preds)
    roc_auc = roc_auc_score(y[test], y_scores)
    f1 = f1_score(y[test], y_preds)
    sen = sensitivityCalc(y_preds, y[test])
    spe = specificityCalc(y_preds, y[test])
    results.append([accuracy, sen, spe, f1, roc_auc])

results = np.array(results)
res_mean = np.mean(results, axis=0)
res_std = np.std(results, axis=0)
print(f'************Final Results************')
print(f'acc: {res_mean[0]:.4f} +- {res_std[0]:.4f}\n'
      f'sen: {res_mean[1]:.4f} +- {res_std[1]:.4f}\n'
      f'spe: {res_mean[2]:.4f} +- {res_std[2]:.4f}\n'
      f'f1: {res_mean[3]:.4f} +- {res_std[3]:.4f}\n'
      f'auc: {res_mean[4]:.4f} +- {res_std[4]:.4f}\n')


