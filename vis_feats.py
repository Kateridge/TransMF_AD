from scipy import ndimage, misc
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import SimpleITK as sitk
from sklearn.manifold import TSNE
from matplotlib import cm

mri_paths = glob('../Datasets/ADNI/MRI/NIIGZ_Origin/*')
pet_paths = glob('../Datasets/ADNI/PET/NIIGZ_Origin/*')
mri_downsampled_list = np.zeros(shape=(0, 20*24*20))
pet_downsampled_list = np.zeros(shape=(0, 20*24*20))

for mri_path in mri_paths:
    sub = sitk.ReadImage(mri_path)
    sub_np = sitk.GetArrayFromImage(sub)
    sub_np_downsampled = ndimage.zoom(sub_np, 0.25)
    sub_np_downsampled = sub_np_downsampled.reshape(1, -1)
    mri_downsampled_list = np.concatenate([mri_downsampled_list, sub_np_downsampled], axis=0)

for pet_path in pet_paths:
    sub = sitk.ReadImage(pet_path)
    sub_np = sitk.GetArrayFromImage(sub)
    sub_np_downsampled = ndimage.zoom(sub_np, 0.25)
    sub_np_downsampled = sub_np_downsampled.reshape(1, -1)
    pet_downsampled_list = np.concatenate([pet_downsampled_list, sub_np_downsampled], axis=0)

print('Downsample is done')

fig = plt.figure()
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
# print(mri_downsampled_list.shape)
# ax1.imshow(sub_np[36, :, :])
# ax2.imshow(result[9, :, :])
# plt.show()

tsne = TSNE(2, init='pca', learning_rate='auto')
tsne_proj_mri = tsne.fit_transform(mri_downsampled_list)
tsne_proj_pet = tsne.fit_transform(pet_downsampled_list)
# tsne_proj_mri = tsne.fit_transform(mri_cnn_feats)
# tsne_proj_pet = tsne.fit_transform(pet_cnn_feats)


# Plot those points as a scatter plot and label them based on the pred labels
cmap = cm.get_cmap('tab20')
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.scatter(tsne_proj[0:num_samples // 2, 0], tsne_proj[0:num_samples // 2, 1], label='MRI', alpha=0.5)
# ax.scatter(tsne_proj[num_samples // 2:, 0], tsne_proj[num_samples // 2:, 1], label='PET', alpha=0.5)
ax1.scatter(tsne_proj_mri[:, 0], tsne_proj_mri[:, 1], label='MRI', alpha=0.5)
ax1.legend(fontsize='large', markerscale=2)
ax2.scatter(tsne_proj_pet[:, 0], tsne_proj_pet[:, 1], label='PET', alpha=0.5)
ax2.legend(fontsize='large', markerscale=2)
plt.show()
plt.savefig('DataDistribution.png')