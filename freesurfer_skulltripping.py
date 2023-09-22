import os
import glob

# ---------------------Step 1-----------------------------
# perform N3 NU Intensity Correction， Normalization(make white matter averaged to 110) and Skull Strip
# ls ./sub-ADNI*/ses-M00/anat/*.nii.gz | parallel -j 8 recon-all -i {} -s {.} -autorecon1 -notalairach

# ---------------------Step 2-----------------------------
# # Set up environment of FreeSurfer
# a = "export FREESURFER_HOME=/home/kateridge/freesurfer;"
# b = "source $FREESURFER_HOME/SetUpFreeSurfer.sh;"
#
# # Convert .mgz into .nii.gz
# input_path = "/media/kateridge/data/Datasets/ADNI_FDG/BIDS_SkullTripped" # input data root
# out_path = "/media/kateridge/data/Datasets/ADNI_FDG/ADNI/MRI" # output data root
# mri_skulltripped = glob.glob(os.path.join(input_path, "*", 'mri', 'brainmask.mgz')) # glob all MRI files
# mri_T1 = glob.glob(os.path.join(input_path, "*", 'mri', 'T1.mgz'))
# for file in mri_skulltripped:
#     file_input_brainmask = file
#     file_input_t1 = file.replace('brainmask', 'T1')
#     # get subject id
#     sub_id = file.split('ADNI')[-1]
#     sub_id = sub_id.split('_ses')[0]
#     cmd = 'mkdir ' + os.path.join(out_path, sub_id)
#     os.system(cmd)
#     # get filename
#     filename = file.split('/')[-1]
#     filename = filename.replace('.mgz', '.nii.gz')
#     # path join
#     file_out_brainmask = os.path.join(out_path, sub_id, filename)
#     file_out_T1 = file_out_brainmask.replace('brainmask', 'T1')
#     # convert
#     cmd = a + b + 'mri_convert ' + file_input_brainmask + ' ' + file_out_brainmask
#     os.system(cmd)
#     cmd = a + b + 'mri_convert ' + file_input_t1 + ' ' + file_out_T1
#     os.system(cmd)

# ---------------------Step 3-----------------------------
# FSL processing.
print("FSL start......\n")
cmd = 'source ~/.bashrc'
os.system(cmd)
input_path ="/media/kateridge/data/Datasets/ADNI_FDG/ADNI" # input data root
out_path="/media/kateridge/data/Datasets/ADNI_FDG/ADNI" # output data root
pet_files = glob.glob(os.path.join(input_path, 'PET', '*')) # glob all MRI files
mni_template = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'
a = "export FREESURFER_HOME=/home/kateridge/freesurfer;"
b = "source $FREESURFER_HOME/SetUpFreeSurfer.sh;"
print(len(pet_files))
for file in pet_files:
    sub_id = file.split('/')[-1]
    # 1) Rigid register PET to skulltripped MRI
    ref_img = os.path.join(input_path, 'MRI', sub_id, 'brainmask.nii.gz')
    mov_img = os.path.join(input_path, 'PET', sub_id, 'pet_source.nii.gz')
    out_img = os.path.join(out_path, 'PET', sub_id, 'pet_reg2mri.nii.gz')
    out_mat = os.path.join(out_path, 'PET', sub_id, 'pet_reg2mri.mat')
    print('----------Process Subject----------- ' + sub_id)
    print('1) Rigid register PET to MRI')
    cmd = a + b + 'flirt -ref ' + ref_img + ' -in ' + mov_img + ' -out ' + out_img + ' -omat ' + out_mat + ' -dof 6'
    os.system(cmd)

    # 2) Affine register MRI to MNI152-2mm template
    ref_img = mni_template
    mov_img = os.path.join(input_path, 'MRI', sub_id, 'brainmask.nii.gz')
    out_img = os.path.join(out_path, 'MRI', sub_id, 'T1_mni.nii.gz')
    out_mat = os.path.join(out_path, 'MRI', sub_id, 'mri_reg2mni.mat')
    print('2) Affine register MRI to MNI152 template')
    cmd = a + b + 'flirt -ref ' + ref_img + ' -in ' + mov_img + ' -out ' + out_img + ' -omat ' + out_mat + ' -dof 12'
    os.system(cmd)

    # 3) Affine register PET to MNI152-2mm used affine parameters in 2)
    ref_img = mni_template
    mov_img = os.path.join(input_path, 'PET', sub_id, 'pet_reg2mri.nii.gz')
    out_img = os.path.join(out_path, 'PET', sub_id, 'PET_mni.nii.gz')
    init_mat = os.path.join(out_path, 'MRI', sub_id, 'mri_reg2mni.mat')
    print('3) Transform PET to MNI152 template')
    cmd = a + b + 'flirt -ref ' + ref_img + ' -in ' + mov_img + ' -out ' + out_img + ' -init ' + init_mat + ' -applyxfm'
    os.system(cmd)

print("\n\nEnd")