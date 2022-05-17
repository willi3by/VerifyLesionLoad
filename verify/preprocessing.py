import os
import ants
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from glob import glob
import re
import matplotlib.pyplot as plt
import seaborn

def hd_bet_subject(subj_image_path):
    cmd = 'hd-bet -i ' + subj_image_path + ' -device cpu -mode fast -tta 0'
    os.system(cmd)

def align_lesion_mask_to_mni(root_dir, subj_image_path, template_image_path, lesion_mask):
    #Align template to subject space then apply transform to get lesion in
    #template space.
    #Create condition for emask.

    template_img = ants.image_read(template_image_path)
    subj_img = ants.image_read(subj_image_path)
    subj_lesion_mask = ants.image_read(lesion_mask)
    subj_lesion_mask_inv = 1 - subj_lesion_mask.numpy()
    subj_lesion_mask_inv_ants = subj_lesion_mask.new_image_like(subj_lesion_mask_inv)

    mytx = ants.registration(fixed=subj_img, moving=template_img,
                             mask=subj_lesion_mask_inv_ants,
                             type_of_transform='SyN')

    warped_lesion = ants.apply_transforms(fixed = template_img, moving = subj_lesion_mask,
                                          transformlist = mytx['invtransforms'],
                                          interpolator='nearestNeighbor')
    write_lesion_path = root_dir + 'lesion_to_mni.nii'
    ants.image_write(warped_lesion, filename=write_lesion_path)
    return(warped_lesion)

def preprocess_SMATT_files(path_to_smatt_files):
    smatt_key = path_to_smatt_files + 'SMATT_key.txt'
    f = open(smatt_key, 'r')
    answer = {}
    for line in f:
        k, v = line.strip().split(':')
        answer[k.strip()] = v.strip()
    f.close()
    all_tracts = [answer.get(key) for key in answer.keys()]
    all_tracts_text = ' '.join(all_tracts).replace(',', ' ')
    all_tracts_list = all_tracts_text.split(' ')
    tracts = set(all_tracts_list)
    keys = answer.keys()
    smatt_nii = path_to_smatt_files + 'S-MATT.nii'
    smatt_img = nib.load(smatt_nii)
    affine = smatt_img.affine
    for tract in tracts:
        smatt_tract = smatt_img.get_fdata()
        smatt_tract = np.round(smatt_tract)
        for key in keys:
            tracts_in_vox = answer.get(key)
            num_tracts_in_vox = len(answer.get(key).split(','))
            if tract in tracts_in_vox:
                vox_val = 1 / num_tracts_in_vox
            else:
                vox_val = 0
            np.place(smatt_tract, smatt_tract == int(key), vox_val)
        filename = path_to_smatt_files + tract + '_prob.nii'
        tract_nii = nib.Nifti1Image(smatt_tract, affine)
        nib.save(tract_nii, filename)

