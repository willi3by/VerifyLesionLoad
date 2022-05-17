import os
import ants
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from glob import glob
import re
import matplotlib.pyplot as plt
import seaborn

def calculate_lesion_load(path_to_prob_smatt_files, path_to_aligned_lesion):
    tract_files_path = path_to_prob_smatt_files + '*prob.nii'
    tract_files = glob(tract_files_path)
    all_tract_loads = {}
    for t in tract_files:
        tract_name = re.search("(?<=data/).+?(?=_)", t).group()
        tract = nib.load(t)
        tract_data = tract.get_fdata()
        lesion = nib.load(path_to_aligned_lesion)
        lesion_resamp = resample_from_to(lesion, tract)
        lesion_resamp_data = lesion_resamp.get_fdata().round()
        overlap = tract_data * lesion_resamp_data
        slice_weights = [np.count_nonzero(tract_data[...,i]) for i in range(len(tract_data[...,]))]
        max_area = np.max(slice_weights)
        lesion_load = []
        for i in range(len(overlap[...,])):
            s = np.sum(overlap[...,i])
            weighted_s = s * (max_area / slice_weights[i])
            lesion_load.append(weighted_s)
        all_tract_loads.update({ tract_name : np.nan_to_num(lesion_load)})
    return all_tract_loads

def plot_tract_loads(all_tract_loads):
    for k,v in sorted(all_tract_loads.items()):
        plt.plot(v, label=k)
    plt.xlabel('Slice (Inferior to Superior)')
    plt.ylabel('Weighted Lesion Load (# of voxels)')
    plt.title('Lesion Load by Slice')
    plt.xlim([60, 100])
    plt.ylim([0,70])
    plt.legend()
    plt.rcParams['figure.figsize'] = [10,8]
    seaborn.despine(top=True)
    plt.show()

def calculate_lesion_load_auc(all_tract_loads):
    lesion_load_auc = {}
    for k,v in sorted(all_tract_loads.items()):
        tract_auc = np.trapz(v)
        lesion_load_auc.update({k : tract_auc})
    return lesion_load_auc
