import numpy as np
import torch
import torchio as tio
import torch.nn.functional as F
import monai.transforms as montrans
import math

def rescale_intensity_3D(img):
    img = img[np.newaxis, :, :, :]
    img = tio.RescaleIntensity(out_min_max=(0, 1))(img)
    img = img.squeeze(0)
    return img


def CropOrPad_3D(img, resolution):
    img = img[np.newaxis, :, :, :]
    img = tio.CropOrPad(resolution)(img)
    img = img.squeeze(0)
    return img


def process_tabular_data(tabular_data, label=None):
    
    # handle the missing entries in the tabular data: append the missing indicator mask at the end of the tabular data
    # tabular_data_attr : ['AGE', 'GENDER', 'EDUCATION', 'CSFVol', 'TotalGrayVol', 'CorticalWhiteMatterVol',
    #     'Left-Hippocampus', 'Right-Hippocampus', 'rh_entorhinal_thickness', 'lh_entorhinal_thickness',
    #     'APOE4', 'MMSE', 'ADAS13']
    
    tabular_data = np.array(tabular_data).astype(np.float32)
    
    # for adni data we have all 13 entries
    if len(tabular_data) == 13:
        tabular_data = tabular_data[:13]
        tabular_mask = np.isnan(tabular_data)
        tabular_mask = np.logical_not(tabular_mask)
        tabular_data = np.nan_to_num(tabular_data, copy=False)  
        # concat the mask to the tabular data
        tabular_data = np.concatenate((tabular_data, tabular_mask), axis=0)
    
    else:
        raise ValueError("The length of tabular data is 13.")
  
    tabular_data = torch.from_numpy(tabular_data)
    return tabular_data

