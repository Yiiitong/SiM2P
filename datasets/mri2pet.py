import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T, utils
from torch import nn
import torch.nn.functional as F
import h5py
import monai.transforms as montrans
import torchio as tio
import os
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

from .data_utils import process_tabular_data

LOG = logging.getLogger(__name__)
DIAGNOSIS_MAP = {"CN": 0, "Dementia": 1, "AD": 1, "MCI": 2, "FTD": 2}

class Mri2petDataset(Dataset):
    def __init__(
        self,
        img_size = 80, 
        data_path = ['/dataset1','/dataset2','/dataset3'], # list of data paths
        standardized_tabular = True, # whether to standardize the tabular data
        random_flip = False, # whether to do random flip in data augmentation
        with_SSP = False, # if there is 3D-SSP available in the data
        stage = 'train', # 'train', 'val', or 'test'
        target_modality = 'PET', # 'PET' or 'PETSSP', if 'PETSSP', then the model will take both PET and SSP as target
    ):
        super().__init__()

        self.data_path = data_path
        self.standardized_tabular = standardized_tabular
        self.random_flip = random_flip
        self.with_SSP = with_SSP
        self.stage = stage
        self.target_modality = target_modality

        if stage == 'train':
            self.data_path = [data_path + 'train.h5' for data_path in self.data_path]

        elif stage == 'val':
            self.data_path = [data_path + 'valid.h5' for data_path in self.data_path]

        elif stage == 'test':
            self.data_path = [data_path + 'test.h5' for data_path in self.data_path]

        self.img_size = img_size

        self._load()

    def _load(self):
        mri_data = []
        pet_data = []
        diagnosis = []
        tabular_data = []
        mri_uid = []

        self.tabular_mean = []
        self.tabular_std = []

        if self.with_SSP:
            ssp_data = []

        for data_path in self.data_path: 
            if 'h5' in data_path:       
                print(f'loaded from h5 file: {data_path}')

                with h5py.File(data_path, mode='r') as file:
                    for name, group in tqdm(file.items(), total=len(file)):

                        if name == "stats":
                            self.tabular_mean.append(group["tabular/mean"][:])
                            self.tabular_std.append(group["tabular/stddev"][:])

                        else:
                            input_mri_data = group['MRI/T1/data'][:]
                            input_pet_data = group['PET/FDG/data'][:]

                            mri_data.append(input_mri_data)
                            pet_data.append(input_pet_data)

                            if self.with_SSP:
                                input_ssp_data = group['PET/SSP/data'][:]
                                ssp_data.append(input_ssp_data)

                            tabular_data.append(group['tabular'][:])
                            diagnosis.append(group.attrs['DX'])
                            mri_uid.append(name)

            else:
                raise NotImplementedError(f'Only h5 file is supported for now. {data_path} is not h5 file.')

        self.len_data = len(pet_data)
        self._image_data_mri = mri_data
        self._image_data_pet = pet_data
        if self.with_SSP:
            self._image_data_ssp = ssp_data
            print("Using 3D-SSP as additional input.")
        self._tabular_data = tabular_data
        self._mri_uid = mri_uid

        print("DATASET: ", self.data_path)
        print("SAMPLES: ", self.len_data)
        print("Input Shape: {}".format(mri_data[0].shape))

        labels, counts = np.unique(diagnosis, return_counts=True)
        print("Classes: ", pd.Series(counts, index=labels))     
        self._diagnosis = [DIAGNOSIS_MAP[d] for d in diagnosis]

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):

        mri_scan = self._image_data_mri[idx]
        pet_scan = self._image_data_pet[idx]
        if self.with_SSP:
            ssp_scan = self._image_data_ssp[idx]
        tabular_data = self._tabular_data[idx]

        label = self._diagnosis[idx]
        mri_uid = self._mri_uid[idx]
        
        if self.standardized_tabular:
            tabular_data = (tabular_data - self.tabular_mean[0]) / self.tabular_std[0]
        
        tabular_data = process_tabular_data(tabular_data)

        # data should already be in the intensity range of [0, 1]
        data_transform = montrans.Compose([
            tio.CropOrPad((112, 112, 112)),
            montrans.Resize((self.img_size, self.img_size, self.img_size)),
        ])

        data_aug = montrans.Compose([
            tio.RandomFlip(axes = (0,), flip_probability=0.5) if self.random_flip else nn.Identity(),
        ])

        mri_scan = data_transform(mri_scan[np.newaxis])
        pet_scan = data_transform(pet_scan[np.newaxis])

        # using data augmentation only in training
        if self.stage == 'train':
            mri_scan = data_aug(mri_scan)
            pet_scan = data_aug(pet_scan)

        if self.with_SSP:
            if self.img_size != ssp_scan.shape[0]:
                # reshape to (img_size, img_size, original_depth)
                ssp_scan = montrans.Resize((self.img_size, self.img_size, ssp_scan.shape[-1]))(ssp_scan[np.newaxis])
        
        if self.with_SSP:
            if self.target_modality == 'PETSSP':
                ssp_scan = montrans.Resize((self.img_size, self.img_size, self.img_size))(ssp_scan[np.newaxis])
                pet_scan = torch.cat((pet_scan, ssp_scan), dim=0)
                mri_scan = torch.cat((mri_scan, mri_scan), dim=0)
            else:
                ssp_scan = ssp_scan[np.newaxis]
                ssp_scan = ssp_scan[:,:,:,[0, -2]]
        
        output = {
            'pet_scan': pet_scan,
            'mri_scan': mri_scan,
            'idx': idx,
            'label': label,
            'mri_uid': mri_uid,
            'tabular_data': tabular_data,
        }
        if self.with_SSP:
            output['ssp_scan'] = ssp_scan

        return output

