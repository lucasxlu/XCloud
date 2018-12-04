import sys
import os
import math

import numpy as np
import pandas as pd
from skimage import io
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

sys.path.append('../')
from research.age.cfg import cfg


class UTKFaceDataset(Dataset):
    """
    UTKFace dataset
    """

    def __init__(self, type='train', transform=None):
        files = os.listdir(os.path.join(cfg['root'], 'UTKFace'))
        ages = [int(fname.split("_")[0]) for fname in files]

        train_files, test_files, train_ages, test_ages = train_test_split(files, ages, test_size=0.2, random_state=42)
        train_files, val_files, train_ages, val_ages = train_test_split(files, ages, test_size=0.05, random_state=2)

        if type == 'train':
            self.filelist = train_files
            self.agelist = train_ages
            self.genderlist = [int(fname.split("_")[1]) for fname in train_files]
            self.racelist = [int(fname.split("_")[2]) if len(fname.split("_")[2]) == 1 else 4 for fname in train_files]
        elif type == 'val':
            self.filelist = val_files
            self.agelist = val_ages
            self.genderlist = [int(fname.split("_")[1]) for fname in val_files]
            self.racelist = [int(fname.split("_")[2]) if len(fname.split("_")[2]) == 1 else 4 for fname in val_files]
        elif type == 'test':
            self.filelist = test_files
            self.agelist = test_ages
            self.genderlist = [int(fname.split("_")[1]) for fname in test_files]
            self.racelist = [int(fname.split("_")[2]) if len(fname.split("_")[2]) == 1 else 4 for fname in test_files]
        else:
            print('Invalid data type. It can only be train/val/test...')

        self.transform = transform

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        img_name = os.path.join(cfg['root'], 'UTKFace', self.filelist[idx])

        image = io.imread(img_name)
        sample = {'image': image, 'age': self.agelist[idx], "gender": self.genderlist[idx],
                  "race": self.racelist[idx]}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample
