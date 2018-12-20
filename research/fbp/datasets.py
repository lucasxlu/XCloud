"""
definition of datasets
Author: XuLu
"""
import os
import sys

import numpy as np
from PIL import Image
from skimage import io
from torch.utils.data import Dataset

sys.path.append('../')
from fbp.cfg import cfg


class SCUTFBP5500Dataset(Dataset):
    """
    SCUTFBP5500 dataset
    """

    def __init__(self, train=True, transform=None):
        """
        PyTorch Dataset definition
        :param train:
        :param transform:
        """
        cv64_txt_dir = '/'.join(
            cfg['image_dir'].split('/')[0:-1]) + '/train_test_files/split_of_60%training and 40%testing'

        if train:
            array = []
            with open(os.path.join(cv64_txt_dir, 'train.txt'), mode='rt') as f:
                for l in f.readlines():
                    if l.strip() != '':
                        array.append([l.split(' ')[0], float(l.split(' ')[1].strip())])

            self.img_files = [os.path.join(cfg['image_dir'], array[i][0]) for i in range(len(array))]
            self.labels = [float(array[i][1]) for i in range(len(array))]
        else:
            array = []
            with open(os.path.join(cv64_txt_dir, 'test.txt'), mode='rt') as f:
                for l in f.readlines():
                    if l.strip() != '':
                        array.append([l.split(' ')[0], float(l.split(' ')[1].strip())])

            self.img_files = [os.path.join(cfg['image_dir'], array[i][0]) for i in range(len(array))]
            self.labels = [float(array[i][1]) for i in range(len(array))]

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = io.imread(self.img_files[idx])
        label = self.labels[idx]

        sample = {'image': image, 'label': label, 'class': round(label) - 1, 'filename': self.img_files[idx]}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample
