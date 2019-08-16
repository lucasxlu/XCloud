import os
import sys

import numpy as np
from PIL import Image
from skimage import io
from skimage.color import gray2rgb, rgba2rgb
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

sys.path.append('../')
from research.garbageclassification.cfg import cfg


class GarbageDataset(Dataset):
    """
    garbage dataset
    """

    def __init__(self, type, transform=None):
        files = []
        lbs = []

        for txt in os.listdir(cfg['garbage_classification_root']):
            with open(os.path.join(cfg['garbage_classification_root'], 'train_data', txt), mode='rt') as f:
                files.append(os.path.join(cfg['garbage_classification_root'], 'train_data',
                                          ''.join(f.readlines()).split(',')[0].strip()))
                lbs.append(int(''.join(f.readlines()).split(',')[1].strip()))

        X_train, X_test, y_train, y_test = train_test_split(files, lbs, test_size=0.2, stratify=lbs)
        if type == 'train':
            self.filelist = X_train
            self.typelist = y_train
        else:
            self.filelist = X_test
            self.typelist = y_test

        self.transform = transform

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        img_name = self.filelist[idx]

        image = io.imread(img_name)

        if len(list(image.shape)) < 3:
            image = gray2rgb(image)
        elif len(list(image.shape)) > 3:
            image = rgba2rgb(image)

        sample = {'image': image, "type": self.typelist[idx], 'filename': img_name}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample
