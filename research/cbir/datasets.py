import os
import sys

import numpy as np
from PIL import Image
from skimage import io
from skimage.color import gray2rgb, rgba2rgb
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

sys.path.append('../')
from research.cbir.cfg import cfg


class TissuePhysiologyDataset(Dataset):
    """
    TissuePhysiology dataset
    """

    def __init__(self, type='train', transform=None):
        files = []
        types = []

        mp = {}
        for i, img_lb in enumerate(sorted(os.listdir(cfg['tissue_physiology_img_base']))):
            mp[i] = img_lb
            for img_f in os.listdir(os.path.join(cfg['tissue_physiology_img_base'], img_lb)):
                files.append(os.path.join(cfg['tissue_physiology_img_base'], img_lb, img_f))
                types.append(i)

        train_files, test_files, train_types, test_types = train_test_split(files, types, test_size=0.1,
                                                                            random_state=42)

        if type == 'train':
            self.filelist = train_files
            self.typelist = train_types
        elif type == 'val':
            self.filelist = test_files
            self.typelist = test_types
        elif type == 'test':
            self.filelist = test_files
            self.typelist = test_types
        else:
            print('Invalid data type. It can only be train/val/test...')

        self.transform = transform

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        img_name = self.filelist[idx]

        image = io.imread(img_name)

        if image.shape[-1] == 4:
            image = rgba2rgb(image)
        elif image.shape[-1] == 1 or len(list(image.shape)) < 3:
            image = gray2rgb(image)

        sample = {'image': image, "type": self.typelist[idx], 'filename': img_name}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class LightClothingDataset(Dataset):
    """
    LightClothingDataset
    """

    def __init__(self, type='train', transform=None):
        files = []
        types = []

        mp = {}
        for i, img_lb in enumerate(sorted(os.listdir(cfg['light_clothing_img_base']))):
            mp[i] = img_lb
            for img_f in os.listdir(os.path.join(cfg['light_clothing_img_base'], img_lb)):
                files.append(os.path.join(cfg['light_clothing_img_base'], img_lb, img_f))
                types.append(i)

        train_files, test_files, train_types, test_types = train_test_split(files, types, test_size=0.1,
                                                                            random_state=25)

        if type == 'train':
            self.filelist = train_files
            self.typelist = train_types
        elif type == 'val':
            self.filelist = test_files
            self.typelist = test_types
        elif type == 'test':
            self.filelist = test_files
            self.typelist = test_types
        else:
            print('Invalid data type. It can only be train/val/test...')

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
