import os
import sys

import numpy as np
from PIL import Image
from skimage import io
from skimage.color import gray2rgb, rgba2rgb
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

sys.path.append('../')
from research.kd.cfg import cfg


class ImageDataset(Dataset):
    """
    Image dataset
    """

    def __init__(self, type='train', transform=None):
        files = []
        types = []

        mp = {}
        with open('./predefined_classes.txt', mode='rt', encoding='utf-8') as f:
            for i, itm in enumerate(f.readlines()):
                mp[i] = itm.strip().replace('\n', '')

        for k, v in mp.items():
            for img in os.listdir(os.path.join(cfg['img_base'], v)):
                filename = os.path.join(cfg['img_base'], v, img)
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                    files.append(filename)
                    types.append(k)

        train_files, test_files, train_types, test_types = train_test_split(files, types, test_size=0.2, stratify=types,
                                                                            random_state=42)
        train_files, val_files, train_types, val_types = train_test_split(train_files, train_types, test_size=0.05,
                                                                          stratify=train_types, random_state=2)

        if type == 'train':
            self.filelist = train_files
            self.typelist = train_types
        elif type == 'val':
            self.filelist = val_files
            self.typelist = val_types
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

        image = io.imread(img_name, plugin='pil')

        if len(list(image.shape)) < 3:
            image = gray2rgb(image)
        elif len(list(image.shape)) > 3:
            image = rgba2rgb(image)

        sample = {'image': image, "type": self.typelist[idx], 'filename': img_name}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample
