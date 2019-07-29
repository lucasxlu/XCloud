import os
import sys

import numpy as np
from PIL import Image
from skimage import io
from skimage.color import gray2rgb, rgba2rgb
from torch.utils.data import Dataset

sys.path.append('../')
from research.intelligentagriculture.cfg import cfg


class IP102Dataset(Dataset):
    """
    IP102 dataset
    """

    def __init__(self, type, transform=None):
        self.filelist = []
        self.typelist = []

        with open(os.path.join(cfg['ip102_classification_root'], '{}.txt'.format(type)), mode='rt',
                  encoding='utf-8') \
                as f:
            for _ in f.readlines():
                self.filelist.append(os.path.join(cfg['ip102_classification_root'], 'Images', _.split(' ')[0]))
                self.typelist.append(int(_.split(' ')[-1]))

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
