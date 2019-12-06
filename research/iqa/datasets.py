import os
import sys

import numpy as np
from PIL import Image
from skimage import io
from skimage.color import rgb2gray
from torch.utils.data import Dataset

sys.path.append('../')
from iqa.cfg import cfg


class ImageQualityDataset(Dataset):
    """
    Image Quality Dataset
    """

    def __init__(self, type='train', transform=None):
        assert type in ['train', 'val']
        files = []
        types = []

        for quality_dir in os.listdir(os.path.join(cfg['iqa_img_base'], type)):
            for img_f in os.listdir(os.path.join(cfg['iqa_img_base'], type, quality_dir)):
                files.append(os.path.join(cfg['iqa_img_base'], type, quality_dir, img_f))
                types.append(0 if quality_dir == 'LR' else 1)

        self.files = files
        self.types = types
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        image = io.imread(img_name)

        #  if image.shape[-1] > 1:
        #      image = rgb2gray(image)

        sample = {'image': image, "type": self.types[idx], 'filename': img_name}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample
