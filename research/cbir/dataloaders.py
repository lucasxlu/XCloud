import os

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from skimage.color import gray2rgb, rgba2rgb
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """
    cropped patch dataset
    """

    def __init__(self, img_dir, transform=None):
        files = []

        for img in os.listdir(img_dir):
            filename = os.path.join(img_dir, img)
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                files.append(filename)

        self.filelist = files
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

        sample = {'image': image, 'filename': img_name, 'idx': idx}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


def load_patch_dataset():
    """
    load cropped patch dataset
    :return:
    """
    batch_size = 32
    print('loading PatchDataset...')
    patch_dataset = PatchDataset(img_dir='/home/xulu/DataSet/LightClothingCrops/LightClothing',
                                 transform=transforms.Compose([
                                     transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ])
                                 )

    return DataLoader(patch_dataset, batch_size=batch_size, shuffle=False, num_workers=50, drop_last=False)
