import os

from PIL import Image
from torchvision.datasets import VisionDataset


class ShelfLaminateSegDataset(VisionDataset):
    """ShelfLaminate Segmentation Dataset.

    Args:
        root (string): Root directory of the ShelfLaminate Segmentation Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self,
                 root,
                 image_set='train',
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(ShelfLaminateSegDataset, self).__init__(root, transforms, transform, target_transform)
        self.filename = 'ShelfLaminateSegDataset'
        self.image_set = image_set
        base_dir = os.path.join(root, self.filename)
        image_dir = os.path.join(base_dir, 'JPEGImages')
        mask_dir = os.path.join(base_dir, 'Masks')

        if not os.path.isdir(base_dir):
            raise RuntimeError('Dataset not found or corrupted.')

        splits_dir = os.path.join(base_dir, 'ImageSets/Segmentation')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)
