import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append('../')
from research.iqa.cfg import cfg
from research.iqa.datasets import ImageQualityDataset


def load_image_quality_data():
    """
    load ImageQualityDataset patches
    :return:
    """
    batch_size = cfg['batch_size']
    print('loading ImageQualityDataset...')
    train_dataset = ImageQualityDataset(type='train',
                                        transform=transforms.Compose([
                                            transforms.RandomCrop(32),
                                            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,
                                                                   hue=0.1),
                                            transforms.RandomRotation(90),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485], [0.229]),
                                            # transforms.RandomErasing(p=0.5, scale=(0.1, 0.3), value='random')
                                        ]))
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=50, drop_last=True,
                             pin_memory=True)

    val_dataset = ImageQualityDataset(type='val',
                                      transform=transforms.Compose([
                                          transforms.CenterCrop(32),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485], [0.229])
                                      ]))
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=50, drop_last=True,
                           pin_memory=True)

    test_dataset = ImageQualityDataset(type='val',
                                       transform=transforms.Compose([
                                           transforms.CenterCrop(32),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485], [0.229])
                                       ]))
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=50, drop_last=True,
                            pin_memory=True)

    return trainloader, valloader, testloader
