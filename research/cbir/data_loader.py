import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append('../')
from research.cbir.cfg import cfg
from research.cbir.datasets import TissuePhysiologyDataset, LightClothingDataset


def load_tissuephysiology_data():
    """
    load TissuePhysiology Crop dataset
    :return:
    """
    batch_size = cfg['batch_size']
    print('loading TissuePhysiologyDataset...')
    train_dataset = TissuePhysiologyDataset(type='train',
                                            transform=transforms.Compose([
                                                transforms.Resize(224),
                                                transforms.RandomCrop(224),
                                                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,
                                                                       hue=0.1),
                                                transforms.RandomRotation(60),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomVerticalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ]))
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=50, drop_last=True,
                             pin_memory=True)

    val_dataset = TissuePhysiologyDataset(type='val',
                                          transform=transforms.Compose([
                                              transforms.Resize(224),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ]))
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=50, drop_last=True,
                           pin_memory=True)

    test_dataset = TissuePhysiologyDataset(type='test',
                                           transform=transforms.Compose([
                                               transforms.Resize(224),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ]))
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=50, drop_last=True,
                            pin_memory=True)

    return trainloader, valloader, testloader


def load_lightclothing_data():
    """
    load LightClothing Crop dataset
    :return:
    """
    batch_size = cfg['batch_size']
    print('loading LightClothingDataset...')
    train_dataset = LightClothingDataset(type='train',
                                         transform=transforms.Compose([
                                             transforms.Resize(224),
                                             transforms.RandomCrop(224),
                                             transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,
                                                                    hue=0.1),
                                             transforms.RandomRotation(60),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomVerticalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ]))
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=50, drop_last=True,
                             pin_memory=True)

    val_dataset = LightClothingDataset(type='val',
                                       transform=transforms.Compose([
                                           transforms.Resize(224),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ]))
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=50, drop_last=True,
                           pin_memory=True)

    test_dataset = LightClothingDataset(type='test',
                                        transform=transforms.Compose([
                                            transforms.Resize(224),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ]))
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=50, drop_last=True,
                            pin_memory=True)

    return trainloader, valloader, testloader
