import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append('../')
from research.garbageclassification.cfg import cfg
from research.garbageclassification.datasets import GarbageDataset


def load_garbage_classification_data():
    """
    load garbage classification data
    :return:
    """
    batch_size = cfg['batch_size']
    print('loading Garbage Classification dataset...')
    train_dataset = GarbageDataset(type='train',
                                   transform=transforms.Compose([
                                       transforms.Resize(224),
                                       transforms.RandomCrop(224),
                                       transforms.ColorJitter(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.RandomRotation(30),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ]))
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20, drop_last=True)

    val_dataset = GarbageDataset(type='val',
                                 transform=transforms.Compose([
                                     transforms.Resize(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ]))
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=True)

    test_dataset = GarbageDataset(type='test',
                                  transform=transforms.Compose([
                                      transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                  ]))
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=True)

    return trainloader, valloader, testloader
