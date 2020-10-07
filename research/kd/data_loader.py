import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append('../')
from research.kd.cfg import cfg
from research.kd.datasets import ImageDataset


def load_mengzhucrop_data():
    """
    load MengZhu Crop dataset
    :return:
    """
    batch_size = cfg['batch_size']
    print('loading Image dataset...')
    train_dataset = ImageDataset(type='train',
                                 transform=transforms.Compose([
                                     transforms.Resize((224, 224)),
                                     transforms.RandomRotation(60),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                     transforms.RandomErasing(p=0.5, scale=(0.1, 0.3), value='random')
                                 ]))
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=50, drop_last=True,
                             pin_memory=True)

    val_dataset = ImageDataset(type='val',
                               transform=transforms.Compose([
                                   transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ]))
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=50, drop_last=False,
                           pin_memory=True)

    test_dataset = ImageDataset(type='test',
                                transform=transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]))
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=50, drop_last=False,
                            pin_memory=True)

    return trainloader, valloader, testloader
