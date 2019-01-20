import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append('../')
from research.imgcensor.cfg import cfg
from research.imgcensor.datasets import NSFWDataset


def load_nsfw_data():
    """
    load NSFW dataset
    :param dataset_name:
    :return:
    """
    batch_size = cfg['batch_size']
    print('loading NSFW dataset...')
    train_dataset = NSFWDataset(type='train',
                                transform=transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.ColorJitter(),
                                    transforms.RandomRotation(30),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]))
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = NSFWDataset(type='val',
                              transform=transforms.Compose([
                                  transforms.Resize(224),
                                  transforms.ColorJitter(),
                                  transforms.RandomRotation(30),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                              ]))
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    test_dataset = NSFWDataset(type='test',
                               transform=transforms.Compose([
                                   transforms.Resize(224),
                                   transforms.ColorJitter(),
                                   transforms.RandomRotation(30),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ]))
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, valloader, testloader
