import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append('../../')
from research.fbp.cfg import cfg
from research.fbp.datasets import SCUTFBP5500Dataset


def load_scutfbp5500_data():
    """
    load SCUTFBP5500 Dataset
    :return:
    """
    batch_size = cfg['batch_size']
    train_dataset = SCUTFBP5500Dataset(train=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(227),
                                           transforms.RandomResizedCrop(224),
                                           transforms.ColorJitter(),
                                           transforms.RandomRotation(30),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           # transforms.Normalize(
                                           #     mean=[131.45376586914062, 103.98748016357422,
                                           #           91.46234893798828], std=[1, 1, 1])
                                       ]))
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = SCUTFBP5500Dataset(train=False,
                                      transform=transforms.Compose([
                                          transforms.Resize(227),
                                          transforms.RandomResizedCrop(224),
                                          transforms.ColorJitter(),
                                          transforms.RandomRotation(30),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          # transforms.Normalize(
                                          #     mean=[131.45376586914062, 103.98748016357422, 91.46234893798828],
                                          #     std=[1, 1, 1])
                                      ]))
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader
