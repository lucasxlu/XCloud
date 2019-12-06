"""
some paper re-implementations proposed in NR-IQA fields
@Author: LucasX
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IQANet(nn.Module):
    """
    Convolutional Neural Networks for No-Reference Image Quality Assessment. CVPR'14
    Sampling a patch with 32*32*1 from a given image and train a simple CNN for quality score regression
    Input size: 32*32*1
    """

    def __init__(self, num_out):
        super(IQANet, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 7)
        self.bn1 = nn.BatchNorm2d(50)
        self.maxpool = nn.MaxPool2d(26)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(2 * 50, 800)  # concatenate MinPool and MaxPool
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, num_out)

    def forward(self, x):
        x = self.bn1(self.conv1(x))  # 26*26*50
        min_pool_x = -1 * self.maxpool(-x)  # MinPool
        min_pool_x = min_pool_x.view(-1, self.num_flat_features(min_pool_x))
        max_pool_x = self.maxpool(x)  # MaxPool
        max_pool_x = max_pool_x.view(-1, self.num_flat_features(max_pool_x))

        # If the size is a square you can only specify a single number
        x = torch.cat((min_pool_x, max_pool_x), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class IQACNNPlusPlus(nn.Module):
    """
    Simultaneous Estimation of Image Quality and Distortion via Multi-task Convolutional Neural Networks. ICIP'15
    Sampling a patch with 32*32*1 from a given image and train a simple CNN for quality score regression
    Input size: 32*32*3
    """

    def __init__(self, num_out):
        super(IQACNNPlusPlus, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)  # 30*30*8
        self.bn1 = nn.BatchNorm2d(8)
        self.maxpool1 = nn.MaxPool2d(2)  # 29*29*8

        self.conv2 = nn.Conv2d(8, 32, 3)  # 27*27*32
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(26)  # 26*26*32

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(64, 128)  # concatenate MinPool and MaxPool
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, num_out)

    def forward(self, x):
        x = self.bn1(self.conv1(x))  # 30*30*8
        x = self.bn2(self.conv2(x))  # 30*30*8
        min_pool_x = -1 * self.maxpool2(-x)  # MinPool
        min_pool_x = min_pool_x.view(-1, self.num_flat_features(min_pool_x))
        max_pool_x = self.maxpool2(x)  # MaxPool
        max_pool_x = max_pool_x.view(-1, self.num_flat_features(max_pool_x))

        # If the size is a square you can only specify a single number
        x = torch.cat((min_pool_x, max_pool_x), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
