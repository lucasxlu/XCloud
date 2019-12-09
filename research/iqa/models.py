"""
some paper re-implementations proposed in NR-IQA fields
@Author: LucasX
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class IQANet(nn.Module):
    """
    Convolutional Neural Networks for No-Reference Image Quality Assessment. CVPR'14
    Sampling a patch with 32*32*3 from a given image and train a simple CNN for quality score regression
    Input size: 32*32*3
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
    Sampling a patch with 32*32*3 from a given image and train a simple CNN for quality score regression/classification
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

    
class DeepPatchCNN(nn.Module):
    """
    A deep neural network for image quality assessment. ICIP'16
    Sampling a patch with 32*32*3 from a given image and train a simple CNN for quality score regression
    Input size: 32*32*3
    """

    def __init__(self, num_out):
        super(DeepPatchCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 32*32*32
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)  # 32*32*32
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)  # 16*16*32

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # 16*16*64
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)  # 16*16*64
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(2, 2)  # 8*8*64

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)  # 8*8*128
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)  # 8*8*128
        self.bn6 = nn.BatchNorm2d(128)
        self.relu6 = nn.ReLU()
        self.maxpool6 = nn.MaxPool2d(2, 2)  # 4*4*128

        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)  # 4*4*256
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)  # 4*4*256
        self.bn8 = nn.BatchNorm2d(256)
        self.relu8 = nn.ReLU()
        self.maxpool8 = nn.MaxPool2d(2, 2)  # 2*2*256

        self.conv9 = nn.Conv2d(256, 512, 3, padding=1)  # 2*2*512
        self.bn9 = nn.BatchNorm2d(512)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)  # 2*2*512
        self.bn10 = nn.BatchNorm2d(512)
        self.relu10 = nn.ReLU()
        self.maxpool10 = nn.MaxPool2d(2, 2)  # 1*1*512

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_out)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))

        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool4(self.relu4(self.bn4(self.conv4(x))))

        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.maxpool6(self.relu6(self.bn6(self.conv6(x))))

        x = self.relu7(self.bn7(self.conv7(x)))
        x = self.maxpool8(self.relu8(self.bn8(self.conv8(x))))

        x = self.relu9(self.bn9(self.conv9(x)))
        x = self.maxpool10(self.relu10(self.bn10(self.conv10(x))))

        x = x.view(-1, self.num_flat_features(x))

        # If the size is a square you can only specify a single number
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class NIMA(nn.Module):
    """
    PyTorch implementation of <NIMA: Neural Image Assessment> published at TIP'18
    Input size: 224*224*3
    @Author: LucasX
    """

    def __init__(self, backbone, num_out):
        super(NIMA, self).__init__()
        if backbone.lower() == 'resnet':
            resnet50 = models.resnet50(pretrained=True)
            num_ftrs = resnet50.fc.in_features
            resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, num_out), nn.Softmax(dim=1))
            print('[INFO] initialize NIMA with ResNet50 as backbone')
            self.backbone = resnet50
        elif backbone.lower() == 'densenet':
            densenet169 = models.densenet169(pretrained=True)
            num_ftrs = densenet169.classifier.in_features
            densenet169.classifier = nn.Sequential(nn.Linear(num_ftrs, num_out), nn.Softmax(dim=1))
            print('[INFO] initialize NIMA with DenseNet169 as backbone')
            self.backbone = densenet169
        else:
            raise ValueError('Invalid backbone, it can only be [resnet/densenet]')

    def forward(self, x):
        return self.backbone(x)


class EDMLoss(nn.Module):
    """
    EDMLoss used in conjunction with NIMA model
    """

    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, p_target: torch.Tensor, p_estimate: torch.Tensor):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))

        return samplewise_emd.mean()
