import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CRNet(nn.Module):
    """
    definition of CRNet
    """

    def __init__(self):
        super(CRNet, self).__init__()
        self.meta = {'mean': [131.45376586914062, 103.98748016357422, 91.46234893798828],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}

        model_ft = models.resnet18(pretrained=True)

        self.model = model_ft
        self.regressor = Regressor(model_ft)
        self.classifier = Classifier(model_ft, num_cls=5)

    def forward(self, x):
        for name, module in self.model.named_children():
            if name != 'fc':
                x = module(x)

        reg_out = self.regressor.forward(x.view(-1, self.num_flat_features(x)))
        cls_out = self.classifier.forward(x.view(-1, self.num_flat_features(x)))

        return reg_out, cls_out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class Regressor(nn.Module):

    def __init__(self, model):
        super(Regressor, self).__init__()

        num_ftrs = model.fc.in_features
        self.fc1 = nn.Linear(num_ftrs, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x2 = F.relu(self.fc2(x1))
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x3 = self.fc3(x2)

        return x3

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class Classifier(nn.Module):

    def __init__(self, model, num_cls=5):
        super(Classifier, self).__init__()

        num_ftrs = model.fc.in_features
        self.fc1 = nn.Linear(num_ftrs, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_cls)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x2 = F.relu(self.fc2(x1))
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x3 = self.fc3(x2)

        return x3

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class SCUTNet(nn.Module):
    """
    Paper: SCUT-FBP: A Benchmark Dataset for Facial Beauty Perception
    Link: https://arxiv.org/ftp/arxiv/papers/1511/1511.02459.pdf
    Note: input size is 227*227
    Performance:

    """

    def __init__(self):
        super(SCUTNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(50, 100, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(100, 150, 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(150, 200, 4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(200, 250, 4)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(250, 300, 2)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(300 * 1 * 1, 500)
        self.fc2 = nn.Linear(500, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.pool6(F.relu(self.conv6(x)))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PICNN(nn.Module):
    """
    Paper: Facial attractiveness prediction using psychologically inspired convolutional neural network (PI-CNN)
    Link: https://ieeexplore.ieee.org/abstract/document/7952438/
    Note: input size is 227*227
    """

    def __init__(self):
        super(PICNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(50, 100, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(100, 150, 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv4 = nn.Conv2d(150, 200, 4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv5 = nn.Conv2d(200, 250, 3)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(250, 300, 2)
        self.pool6 = nn.AvgPool2d(kernel_size=2, stride=2)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(300 * 2 * 2, 500)
        self.fc2 = nn.Linear(500, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.pool6(F.relu(self.conv6(x)))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
