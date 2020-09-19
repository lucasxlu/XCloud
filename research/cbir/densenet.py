import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torchvision import models


def myphi(x, m):
    x = x * m
    return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) + \
           x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)


class Flatten(nn.Module):
    """
    self constructed Flatten Module
    Note: use nn.Flatten() directly after PyTorch 1.5 or higher
    """

    __constants__ = ['start_dim', 'end_dim']
    start_dim: int
    end_dim: int

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        return input.flatten(self.start_dim, self.end_dim)


class DenseNet121(nn.Module):
    """
    DenseNet121 as backbone, constructed for ASoftmaxLoss
    """

    def __init__(self, num_cls, embedding_dim=1024):
        super(DenseNet121, self).__init__()
        self.__class__.__name__ = 'DenseNet121'
        densenet121 = models.densenet121(pretrained=True)
        self.features = densenet121.features
        num_ftrs = densenet121.classifier.in_features
        self.embedding = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(num_ftrs, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim)
        )
        self.classifier = AngleLinear(embedding_dim, num_cls)

    def forward(self, x):
        """
        feedforward an image, return pooling features (with BNNeck) and logits before softmax layer
        :param x:
        :return:
        """
        feats = self.embedding(self.features(x))
        logits = self.classifier(feats)

        return feats, logits

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F, ClassNum) F=in_features  ClassNum=out_features

        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        wlen = ww.pow(2).sum(0).pow(0.5)  # size= ClassNum

        cos_theta = x.mm(ww)  # size=(B, ClassNum)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = cos_theta.data.acos()
            k = (self.m * theta / 3.14159265).floor()
            n_one = k * 0.0 - 1
            phi_theta = (n_one ** k) * cos_m_theta - 2 * k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta, self.m)
            phi_theta = phi_theta.clamp(-1 * self.m, 1)

        cos_theta = cos_theta * xlen.view(-1, 1)
        phi_theta = phi_theta * xlen.view(-1, 1)
        output = (cos_theta, phi_theta)

        return output  # size=(B, ClassNum, 2)
