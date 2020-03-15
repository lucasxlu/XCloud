# run unsupervised clustering algorithms to automatically find new categories
# Please carefully tune the hyper-param k, to make sure that max(x_i - \mu_i) <= \tau (such as 0.3)
# to avoid introducing noise
# author: @LucasX
import os
import math
import argparse

from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import shutil
from skimage import io

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from skimage.color import gray2rgb, rgba2rgb
from torchvision import models
from torchvision.transforms import transforms

parser = argparse.ArgumentParser()
parser.add_argument('-k', type=int)
parser.add_argument('-out_num', type=int, default=369)
parser.add_argument('-checkpoint', type=str,
                    default='/data/lucasxu/ModelZoo/DenseNet121_Embedding_AngularLoss.pth')
parser.add_argument('-max_l1_dist', type=float)
parser.add_argument('-use_gpu', type=bool, default=True)
parser.add_argument('-algorithm', type=str, default='kmeans')
parser.add_argument('-img_dir', type=str, help='unrecognizable images directory')
parser.add_argument('-save_dir', type=str, default='./ClusteringResult')
args = vars(parser.parse_args())

for key, value in args.items():
    print('%s = %s' % (key, value))


def myphi(x, m):
    x = x * m
    return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) + \
           x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)


class DenseNet121(nn.Module):
    """
    DenseNet with features, constructed for AngularLoss
    """

    def __init__(self, num_cls):
        super(DenseNet121, self).__init__()
        self.__class__.__name__ = 'DenseNet121'
        densenet121 = models.densenet121(pretrained=True)
        num_ftrs = densenet121.classifier.in_features
        densenet121.classifier = nn.Sequential(nn.Linear(num_ftrs, 1024), AngleLinear(1024, num_cls))
        self.model = densenet121

    def forward(self, x):
        for name, module in self.model.named_children():
            if name == 'features':
                feats = module(x)
                feats = F.relu(feats, inplace=True)
                feats = F.avg_pool2d(feats, kernel_size=7, stride=1).view(feats.size(0), -1)
            elif name == 'classifier':
                out = module(feats)

        return feats, out

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
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum

        cos_theta = x.mm(ww)  # size=(B,Classnum)
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

        return output  # size=(B,Classnum,2)


class AngularLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngularLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta, phi_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()

        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        output = cos_theta * 1.0  # size=(B,Classnum)
        output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)

        logpt = F.log_softmax(output, dim=0)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss.mean()

        return loss


def load_model_with_weights(model, model_path):
    print(model)
    model = model.float()
    model_name = model.__class__.__name__
    device = torch.device('cuda' if torch.cuda.is_available() and args['use_gpu'] else 'cpu')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    if model_path is not None and model_name != "":
        model.load_state_dict(torch.load(model_path))
    model.eval()

    if torch.cuda.device_count() > 1:
        model = model.module

    return model


def ext_deep_feat(model_with_weights, img_filepath):
    """
    extract deep feature from an image filepath
    :param model_with_weights:
    :param img_filepath:
    :return:
    """
    model_name = model_with_weights.__class__.__name__
    device = torch.device('cuda' if torch.cuda.is_available() and args['use_gpu'] else 'cpu')
    model_with_weights = model_with_weights.to(device)
    model_with_weights.eval()
    if model_name.startswith('DenseNet'):
        with torch.no_grad():
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            image = io.imread(img_filepath)
            if len(list(image.shape)) < 3:
                image = gray2rgb(image)
            elif len(list(image.shape)) > 3:
                image = rgba2rgb(image)

            img = preprocess(Image.fromarray(image.astype(np.uint8)))
            img.unsqueeze_(0)

            inputs = img.to(device)

            feat = model_with_weights.model.features(inputs)
            feat = F.relu(feat, inplace=True)
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)

            # print('feat size = {}'.format(feat.shape))

    feat = feat.to('cpu').detach().numpy().ravel()

    return feat / np.linalg.norm(feat)


def run_clustering(X, imgs):
    """
    run clustering algorithm, and assign a pseudo label to each sample
    Note that each feature vector in X relates to an image file in imgs
    :param X:
    :param imgs:
    :return:
    """
    assert len(X) == len(imgs)
    kmeans = KMeans(n_clusters=args['k'], random_state=0, verbose=True, max_iter=1000).fit(X)
    print('Categories: ', kmeans.labels_)
    for x, img in zip(X, imgs):
        pseudo_label = kmeans.predict(np.array(x).T.reshape(1, -1))[0]
        print('[INFO] assign {} with a pseudo label {}'.format(img, pseudo_label))
        if not os.path.exists(os.path.join(args['save_dir'], str(pseudo_label))):
            os.makedirs(os.path.join(args['save_dir'], str(pseudo_label)))
        shutil.copy(img, os.path.join(args['save_dir'], str(pseudo_label), os.path.basename(img)))


if __name__ == '__main__':
    densenet121 = DenseNet121(num_cls=args['out_num'])
    state_dict = torch.load(args['checkpoint'])
    try:
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        densenet121.load_state_dict(new_state_dict)
    except:
        densenet121.load_state_dict(state_dict)
    densenet121.eval()

    imgs = []
    feats = []

    print('Feature Extraction has started!')
    for img in os.listdir(args['img_dir']):
        feat = ext_deep_feat(densenet121, os.path.join(args['img_dir'], img))
        feats.append(feat.tolist())
        imgs.append(os.path.join(args['img_dir'], img))
        print('[INFO] extract feature for {} successfully, shape = {}'.format(img, np.array(feat).shape))

    print('Feature Extraction has finished!')
    print('Run Clustering algorithm...')
    run_clustering(feats, imgs)
    print('Finish Clustering!')
