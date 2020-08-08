"""
train and test with ArcLoss
DenseNet121 as backbone
Author: LucasX
"""
import copy
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.optim import lr_scheduler
from torchvision import models

sys.path.append('../')
from research.cbir import data_loader
from research.cbir.cfg import cfg
from research.cbir.file_utils import mkdir_if_not_exist

EMBEDDING_SIZE = 1024


class ArcLoss(nn.Module):

    def __init__(self, embedding_size, class_num, s=30.0, m=0.50):
        """ArcFace formula:
            cos(m + theta) = cos(m)cos(theta) - sin(m)sin(theta)
        Note that:
            0 <= m + theta <= Pi
        So if (m + theta) >= Pi, then theta >= Pi - m. In [0, Pi]
        we have:
            cos(theta) < cos(Pi - m)
        So we can use cos(Pi - m) as threshold to check whether
        (m + theta) go out of [0, Pi]
        Args:
            embedding_size: usually 128, 256, 512 ...
            class_num: num of people when training
            s: scale, see normface https://arxiv.org/abs/1704.06369
            m: margin, see SphereFace, CosFace, and ArcFace paper
        """
        super().__init__()
        self.in_features = embedding_size
        self.out_features = class_num
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight.to("cuda:0")))
        sine = ((1.0 - cosine.pow(2)).clamp(0, 1)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # drop to CosFace
        # update y_i by phi in cosine
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]

        return output * self.s


def train_model(model, dataloaders, criterion, optimizer, metric, scheduler, num_epochs, inference=False):
    """
    :param model:
    :param dataloaders:
    :param criterion:
    :param optimizer:
    :param metric:
    :param scheduler:
    :param num_epochs:
    :param inference:
    :return:
    """
    print(model)
    model_name = model.__class__.__name__
    model = model.float()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    dataset_sizes = {x: dataloaders[x].__len__() * cfg['batch_size'] for x in ['train', 'val', 'test']}

    for k, v in dataset_sizes.items():
        print('Dataset size of {0} is {1}...'.format(k, v))

    if not inference:
        print('Start training %s...' % model_name)
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('-' * 100)
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                # for data in dataloaders[phase]:
                for i, data in enumerate(dataloaders[phase], 0):

                    inputs, labels = data['image'], data['type']
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # feats, outputs = model(inputs)
                        outputs = model(inputs)
                        thetas = metric(outputs, labels)
                        loss = criterion(thetas, labels)

                        _, preds = torch.max(thetas, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    tmp_correct = 0
                    tmp_total = 0
                    tmp_y_pred = []
                    tmp_y_true = []
                    tmp_filenames = []

                    for data in dataloaders['val']:
                        images, labels, filename = data['image'], data['type'], data['filename']
                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = model(images)
                        thetas = metric(outputs, labels)
                        _, predicted = torch.max(thetas.data, 1)
                        tmp_total += labels.size(0)
                        tmp_correct += (predicted == labels).sum().item()

                        tmp_y_pred += predicted.to("cpu").detach().numpy().tolist()
                        tmp_y_true += labels.to("cpu").detach().numpy().tolist()
                        tmp_filenames += filename

                    tmp_acc = tmp_correct / tmp_total

                    print('Confusion Matrix of {0} on val set: '.format(model_name))
                    cm = confusion_matrix(tmp_y_true, tmp_y_pred)
                    print(cm)
                    cm = np.array(cm)

                    print('Accuracy = {0}'.format(tmp_acc))
                    precisions = []
                    recalls = []

                    for i in range(len(cm)):
                        precisions.append(cm[i][i] / sum(cm[:, i].tolist()))
                        recalls.append(cm[i][i] / sum(cm[i, :].tolist()))

                    print("Precision of {0} on val set = {1}".format(model_name, sum(precisions) / len(precisions)))
                    print("Recall of {0} on val set = {1}".format(model_name, sum(recalls) / len(recalls)))

                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                    model.load_state_dict(best_model_wts)
                    model_path_dir = './model'
                    mkdir_if_not_exist(model_path_dir)
                    torch.save(model.state_dict(),
                               './model/{0}_best_epoch-{1}.pth'.format(model_name, epoch))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        model_path_dir = './model'
        mkdir_if_not_exist(model_path_dir)
        torch.save(model.state_dict(), './model/%s.pth' % model_name)

    else:
        print('Start testing %s...' % model.__class__.__name__)
        model.load_state_dict(torch.load(os.path.join('./model/%s.pth' % model_name)))

    model.eval()

    correct = 0
    total = 0
    y_pred = []
    y_true = []
    filenames = []
    probs = []

    with torch.no_grad():
        for data in dataloaders['test']:
            images, labels, filename = data['image'], data['type'], data['filename']
            images = images.to(device)
            labels = labels.to(device)

            # feats, outputs = model(images)
            outputs = model(images)
            thetas = metric(outputs, labels)
            _, predicted = torch.max(thetas.data, 1)

            outputs = F.softmax(outputs)
            # get TOP-K output labels and corresponding probabilities
            topK_prob, topK_label = torch.topk(outputs, 1)
            probs += topK_prob.to("cpu").detach().numpy().tolist()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_pred += predicted.to("cpu").detach().numpy().tolist()
            y_true += labels.to("cpu").detach().numpy().tolist()
            filenames += filename

    print('Accuracy of {0} on test set: {1}% '.format(model_name, 100 * correct / total))
    print(
        'Confusion Matrix of {0} on test set: '.format(model_name))

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    cm = np.array(cm)

    precisions = []
    recalls = []
    for i in range(len(cm)):
        precisions.append(cm[i][i] / sum(cm[:, i].tolist()))
        recalls.append(cm[i][i] / sum(cm[i, :].tolist()))

    print('Precision List: ')
    print(precisions)
    print('Recall List: ')
    print(recalls)

    print("Precision of {0} on val set = {1}".format(model_name,
                                                     sum(precisions) / len(precisions)))
    print(
        "Recall of {0} on val set = {1}".format(model_name, sum(recalls) / len(recalls)))

    print('Output CSV...')
    col = ['filename', 'gt', 'pred', 'prob']
    df = pd.DataFrame([[filenames[i], y_true[i], y_pred[i], probs[i][0]] for i in range(len(filenames))],
                      columns=col)
    df.to_csv("./%s.csv" % model_name, index=False)
    print('CSV has been generated...')


def main_with_arcloss(model, epoch, data_name):
    """
    train model
    :param model:
    :param epoch:
    :param data_name:
    :return:
    """
    criterion_xent_loss = nn.CrossEntropyLoss()
    arc_metric = ArcLoss(EMBEDDING_SIZE, cfg['out_num'])
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

    if data_name == 'TissuePhysiology':
        print('start loading TissuePhysiology dataset...')
        trainloader, valloader, testloader = data_loader.load_tissuephysiology_data()
    elif data_name == 'LightClothing':
        print('start loading LightClothing dataset...')
        trainloader, valloader, testloader = data_loader.load_lightclothing_data()
    elif data_name == 'AdditiveFood':
        print('start loading AdditiveFood dataset...')
        trainloader, valloader, testloader = data_loader.load_additivefood_data()
    else:
        print('Invalid data name. It can only be TissuePhysiology, LightClothing or AdditiveFood...')
        sys.exit(0)

    dataloaders = {
        'train': trainloader,
        'val': valloader,
        'test': testloader
    }

    train_model(model=model, dataloaders=dataloaders, criterion=criterion_xent_loss,
                optimizer=optimizer, metric=arc_metric,
                scheduler=exp_lr_scheduler, num_epochs=epoch, inference=False)


class DenseNet121(nn.Module):
    """
    DenseNet with features, constructed for ArcLoss
    """

    def __init__(self):
        super(DenseNet121, self).__init__()
        self.__class__.__name__ = 'DenseNet121'
        densenet121 = models.densenet121(pretrained=True)
        num_ftrs = densenet121.classifier.in_features
        self.features = densenet121.features
        self.embedding = nn.Sequential(nn.Linear(num_ftrs, EMBEDDING_SIZE), nn.BatchNorm1d(EMBEDDING_SIZE))

    def forward(self, x):
        x = self.features(x)
        feats = F.relu(x, inplace=True)
        feats = F.avg_pool2d(feats, kernel_size=7, stride=1).view(feats.size(0), -1)

        return self.embedding(feats)


if __name__ == '__main__':
    # densenet121 = models.densenet121(pretrained=True)
    # num_ftrs = densenet121.classifier.in_features
    # densenet121.classifier = nn.Linear(num_ftrs, cfg['out_num'])

    densenet121 = DenseNet121()

    main_with_arcloss(model=densenet121, epoch=cfg['epoch'], data_name='TissuePhysiology')
