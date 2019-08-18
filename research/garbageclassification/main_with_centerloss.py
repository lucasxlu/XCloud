"""
train and test with CenterLoss
Author: XuLu
"""
import os
import sys
import time
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.optim import lr_scheduler
from torchvision import models

sys.path.append('../../')
from research.garbageclassification.losses import CenterLoss
from research.garbageclassification import data_loader
from research.garbageclassification.utils import mkdirs_if_not_exist
from research.garbageclassification.cfg import cfg


def train_model(model, dataloaders, criterion_xent, criterion_cent, optimizer_model, optimizer_centloss, scheduler,
                num_epochs, inference=False):
    """
    train model
    :param optimizer_centloss:
    :param optimizer_model:
    :param criterion_cent:
    :param criterion_xent:
    :param model:
    :param dataloaders:
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

    for _ in dataset_sizes.keys():
        print('Dataset size of {0} is {1}...'.format(_, dataloaders[_].__len__() * cfg['batch_size']))

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
                    optimizer_model.zero_grad()
                    optimizer_centloss.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        feats, outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)

                        xent_loss = criterion_xent(outputs, labels)

                        loss = criterion_cent(feats, labels) * 0.001 + xent_loss

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer_model.step()

                            # multiple (1./alpha) in order to remove the effect of alpha on updating centers
                            for param in criterion_cent.parameters():
                                param.grad.data *= (1. / 1.)
                            optimizer_centloss.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / (dataset_sizes[phase] * cfg['batch_size'])
                epoch_acc = running_corrects.double() / (dataset_sizes[phase])

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

                        feats, outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
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
                    mkdirs_if_not_exist(model_path_dir)
                    torch.save(model.state_dict(),
                               './model/{0}_best_epoch-{1}.pth'.format(model_name, epoch))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        model_path_dir = './model'
        mkdirs_if_not_exist(model_path_dir)
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
            images, labels, filename = data['image'], data['label'], data['filename']
            images = images.to(device)
            labels = labels.to(device)

            feats, outputs = model(images)

            outputs = F.softmax(outputs)
            # get TOP-K output labels and corresponding probabilities
            topK_prob, topK_label = torch.topk(outputs, 2)
            probs += topK_prob.to("cpu").detach().numpy().tolist()

            _, predicted = torch.max(outputs.data, 1)
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


def main_with_centerloss(model, epoch):
    """
    train model
    :param model:
    :param epoch:
    :param data_name: ISIC/SD198
    :return:
    """

    criterion_xent = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=40, feat_dim=1024)
    optimizer_model = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    optimizer_centloss = optim.SGD(criterion_cent.parameters(), lr=0.5)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_model, step_size=80, gamma=0.1)

    trainloader, valloader, testloader = data_loader.load_garbage_classification_data()

    dataloaders = {
        'train': trainloader,
        'val': valloader,
        'test': testloader,
    }

    train_model(model=model, dataloaders=dataloaders, criterion_xent=criterion_xent, criterion_cent=criterion_cent,
                optimizer_model=optimizer_model, optimizer_centloss=optimizer_centloss, scheduler=exp_lr_scheduler,
                num_epochs=epoch, inference=False)


class DenseNet121(nn.Module):
    """
    DenseNet with features, constructed for CenterLoss
    """

    def __init__(self, num_cls=40):
        super(DenseNet121, self).__init__()
        self.__class__.__name__ = 'DenseNet121'
        densenet121 = models.densenet121(pretrained=True)
        num_ftrs = densenet121.classifier.in_features
        densenet121.classifier = nn.Linear(num_ftrs, num_cls)
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


class DenseNet169(nn.Module):
    """
    DenseNet with features, constructed for CenterLoss
    """

    def __init__(self, num_cls=40):
        super(DenseNet169, self).__init__()
        self.__class__.__name__ = 'DenseNet169'
        densenet169 = models.densenet169(pretrained=True)
        num_ftrs = densenet169.classifier.in_features
        densenet169.classifier = nn.Linear(num_ftrs, num_cls)
        self.model = densenet169

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


if __name__ == '__main__':
    # densenet121 = models.densenet121(pretrained=True)
    # num_ftrs = densenet121.classifier.in_features
    # densenet121.classifier = nn.Linear(num_ftrs, 40)

    # densenet121 = DenseNet(num_classes=40)

    densenet = DenseNet121(num_cls=40)
    main_with_centerloss(model=densenet, epoch=cfg['epoch'])
