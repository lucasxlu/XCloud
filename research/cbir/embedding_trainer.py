"""
train deep feature embedding with SoftmaxLoss/CenterLoss/ASoftmaxLoss/ArcLoss
DenseNet121 as backbone
@Author: LucasX
"""
import ast
import copy
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.optim import lr_scheduler

from research.cbir.densenet import DenseNet121
from research.cbir.losses import AngularLoss, ArcLoss, CenterLoss

sys.path.append('../')
from research.cbir import data_loader
from research.cbir.cfg import cfg
from research.cbir.file_utils import mkdir_if_not_exist

parser = argparse.ArgumentParser()
parser.add_argument('-loss', type=str, choices=['SoftmaxLoss', 'CenterLoss', 'ASoftmaxLoss', 'ArcLoss'])
parser.add_argument('-arch', type=str, choices=['DenseNet121'])
parser.add_argument('-infer', type=ast.literal_eval, dest='flag')
parser.add_argument('-dim', type=int, default=1024, help='embedding dimension size')

args = vars(parser.parse_args())


def train_model_with_modified_softmax_loss(model, dataloaders, criterion, optimizer, scheduler):
    """
    train model with modified SoftmaxLoss, such as vanilla Softmax and ASoftmax Loss
    :param optimizer:
    :param criterion:
    :param model:
    :param dataloaders:
    :param scheduler:
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

    if not args['infer']:
        print('Start training %s...' % model_name)
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(cfg['epoch']):
            print('-' * 100)
            print('Epoch {}/{}'.format(epoch, cfg['epoch'] - 1))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    if torch.__version__ <= '1.1.0':
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
                        feats, outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        outputs = outputs[0]  # 0=cos_theta 1=phi_theta
                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    if torch.__version__ >= '1.1.0':
                        scheduler.step()

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

                        feats, outputs = model(images)
                        outputs = outputs[0]  # 0=cos_theta 1=phi_theta
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
                    mkdir_if_not_exist(model_path_dir)
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(),
                                   './model/{0}_best_epoch-{1}.pth'.format(model_name, epoch))
                    else:
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
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), './model/%s.pth' % model_name)
        else:
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

            feats, outputs = model(images)
            outputs = outputs[0]  # 0=cos_theta 1=phi_theta

            _, predicted = torch.max(outputs.data, 1)

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


def train_model_for_centerloss(model, dataloaders, criterion_xent, criterion_cent, optimizer_model, optimizer_centloss,
                               scheduler):
    """
    train model with CenterLoss
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

    for k, v in dataset_sizes.items():
        print('Dataset size of {0} is {1}...'.format(k, v))

    if not args['infer']:
        print('Start training %s...' % model_name)
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(cfg['epoch']):
            print('-' * 100)
            print('Epoch {}/{}'.format(epoch, cfg['epoch'] - 1))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    if torch.__version__ <= '1.1.0':
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

                if phase == 'train':
                    if torch.__version__ >= '1.1.0':
                        scheduler.step()

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
                    mkdir_if_not_exist(model_path_dir)
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(),
                                   './model/{0}_best_epoch-{1}.pth'.format(model_name, epoch))
                    else:
                        torch.save(model.state_dict(), './model/{0}_best_epoch-{1}.pth'.format(model_name, epoch))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        model_path_dir = './model'
        mkdir_if_not_exist(model_path_dir)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), './model/%s.pth' % model_name)
        else:
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


def main_with_centerloss(model):
    """
    train model
    :param model:
    :param epoch:
    :param data_name:
    :return:
    """

    criterion_xent = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=cfg['out_num'], feat_dim=1024)
    optimizer_model = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    optimizer_centloss = optim.SGD(criterion_cent.parameters(), lr=0.5)

    cosine_anneal_warmup_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_model, T_0=10, T_mult=10,
                                                                                 eta_min=0, last_epoch=-1)

    print('start loading ImageDataset...')
    trainloader, valloader, testloader = data_loader.load_imagedataset_data()

    dataloaders = {
        'train': trainloader,
        'val': valloader,
        'test': testloader
    }

    train_model_for_centerloss(model=model, dataloaders=dataloaders, criterion_xent=criterion_xent,
                               criterion_cent=criterion_cent,
                               optimizer_model=optimizer_model, optimizer_centloss=optimizer_centloss,
                               scheduler=cosine_anneal_warmup_lr_scheduler)


def train_model_for_arcloss(model, dataloaders, criterion, optimizer, metric, scheduler):
    """
    :param model:
    :param dataloaders:
    :param criterion:
    :param optimizer:
    :param metric:
    :param scheduler:
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

    if not args['infer']:
        print('Start training %s...' % model_name)
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(arch['epoch']):
            print('-' * 100)
            print('Epoch {}/{}'.format(epoch, arch['epoch'] - 1))

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
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(),
                                   './model/{0}_best_epoch-{1}.pth'.format(model_name, epoch))
                    else:
                        torch.save(model.state_dict(), './model/{0}_best_epoch-{1}.pth'.format(model_name, epoch))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        model_path_dir = './model'
        mkdir_if_not_exist(model_path_dir)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), './model/%s.pth' % model_name)
        else:
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


def main_with_softmaxloss(model):
    """
    train model with vanilla SoftmaxLoss as supervision
    :param model:
    :return:
    """
    criterion_softmaxloss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg['init_lr'], momentum=0.9, weight_decay=cfg['weight_decay'])
    cosine_anneal_warmup_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=10,
                                                                                 eta_min=0, last_epoch=-1)

    print('start loading ImageDataset...')
    trainloader, valloader, testloader = data_loader.load_imagedataset_data()

    dataloaders = {
        'train': trainloader,
        'val': valloader,
        'test': testloader
    }

    train_model_with_modified_softmax_loss(model=model, dataloaders=dataloaders, criterion=criterion_softmaxloss,
                                           optimizer=optimizer,
                                           scheduler=cosine_anneal_warmup_lr_scheduler)


def main_with_asoftmaxloss(model):
    """
    train model with vanilla ASoftmaxLoss as supervision
    :param model:
    :return:
    """
    criterion_aloss = AngularLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg['init_lr'], momentum=0.9, weight_decay=cfg['weight_decay'])
    cosine_anneal_warmup_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=10,
                                                                                 eta_min=0, last_epoch=-1)

    print('start loading ImageDataset...')
    trainloader, valloader, testloader = data_loader.load_imagedataset_data()

    dataloaders = {
        'train': trainloader,
        'val': valloader,
        'test': testloader
    }

    train_model_with_modified_softmax_loss(model=model, dataloaders=dataloaders, criterion=criterion_aloss,
                                           optimizer=optimizer,
                                           scheduler=cosine_anneal_warmup_lr_scheduler)


def main_with_arcloss(model):
    """
    train model
    :param model:
    :param epoch:
    :return:
    """
    criterion_xent_loss = nn.CrossEntropyLoss()
    arc_metric = ArcLoss(args['dim'], cfg['out_num'])
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

    print('start loading ImageDataset...')
    trainloader, valloader, testloader = data_loader.load_imagedataset_data()

    dataloaders = {
        'train': trainloader,
        'val': valloader,
        'test': testloader
    }

    train_model_for_arcloss(model=model, dataloaders=dataloaders, criterion=criterion_xent_loss,
                            optimizer=optimizer, metric=arc_metric,
                            scheduler=exp_lr_scheduler)


if __name__ == '__main__':
    if args['arch'] == 'DenseNet121':
        arch = DenseNet121(num_cls=cfg['out_num'])

    if args['loss'] == 'SoftmaxLoss':
        main_with_softmaxloss(model=arch)
    elif args['loss'] == 'ASoftmaxLoss':
        main_with_asoftmaxloss(model=arch)
    elif args['loss'] == 'ArcLoss':
        main_with_arcloss(model=arch)
    elif args['loss'] == 'CenterLoss':
        main_with_centerloss(model=arch)
