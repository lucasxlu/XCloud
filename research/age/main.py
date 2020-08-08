"""
train and eval ResNet for Age Estimation
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
from research.age import data_loader
from research.age.utils import mkdirs_if_not_exist
from research.age.cfg import cfg


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, inference=False):
    """
    train model
    :param model:
    :param dataloaders:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :param inference:
    :return:
    """
    model = model.float()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}

    for _ in dataset_sizes.keys():
        print('Dataset size of {0} is {1}...'.format(_, dataset_sizes[_]))

    if not inference:
        print('Start training %s...' % model.__class__.__name__)
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
                # for inputs, labels, filenames in dataloaders[phase]:
                for i, data in enumerate(dataloaders[phase], 0):

                    inputs, ages = data['image'], data['age']
                    inputs = inputs.to(device)
                    ages = ages.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, ages)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == ages.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    tmp_correct = 0
                    tmp_total = 0
                    tmp_y_pred = []
                    tmp_y_true = []
                    tmp_filenames = []

                    for data in dataloaders['val']:
                        images, ages, filename = data['image'], data['age'], data['filename']
                        images = images.to(device)
                        ages = ages.to(device)

                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        tmp_total += ages.size(0)
                        tmp_correct += (predicted == ages).sum().item()

                        tmp_y_pred += predicted.to("cpu").detach().numpy().tolist()
                        tmp_y_true += ages.to("cpu").detach().numpy().tolist()
                        tmp_filenames += filename

                    tmp_acc = tmp_correct / tmp_total

                    print('Confusion Matrix of {0} on val set: '.format(model.__class__.__name__))
                    cm = confusion_matrix(tmp_y_true, tmp_y_pred)
                    print(cm)
                    cm = np.array(cm)

                    print('Accuracy = {0}'.format(tmp_acc))
                    precisions = []
                    recalls = []

                    for i in range(len(cm)):
                        precisions.append(cm[i][i] / sum(cm[:, i].tolist()))
                        recalls.append(cm[i][i] / sum(cm[i, :].tolist()))

                    print("Precision of {0} on val set = {1}".format(model.__class__.__name__,
                                                                     sum(precisions) / len(precisions)))
                    print(
                        "Recall of {0} on val set = {1}".format(model.__class__.__name__, sum(recalls) / len(recalls)))

                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                    model.load_state_dict(best_model_wts)
                    model_path_dir = './model'
                    mkdirs_if_not_exist(model_path_dir)
                    torch.save(model.module.state_dict(),
                               './model/{0}_best_epoch-{1}.pth'.format(model.__class__.__name__, epoch))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        model_path_dir = './model'
        mkdirs_if_not_exist(model_path_dir)
        torch.save(model.module.state_dict(), './model/%s.pth' % model.__class__.__name__)

    else:
        print('Start testing %s...' % model.__class__.__name__)
        model.load_state_dict(torch.load(os.path.join('./model/%s.pth' % model.__class__.__name__)))

    model.eval()

    correct = 0
    total = 0
    y_pred = []
    y_true = []
    filenames = []
    probs = []

    with torch.no_grad():
        for data in dataloaders['test']:
            images, ages, filename = data['image'], data['age'], data['filename']
            images = images.to(device)
            ages = ages.to(device)

            outputs = model(images)

            outputs = F.softmax(outputs)
            # get TOP-K output labels and corresponding probabilities
            topK_prob, topK_label = torch.topk(outputs, 2)
            probs += topK_prob.to("cpu").detach().numpy().tolist()

            _, predicted = torch.max(outputs.data, 1)
            total += ages.size(0)
            correct += (predicted == ages).sum().item()

            y_pred += predicted.to("cpu").detach().numpy().tolist()
            y_true += ages.to("cpu").detach().numpy().tolist()
            filenames += filename

    print('Accuracy of {0} on test set: {1}% '.format(model.__class__.__name__, 100 * correct / total))
    print(
        'Confusion Matrix of {0} on test set: '.format(model.__class__.__name__))

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

    print("Precision of {0} on test set = {1}".format(model.__class__.__name__,
                                                     sum(precisions) / len(precisions)))
    print(
        "Recall of {0} on test set = {1}".format(model.__class__.__name__, sum(recalls) / len(recalls)))

    print('Output CSV...')
    col = ['filename', 'gt', 'pred', 'prob']
    df = pd.DataFrame([[filenames[i], y_true[i], y_pred[i], probs[i][0]] for i in range(len(filenames))],
                      columns=col)
    df.to_csv("./output-%s.csv" % model.__class__.__name__, index=False)
    print('CSV has been generated...')


def run_age_estimator(model, epoch):
    """
    run age estimator
    :param model:
    :param epoch:
    :return:
    """
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

    print('start loading UTKFaceDataset...')
    trainloader, valloader, testloader = data_loader.load_data("UTKFace")

    dataloaders = {
        'train': trainloader,
        'val': valloader,
        'test': testloader,
    }

    train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer_ft,
                scheduler=exp_lr_scheduler, num_epochs=epoch, inference=False)


def batch_inference(model, inferencedataloader):
    """
    Inference in Batch Mode
    :param inferencedataloader:
    :param model:
    :return:
    """
    model = model.float()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    print('Start testing %s...' % model.__class__.__name__)
    model.load_state_dict(torch.load(os.path.join('./model/%s.pth' % model.__class__.__name__)))
    model.eval()

    y_pred = []
    filenames = []
    probs = []

    with torch.no_grad():
        for data in inferencedataloader:
            images, filename = data['image'], data['filename']
            images = images.to(device)

            outputs = model(images)

            outputs = F.softmax(outputs)
            # get TOP-K output labels and corresponding probabilities
            topK_prob, topK_label = torch.topk(outputs, 2)
            probs += topK_prob.to("cpu").detach().numpy().tolist()

            _, predicted = torch.max(outputs.data, 1)

            y_pred += predicted.to("cpu").detach().numpy().tolist()
            filenames += filename

    print('Output CSV...')
    col = ['filename', 'pred', 'prob']
    df = pd.DataFrame([[filenames[i], y_pred[i], probs[i][0]] for i in range(len(filenames))],
                      columns=col)
    df.to_csv("./submission.csv", index=False)
    print('CSV has been generated...')


if __name__ == '__main__':
    resnet = models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, cfg['num_of_class_in_utkface'])

    run_age_estimator(model=resnet, epoch=200)

    # batch_inference(resnet, data_loader.load_data('UTKFace'))
