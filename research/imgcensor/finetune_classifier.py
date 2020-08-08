"""
fine-tune Deep Models
Author: LucasX
"""
import argparse
import copy
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
from research.imgcensor import data_loader
from research.imgcensor.utils import mkdirs_if_not_exist

parse = argparse.ArgumentParser(description="config for fine-tune deep models")
parse.add_argument('-bs', '--batch_size', help='batch size', default=64)
parse.add_argument('-lr', '--learning_rate', help='learning rate', default=1e-3)
parse.add_argument('-step', '--step', help='learning rate decay step', default=40)
parse.add_argument('-wd', '--weight_decay', help='weight decay', default=1e-4)
parse.add_argument('-out_num_before', '--out_num_before', help='output num before', type=int, default=5)
parse.add_argument('-out_num_after', '--out_num_after', help='output num after', type=int, default=6)
parse.add_argument('-checkpoint', '--checkpoint', help='checkpoint')
parse.add_argument('-epoch', '--epoch', help='fine-tune epoch', default=100)
parse.add_argument('-infer', '--inference', help='whether to perform inference', default=False)
args = vars(parse.parse_args())


def finetune_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, inference=False):
    """
    fine-tune model
    :param model:to
    :param dataloaders:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :param inference:
    :return:
    """
    model_name = model.__class__.__name__
    model = model.float()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('[INFO] start loading deep model weights...')
    state_dict = torch.load(args['checkpoint'])
    try:
        model.load_state_dict(state_dict)
    except:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    print('[INFO] finish loading deep model weights...')

    print('-' * 100)
    if model_name.startswith('DenseNet'):
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, args['out_num_after'])
    elif model_name.startswith('ResNet'):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args['out_num_after'])

    print(model)

    print('-' * 100)
    for k, v in args.items():
        print(k, v)
    print('-' * 100)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}

    for _ in dataset_sizes.keys():
        print('Dataset size of {0} is {1}...'.format(_, dataset_sizes[_]))

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
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for i, data in enumerate(dataloaders[phase], 0):
                    inputs, types = data['image'], data['type']
                    inputs = inputs.to(device)
                    types = types.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, types)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == types.data)

                if phase == 'train':
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
                        images, types, filename = data['image'], data['type'], data['filename']
                        images = images.to(device)
                        types = types.to(device)

                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        tmp_total += types.size(0)
                        tmp_correct += (predicted == types).sum().item()

                        tmp_y_pred += predicted.to("cpu").detach().numpy().tolist()
                        tmp_y_true += types.to("cpu").detach().numpy().tolist()
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

                    print("Precision of {0} on val set = {1}".format(model_name,
                                                                     sum(precisions) / len(precisions)))
                    print(
                        "Recall of {0} on val set = {1}".format(model_name, sum(recalls) / len(recalls)))

                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                    model.load_state_dict(best_model_wts)
                    model_path_dir = './model'
                    mkdirs_if_not_exist(model_path_dir)
                    torch.save(model.module.state_dict(),
                               './model/{0}_best_epoch-{1}.pth'.format(model_name, epoch))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        model_path_dir = './model'
        mkdirs_if_not_exist(model_path_dir)
        torch.save(model.module.state_dict(), './model/%s.pth' % model_name)

    else:
        print('Start testing %s...' % model_name)
        model.load_state_dict(torch.load(os.path.join('./model/%s.pth' % model_name)))

    model.eval()

    correct = 0
    total = 0
    y_pred = []
    y_true = []
    filenames = []
    probs = []

    with torch.no_grad():
        for i, data in enumerate(dataloaders['test'], 0):
            images, types, filename = data['image'], data['type'], data['filename']
            images = images.to(device)
            types = types.to(device)

            outputs = model(images)

            outputs = F.softmax(outputs)
            # get TOP-K output labels and corresponding probabilities
            topK_prob, topK_label = torch.topk(outputs, 2)
            probs += topK_prob.to("cpu").detach().numpy().tolist()

            _, predicted = torch.max(outputs.data, 1)
            total += types.size(0)
            correct += (predicted == types).sum().item()

            y_pred += predicted.to("cpu").detach().numpy().tolist()
            y_true += types.to("cpu").detach().numpy().tolist()
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


def finetune_nsfw_classification(model, epoch):
    """
    finetune NSFW classification
    :param model:
    :param epoch:
    :return:
    """
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=0.9,
                             weight_decay=args['weight_decay'])

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args['step'], gamma=0.1)

    print('start loading NSFW Dataset...')
    trainloader, valloader, testloader = data_loader.load_nsfw_data()

    dataloaders = {
        'train': trainloader,
        'val': valloader,
        'test': testloader,
    }

    finetune_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer_ft,
                   scheduler=exp_lr_scheduler, num_epochs=epoch, inference=args['inference'])


if __name__ == '__main__':
    densenet121 = models.densenet121(pretrained=False)
    num_ftrs = densenet121.classifier.in_features
    densenet121.classifier = nn.Linear(num_ftrs, args['out_num_before'])

    finetune_nsfw_classification(model=densenet121, epoch=args['epoch'])
