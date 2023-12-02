"""
@Note: Implementation of Knowledge Distillation Algorithms
@Author: LucasXU
"""
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
from research.kd import data_loader
from research.kd.cfg import cfg
from research.kd.losses import KDLoss, RegularizedTfKDLoss, SelfTfKDLoss


def train_model_with_kd(use_lsr, teacher_model_w_weights, student_model_wo_weights, dataloaders, criterion,
                        optimizer, scheduler,
                        num_epochs, inference=False):
    """
    train model with Knowledge Distillation
    :param use_lsr: whether to use LabelSmoothingRegularization
    :param teacher_model_w_weights:
    :param student_model_wo_weights:
    :param dataloaders:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :param inference:
    :return:
    """
    print(student_model_wo_weights)
    model_name = student_model_wo_weights.__class__.__name__
    student_model_wo_weights = student_model_wo_weights.float()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    student_model_wo_weights = student_model_wo_weights.to(device)
    teacher_model_w_weights = teacher_model_w_weights.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        student_model_wo_weights = nn.DataParallel(student_model_wo_weights)

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}

    for _ in dataset_sizes.keys():
        print('Dataset size of {0} is {1}...'.format(_, dataset_sizes[_]))

    if not inference:
        print('Start training %s...' % model_name)
        since = time.time()

        best_model_wts = copy.deepcopy(student_model_wo_weights.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('-' * 100)
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    if torch.__version__ <= '1.1.0':
                        scheduler.step()
                    student_model_wo_weights.train()  # Set model to training mode
                else:
                    student_model_wo_weights.eval()  # Set model to evaluate mode

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
                        student_outputs = student_model_wo_weights(inputs)
                        _, preds = torch.max(student_outputs, 1)
                        if not use_lsr:
                            teacher_outputs = teacher_model_w_weights(inputs)
                        if use_lsr:
                            loss = criterion(student_outputs, types)
                        else:
                            loss = criterion(teacher_outputs, student_outputs, types)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == types.data)

                if phase == 'train':
                    if torch.__version__ >= '1.1.0':
                        scheduler.step()

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
                        images, types, filename = data['image'], data['type'], data['filename']
                        images = images.to(device)
                        types = types.to(device)

                        outputs = student_model_wo_weights(images)
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
                    best_model_wts = copy.deepcopy(student_model_wo_weights.state_dict())

                    student_model_wo_weights.load_state_dict(best_model_wts)
                    model_path_dir = './model'
                    os.makedirs(model_path_dir, exist_ok=True)
                    if torch.cuda.device_count() > 1:
                        torch.save(student_model_wo_weights.module.state_dict(),
                                   './model/{0}_best_epoch-{1}.pth'.format(model_name, epoch))
                    else:
                        torch.save(student_model_wo_weights.state_dict(),
                                   './model/{0}_best_epoch-{1}.pth'.format(model_name, epoch))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        student_model_wo_weights.load_state_dict(best_model_wts)
        model_path_dir = './model'
        os.makedirs(model_path_dir, exist_ok=True)
        if torch.cuda.device_count() > 1:
            torch.save(student_model_wo_weights.module.state_dict(), './model/%s.pth' % model_name)
        else:
            torch.save(student_model_wo_weights.state_dict(), './model/%s.pth' % model_name)

    else:
        print('Start testing %s...' % model_name)
        student_model_wo_weights.load_state_dict(torch.load(os.path.join('./model/%s.pth' % model_name)))

    student_model_wo_weights.eval()

    correct = 0
    total = 0
    y_pred = []
    y_true = []
    filenames = []
    probs = []

    with torch.no_grad():
        for data in dataloaders['test']:
            images, types, filename = data['image'], data['type'], data['filename']
            images = images.to(device)
            types = types.to(device)

            outputs = student_model_wo_weights(images)

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


def run_img_classification(teacher_w_weights, student_wo_weights, epoch):
    """
    run image classification
    :param teacher_w_weights:
    :param student_wo_weights:
    :param epoch:
    :return:
    """
    if cfg['use_lsr']:
        criterion = RegularizedTfKDLoss(alpha=0.5, temperature=10)
    else:
        criterion = KDLoss(alpha=0.5, temperature=10)  # vanilla KD Loss
    teacher_w_weights.eval()

    optimizer_ft = optim.SGD(student_wo_weights.parameters(), lr=cfg['init_lr'], momentum=0.9,
                             weight_decay=cfg['weight_decay'])

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=cfg['lr_decay_step'], gamma=0.1)
    # cosine_anneal_warmup_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ft, T_0=10, T_mult=10,
    #                                                                              eta_min=1e-5, last_epoch=-1)

    trainloader, valloader, testloader = data_loader.load_mengzhucrop_data()

    dataloaders = {
        'train': trainloader,
        'val': valloader,
        'test': testloader,
    }

    train_model_with_kd(use_lsr=cfg['use_lsr'], teacher_model_w_weights=teacher_w_weights,
                        student_model_wo_weights=student_wo_weights,
                        dataloaders=dataloaders, criterion=criterion, optimizer=optimizer_ft,
                        scheduler=exp_lr_scheduler, num_epochs=epoch, inference=False)


if __name__ == '__main__':
    densenet169 = models.densenet169(pretrained=False)
    num_ftrs = densenet169.classifier.in_features
    densenet169.classifier = nn.Linear(num_ftrs, cfg['out_num'])
    densenet169.load_state_dict(torch.load("/home/lucasxu/ModelZoo/DenseNet169.pth"))

    # shufflenet_v2 = models.shufflenet_v2_x1_0(pretrained=True)
    # num_ftrs = shufflenet_v2.fc.in_features
    # shufflenet_v2.fc = nn.Linear(num_ftrs, cfg['out_num'])

    mobilenet_v2 = models.mobilenet_v2(pretrained=True)
    num_ftrs = mobilenet_v2.classifier[1].in_features
    mobilenet_v2.classifier[1] = nn.Linear(num_ftrs, cfg['out_num'])

    # resnet18 = models.resnet18(pretrained=True)
    # num_ftrs = resnet18.fc.in_features
    # resnet18.fc = nn.Linear(num_ftrs, 6)

    # mixnet_m = ptcv_get_model("mixnet_m", pretrained=True)
    # num_ftrs = mixnet_m.output.in_features
    # mixnet_m.output = nn.Linear(num_ftrs, 6)

    # condensenet74 = ptcv_get_model("condensenet74_c4_g4", pretrained=True)
    # condensenet74.output.linear = nn.Linear(1032, 6, bias=True)

    run_img_classification(teacher_w_weights=densenet169, student_wo_weights=mobilenet_v2, epoch=cfg['epoch'])
