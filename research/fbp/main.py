"""
Deep Learning for Facial Beauty Prediction
Author: XuLu
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.optim import lr_scheduler

sys.path.append('../')
from fbp.losses import CRLoss
from fbp.models import CRNet, SCUTNet
from fbp.utils import mkdirs_if_not_exist
from fbp import data_loader, cfg
from fbp.shufflenet_v2 import ShuffleNetV2


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs,
                inference=False):
    """
    model training
    :param model:
    :param train_dataloader:
    :param test_dataloader:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :param model_name:
    :param inference:
    :return:
    """
    model = model.float()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    if not inference:
        print('Start training %s...' % model.__class__.__name__)
        for epoch in range(num_epochs):
            model.train()
            scheduler.step()

            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                inputs, label = data['image'], data['label']

                inputs = inputs.to(device)
                label = label.to(device).float()

                optimizer.zero_grad()

                inputs = inputs.float()
                # label = label.view(cfg['batch_size'], 1)  # for regression only

                out = model(inputs)
                loss = criterion(out, label.unsqueeze(-1))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:  # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0

            if epoch % 10 == 9:
                model.eval()
                with torch.no_grad():
                    tmp_y_pred = []
                    tmp_y_true = []
                    tmp_filenames = []

                    for data in test_dataloader:
                        images, labels, filename = data['image'], data['label'], data['filename']
                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = model(images)
                        # _, predicted = torch.max(outputs.data, 1)

                        tmp_y_pred += outputs.to("cpu").detach().numpy().tolist()
                        tmp_y_true += labels.to("cpu").detach().numpy().tolist()
                        tmp_filenames += filename

                model_path_dir = './model'
                mkdirs_if_not_exist(model_path_dir)
                torch.save(model.state_dict(),
                           os.path.join(model_path_dir, model.__class__.__name__ + '-epoch-%d.pth' % epoch))

                # print(len(tmp_y_true))
                # print(len(tmp_y_pred))
                #
                # print(np.array(tmp_y_true).shape)
                # print(np.array(tmp_y_pred).shape)

                rmse_lr = round(np.math.sqrt(mean_squared_error(np.array(tmp_y_true), np.array(tmp_y_pred).ravel())), 4)
                mae_lr = round(mean_absolute_error(np.array(tmp_y_true), np.array(tmp_y_pred).ravel()), 4)
                pc = round(np.corrcoef(np.array(tmp_y_true), np.array(tmp_y_pred).ravel())[0, 1], 4)
                print('RMSE of {0} on test set: {1} '.format(model.__class__.__name__, rmse_lr))
                print('MAE of {0} on test set: {1} '.format(model.__class__.__name__, mae_lr))
                print('PC of {0} on test set: {1} '.format(model.__class__.__name__, pc))

                model.train()

        print('Finished training %s...\n' % model.__class__.__name__)
        print('Saving trained model...')
        model_path_dir = './model'
        mkdirs_if_not_exist(model_path_dir)
        torch.save(model.state_dict(), os.path.join(model_path_dir, '%s.pth' % model.__class__.__name__))
        print('%s has been saved successfully~' % model.__class__.__name__)

    else:
        print('Loading pre-trained model...')
        model.load_state_dict(torch.load(os.path.join('./model/%s.pth' % model.__class__.__name__)))

    model.eval()
    print('Start testing %s...' % model.__class__.__name__)
    y_pred = []
    y_true = []
    filenames = []

    with torch.no_grad():
        for data in test_dataloader:
            images, labels, filename = data['image'], data['label'], data['filename']
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # _, predicted = torch.max(outputs.data, 1)

            y_pred += outputs.to("cpu").detach().numpy().tolist()
            y_true += labels.to("cpu").detach().numpy().tolist()
            filenames += filename

    rmse_lr = round(np.math.sqrt(mean_squared_error(np.array(y_true), np.array(y_pred).ravel())), 4)
    mae_lr = round(mean_absolute_error(np.array(y_true), np.array(y_pred).ravel()), 4)
    pc = round(np.corrcoef(np.array(y_true), np.array(y_pred).ravel())[0, 1], 4)
    print('RMSE of {0} on test set: {1} '.format(model.__class__.__name__, rmse_lr))
    print('MAE of {0} on test set: {1} '.format(model.__class__.__name__, mae_lr))
    print('PC of {0} on test set: {1} '.format(model.__class__.__name__, pc))

    print('Output CSV...')
    col = ['filename', 'gt', 'pred']
    df = pd.DataFrame([[filenames[i], y_true[i], y_pred[i]] for i in range(len(filenames))],
                      columns=col)
    df.to_csv("./output.csv", index=False)
    print('CSV has been generated...')


def train_model_with_crloss(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs=25,
                            inference=False):
    """
    train and eval with CRLoss
    :param model:
    :param train_dataloader:
    :param test_dataloader:
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

    if not inference:
        model.train()
        print('Start training CRNet...')
        for epoch in range(num_epochs):
            scheduler.step()

            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                inputs, scores, classes = data['image'], data['label'], data['class']

                inputs = inputs.to(device)
                scores = scores.to(device)
                classes = classes.to(device)

                optimizer.zero_grad()

                inputs = inputs.float()
                # scores = scores.float().view(cfg['batch_size'], -1)
                # classes = classes.int().view(cfg['batch_size'], 3)

                reg_out, cls_out = model(inputs)
                loss = criterion(cls_out, classes, reg_out, scores.float().unsqueeze(-1))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:  # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0

            if epoch % 10 == 9:
                model.eval()
                with torch.no_grad():
                    tmp_y_pred = []
                    tmp_y_true = []
                    tmp_filenames = []

                    for data in test_dataloader:
                        images, labels, classes, filename = data['image'], data['label'], data['class'], data[
                            'filename']

                        images = images.to(device)
                        labels = labels.to(device)

                        reg_outputs, cls_outputs = model(images)
                        # _, predicted = torch.max(outputs.data, 1)

                        tmp_y_true += labels.to("cpu").detach().numpy().tolist()
                        tmp_y_pred += reg_outputs.to("cpu").detach().numpy().tolist()
                        tmp_filenames += filename

                model_path_dir = './model'
                mkdirs_if_not_exist(model_path_dir)
                torch.save(model.state_dict(),
                           os.path.join(model_path_dir, model.__class__.__name__ + '-epoch-%d.pth' % (epoch + 1)))

                rmse_lr = round(np.math.sqrt(mean_squared_error(np.array(tmp_y_true), np.array(tmp_y_pred).ravel())), 4)
                mae_lr = round(mean_absolute_error(np.array(tmp_y_true), np.array(tmp_y_pred).ravel()), 4)
                pc = round(np.corrcoef(np.array(tmp_y_true), np.array(tmp_y_pred).ravel())[0, 1], 4)
                print('RMSE of {0} on test set: {1} '.format(model.__class__.__name__, rmse_lr))
                print('MAE of {0} on test set: {1} '.format(model.__class__.__name__, mae_lr))
                print('PC of {0} on test set: {1} '.format(model.__class__.__name__, pc))

                model.train()

        print('Finished training CRNet...\n')
        print('Saving trained model...')
        model_path_dir = './model'
        mkdirs_if_not_exist(model_path_dir)
        torch.save(model.state_dict(), os.path.join(model_path_dir, '%s.pth' % model.__class__.__name__))
        print('CRNet has been saved successfully~')

    else:
        print('Loading pre-trained model...')
        model.load_state_dict(torch.load(os.path.join('./model/%s.pth' % model.__class__.__name__)))

    model.eval()

    print('Start testing CRNet...')
    predicted_labels = []
    gt_labels = []
    filenames = []
    for data in test_dataloader:
        images, scores, classes, filename = data['image'], data['label'], data['class'], data['filename']
        images = images.to(device)

        reg_out, cls_out = model.forward(images)

        # bat_list = []
        # for out in F.softmax(cls_out).to("cpu"):
        #     tmp = 0
        #     for i in range(0, 3, 1):
        #         tmp += out[i] * (i - 1)
        #     bat_list.append(float(tmp.detach().numpy()))

        # predicted_labels += (0.6 * reg_out.to("cpu").detach().numpy() + 0.4 * np.array(bat_list)).tolist()

        predicted_labels += reg_out.to("cpu").detach().numpy().tolist()
        gt_labels += scores.to("cpu").detach().numpy().tolist()
        filenames += filename

    mae_lr = round(mean_absolute_error(np.array(gt_labels), np.array(predicted_labels).ravel()), 4)
    rmse_lr = round(np.math.sqrt(mean_squared_error(np.array(gt_labels), np.array(predicted_labels).ravel())), 4)
    pc = round(np.corrcoef(np.array(gt_labels), np.array(predicted_labels).ravel())[0, 1], 4)

    print('===============The Mean Absolute Error of CRNet is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of CRNet is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of CRNet is {0}===================='.format(pc))

    col = ['filename', 'gt', 'pred']
    df = pd.DataFrame([[filenames[i], gt_labels[i], predicted_labels[i][0]] for i in range(len(gt_labels))],
                      columns=col)
    df.to_excel("./output.xlsx", sheet_name='Output', index=False)
    print('Output Excel has been generated~')


def run_fbp(model, epoch):
    """
    run FBP
    :param model:
    :param epoch:
    :return:
    """
    criterion = nn.MSELoss()

    optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

    print('start loading SCUTFBP5500 dataset...')
    trainloader, testloader = data_loader.load_scutfbp5500_data()

    train_model(model=model, train_dataloader=trainloader, test_dataloader=testloader,
                criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, num_epochs=epoch,
                inference=False)


def run_crnet(model, epoch=100):
    """
    run CRNet on SCUT-FBP5500
    :param model:
    :param epoch:
    :return:
    """
    criterion = CRLoss()

    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

    print('start loading SCUTFBP5500 dataset...')
    train_loader, test_loader = data_loader.load_scutfbp5500_data()
    train_model_with_crloss(model=model, train_dataloader=train_loader, test_dataloader=test_loader,
                            criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, num_epochs=epoch,
                            inference=False)


if __name__ == '__main__':
    # run ResNet18 on SCUT-FBP5500
    # resnet = models.resnet18(pretrained=False)
    # num_ftrs = resnet.fc.in_features
    # resnet.fc = nn.Linear(num_ftrs, 1)
    # run_fbp(model=resnet, epoch=200)

    # scutnet = SCUTNet()
    shufflenetv2 = ShuffleNetV2()
    run_fbp(model=shufflenetv2, epoch=200)

    # crnet = CRNet()
    # run_crnet(crnet, epoch=250)
