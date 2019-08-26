"""
convert pretrained PyTorch model to ONNX/TensorRT model
TensorRT V5.1.2
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from torchvision import models
from torch2trt import torch2trt
from torch2trt import TRTModule

sys.path.append('../')
from mengzhu import data_loader


def cvt_pytorch_model_to_trt_model(model, pretrained_pytorch_weights, save_to_trt_weights):
    """
    convert pytorch model to TensorRT model
    :param model:
    :param pretrained_pytorch_weights:
    :param save_to_trt_weights:
    :return:
    """
    state_dict = torch.load(pretrained_pytorch_weights)
    try:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(state_dict)

    model = model.cuda().eval()
    data_placeholder = torch.randn((1, 3, 224, 224)).cuda()
    model_trt = torch2trt(model, [data_placeholder], fp16_mode=True)
    torch.save(model_trt.state_dict(), save_to_trt_weights)
    print('done!')


def eval_trt_model(trt_model_weights, py_model, py_model_weights, dataloader):
    """
    evaluate TensorRT model
    :param trt_model_weights:
    :param py_model:
    :param py_model_weights:
    :param dataloader:
    :return:
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[INFO] start loading TensorRT model weights')
    trt_model = TRTModule()
    trt_model.load_state_dict(torch.load(trt_model_weights))
    trt_model.eval()
    print('[INFO] finish loading TensorRT model weights')

    print('[INFO] start loading PyTorch model weights')
    state_dict = torch.load(py_model_weights)
    try:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        py_model.load_state_dict(new_state_dict)
    except:
        py_model.load_state_dict(state_dict)
    print('[INFO] finish loading PyTorch model weights')
    py_model = py_model.to(device)
    py_model.eval()

    y_py_pred = []
    filenames = []
    gts = []
    py_probs = []
    tik = time.time()
    with torch.no_grad():
        for data in dataloader:
            images, types, filename = data['image'], data['type'], data['filename']
            images = images.to(device)

            outputs = py_model(images)
            outputs = F.softmax(outputs)
            # get TOP-K output labels and corresponding probabilities
            topK_prob, topK_label = torch.topk(outputs, 2)
            py_probs += topK_prob.to("cpu").detach().numpy().tolist()

            _, predicted = torch.max(outputs.data, 1)

            y_py_pred += predicted.to("cpu").detach().numpy().tolist()
            gts += types.tolist()
            filenames += filename
    tok = time.time()
    print('Output CSV of PyTorch Model')
    col = ['filename', 'pred', 'prob']
    df = pd.DataFrame([[filenames[i], y_py_pred[i], py_probs[i][0]] for i in range(len(filenames))],
                      columns=col)
    df.to_csv("./InferencePy.csv", index=False)
    print('CSV of PyTorch Model has been generated...')
    print('-' * 100)
    print('[INFO] Confusion Matrix of PyTorch Model:')
    cm = confusion_matrix(gts, y_py_pred)
    print(cm)
    print('[INFO] Accuracy of PyTorch Model: {}'.format(accuracy_score(gts, y_py_pred)))
    print('[INFO] Precision of PyTorch Model: {}'.format(precision_score(gts, y_py_pred, average='macro')))
    print('[INFO] Recall of PyTorch Model: {}'.format(recall_score(gts, y_py_pred, average='macro')))
    print('[INFO] FPS of PyTorch model: {}'.format(dataloader.__len__() / (tok - tik)))
    print('-' * 100)

    y_trt_pred = []
    filenames = []
    gts = []
    trt_probs = []
    tik = time.time()
    with torch.no_grad():
        for data in dataloader:
            images, types, filename = data['image'], data['type'], data['filename']
            images = images.to(device)

            outputs = trt_model(images)
            outputs = F.softmax(outputs)
            # get TOP-K output labels and corresponding probabilities
            topK_prob, topK_label = torch.topk(outputs, 2)
            trt_probs += topK_prob.to("cpu").detach().numpy().tolist()

            _, predicted = torch.max(outputs.data, 1)

            y_trt_pred += predicted.to("cpu").detach().numpy().tolist()
            gts += types.tolist()
            filenames += filename
    tok = time.time()
    print('Output CSV of TensorRT Model')
    col = ['filename', 'pred', 'prob']
    df = pd.DataFrame([[filenames[i], y_trt_pred[i], trt_probs[i][0]] for i in range(len(filenames))],
                      columns=col)
    df.to_csv("./InferenceTRT.csv", index=False)
    print('CSV of TensorRT Model has been generated...')
    print('-' * 100)
    print('[INFO] Confusion Matrix of TensorRT Model:')
    cm = confusion_matrix(gts, y_trt_pred)
    print(cm)
    print('[INFO] Accuracy of TensorRT Model: {}'.format(accuracy_score(gts, y_trt_pred)))
    print('[INFO] Precision of TensorRT Model: {}'.format(precision_score(gts, y_trt_pred, average='macro')))
    print('[INFO] Recall of TensorRT Model: {}'.format(recall_score(gts, y_trt_pred, average='macro')))
    print('[INFO] FPS of TensorRT model: {}'.format(dataloader.__len__() / (tok - tik)))
    print('-' * 100)


if __name__ == '__main__':
    densenet169 = models.densenet169(pretrained=True)
    num_ftrs = densenet169.classifier.in_features
    densenet169.classifier = nn.Linear(num_ftrs, 10)
    cvt_pytorch_model_to_trt_model(densenet169, '/data/lucasxu/ModelZoo/DenseNet169_DailyFreshMeal.pth',
                                   '/data/lucasxu/ModelZoo/DenseNet169_DailyFreshMeal_trt.pth')

    # trainloader, valloader, testloader = data_loader.load_daily_fresh_meal_data()
    # eval_trt_model(trt_model_weights='/data/lucasxu/ModelZoo/DenseNet169_DailyFreshMeal_trt.pth',
    #                py_model=densenet169, py_model_weights='/data/lucasxu/ModelZoo/DenseNet169_DailyFreshMeal.pth',
    #                dataloader=testloader)
