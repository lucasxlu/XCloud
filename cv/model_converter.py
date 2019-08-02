"""
convert pretrained PyTorch model to ONNX/TensorRT
TensorRT V5.1.2
"""
import os

import torch
import torch.nn as nn
from torchvision import models
from torch2trt import torch2trt


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


if __name__ == '__main__':
    resnet18 = models.resnet18(pretrained=True)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 6)
    cvt_pytorch_model_to_trt_model(resnet18, '/data/lucasxu/ModelZoo/ResNet18_MengZhu_Discard_Quality.pth',
                                   '/data/lucasxu/ModelZoo/ResNet18_MengZhu_Discard_Quality_trt.pth')
