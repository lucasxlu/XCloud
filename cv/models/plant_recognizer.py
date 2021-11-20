import os

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision.transforms import transforms

from cv.cfg import cfg


class PlantRecognizer:
    """
    Plant Recognition Class Wrapper
    """

    def __init__(self, pretrained_model_path=os.path.join(cfg['model_zoo_base'], "ResNet50_Plant.pth"), num_cls=998):
        assert os.path.exists(pretrained_model_path)

        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_cls)

        model = model.float()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        if torch.cuda.device_count() > 1:
            print("We are running on", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(pretrained_model_path))
        else:
            state_dict = torch.load(pretrained_model_path, map_location=torch.device(
                'cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

        model.eval()

        df = pd.read_csv('cv/label.csv')
        key_type = {}
        for i in range(len(df['category_name'].tolist())):
            key_type[int(df['category_name'].tolist()[i].split('_')[-1]) - 1] = df['label'].tolist()[i]

        self.device = device
        self.model = model
        self.key_type = key_type
        self.topK = 5

    def infer(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
        mat = Image.fromarray(img)
        b, g, r = mat.split()
        img = Image.merge("RGB", (r, g, b))

        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = preprocess(img)
        img.unsqueeze_(0)

        img = img.to(self.device)

        outputs = self.model.forward(img)
        outputs = F.softmax(outputs, dim=1)

        # get TOP-K output labels and corresponding probabilities
        topK_prob, topK_label = torch.topk(outputs, self.topK)
        prob = topK_prob.to("cpu").detach().numpy().tolist()

        print('[WARN]', prob[0][0])

        if prob[0][0] >= cfg['thresholds']['plant_recognition']:
            return [
                {
                    'name': self.key_type[int(topK_label[0][i].to("cpu"))],
                    'category_id': int(topK_label[0][i].data.to("cpu").numpy()) + 1,
                    'prob': round(prob[0][i], 4)
                } for i in range(self.topK)
            ]
        else:
            return [
                {
                    'name': "Unknown",
                    'category_id': -1,
                    'prob': round(prob[0][0], 4)
                }
            ]


plant_recognizer = PlantRecognizer(num_cls=998)
