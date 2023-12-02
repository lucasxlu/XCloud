import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision.transforms import transforms

from cv.cfg import cfg


class SkinDiseaseRecognizer:
    """
    Skin Disease Recognizer
    pre-trained checkpoint: https://drive.google.com/file/d/1yqSDfVm32Kjn_uNMeANuApPcdGsbijXQ/view?usp=sharing
    """

    def __init__(self, num_cls, pretrained_model=os.path.join(cfg['model_zoo_base'], 'DenseNet121_SD198.pth')):
        assert os.path.exists(pretrained_model)

        self.num_cls = num_cls
        densenet121 = models.densenet121(pretrained=False)
        num_ftrs = densenet121.classifier.in_features
        densenet121.classifier = nn.Linear(num_ftrs, self.num_cls)

        model = densenet121.float()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        model.load_state_dict(torch.load(pretrained_model, map_location=torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')))
        # model.load_state_dict(torch.load(pretrained_model, map_location=lambda storage, loc: storage))

        model.eval()
        self.model = model
        self.device = device
        self.topK = 5
        self.mapping = {}

        with open('cv/classes.txt', mode='rt', encoding='utf-8') as f:
            for line in f.readlines():
                self.mapping[int(line.split(' ')[0].strip()) - 1] = line.split(' ')[1]

    def infer(self, img):
        if isinstance(img, str):
            img = Image.open(img)

        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = preprocess(img)
        img.unsqueeze_(0)

        img = img.to(self.device)

        outputs = self.model(img)
        outputs = F.softmax(outputs, dim=1)

        topK_prob, topK_label = torch.topk(outputs, self.topK)
        prob = topK_prob.to("cpu").detach().numpy().tolist()

        _, predicted = torch.max(outputs.data, 1)

        if prob[0][0] >= cfg['thresholds']['skin_disease_recognition']:
            return [
                {
                    "disease": self.mapping[int(topK_label[0][i].to("cpu"))],
                    "probability": round(prob[0][i], 4),
                } for i in range(self.topK)
            ]
        else:
            return [
                {
                    "disease": "Unknown",
                    "probability": round(prob[0][0], 4),
                }
            ]


skin_disease_recognizer = SkinDiseaseRecognizer(num_cls=198)
