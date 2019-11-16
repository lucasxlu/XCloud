import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from torchvision import models
from torchvision.transforms import transforms

from cv.cfg import cfg


class FoodRecognizer:
    """
    Food Recognition Class Wrapper
    """

    def __init__(self, pretrained_model_path=os.path.join(cfg['model_zoo_base'], "DenseNet161_iFood.pth"), num_cls=251):
        model = models.densenet161(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_cls)

        model = model.float()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')))

        # if torch.cuda.device_count() > 1:
        #     print("We are running on", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model)
        #     model.load_state_dict(torch.load(pretrained_model_path))
        # else:
        #     state_dict = torch.load(pretrained_model_path)
        #     from collections import OrderedDict
        #     new_state_dict = OrderedDict()
        #     for k, v in state_dict.items():
        #         name = k[7:]  # remove `module.`
        #         new_state_dict[name] = v
        #
        #     model.load_state_dict(new_state_dict)

        model.eval()

        self.device = device
        self.model = model
        self.key_type = {}

        with open('cv/food_class_list.txt', encoding='utf-8', mode='rt') as f:
            for line in f.readlines():
                self.key_type[int(line.split(' ')[0].strip())] = line.split(' ')[1].strip()

        self.topK = 5

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

        # get TOP-K output labels and corresponding probabilities
        topK_prob, topK_label = torch.topk(outputs, self.topK)
        prob = topK_prob.to("cpu").detach().numpy().tolist()

        _, predicted = torch.max(outputs.data, 1)

        if prob[0][0] >= cfg['thresholds']['food_recognition']:
            return {
                'status': 0,
                'message': 'success',
                'results': [
                    {
                        'name': self.key_type[int(topK_label[0][i].to("cpu"))] if int(
                            topK_label[0][i].to("cpu")) in self.key_type.keys() else 'Unknown',
                        'category': int(topK_label[0][i].data.to("cpu").numpy()),
                        'prob': round(prob[0][i], 4)
                    } for i in range(self.topK)
                ]
            }
        else:
            return {
                'status': 0,
                'message': 'success',
                'results': [
                    {
                        'name': 'Unknown',
                        'category': -1,
                        'prob': round(prob[0][0], 4)
                    }
                ]
            }


food_recognizer = FoodRecognizer(num_cls=251)
