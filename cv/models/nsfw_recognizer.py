import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from torchvision import models
from torchvision.transforms import transforms

from cv.cfg import cfg


class NSFWEstimator:
    """
    NSFW Estimator Class Wrapper
    """

    def __init__(self, pretrained_model_path=os.path.join(cfg['model_zoo_base'], "DenseNet121_NSFW.pth"), num_cls=5):
        self.num_cls = num_cls
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, self.num_cls)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')))

        model.eval()

        self.device = device
        self.model = model
        self.topK = 5
        self.mapping = {
            0: 'drawings',
            1: 'hentai',
            2: 'neutral',
            3: 'porn',
            4: 'sexy'
        }

    def infer(self, img_file):
        img = Image.open(img_file)

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

        return [
            {
                "prob": round(prob[0][i], 4),
                "type": self.mapping[int(topK_label[0][i].to("cpu"))],
            } for i in range(self.topK)
        ]


nsfw_estimator = NSFWEstimator(num_cls=5)
