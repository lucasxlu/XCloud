# Model Inference for NSFW Estimation
# Author: LucasXU
import sys
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

sys.path.append('../')
from research.imgcensor.cfg import cfg


class NSFWEstimator:
    """
    NSFW Estimator Class Wrapper
    """

    def __init__(self, pretrained_model_path):
        model = models.densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, cfg['out_num'])

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model.load_state_dict(torch.load(pretrained_model_path))

        model.to(device)
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
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img = Image.open(img_file)
        if img.mode != 'RGB':  # handle gray-scale & RGBA images
            img = img.convert('RGB')
        
        img = preprocess(img)
        img.unsqueeze_(0)

        img = img.to(self.device)

        outputs = self.model(img)
        outputs = F.softmax(outputs, dim=1)

        topK_prob, topK_label = torch.topk(outputs, self.topK)
        prob = topK_prob.to("cpu").detach().numpy().tolist()

        _, predicted = torch.max(outputs.data, 1)

        return {
            "status": 0,
            "message": "success",
            "results": [
                {
                    "prob": round(prob[0][i], 4),
                    "type": self.mapping[int(topK_label[0][i].to("cpu"))],
                } for i in range(self.topK)
            ]
        }


if __name__ == '__main__':
    nsfw = NSFWEstimator('./model/DenseNet121_NSFW.pth')
    pprint(nsfw.infer('./3.jpg'))
