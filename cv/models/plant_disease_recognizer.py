import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision.transforms import transforms

from cv.cfg import cfg


class PlantDiseaseRecognizer:
    """
    Plant Disease Recognition Class Wrapper
    pre-trained checkpoint: https://drive.google.com/file/d/1yqSDfVm32Kjn_uNMeANuApPcdGsbijXQ/view?usp=sharing
    """

    def __init__(self, pretrained_model_path=os.path.join(cfg['model_zoo_base'], "ResNet18_PDR.pth"),
                 num_cls=61):
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_cls)

        model = model.float()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # model = nn.DataParallel(model)
        # model.load_state_dict(torch.load(pretrained_model_path))

        if torch.cuda.device_count() > 1:
            print("We are running on", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device(
                'cuda:0') if torch.cuda.is_available() else torch.device('cpu')))
        else:
            state_dict = torch.load(pretrained_model_path)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        model.eval()

        self.device = device
        self.model = model
        self.key_type = {
            0: '苹果健康',
            1: '苹果黑星病一般',
            2: '苹果黑星病严重',
            3: '苹果灰斑病',
            4: '苹果雪松锈病一般',
            5: '苹果雪松锈病严重',
            6: '樱桃健康',
            7: '樱桃白粉病一般',
            8: '樱桃白粉病严重',
            9: '玉米健康',
            10: '玉米灰斑病一般',
            11: '玉米灰斑病严重',
            12: '玉米锈病一般',
            13: '玉米锈病严重',
            14: '玉米叶斑病一般',
            15: '玉米叶斑病严重',
            16: '玉米花叶病毒病',
            17: '葡萄健康',
            18: '葡萄黑腐病一般',
            19: '葡萄黑腐病严重',
            20: '葡萄轮斑病一般',
            21: '葡萄轮斑病严重',
            22: '葡萄褐斑病一般',
            23: '葡萄褐斑病严重',
            24: '柑桔健康',
            25: '柑桔黄龙病一般',
            26: '柑桔黄龙病严重',
            27: '桃健康',
            28: '桃疮痂病一般',
            29: '桃疮痂病严重',
            30: '辣椒健康',
            31: '辣椒疮痂病一般',
            32: '辣椒疮痂病严重',
            33: '马铃薯健康',
            34: '马铃薯早疫病一般',
            35: '马铃薯早疫病严重',
            36: '马铃薯晚疫病一般',
            37: '马铃薯晚疫病严重',
            38: '草莓健康',
            39: '草莓叶枯病一般',
            40: '草莓叶枯病严重',
            41: '番茄健康',
            42: '番茄白粉病一般',
            43: '番茄白粉病严重',
            44: '番茄疮痂病一般',
            45: '番茄疮痂病严重',
            46: '番茄早疫病一般',
            47: '番茄早疫病严重',
            48: '番茄晚疫病菌一般',
            49: '番茄晚疫病菌严重',
            50: '番茄叶霉病一般',
            51: '番茄叶霉病严重',
            52: '番茄斑点病一般',
            53: '番茄斑点病严重',
            54: '番茄斑枯病一般',
            55: '番茄斑枯病严重',
            56: '番茄红蜘蛛损伤一般',
            57: '番茄红蜘蛛损伤严重',
            58: '番茄黄化曲叶病毒病一般',
            59: '番茄黄化曲叶病毒病严重',
            60: '番茄花叶病毒病'
        }
        self.topK = 5

    def infer(self, img_file):
        img = Image.open(img_file)

        preprocess = transforms.Compose([
            transforms.Resize(227),
            transforms.RandomCrop(224),
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

        if prob[0][0] >= cfg['thresholds']['plant_disease_recognition']:
            return [
                {
                    'name': self.key_type[int(topK_label[0][i].to("cpu"))],
                    'disease': int(topK_label[0][i].data.to("cpu").numpy()),
                    'prob': round(prob[0][i], 4)
                } for i in range(self.topK)
            ]
        else:
            return [
                {
                    'name': "Unknown",
                    'disease': -1,
                    'prob': round(prob[0][0], 4)
                }
            ]


pdr = PlantDiseaseRecognizer(num_cls=61)
