import base64
import io
import json
import os
import time

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.http import HttpResponse
from skimage import io
from torchvision import models
from torchvision.transforms import transforms

from cv.cfg import cfg

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROB_THRESH = 0.3
URL_PORT = 'http://localhost:8000'


class PlantRecognizer:
    """
    Plant Recognition Class Wrapper
    """

    def __init__(self, pretrained_model_path="cv/model/ResNet50_Plant.pth", num_cls=998):
        model = models.resnet50(pretrained=True)
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

    def infer(self, img_file):
        tik = time.time()
        img = io.imread(img_file)
        img = Image.fromarray(img.astype(np.uint8))

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

        _, predicted = torch.max(outputs.data, 1)

        tok = time.time()

        if prob[0][0] >= cfg['thresholds']['plant_recognition']:
            return {
                'status': 0,
                'message': 'success',
                'elapse': tok - tik,
                'results': [
                    {
                        'name': self.key_type[int(topK_label[0][i].to("cpu"))],
                        'category_id': int(topK_label[0][i].data.to("cpu").numpy()) + 1,
                        'prob': round(prob[0][i], 4)
                    } for i in range(self.topK)
                ]
            }
        else:
            return {
                'status': 0,
                'message': 'success',
                'elapse': tok - tik,
                'results': [
                    {
                        'name': "Unknown",
                        'category_id': -1,
                        'prob': round(prob[0][0], 4)
                    }
                ]
            }

    def infer_from_img_url(self, img_url):
        tik = time.time()
        response = requests.get(img_url, timeout=20)
        if response.status_code in [403, 404, 500]:
            return {
                'status': 2,
                'message': 'Invalid URL',
                'elapse': time.time() - tik,
                'results': None
            }

        else:
            img_content = response.content

            import io
            from PIL import Image
            img_np = np.array(Image.open(io.BytesIO(img_content)))
            img = Image.fromarray(img_np.astype(np.uint8))

            preprocess = transforms.Compose([
                transforms.Resize(227),
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

            _, predicted = torch.max(outputs.data, 1)

            tok = time.time()

            if prob[0][0] >= cfg['thresholds']['plant_recognition']:
                return {
                    'status': 0,
                    'message': 'success',
                    'elapse': tok - tik,
                    'results': [
                        {
                            'name': self.key_type[int(topK_label[0][i].to("cpu"))],
                            'category_id': int(topK_label[0][i].data.to("cpu").numpy()) + 1,
                            'prob': round(prob[0][i], 4)
                        } for i in range(self.topK)
                    ]
                }
            else:
                return {
                    'status': 0,
                    'message': 'success',
                    'elapse': tok - tik,
                    'results': [
                        {
                            'name': "Unknown",
                            'category_id': -1,
                            'prob': round(prob[0][0], 4)
                        }
                    ]
                }


plant_recognizer = PlantRecognizer(num_cls=998)


def upload_and_rec_plant(request):
    """
    upload and recognize plant image
    :param request:
    :return:
    """
    image_dir = 'cv/static/PlantUpload'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    result = {}

    if request.method == "POST":
        image = request.FILES.get("image", None)
        if not isinstance(image, InMemoryUploadedFile):
            imgstr = request.POST.get("image", None)
            if 'http' in imgstr:
                response = requests.get(imgstr)
                image = InMemoryUploadedFile(io.BytesIO(response.content), name="{}.jpg".format(str(time.time())),
                                             field_name="image", content_type="image/jpeg", size=1347415, charset=None)
            else:
                image = InMemoryUploadedFile(io.BytesIO(base64.b64decode(imgstr)),
                                             name="{}.jpg".format(str(time.time())), field_name="image",
                                             content_type="image/jpeg", size=1347415, charset=None)

        if not image:
            result['code'] = 3
            result['msg'] = 'Invalid Path for Image'
            result['results'] = None

            json_result = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_result)
        else:
            destination = open(os.path.join(image_dir, image.name), 'wb+')
            for chunk in image.chunks():
                destination.write(chunk)
            destination.close()

            tik = time.time()
            imagepath = URL_PORT + '/static/PlantUpload/' + image.name

            plant_result = plant_recognizer.infer(os.path.join(image_dir, image.name))

            result['code'] = 0
            result['msg'] = 'success'
            result['imgpath'] = imagepath
            result['results'] = plant_result['results']
            result['elapse'] = round(time.time() - tik, 2)

            json_str = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_str)
    else:
        result['code'] = 3
        result['msg'] = 'Invalid HTTP Method'
        result['data'] = None

        json_result = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_result)
