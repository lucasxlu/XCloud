import json
import os
import io
import base64
import time

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from django.http import HttpResponse
from torchvision import models
from torchvision.transforms import transforms
from django.core.files.uploadedfile import InMemoryUploadedFile

from cv.cfg import cfg

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROB_THRESH = 0.3
URL_PORT = 'http://localhost:8000'


class FoodRecognizer:
    """
    Food Recognition Class Wrapper
    """

    def __init__(self, pretrained_model_path="cv/model/DenseNet161_iFood.pth", num_cls=251):
        model = models.densenet161(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_cls)

        model = model.float()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # model = nn.DataParallel(model)
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


def upload_and_rec_food(request):
    """
    upload and recognize food
    :param request:
    :return:
    """
    image_dir = 'cv/static/FoodUpload'
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
            imagepath = URL_PORT + '/static/FoodUpload/' + image.name

            plant_result = food_recognizer.infer(os.path.join(image_dir, image.name))

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
