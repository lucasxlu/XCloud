import base64
import io
import json
import os
import time

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.http import HttpResponse
from torchvision import models
from torchvision.transforms import transforms

from cv.cfg import cfg

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROB_THRESH = 0.3
URL_PORT = 'http://localhost:8000'


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, np.float32):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)


class SkinDiseaseRecognizer:
    """
    Skin Disease Recognizer
    """

    def __init__(self, num_cls, pretrained_model='cv/model/DenseNet121_SD198.pth'):
        self.num_cls = num_cls

        densenet121 = models.densenet121(pretrained=True)
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

    def infer_from_img_file(self, img_path):
        img = Image.open(img_path)

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
            return {
                "status": 0,
                "message": "success",
                "results": [
                    {
                        "disease": self.mapping[int(topK_label[0][i].to("cpu"))],
                        "probability": round(prob[0][i], 4),
                    } for i in range(self.topK)
                ]
            }
        else:
            return {
                "status": 0,
                "message": "success",
                "results": [
                    {
                        "disease": "Unknown",
                        "probability": round(prob[0][0], 4),
                    }
                ]
            }

    def infer_from_img(self, img):
        import io
        from PIL import Image
        img_np = np.array(Image.open(io.BytesIO(img)))
        img = Image.fromarray(img_np.astype(np.uint8))

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
            return {
                "status": 0,
                "message": "success",
                "results": [
                    {
                        "disease": self.mapping[int(topK_label[0][i].to("cpu"))],
                        "probability": round(prob[0][i], 4),
                    } for i in range(self.topK)
                ]
            }
        else:
            return {
                "status": 0,
                "message": "success",
                "results": [
                    {
                        "disease": "Unknown",
                        "probability": round(prob[0][0], 4),
                    }
                ]
            }


skin_disease_recognizer = SkinDiseaseRecognizer(num_cls=198)


def upload_and_rec_skin_disease(request):
    """
    upload and recognize skin disease
    :param request:
    :return:
    """
    image_dir = 'cv/static/SkinUpload'
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
            result['msg'] = 'Invalid Path for Skin Image'
            result['results'] = None

            json_result = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_result)
        else:
            destination = open(os.path.join(image_dir, image.name), 'wb+')
            for chunk in image.chunks():
                destination.write(chunk)
            destination.close()

            tik = time.time()
            imagepath = URL_PORT + '/static/SkinUpload/' + image.name

            skin_disease = skin_disease_recognizer.infer_from_img_file(os.path.join(image_dir, image.name))
            # skin_disease = skin_disease_recognizer.infer_from_img(destination)

            result['code'] = 0
            result['msg'] = 'success'
            result['imgpath'] = imagepath
            if max(_['probability'] for _ in skin_disease['results']) > PROB_THRESH:
                result['results'] = skin_disease['results']
            else:
                result['results'] = [{'disease': 'Unknown', 'probability': skin_disease['results'][0]['probability']}]
            result['elapse'] = round(time.time() - tik, 2)

            json_str = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_str)
    else:
        result['code'] = 3
        result['msg'] = 'Invalid HTTP Method'
        result['data'] = None
        result['elapse'] = 0

        json_result = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_result)
