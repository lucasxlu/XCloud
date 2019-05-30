import json
import os
import requests
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.linalg import norm
import pandas as pd
import cv2
from PIL import Image
from django.http import HttpResponse
from mtcnn.mtcnn import MTCNN
from skimage import io
from sklearn.externals import joblib
from torchvision import models
from torchvision.transforms import transforms

from cv import features
from cv.feat_extractor import ext_feats
from cv.net_sphere import SphereFaceNet
from cv.shufflenet_v2 import ShuffleNetV2
from cv.cfg import cfg

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROB_THRESH = 0.3
URL_PORT = 'http://localhost:8000'


class BeautyRecognizerML:
    """
    non-deep learning based facial beauty predictor
    """

    def __init__(self, pretrained_model='cv/model/GradientBoostingRegressor.pkl'):
        gbr = joblib.load(pretrained_model)
        self.model = gbr
        self.detector = MTCNN()

    def infer(self, img_path):
        img = cv2.imread(img_path)
        mtcnn_result = detect_face(self.detector, img_path)
        bbox = mtcnn_result[0]['box']

        margin_pixel = 10
        face_region = img[bbox[0] - margin_pixel: bbox[0] + bbox[2] + margin_pixel,
                      bbox[1] - margin_pixel: bbox[1] + bbox[3] + margin_pixel]

        ratio = max(face_region.shape[0], face_region.shape[1]) / min(face_region.shape[0], face_region.shape[1])
        if face_region.shape[0] < face_region.shape[1]:
            face_region = cv2.resize(face_region, (int(ratio * 64), 64))
            face_region = face_region[:,
                          int((face_region.shape[0] - 64) / 2): int((face_region.shape[0] - 64) / 2) + 64]
        else:
            face_region = cv2.resize(face_region, (64, int(ratio * 64)))
            face_region = face_region[int((face_region.shape[1] - 64) / 2): int((face_region.shape[1] - 64) / 2) + 64,
                          :]

        return self.model.predict(np.array(features.HOG_from_cv(face_region).reshape(1, -1)))[0]


class BeautyRecognizer:
    """
    Facial Beauty Predictor Powered by ShuffleNetV2
    """

    def __init__(self, pretrained_model='cv/model/ShuffleNetV2.pth'):
        model = ShuffleNetV2()

        model = model.float()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # model.load_state_dict(torch.load(pretrained_model))
        model.load_state_dict(torch.load(pretrained_model, map_location=lambda storage, loc: storage))

        model.eval()
        self.model = model
        self.device = device
        self.detector = MTCNN()

    def infer(self, img_path):
        img = cv2.imread(img_path)
        mtcnn_result = detect_face(self.detector, img_path)

        if len(mtcnn_result) > 0:
            bbox = mtcnn_result[0]['box']

            margin_pixel = 10
            face_region = img[bbox[0] - margin_pixel: bbox[0] + bbox[2] + margin_pixel,
                          bbox[1] - margin_pixel: bbox[1] + bbox[3] + margin_pixel]

            cv2.rectangle(img, (bbox[0] - margin_pixel, bbox[1] - margin_pixel),
                          (bbox[0] + bbox[2] + margin_pixel, bbox[1] + bbox[3] + margin_pixel), (232, 171, 74), 2)
            cv2.imwrite(img_path, img)

            ratio = max(face_region.shape[0], face_region.shape[1]) / min(face_region.shape[0], face_region.shape[1])
            if face_region.shape[0] < face_region.shape[1]:
                face_region = cv2.resize(face_region, (int(ratio * 64), 64))
                face_region = face_region[:,
                              int((face_region.shape[0] - 64) / 2): int((face_region.shape[0] - 64) / 2) + 64]
            else:
                face_region = cv2.resize(face_region, (64, int(ratio * 64)))
                face_region = face_region[int((face_region.shape[1] - 64) / 2): int((face_region.shape[1] - 64) / 2) +
                                                                                64, :]

            face_region = Image.fromarray(face_region.astype(np.uint8))
            preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            face_region = preprocess(face_region)
            face_region.unsqueeze_(0)
            face_region = face_region.to(self.device)

            return {"beauty": float(self.model.forward(face_region).data.to("cpu").numpy()), "mtcnn": mtcnn_result[0]}
        else:
            return None


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


class NSFWEstimator:
    """
    NSFW Estimator Class Wrapper
    """

    def __init__(self, pretrained_model_path="cv/model/DenseNet121_NSFW.pth", num_cls=5):
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


class PlantDiseaseRecognizer:
    """
    Plant Disease Recognition Class Wrapper
    """

    def __init__(self, pretrained_model_path="cv/model/ResNet18_PDR.pth", num_cls=61):
        model = models.resnet18(pretrained=True)
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
        tik = time.time()
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
        tok = time.time()

        if prob[0][0] >= cfg['thresholds']['plant_disease_recognition']:
            return {
                'status': 0,
                'message': 'success',
                'elapse': tok - tik,
                'results': [
                    {
                        'name': self.key_type[int(topK_label[0][i].to("cpu"))],
                        'disease': int(topK_label[0][i].data.to("cpu").numpy()),
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
                        'disease': -1,
                        'prob': round(prob[0][0], 4)
                    }
                ]
            }


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


class FaceSearcher:
    """
    Face Search Class Wrapper
    """

    def __init__(self, face_feats_path="cv/model/hzau_master_face_features.pkl"):
        with open(face_feats_path, mode='rb') as f:
            face_feats_list = pickle.load(f)

        self.topK = 10
        self.face_feats_list = face_feats_list

    def search(self, img_file):
        face_feat = ext_feats(sphere_face=SphereFaceNet(feature=True), img_path=img_file)

        compare_result = {}
        for face_obj in self.face_feats_list:
            cos_sim = np.dot(face_feat, face_obj['feature']) / (norm(face_feat) * norm(face_obj['feature']))
            compare_result[face_feat['studentid']] = cos_sim

        sorted_compare_result = sorted(compare_result.items(), key=lambda kv: kv[1])

        return {
            'status': 0,
            'message': 'success',
            'results': sorted_compare_result[self.topK]
        }


beauty_recognizer = BeautyRecognizer()
skin_disease_recognizer = SkinDiseaseRecognizer(num_cls=198)
nsfw_estimator = NSFWEstimator(num_cls=5)
plant_recognizer = PlantRecognizer(num_cls=998)
pdr = PlantDiseaseRecognizer(num_cls=61)
food_recognizer = FoodRecognizer(num_cls=251)
face_searcher = FaceSearcher()


def upload_and_rec_beauty(request):
    """
    upload and recognize image
    :param request:
    :return:
    """
    image_dir = 'cv/static/FaceUpload'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    result = {}

    if request.method == "POST":
        image = request.FILES.get("image", None)
        if not image:
            result['code'] = 1
            result['msg'] = 'Invalid Path for Face Image'
            result['data'] = None

            json_result = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_result)
        else:
            destination = open(os.path.join(image_dir, image.name), 'wb+')
            for chunk in image.chunks():
                destination.write(chunk)
            destination.close()

            tik = time.time()
            imagepath = URL_PORT + '/static/FaceUpload/' + image.name

            res = beauty_recognizer.infer(os.path.join(image_dir, image.name))

            if res is not None:
                result['code'] = 0
                result['msg'] = 'success'
                result['data'] = {
                    'imgpath': imagepath,
                    'beauty': round(res['beauty'], 2),
                    'detection': res['mtcnn']
                }
                result['elapse'] = round(time.time() - tik, 2)
            else:
                result['code'] = 3
                result['msg'] = 'None face is detected'
                result['data'] = None
                result['elapse'] = round(time.time() - tik, 2)

            json_str = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_str)
    else:
        result['code'] = 2
        result['msg'] = 'Invalid HTTP Method'
        result['data'] = None

        json_result = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_result)


def detect_face(detector, img_path):
    """
    detect face with MTCNN
    :param img_path:
    :return:
    """
    img = cv2.imread(img_path)
    if detector is None:
        detector = MTCNN()
    mtcnn_result = detector.detect_faces(img)

    return mtcnn_result


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


def upload_and_rec_porn(request):
    """
    upload and recognize porn image
    :param request:
    :return:
    """
    image_dir = 'cv/static/ImgCensorUpload'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    result = {}

    if request.method == "POST":
        image = request.FILES.get("image", None)
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
            imagepath = URL_PORT + '/static/ImgCensorUpload/' + image.name

            nswf_result = nsfw_estimator.infer(os.path.join(image_dir, image.name))

            result['code'] = 0
            result['msg'] = 'success'
            result['imgpath'] = imagepath
            if max([_['prob'] for _ in nswf_result['results']]) > PROB_THRESH:
                result['results'] = nswf_result['results']
            else:
                result['results'] = [{'type': 'Unknown', 'prob': nswf_result['results'][0]['prob']}]
            result['elapse'] = round(time.time() - tik, 2)

            json_str = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_str)
    else:
        result['code'] = 3
        result['msg'] = 'Invalid HTTP Method'
        result['data'] = None

        json_result = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_result)


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


def upload_and_rec_plant_disease(request):
    """
    upload and recognize plant disease
    :param request:
    :return:
    """
    image_dir = 'cv/static/PDRUpload'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    result = {}

    if request.method == "POST":
        image = request.FILES.get("image", None)
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
            imagepath = URL_PORT + '/static/PDRUpload/' + image.name

            plant_result = pdr.infer(os.path.join(image_dir, image.name))

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


def upload_and_ext_face_feats(request):
    """
    upload and extract face features
    :param request:
    :return:
    """
    image_dir = 'cv/static/FaceUpload'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    result = {}

    if request.method == "POST":
        image = request.FILES.get("image", None)
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
            imagepath = URL_PORT + '/static/FaceUpload/' + image.name

            face_feats_result = ext_feats(os.path.join(image_dir, image.name))

            result['code'] = 0
            result['msg'] = 'success'
            result['imgpath'] = imagepath
            result['results'] = face_feats_result['feature']
            result['elapse'] = round(time.time() - tik, 2)

            json_str = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_str)
    else:
        result['code'] = 3
        result['msg'] = 'Invalid HTTP Method'
        result['data'] = None

        json_result = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_result)


def upload_and_search_face(request):
    """
    upload and search face
    :param request:
    :return:
    """
    image_dir = 'cv/static/FaceUpload'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    result = {}

    if request.method == "POST":
        image = request.FILES.get("image", None)
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
            imagepath = URL_PORT + '/static/FaceUpload/' + image.name

            result['code'] = 0
            result['msg'] = 'success'
            result['imgpath'] = imagepath
            result['results'] = face_searcher.search(imagepath)
            result['elapse'] = round(time.time() - tik, 2)

            json_str = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_str)
    else:
        result['code'] = 3
        result['msg'] = 'Invalid HTTP Method'
        result['data'] = None

        json_result = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_result)
