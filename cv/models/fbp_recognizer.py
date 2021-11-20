import logging
import os

import cv2
import numpy as np
import torch
from PIL import Image
from mtcnn.mtcnn import MTCNN
from joblib import dump, load
from torchvision.transforms import transforms

from cv.cfg import cfg
from cv.models.shufflenet_v2 import ShuffleNetV2
from research.fbp import features

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def detect_face(detector, img):
    """
    detect face with MTCNN
    :param img_path:
    :return:
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    if detector is None:
        detector = MTCNN()
    mtcnn_result = detector.detect_faces(img)

    return mtcnn_result


class BeautyRecognizerML:
    """
    non-deep learning based facial beauty predictor
    """

    def __init__(self, pretrained_model=os.path.join(cfg['model_zoo_base'], 'GradientBoostingRegressor.pkl')):
        assert os.path.exists(pretrained_model)

        logger.info('Initiate BeautyRecognizerML')
        gbr = load(pretrained_model)
        self.model = gbr
        self.detector = MTCNN()

    def infer(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
        mtcnn_result = detect_face(self.detector, img)
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

    def __init__(self, pretrained_model=os.path.join(cfg['model_zoo_base'], 'ShuffleNetV2.pth')):
        assert os.path.exists(pretrained_model)

        logger.info('Initiate BeautyRecognizer (ShuffleNet V2 as backbone)')
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

    def infer(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
        mtcnn_result = detect_face(self.detector, img)

        if len(mtcnn_result) > 0:
            bbox = mtcnn_result[0]['box']

            margin_pixel = 10
            face_region = img[bbox[0] - margin_pixel: bbox[0] + bbox[2] + margin_pixel,
                          bbox[1] - margin_pixel: bbox[1] + bbox[3] + margin_pixel]

            cv2.rectangle(img, (bbox[0] - margin_pixel, bbox[1] - margin_pixel),
                          (bbox[0] + bbox[2] + margin_pixel, bbox[1] + bbox[3] + margin_pixel), (232, 171, 74), 2)
            # cv2.imwrite(img_path, img)

            ratio = max(face_region.shape[0], face_region.shape[1]) / min(face_region.shape[0], face_region.shape[1])
            if face_region.shape[0] < face_region.shape[1]:
                face_region = cv2.resize(face_region, (int(ratio * 64), 64))
                face_region = face_region[:,
                              int((face_region.shape[0] - 64) / 2): int((face_region.shape[0] - 64) / 2) + 64]
            else:
                face_region = cv2.resize(face_region, (64, int(ratio * 64)))
                face_region = face_region[int((face_region.shape[1] - 64) / 2): int((face_region.shape[1] - 64) / 2)
                                                                                + 64, :]

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
            return {"beauty": -1, "mtcnn": []}


beauty_recognizer = BeautyRecognizer()
