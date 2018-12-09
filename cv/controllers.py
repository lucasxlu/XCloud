import json
import os
import time

import numpy as np
import cv2
from django.http import HttpResponse
from mtcnn.mtcnn import MTCNN
from sklearn.externals import joblib

from cv import features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

URL_PORT = 'http://localhost:8000'


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

            beauty = BeautyRecognizer().infer(os.path.join(image_dir, image.name))

            result['code'] = 0
            result['msg'] = 'success'
            result['data'] = {
                'imgpath': imagepath,
                'beauty': round(beauty, 2)
            }
            result['elapse'] = round(time.time() - tik, 2)

            json_str = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_str)
    else:
        result['code'] = 2
        result['msg'] = 'Invalid HTTP Method'
        result['data'] = None

        json_result = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_result)


def detect_face(img_path):
    """
    detect face with MTCNN
    :param img_path:
    :return:
    """
    img = cv2.imread(img_path)
    detector = MTCNN()
    mtcnn_result = detector.detect_faces(img)

    return mtcnn_result


class BeautyRecognizer:
    def __init__(self, pretrained_model='./model/GradientBoostingRegressor.pkl'):
        gbr = joblib.load(pretrained_model)
        self.model = gbr

    def infer(self, img_path):
        img = cv2.imread(img_path)
        mtcnn_result = detect_face(img_path)
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
