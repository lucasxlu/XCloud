import json
import os
import time
from random import randint

import cv2
from django.http import HttpResponse
from mtcnn.mtcnn import MTCNN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

URL_PORT = 'http://localhost:8000'


def upload_and_rec(request):
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

            result['code'] = 0
            result['msg'] = 'success'
            result['data'] = {
                'imgpath': imagepath,
                'beauty': randint(0, 9)
            }
            result['elapse'] = time.time() - tik

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
    def __init__(self, pretrained_model):
        self.model = None

    def infer(self, img_path):
        img = cv2.imread(img_path)
        mtcnn_result = detect_face(img_path)
        bbox = mtcnn_result['box']
        face_region = img[bbox[0] - int(bbox[2] / 2): bbox[0] + int(bbox[2] / 2),
                      bbox[1] - int(bbox[3] / 2): bbox[1] + int(bbox[3] / 2)]

        return self.model.infer(face_region)
