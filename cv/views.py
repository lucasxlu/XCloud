import time
import json
from collections import OrderedDict

import cv2
from django.http import HttpResponse
from django.shortcuts import render
from mtcnn.mtcnn import MTCNN


# Create your views here.


def welcome(request):
    """
    welcome page for computer vision welcome
    :param request:
    :return:
    """
    return render(request, 'welcome.html')


def index(request):
    return render(request, 'index.html')


def mcloud(request):
    return render(request, 'mcloud.html')


def fbp(request):
    from cv import controllers
    return controllers.upload_and_rec_beauty(request)


def fbp_view(request):
    return render(request, 'fbp.html')


def detect_face(request):
    """
    face detection
    @Note: currently supported by MTCNN
    :param request:
    :return:
    """
    face_img_path = request.GET.get('faceImagePath')
    result = OrderedDict()

    if not face_img_path or face_img_path.strip() == "":
        result['code'] = 1
        result['msg'] = 'Invalid Path for Face Image'
        result['data'] = None
    else:
        img = cv2.imread(face_img_path)
        detector = MTCNN()
        result['code'] = 0
        result['msg'] = 'success'
        result['data'] = detector.detect_faces(img)

    json_result = json.dumps(result, ensure_ascii=False)

    return HttpResponse(json_result)


def rec_skin(request):
    """
    recognize 198 skin disease
    :param request:
    :return:
    """
    face_img_path = request.GET.get('faceImagePath')
    tik = time.time()

    result = OrderedDict()
    result['code'] = 0
    result['msg'] = 'success'
    result['results'] = [
        {'disease': 'Median_Nail_Dystrophy', 'probability': 0.96},
        {'disease': 'Acute_Eczema', 'probability': 0.01},
        {'disease': 'Keloid', 'probability': 0.01},
        {'disease': 'Lipoma', 'probability': 0.01},
        {'disease': 'Myxoid_Cyst', 'probability': 0.1},
    ]
    tok = time.time()
    result['elapse'] = tok - tik

    json_result = json.dumps(result, ensure_ascii=False)

    return HttpResponse(json_result)


def stat_skin(request):
    """
    skin API statistics
    :param request:
    :return:
    """
    username = request.GET.get('username')
    result = OrderedDict()
    result['code'] = 0
    result['msg'] = 'success'
    result['api'] = {'name': 'cv/mcloud/skin', 'count': 10256}
    result['history'] = [
        {'disease': 'Median_Nail_Dystrophy', 'count': 1002},
        {'disease': 'Keloid', 'count': 526},
        {'disease': 'Lipoma', 'count': 295},
    ]

    json_result = json.dumps(result, ensure_ascii=False)

    return HttpResponse(json_result)
