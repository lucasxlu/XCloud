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
