import datetime
import json
import sys
import time
from collections import OrderedDict

import cv2
from django.http import HttpResponse
from django.shortcuts import render
from mtcnn.mtcnn import MTCNN

sys.path.append('../')
from cv.controllers.face_search_controller import upload_and_search_face
from cv.controllers.food_controller import upload_and_rec_food
from cv.controllers.nswf_controller import upload_and_rec_porn
from cv.controllers.plant_controller import upload_and_rec_plant
from cv.controllers.plant_disease_controller import upload_and_rec_plant_disease
from cv.controllers.fbp_controller import upload_and_rec_beauty
from cv.controllers.skin_disease_controller import upload_and_rec_skin_disease
from cv.cfg import cfg
from cv import db_utils


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


def fbp_view(request):
    return render(request, 'fbp.html')


def fbp(request):
    return upload_and_rec_beauty(request)


def food_view(request):
    return render(request, 'food.html')


def food(request):
    return upload_and_rec_food(request)


def plant_view(request):
    return render(request, 'plant.html')


def plant(request):
    return upload_and_rec_plant(request)


def skin_view(request):
    return render(request, 'skin.html')


def nsfw_view(request):
    return render(request, 'nsfw.html')


def nsfw(request):
    return upload_and_rec_porn(request)


def pdr_view(request):
    return render(request, 'pdr.html')


def pdr(request):
    return upload_and_rec_plant_disease(request)


def face_search_view(request):
    return render(request, 'facesearch.html')


def face_search(request):
    return upload_and_search_face(request)


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
    # skin_img_path = request.GET.get('skinImagePath')
    # tik = time.time()
    #
    # result = OrderedDict()
    # result['code'] = 0
    # result['msg'] = 'success'
    # result['results'] = [
    #     {'disease': 'Median_Nail_Dystrophy', 'probability': 0.96},
    #     {'disease': 'Acute_Eczema', 'probability': 0.01},
    #     {'disease': 'Keloid', 'probability': 0.01},
    #     {'disease': 'Lipoma', 'probability': 0.01},
    #     {'disease': 'Myxoid_Cyst', 'probability': 0.1},
    # ]
    # tok = time.time()
    # result['elapse'] = tok - tik
    #
    # json_result = json.dumps(result, ensure_ascii=False)
    #
    # return HttpResponse(json_result)

    skin_disease_result = upload_and_rec_skin_disease(request)
    skin_disease_result_json = json.loads(skin_disease_result.content.decode('utf-8'))

    print(skin_disease_result_json)

    if cfg['use_mysql'] and skin_disease_result_json['code'] == 0:
        conn = db_utils.connect_mysql_db()
        db_utils.insert_to_api(conn, 'LucasX', 'cv/mcloud/skin', skin_disease_result_json['elapse'],
                               datetime.time(), 0, skin_disease_result_json['imgpath'],
                               skin_disease_result_json['results'][0]['disease'])

    return skin_disease_result


def stat_skin(request):
    """
    skin API statistics
    :param request:
    :return:
    """
    username = request.GET.get('username')
    result = OrderedDict()
    tik = time.time()

    conn = db_utils.connect_mysql_db()

    result['code'] = 0
    result['msg'] = 'success'
    result['api'] = db_utils.query_api_hist(conn, username)
    tok = time.time()
    result['epalse'] = tok - tik

    json_result = json.dumps(result, ensure_ascii=False)

    return HttpResponse(json_result)
