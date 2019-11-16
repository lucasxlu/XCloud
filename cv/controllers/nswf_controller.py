import base64
import json
import os
import time

import cv2
import numpy as np
import requests
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from django.http import HttpResponse

from cv.cfg import cfg
from cv.models.nsfw_recognizer import nsfw_estimator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_PORT = 'http://localhost:8001'


def upload_and_rec_porn(request):
    """
    upload and recognize porn image
    :param request:
    :return:
    """
    image_dir = 'cv/static/NSFWUpload'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    result = {}
    imagepath = ''
    if request.method == "POST":
        image = request.FILES.get("image", None)
        if not isinstance(image, InMemoryUploadedFile) and not isinstance(image, TemporaryUploadedFile):
            imgstr = request.POST.get("image", None)
            if imgstr is None or imgstr.strip() == '':
                result['code'] = 1
                result['msg'] = 'Invalid Image'
                result['data'] = None
                result['elapse'] = 0
                json_result = json.dumps(result, ensure_ascii=False)

                return HttpResponse(json_result)
            elif 'http://' in imgstr or 'https://' in imgstr:
                response = requests.get(imgstr)
                image = np.asarray(bytearray(response.content), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            else:
                img_base64 = base64.b64decode(imgstr)
                image = np.frombuffer(img_base64, dtype=np.float64)
        else:
            if image is None:
                result['code'] = 1
                result['msg'] = 'Invalid Image'
                result['data'] = None
                result['elapse'] = 0
                json_result = json.dumps(result, ensure_ascii=False)

                return HttpResponse(json_result)

            destination = open(os.path.join(image_dir, image.name), 'wb+')
            for chunk in image.chunks():
                destination.write(chunk)
            destination.close()

            imagepath = URL_PORT + '/static/NSFWUpload/' + image.name
            image = 'cv/static/NSFWUpload/' + image.name

        tik = time.time()
        nswf_result = nsfw_estimator.infer(image)

        result['code'] = 0
        result['msg'] = 'Success'
        result['imgpath'] = imagepath
        if max([_['prob'] for _ in nswf_result['results']]) > cfg['thresholds']['nsfw_recognition']:
            result['results'] = nswf_result['results']
        else:
            result['results'] = [{'type': 'Unknown', 'prob': nswf_result['results'][0]['prob']}]
        result['elapse'] = round(time.time() - tik, 2)

        json_str = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_str)
    else:
        result['code'] = 2
        result['msg'] = 'Invalid HTTP Method'
        result['data'] = None

        json_result = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_result)
