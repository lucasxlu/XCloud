import base64
import io
import json
import os
import time

import cv2
import numpy as np
import requests
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from django.http import HttpResponse

from cv.models.plant_recognizer import plant_recognizer

URL_PORT = 'http://localhost:8001'


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

            imagepath = URL_PORT + '/static/PlantUpload/' + image.name
            image = 'cv/static/PlantUpload/' + image.name

        tik = time.time()
        plant_result = plant_recognizer.infer(image)

        result['code'] = 0
        result['msg'] = 'Success'
        result['imgpath'] = imagepath
        result['results'] = plant_result['results']
        result['elapse'] = round(time.time() - tik, 2)

        json_str = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_str)
    else:
        result['code'] = 2
        result['msg'] = 'Invalid HTTP Method'
        result['data'] = None

        json_result = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_result)
