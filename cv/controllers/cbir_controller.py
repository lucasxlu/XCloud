import base64
import json
import os
import time

import cv2
import numpy as np
import requests
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from django.http import HttpResponse

from cv.models.cbir_seacher import image_searcher

URL_PORT = 'http://localhost:8001'


def upload_and_search(request):
    """
    upload and search image
    :param request:
    :return:
    """
    image_dir = 'cv/static/CBIRUpload'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    result = {}
    if request.method == "POST":
        image = request.FILES.get("image", None)
        if not image:
            result['code'] = 1
            result['msg'] = 'Invalid Image'
            result['results'] = None
            json_result = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_result)
        if not isinstance(image, InMemoryUploadedFile) and not isinstance(image, TemporaryUploadedFile):
            imgstr = request.POST.get("image", None)
            if 'http://' in imgstr or 'https://' in imgstr:
                response = requests.get(imgstr)
                image = np.asarray(bytearray(response.content), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            else:
                img_base64 = base64.b64decode(imgstr)
                image = np.frombuffer(img_base64, dtype=np.float64)
        else:
            destination = open(os.path.join(image_dir, image.name), 'wb+')
            for chunk in image.chunks():
                destination.write(chunk)
            destination.close()
            imagepath = URL_PORT + '/static/CBIRUpload/' + image.name
            image = 'cv/static/CBIRUpload/' + image.name

        tik = time.time()
        res = image_searcher.search(image)

        result['code'] = 0
        result['msg'] = 'Success'
        result['imgpath'] = imagepath
        result['data'] = res
        result['elapse'] = round(time.time() - tik, 2)

        json_str = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_str)
    else:
        result['code'] = 2
        result['msg'] = 'Invalid HTTP Method'
        result['data'] = None

        json_result = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_result)
