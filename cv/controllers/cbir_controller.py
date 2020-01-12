import base64
import json
import os
import time

import cv2
import numpy as np
import requests
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from django.http import HttpResponse

from cv.controllers.log import logger
from cv.models.cbir_seacher import image_searcher

URL_PORT = 'http://localhost:8001'


def upload_and_search(request):
    """
    upload and search image
    :param request:
    :return:
    """
    tik = time.time()
    image_dir = 'cv/static/CBIRUpload'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    result = {'data': {}}
    image_url_record = ''
    network_latency = 0

    if request.method == "POST":
        img_f = request.FILES.get("image", None)
        image = img_f if img_f is not None else request.POST.get("image", None)

        if image is None:
            result['code'] = 1
            result['msg'] = 'Invalid Image'
            result['elapse'] = 0
            json_result = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_result)
        else:
            network_latency_tik = time.time()
            if not isinstance(image, InMemoryUploadedFile) and not isinstance(image, TemporaryUploadedFile):
                # imgstr = request.POST.get("image", None)
                imgstr = image
                if 'http://' in imgstr or 'https://' in imgstr:
                    image_url_record = imgstr
                    # in case of bad image URL
                    try:
                        response = requests.get(image_url_record)
                        image = np.asarray(bytearray(response.content), dtype="uint8")
                        network_latency_tok = time.time()
                        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                        data = image_searcher.search(image)
                        result['data'] = data
                        result['code'] = 0
                        result['msg'] = 'success'
                    except:
                        result['code'] = 1
                        result['msg'] = 'Invalid Image'
                        result['data'] = {}
                        result['elapse'] = 0
                else:  # base64
                    # in case of broken base64 image
                    try:
                        img_base64 = base64.b64decode(imgstr)
                        image = np.frombuffer(img_base64, dtype=np.float64)
                        network_latency_tok = time.time()
                        data = image_searcher.search(image)
                        result['data'] = data
                        result['code'] = 0
                        result['msg'] = 'success'
                    except:
                        result['code'] = 1
                        result['msg'] = 'Invalid Image'
                        result['elapse'] = 0
            else:  # used in browser
                destination = open(os.path.join(image_dir, image.name), 'wb+')
                for chunk in image.chunks():
                    destination.write(chunk)
                destination.close()
                network_latency_tok = time.time()
                data = image_searcher.search(os.path.join(image_dir, image.name))
                result['data'] = data
                result['code'] = 0
                result['msg'] = 'success'

            tok = time.time()
            result['elapse'] = tok - tik
            network_latency = network_latency_tok - network_latency_tik
            logger.debug("NetworkLatency {}".format(network_latency))
            json_str = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_str)
    else:
        result['code'] = 2
        result['msg'] = 'Invalid HTTP Method'
        result['data'] = None
        result['elapse'] = 0

        json_result = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_result)
