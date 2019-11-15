import base64
import io
import json
import os
import pickle
import time

import numpy as np
import requests
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.http import HttpResponse
from numpy.linalg import norm

from utils.feat_extractor import ext_feats
from cv.models.net_sphere import SphereFaceNet

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_PORT = 'http://localhost:8000'


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, np.float32):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)


class FaceSearcher:
    """
    Face Search Class Wrapper
    """

    def __init__(self, face_feats_path="cv/model/hzau_master_face_features.pkl"):
        with open(face_feats_path, mode='rb') as f:
            face_feats_list = pickle.load(f)

        self.topK = 10
        self.face_feats_list = face_feats_list
        self.sphere_face = SphereFaceNet(feature=True)

    def search(self, img_file):
        face_feat = ext_feats(sphere_face=SphereFaceNet(feature=True), img_path=img_file)

        compare_result = {}
        for face_obj in self.face_feats_list:
            norm_face_feature = face_feat['feature'] / np.linalg.norm(face_feat['feature'])
            norm_face_obj_feature = face_obj['feature'] / np.linalg.norm(face_obj['feature'])

            cos_sim = np.dot(norm_face_feature, norm_face_obj_feature) / \
                      (norm(norm_face_feature) * norm(norm_face_obj_feature))
            compare_result[face_obj['studentid']] = cos_sim

        sorted_compare_result = sorted(compare_result.items(), key=lambda kv: kv[1], reverse=True)

        return {
            'status': 0,
            'message': 'success',
            'results': sorted_compare_result[0: self.topK]
        }


face_searcher = FaceSearcher()


def upload_and_ext_face_feats(request):
    """
    upload and extract face features
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
            result['code'] = 3
            result['msg'] = 'Invalid Path for Image'
            result['results'] = None

            json_result = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_result)
        else:
            destination = open(os.path.join(image_dir, image.name), 'wb+')
            for chunk in image.chunks():
                destination.write(chunk)
            destination.close()

            tik = time.time()
            imagepath = URL_PORT + '/static/FaceUpload/' + image.name

            face_feats_result = ext_feats(os.path.join(image_dir, image.name))

            result['code'] = 0
            result['msg'] = 'success'
            result['imgpath'] = imagepath
            result['results'] = face_feats_result['feature']
            result['elapse'] = round(time.time() - tik, 2)

            json_str = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_str)
    else:
        result['code'] = 3
        result['msg'] = 'Invalid HTTP Method'
        result['data'] = None

        json_result = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_result)


def upload_and_search_face(request):
    """
    upload and search face
    :param request:
    :return:
    """
    image_dir = 'cv/static/FaceUpload'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    result = {}

    if request.method == "POST":
        image = request.FILES.get("image", None)
        if not isinstance(image, InMemoryUploadedFile):
            imgstr = request.POST.get("image", None)
            if 'http' in imgstr:
                response = requests.get(imgstr)
                image = InMemoryUploadedFile(io.BytesIO(response.content), name="{}.jpg".format(str(time.time())),
                                             field_name="image", content_type="image/jpeg", size=1347415, charset=None)
            else:
                image = InMemoryUploadedFile(io.BytesIO(base64.b64decode(imgstr)),
                                             name="{}.jpg".format(str(time.time())), field_name="image",
                                             content_type="image/jpeg", size=1347415, charset=None)

        if not image:
            result['code'] = 3
            result['msg'] = 'Invalid Path for Image'
            result['results'] = None

            json_result = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_result)
        else:
            destination = open(os.path.join(image_dir, image.name), 'wb+')
            for chunk in image.chunks():
                destination.write(chunk)
            destination.close()

            tik = time.time()
            imagepath = URL_PORT + '/static/FaceUpload/' + image.name

            print(imagepath)

            result['code'] = 0
            result['msg'] = 'success'
            result['imgpath'] = imagepath
            result['results'] = face_searcher.search(os.path.join(image_dir, image.name))['results']
            result['elapse'] = round(time.time() - tik, 2)

            json_str = json.dumps(result, ensure_ascii=False, cls=NumpyEncoder)

            return HttpResponse(json_str)
    else:
        result['code'] = 3
        result['msg'] = 'Invalid HTTP Method'
        result['data'] = None

        json_result = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_result)
