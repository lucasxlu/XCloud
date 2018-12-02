import json
from random import randint
from collections import OrderedDict

from django.http import HttpResponse
from django.shortcuts import render


# Create your views here.


def welcome(request):
    """
    welcome page for computer vision welcome
    :param request:
    :return:
    """
    return render(request, 'welcome.html')


def fbp(request):
    """
    calculate facial beauty score
    :param request:
    faceImagePath: face image path
    :return:
    """
    face_img_path = request.GET.get('faceImagePath')
    result = OrderedDict()

    if not face_img_path or face_img_path.strip() == '':
        result['code'] = 1
        result['msg'] = 'Invalid Path for Face Image'
        result['data'] = None
    else:
        result['code'] = 0
        result['msg'] = 'success'
        result['data'] = {
            'beauty': randint(0, 9)
        }

    json_result = json.dumps(result, ensure_ascii=False)

    return HttpResponse(json_result)
