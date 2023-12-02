# controllers for Image Deblurring
# Author: @LucasXU
# Mail: xulu0620@gmail.com

import base64
import json
import time

import requests
from django.http import HttpResponse

from cv.models.deblur_gan_v2 import *


def deblur(sr_model, img_f: str):
    """
    run Deblur-GAN v2 to infer an image
    @Author: LucasXU
    """

    def sorted_glob(pattern):
        return sorted(glob(pattern))

    mask = None
    f_img, f_mask = img_f, mask
    img, mask = map(cv2.imread, (f_img, f_mask))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pred = sr_model(img, mask)
    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    sr_img_path = img_f.replace(os.path.basename(img_f), "sr_" + os.path.basename(img_f))
    cv2.imwrite(sr_img_path, pred)

    return sr_img_path


def upload_and_deblur(request):
    """
    upload and do Image Deblurring
    :param request:
    :return:
    """
    tik = time.time()
    from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
    import io

    image_dir = 'cv/static/SRImgs'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    result = {}

    if request.method == "POST":
        image = request.FILES.get("image", None)
        if not isinstance(image, InMemoryUploadedFile) and not isinstance(image, TemporaryUploadedFile):
            imgstr = request.POST.get("image", None)
            if imgstr is None:
                result['code'] = 3
                result['msg'] = 'Do not upload image larger than 2.5MB'
                result['elapse'] = time.time() - tik
                result['data'] = None

                json_result = json.dumps(result, ensure_ascii=False)

                return HttpResponse(json_result)
            else:
                if 'http://' in imgstr or 'https://' in imgstr:
                    response = requests.get(imgstr)
                    image = InMemoryUploadedFile(io.BytesIO(response.content), name="{}.jpg".format(str(time.time())),
                                                 size=100, field_name="image", content_type="image/jpeg", charset=None)
                else:
                    image = InMemoryUploadedFile(io.BytesIO(base64.b64decode(imgstr)),
                                                 name="{}.jpg".format(str(time.time())), size=100, field_name="image",
                                                 content_type="image/jpeg", charset=None)
        if not image:
            result['code'] = 2
            result['msg'] = 'Invalid Image Path'
            result['elapse'] = time.time() - tik
            result['data'] = None

            json_result = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_result)
        else:
            destination = open(os.path.join(image_dir, image.name), 'wb+')
            for chunk in image.chunks():
                destination.write(chunk)
            destination.close()

            result['code'] = 0
            result['msg'] = 'success'
            result['elapse'] = time.time() - tik

            sr_img_path = deblur(deblur_gan_v2, os.path.join(image_dir, image.name))
            result['sr_img'] = sr_img_path

            json_str = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_str)
    else:
        result['code'] = 1
        result['msg'] = 'Invalid HTTP Method'
        result['elapse'] = time.time() - tik
        result['data'] = None

        json_result = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_result)
