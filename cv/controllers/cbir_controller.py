import base64
import io
import json
import os
import pickle
import time

import requests
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.http import HttpResponse
from skimage import io
from skimage.color import gray2rgb, rgba2rgb
from torchvision import models
from torchvision.transforms import transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_PORT = 'http://localhost:8000'
USE_GPU = True


def build_faiss_index(nd_feats_array, mode):
    """
    build index on multi GPUs
    :param nd_feats_array:
    :param mode: 0: CPU; 1: GPU; 2: Multi-GPU
    :return:
    """
    d = nd_feats_array.shape[1]

    cpu_index = faiss.IndexFlatL2(d)  # build the index on CPU
    if mode == 0:
        print("[INFO] Is trained? >> {}".format(cpu_index.is_trained))
        cpu_index.add(nd_feats_array)  # add vectors to the index
        print("[INFO] Capacity of gallery: {}".format(cpu_index.ntotal))

        return cpu_index
    elif mode == 1:
        ngpus = faiss.get_num_gpus()
        print("[INFO] number of GPUs:", ngpus)
        res = faiss.StandardGpuResources()  # use a single GPU
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(nd_feats_array)  # add vectors to the index
        print("[INFO] Capacity of gallery: {}".format(gpu_index.ntotal))

        return gpu_index
    elif mode == 2:
        multi_gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)  # build the index on multi GPUs
        multi_gpu_index.add(nd_feats_array)  # add vectors to the index
        print("[INFO] Capacity of gallery: {}".format(multi_gpu_index.ntotal))

        return multi_gpu_index


def search(index, query_feat, topK):
    """
    search TopK results
    :param index:
    :param topK:
    :return:
    """
    xq = query_feat.astype('float32')
    D, I = index.search(xq, topK)  # actual search
    # print(D[:5])  # neighbors of the 5 first queries
    # print(I[:5])  # neighbors of the 5 first queries

    print(I)
    print('-' * 100)
    print(D)

    return I.ravel()


def ext_deep_feat(model_with_weights, img_filepath):
    """
    extract deep feature from an image filepath
    :param model_with_weights:
    :param img_filepath:
    :return:
    """
    model_name = model_with_weights.__class__.__name__
    device = torch.device('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu')
    model_with_weights = model_with_weights.to(device)
    model_with_weights.eval()
    if model_name.startswith('DenseNet'):
        # print('extracting deep features of {}...'.format(model_name))

        with torch.no_grad():
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            image = io.imread(img_filepath)
            if len(list(image.shape)) < 3:
                image = gray2rgb(image)
            elif len(list(image.shape)) > 3:
                image = rgba2rgb(image)

            img = preprocess(Image.fromarray(image.astype(np.uint8)))
            img.unsqueeze_(0)
            img = img.to(device)

            inputs = img.to(device)

            feat = model_with_weights.features(inputs)
            feat = F.relu(feat, inplace=True)
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)

            # print('feat size = {}'.format(feat.shape))

    return feat.to('cpu').detach().numpy()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, np.float32):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)


class ImageSearcher:
    def __index__(self, index_filename_pkl='./filenames.pkl', feats_pkl='./feats.pkl'):
        densenet121 = models.densenet121(pretrained=True)
        densenet121.eval()

        print('[INFO] loading gallery features')
        with open(feats_pkl, mode='rb') as f:
            nd_feats_array = pickle.load(f).astype('float32')
        print(nd_feats_array.shape)
        print('[INFO] finish loading gallery\n[INFO] building index...')
        index = build_faiss_index(nd_feats_array, mode=0)
        print('[INFO] finish building index...')

        with open(index_filename_pkl, mode='rb') as f:
            idx_filename = pickle.load(f)

        self.index = index
        self.idx_filename = idx_filename
        self.model = densenet121

    def search(self, query_img, topK=10):
        """
        search TopK results:
        :param topK:
        :return:
        """
        query_feat = ext_deep_feat(self.model, query_img)
        xq = query_feat.astype('float32')
        D, I = self.index.search(xq, topK)  # actual search
        # print(D[:5])  # neighbors of the 5 first queries
        # print(I[:5])  # neighbors of the 5 first queries

        print(I)
        print('-' * 100)
        print(D)

        returned_indices = I.ravel()

        return [os.path.basename(self.idx_filename[idx]) for idx in returned_indices]


image_searcher = ImageSearcher()


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
            result['code'] = 1
            result['msg'] = 'Invalid Image'
            result['data'] = None

            json_result = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_result)
        else:
            destination = open(os.path.join(image_dir, image.name), 'wb+')
            for chunk in image.chunks():
                destination.write(chunk)
            destination.close()

            tik = time.time()
            imagepath = URL_PORT + '/static/CBIRUpload/' + image.name

            res = image_searcher.search(imagepath)

            if res is not None:
                result['code'] = 0
                result['msg'] = 'success'
                result['data'] = res
                result['elapse'] = round(time.time() - tik, 2)
            else:
                result['code'] = 3
                result['msg'] = 'None face is detected'
                result['data'] = None
                result['elapse'] = round(time.time() - tik, 2)

            json_str = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_str)
    else:
        result['code'] = 2
        result['msg'] = 'Invalid HTTP Method'
        result['data'] = None

        json_result = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_result)
