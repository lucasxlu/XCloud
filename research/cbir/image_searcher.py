"""
build index and search TopK results with faiss
"""
import pickle
import sys
import os
import shutil

import numpy as np
import faiss
from torchvision import models

sys.path.append('../../')
from research.cbir.ext_feats import ext_deep_feat


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


if __name__ == '__main__':
    print('[INFO] loading gallery features')
    with open('./feats.pkl', mode='rb') as f:
        nd_feats_array = pickle.load(f).astype('float32')
    print(nd_feats_array.shape)
    print('[INFO] finish loading gallery\n[INFO] building index...')
    index = build_faiss_index(nd_feats_array, mode=0)
    print('[INFO] finish building index...')

    with open('./filenames.pkl', mode='rb') as f:
        idx_filename = pickle.load(f)

    densenet121 = models.densenet121(pretrained=True)
    feat = ext_deep_feat(densenet121, './query.jpg')
    returned_indices = search(index, feat, topK=50)
    print(returned_indices)

    if not os.path.exists('./retrievalimg'):
        os.makedirs('./retrievalimg')

    print('[INFO] the retrieved images are:')
    for idx in returned_indices:
        print(idx, idx_filename[idx])
        shutil.copy(idx_filename[idx], os.path.join('./retrievalimg', idx_filename[idx].split('/')[-1]))
