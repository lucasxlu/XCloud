import pickle
import os
import json

import numpy as np

from torchvision import models

from cv.controllers.cbir_controller import build_faiss_index, ext_deep_feat


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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, np.float32):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)
