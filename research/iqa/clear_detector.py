"""
image clearness detection
"""
import math
import os
import time

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


class ClearDetector:
    """
    image clearness detector, a bigger value represents clearer image
    """

    def __init__(self, clear_value_threshold=0.8, show=False):
        self.clear_value_threshold = clear_value_threshold
        self.show = show

    def preprocess_img(self, img, resize_width, resize_height):
        """
        resize and convert to Gray scale
        :param img:
        :param resize_width:
        :param resize_height:
        :return:
        """
        x, y, z = img.shape
        new_pic = img
        if ((x >= resize_height) & (y >= resize_width) | (x < resize_height) & (y < resize_width)):
            new_pic = img
        elif (x < resize_height) & (y >= resize_width):
            new_pic = img[:, int(y / 2 - resize_width / 2): int(y / 2 + resize_width / 2), :]
        elif (x >= resize_height) & (y < resize_width):
            new_pic = img[int(x / 2 - resize_height / 2): int(x / 2 + resize_height / 2), :]
        elif (x >= resize_height) & (y < resize_width):
            new_pic = img[int(x / 2 - resize_height / 2): int(x / 2 + resize_height / 2), :]

        new_picture = cv2.resize(new_pic, (resize_height, resize_width))
        if len(new_picture.shape) == 3:
            new_picture = cv2.cvtColor(new_picture, cv2.COLOR_BGR2GRAY)

        return new_picture

    def cal_img_ssim(self, gray_img, block_stride=11):
        """
        calculate SSIM of the Gray scale image with block size of block_stride
        :param gray_img:
        :param block_stride:
        :return:
        """
        if len(gray_img) == 3:
            gray_img = self.preprocess_img(gray_img)
        x, y = gray_img.shape
        res_list = []
        for i in range(0, int(x), int(x / block_stride) + 3):
            for j in range(0, int(y), int(y / block_stride) + 3):
                res_list.append(self.cal_ssim(gray_img[i:i + block_stride, j:j + block_stride]))

        res_list = np.array(res_list)
        res_list_sort = res_list[np.lexsort(-res_list.T)]
        res_list = res_list_sort[:, :1]
        res = np.mean(res_list[:10])

        return res if res < 0 else 1 - res

    def cal_entropy(self, img):
        """
        calculate entropy value
        :param img:
        :return:
        """
        res = 0
        tmp = [0] * 256
        img_list = []
        for i in range(len(img)):
            img_list.extend(map(int, img[i]))
        img_list_set = set(img_list)
        for i in img_list_set:
            tmp[i] = float(img_list.count(i)) / 256

        for i in range(len(tmp)):
            if tmp[i] != 0:
                res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))

        return res

    def cal_ssim(self, img):
        """
        calculate SSIM
        :param img:
        :return:
        """
        x, y = img.shape
        res_entropy = self.cal_entropy(img)
        tr = cv2.GaussianBlur(img, (5, 5), 3)
        g = cv2.Sobel(img, cv2.CV_16S, 2, 2) / 5
        gr = cv2.Sobel(tr, cv2.CV_16S, 2, 2) / 5
        ux = np.mean(g)
        uy = np.mean(gr)
        vx = np.var(g)
        vy = np.var(gr)
        vxy = (1 / (x * y - 1)) * np.sum((g - ux) * (gr - uy))
        r, k1, k2 = 255, 0.03, 0.01
        c1 = (k1 * r) ** 2
        c2 = (k2 * r) ** 2

        a1 = 2 * ux * uy + c1
        a2 = 2 * vxy + c2
        b1 = ux ** 2 + uy ** 2 + c1
        b2 = vx + vy + c2
        ssim = (a1 * a2) / (b1 * b2)

        return ssim, res_entropy


if __name__ == '__main__':
    tik = time.time()
    preds = []
    gts = []

    clear_detector = ClearDetector()

    img_dir = "./blur"
    for _ in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, _))
        img = clear_detector.preprocess_img(img, 200, 300)
        res = clear_detector.cal_img_ssim(img)
        print(res)
        if res < clear_detector.clear_value_threshold:
            preds.append(1)
        else:
            preds.append(0)

        gts.append(1)

    img_dir = "./normal"
    for _ in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, _))
        img = clear_detector.preprocess_img(img, 200, 300)
        res = clear_detector.cal_img_ssim(img)
        print(res)
        if res >= clear_detector.clear_value_threshold:
            preds.append(0)
        else:
            preds.append(1)

        gts.append(0)

    tok = time.time()
    print('FPS={}'.format(len(os.listdir(img_dir)) / (tok - tik)))

    print(confusion_matrix(gts, preds))
    print('Precision = %f' % precision_score(gts, preds))
    print('Recall = %f' % recall_score(gts, preds))
    print('Accuracy = %f' % accuracy_score(gts, preds))
