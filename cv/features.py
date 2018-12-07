"""
feature extractor
"""
import cv2
import dlib
import numpy as np
import skimage.color
from skimage import io
from skimage.feature import hog, local_binary_pattern, corner_harris

DLIB_MODEL = "E:\ModelZoo\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(DLIB_MODEL)
detector = dlib.get_frontal_face_detector()


def HOG(img_path):
    """
    extract HOG feature
    :param img_path:
    :return:
    :version: 1.0
    """
    img = io.imread(img_path)
    img = skimage.color.rgb2gray(img)
    img = (img - np.mean(img)) / np.std(img)
    feature = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys')

    return feature


def LBP(img_path):
    """
    extract LBP features
    :param img_path:
    :return:
    """
    img = io.imread(img_path)
    img = skimage.color.rgb2gray(img)
    img = (img - np.mean(img)) / np.std(img)
    feature = local_binary_pattern(img, P=8, R=0.2)
    # im = Image.fromarray(np.uint8(feature))
    # im.show()

    return feature.reshape(feature.shape[0] * feature.shape[1])


def LBP_from_cv(img):
    """
    extract LBP features from opencv region
    :param img:
    :return:
    """
    img = skimage.color.rgb2gray(img)
    img = (img - np.mean(img)) / np.std(img)
    feature = local_binary_pattern(img, P=8, R=0.2)
    # im = Image.fromarray(np.uint8(feature))
    # im.show()

    return feature.reshape(feature.shape[0] * feature.shape[1])


def HARRIS(img_path):
    """
    extract HARR features
    :param img_path:
    :return:
    :Version:1.0
    """
    img = io.imread(img_path)
    img = skimage.color.rgb2gray(img)
    img = (img - np.mean(img)) / np.std(img)
    feature = corner_harris(img, method='k', k=0.05, eps=1e-06, sigma=1)

    return feature.reshape(feature.shape[0] * feature.shape[1])


def RAW(img_path):
    img = io.imread(img_path)
    img = skimage.color.rgb2gray(img)
    img = (img - np.mean(img)) / np.std(img)

    return img.reshape(img.shape[0] * img.shape[1])


def HOG_from_cv(img):
    """
    extract HOG feature from opencv image object
    :param img:
    :return:
    :Version:1.0
    """
    img = skimage.color.rgb2gray(img)
    img = (img - np.mean(img)) / np.std(img)

    return hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys')


def Geo_from_cv(img):
    """
    68 facial landmarks as geometry feature
    :param img:
    :return:
    """
    faces = detector(img, 1)

    result = {}
    if len(faces) > 0:
        for k, d in enumerate(faces):
            shape = predictor(img, d)
            result[k] = {"bbox": [d.left(), d.top(), d.right(), d.bottom()],
                         "landmarks": [[shape.part(i).x, shape.part(i).y] for i in range(68)]}

    xs = np.array([_[0] for _ in result[0]['landmarks']])
    ys = np.array([_[1] for _ in result[0]['landmarks']])

    return list(xs - np.mean(xs)) + list(ys - np.mean(ys))
