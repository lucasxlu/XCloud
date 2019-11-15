"""
train and test FBP with traditional ML, instead of DL
"""
import os
import sys

import pandas as pd
from mtcnn.mtcnn import MTCNN
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import cv2

sys.path.append('../')
from research.fbp.features import *

SCUT5500_DIR = "E:\DataSet\Face\SCUT-FBP5500\Images"
SCUT5500_CROP_DIR = "E:\DataSet\Face\SCUT-FBP5500\Crop"
LABEL_CSV = "E:\DataSet\Face\SCUT-FBP5500/train_test_files\SCUT-FBP5500-With-Head.csv"


def prepare_data():
    features = []
    lbs = []

    df = pd.read_csv(LABEL_CSV)
    files = df['file']
    scores = df['score']
    detector = MTCNN()

    for i in range(len(files)):
        img = cv2.imread(os.path.join(SCUT5500_DIR, files[i]))
        mtcnn_result = detector.detect_faces(img)

        if len(mtcnn_result) > 0:
            bbox = mtcnn_result[0]['box']
            print(bbox)

            margin_pixel = 10
            face_region = img[bbox[0] - margin_pixel: bbox[0] + bbox[2] + margin_pixel,
                          bbox[1] - margin_pixel: bbox[1] + bbox[3] + margin_pixel]

            # cv2.imshow('face', face_region)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            ratio = max(face_region.shape[0], face_region.shape[1]) / min(face_region.shape[0], face_region.shape[1])
            if face_region.shape[0] > face_region.shape[1]:
                face_region = cv2.resize(face_region,
                                         (int((face_region.shape[0] / ratio) * 64 / face_region.shape[1]), 64))
            else:
                face_region = cv2.resize(face_region,
                                         (64, int((face_region.shape[1] / ratio) * 64 / face_region.shape[0])))
        else:
            face_region = cv2.resize(img, (64, 64))

        lbp = LBP_from_cv(face_region)
        # hog = HOG_from_cv(face_region)
        # ldmk = Geo_from_cv(img)

        feature = lbp
        features.append(feature)
        lbs.append(scores[i])

    return features, lbs


def prepare_data_from_crop():
    """
    prepare face datasets from cropped facial regions
    :return:
    """
    features = []
    lbs = []

    df = pd.read_csv(LABEL_CSV)
    files = df['file']
    scores = df['score']

    for i in range(len(files)):
        img = cv2.imread(os.path.join(SCUT5500_CROP_DIR, files[i]))

        face_region = cv2.resize(img, (64, 64))

        hog = HOG_from_cv(face_region)
        # raw = RAW_from_cv(face_region)
        # lbp = LBP_from_cv(face_region)
        # ldmk = Geo_from_cv(img)

        feature = list(hog)
        features.append(feature)
        lbs.append(scores[i])

    return features, lbs


def train_fbp(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    gbr = GradientBoostingRegressor()
    print('start training regressor...')
    gbr.fit(X_train, y_train)

    if not os.path.exists('./model'):
        os.makedirs('./model')

    joblib.dump(gbr, './model/GradientBoostingRegressor.pkl')
    print('The regression model has been persisted...')

    y_pred = gbr.predict(X_test)

    mae_lr = round(mean_absolute_error(y_test, np.array(y_pred).ravel()), 4)
    rmse_lr = round(np.math.sqrt(mean_squared_error(np.array(y_test), np.array(y_pred).ravel())), 4)
    pc = round(np.corrcoef(np.array(y_test), np.array(y_pred).ravel())[0, 1], 4)

    print('===============The Mean Absolute Error of ANet is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of ANet is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of ANet is {0}===================='.format(pc))


if __name__ == '__main__':
    X, y = prepare_data_from_crop()
    train_fbp(X, y)
