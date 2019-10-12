"""
a porn image detector based on machine learning and skin color model
"""
import os

import numpy as np
import skimage.color
import skimage.filters
from PIL import Image
from skimage import data
from skimage import io, transform
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

BASE_DIR = 'E:/DataSet/CV/TrainAndTestPornImages'


def get_skin_area(image_filepath):
    image = io.imread(image_filepath)
    hsv_img = skimage.color.rgb2hsv(image)
    gray_img = skimage.color.rgb2gray(image)

    print(hsv_img[:, :, 0])

    thresh = skimage.filters.threshold_otsu(gray_img)
    dst = (gray_img <= thresh) * 1.0
    io.imsave('./image.jpg', dst)


def get_skin_ratio(im):
    """
    get skin area ratio of an image
    :param im:
    :return:
    :version: 1.0
    """
    im = im.crop((int(im.size[0] * 0.2), int(im.size[1] * 0.2), im.size[0] - int(im.size[0] * 0.2),
                  im.size[1] - int(im.size[1] * 0.2)))
    skin = sum(
        [count for count, rgb in im.getcolors(im.size[0] * im.size[1]) if rgb[0] > 60 and rgb[1] < (rgb[0] * 0.85)
         and rgb[2] < (rgb[0] * 0.7) and rgb[1] > (rgb[0] * 0.4) and rgb[2] > (rgb[0] * 0.2)])

    return float(skin) / float(im.size[0] * im.size[1])


def HOG(img_path):
    """
    extract HOG feature
    :param img_path:
    :return:
    :version: 1.0
    """
    img = io.imread(img_path)
    img = skimage.color.rgb2gray(img)
    feature = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))

    return feature


def LBP(img_path):
    # settings for LBP
    radius = 3
    n_points = 8 * radius
    METHOD = 'uniform'
    image = data.load(img_path)
    gray_image = skimage.color.rgb2gray(image)
    lbp = local_binary_pattern(gray_image, n_points, radius, METHOD)

    return lbp


def batch_rename(directory):
    """
    batch rename the files in a directory from 1 to files' number
    :param directory:
    :return:
    :version:1.0
    """
    index = 1
    for _ in os.listdir(directory):
        os.rename(os.path.join(directory, _), os.path.join(directory, str(index) + '.' + _.split(".")[-1]))
        index += 1
    print('all have been renamed!')


def remove_abnomal_images(directory):
    """
    remove abnormal images
    :param directory:
    :return:
    :version: 1.0
    """
    for _ in os.listdir(directory):
        try:
            feature = HOG(os.path.join(directory, _))
            # print(len(feature))
            print(_)
        except:
            os.remove(directory + '/' + _)


def detect_image(image_filepath):
    """
    detect an image if it is a porn, sexy or normal image
    :param image_filepath:
    :return:
    :version: 1.0
    """
    skin_percent = get_skin_ratio(Image.open(image_filepath)) * 100
    if skin_percent > 40:
        print("PORN {0} has {1:.0f}% skin".format(image_filepath, skin_percent))
        flag = True
    else:
        # print("CLEAN {0} has {1:.0f}% skin".format(image_filepath, skin_percent))
        flag = False

    return flag


def extract_features(directory):
    features = []
    for _ in os.listdir(directory):
        img = skimage.color.rgb2gray(io.imread(os.path.join(directory, _)))
        out_size = (64, 64)  # height, width
        resized_img = transform.resize(img, out_size, order=1, mode='reflect')
        feature = hog(resized_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        features.append(feature)

    return np.array(features)


def start_train():
    if os.path.exists('./model/porn_det_ml.pkl'):
        os.remove('./model/porn_det_ml.pkl')

    print('starting training machine learning model...')

    train_feature_normal = extract_features(os.path.join(BASE_DIR, 'train', '0'))
    train_feature_sexy = extract_features(os.path.join(BASE_DIR, 'train', '1'))
    train_feature_porn = extract_features(os.path.join(BASE_DIR, 'train', '2'))

    test_feature_normal = extract_features(os.path.join(BASE_DIR, 'test', '0'))
    test_feature_sexy = extract_features(os.path.join(BASE_DIR, 'test', '1'))
    test_feature_porn = extract_features(os.path.join(BASE_DIR, 'test', '2'))

    train_y_normal = [0 for i in range(len(train_feature_normal))]
    train_y_sexy = [1 for i in range(len(train_feature_sexy))]
    train_y_porn = [2 for i in range(len(train_feature_porn))]

    test_y_normal = [0 for i in range(len(test_feature_normal))]
    test_y_sexy = [1 for i in range(len(test_feature_sexy))]
    test_y_porn = [2 for i in range(len(test_feature_porn))]

    # svc = svm.SVC()
    # svc.fit(np.concatenate((train_feature_normal, train_feature_sexy, train_feature_porn), axis=0),
    #         np.concatenate((train_y_normal, train_y_sexy, train_y_porn)))
    #
    # accuracy_svc = svc.score(np.concatenate((test_feature_normal, test_feature_sexy, test_feature_porn), axis=0),
    #                          np.concatenate((test_y_normal, test_y_sexy, test_y_porn)))
    #
    # print('=================The accuracy of SVM is {:.5%}================='.format(accuracy_svc))
    #
    # dtree = tree.DecisionTreeClassifier()
    # dtree.fit(np.concatenate((train_feature_normal, train_feature_sexy, train_feature_porn), axis=0),
    #           np.concatenate((train_y_normal, train_y_sexy, train_y_porn)))
    #
    # accuracy_dtree = dtree.score(np.concatenate((test_feature_normal, test_feature_sexy, test_feature_porn), axis=0),
    #                              np.concatenate((test_y_normal, test_y_sexy, test_y_porn)))
    # print('=============The accuracy of Dtree is {:.5%}================='.format(accuracy_dtree))

    # mlp = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(300, 200, 150, 150, 100), random_state=1)
    # mlp.fit(np.concatenate((train_feature_normal, train_feature_sexy, train_feature_porn), axis=0),
    #         np.concatenate((train_y_normal, train_y_sexy, train_y_porn)))
    # accuracy_mlp = mlp.score(np.concatenate((test_feature_normal, test_feature_sexy, test_feature_porn), axis=0),
    #                          np.concatenate((test_y_normal, test_y_sexy, test_y_porn)))
    # print('=============The accuracy of MLP is {:.5%}==================='.format(accuracy_mlp))

    lr = LogisticRegression()
    lr.fit(np.concatenate((train_feature_normal, train_feature_sexy, train_feature_porn), axis=0),
           np.concatenate((train_y_normal, train_y_sexy, train_y_porn), axis=0))
    accurracy_lr = lr.score(np.concatenate((test_feature_normal, test_feature_sexy, test_feature_porn), axis=0),
                            np.concatenate((test_y_normal, test_y_sexy, test_y_porn), axis=0))
    print('===============The accuracy of LR is {:.5%}===================='.format(accurracy_lr))

    rf = RandomForestClassifier(n_estimators=20)
    rf.fit(np.concatenate((train_feature_normal, train_feature_sexy, train_feature_porn), axis=0),
           np.concatenate((train_y_normal, train_y_sexy, train_y_porn)))

    print('finishing training machine learning model...')

    accuracy_rf = rf.score(np.concatenate((test_feature_normal, test_feature_sexy, test_feature_porn), axis=0),
                           np.concatenate((test_y_normal, test_y_sexy, test_y_porn)))

    joblib.dump(rf, './model/porn_det_ml.pkl')

    print('===============The accuracy of RF is {:.5%}===================='.format(accuracy_rf))


def predict(image_filepath):
    model_path = BASE_DIR + '/migrations/porn_detector.pkl'
    model = joblib.load(model_path)
    img = skimage.color.rgb2gray(io.imread(image_filepath))
    out_size = (128, 128)  # height, width
    resized_img = transform.resize(img, out_size, order=1, mode='reflect')
    feature = hog(resized_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))

    val = model.predict(feature)

    if val == 0:
        return '正常'
    elif val == 1:
        return '性感'
    elif val == 2:
        return "色情"


if __name__ == '__main__':
    # batch_rename("E:/DataSet/CV/TrainAndTestPornImages/PornImages/2")
    start_train()
    # count = 0
    # for image in os.listdir(directory):
    #     for image_file in glob.glob(os.path.join(directory, image)):
    #         try:
    #             flag = detect_image(os.path.join(directory, image))
    #             if flag is True:
    #                 count += 1
    #         except:
    #             pass
    #
    # print('=============The algorithm detects %d in total================' % count)
