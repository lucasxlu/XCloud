"""
Python3 implementation of SamplePairing (offline version)
Inoue, Hiroshi. "Data augmentation by pairing samples for images classification." arXiv preprint arXiv:1801.02929 (2018).
Note: your image file structure should be:
- base_img_root
    - category 1
    - category 2
    - category 3
    - ...
    - category N
after running this code, you will get sample-paired images as:
- SamplePairingImg
    - category 1
    - category 2
    - category 3
    - ...
    - category N
@Author: LucasX
"""
import os
import random

import cv2

DATA_AUG_DIR = './SamplePairingImg'
NOT_JOIN_SP_CAT = ['Naturally-Blurred']


def do_sample_pairing(base_img_root, sample_number_of_each_category=50):
    """
    perform SamplePairing
    :param base_img_root:
    :param sample_number_of_each_category:
    :return:
    """
    assert os.path.isdir(base_img_root)
    visited_label_img_set = set()

    if not os.path.exists(DATA_AUG_DIR):
        os.makedirs(DATA_AUG_DIR)

    cat_list = [_ for _ in os.listdir(base_img_root) if os.path.isdir(os.path.join(base_img_root, _))]
    for cat in cat_list:
        if cat not in NOT_JOIN_SP_CAT:
            number = 0
            fg_imgs = [os.path.join(base_img_root, cat, _) for _ in os.listdir(os.path.join(base_img_root, cat))]
            print('[INFO] generate sample-paired images for {}'.format(cat))
            if not os.path.exists(os.path.join(DATA_AUG_DIR, cat)):
                os.makedirs(os.path.join(DATA_AUG_DIR, cat))
            bg_dirs = list(set(cat_list) - set(cat))
            bg_imgs = []
            for bg_dir in bg_dirs:
                for _ in os.listdir(os.path.join(base_img_root, bg_dir)):
                    bg_imgs.append(os.path.join(base_img_root, bg_dir, _))

            while number < sample_number_of_each_category and len(bg_imgs) > 0:
                random_fg = fg_imgs[random.randint(0, len(fg_imgs) - 1)]
                random_bg = bg_imgs.pop(random.randint(0, len(bg_imgs) - 1))

                if random_fg not in visited_label_img_set and random_bg not in visited_label_img_set:
                    mixed_img = mix_img(random_fg, random_bg)
                    cv2.imwrite(os.path.join(DATA_AUG_DIR, cat, 'SamplePairing_{}'.format(os.path.basename(random_fg))),
                                mixed_img)
                    visited_label_img_set.add(random_fg)
                    visited_label_img_set.add(random_bg)

                    number += 1

    print('[INFO] processing done!')


def mix_img(img_a, img_b):
    """
    mix two images to form a new image with the same size as the smaller one
    :param img_a:
    :param img_b:
    :return:
    """
    if isinstance(img_a, str):
        img_a = cv2.imread(img_a)
    if isinstance(img_b, str):
        img_b = cv2.imread(img_b)

    (w, h, c) = img_a.shape if img_a.shape[0] * img_a.shape[1] < img_b.shape[0] * img_b.shape[1] else img_b.shape
    img_a = cv2.resize(img_a, (w, h))
    img_b = cv2.resize(img_b, (w, h))

    mixed_img = 0.5 * img_a + 0.5 * img_b

    return mixed_img


if __name__ == '__main__':
    do_sample_pairing("D:/Datasets/CERTH_ImageBlurDataset/TrainingSet", 300)
