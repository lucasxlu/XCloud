# Blur Creator, used to create blurry images, for Image Quality Analysis/Super Resolution/Image Deblurring
# author: @LucasX

import os
import cv2
import random

HR_DIR = "./DeblurDataset/train/HR"
LR_DIR = "./DeblurDataset/train/LR"


def blur_img(img_f, blur_type):
    """
    blur image with given blur type
    :param img_f:
    :param blur_type: Avg, Gaussian, Median
    :return:
    """
    if not os.path.exists(LR_DIR):
        os.makedirs(LR_DIR)
    img = cv2.imread(img_f)

    if blur_type == 'Avg':
        kernerl_size = random.randint(5, 10)
        blurry_img = cv2.blur(img, (kernerl_size, kernerl_size))  # Avg Blur
        cv2.imwrite(os.path.join(LR_DIR, img_f.split(os.path.sep)[-1]), blurry_img)
        print('[INFO] blurring {} using kernel [{}, {}]'.format(img_f, kernerl_size, kernerl_size))
    elif blur_type == 'Gaussian':
        kernerl_size = 5
        blurry_img = cv2.GaussianBlur(img, (kernerl_size, kernerl_size), 20)  # Gaussian Blur
        cv2.imwrite(os.path.join(LR_DIR, img_f.split(os.path.sep)[-1]), blurry_img)
        print('[INFO] blurring {} using kernel [{}, {}]'.format(img_f, kernerl_size, kernerl_size))
    elif blur_type == 'Median':
        kernerl_size = 5
        blurry_img = cv2.medianBlur(img, kernerl_size)
        cv2.imwrite(os.path.join(LR_DIR, img_f.split(os.path.sep)[-1]), blurry_img)
        print('[INFO] blurring {} using kernel [{}, {}]'.format(img_f, kernerl_size, kernerl_size))
    elif blur_type == 'Bilateral':
        blurry_img = cv2.bilateralFilter(img, 9, 75, 75)
        cv2.imwrite(os.path.join(LR_DIR, img_f.split(os.path.sep)[-1]), blurry_img)
        print('[INFO] blurring {} using Bilateral Filter'.format(img_f))
    else:
        print('[ERROR] Invalid BLUR_TYPE! It can only in [Avg]/[Gaussian]/[Median]/[Bilateral]')

    # cv2.imshow('blurry_img', blurry_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':

    for img_f in os.listdir(HR_DIR):
        blur_img(os.path.join(HR_DIR, img_f), blur_type='Gaussian')
