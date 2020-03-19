# generate blurry samples from clear images
# author: @LucasX
import argparse
import os
import random
import numpy as np

import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-hd_dir', type=str,
                    default='/Users/lucasx/Documents/Dataset/normal')
parser.add_argument('-ld_dir', type=str,
                    default='/Users/lucasx/Documents/Dataset/blur')
parser.add_argument('-blur_type', type=str, default='Motion')
parser.add_argument('-show', type=bool, default=False)
args = vars(parser.parse_args())

print('-' * 100)
for key, value in args.items():
    print('%s = %s' % (key, value))
print('-' * 100)


def blur_img(img_f):
    """
    blur image with given blur type
    :param img_f:
    :param blur_type: Avg, Gaussian, Median
    :return:
    """
    if img_f.endswith('.jpg') or img_f.endswith('.png') or img_f.endswith('.jpeg'):
        if not os.path.exists(args['ld_dir']):
            os.makedirs(args['ld_dir'])
        img = cv2.imread(img_f)
        blurry_img = None

        if args['blur_type'] == 'Avg':
            kernerl_size = random.randint(5, 10)
            blurry_img = cv2.blur(img, (kernerl_size, kernerl_size))  # Avg Blur
            cv2.imwrite(os.path.join(args['ld_dir'], '{}_{}'.format(args['blur_type'], img_f.split(os.path.sep)[-1])),
                        blurry_img)
            print('[INFO] blurring {} using kernel [{}, {}]'.format(img_f, kernerl_size, kernerl_size))
        elif args['blur_type'] == 'Gaussian':
            kernerl_size = 5
            blurry_img = cv2.GaussianBlur(img, (kernerl_size, kernerl_size), 20)  # Gaussian Blur
            cv2.imwrite(os.path.join(args['ld_dir'], '{}_{}'.format(args['blur_type'], img_f.split(os.path.sep)[-1])),
                        blurry_img)
            print('[INFO] blurring {} using kernel [{}, {}]'.format(img_f, kernerl_size, kernerl_size))
        elif args['blur_type'] == 'Median':
            kernerl_size = 5
            blurry_img = cv2.medianBlur(img, kernerl_size)
            cv2.imwrite(os.path.join(args['ld_dir'], '{}_{}'.format(args['blur_type'], img_f.split(os.path.sep)[-1])),
                        blurry_img)
            print('[INFO] blurring {} using kernel [{}, {}]'.format(img_f, kernerl_size, kernerl_size))
        elif args['blur_type'] == 'Bilateral':
            blurry_img = cv2.bilateralFilter(img, 9, 75, 75)
            cv2.imwrite(os.path.join(args['ld_dir'], '{}_{}'.format(args['blur_type'], img_f.split(os.path.sep)[-1])),
                        blurry_img)
            print('[INFO] blurring {} using Bilateral Filter'.format(img_f))
        elif args['blur_type'] == 'Motion':
            # Specify the kernel size.
            # The greater the size, the more the motion.
            kernel_size = random.randint(10, 30)
            # Create the vertical kernel.
            kernel_v = np.zeros((kernel_size, kernel_size))
            # Create a copy of the same for creating the horizontal kernel.
            kernel_h = np.copy(kernel_v)
            # Fill the middle row with ones.
            kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
            kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

            # Normalize.
            kernel_v /= kernel_size
            kernel_h /= kernel_size

            # Apply the vertical kernel.
            vertical_mb = cv2.filter2D(img, -1, kernel_v)
            # Apply the horizontal kernel.
            horizonal_mb = cv2.filter2D(img, -1, kernel_h)

            blurry_img = horizonal_mb
            cv2.imwrite(os.path.join(args['ld_dir'], '{}_{}'.format(args['blur_type'], img_f.split(os.path.sep)[-1])),
                        blurry_img)
            print('[INFO] blurring {} using Motion Filter with kernel [{}, {}]'.format(img_f, kernel_size, kernel_size))
        else:
            print('[ERROR] Invalid BLUR_TYPE! It can only in [Avg]/[Gaussian]/[Median]/[Bilateral]')

        if args['show'] and blurry_img is not None:
            cv2.imshow('blurry_img', blurry_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    for img_f in os.listdir(args['hd_dir']):
        blur_img(os.path.join(args['hd_dir'], img_f))
