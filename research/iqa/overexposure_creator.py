# generate overexposure samples from clear images
# author: @LucasX
import argparse
import os
import random
from multiprocessing import Queue, Process

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-orig_dir', type=str,
                    default='C:/Users/LucasX/Desktop/ShelfExposure/normal')
parser.add_argument('-outpur_dir', type=str,
                    default='C:/Users/LucasX/Desktop/ShelfExposure/exposure')
parser.add_argument('-alpha', type=float, default=2.0)
parser.add_argument('-beta', type=float, default=0.0)
parser.add_argument('-procs', type=int, default=2)
parser.add_argument('-show', type=bool, default=False)
args = vars(parser.parse_args())

print('-' * 100)
for key, value in args.items():
    print('%s = %s' % (key, value))
print('-' * 100)


def modify_img_saturation(img_f):
    """
    modify image saturation to imitate overexposure effect
    :param img_f:
    :return:
    """
    if img_f.endswith('.jpg') or img_f.endswith('.png') or img_f.endswith('.jpeg'):
        if not os.path.exists(args['outpur_dir']):
            os.makedirs(args['outpur_dir'])
        image = cv2.imread(img_f)

        overexposure_image = np.zeros(image.shape, image.dtype)
        # alpha = args['alpha']
        alpha = random.randint(2, 10)

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):

                    overexposure_image[y, x, c] = np.clip(alpha * image[y, x, c] + args['beta'], 0, 255)

        if args['show'] and overexposure_image is not None:
            cv2.imshow('overexposure_image', overexposure_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.imwrite(os.path.join(args['outpur_dir'], os.path.basename(img_f)), overexposure_image)
        print('[INFO] generate overexposure image {} successfully'.format(os.path.basename(img_f)))


def multi_proc_modify_img_saturation(imgs_queue):
    """
    modify image saturation to imitate overexposure effect in multi-processing mode
    :param imgs_queue:
    :return:
    """
    if not imgs_queue.empty():
        img_f = imgs_queue.get()
        if img_f.endswith('.jpg') or img_f.endswith('.png') or img_f.endswith('.jpeg'):
            if not os.path.exists(args['outpur_dir']):
                os.makedirs(args['outpur_dir'])
            image = cv2.imread(img_f)

            overexposure_image = np.zeros(image.shape, image.dtype)
            # alpha = args['alpha']
            alpha = random.randint(2, 10)

            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    for c in range(image.shape[2]):

                        overexposure_image[y, x, c] = np.clip(alpha * image[y, x, c] + args['beta'], 0, 255)

            if args['show'] and overexposure_image is not None:
                cv2.imshow('overexposure_image', overexposure_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            cv2.imwrite(os.path.join(args['outpur_dir'], os.path.basename(img_f)), overexposure_image)
            print('[INFO] generate overexposure image {} successfully'.format(os.path.basename(img_f)))


if __name__ == '__main__':
    # multi-thread processing version
    imgs_queue = Queue()
    for img_f in os.listdir(args['orig_dir']):
        imgs_queue.put(os.path.join(args['orig_dir'], img_f))

    for i in range(args['procs']):
        Process(target=multi_proc_modify_img_saturation, args=(imgs_queue,)).start()

    # single-thread processing version
    # for img_f in os.listdir(args['orig_dir']):
    #     modify_img_saturation(os.path.join(args['orig_dir'], img_f))
