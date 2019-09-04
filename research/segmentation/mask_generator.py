import json
import os

import cv2
import numpy as np

CLASS_COLOR = {
    'Background': [0, 0, 0],
    'CengBan': [128, 0, 0],
    'HuoJia': [0, 128, 0],
    'GuaWang': [0, 0, 128],
}


def generate_mask(img_filepath, json_labelme_file):
    """
    generate GT mask for semantic segmentation
    :param img_filepath:
    :param json_labelme_file:
    :return:
    """
    gt_mask_dir = './Masks'
    if not os.path.exists(gt_mask_dir):
        os.makedirs(gt_mask_dir)

    with open(json_labelme_file, mode='rt') as f:
        json_info = json.load(f)
    img = cv2.imread(img_filepath, cv2.CV_8S)
    mask = np.zeros_like(img)

    for shape in json_info['shapes']:
        if shape['label'] == 'HuoJia':
            pts = np.array(shape['points']).astype(np.int32)
            cv2.drawContours(mask, [pts], -1, CLASS_COLOR[shape['label']], -1, cv2.LINE_AA)
            cv2.drawContours(mask, [pts], -1, [255, 255, 255], 3, cv2.LINE_AA)

    for shape in json_info['shapes']:
        if shape['label'] == 'GuaWang':
            pts = np.array(shape['points']).astype(np.int32)
            cv2.drawContours(mask, [pts], -1, CLASS_COLOR[shape['label']], -1, cv2.LINE_AA)
            cv2.drawContours(mask, [pts], -1, [255, 255, 255], 3, cv2.LINE_AA)

    for shape in json_info['shapes']:
        if shape['label'] == 'CengBan':
            pts = np.array(shape['points']).astype(np.int32)
            cv2.drawContours(mask, [pts], -1, CLASS_COLOR[shape['label']], -1, cv2.LINE_AA)
            cv2.drawContours(mask, [pts], -1, [255, 255, 255], 3, cv2.LINE_AA)

    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)
    mask = mask.astype("uint8")

    p = img_filepath.split(os.path.sep)[-1].replace('.jpg', '.png')
    cv2.imwrite(os.path.join(gt_mask_dir, '{}'.format(p)),
                mask)
    print('[INFO] finish generating mask {0} for {1}'.format(p, img_filepath.split(os.path.sep)[-1]))
