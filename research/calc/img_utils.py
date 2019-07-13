"""
image utils
"""
import os
import sys

import cv2
from lxml import etree
from skimage import io

sys.path.append('../')
from research.calc.file_utils import mkdir_if_not_exist


def clean_bad_imgs(root):
    """
    clean bad images in root directory
    :param root:
    :return:
    """
    for d in os.listdir(root):
        if os.path.isdir(os.path.join(root, d)):
            clean_bad_imgs(os.path.join(root, d))
        else:
            filename = os.path.join(root, d)
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                try:
                    image = io.imread(filename)
                except:
                    os.remove(filename)
                    print('remove {0}'.format(filename))

    print('done!')


def crop_subregion_from_img(label_me_xml_path, img_dir, save_to_base_dir):
    """
    crop image patches from VOC format
    :param label_me_xml_path:
    :param img_dir:
    :param save_to_base_dir:
    :return:
    """
    mkdir_if_not_exist(save_to_base_dir)
    if label_me_xml_path.endswith('.xml'):
        print('processing %s...' % label_me_xml_path)
        tree = etree.parse(label_me_xml_path)
        objects = tree.xpath('//object')
        label_me_xml_name = label_me_xml_path.split(os.path.sep)[-1]

        img_path = os.path.join(img_dir, label_me_xml_name[0: -4] + '.jpg')
        img = cv2.imread(img_path)

        for i, object in enumerate(objects):
            sub_region = img[int(object.xpath('bndbox/ymin/text()')[0]): int(object.xpath('bndbox/ymax/text()')[0]),
                         int(object.xpath('bndbox/xmin/text()')[0]): int(object.xpath('bndbox/xmax/text()')[0])]

            print('current bbox is %s.' % object.xpath('name/text()')[0])
            bbox_obj_class = object.xpath('name/text()')[0]

            mkdir_if_not_exist(os.path.join(save_to_base_dir, bbox_obj_class))
            cv2.imwrite(os.path.join(save_to_base_dir, bbox_obj_class,
                                     "{0}_{1}_{2}.jpg".format(label_me_xml_name[0: -4], bbox_obj_class, i)), sub_region)

    print('done!')
