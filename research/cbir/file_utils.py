"""
file io utils
"""

import os

import cv2
from lxml import etree


def mkdir_if_not_exist(dir_name):
    """
    make directory if not exist
    :param dir_name:
    :return:
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def crop_subregion_from_img(label_me_xml_path, img_dir, save_to_base_dir):
    """
    crop sub-region patch from image
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


if __name__ == '__main__':
    anno_dir = "/path/to/Annotations"
    img_dir = "/path/to/JPEGImages"
    crop_save_to_dir = "/path/to/Crops"

    for xml in os.listdir(anno_dir):
        xml_path = os.path.join(anno_dir, xml)
        try:
            crop_subregion_from_img(xml_path, img_dir, crop_save_to_dir)
        except:
            pass

    print('done!')
