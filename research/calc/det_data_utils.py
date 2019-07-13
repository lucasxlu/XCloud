"""
data proc util prepared for mm-detection
"""
import os
import sys
import random
import shutil

import cv2

from lxml import etree
import Augmentor

sys.path.append('../')


def get_obj_class_as_dict(predefined_classes_txt):
    """
    get object as dict
    :param predefined_classes_txt:
    :return:
    """
    obj_dict = {}
    with open(predefined_classes_txt, mode='rt', encoding='utf-8') as f:
        for i, cls in enumerate(f.readlines()):
            obj_dict[i] = cls.strip()

    return obj_dict


def cvt_yolo2voc_format(labelme_yolo_path, save_to_dir, obj_dict, img_dir):
    """
    convert YOLO format to Pascal VOC format
    :param labelme_yolo_path:
    :param save_to_dir:
    :param obj_dict: return of function get_obj_class_as_dict()
    :param img_dir:
    :return:
    """
    if labelme_yolo_path.endswith('.txt'):
        print("processing %s" % labelme_yolo_path)
        img_h, img_w, img_c = cv2.imread(
            os.path.join(img_dir, labelme_yolo_path.split('/')[-1].replace('.txt', '.jpg'))).shape

        annotation = etree.Element("annotation")
        etree.SubElement(annotation, "filename").text = labelme_yolo_path.split('/')[-1].replace('.txt', '.jpg')
        size = etree.SubElement(annotation, "size")
        etree.SubElement(size, "width").text = str(img_w)
        etree.SubElement(size, "height").text = str(img_h)
        etree.SubElement(size, "depth").text = str(img_c)

        with open(labelme_yolo_path, mode='rt', encoding='utf-8') as f:
            for _ in f.readlines():
                if _.strip() is not "":
                    cls_name = obj_dict[int(_.split(" ")[0].strip())]
                    bbox_w = float(_.split(" ")[3].strip()) * img_w
                    bbox_h = float(_.split(" ")[4].strip()) * img_h
                    xmin = int(float(_.split(' ')[1]) * img_w - bbox_w / 2)
                    xmax = int(float(_.split(' ')[1]) * img_w + bbox_w / 2)
                    ymin = int(float(_.split(' ')[2]) * img_h - bbox_h / 2)
                    ymax = int(float(_.split(' ')[2]) * img_h + bbox_h / 2)

                    object = etree.SubElement(annotation, "object")
                    etree.SubElement(object, "name").text = cls_name
                    bndbox = etree.SubElement(object, "bndbox")
                    etree.SubElement(bndbox, "xmin").text = str(xmin)
                    etree.SubElement(bndbox, "ymin").text = str(ymin)
                    etree.SubElement(bndbox, "xmax").text = str(xmax)
                    etree.SubElement(bndbox, "ymax").text = str(ymax)

            tree = etree.ElementTree(annotation)
            tree.write('{0}/{1}'.format(save_to_dir, labelme_yolo_path.split('/')[-1].replace('.txt', '.xml')),
                       pretty_print=True,
                       xml_declaration=True, encoding="utf-8")
            print('write %s successfully.' % labelme_yolo_path.split('/')[-1].replace('.txt', '.xml'))


def split_train_val_test_detection_data(xml_dir):
    """
    prepare train/val/test dataset for detection
    :param xml_dir:
    :return:
    """
    filenames = [_.replace('.xml', '') for _ in os.listdir(xml_dir)]
    random.shuffle(filenames)

    TEST_RATIO = 0.2

    train = filenames[0:int(len(filenames) * (1 - TEST_RATIO))]
    test = filenames[int(len(filenames) * (1 - TEST_RATIO)) + 1:]

    val = train[0:int(len(train) * 0.1)]
    train = train[int(len(train) * 0.1) + 1:]

    with open('./train.txt', mode='wt', encoding='utf-8') as f:
        f.writelines('\n'.join(train))

    with open('./val.txt', mode='wt', encoding='utf-8') as f:
        f.writelines('\n'.join(val))

    with open('./test.txt', mode='wt', encoding='utf-8') as f:
        f.writelines('\n'.join(test))


def data_augment(path_to_images_dir, data_aug_samples):
    """
    data augmentation
    :param path_to_images_dir:
    :param data_aug_samples:
    :return:
    """
    p = Augmentor.Pipeline(path_to_images_dir)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=1, min_factor=1.1, max_factor=1.5)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.rotate90(0.5)
    p.rotate270(0.5)
    p.sample(data_aug_samples)


def batch_augment_all_subdirs_in_basedir(base_dir):
    """
    batch augment all subdirs in basedir
    Note: the directory path tree should like: base_dir/d_1, base_dir/d_2, base_dir/d_3, ...
    :param base_dir:
    :return:
    """
    for d in os.listdir(base_dir):
        idx = 0
        print('process {0}...'.format(d))
        data_augment(os.path.join(base_dir, d))
        if os.path.isdir(os.path.join(base_dir, d)):
            for aug_img in os.listdir(os.path.join(base_dir, d, 'output')):
                os.rename(os.path.join(base_dir, d, 'output', aug_img),
                          os.path.join(base_dir, d, 'output', "Aug{0}_{1}".format(idx, aug_img)))
                shutil.copy(os.path.join(base_dir, d, "output", "Aug{0}_{1}".format(idx, aug_img)),
                            os.path.join(base_dir, d))
                idx += 1

            shutil.rmtree(os.path.join(base_dir, d, 'output'))
