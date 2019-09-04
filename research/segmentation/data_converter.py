import base64
import json
import os

from lxml import etree

IMG_DIR = '/path/to/JPEGImages'


def cvt_jingling_assist_xml_to_labelme_json(pascal_voc_xml_dir, labelme_json_dir):
    """
    convert JingLing Assist XML format to labelme json format
    :param pascal_voc_xml_dir:
    :param labelme_json_dir:
    :return:
    """
    if not os.path.exists(labelme_json_dir):
        os.makedirs(labelme_json_dir)

    for pascal_voc_xml in os.listdir(pascal_voc_xml_dir):
        if pascal_voc_xml.endswith('.xml'):
            print('processing {}'.format(os.path.join(pascal_voc_xml_dir, pascal_voc_xml)))
            tree = etree.parse(os.path.join(pascal_voc_xml_dir, pascal_voc_xml))
            if len(tree.xpath('//item')) == 0:
                continue
            json_info = {"version": "3.16.6", "flags": {}, "shapes": [], "imageHeight": 1296, "imageWidth": 972,
                         "lineColor": [
                             0,
                             255,
                             0,
                             128
                         ], "fillColor": [
                    255,
                    0,
                    0,
                    128
                ], 'imagePath': tree.xpath('//path/text()')[0]}
            with open(os.path.join(IMG_DIR, json_info['imagePath'].split(os.path.sep)[-1]), "rb") as fid:
                data = fid.read()
            b64_bytes = base64.b64encode(data)
            b64_string = b64_bytes.decode()
            json_info['imageData'] = b64_string

            if len(tree.xpath('//item')) == 0:
                continue

            for item in tree.xpath('//item'):
                shape = {
                    "label": item.xpath('name/text()')[0].lower().strip(),
                    "line_color": None,
                    "fill_color": None,
                    "points": [],
                    "shape_type": "polygon",
                    "flags": {}
                }
                polygons = item.xpath('polygon')
                for polygon in polygons:
                    for i in range(int(len(polygon) / 2)):
                        shape['points'].append([max(float(polygon.xpath('x{0}/text()'.format(i + 1))[0]), 0),
                                                max(float(polygon.xpath('y{0}/text()'.format(i + 1))[0]), 0)])

                json_info['shapes'].append(shape)

        with open(os.path.join(labelme_json_dir, '{}.json'.format(pascal_voc_xml.split('.xml')[0])), mode='wt') as f:
            json.dump(json_info, f, indent=2, sort_keys=True)

    print('finish convert annotation files.')


def cvt_pascal_voc_xml_to_labelme_json(pascal_voc_xml_dir, labelme_json_dir):
    """
    convert Pascal VOC XML format to labelme json format
    :param pascal_voc_xml_dir:
    :param labelme_json_dir:
    :return:
    """
    if not os.path.exists(labelme_json_dir):
        os.makedirs(labelme_json_dir)

    for pascal_voc_xml in os.listdir(pascal_voc_xml_dir):
        if pascal_voc_xml.endswith('.xml'):
            print('processing {}'.format(os.path.join(pascal_voc_xml_dir, pascal_voc_xml)))
            tree = etree.parse(os.path.join(pascal_voc_xml_dir, pascal_voc_xml))
            if len(tree.xpath('//size')) == 0:
                os.remove(os.path.join(pascal_voc_xml_dir, pascal_voc_xml))
                continue

            json_info = {
                "version": "3.16.6",
                "flags": {},
                "imagePath": tree.xpath('//path/text()')[0],
                "shapes": [],
                "imageHeight": int(tree.xpath('//height/text()')[0]),
                "imageWidth": int(tree.xpath('//width/text()')[0]),
                "lineColor": [
                    0,
                    255,
                    0,
                    128
                ],
                "fillColor": [
                    255,
                    0,
                    0,
                    128
                ],
            }

            with open(os.path.join(IMG_DIR, json_info['imagePath'].split(os.path.sep)[-1]), "rb") as fid:
                data = fid.read()
            b64_bytes = base64.b64encode(data)
            b64_string = b64_bytes.decode()
            json_info['imageData'] = b64_string

            if len(tree.xpath('//object')) == 0:
                continue

            for object in tree.xpath('//object'):
                shape = {
                    "label": object.xpath('name/text()')[0].lower().strip(),
                    "line_color": None,
                    "fill_color": None,
                    "points": [],
                    "shape_type": "polygon",
                    "flags": {}
                }
                polygons = object.xpath('polygon')
                for polygon in polygons:
                    for i in range(int(len(polygon) / 2)):
                        shape['points'].append([max(float(polygon.xpath('x{0}/text()'.format(i + 1))[0]), 0),
                                                max(float(polygon.xpath('y{0}/text()'.format(i + 1))[0]), 0)])

                json_info['shapes'].append(shape)

        with open(os.path.join(labelme_json_dir, '{}.json'.format(pascal_voc_xml.split('.xml')[0])), mode='wt',
                  encoding='utf-8') as f:
            json.dump(json_info, f, indent=2)

    print('finish convert annotation files.')


def get_classes_from_mask_anno(json_anno_dir):
    """
    get all categories from mask annotation json files
    :param json_anno_dir:
    :return:
    """
    classes = []
    for js in os.listdir(json_anno_dir):
        if js.endswith('.json'):
            with open(os.path.join(json_anno_dir, js), mode='rt') as f:
                json_info = json.load(f)
                classes += [shape['label'] for shape in json_info['shapes']]

    print(set(classes))


def rectify_label(json_anno_dir):
    """
    rectify wrong labels
    :param json_anno_dir:
    :return:
    """
    for js in os.listdir(json_anno_dir):
        if js.endswith('.json'):
            with open(os.path.join(json_anno_dir, js), mode='rt') as f:
                json_info = json.load(f)
                json_txt = json.dumps(json_info)
                json_txt = json_txt.replace('banceng', 'CengBan').replace('cengban', 'CengBan').replace('guawang',
                                                                                                        'GuaWang').replace(
                    'huojia', 'HuoJia').replace('Cengban', 'CengBan')

            json_info = json.loads(json_txt)
            with open(os.path.join(json_anno_dir, js), mode='wt') as f:
                json.dump(json_info, f, indent=2)
