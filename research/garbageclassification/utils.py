import json
import os
import shutil
import sys

import pandas as pd
from skimage import io
from skimage.color import gray2rgb, rgba2rgb

sys.path.append('../../')
from research.garbageclassification.cfg import cfg


def mkdirs_if_not_exist(dir_name):
    """
    create new folder if not exist
    :param dir_name:
    :return:
    """
    if not os.path.isdir(dir_name) or not os.path.exists(dir_name):
        os.makedirs(dir_name)


def over_sample(from_target_dir, to_target_dir, copy_time):
    """
    over sample
    :param from_target_dir:
    :param to_target_dir:
    :param copy_time:
    :return:
    """
    mkdirs_if_not_exist(to_target_dir)

    for i in range(copy_time):
        print('Copying Round %d...' % i)
        for _ in os.listdir(from_target_dir):
            shutil.copyfile(os.path.join(from_target_dir, _), os.path.join(to_target_dir, 'cp_{0}_{1}'.format(i, _)))

    print('Processing done!')


def cvt_submission_format(csv_path):
    """
    convert CSV file to JSON submission format
    :param csv_path:
    :return:
    """
    df = pd.read_csv(csv_path)
    submission = []
    filenames = df['filename']
    pred = df['pred']

    for i in range(len(df)):
        submission.append({
            "image_id": filenames[i].split('/')[-1],
            "disease_class": pred[i]
        })

    with open('./submission.json', mode='wt', encoding='utf-8') as f:
        json.dump(str(submission), f)

    print('Converting JSON successfully~~~')


def remove_mal_images():
    types = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

    for tp in types:
        print('process {0} ...'.format(tp))
        for img in os.listdir(os.path.join(cfg['root'], tp, 'IMAGES')):
            filename = os.path.join(cfg['root'], tp, 'IMAGES', img)
            print(filename)
            if '.jpg?' in filename:
                dst = filename.split('?')[0]
                shutil.move(filename, dst)
                filename = dst

            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                try:
                    image = io.imread(filename)
                    if len(list(image.shape)) < 3:
                        image = gray2rgb(filename)
                    elif len(list(image.shape)) > 3:
                        image = rgba2rgb(image)
                    io.imsave(filename, image)
                except:
                    os.remove(filename)
            else:
                os.remove(filename)


if __name__ == '__main__':
    remove_mal_images()
