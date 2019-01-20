import os
import shutil
import json

import pandas as pd


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


if __name__ == '__main__':
    cvt_submission_format('./ResNet.csv')
