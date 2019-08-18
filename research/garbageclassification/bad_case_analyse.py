import os
import shutil
import pandas as pd


def select_bad_cases(output_csv):
    """
    select bad cases
    :param output_csv:
    :return:
    """
    df = pd.read_csv(output_csv)
    gts = df['gt'].tolist()
    preds = df['pred'].tolist()
    filenames = df['filename'].tolist()
    probs = df['prob'].tolist()

    if not os.path.exists('./BadCases'):
        os.makedirs('./BadCases')

    for i, filename in enumerate(filenames):
        if str(gts[i]) != str(preds[i]):
            shutil.copy(filename, './BadCases/{0}_{1}_{2}_{3}'.format(gts[i], preds[i], probs[i], filename.split('/')[-1]))

    print('done!')


if __name__ == '__main__':
    select_bad_cases('./DenseNet.csv')
