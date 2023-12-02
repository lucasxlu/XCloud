"""
math toolkit
@author: Lucas Xu
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.multiclass import unique_labels


def cal_sensitivity(gts, preds):
    """
    calculate sensitivity from the given gts and preds
    Note: the rows are predicted values, and columns are groundtruth values in sklearn
    :param gts
    :param preds
    :return:
    """
    sensitivity = recall_score(gts, preds, average='macro')

    return sensitivity


def cal_specificity(gts, preds):
    """
    calculate specificity from the given gts and preds
    Note: the rows are predicted values, and columns are groundtruth values in sklearn
    :param gts
    :param preds
    :return:
    """
    cm = confusion_matrix(gts, preds)

    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)

    # Specificity or true negative rate
    specificity_list = tn / (tn + fp)

    return sum(specificity_list) / len(specificity_list)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          show_text=False,
                          title=None,
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)

    # classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[1]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if show_text:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

    return ax


if __name__ == '__main__':
    df = pd.read_csv("C:/Users/LucasXU/Desktop/DenseNet.csv")
    gts = df['gt']
    preds = df['pred']

    print('Top1 Acc = {}%'.format(accuracy_score(gts, preds) * 100))
    print('Precision = %f ' % precision_score(gts, preds, average='macro'))
    print('Recall = %f ' % recall_score(gts, preds, average='macro'))
    print('F1 = %f ' % f1_score(gts, preds, average='macro'))
    print('Sens = %f' % cal_sensitivity(gts, preds))
    print('Spec = %f' % cal_specificity(gts, preds))

    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plot_confusion_matrix(gts, preds, classes=[_ for _ in range(40)], normalize=True,
                          title='Normalized confusion matrix')
    plt.show()
