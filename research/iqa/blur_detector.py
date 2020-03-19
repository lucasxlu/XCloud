# a class wrapper for Blur Detection
import cv2
from sklearn.metrics.classification import accuracy_score, confusion_matrix, recall_score, precision_score


class BlurDetector:
    def __init__(self, var_threshold=400, show=False):
        self.var_threshold = var_threshold
        self.show = show

    def cal_variance_of_laplacian(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)

        if self.show:
            cv2.imshow('img', image)
            cv2.waitKey()
            cv2.destroyAllWindows()

        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def judge_blur_or_not(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        var_laplacian = self.cal_variance_of_laplacian(image)
        if self.show:
            cv2.putText(image, str(round(var_laplacian, 2)), (0, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
            cv2.imshow('img', image)
            cv2.waitKey()
            cv2.destroyAllWindows()

        return {'var': var_laplacian, 'blur': var_laplacian < self.var_threshold}


if __name__ == '__main__':
    blur_detector = BlurDetector()
    import os
    import time

    tik = time.time()
    preds = []
    gts = []

    img_dir = 'C:/Users/Administrator/Desktop/Normal'
    for _ in os.listdir(img_dir):
        res = blur_detector.judge_blur_or_not(os.path.join(img_dir, _))
        print(res)
        if res['desc'] == 'Not Blurry':
            preds.append(0)
        else:
            preds.append(1)

        gts.append(0)

    img_dir = 'C:/Users/Administrator/Desktop/Blur'
    for _ in os.listdir(img_dir):
        res = blur_detector.judge_blur_or_not(os.path.join(img_dir, _))
        print(res)
        if res['desc'] == 'Not Blurry':
            preds.append(0)
        else:
            preds.append(1)

        gts.append(1)

    tok = time.time()
    print('FPS={}'.format(len(os.listdir(img_dir)) / (tok - tik)))

    print(confusion_matrix(gts, preds))
    print('Precision = %f' % precision_score(gts, preds))
    print('Recall = %f' % recall_score(gts, preds))
    print('Accuracy = %f' % accuracy_score(gts, preds))
