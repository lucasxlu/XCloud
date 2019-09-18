# a class wrapper for Blur Detection
import cv2


class BlurDetector:
    def __init__(self, var_threshold=100):
        self.var_threshold = var_threshold
        self.var_laplacian = 0.0

    def cal_variance_of_laplacian(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def judge_blur_or_not(self, image):
        var_laplacian = self.cal_variance_of_laplacian(image)
        if var_laplacian > self.var_threshold:
            return {'var': var_laplacian, 'desc': 'Not Blurry'}
        else:
            return {'var': var_laplacian, 'desc': 'Blurry'}
