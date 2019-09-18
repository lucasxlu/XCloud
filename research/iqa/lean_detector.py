# a class wrapper for Lean Detection
import math

import cv2
import numpy as np


class LeanDetector:
    def __init__(self, lean_degree_threshold=5):
        # the hyper-param lean_degree_threshold should be tuned according to your application scene
        self.lean_degree_threshold = lean_degree_threshold

    def calculate_lean_angle(self, img):
        """
        calculate lean angle of a Region of Interest (RoI)
        if angle degree > 0, then take (x_min, y_min, x_max, y_max) as diagonal
        else take (x_min, y_max, x_max, y_min) as diagonal
        :param img:
        :return:
        """
        if isinstance(img, str):
            img = cv2.imread(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 150, apertureSize=3)

        minLineLength = 50
        maxLineGap = 10

        longest_line = (0, 0, 0, 0)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength, maxLineGap, 10)
        if lines is not None and len(lines) > 0:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) > np.linalg.norm(
                            np.array([longest_line[0], longest_line[1]]) - np.array(
                                [longest_line[2], longest_line[3]])):
                        longest_line = (x1, y1, x2, y2)

            if longest_line[0] <= longest_line[2]:
                x_l = longest_line[0]
                y_l = longest_line[1]
                x_r = longest_line[2]
                y_r = longest_line[3]
            else:
                x_l = longest_line[2]
                y_l = longest_line[3]
                x_r = longest_line[0]
                y_r = longest_line[1]

            return np.degrees(np.arctan(((y_r - y_l) / (x_r - x_l))))
        else:
            return 0

    def judge_lean(self, img):
        degree = self.calculate_lean_angle(img)

        return {'degree': degree, 'desc': 'Not Lean' if math.fabs(degree) == 0 else 'Lean'}
