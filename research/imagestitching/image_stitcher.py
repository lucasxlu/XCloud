import numpy as np
import cv2

import imutils
from imutils import paths


def stich_images(img_dir="./shelf"):
    """
    stitch images in a given directory
    """
    print("[INFO] loading images...")
    imagePaths = sorted(list(paths.list_images(img_dir)))
    images = []

    # loop over the image paths, load each one, and add them to our
    # images to stitch list
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        images.append(image)

    print("[INFO] stitching images...")
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)
    # if the status is 0, then OpenCV successfully performed image stitching
    if status == 0:
        # write the output stitched image to disk
        cv2.imwrite("./pano.jpg", stitched)

        # display the output stitched image to our screen
        # cv2.imshow("Stitched", stitched)
        # cv2.waitKey(0)

    # otherwise the stitching failed, likely due to not enough keypoints) being detected
    else:
        print("[INFO] image stitching failed ({})".format(status))


def concat_images(img_dir="C:/Users/Administrator/Desktop/shelf"):
    """
    concatenate two image patches in a given directory
    :param img_dir: 
    :return: 
    """
    print("[INFO] loading images...")
    imagePaths = sorted(list(paths.list_images(img_dir)))
    images = []
    cols, rows = [], []

    # loop over the image paths, load each one, and add them to our
    # images to stitch list
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        images.append(image)
        cols.append(image.shape[1])
        rows.append(image.shape[0])

    print("[INFO] concatenating images...")
    max_r = sum(rows)
    max_c = max(cols)

    concated_img = np.ones((max_r, max_c, 3)) * 255
    for i, image in enumerate(images):
        concated_img[0 if i == 0 else rows[i - 1]: sum(rows[: i + 1]),
        int((max_c - cols[i]) / 2):int((max_c + cols[i]) / 2), :] = image

    cv2.imwrite("./concat.jpg", concated_img)


if __name__ == '__main__':
    # stich_images()
    concat_images()
