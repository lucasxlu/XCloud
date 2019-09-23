"""
Selective Search in OpenCV, used to create patch for IQA
"""

import cv2


def generate_candidates_by_selective_search(imgpath, ss_type='f', show_result=False):
    """
    generate candidates by selective search algorithm
    :param imgpath:
    :param ss_type: fast (f) or quality (q)
    :param show_result
    :return:
    """
    # number of region proposals to show
    num_show_rects = 10

    # speed-up using multi-threads
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    # read image
    im = cv2.imread(imgpath)

    # resize image to a smaller size for fast computation
    new_height = 200
    new_width = int(im.shape[1] * new_height / im.shape[0])
    im = cv2.resize(im, (new_width, new_height))

    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # set input image on which we will run segmentation
    ss.setBaseImage(im)

    # Switch to fast but low recall Selective Search method
    if ss_type == 'f':
        ss.switchToSelectiveSearchFast()

    # Switch to high recall but slow Selective Search method
    elif ss_type == 'q':
        ss.switchToSelectiveSearchQuality()
    # if argument is neither f nor q print help message
    else:
        print('Invalid type!')

    # run selective search segmentation on input image
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))

    if show_result:
        im_out = im.copy()

        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till num_show_rects
            if i < num_show_rects:
                x, y, w, h = rect
                cv2.rectangle(im_out, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break

        cv2.imshow("Output", im_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
