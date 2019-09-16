import os
import numpy as np

from PIL import Image

ROOT = '../'
SRC_DIR = 'img'
DEST_DIR = 'result'


def get_filenames():
    """Return list of tuples for source and template destination
       filenames(absolute filepath)."""
    filenames = []
    for img_fname in os.listdir(os.path.join(ROOT, SRC_DIR)):
        filenames.append((os.path.join(ROOT, SRC_DIR, img_fname),
                          os.path.join(ROOT, DEST_DIR, img_fname)))

    print(filenames)

    return filenames
