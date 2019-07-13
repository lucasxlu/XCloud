"""
file io utils
"""

import os


def mkdir_if_not_exist(dir_name):
    """
    make directory if not exist
    :param dir_name:
    :return:
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
