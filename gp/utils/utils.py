import os
import numpy as np
import random
import tensorflow as tf


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return:
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        print("Creating directories error: {0}".format(err))


def set_all_global_seeds(i):
    try:
        tf.set_random_seed(i)
        np.random.seed(i)
        random.seed(i)
    except:
        return ImportError
