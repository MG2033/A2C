import os
import numpy as np
import random
import tensorflow as tf
from bunch import Bunch
import argparse
import json


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def parse_args():
    """
    Parse the arguments of the program
    :return: (config_args)
    :rtype: tuple
    """
    # Create a parser
    parser = argparse.ArgumentParser(description="A2C Tensorflow implementation")
    parser.add_argument('--version', action='version', version='%(prog)s 0.0.1')
    parser.add_argument('--config', default=None, type=str, help='Configuration file')

    # Parse the arguments
    args = parser.parse_args()

    # parse the configurations from the config json file provided
    with open(args.config, 'r') as config_file:
        config_args_dict = json.load(config_file)
    # convert the dictionary to a namespace using bunch lib
    config_args = Bunch(config_args_dict)

    print(config_args)
    return config_args


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


def create_experiment_dirs(exp_dir):
    """
    Create Directories of a regular tensorflow experiment directory
    :param exp_dir:
    :return summary_dir, checkpoint_dir:
    """
    experiment_dir = "experiments/" + exp_dir + "/"
    summary_dir = experiment_dir + 'summaries/'
    checkpoint_dir = experiment_dir + 'checkpoints/'
    output_dir = experiment_dir + 'output/'
    test_dir = experiment_dir + 'test/'
    dirs = [summary_dir, checkpoint_dir, output_dir, test_dir]
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        print("Experiment directories created")
        return experiment_dir, summary_dir, checkpoint_dir, output_dir, test_dir
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def create_list_dirs(input_dir, prefix_name, count):
    dirs_path = []
    for i in range(count):
        dirs_path.append(input_dir + prefix_name + '-' + str(i))
        create_dirs([input_dir + prefix_name + '-' + str(i)])
    return dirs_path


def set_all_global_seeds(i):
    try:
        tf.set_random_seed(i)
        np.random.seed(i)
        random.seed(i)
    except:
        return ImportError


def find_trainable_variables(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()
