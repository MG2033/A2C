import numpy as np
import tensorflow as tf
from gp.layers.utils import orthogonal_initializer
from gp.layers.convolution import conv2d
from gp.layers.dense import flatten


class A2CCNN:
    def __init__(self, scope, observation):
        with tf.variable_scope(scope):
            conv1 = conv2d('conv1', observation, num_filters=32, kernel_size=(8, 8),
                           padding='VALID', stride=(4, 4),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=False)

            conv2 = conv2d('conv2', conv1, num_filters=64, kernel_size=(4, 4), padding='VALID', stride=(2, 2),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=False)

            conv3 = conv2d('conv3', conv2, num_filters=64, kernel_size=(3, 3), padding='VALID', stride=(1, 1),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=False)

            self.conv3_flattened = flatten(conv3)

    def cnn_output(self):
        return self.conv3_flattened