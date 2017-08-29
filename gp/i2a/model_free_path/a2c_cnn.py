import numpy as np
import tensorflow as tf
from gp.layers.utils import orthogonal_initializer, noise_and_argmax
from gp.layers.convolution import conv2d
from gp.layers.dense import flatten, dense


class A2CCNN:
    def __init__(self, input_shape, reuse=False, is_training=True):
        with tf.name_scope("a2c_cnn_policy_input"):
            self.X_input = tf.placeholder(tf.uint8, input_shape)
        with tf.variable_scope("a2c_cnn_policy", reuse=reuse):
            conv1 = conv2d('conv1', tf.cast(self.X_input, tf.float32) / 255, num_filters=32, kernel_size=(8, 8),
                           padding='VALID', stride=(4, 4),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=is_training)

            conv2 = conv2d('conv2', conv1, num_filters=64, kernel_size=(4, 4), padding='VALID', stride=(2, 2),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=is_training)

            conv3 = conv2d('conv3', conv2, num_filters=64, kernel_size=(3, 3), padding='VALID', stride=(1, 1),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=is_training)

            conv3_flattened = flatten(conv3)

            self.encoded_output = conv3_flattened

    def get_output(self):
        return self.encoded_output