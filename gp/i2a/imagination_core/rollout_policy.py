import numpy as np
import tensorflow as tf
from gp.layers.utils import orthogonal_initializer, noise_and_argmax
from gp.layers.convolution import conv2d
from gp.layers.dense import flatten, dense


class RolloutPolicy:
    def __init__(self, input_shape, num_actions, reuse=False, is_training=True, name='rollout'):
        with tf.name_scope(name + "_policy_input"):
            self.X_input = tf.placeholder(tf.uint8, input_shape)
        with tf.variable_scope(name + "_policy", reuse=reuse):
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

            fc4 = dense('fc4', conv3_flattened, output_dim=512, initializer=orthogonal_initializer(np.sqrt(2)),
                        activation=tf.nn.relu, is_training=is_training)

            self.policy_logits = dense('policy_logits', fc4, output_dim=num_actions,
                                       initializer=orthogonal_initializer(np.sqrt(1.0)), is_training=is_training)

            self.value_function = dense('value_function', fc4, output_dim=1,
                                        initializer=orthogonal_initializer(np.sqrt(1.0)), is_training=is_training)

            with tf.name_scope('value'):
                self.value_s = self.value_function[:, 0]

            with tf.name_scope('action'):
                self.action_s = noise_and_argmax(self.policy_logits)

            with tf.name_scope('action_original'):
                self.action_o = tf.argmax(self.policy_logits, 1)

    def train_step(self, observation, *_args, **_kwargs):
        # Take a step using the model and return the predicted policy and value function
        action, value = self.sess.run([self.action_s, self.value_s], {self.X_input: observation})
        return action, value

    def test_step(self, observation, *_args, **_kwargs):
        # Take a step using the model and return the predicted policy and value function
        action, value = self.sess.run([self.action_o, self.value_s], {self.X_input: observation})
        return action, value

    def loss(self, behavioural_policy_argmaxed):
        pass

    def value(self, observation, *_args, **_kwargs):
        # Return the predicted value function for a given observation
        return self.sess.run(self.value_s, {self.X_input: observation})
