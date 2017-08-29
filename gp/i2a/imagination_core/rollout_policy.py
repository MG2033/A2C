import numpy as np
import tensorflow as tf
from gp.layers.utils import orthogonal_initializer
from gp.layers.convolution import conv2d
from gp.layers.dense import flatten, dense


class RolloutPolicy:
    @staticmethod
    def policy_template(scope, num_actions):
        def template(observation):
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

                conv3_flattened = flatten(conv3)

                fc4 = dense('fc4', conv3_flattened, output_dim=512, initializer=orthogonal_initializer(np.sqrt(2)),
                            activation=tf.nn.relu, is_training=False)

                policy_logits = dense('policy_logits', fc4, output_dim=num_actions,
                                      initializer=orthogonal_initializer(np.sqrt(1.0)), is_training=False)

                value_function = dense('value_function', fc4, output_dim=1,
                                       initializer=orthogonal_initializer(np.sqrt(1.0)), is_training=False)

                with tf.name_scope('action_softmax'):
                    action_softmax = tf.nn.softmax(policy_logits)

            return action_softmax

        return template

    @staticmethod
    def loss(scope, behavioural_policy, rollout_policy):
        # Apply the rollout auxillary loss handler.
        with tf.name_scope(scope):
            ep = 1e-6
            return tf.reduce_mean(-tf.reduce_sum(behavioural_policy * tf.log(rollout_policy + ep), axis=-1))
