import numpy as np
import tensorflow as tf
from models.base_policy import BasePolicy
from layers import conv2d, flatten, dense
from layers import orthogonal_initializer, noise_and_argmax


class CNNPolicy(BasePolicy):
    def __init__(self, sess, input_shape, num_actions, reuse=False, is_training=True, name='train'):
        super().__init__(sess, reuse)
        self.initial_state = []
        with tf.name_scope(name + "policy_input"):
            self.X_input = tf.placeholder(tf.uint8, input_shape)
        with tf.variable_scope("policy", reuse=reuse):
            conv1 = conv2d('conv1', tf.cast(self.X_input, tf.float32) / 255., num_filters=32, kernel_size=(8, 8),
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

    def step(self, observation, *_args, **_kwargs):
        # Take a step using the model and return the predicted policy and value function
        action, value = self.sess.run([self.action_s, self.value_s], {self.X_input: observation})
        return action, value, []  # dummy state

    def value(self, observation, *_args, **_kwargs):
        # Return the predicted value function for a given observation
        return self.sess.run(self.value_s, {self.X_input: observation})
