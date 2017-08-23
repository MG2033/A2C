import tensorflow as tf
import numpy as np


class RolloutsEncoder:
    def __init__(self, rollouts_observations, rollouts_rewards, config):
        """
        :param rollouts_observations: the observation output of the rollouts as a tensor of shape:
                                                        [actions_num, rollouts_steps, frame_w, frame_h, frame_c]
        :param rollouts_rewards: the reward output of the rollouts as a tensor of shape:
                                                        [actions_num, rollouts_steps, 1]
        :param config: configuration object
        """
        self.__config = config
        self.__rollouts_observations = rollouts_observations
        self.__rollouts_rewards = rollouts_rewards

        self.__build_model()

    def cnn_encoder(self, x):
        """
        a cnn encoder block with fc layer
        :param x: input tensor of shape: [actions_num, frame_w, frame_h, frame_c]
        :return: output tensor of shape [actions_num, 512]
        """
        with tf.variable_scope('cnn_encoder'):
            for i, (kernals_num, kernals_size, strides) in enumerate(zip(self.__config.cnn_kernals_num, self.__config.cnn_kernal_sizes, self.__config.cnn_strides)):
                x = tf.layers.conv2d(
                    x,
                    kernals_num,
                    kernals_size,
                    strides=strides,
                    padding='same',
                    dilation_rate=(1, 1),
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name='conv_{0}'.format(i + 1))

            # x = tf.contrib.layers.flatten(x)
            # x = tf.layers.dense(
            #     x,
            #     self.__config.fc_units,
            #     activation=tf.nn.relu,
            #     kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #     name='fc')

            return x

    def __template(self, observations, rewards, lstm_state):
        """
        __template function to unroll the lstm
        :param observations: input tensor of shape: [actions_num, frame_w, frame_h, frame_c]
        :param rewards: input tensor of shape: [actions_num, 1]
        :param lstm_state: input tensor of shape: [2, actions_num, 512]
        :return: the output and the lstm hidden state
        """
        lstm_state = tf.contrib.rnn.LSTMStateTuple(lstm_state[0], lstm_state[1])

        encoded_observations = self.cnn_encoder(observations)

        encoded_observations_shape = encoded_observations.get_shape().as_list()

        broadcasted_rewards = tf.ones([encoded_observations_shape[0], encoded_observations_shape[1], encoded_observations_shape[2], 1]) * rewards

        lstm_input = tf.concat([encoded_observations, broadcasted_rewards], axis=0)

    def __build_model(self):
        pass
