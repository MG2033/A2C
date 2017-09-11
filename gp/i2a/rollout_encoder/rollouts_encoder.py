import tensorflow as tf
from gp.logger.logger import Logger
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

        self.rollout_encoding = self.__build_model()

        self.summaries = tf.summary.merge_all('rollout_encoder')

    def __cnn_encoder(self, x):
        """
        a cnn encoder block with fc layer
        :param x: input tensor of shape: [actions_num, frame_w, frame_h, frame_c]
        :return: output tensor of shape [actions_num, 512]
        """
        with tf.variable_scope('__cnn_encoder'):
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

                Logger.summarize_layer(x, 'rollout_encoder')

            return x

    def __template(self, observations, rewards, lstm_state):
        """
        __template function to unroll the lstm
        :param observations: input tensor of shape: [actions_num, frame_w, frame_h, frame_c]
        :param rewards: input tensor of shape: [actions_num, 1]
        :param lstm_state: input tensor of shape: [2, actions_num, 512]
        :return: the output and the lstm hidden state
        """
        encoded_observations = self.__cnn_encoder(observations)
        encoded_observations_shape = encoded_observations.get_shape().as_list()

        broadcasted_rewards = tf.ones([encoded_observations_shape[0], encoded_observations_shape[1] * encoded_observations_shape[2]]) * rewards
        broadcasted_rewards = tf.reshape(broadcasted_rewards, [encoded_observations_shape[0], encoded_observations_shape[1], encoded_observations_shape[2], 1])

        lstm_state = tf.contrib.rnn.LSTMStateTuple(lstm_state[0], lstm_state[1])
        lstm_input = tf.contrib.layers.flatten(tf.concat([encoded_observations, broadcasted_rewards], axis=3))

        lstm = tf.contrib.rnn.BasicLSTMCell(self.__config.lstm_units)
        lstm_output, lstm_next_state = lstm(lstm_input, lstm_state)

        Logger.summarize_layer(lstm_output, 'rollout_encoder')

        return lstm_output, lstm_next_state

    def __build_model(self):
        encoder_template = tf.make_template('encoder', self.__template)

        lstm_state = tf.zeros([2, self.__config.actions_num, self.__config.lstm_units])
        lstm_state = tf.contrib.rnn.LSTMStateTuple(lstm_state[0], lstm_state[1])

        for i in range(self.__config.rollouts_steps):
            step_output, lstm_state = encoder_template(self.__rollouts_observations[:, i], self.__rollouts_rewards[:, i], lstm_state)

        self.__output = tf.reshape(step_output, [-1])

        return self.__output
